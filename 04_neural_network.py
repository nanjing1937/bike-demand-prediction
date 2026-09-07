# -*- coding: utf-8 -*-
"""
04_neural_network.py — 神经网络对比实验：单层网络 vs 多层感知机
==============================================================
共享单车租赁需求预测 | 阶段四（追加批）

目的：回答"表格数据上，神经网络与树模型差距多大"，同时补上
      实验"已知限制"中"未尝试神经网络"一条。

上游：data/processed/ 四个 csv（02 落盘、03 同款划分——对比公平的基础）
输出：
    - inbox/results_summary.json       追加 SingleLayerNN / MLP 两键
    - inbox/figures/fig9_training_curves.png    两模型训练曲线
    - inbox/figures/fig10_nn_pred_vs_true.png   两模型预测vs真实散点

评估口径（与 03 严格一致）：
    - 目标列已是 log1p(租赁量)：训练在 log 空间，预测后 expm1 还原
    - 还原后 clip(lower=0)（租赁量物理非负），四指标在原始量纲计算
    - 测试集只在两个模型完全定型后各评估一次；
      早停验证集从训练集内再切 20%，测试集零参与任何决策

两个模型的设计（为什么这么搭）：
    1) 单层网络：仅一个线性输出层、无隐藏层——数学上就是线性回归。
       预期贴着 LinearRegression 基线（R²≈0.57），用来验证
       "神经网络是可微分的参数化函数，退化结构=线性模型"这一理论
    2) MLP：3 隐藏层 256-128-64，ReLU + BatchNorm + Dropout，
       Adam + 权重衰减 + 早停——表格数据神经网络的常规强配置
    特征：19 列全保留 + StandardScaler 标准化。
       03 的线性族剔除露点温度是为 OLS 共线性的数值稳定（Ridge 实证
       两种版本 R² 差 <0.001）；神经网络对共线性不敏感，故全保留，
       与树模型的特征口径一致。

技术选型（自行决定，报告注明）：单层网络 lr=0.01/batch=512/上限 500 轮；
MLP lr=1e-3/weight_decay=1e-4/batch=256/上限 300 轮；早停耐心 30~50 轮。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- 路径：相对脚本位置推导（可移植） ----------
ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "inbox" / "figures"
INBOX = ROOT / "inbox"

RANDOM_STATE = 42
DEVICE = torch.device("cpu")          # 明确 CPU 训练：数据量小，CPU 足够且可复现
torch.manual_seed(RANDOM_STATE)       # 固定全局种子：权重初始化/批次洗牌可复现
np.random.seed(RANDOM_STATE)

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3})


# ---------- 数据 ----------
def load_data():
    """读取 02 落盘的四份数据（与 03 完全同一份划分）。"""
    X_train = pd.read_csv(PROC / "X_train.csv")
    X_test = pd.read_csv(PROC / "X_test.csv")
    y_train = pd.read_csv(PROC / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(PROC / "y_test.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


# ---------- 网络结构 ----------
class SingleLayerNN(nn.Module):
    """单层网络：只有输出层。y = Wx + b，与 LinearRegression 同构。"""

    def __init__(self, n_features):
        super().__init__()
        self.out = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.out(x).squeeze(-1)   # (N,1)->(N,)，方便与 (N,) 目标直接算损失


class MLP(nn.Module):
    """多层感知机：256-128-64，ReLU + BatchNorm + Dropout。

    BatchNorm 稳定各层输入分布、允许稍大学习率；
    Dropout 压制神经元共适应（co-adaptation），两者是表格 MLP 常规配置。
    """

    def __init__(self, n_features, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- 训练循环（手写，理解每一步） ----------
def train_model(model, X_tr, y_tr, X_val, y_val, *, lr, weight_decay,
                batch_size, max_epochs, patience, tag):
    """
    手写 mini-batch 训练循环 + 早停。

    为什么需要早停：MLP 容量大（约 4.6 万参数），6090 行训练数据
    很容易过拟合——验证损失不再下降时停止，并回滚到验证最优的那份权重。
    验证集来自训练集内部再切分，与测试集无关（防泄漏）。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    loss_fn = nn.MSELoss()                      # 与 03 的回归口径一致：均方误差
    n = X_tr.shape[0]
    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0

    for epoch in range(1, max_epochs + 1):
        # --- 训练模式：Dropout 生效、BatchNorm 用批次统计 ---
        model.train()
        perm = torch.randperm(n)                # 每轮洗牌，打破批次相关性
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()               # 清上一步梯度，防累加
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()                     # 反向传播求梯度
            optimizer.step()                    # Adam 更新参数
            epoch_loss += loss.item()
            n_batches += 1
        history["train"].append(epoch_loss / n_batches)

        # --- 评估模式：Dropout 关闭、BatchNorm 用全局统计 ---
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        history["val"].append(val_loss)

        # --- 早停判定：验证损失连续 patience 轮无改善则停 ---
        if val_loss < best_val - 1e-5:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    [{tag}] 早停于第 {epoch} 轮（最优第 {best_epoch} 轮，"
                      f"val MSE={best_val:.4f}）")
                break
        if epoch % 50 == 0 or epoch == 1:
            print(f"    [{tag}] epoch {epoch:>4}  train MSE={history['train'][-1]:.4f}  "
                  f"val MSE={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)       # 回滚到验证最优权重
    return history, best_epoch


def evaluate(name, model, X_te, y_log_te, results, config=None):
    """统一评估（与 03 同一条路径）：log 空间预测 → expm1 → clip(0) → 原始量纲四指标。"""
    model.eval()
    with torch.no_grad():
        pred_log = model(X_te).numpy()
    pred = np.clip(np.expm1(pred_log), 0, None)
    true = np.expm1(np.asarray(y_log_te))
    mse = mean_squared_error(true, pred)
    res = {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(true, pred)),
        "R2": float(r2_score(true, pred)),
    }
    if config:
        res["best_params"] = config
    results[name] = res
    print(f"  [{name:<14}] RMSE={res['RMSE']:7.1f}  MAE={res['MAE']:7.1f}  "
          f"R2={res['R2']:.4f}  MSE={res['MSE']:.1f}")
    return pred


# ---------- 可视化 ----------
def plot_training_curves(hist_sl, hist_mlp):
    """Fig.9 两模型训练曲线（log 纵轴：损失跨数量级时更清晰）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, (title, h) in zip(axes, [("SingleLayerNN", hist_sl), ("MLP", hist_mlp)]):
        ax.plot(h["train"], label="train loss", color="#4C72B0")
        ax.plot(h["val"], label="val loss", color="#DD8452")
        ax.set(xlabel="Epoch", ylabel="MSE (log scale)", title=title,
               yscale="log")
        ax.legend()
    fig.suptitle("Fig.9 Training Curves — early stopping on inner validation set", y=1.04)
    fig.savefig(FIG_DIR / "fig9_training_curves.png")
    plt.close(fig)


def plot_pred_vs_true(true, pred_sl, pred_mlp):
    """Fig.10 两模型预测 vs 真实散点（越贴对角线越好）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    for ax, (title, pred) in zip(axes, [("SingleLayerNN", pred_sl), ("MLP", pred_mlp)]):
        ax.scatter(true, pred, s=6, alpha=0.25, color="#4C72B0")
        lim = [0, max(true.max(), pred.max()) * 1.02]
        ax.plot(lim, lim, "r--", lw=1.2, label="y = x")
        ax.set(xlim=lim, ylim=lim, xlabel="Actual rented bikes", ylabel="Predicted",
               title=title)
        ax.legend()
    fig.suptitle("Fig.10 Neural Networks — Pred vs Actual (test set)", y=1.02)
    fig.savefig(FIG_DIR / "fig10_nn_pred_vs_true.png")
    plt.close(fig)


# ---------- 主流程 ----------
def main():
    print("=" * 60)
    print("阶段四：神经网络对比（单层网络 + MLP，CPU 训练）")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()
    print(f"[数据] 训练 {X_train.shape} / 测试 {X_test.shape}（与 03 同一份划分）")

    # 标准化：神经网络对输入量纲敏感（梯度尺度不一）——scaler 只在训练集上 fit，
    # 再变换测试集，防止测试集统计量泄漏进训练过程
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)
    ytr = y_train.to_numpy(dtype=np.float32)
    yte = y_test.to_numpy(dtype=np.float32)

    # 早停用验证集：从训练集内再切 20%（测试集零参与）
    Xtr_fit, Xval, ytr_fit, yval = train_test_split(
        Xtr, ytr, test_size=0.2, random_state=RANDOM_STATE)
    tensors = {
        "fit": (torch.tensor(Xtr_fit), torch.tensor(ytr_fit)),
        "val": (torch.tensor(Xval), torch.tensor(yval)),
        "test": (torch.tensor(Xte), torch.tensor(yte)),
    }
    X_fit, y_fit = tensors["fit"]
    X_val, y_val = tensors["val"]
    X_te, y_te = tensors["test"]
    print(f"[切分] 早停验证集 {Xval.shape[0]} 行（来自训练集内部）")

    results = {}

    # ================= 1) 单层网络（理论对照） =================
    print("\n---- 1. 单层网络（无隐藏层，≈线性回归） ----")
    torch.manual_seed(RANDOM_STATE)             # 两组实验各自固定初始化
    sl = SingleLayerNN(Xtr.shape[1]).to(DEVICE)
    hist_sl, ep_sl = train_model(
        sl, X_fit, y_fit, X_val, y_val,
        lr=0.01, weight_decay=0.0, batch_size=512, max_epochs=500,
        patience=50, tag="SingleLayerNN")
    n_params_sl = sum(p.numel() for p in sl.parameters())
    print(f"    参数量 {n_params_sl}（= 特征数 {Xtr.shape[1]} 个权重 + 1 个截距）")
    pred_sl = evaluate("SingleLayerNN", sl, X_te, y_te, results,
                       config={"lr": 0.01, "batch_size": 512,
                               "best_epoch": ep_sl, "n_params": n_params_sl})

    # ================= 2) MLP =================
    print("\n---- 2. 多层感知机 MLP（256-128-64 + BN + Dropout） ----")
    torch.manual_seed(RANDOM_STATE)
    mlp = MLP(Xtr.shape[1]).to(DEVICE)
    hist_mlp, ep_mlp = train_model(
        mlp, X_fit, y_fit, X_val, y_val,
        lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=300,
        patience=30, tag="MLP")
    n_params_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"    参数量 {n_params_mlp}")
    pred_mlp = evaluate("MLP", mlp, X_te, y_te, results,
                        config={"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 256,
                                "hidden": "256-128-64", "dropout": 0.2,
                                "best_epoch": ep_mlp, "n_params": n_params_mlp})

    # ================= 3) 可视化 + 落盘 =================
    plot_training_curves(hist_sl, hist_mlp)
    plot_pred_vs_true(np.expm1(yte), pred_sl, pred_mlp)

    # 追加进既有 results_summary.json：先读旧内容，新增两键，保持原有键序不变
    out = INBOX / "results_summary.json"
    with open(out, encoding="utf-8") as f:
        summary = json.load(f)
    summary.update(results)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== 神经网络结果（30% 测试集，原始量纲） =====")
    hdr = f"{'Model':<16}{'MSE':>12}{'RMSE':>10}{'MAE':>9}{'R2':>9}"
    print(hdr); print("-" * len(hdr))
    for n, r in results.items():
        print(f"{n:<16}{r['MSE']:>12.1f}{r['RMSE']:>10.1f}{r['MAE']:>9.1f}{r['R2']:>9.4f}")
    print(f"\n[落盘] {out.relative_to(ROOT)}（追加 SingleLayerNN / MLP）")
    print("[图] fig9 / fig10 输出至 inbox/figures/")


if __name__ == "__main__":
    main()
