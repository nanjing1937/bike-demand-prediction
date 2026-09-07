# -*- coding: utf-8 -*-
"""
05_pure_nn.py — 纯神经网络独立版：单层网络 + 多层感知机（MLP）
================================================================
共享单车租赁需求预测 | 阶段五

目的：对应《动手学深度学习》主线（线性网络 → 多层感知机），
      提供一个**不依赖任何前序脚本产物**的独立全流程——
      自己读原始数据、自己清洗、自己调参，一条命令跑通：
          python 05_pure_nn.py

与前序脚本的关系：
    - 清洗与特征工程协议复刻自 02_preprocess.py（保证数据口径一致，
      结果可与 03/04 对照），但**不读取** data/processed/ 下的任何产物
    - 04_neural_network.py 的重点是"神经网络 vs 树模型"的对比实验；
      本脚本的重点是"纯神经网络视角"的自包含学习路径

流程：
    1. 读 data/BikeData.csv → 清洗（与 02 协议一致）→ 特征工程
    2. 目标 log1p；30% 测试集（random_state=42，与 02/03 完全一致）
    3. 单层网络（无隐藏层 ≈ 线性回归）作基线
    4. MLP 网格调参（隐藏层宽度 × 学习率，共 5 组）：
       全部在训练集内切出的验证集上选优，测试集零参与；
       最优组合确定后在全训练集重训（早停仍看该验证集）
    5. 测试集只在最终评估用一次：MSE/RMSE/MAE/R²，原始量纲

输出：
    - inbox/results_pure_nn.json      两模型四指标 + 最优网格组合 + 数据行数
    - inbox/figures/fig11_pure_nn_curves.png   单层 + 最优 MLP 训练曲线
    - inbox/figures/fig12_pure_nn_pred.png     最优 MLP 预测 vs 真实散点

评估口径（与 03/04 严格一致）：
    log 空间训练 → expm1 还原 → clip(lower=0)（租赁量物理非负）
    → 原始量纲四指标。测试集只在两模型完全定型后各评估一次。
"""

import json
import time
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

# ---------- 路径：相对脚本位置推导（可移植，clone 后直接跑） ----------
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "BikeData.csv"
INBOX = ROOT / "inbox"
FIG_DIR = INBOX / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DEVICE = torch.device("cpu")   # 明确 CPU：数据量小，CPU 足够且可复现

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3})

# MLP 网格：隐藏层宽度 × 学习率（宽度对应《动手学深度学习》由大到小的
# 容量阶梯；学习率覆盖 Adam 常用两档。其余超参固定，见 GRID_FIXED）
GRID = [
    ((256, 128, 64), 1e-3),
    ((256, 128, 64), 3e-4),
    ((128, 64, 32), 1e-3),
    ((128, 64, 32), 3e-4),
    ((64, 32), 1e-3),
]
GRID_FIXED = {"dropout": 0.2, "weight_decay": 1e-4, "batch_size": 256,
              "max_epochs": 300, "patience": 30}


# ---------- 数据：复刻 02_preprocess.py 的清洗与特征工程协议 ----------
def load_and_clean() -> pd.DataFrame:
    """读原始 csv → 清洗三步（与 02 协议一致，注释见各步）。

    为什么自己清洗而不读 data/processed/：本脚本的定位是"独立全流程"，
    学习路径上从原始数据到评估一条龙；同时复刻同一套协议，
    保证结果与 03/04 可直接对照。
    """
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    n0 = len(df)
    print(f"[读取] {RAW_PATH.name}  shape={df.shape}")

    # 1) 类别大小写统一（EDA 实证：'winter' 80 行、'Y' 80 行）——与 02 协议一致
    df["Seasons"] = df["Seasons"].str.strip().str.title()
    df["Functioning Day"] = df["Functioning Day"].replace({"Y": "Yes", "N": "No"})
    # 2) 完全重复行删除：去重后应恰为 8760 = 365×24 整一年 ——与 02 协议一致
    df = df.drop_duplicates()
    n_dup = n0 - len(df)
    # 3) 目标负值删除（租赁量 >= 0，负值属错误记录）——与 02 协议一致
    df = df[df["Rented Bike Count"] >= 0]
    n_neg = n0 - n_dup - len(df)
    print(f"[清洗] {n0} → 修大小写 → 去重(-{n_dup}) → 删负目标(-{n_neg}) → {len(df)} 行")
    assert len(df) == 8700, f"清洗后行数异常：{len(df)}（应为 8700）"
    return df.reset_index(drop=True)


def feature_engineer(df: pd.DataFrame):
    """特征工程（与 02 协议一致）：Date 派生 + Hour 周期编码 + one-hot。"""
    d = df.copy()
    # A. Date 派生：月份/星期/周末标记（原始 Date 字符串不进模型）
    dt = pd.to_datetime(d["Date"], format="%d/%m/%Y")
    d["Month"] = dt.dt.month
    d["DayOfWeek"] = dt.dt.dayofweek            # 0=周一 … 6=周日
    d["IsWeekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # B. Hour 周期编码：sin/cos 让 23 点与 0 点在特征空间相邻——
    #    纯线性结构的模型（本脚本的单层网络）尤其依赖这一编码
    d["Hour_sin"] = np.sin(2 * np.pi * d["Hour"] / 24)
    d["Hour_cos"] = np.cos(2 * np.pi * d["Hour"] / 24)

    # C. 类别 one-hot（drop_first 避免虚拟变量陷阱）
    d = pd.get_dummies(d, columns=["Seasons", "Holiday", "Functioning Day"],
                       drop_first=True, dtype=int)

    d = d.drop(columns=["Date"])
    y = np.log1p(d.pop("Rented Bike Count"))    # 目标 log1p（右偏→近正态）
    return d, y


# ---------- 网络结构 ----------
class SingleLayerNet(nn.Module):
    """单层网络：只有输出层。y = Wx + b，数学上就是线性回归。

    为什么保留它：验证"神经网络是可微分的参数化函数，退化结构=线性模型"——
    它是多层网络的学习起点（《动手学深度学习》第 3→4 章的衔接点），
    也是后续 MLP 每一分提升的归因基准。
    """

    def __init__(self, n_features):
        super().__init__()
        self.out = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.out(x).squeeze(-1)   # (N,1)->(N,)，与 (N,) 目标直接算损失


class MLP(nn.Module):
    """多层感知机：可配置隐藏层宽度，ReLU + BatchNorm + Dropout。

    BatchNorm 稳定各层输入分布、允许稍大学习率；Dropout 压制神经元
    共适应；最后隐藏层到输出层之间不设 Dropout（输出端不需要额外噪声）。
    """

    def __init__(self, n_features, hidden=(256, 128, 64), p_drop=0.2):
        super().__init__()
        layers, in_dim = [], n_features
        for i, h in enumerate(hidden):
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            if i < len(hidden) - 1:
                layers.append(nn.Dropout(p_drop))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- 训练循环（手写，理解每一步） ----------
def train_model(model, X_tr, y_tr, X_val, y_val, *, lr, weight_decay,
                batch_size, max_epochs, patience, tag, verbose=True):
    """手写 mini-batch 训练循环 + 早停。

    为什么需要早停：MLP 容量大（数万参数）而训练数据仅数千行，
    很容易过拟合——验证损失不再下降时停止，并回滚到验证最优权重。
    验证集来自训练集内部，与测试集无关（防泄漏）。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    loss_fn = nn.MSELoss()                  # 回归口径：均方误差（log 空间）
    n = X_tr.shape[0]
    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()                       # Dropout 生效、BN 用批次统计
        perm = torch.randperm(n)            # 每轮洗牌，打破批次相关性
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()           # 清上一步梯度，防累加
            loss = loss_fn(model(xb), yb)
            loss.backward()                 # 反向传播求梯度
            optimizer.step()                # Adam 更新参数
            epoch_loss += loss.item()
            n_batches += 1
        history["train"].append(epoch_loss / n_batches)

        model.eval()                        # Dropout 关闭、BN 用全局统计
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        history["val"].append(val_loss)

        # 早停判定：验证损失连续 patience 轮无改善则停
        if val_loss < best_val - 1e-5:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    [{tag}] 早停于第 {epoch} 轮（最优第 {best_epoch} 轮，"
                          f"val MSE={best_val:.4f}）")
                break
        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"    [{tag}] epoch {epoch:>4}  train MSE={history['train'][-1]:.4f}  "
                  f"val MSE={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)   # 回滚到验证最优权重
    return history, best_epoch, best_val


def evaluate(name, model, X_te, y_log_te, results, config=None):
    """统一评估：log 空间预测 → expm1 还原 → clip(0) → 原始量纲四指标。

    与 03/04 同一条路径——只有口径一致，跨脚本的数字才有可比性。
    """
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
        res["config"] = config
    results[name] = res
    print(f"  [{name:<14}] RMSE={res['RMSE']:7.1f}  MAE={res['MAE']:7.1f}  "
          f"R2={res['R2']:.4f}  MSE={res['MSE']:.1f}")
    return pred, true


# ---------- 可视化（标签用英文，避免环境缺中文字体出现方框） ----------
def plot_curves(hist_sl, hist_mlp, best_tag):
    """Fig.11 单层 + 最优 MLP 训练曲线（log 纵轴：损失跨数量级更清晰）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, (title, h) in zip(axes, [("SingleLayerNet", hist_sl), (best_tag, hist_mlp)]):
        ax.plot(h["train"], label="train loss", color="#4C72B0")
        ax.plot(h["val"], label="val loss", color="#DD8452")
        ax.set(xlabel="Epoch", ylabel="MSE (log scale)", title=title, yscale="log")
        ax.legend()
    fig.suptitle("Fig.11 Pure-NN Training Curves — early stopping on inner validation set",
                 y=1.04)
    fig.savefig(FIG_DIR / "fig11_pure_nn_curves.png")
    plt.close(fig)


def plot_pred(true, pred):
    """Fig.12 最优 MLP 预测 vs 真实散点（越贴对角线越好）。"""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(true, pred, s=6, alpha=0.25, color="#4C72B0")
    lim = [0, max(true.max(), pred.max()) * 1.02]
    ax.plot(lim, lim, "r--", lw=1.2, label="y = x")
    ax.set(xlim=lim, ylim=lim, xlabel="Actual rented bikes", ylabel="Predicted",
           title="Best MLP (grid-searched)")
    ax.legend()
    fig.suptitle("Fig.12 Pure-NN Best MLP — Pred vs Actual (test set)", y=1.02)
    fig.savefig(FIG_DIR / "fig12_pure_nn_pred.png")
    plt.close(fig)


# ---------- 主流程 ----------
def main():
    print("=" * 60)
    print("阶段五：纯神经网络独立版（单层 + MLP 网格调参，CPU 训练）")
    print("=" * 60)

    # ---- 1) 数据：原始 csv → 清洗 → 特征工程 → 划分（全部自包含） ----
    df = load_and_clean()
    X, y = feature_engineer(df)
    print(f"[特征工程] 派生 Month/DayOfWeek/IsWeekend + Hour sin/cos + one-hot "
          f"→ {X.shape[1]} 个特征")

    # 划分：30% 测试集，random_state=42 与 02/03 完全一致——
    # 同一种子+同一协议 → 与 03/04 逐行相同的测试集，结果可直接对照
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE)
    print(f"[划分] 训练 {X_train.shape[0]} 行 / 测试 {X_test.shape[0]} 行"
          f"（30%，random_state=42）")
    assert X_test.shape[0] == 2610, f"测试集行数异常：{X_test.shape[0]}（应为 2610）"

    # 标准化：神经网络对输入量纲敏感（各特征梯度尺度不一）。
    # scaler 只在训练集上 fit 再变换测试集——防止测试集统计量泄漏进训练
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)
    ytr = y_train.to_numpy(dtype=np.float32)
    yte = y_test.to_numpy(dtype=np.float32)

    # 调参/早停用验证集：从训练集内再切 20%（测试集零参与）
    X_fit, X_val, y_fit, y_val = train_test_split(
        Xtr, ytr, test_size=0.2, random_state=RANDOM_STATE)
    X_fit, y_fit = torch.tensor(X_fit), torch.tensor(y_fit)
    X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
    X_te, y_te = torch.tensor(Xte), torch.tensor(yte)
    print(f"[切分] 网格调参/早停验证集 {X_val.shape[0]} 行（来自训练集内部）")

    results = {}

    # ================= 2) 单层网络（基线，≈线性回归） =================
    print("\n---- 1. 单层网络（无隐藏层，≈线性回归） ----")
    torch.manual_seed(RANDOM_STATE)         # 固定初始化，重跑幂等
    np.random.seed(RANDOM_STATE)
    sl = SingleLayerNet(Xtr.shape[1]).to(DEVICE)
    hist_sl, ep_sl, _ = train_model(
        sl, X_fit, y_fit, X_val, y_val,
        lr=0.01, weight_decay=0.0, batch_size=512, max_epochs=500,
        patience=50, tag="SingleLayerNet")
    n_params_sl = sum(p.numel() for p in sl.parameters())
    print(f"    参数量 {n_params_sl}（= 特征数 {Xtr.shape[1]} 个权重 + 1 个截距）")
    pred_sl, true_te = evaluate(
        "SingleLayerNN", sl, X_te, y_te, results,
        config={"lr": 0.01, "batch_size": 512, "best_epoch": ep_sl,
                "n_params": n_params_sl})

    # ================= 3) MLP 网格调参（宽度 × 学习率，5 组） =================
    print(f"\n---- 2. MLP 网格调参（{len(GRID)} 组：宽度 × 学习率，"
          f"其余固定 {GRID_FIXED}） ----")
    grid_rows = []
    t0 = time.time()
    for i, (hidden, lr) in enumerate(GRID, 1):
        tag = f"{len(hidden)}x{hidden[0]} lr={lr:g}"
        torch.manual_seed(RANDOM_STATE)     # 每组同种子：差异只来自超参本身
        np.random.seed(RANDOM_STATE)
        m = MLP(Xtr.shape[1], hidden=hidden, p_drop=GRID_FIXED["dropout"]).to(DEVICE)
        _, ep, val_mse = train_model(
            m, X_fit, y_fit, X_val, y_val,
            lr=lr, weight_decay=GRID_FIXED["weight_decay"],
            batch_size=GRID_FIXED["batch_size"], max_epochs=GRID_FIXED["max_epochs"],
            patience=GRID_FIXED["patience"], tag=tag, verbose=False)
        n_par = sum(p.numel() for p in m.parameters())
        grid_rows.append({"hidden": list(hidden), "lr": lr, "val_MSE": round(val_mse, 5),
                          "best_epoch": ep, "n_params": n_par})
        print(f"  [{i}/{len(GRID)}] hidden={'-'.join(map(str, hidden)):<11} lr={lr:<7g}"
              f" val MSE={val_mse:.5f}  epoch={ep:<4d} params={n_par}")
    grid_time = time.time() - t0
    best = min(grid_rows, key=lambda r: r["val_MSE"])
    best_hidden, best_lr = tuple(best["hidden"]), best["lr"]
    print(f"[网格] 耗时 {grid_time:.1f}s，最优组合：hidden={'-'.join(map(str, best_hidden))}"
          f" lr={best_lr:g}（val MSE={best['val_MSE']:.5f}）")

    # ---- 最优组合在全训练集重训：网格已用验证集选出结构，最终模型
    #      吃回全部 6090 行训练数据；早停仍看同一验证集（测试集零参与） ----
    print("\n---- 3. 最优组合全训练集重训 ----")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    X_all, y_all = torch.tensor(Xtr), torch.tensor(ytr)
    mlp = MLP(Xtr.shape[1], hidden=best_hidden,
              p_drop=GRID_FIXED["dropout"]).to(DEVICE)
    hist_mlp, ep_mlp, _ = train_model(
        mlp, X_all, y_all, X_val, y_val,
        lr=best_lr, weight_decay=GRID_FIXED["weight_decay"],
        batch_size=GRID_FIXED["batch_size"], max_epochs=GRID_FIXED["max_epochs"],
        patience=GRID_FIXED["patience"], tag="MLP-final")
    n_params_mlp = sum(p.numel() for p in mlp.parameters())
    best_tag = f"MLP {'-'.join(map(str, best_hidden))}"
    pred_mlp, _ = evaluate(
        "MLP", mlp, X_te, y_te, results,
        config={"hidden": "-".join(map(str, best_hidden)), "lr": best_lr,
                "dropout": GRID_FIXED["dropout"],
                "weight_decay": GRID_FIXED["weight_decay"],
                "batch_size": GRID_FIXED["batch_size"],
                "best_epoch": ep_mlp, "n_params": n_params_mlp,
                "refit_on": "full train set"})

    # ================= 4) 可视化 + 落盘 =================
    plot_curves(hist_sl, hist_mlp, best_tag)
    plot_pred(true_te, pred_mlp)

    out = {
        "meta": {
            "script": "05_pure_nn.py",
            "pipeline": "独立全流程：读原始数据 → 清洗（复刻 02 协议）→ "
                        "特征工程 → 30% 划分 → 单层 + MLP 网格调参 → 测试集一次评估",
            "rows_raw": 8820, "rows_after_dedup": 8760, "rows_clean": 8700,
            "train_rows": int(Xtr.shape[0]), "test_rows": int(Xte.shape[0]),
            "n_features": int(Xtr.shape[1]),
            "random_state": RANDOM_STATE,
            "device": "cpu",
        },
        "SingleLayerNN": results["SingleLayerNN"],
        "MLP": results["MLP"],
        "grid_search": {
            "protocol": "网格与早停只看训练集内 20% 验证集；"
                        "最优组合在全训练集重训后测试集仅评估一次",
            "fixed": GRID_FIXED,
            "candidates": grid_rows,
            "best": best,
            # 注：网格耗时只打到 stdout / 写入实验报告，不进本 JSON——
            # 墙钟时间每次运行天然波动，写进文件会破坏逐字节幂等
        },
    }
    out_path = INBOX / "results_pure_nn.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n===== 纯神经网络独立版结果（30% 测试集，原始量纲） =====")
    hdr = f"{'Model':<16}{'MSE':>12}{'RMSE':>10}{'MAE':>9}{'R2':>9}"
    print(hdr); print("-" * len(hdr))
    for n in ("SingleLayerNN", "MLP"):
        r = results[n]
        print(f"{n:<16}{r['MSE']:>12.1f}{r['RMSE']:>10.1f}{r['MAE']:>9.1f}{r['R2']:>9.4f}")
    print(f"\n[落盘] {out_path.relative_to(ROOT)}")
    print("[图] fig11 / fig12 输出至 inbox/figures/")


if __name__ == "__main__":
    main()
