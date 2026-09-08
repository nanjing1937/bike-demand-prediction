# -*- coding: utf-8 -*-
"""
04_weight_decay.py — 组件消融延伸实验：MLP Adam 加 weight_decay
==============================================================
共享单车租赁需求预测 | 延伸实验（weight_decay 单变量）

本脚本为组件消融延伸实验：在 03_train_nn.py 极简协议的基础上，
**唯一变量**是 MLP 优化器 Adam 加 weight_decay=1e-4（取自此前的
调参结论）；SingleLayerNet 保持与 03 完全一致（不加），作为对照锚。
主线仍是 03 的极简协议，本脚本只回答一个问题：
    去掉正则化损失的那 3 个百分点，有多少能靠 weight_decay 找回来？
一条命令跑通：
    python 04_weight_decay.py

与前序脚本的关系：
    - 清洗与特征工程协议复刻自 02_preprocess.py（保证数据口径一致），
      但**不读取** data/processed/ 下的任何产物
    - 01/02 仍是独立可跑的 EDA 与预处理阶段；本脚本可直接从原始数据跑通

本版本刻意极简，仅含 Linear 与 ReLU，不含正则化/归一化组件。

两阶段训练协议（k 折交叉验证定轮数 → 全量重训）：
    阶段一（定轮数）：5 折交叉验证切训练全集 6090 行；每折按现行超参
        从头训练（折内再切 20% 做早停验证），记录该折最优轮数 best_epoch；
        取五折中位数（int(np.median)）定为 EPOCHS_FINAL。
        —— 每个数据点都当过验证点，轮数估计不再永久扣留 20% 样本。
    阶段二（全量重训）：在 6090 行全集上以固定 epoch=EPOCHS_FINAL
        从头训练，无验证集、无早停；早停机制只保留在阶段一。
    测试集 30%（2610 行）从头到尾只在最终评估用一次，口径不变。

输出：
    - inbox/results_wd.json           两模型四指标 + 协议信息
    - inbox/figures/fig13_wd_curves.png        两模型阶段二训练曲线
    - inbox/figures/fig14_wd_pred.png          MLP 预测 vs 真实散点

评估口径（与 02 预处理严格一致）：
    log 空间训练 → expm1 还原 → clip(lower=0)（租赁量物理非负）
    → 原始量纲四指标。测试集只在两模型完全定型后各评估一次。
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- 路径：相对脚本位置推导（可移植，clone 后直接跑） ----------
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "BikeData.csv"
INBOX = ROOT / "inbox"
FIG_DIR = INBOX / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
DEVICE = torch.device("cpu")   # 明确 CPU：数据量小，CPU 足够且可复现

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


# ---------- 数据：复刻 02_preprocess.py 的清洗与特征工程协议 ----------
def load_and_clean() -> pd.DataFrame:
    """读原始 csv → 清洗三步（与 02 协议一致，注释见各步）。

    为什么自己清洗而不读 data/processed/：本脚本的定位是"独立全流程"，
    学习路径上从原始数据到评估一条龙；同时复刻同一套协议，
    保证结果与 02 产出的划分口径一致。
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


# ---------- 网络结构（本版本刻意极简，仅 Linear 与 ReLU） ----------
class SingleLayerNet(nn.Module):
    """单层网络：只有输出层。y = Wx + b，数学上就是线性回归。

    为什么保留它：验证"神经网络是可微分的参数化函数，退化结构=线性模型"——
    它是多层网络的学习起点（《动手学深度学习》第 3→4 章的衔接点），
    也是 MLP 每一分提升的归因基准。
    """

    def __init__(self, n_features):
        super().__init__()
        self.out = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.out(x).squeeze(-1)   # (N,1)->(N,)，与 (N,) 目标直接算损失


class MLP(nn.Module):
    """多层感知机：每层 Linear + ReLU，末层 Linear 到输出。

    结构固定 256-128-64（容量阶梯由大到小）；不含任何其他模块。
    """

    def __init__(self, n_features, hidden=(256, 128, 64)):
        super().__init__()
        layers, in_dim = [], n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- 训练循环（手写，理解每一步） ----------
def train_model(model, X_tr, y_tr, X_val, y_val, *, lr,
                batch_size, max_epochs, patience, tag, verbose=True,
                weight_decay=0.0):
    """手写 mini-batch 训练循环 + 早停（两阶段协议的阶段一专用）。

    为什么需要早停：MLP 容量大（数万参数）而训练数据仅数千行，
    容易过拟合——验证损失不再下降时停止，并回滚到验证最优权重。
    验证集来自该折训练数据内部，与测试集无关（防泄漏）。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    loss_fn = nn.MSELoss()                  # 回归口径：均方误差（log 空间）
    n = X_tr.shape[0]
    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
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

        model.eval()
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


def train_full(model, X_tr, y_tr, *, lr, batch_size, epochs, tag,
              weight_decay=0.0):
    """阶段二：固定轮数全量重训（无验证集、无早停）。

    轮数已由阶段一五折中位数确定；这里在训练全集（6090 行）上从头训练，
    每一行数据都参与拟合——不再永久扣留 20% 样本做早停验证。
    训练损失曲线完整记录（用于 fig11）。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    n = X_tr.shape[0]
    history = {"train": []}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        history["train"].append(epoch_loss / n_batches)
        if epoch % 50 == 0 or epoch == 1:
            print(f"    [{tag}] epoch {epoch:>4}  train MSE={history['train'][-1]:.4f}")
    return history


def run_two_stage(tag, make_model, X_all, y_all, *, lr, batch_size,
                  max_epochs, patience, weight_decay=0.0):
    """两阶段协议（对每个模型执行）：5 折定轮数 → 全量重训。

    阶段一：KFold(5, shuffle, random_state=42) 切训练全集；每折内再切
            20% 做早停验证（与旧协议同口径，测试集零参与），从头训练
            并记录该折 best_epoch；五折中位数定为 EPOCHS_FINAL。
    阶段二：重置种子后在全集上以固定轮数从头训练。
    每次训练（每折、每阶段）前都重置种子，保证两遍运行逐字节幂等。
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_best = []
    for k, (tr_idx, _) in enumerate(kf.split(X_all), 1):
        Xk, yk = X_all[tr_idx], y_all[tr_idx]
        X_fit, X_val, y_fit, y_val = train_test_split(
            Xk, yk, test_size=0.2, random_state=RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)     # 每折同种子：折间可比、重跑幂等
        np.random.seed(RANDOM_STATE)
        model = make_model().to(DEVICE)
        _, best_epoch, best_val = train_model(
            model, torch.tensor(X_fit), torch.tensor(y_fit),
            torch.tensor(X_val), torch.tensor(y_val),
            lr=lr, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, tag=f"{tag}-fold{k}", verbose=False,
            weight_decay=weight_decay)
        fold_best.append(best_epoch)
        print(f"  [{tag}] 折 {k}/{N_FOLDS}: best_epoch={best_epoch:>3}  "
              f"(best val MSE={best_val:.4f})")

    epochs_final = int(np.median(fold_best))
    print(f"  [{tag}] 五折 best_epoch 中位数 = {epochs_final} → 阶段二固定轮数")

    torch.manual_seed(RANDOM_STATE)         # 阶段二同样固定种子
    np.random.seed(RANDOM_STATE)
    model = make_model().to(DEVICE)
    hist = train_full(model, torch.tensor(X_all), torch.tensor(y_all),
                      lr=lr, batch_size=batch_size,
                      epochs=epochs_final, tag=f"{tag}-final",
                      weight_decay=weight_decay)
    return model, hist, epochs_final


def evaluate(name, model, X_te, y_log_te, results, config=None):
    """统一评估：log 空间预测 → expm1 还原 → clip(0) → 原始量纲四指标。

    与 02 预处理同一条路径——只有口径一致，跨脚本的数字才有可比性。
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
def plot_curves(hist_sl, hist_mlp):
    """Fig.11 单层 + MLP 阶段二训练曲线（log 纵轴：损失跨数量级更清晰）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, (title, h) in zip(axes, [("SingleLayerNet", hist_sl), ("MLP 256-128-64", hist_mlp)]):
        ax.plot(h["train"], label="train loss", color="#4C72B0")
        ax.set(xlabel="Epoch", ylabel="MSE (log scale)", title=title, yscale="log")
        ax.legend()
    fig.suptitle("Fig.13 Weight-Decay MLP — Training Curves at k-fold median epochs",
                 y=1.04)
    fig.savefig(FIG_DIR / "fig13_wd_curves.png")
    plt.close(fig)


def plot_pred(true, pred):
    """Fig.12 MLP 预测 vs 真实散点（越贴对角线越好）。"""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(true, pred, s=6, alpha=0.25, color="#4C72B0")
    lim = [0, max(true.max(), pred.max()) * 1.02]
    ax.plot(lim, lim, "r--", lw=1.2, label="y = x")
    ax.set(xlim=lim, ylim=lim, xlabel="Actual rented bikes", ylabel="Predicted",
           title="MLP 256-128-64")
    ax.legend()
    fig.suptitle("Fig.14 Weight-Decay MLP — Pred vs Actual (test set)", y=1.02)
    fig.savefig(FIG_DIR / "fig14_wd_pred.png")
    plt.close(fig)


# ---------- 主流程 ----------
def main():
    print("=" * 60)
    print("延伸实验：weight_decay 单变量消融（5折定轮数 + 全量重训）")
    print("=" * 60)

    # ---- 1) 数据：原始 csv → 清洗 → 特征工程 → 划分（全部自包含） ----
    df = load_and_clean()
    X, y = feature_engineer(df)
    print(f"[特征工程] 派生 Month/DayOfWeek/IsWeekend + Hour sin/cos + one-hot "
          f"→ {X.shape[1]} 个特征")

    # 划分：30% 测试集，random_state=42 与 02 完全一致——
    # 同一种子+同一协议 → 与 02 逐行相同的测试集，结果可直接对照
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE)
    print(f"[划分] 训练 {X_train.shape[0]} 行 / 测试 {X_test.shape[0]} 行"
          f"（30%，random_state=42）")
    assert X_test.shape[0] == 2610, f"测试集行数异常：{X_test.shape[0]}（应为 2610）"
    assert X_train.shape[0] == 6090, f"训练集行数异常：{X_train.shape[0]}（应为 6090）"

    # 标准化：神经网络对输入量纲敏感（各特征梯度尺度不一）。
    # scaler 只在训练集上 fit 再变换测试集——防止测试集统计量泄漏进训练
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)
    ytr = y_train.to_numpy(dtype=np.float32)
    yte = y_test.to_numpy(dtype=np.float32)
    X_te, y_te = torch.tensor(Xte), torch.tensor(yte)
    print(f"[协议] 阶段一 KFold({N_FOLDS}) 定轮数 → 阶段二 {Xtr.shape[0]} 行全量重训")

    results = {}

    # ================= 2) 单层网络（基线，≈线性回归） =================
    print("\n---- 1. 单层网络（无隐藏层，≈线性回归） ----")
    sl, hist_sl, ep_sl = run_two_stage(
        "SingleLayerNet", lambda: SingleLayerNet(Xtr.shape[1]),
        Xtr, ytr, lr=0.01, batch_size=512, max_epochs=500, patience=50)
    n_params_sl = sum(p.numel() for p in sl.parameters())
    print(f"    参数量 {n_params_sl}（= 特征数 {Xtr.shape[1]} 个权重 + 1 个截距）")
    pred_sl, true_te = evaluate(
        "SingleLayerNN", sl, X_te, y_te, results,
        config={"lr": 0.01, "batch_size": 512, "folds": N_FOLDS,
                "epochs_final": ep_sl, "n_params": n_params_sl})

    # ================= 3) MLP（256-128-64） =================
    print("\n---- 2. MLP（hidden 256-128-64，Linear+ReLU） ----")
    # 唯一实验变量：MLP 的 Adam 加 weight_decay=1e-4（SingleLayerNet 不加=对照锚）
    mlp, hist_mlp, ep_mlp = run_two_stage(
        "MLP", lambda: MLP(Xtr.shape[1], hidden=(256, 128, 64)),
        Xtr, ytr, lr=1e-3, batch_size=256, max_epochs=300, patience=30,
        weight_decay=1e-4)
    n_params_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"    参数量 {n_params_mlp}")
    pred_mlp, _ = evaluate(
        "MLP", mlp, X_te, y_te, results,
        config={"hidden": "256-128-64", "lr": 1e-3, "batch_size": 256,
                "folds": N_FOLDS, "epochs_final": ep_mlp,
                "weight_decay": 1e-4,
                "n_params": n_params_mlp})

    # ================= 4) 可视化 + 落盘 =================
    plot_curves(hist_sl, hist_mlp)
    plot_pred(true_te, pred_mlp)

    out = {
        "meta": {
            "script": "04_weight_decay.py",
            "note": "MLP Adam weight_decay=1e-4，其余与 03 协议一致",
            "pipeline": "kfold5-determine-epochs + full-retrain",
            "rows_raw": 8820, "rows_after_dedup": 8760, "rows_clean": 8700,
            "train_rows_final": int(Xtr.shape[0]),
            "test_rows": int(Xte.shape[0]),
            "n_features": int(Xtr.shape[1]),
            "random_state": RANDOM_STATE,
            "device": "cpu",
        },
        "SingleLayerNN": results["SingleLayerNN"],
        "MLP": results["MLP"],
    }
    out_path = INBOX / "results_wd.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n===== weight_decay 消融结果（30% 测试集，原始量纲） =====")
    hdr = f"{'Model':<16}{'MSE':>12}{'RMSE':>10}{'MAE':>9}{'R2':>9}"
    print(hdr); print("-" * len(hdr))
    for n in ("SingleLayerNN", "MLP"):
        r = results[n]
        print(f"{n:<16}{r['MSE']:>12.1f}{r['RMSE']:>10.1f}{r['MAE']:>9.1f}{r['R2']:>9.4f}")
    print(f"\n[落盘] {out_path.relative_to(ROOT)}")
    print("[图] fig13 / fig14 输出至 inbox/figures/")


if __name__ == "__main__":
    main()
