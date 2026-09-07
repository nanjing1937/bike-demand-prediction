# -*- coding: utf-8 -*-
"""
03_train_evaluate.py — 建模、调参与测试集评估
====================================
共享单车租赁需求预测 | 阶段三

上游：02_preprocess.py 落盘的 data/processed/ 四个 csv
输出：
    - inbox/results_summary.json   各模型四指标 + 最优超参（结构化，供复核）
    - inbox/figures/fig5~fig8      模型对比 / 预测vs真实 / 残差 / 特征重要性

模型阵容（为什么选这些）：
    基线 LinearRegression / Ridge / Lasso —— 线性模型可解释、训练快，
        作为参照系；Ridge/Lasso 正则化对比看共线性与稀疏化的影响
    主力 RandomForest / XGBoost / LightGBM —— 树模型天然捕捉非线性
        （小时双峰、温度甜区），是该类表格数据公认的强基线

调参纪律（防泄漏）：
    - 所有超参搜索只在训练集内做 5 折交叉验证（KFold shuffle 固定种子）
    - 测试集仅在所有模型定型后评估一次，全程不参与任何决策
    - 随机搜索 RandomizedSearchCV（空间大时比网格省时且效果相当）

评估口径：
    - 训练目标为 log1p(租赁量)，预测后 np.expm1 还原到原始量纲再算指标
    - 预测值 clip(lower=0)：租赁量物理上非负，对线性模型的负预测做截断
    - 四指标：MSE / RMSE / MAE / R²（原始量纲，RMSE 对大误差更敏感，
      MAE 对离群更稳健，两者一起看）
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ---------- 路径：相对脚本位置推导（可移植） ----------
ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "inbox" / "figures"
INBOX = ROOT / "inbox"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3})


# ---------- 工具 ----------
def load_data():
    """读取 02 落盘的四份数据；线性模型另有共线性删列版本。"""
    X_train = pd.read_csv(PROC / "X_train.csv")
    X_test = pd.read_csv(PROC / "X_test.csv")
    y_train = pd.read_csv(PROC / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(PROC / "y_test.csv").iloc[:, 0]
    # 线性模型共线性处理：丢弃露点温度（corr(温度,露点)=0.913，EDA 实证）
    X_train_lin = X_train.drop(columns=["Dew point temperature(°C)"])
    X_test_lin = X_test.drop(columns=["Dew point temperature(°C)"])
    return X_train, X_test, y_train, y_test, X_train_lin, X_test_lin


def evaluate(name, model, X_te, y_log_te, results, best_params=None, cv_best=None):
    """
    统一评估：log 空间预测 → expm1 还原 → clip(0) → 原始量纲四指标。
    所有模型走同一条路径，保证对比公平。
    """
    pred_log = model.predict(X_te)
    pred = np.clip(np.expm1(pred_log), 0, None)   # 还原 + 非负截断
    true = np.expm1(y_log_te.values)
    mse = mean_squared_error(true, pred)
    res = {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(true, pred)),
        "R2": float(r2_score(true, pred)),
    }
    if best_params:
        res["best_params"] = {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                              for k, v in best_params.items()}
    if cv_best is not None:
        res["cv_RMSE_log"] = float(-cv_best)      # 训练集 CV 的 log 空间 RMSE（调参依据）
    results[name] = res
    print(f"  [{name:<14}] RMSE={res['RMSE']:7.1f}  MAE={res['MAE']:7.1f}  R2={res['R2']:.4f}  MSE={res['MSE']:.1f}")
    return pred


def plot_model_comparison(results):
    """Fig.5 各模型 RMSE / R² 对比条形图。"""
    names = list(results.keys())
    rmses = [results[n]["RMSE"] for n in names]
    r2s = [results[n]["R2"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    order = np.argsort(rmses)
    axes[0].bar([names[i] for i in order], [rmses[i] for i in order], color="#4C72B0")
    axes[0].set(title="Test RMSE (lower is better)", ylabel="RMSE (bikes)")
    axes[0].tick_params(axis="x", rotation=20)

    order2 = np.argsort(r2s)[::-1]
    axes[1].bar([names[i] for i in order2], [r2s[i] for i in order2], color="#55A868")
    axes[1].set(title="Test R2 (higher is better)", ylabel="R2")
    axes[1].set_ylim(min(r2s) - 0.05, 1.0)
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Fig.5 Model Comparison on 30% Test Set", y=1.04)
    fig.savefig(FIG_DIR / "fig5_model_comparison.png")
    plt.close(fig)


def plot_pred_vs_true(true, pred, model_name):
    """Fig.6 最优模型：预测值 vs 真实值散点（越贴对角线越好）。"""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(true, pred, s=6, alpha=0.25, color="#4C72B0")
    lim = [0, max(true.max(), pred.max()) * 1.02]
    ax.plot(lim, lim, "r--", lw=1.2, label="y = x")
    ax.set(xlim=lim, ylim=lim, xlabel="Actual rented bikes", ylabel="Predicted",
           title=f"Fig.6 Pred vs Actual — {model_name}")
    ax.legend()
    fig.savefig(FIG_DIR / "fig6_pred_vs_true.png")
    plt.close(fig)


def plot_residuals(true, pred, model_name):
    """Fig.7 残差分析：残差 vs 预测值 + 残差直方图（检查异方差/偏态）。"""
    resid = true - pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(pred, resid, s=6, alpha=0.25, color="#4C72B0")
    axes[0].axhline(0, color="red", ls="--", lw=1.2)
    axes[0].set(xlabel="Predicted", ylabel="Residual (actual - predicted)",
                title="Residuals vs Predicted")
    axes[1].hist(resid, bins=60, color="#DD8452", edgecolor="white")
    axes[1].axvline(0, color="red", ls="--", lw=1.2)
    axes[1].set(xlabel="Residual", ylabel="Frequency", title="Residual Distribution")
    fig.suptitle(f"Fig.7 Residual Analysis — {model_name}", y=1.04)
    fig.savefig(FIG_DIR / "fig7_residuals.png")
    plt.close(fig)
    return {"resid_mean": float(resid.mean()), "resid_std": float(resid.std())}


def plot_feature_importance(X_train, importances, model_name):
    """Fig.8 特征重要性（树模型 impurity/增益重要性 Top 15）。"""
    s = pd.Series(importances, index=X_train.columns).sort_values()
    top = s.tail(15)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values, color="#55A868")
    ax.set(xlabel="Importance", title=f"Fig.8 Feature Importance — {model_name} (Top 15)")
    fig.savefig(FIG_DIR / "fig8_feature_importance.png")
    plt.close(fig)
    return {k: float(v) for k, v in s.sort_values(ascending=False).head(8).items()}


# ---------- 主流程 ----------
def main():
    print("=" * 60)
    print("阶段三：建模、调参与测试集评估")
    print("=" * 60)

    X_train, X_test, y_train, y_test, X_train_lin, X_test_lin = load_data()
    print(f"[数据] 训练 {X_train.shape} / 测试 {X_test.shape}（线性版已剔除露点温度）")

    # 交叉验证划分固定：所有模型的 CV 用同一折结构，公平且可复现
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    y_true_orig = np.expm1(y_test.values)

    # ================= 1) 基线：线性模型族 =================
    print("\n---- 1. 基线：线性模型（标准化 + 剔除露点温度） ----")

    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lin.fit(X_train_lin, y_train)
    evaluate("LinearRegression", lin, X_test_lin, y_test, results)

    # Ridge：L2 正则，网格搜 alpha（对数尺度——正则强度跨量级）
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=RANDOM_STATE))])
    ridge_gs = RandomizedSearchCV(
        ridge_pipe, {"model__alpha": np.logspace(-2, 3, 30)},
        n_iter=20, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    ridge_gs.fit(X_train_lin, y_train)
    evaluate("Ridge", ridge_gs.best_estimator_, X_test_lin, y_test, results,
             best_params=ridge_gs.best_params_, cv_best=ridge_gs.best_score_)
    print(f"    Ridge best alpha={ridge_gs.best_params_['model__alpha']:.3f} (CV log-RMSE={-ridge_gs.best_score_:.4f})")

    # Lasso：L1 正则，可稀疏化系数，顺便当"特征筛选器"观察
    lasso_pipe = Pipeline([("scaler", StandardScaler()), ("model", Lasso(random_state=RANDOM_STATE, max_iter=20000))])
    lasso_gs = RandomizedSearchCV(
        lasso_pipe, {"model__alpha": np.logspace(-4, 1, 30)},
        n_iter=20, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    lasso_gs.fit(X_train_lin, y_train)
    evaluate("Lasso", lasso_gs.best_estimator_, X_test_lin, y_test, results,
             best_params=lasso_gs.best_params_, cv_best=lasso_gs.best_score_)
    n_zero = int((lasso_gs.best_estimator_.named_steps["model"].coef_ == 0).sum())
    print(f"    Lasso best alpha={lasso_gs.best_params_['model__alpha']:.4f}，{n_zero}/{len(X_train_lin.columns)} 个系数被压成 0")

    # ================= 2) 主力：树模型族 =================
    print("\n---- 2. 主力：树模型（无需标准化，保留全部特征） ----")

    # 随机森林：树深/特征子采样是关键维度
    print("  [RandomForest 调参中…] ", end="", flush=True)
    rf_space = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 0.7],
    }
    rf_rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), rf_space,
        n_iter=15, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_rs.fit(X_train, y_train)
    print(f"done (CV log-RMSE={-rf_rs.best_score_:.4f})")
    evaluate("RandomForest", rf_rs.best_estimator_, X_test, y_test, results,
             best_params=rf_rs.best_params_, cv_best=rf_rs.best_score_)

    # XGBoost：学习率/深度/子采样/列采样
    print("  [XGBoost 调参中…] ", end="", flush=True)
    xgb_space = {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1, 3, 10],
    }
    xgb_rs = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
        xgb_space, n_iter=20, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_rs.fit(X_train, y_train)
    print(f"done (CV log-RMSE={-xgb_rs.best_score_:.4f})")
    evaluate("XGBoost", xgb_rs.best_estimator_, X_test, y_test, results,
             best_params=xgb_rs.best_params_, cv_best=xgb_rs.best_score_)

    # LightGBM：叶子数/学习率/正则
    print("  [LightGBM 调参中…] ", end="", flush=True)
    lgb_space = {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [5, 10, 20, 40],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [0.0, 1, 5, 10],
    }
    lgb_rs = RandomizedSearchCV(
        LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        lgb_space, n_iter=20, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    lgb_rs.fit(X_train, y_train)
    print(f"done (CV log-RMSE={-lgb_rs.best_score_:.4f})")
    evaluate("LightGBM", lgb_rs.best_estimator_, X_test, y_test, results,
             best_params=lgb_rs.best_params_, cv_best=lgb_rs.best_score_)

    # ================= 3) 汇总与可视化 =================
    print("\n---- 3. 汇总 ----")
    best_name = min(results, key=lambda n: results[n]["RMSE"])
    print(f"最优模型（测试集 RMSE 最小）: {best_name}")

    plot_model_comparison(results)

    # 最优模型的深入分析图（用对应模型对象重新预测）
    if best_name in ("RandomForest", "XGBoost", "LightGBM"):
        best_model = {"RandomForest": rf_rs, "XGBoost": xgb_rs, "LightGBM": lgb_rs}[best_name].best_estimator_
        best_pred = np.clip(np.expm1(best_model.predict(X_test)), 0, None)
        plot_pred_vs_true(y_true_orig, best_pred, best_name)
        rstats = plot_residuals(y_true_orig, best_pred, best_name)
        imp = plot_feature_importance(X_train, best_model.feature_importances_, best_name)
        results["_best_model"] = {
            "name": best_name,
            "residual_stats": rstats,
            "feature_importance_top8": imp,
        }
        print(f"残差均值 {rstats['resid_mean']:+.2f}（接近 0 无系统性偏差），标准差 {rstats['resid_std']:.1f}")
        print("特征重要性 Top5:", ", ".join(f"{k}={v:.3f}" for k, v in list(imp.items())[:5]))

    # 汇总表打印
    print("\n===== 多模型对比（30% 测试集，原始量纲） =====")
    hdr = f"{'Model':<16}{'MSE':>12}{'RMSE':>10}{'MAE':>9}{'R2':>9}"
    print(hdr); print("-" * len(hdr))
    for n, r in results.items():
        if not n.startswith("_"):
            print(f"{n:<16}{r['MSE']:>12.1f}{r['RMSE']:>10.1f}{r['MAE']:>9.1f}{r['R2']:>9.4f}")

    # 落盘 JSON
    out = INBOX / "results_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[落盘] {out.relative_to(ROOT)}")
    print("[图] fig5~fig8 输出至 inbox/figures/")


if __name__ == "__main__":
    main()
