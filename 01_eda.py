# -*- coding: utf-8 -*-
"""
01_eda.py — 探索性数据分析（EDA）
====================================
共享单车租赁需求预测 | 阶段一

为什么单独做 EDA：
    在建模之前先"认识数据"——看目标分布、时间模式、天气影响，
    才能决定预处理方案（比如哪些特征要变换、异常值怎么处理），
    避免"上来就跑模型"的盲目性。

做什么：
    1) 数据质量体检：脏值（大小写错误 / -1 租赁量 / 重复行）的发现与量化
    2) 目标分布：直方图 + 分位数
    3) 时间模式：按小时 / 星期 / 月份的平均租赁量
    4) 天气影响：天气特征与目标的相关性（含多重共线性检查）
    5) 类别分组：季节 / 节假日 / 营业日的分组统计

输出：
    - inbox/figures/ 下 5 张图（标签用英文，避免中文字体方框问题）
    - 终端打印 EDA 摘要（供报告引用）
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面环境下渲染，脚本化运行必须
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- 路径：相对脚本位置推导，clone 后可直接跑（不写死本机路径） ----------
ROOT = Path(__file__).resolve().parent          # 项目根目录
DATA_PATH = ROOT / "data" / "BikeData.csv"
FIG_DIR = ROOT / "inbox" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

RANDOM_STATE = 42  # 全项目统一随机种子（虽然 EDA 不涉及随机性，统一声明便于复现）


def load_raw() -> pd.DataFrame:
    """读取原始数据。encoding='utf-8-sig' 处理 UTF-8 BOM（防止列名带 \\ufeff）。"""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    return df


def data_quality_audit(df: pd.DataFrame) -> dict:
    """
    数据质量体检：课程数据常被故意埋脏点，先量化再处理。

    为什么逐项检查这些：
    - 大小写不一致：会让 one-hot 编码凭空多出 'winter' 列（与 'Winter' 重复）
    - 负值目标：租赁量不可能为负，属于占位/错误值
    - 完全重复行：同一小时出现两次会放大数据中某些模式的权重
    """
    audit = {}
    audit["shape"] = df.shape
    audit["missing_total"] = int(df.isna().sum().sum())

    # 1) 类别列取值盘点（发现 winter / Y 这类大小写脏值）
    cat_cols = ["Seasons", "Holiday", "Functioning Day"]
    audit["category_values"] = {
        c: df[c].value_counts().to_dict() for c in cat_cols
    }

    # 2) 目标列非法值（负值）
    audit["negative_target"] = int((df["Rented Bike Count"] < 0).sum())

    # 3) 完全重复行
    audit["duplicate_rows"] = int(df.duplicated().sum())

    # 4) 数值范围合理性（温度/湿度等物理常识检查）
    audit["num_summary"] = df.describe().T[["min", "max", "mean"]].round(2).to_dict("index")

    return audit


def plot_target_distribution(df: pd.DataFrame) -> dict:
    """目标分布：右偏明显 → 建模前考虑 log1p 变换（在预处理阶段实施）。"""
    y = df["Rented Bike Count"]
    # 原始数据含 -1 脏值（log1p 会产生 -inf），画分布图前截断到 0；正式清洗在 02_preprocess
    y_plot = y.clip(lower=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(y_plot, bins=60, color="#4C72B0", edgecolor="white")
    axes[0].set(title="Target Distribution (raw)", xlabel="Rented Bike Count", ylabel="Frequency")

    axes[1].hist(np.log1p(y_plot), bins=60, color="#55A868", edgecolor="white")
    axes[1].set(title="Target Distribution (log1p)", xlabel="log1p(Rented Bike Count)", ylabel="Frequency")

    fig.suptitle("Fig.1 Target Distribution: right-skewed, log1p makes it near-normal", y=1.04)
    fig.savefig(FIG_DIR / "fig1_target_distribution.png")
    plt.close(fig)

    skew = y.skew()
    return {"skewness": float(skew), "mean": float(y.mean()), "median": float(y.median())}


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 Date 派生时间特征（EDA 与预处理共用同一套派生逻辑，保证一致性）。

    为什么派生这些：
    - Hour 已有；月/星期/是否周末捕捉周期性需求模式
    - dayofyear 供连续趋势观察
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], format="%d/%m/%Y")
    d["Month"] = d["Date"].dt.month
    d["DayOfWeek"] = d["Date"].dt.dayofweek  # 0=周一 … 6=周日
    d["IsWeekend"] = (d["DayOfWeek"] >= 5).astype(int)
    return d


def plot_time_patterns(df: pd.DataFrame) -> dict:
    """时间模式：小时曲线（通勤双峰？）、星期、月份对比。"""
    d = build_time_features(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    hourly = d.groupby("Hour")["Rented Bike Count"].mean()
    axes[0].plot(hourly.index, hourly.values, marker="o", color="#4C72B0")
    axes[0].set(title="Avg Count by Hour (commute peaks)", xlabel="Hour of day", ylabel="Avg rented bikes")

    dow = d.groupby("DayOfWeek")["Rented Bike Count"].mean()
    axes[1].bar(dow.index, dow.values, color="#55A868")
    axes[1].set(title="Avg Count by Day of Week", xlabel="Day of week (0=Mon)", ylabel="Avg rented bikes")

    monthly = d.groupby("Month")["Rented Bike Count"].mean()
    axes[2].plot(monthly.index, monthly.values, marker="o", color="#C44E52")
    axes[2].set(title="Avg Count by Month (summer peak)", xlabel="Month", ylabel="Avg rented bikes")

    fig.suptitle("Fig.2 Temporal Patterns", y=1.04)
    fig.savefig(FIG_DIR / "fig2_temporal_patterns.png")
    plt.close(fig)

    return {
        "peak_hour": int(hourly.idxmax()),
        "peak_hour_avg": float(hourly.max()),
        "night_avg_0_5": float(hourly.loc[0:5].mean()),
        "month_max": int(monthly.idxmax()),
        "month_min": int(monthly.idxmin()),
    }


def plot_weather_correlation(df: pd.DataFrame) -> dict:
    """天气相关性：数值特征 vs 目标（皮尔逊与斯皮尔曼，后者对非线性和缓的树模型更友好）。"""
    num_cols = [
        "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)",
        "Dew point temperature(°C)", "Solar Radiation (MJ/m2)",
        "Rainfall(mm)", "Snowfall (cm)", "Hour",
    ]
    target = df["Rented Bike Count"]
    pear = {c: float(target.corr(df[c])) for c in num_cols}
    spear = {c: float(target.corr(df[c], method="spearman")) for c in num_cols}

    # 特征间共线性：温度 vs 露点温度（派活单点名检查项）
    collinear = float(df["Temperature(°C)"].corr(df["Dew point temperature(°C)"]))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(num_cols))
    w = 0.38
    ax.bar(x - w/2, [pear[c] for c in num_cols], w, label="Pearson", color="#4C72B0")
    ax.bar(x + w/2, [spear[c] for c in num_cols], w, label="Spearman", color="#DD8452")
    ax.set_xticks(x)
    corr_labels = [c.replace("temperature", "temp.").replace("Radiation", "Rad.").replace("(", "\n(") for c in num_cols]
    ax.set_xticklabels(corr_labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Fig.3 Feature-Target Correlation (Pearson vs Spearman)", ylabel="corr with Rented Bike Count")
    ax.legend()
    fig.savefig(FIG_DIR / "fig3_weather_correlation.png")
    plt.close(fig)

    return {"pearson": pear, "spearman": spear, "corr_temp_dewpoint": collinear}


def plot_category_stats(df: pd.DataFrame) -> dict:
    """类别分组统计：季节 / 节假日 / 营业日 的均值与中位数（均值会被 FD=No 的 0 值拉低）。"""
    d = build_time_features(df)

    def group_stats(gcol: str) -> pd.DataFrame:
        g = d.groupby(gcol)["Rented Bike Count"].agg(["mean", "median", "count"])
        return g

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    s = group_stats("Seasons").reindex(seasons)
    axes[0].bar(s.index, s["mean"], color="#4C72B0", label="mean")
    axes[0].set(title="Avg Count by Season", xlabel="Season", ylabel="Avg rented bikes")

    h = group_stats("Holiday")
    axes[1].bar(["Holiday", "No Holiday"], h["mean"].reindex(["Holiday", "No Holiday"]), color=["#C44E52", "#55A868"])
    axes[1].set(title="Avg Count: Holiday vs Not", xlabel="Holiday", ylabel="Avg rented bikes")

    f = group_stats("Functioning Day")
    axes[1].set_ylim(0, None)
    axes[2].bar(["Yes", "No"], f["mean"].reindex(["Yes", "No"]), color=["#55A868", "#999999"])
    axes[2].set(title="Avg Count: Functioning Day (No ⇒ count=0)", xlabel="Functioning Day", ylabel="Avg rented bikes")

    fig.suptitle("Fig.4 Categorical Group Stats", y=1.04)
    fig.savefig(FIG_DIR / "fig4_category_stats.png")
    plt.close(fig)

    return {
        "season_means": {k: float(v) for k, v in s["mean"].items()},
        "holiday_means": {k: float(v) for k, v in h["mean"].reindex(["Holiday", "No Holiday"]).items()},
        "fd_no_mean": float(f["mean"]["No"]) if "No" in f.index else None,
        "fd_no_rows": int((d["Functioning Day"] == "No").mean() * len(d)),
    }


def main():
    print("=" * 60)
    print("阶段一：探索性数据分析（EDA）")
    print("=" * 60)

    raw = load_raw()
    print(f"\n[读取] {DATA_PATH.name}  shape={raw.shape}")

    # ---- 1) 数据质量体检 ----
    audit = data_quality_audit(raw)
    print("\n---- 1. 数据质量体检 ----")
    print(f"缺失值总数: {audit['missing_total']}")
    print(f"目标负值行: {audit['negative_target']}（租赁量不可能为负 → 预处理删除）")
    print(f"完全重复行: {audit['duplicate_rows']}（去重后应剩 {len(raw) - audit['duplicate_rows']} 行 = 365×24 整一年）")
    for c, vals in audit["category_values"].items():
        odd = {k: v for k, v in vals.items() if k not in {
            "Spring", "Summer", "Autumn", "Winter",
            "Holiday", "No Holiday", "Yes", "No"}}
        if odd:
            print(f"  [{c}] 发现大小写脏值: {odd} → 预处理统一 Title Case / 映射 Y/N")
        else:
            print(f"  [{c}] 取值正常: {list(vals.keys())}")

    # ---- 2) 目标分布 ----
    print("\n---- 2. 目标分布 ----")
    tstat = plot_target_distribution(raw)
    print(f"偏度 skewness={tstat['skewness']:.2f}（>1 显著右偏） mean={tstat['mean']:.0f} median={tstat['median']:.0f}")
    print("log1p 后接近正态 → 回归目标在预处理阶段做 log1p 变换（评估时 np.expm1 还原）")

    # ---- 3) 时间模式 ----
    print("\n---- 3. 时间模式 ----")
    tpat = plot_time_patterns(raw)
    print(f"峰值小时: {tpat['peak_hour']}:00（均值 {tpat['peak_hour_avg']:.0f} 辆）；凌晨 0-5 点均值仅 {tpat['night_avg_0_5']:.0f}")
    print(f"月份: 最高 {tpat['month_max']} 月 / 最低 {tpat['month_min']} 月（冬低夏高）")

    #---- 4) 天气相关性 ----
    print("\n---- 4. 天气与目标的相关性 ----")
    wcorr = plot_weather_correlation(raw)
    print("Pearson  top: " + ", ".join(f"{c}={v:.2f}" for c, v in sorted(wcorr["pearson"].items(), key=lambda kv: -abs(kv[1]))[:4]))
    print("Spearman top: " + ", ".join(f"{c}={v:.2f}" for c, v in sorted(wcorr["spearman"].items(), key=lambda kv: -abs(kv[1]))[:4]))
    print(f"共线性: corr(温度, 露点温度)={wcorr['corr_temp_dewpoint']:.3f}（>0.9 高度共线 → 预处理对线性模型丢弃露点温度）")

    # ---- 5) 类别分组 ----
    print("\n---- 5. 类别分组统计 ----")
    cstats = plot_category_stats(raw)
    print("季节均值: " + ", ".join(f"{k}={v:.0f}" for k, v in cstats["season_means"].items()))
    print(f"节假日: Holiday={cstats['holiday_means']['Holiday']:.0f} vs No Holiday={cstats['holiday_means']['No Holiday']:.0f}")
    print(f"FD=No 行数: {cstats['fd_no_rows'] if cstats['fd_no_rows'] else int((raw['Functioning Day']=='No').sum())}（占比 {(raw['Functioning Day']=='No').mean()*100:.1f}%，租赁量全为 0）")

    print("\n[EDA 完成] 图表输出至 inbox/figures/")


if __name__ == "__main__":
    main()
