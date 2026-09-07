# -*- coding: utf-8 -*-
"""
02_preprocess.py — 数据清洗与特征工程
====================================
共享单车租赁需求预测 | 阶段二

上游：01_eda.py 的发现（三类脏值 + log1p 动机 + 共线性证据）
输出：data/processed/ 下的训练/测试集（磁盘落盘，保证 03 可一键复现）

清洗规则（全部来自 EDA 实证）：
    1) 类别大小写统一：Seasons 'winter'→'Winter'；Functioning Day 'Y'→'Yes'（各 80 行）
    2) 完全重复行删除：60 组 → 去重后 8760 = 365×24 整一年
    3) 目标负值删除：60 行（租赁量不可能为负，属错误记录）
    4) 最终 8700 行

特征工程：
    A. Date 派生：Month / DayOfWeek / IsWeekend（Hour 已有）
    B. Hour 用 sin/cos 周期编码——因为 23 点和 0 点在钟表上相邻，
       直接用原始整数会让线性模型以为它们相距 23 小时（树模型不受影响，
       但统一编码便于多模型对比公平）
    C. 类别 one-hot：Seasons(4)、Holiday(1)、Functioning Day(1)。
       one-hot 而非 ordinal 的理由：季节/节假日无大小序关系，序数编码会
       给树模型之外的线性模型注入假排序信息
    D. 目标 log1p 变换：右偏(skew=1.16)→近正态，改善线性模型拟合与
       梯度提升的损失景观；评估时用 np.expm1 还原到原始量纲
    E. 共线性处理：corr(温度, 露点)=0.913>0.9 → 线性模型丢弃露点温度；
       树模型对共线性稳健且能自选分裂特征，保留露点对比观察
    F. 数值标准化 StandardScaler：线性模型需要；树模型不敏感，但 Pipeline
       内统一标准化不伤害树模型，保证预处理一致性

数据划分：
    30% 测试集，random_state=42。测试集全程隔离，只在 03 的最终评估用一次。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- 路径：相对脚本位置推导（可移植，clone 后直接跑） ----------
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "BikeData.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42  # 全项目统一随机种子

NUM_COLS = [
    "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)",
    "Dew point temperature(°C)", "Solar Radiation (MJ/m2)",
    "Rainfall(mm)", "Snowfall (cm)", "Hour",
]
# 线性模型共线性处理：丢弃露点温度（与温度 corr=0.913）
NUM_COLS_LINEAR = [c for c in NUM_COLS if c != "Dew point temperature(°C)"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """清洗三步：大小写 → 去重 → 删负目标。返回清洗后数据与清理计数。"""
    n0 = len(df)
    # 1) 类别大小写统一（EDA 发现 'winter' 80 行、'Y' 80 行）
    df["Seasons"] = df["Seasons"].str.strip().str.title()
    df["Functioning Day"] = df["Functioning Day"].replace({"Y": "Yes", "N": "No"})
    # 2) 完全重复行
    df = df.drop_duplicates()
    n_dup = n0 - len(df)
    # 3) 目标负值（-1 为错误记录，租赁量 >= 0）
    df = df[df["Rented Bike Count"] >= 0]
    n_neg = n0 - n_dup - len(df)
    print(f"[清洗] {n0} → 修大小写 → 去重(-{n_dup}) → 删负目标(-{n_neg}) → {len(df)} 行")
    return df.reset_index(drop=True)


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程：Date 派生 + Hour 周期编码 + one-hot。返回特征矩阵（含目标列）。"""
    d = df.copy()
    # A. Date 派生
    dt = pd.to_datetime(d["Date"], format="%d/%m/%Y")
    d["Month"] = dt.dt.month
    d["DayOfWeek"] = dt.dt.dayofweek            # 0=周一 … 6=周日
    d["IsWeekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # B. Hour 周期编码：sin/cos 让 23 点与 0 点在特征空间相邻
    d["Hour_sin"] = np.sin(2 * np.pi * d["Hour"] / 24)
    d["Hour_cos"] = np.cos(2 * np.pi * d["Hour"] / 24)

    # C. 类别 one-hot（drop_first 避免虚拟变量陷阱，线性模型尤需）
    d = pd.get_dummies(
        d, columns=["Seasons", "Holiday", "Functioning Day"],
        drop_first=True, dtype=int,
    )
    return d


def main():
    print("=" * 60)
    print("阶段二：数据清洗与特征工程")
    print("=" * 60)

    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    print(f"[读取] {RAW_PATH.name}  shape={raw.shape}")

    df = clean(raw)
    df = feature_engineer(df)
    print(f"[特征工程] 派生 Month/DayOfWeek/IsWeekend + Hour sin/cos + one-hot → {df.shape[1]-1} 个特征")

    # ---- 划分：30% 测试集，random_state 固定保证可复现 ----
    # Date 已派生出 Month/DayOfWeek/IsWeekend，原始字符串列不进模型
    d_fe = df.drop(columns=["Date"])
    y = np.log1p(d_fe.pop("Rented Bike Count"))   # D. 目标 log1p
    X = d_fe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    print(f"[划分] 训练 {X_train.shape[0]} 行 / 测试 {X_test.shape[0]} 行（30%，random_state=42）")

    # ---- 落盘（列名含特殊字符，用 csv 保留原列名；03 重新读取） ----
    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_frame().to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_frame().to_csv(OUT_DIR / "y_test.csv", index=False)
    print(f"[落盘] {OUT_DIR.relative_to(ROOT)}\\ 下 4 个 csv（X/y × train/test）")
    print("[列清单]", list(X_train.columns))


if __name__ == "__main__":
    main()
