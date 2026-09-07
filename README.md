# 共享单车租赁需求预测（首尔）

机器学习课程作业：基于首尔多气象与时间数据，按标准流程（EDA → 预处理 → 建模调优 → 测试集评估）完成共享单车每小时租赁量的回归预测。

## ① 项目介绍

- **任务**：回归——预测某小时的共享单车租赁数量（`Rented Bike Count`，0~3556 辆）
- **数据**：[Seoul Bike Sharing Demand Data (UCI)](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)，2017-12-01 ~ 2018-11-30 全年小时级记录，8820 行 × 14 列（含三类需清洗的脏数据，见实现说明）
- **特征**：小时、温度、湿度、风速、能见度、露点、太阳辐射、降雨、降雪、季节、节假日、是否营业日
- **结论先行**：最优模型 XGBoost，30% 测试集 RMSE=144.3 / R²=0.949，完整对比见 `inbox/实验报告.md`

## ② 运行方式

环境：Python 3.11（3.10+ 均可）

```bash
pip install -r requirements.txt

# 按顺序运行（也可用 && 连成一行）
python 01_eda.py            # EDA：4 张图 → inbox/figures/fig1~4
python 02_preprocess.py     # 清洗+特征工程 → data/processed/*.csv
python 03_train_evaluate.py # 建模调优评估 → fig5~8 + inbox/results_summary.json
python 04_neural_network.py # 神经网络对比（追加批）→ fig9~10 + 追加 results_summary.json
```

预期输出：`inbox/figures/` 下 10 张 PNG、`data/processed/` 下 4 个 csv、`inbox/results_summary.json`（8 个模型键）。03 全流程（含三模型 RandomizedSearchCV 调参）约 3~4 分钟；04 为 CPU 训练的 PyTorch 神经网络，约 1 分钟，结果以"追加"方式写入同一份 results_summary.json，重跑幂等。

> 注：脚本的写盘位置为 `脚本所在目录` 下的 `data/` 与 `inbox/`（脚本内以 `Path(__file__).resolve().parent` 推导根目录），请保持目录结构不变。

## ③ 实现说明（关键决策）

1. **数据清洗**（EDA 发现三类脏点，全部实证量化）：
   - `Seasons` 小写 `winter` / `Functioning Day` 取值 `Y` 各 80 行 → 统一大小写
   - 60 组完全重复行 → 删除（去重后 8760 = 365×24 恰为整年，证明重复系注入）
   - 目标列 60 个 -1（租赁量不可能为负）→ 删除；清洗后 8700 行
2. **特征工程**：Date 派生 Month/DayOfWeek/IsWeekend；Hour 用 sin/cos 周期编码（23 点与 0 点相邻）；类别 one-hot（drop_first 防虚拟变量陷阱）
3. **目标变换**：右偏（skew=1.16）→ log1p 训练、expm1 还原评估、clip(0) 非负截断
4. **共线性处理**：corr(温度, 露点)=0.913 > 0.9 → 线性模型剔除露点温度；树模型对共线性稳健，保留全特征
5. **Functioning Day=No**（294 行，3.4%）：保留不删——停运时段"租赁量为 0"是真实业务状态，one-hot 让模型直接学到该规则；删行反而造成分布失真
6. **调参策略**：RandomizedSearchCV（Ridge/Lasso 20 点、RF 15 组、XGB/LGB 20 组），训练集内 5 折 CV（统一 KFold random_state=42），测试集零参与
7. **模型阵容**：LinearRegression / Ridge / Lasso（基线+正则对比）+ RandomForest / XGBoost / LightGBM（非线性主力）；追加批补神经网络——SingleLayerNN（无隐藏层，理论探针）+ MLP 256-128-64（ReLU+BatchNorm+Dropout），详细对比见实验报告第七节
8. **神经网络协议**（追加批）：与 03 同一份数据划分与评估口径（log1p 训练、expm1+clip 还原、原始量纲四指标）；早停验证集从训练集内再切 20%，测试集零参与；torch 种子 42 可复现

## ④ 测试与验证方式

- **复现**：全流程 random_state=42（数据划分、KFold、RandomizedSearchCV、模型种子、torch），重跑脚本应得到与 `inbox/results_summary.json` 完全一致的数字
- **核对**：`inbox/results_summary.json` 含八模型（六经典 + SingleLayerNN + MLP）MSE/RMSE/MAE/R² + 最优超参 + 最优模型残差统计
- **合理性范围**：线性基线 R²≈0.57，树模型 R² 0.93~0.95，最优 RMSE≈144（约为目标均值 704 的 20%）；神经网络：单层应贴着线性基线（R²≈0.57，理论探针）、MLP 预期 0.85~0.93——均低于 XGBoost 属预期结果（数据规模与归纳偏置所致，见实验报告第七节）；若结果偏差大，先查数据清洗是否生效（行数应为 8700）

## ⑤ 已知限制

- **随机划分的时间泄漏**：按课程要求随机 30% 划分，测试集含训练时段的相邻小时，指标偏乐观；严格应做时序前向验证（详见实验报告第六节）
- 未构造滞后特征（t-1 租赁量等）——随机划分下会造成泄漏，需与时序验证配套
- 单城市单年数据，跨城市/跨年泛化未验证
- 未尝试 CatBoost 与 1D-CNN（单层网络与 MLP 已于追加批完成——结果低于 GBDT 属预期，分析见实验报告第七节）

## ⑥ 目录结构

```
├── data/
│   ├── BikeData.csv          # 原始数据（UCI 公开数据集）
│   └── processed/            # 02 产物：X/y × train/test 四个 csv（git 忽略）
├── inbox/
│   ├── figures/              # 全部图表 fig1~fig10
│   ├── 实验报告.md            # 完整中文实验报告
│   └── results_summary.json  # 各模型四指标结构化结果
├── 01_eda.py                 # 阶段一：探索性数据分析
├── 02_preprocess.py          # 阶段二：清洗与特征工程
├── 03_train_evaluate.py      # 阶段三：建模、调参与评估
├── 04_neural_network.py      # 阶段四（追加批）：神经网络对比
├── requirements.txt
└── README.md
```
