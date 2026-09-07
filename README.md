# 共享单车租赁需求预测（首尔多）

机器学习课程作业：基于首尔多气象与时间数据，按《动手学深度学习》标准流程完成共享单车租赁数量回归预测。

- 数据：Seoul Bike Sharing Demand Data (UCI)，8820 行 × 14 列，时间跨度 2017-12 ~ 2018-11
- 目标：`Rented Bike Count`（每小时租赁量，连续值）
- 流程：EDA → 预处理 → 基线线性模型 → 随机森林/梯度提升 → 交叉验证调优 → 30% 测试集评估

## 运行方式（待补全：后续提交填入）

依次运行：
```
python 01_eda.py
python 02_preprocess.py
python 03_train_evaluate.py
```

## 目录结构（待补全）

- `data/BikeData.csv` 原始数据
- `scripts/` 各阶段脚本（规划中）
- `inbox/figures/` 图表输出
