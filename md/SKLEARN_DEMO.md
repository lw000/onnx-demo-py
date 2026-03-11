# Sklearn 基础使用示例

## 概述

这是一个 Sklearn 基础使用示例，演示数据加载、探索和缺失值处理的基本操作。

## 功能说明

- CSV 数据加载
- 数据统计探索
- 缺失值检查
- 相关性可视化
- 缺失值填充示例

## 使用方法

```bash
python scripts/sklearn-demo.py
```

## 代码示例

### 1. 数据加载

```python
import pandas as pd

# 加载 CSV 数据
df = pd.read_csv('samples/inverter_health_samples.csv')
```

### 2. 数据探索

```python
# 查看数据统计信息
print(df.describe())

# 检查缺失值
print(df.isnull().sum())
```

### 3. 相关性可视化

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 绘制相关性热力图
sns.heatmap(df.corr(), annot=True)
plt.show()
```

### 4. 缺失值处理

```python
# 向前填充 (适合状态量，如开关状态)
df['state_filled'] = df['temp'].fillna(method='ffill')

# 用平均值填充
df['temp_mean'] = df['temp'].fillna(df['temp'].mean())
```

## 输出示例

```
       time  load_factor  temperature     ripple  label_life  label_thermal
count  60.0    60.000000    60.000000  60.000000   60.000000      60.000000
mean   29.5     0.647143    76.354743   4.673583   86.987313       0.150000
...

time              0
load_factor       0
temperature       0
ripple            0
label_life        0
label_thermal     0
dtype: int64
```

## 文件位置

| 文件 | 路径 |
|------|------|
| 脚本 | `scripts/sklearn-demo.py` |
| 数据 | `samples/inverter_health_samples.csv` |
