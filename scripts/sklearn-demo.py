import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载
df = pd.read_csv('data/inverter_health_samples.csv')

# 2. 探索
print(df.describe()) 
print(df.isnull().sum()) # 检查缺失值

# 3. 可视化相关性
sns.heatmap(df.corr(), annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.show()

df1 = pd.DataFrame({'temp': [50, 60, 70, 80, np.nan,np.nan, 54, 65, 75, 85,np.nan, 100]})
# 向前填充 (适合状态量，如开关状态)
df1['state_filled'] = df1['temp'].fillna(method='ffill') # 用平均值填充缺失值