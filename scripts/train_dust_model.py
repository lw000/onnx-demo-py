"""
煤矿粉尘测点趋势预测模型 - 训练脚本

================================================================================
模型信息
================================================================================
- 模型名称: dust_prediction_model.onnx
- 模型类型: 时间序列回归模型
- 算法: Random Forest Regressor
- 特征数量: 20维 (4个当前特征 + 12个滞后特征 + 4个环境特征)
- 输出: PM10浓度预测值 (μg/m³)

================================================================================
功能说明
================================================================================
本模型用于预测煤矿井下粉尘浓度(PM10)，通过引入时间滞后特征来模拟粉尘
从产生到被传感器检测的运移延迟，实现更精确的趋势预测。

核心特性:
- 多测点监测: 支持进风侧、主作业区、回风侧等多点数据
- 滞后特征: 捕捉粉尘扩散的惯性和滞后性
- 环境关联: 考虑风速、湿度、温度等因素的影响

================================================================================
特征定义
================================================================================

【基本特征 - 4个传感器 × 1个时间点 = 4维】
┌─────────────┬──────────────────────┬────────────────────────────┐
│  特征名      │  说明                 │  取值范围                  │
├─────────────┼──────────────────────┼────────────────────────────┤
│  pm10_s0     │  进风侧 PM10 浓度    │  5-500 μg/m³              │
│  pm10_s1     │  主作业区 PM10 浓度  │  5-500 μg/m³ (预测目标)  │
│  pm10_s2     │  回风侧中部 PM10     │  5-500 μg/m³              │
│  pm10_s3     │  回风侧 PM10 浓度    │  5-500 μg/m³              │
└─────────────┴──────────────────────┴────────────────────────────┘

【衍生特征 - 滞后特征: 4个传感器 × 3个时间步 = 12维】
┌─────────────┬──────────────────────┬────────────────────────────┐
│  特征名      │  说明                 │  含义                      │
├─────────────┼──────────────────────┼────────────────────────────┤
│ pm10_lag1_* │  1个时间步滞后        │  ~10秒前 (采样间隔)        │
│ pm10_lag2_* │  2个时间步滞后        │  ~20秒前                   │
│ pm10_lag3_* │  3个时间步滞后        │  ~30秒前                   │
└─────────────┴──────────────────────┴────────────────────────────┘
* = s0, s1, s2, s3 (对应4个传感器)

【环境特征 - 4维】
┌─────────────┬──────────────────────┬────────────────────────────┐
│  特征名      │  说明                 │  取值范围                  │
├─────────────┼──────────────────────┼────────────────────────────┤
│  wind_speed │  风速                 │  0.5-5.0 m/s               │
│  humidity   │  相对湿度             │  20-95 %                   │
│  temp       │  环境温度             │  15-35 °C                  │
│  machine_on │  采煤机状态           │  0 (停机) / 1 (运行)       │
└─────────────┴──────────────────────┴────────────────────────────┘

================================================================================
特征顺序 (C++推理必须按此顺序)
================================================================================
索引  特征名              索引  特征名
──────────────────────────────────────
0     pm10_s0            10    pm10_lag3_s2
1     pm10_s1            11    pm10_lag3_s3
2     pm10_s2            12    wind_speed
3     pm10_s3            13    humidity
4     pm10_lag1_s0       14    temp
5     pm10_lag1_s1       15    machine_on
6     pm10_lag1_s2       
7     pm10_lag1_s3       
8     pm10_lag2_s0       
9     pm10_lag2_s1       
10    pm10_lag2_s2       
11    pm10_lag2_s3       

================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# ==========================================
# 配置
# ==========================================
N_SENSORS = 4      # 传感器数量 (进风侧 s0, 主作业区 s1, 回风侧中部 s2, 回风侧 s3)
LAG_STEPS = 3      # 滞后时间步数 (模拟粉尘运移延迟)

# 目录配置
base_dir = os.path.dirname(os.path.dirname(__file__))
model_dir = os.path.join(base_dir, "models")
samples_dir = os.path.join(base_dir, "data")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

# ==========================================
# 1. 时间序列数据生成与滞后特征构建
# ==========================================

def generate_time_series_data(n_timepoints=5000, n_sensors=4, lag_steps=3):
    """
    模拟煤矿多测点时间序列传感器数据
    
    参数:
        n_timepoints: 时间序列长度
        n_sensors: 传感器数量 (进风侧、主作业区、回风侧等)
        lag_steps: 滞后时间步数 (模拟粉尘运移延迟)
    
    返回:
        DataFrame: 包含原始特征和滞后特征的完整数据
    """
    np.random.seed(42)
    
    # 时间序列模拟：考虑粉尘扩散的惯性
    dt = 1  # 时间间隔 (秒)
    
    # 初始化状态
    pm10_history = np.zeros((n_timepoints, n_sensors))
    wind_speed_history = np.zeros(n_timepoints)
    humidity_history = np.zeros(n_timepoints)
    temp_history = np.zeros(n_timepoints)
    machine_on_history = np.zeros(n_timepoints)
    
    # 初始条件
    pm10_history[0] = np.random.uniform(20, 80, n_sensors)
    wind_speed_history[0] = 2.5
    humidity_history[0] = 50.0
    temp_history[0] = 22.0
    
    # 模拟采煤作业周期 (工作4小时，休息1小时)
    cycle_length = 300  # 5分钟一个周期
    work_ratio = 0.8
    
    for t in range(1, n_timepoints):
        # 采煤机状态 (周期性变化)
        cycle_pos = t % cycle_length
        machine_on = 1 if (cycle_pos < cycle_length * work_ratio) else 0
        machine_on_history[t] = machine_on
        
        # 风机状态 (通常持续运行，但有轻微波动)
        fan_on = 1 if (machine_on or np.random.random() > 0.05) else 0
        
        # 风速: 采煤机开启时风速略低 (人员作业区优化)
        base_wind = 2.5 + fan_on * 0.5
        wind_speed_history[t] = base_wind + np.random.normal(0, 0.3)
        wind_speed_history[t] = np.clip(wind_speed_history[t], 0.5, 5.0)
        
        # 湿度: 洒水系统会提高湿度
        spray_on = machine_on * np.random.choice([0, 1], p=[0.3, 0.7])
        humidity_history[t] = humidity_history[t-1] + spray_on * 2.0 - 0.5
        humidity_history[t] = np.clip(humidity_history[t], 20, 95)
        
        # 温度: 采煤机产热
        temp_history[t] = temp_history[t-1] + machine_on * 0.1 - 0.05
        temp_history[t] = np.clip(temp_history[t], 15, 35)
        
        # 粉尘浓度: 考虑惯性、滞后和空间传播
        for s in range(n_sensors):
            # 传感器位置因子: 越靠近采煤机，浓度越高
            pos_factor = 1.0 + (s / n_sensors) * 0.5
            
            # 粉尘产生: 采煤机截割产生粉尘
            dust_source = machine_on * 150 * pos_factor
            
            # 粉尘扩散: 风速将粉尘推向回风侧 (从进风口到回风侧)
            if s > 0:
                # 前一测点的粉尘扩散到当前测点 (滞后效应)
                diffusion_in = pm10_history[t-1, s-1] * 0.3
            else:
                diffusion_in = 0
            
            # 粉尘消散: 风速越大，消散越快
            dissipation = wind_speed_history[t] * 20
            
            # 粉尘沉降: 湿度越高，沉降越快
            settlement = humidity_history[t] * 0.3
            
            # 历史惯性: 当前浓度与上一时刻相关
            inertia = pm10_history[t-1, s] * 0.85  # 惯性系数
            
            # 随机扰动
            noise = np.random.normal(0, 5)
            
            # 更新浓度
            pm10_history[t, s] = (
                inertia + dust_source + diffusion_in - 
                dissipation - settlement + noise
            )
            pm10_history[t, s] = max(pm10_history[t, s], 5)
    
    # 构建特征矩阵
    records = []
    for t in range(lag_steps, n_timepoints):
        record = {}
        
        # 当前时刻各测点PM10
        for s in range(n_sensors):
            record[f'pm10_s{s}'] = pm10_history[t, s]
        
        # 滞后特征: 反映粉尘运移延迟
        for lag in range(1, lag_steps + 1):
            for s in range(n_sensors):
                record[f'pm10_lag{lag}_s{s}'] = pm10_history[t - lag, s]
        
        # 当前环境参数
        record['wind_speed'] = wind_speed_history[t]
        record['humidity'] = humidity_history[t]
        record['temp'] = temp_history[t]
        record['machine_on'] = machine_on_history[t]
        
        # 目标: 预测下一时刻主作业区(s=1)的PM10
        record['target_pm10'] = pm10_history[t + 1, 1] if t + 1 < n_timepoints else pm10_history[t, 1]
        
        records.append(record)
    
    df = pd.DataFrame(records)
    return df, pm10_history, wind_speed_history, humidity_history

# 生成时间序列数据
lag_steps = 3  # 3个时间步的滞后 (如传感器采样间隔为10秒，则对应30秒延迟)
n_timepoints = 5000

print(f"生成 {n_timepoints} 个时间点的传感器数据...")
print(f"滞后特征步数: {lag_steps} (模拟粉尘运移延迟)")

df, pm10_history, wind_history, humidity_history = generate_time_series_data(
    n_timepoints=n_timepoints, 
    n_sensors=4, 
    lag_steps=lag_steps
)

print(f"数据集大小: {df.shape}")
print(f"特征数量: {df.shape[1] - 1}")  # 减去目标列

# 保存训练数据
csv_path = os.path.join(samples_dir, "dust_time_series_samples.csv")
df.to_csv(csv_path, index=False)
print(f"训练数据已保存: {csv_path}")

# 分离特征和目标
feature_cols = [col for col in df.columns if col != 'target_pm10']
X = df[feature_cols].values
y = df['target_pm10'].values.reshape(-1, 1)

# 特征名称列表
feature_names = feature_cols

# 划分训练集和测试集 (按时间顺序，避免数据泄露)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# ==========================================
# 2. 数据标准化
# ==========================================
print("\n标准化特征...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)

# 保存标准化器
import pickle
scaler_path = os.path.join(model_dir, "dust_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y, 'feature_names': feature_names}, f)
print(f"标准化器已保存: {scaler_path}")

# ==========================================
# 3. 模型训练 (sklearn)
# ==========================================
print("\n正在训练随机森林模型...")
model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train_scaled.ravel())

# 评估
train_score = model.score(X_train_scaled, y_train_scaled)
test_score = model.score(X_test_scaled, y_test)
print(f"训练集 R^2 分数: {train_score:.4f}")
print(f"测试集 R^2 分数: {test_score:.4f}")

# 特征重要性
print("\n--- 特征重要性 Top 10 ---")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

# ==========================================
# 4. 导出为 ONNX 模型
# ==========================================
print("\n正在导出 ONNX 模型...")

# 定义输入张量的名称和类型
# 输入: [Batch_Size, n_features] - 包含当前时刻和滞后特征
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

# 转换模型
onnx_model = convert_sklearn(model, initial_types=initial_type)

# 保存模型
model_filename = os.path.join(model_dir, "dust_prediction_model.onnx")
with open(model_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"模型已成功保存为: {model_filename}")

# ==========================================
# 5. Python 验证 ONNX 推理
# ==========================================
import onnxruntime as ort

print("\n--- 验证 ONNX 推理 ---")

# 加载并验证模型
ort_session = ort.InferenceSession(model_filename)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

print(f"模型输入名称: {input_name}")
print(f"模型输出名称: {output_name}")
print(f"输入形状: {ort_session.get_inputs()[0].shape}")
print(f"输出形状: {ort_session.get_outputs()[0].shape}")

# 准备测试数据
test_input = X_test_scaled[:5].astype(np.float32)

# ONNX 推理
ort_inputs = {input_name: test_input}
ort_outputs = ort_session.run([output_name], ort_inputs)
prediction_scaled = ort_outputs[0]

# 反归一化预测结果 (y_test 已是原始值，无需转换)
prediction_real = scaler_y.inverse_transform(prediction_scaled)

print(f"\n--- 预测结果对比 ---")
for i in range(5):
    actual = y_test[i, 0]  # y_test 已是原始值
    diff = abs(prediction_real[i, 0] - actual)
    print(f"样本 {i+1}: 预测={prediction_real[i,0]:.1f}, 实际={actual:.1f}, 误差={diff:.1f}")

# 计算整体误差
from sklearn.metrics import mean_absolute_error, mean_squared_error
y_pred_scaled = model.predict(X_test_scaled)
y_pred_all = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
mae = mean_absolute_error(y_test, y_pred_all)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_all))
print(f"\n整体误差: MAE={mae:.2f}, RMSE={rmse:.2f}")

# 分析数据分布
print("\n--- 数据分布分析 ---")
print(f"训练集目标范围: [{y_train.min():.1f}, {y_train.max():.1f}]")
print(f"测试集目标范围: [{y_test.min():.1f}, {y_test.max():.1f}]")
print(f"测试集预测范围: [{y_pred_all.min():.1f}, {y_pred_all.max():.1f}]")

# ==========================================
# 6. 滞后特征说明与推理示例
# ==========================================
print("\n" + "="*60)
print("滞后特征说明")
print("="*60)
print("""
粉尘从产生到被传感器检测存在运移延迟。本模型通过引入
滞后特征来捕捉这一物理过程：

特征命名规则:
  - pm10_s0 ~ pm10_s3 : 当前时刻各测点PM10浓度
  - pm10_lag1_s0      : 1个时间步滞后的PM10浓度
  - pm10_lag2_s0      : 2个时间步滞后的PM10浓度
  - pm10_lag3_s0      : 3个时间步滞后的PM10浓度

测点分布:
  - s0: 进风侧 (远离作业区)
  - s1: 主作业区 (预测目标)
  - s2: 回风侧中部
  - s3: 回风侧

推理时需要提供:
  1. 当前时刻各测点传感器数据
  2. 前1、2、3个时间步的历史数据
  3. 当前环境参数 (风速、湿度、温度、设备状态)
""")

# 保存特征顺序说明
feature_order_path = os.path.join(model_dir, "dust_feature_order.txt")
with open(feature_order_path, 'w') as f:
    f.write("特征顺序说明 (用于C++推理):\n\n")
    for i, name in enumerate(feature_names):
        f.write(f"{i}: {name}\n")
print(f"特征顺序已保存: {feature_order_path}")