# 煤矿粉尘测点趋势预测模型

## 模型概述

本模型用于预测煤矿井下粉尘浓度（PM10），通过引入**时间滞后特征**模拟粉尘从产生到被传感器检测的运移延迟，实现更精确的趋势预测。

### 核心改进：时间滞后特征

粉尘扩散和沉降具有显著的惯性和滞后性：
- 采煤机截割产生的粉尘不会瞬间到达回风侧传感器
- 粉尘需要随风流运移一段时间才能被检测到
- 本模型通过引入滞后特征捕捉这一物理过程

### 应用场景

- 煤矿井下粉尘浓度监测
- 采煤作业粉尘预警
- 通风系统优化决策
- 职业健康风险评估

## 模型信息

| 属性 | 值 |
|------|-----|
| 模型类型 | 时间序列回归模型 |
| 算法 | Random Forest Regressor |
| 特征数量 | 25 |
| 滞后步数 | 3 |
| 输出 | PM10 浓度预测值 (μg/m³) |
| 训练样本 | ~4000 |
| ONNX 大小 | ~500 KB |

## 特征说明

### 测点分布（4个传感器）

| 测点 | 位置 | 说明 |
|------|------|------|
| s0 | 进风侧 | 远离作业区，粉尘浓度较低 |
| s1 | 主作业区 | 预测目标位置 |
| s2 | 回风侧中部 | 粉尘扩散区域 |
| s3 | 回风侧 | 粉尘浓度最高区域 |

### 特征列表（25维）

#### 当前时刻特征 (7个)
| 特征名 | 说明 | 取值范围 |
|--------|------|----------|
| pm10_s0 | 进风侧 PM10 | 5-500 μg/m³ |
| pm10_s1 | 主作业区 PM10 | 5-500 μg/m³ |
| pm10_s2 | 回风侧中部 PM10 | 5-500 μg/m³ |
| pm10_s3 | 回风侧 PM10 | 5-500 μg/m³ |
| wind_speed | 风速 | 0.5-5.0 m/s |
| humidity | 相对湿度 | 20-95 % |
| temp | 环境温度 | 15-35 °C |
| machine_on | 采煤机状态 | 0/1 |

#### 滞后特征 (18个)

| 特征名 | 说明 |
|--------|------|
| pm10_lag1_s0 ~ pm10_lag1_s3 | 1个时间步滞后的PM10 (如10秒前) |
| pm10_lag2_s0 ~ pm10_lag2_s3 | 2个时间步滞后的PM10 (如20秒前) |
| pm10_lag3_s0 ~ pm10_lag3_s3 | 3个时间步滞后的PM10 (如30秒前) |

### 滞后特征物理意义

```
时间轴示意 (假设采样间隔10秒):

t-30s: pm10_lag3  ──┐
t-20s: pm10_lag2  ──┼── 捕捉粉尘运移轨迹
t-10s: pm10_lag1  ──┤
t=0:   当前时刻   ──┴── 预测目标: t+10s 的PM10
```

## 使用方法

### 训练模型

```bash
python scripts/train_dust_model.py
```

### Python 推理

```python
import numpy as np
import pickle
import onnxruntime as ort

# 加载标准化器
with open('models/dust_scaler.pkl', 'rb') as f:
    scaler_data = pickle.load(f)
    scaler_X = scaler_data['scaler_X']
    scaler_y = scaler_data['scaler_y']
    feature_names = scaler_data['feature_names']

# 加载模型
session = ort.InferenceSession('models/dust_prediction_model.onnx')

# 准备输入数据 (25个特征)
# 特征顺序: [pm10_s0, pm10_s1, pm10_s2, pm10_s3,
#            pm10_lag1_s0, pm10_lag1_s1, ..., pm10_lag3_s3,
#            wind_speed, humidity, temp, machine_on]
input_data = np.array([[
    # 当前时刻各测点PM10
    50.0, 80.0, 120.0, 180.0,
    # 滞后1步
    45.0, 75.0, 110.0, 160.0,
    # 滞后2步
    42.0, 70.0, 100.0, 145.0,
    # 滞后3步
    40.0, 65.0, 95.0, 130.0,
    # 环境参数
    2.5, 55.0, 23.0, 1
]]).astype(np.float32)

# 标准化
input_scaled = scaler_X.transform(input_data)

# 推理
output = session.run(None, {'float_input': input_scaled})
prediction_scaled = output[0]

# 反标准化
prediction = scaler_y.inverse_transform(prediction_scaled)
print(f"预测 PM10 浓度: {prediction[0,0]:.1f} μg/m³")
```

### C++ 推理

```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

int main() {
    // 创建会话
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dust_prediction");
    Ort::SessionOptions session_options;
    Ort::Session session(env, L"models/dust_prediction_model.onnx", session_options);

    // 输入数据 (25个特征，float32)
    std::vector<float> input_values = {
        // 当前时刻各测点PM10
        50.0f, 80.0f, 120.0f, 180.0f,
        // 滞后1步
        45.0f, 75.0f, 110.0f, 160.0f,
        // 滞后2步
        42.0f, 70.0f, 100.0f, 145.0f,
        // 滞后3步
        40.0f, 65.0f, 95.0f, 130.0f,
        // 环境参数
        2.5f, 55.0f, 23.0f, 1.0f
    };
    std::vector<int64_t> input_shape = {1, 25};

    // 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_values.data(), input_values.size(),
        input_shape.data(), input_shape.size());

    // 推理
    const char* input_names[] = {"float_input"};
    const char* output_names[] = {"variable"};
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
        output_names, 1);

    float predicted_pm10 = output_tensors[0].GetTensorMutableData<float>()[0];
    std::cout << "预测 PM10 浓度: " << predicted_pm10 << " μg/m³" << std::endl;

    return 0;
}
```

## 模型原理

### 粉尘扩散物理模型

```
PM10(t) = 惯性项 + 粉尘源项 + 扩散项 - 消散项 - 沉降项 + 噪声
```

| 物理过程 | 数学表达 | 说明 |
|----------|----------|------|
| 惯性 | PM10(t-1) × 0.85 | 当前浓度与上一时刻相关 |
| 粉尘源 | machine_on × 150 | 采煤机截割产生粉尘 |
| 扩散 | PM10(t-1, s-1) × 0.3 | 沿风流方向的空间传播 |
| 消散 | wind_speed × 20 | 风速越大，粉尘被吹散 |
| 沉降 | humidity × 0.3 | 湿度越高，粉尘沉降越快 |

### 滞后特征的作用

1. **捕捉运移轨迹**: 通过多个时间步的滞后，还原粉尘从产生到检测的传播路径
2. **反映惯性**: 历史数据帮助模型学习粉尘浓度的变化惯性
3. **预测提前量**: 基于当前和历史数据，预测未来时刻的浓度

### 随机森林模型

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 150 | 决策树数量 |
| max_depth | 12 | 树的最大深度 |
| n_jobs | -1 | 使用所有 CPU 核心 |

## 数据预处理

### 标准化

```python
from sklearn.preprocessing import StandardScaler

# 特征标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 目标值标准化
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 保存标准化器
import pickle
with open('models/dust_scaler.pkl', 'wb') as f:
    pickle.dump({
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': feature_names
    }, f)
```

### 重要提醒

- **标准化参数必须保存**: 推理时必须使用训练时的均值和标准差
- **特征顺序必须一致**: 按 `feature_order.txt` 中的顺序排列
- **数据按时间顺序划分**: 训练集使用前80%数据，测试集使用后20%

## 模型文件

| 文件 | 路径 | 说明 |
|------|------|------|
| ONNX 模型 | `models/dust_prediction_model.onnx` | 跨平台部署格式 |
| 训练脚本 | `scripts/train_dust_model.py` | 模型训练代码 |
| 标准化器 | `models/dust_scaler.pkl` | 特征和目标标准化参数 |
| 特征顺序 | `models/dust_feature_order.txt` | C++推理特征顺序 |
| 训练数据 | `data/dust_time_series_samples.csv` | 时间序列样本数据 |

## 预警阈值建议

| PM10 浓度 (μg/m³) | 等级 | 建议措施 |
|-------------------|------|----------|
| < 50 | 优 | 正常作业 |
| 50-100 | 良 | 可正常作业 |
| 100-150 | 轻度污染 | 建议开启辅助降尘 |
| 150-250 | 中度污染 | 必须开启洒水降尘 |
| 250-350 | 重度污染 | 减少作业，加强通风 |
| > 350 | 严重污染 | 停止作业，人员撤离 |

## 扩展建议

1. **多步预测**: 扩展为预测未来多个时间点的粉尘浓度
2. **时空关联**: 引入多个工作面的空间相关性
3. **气象数据**: 接入实时气象数据提高预测精度
4. **设备联动**: 与通风系统控制器联动实现自动调控

## 相关模型

- [皮带机打滑预测](BELT_CONVEYOR_SLIP_PREDICTION.md) - 皮带机运行状态预测
- [泵故障预测](PUMP_FAILURE_PREDICTION.md) - 水泵故障预警
- [压缩机泄漏预测](COMPRESSOR_LEAKAGE_PREDICTION.md) - 压缩机状态监测
- [IGBT温度预测模型](TRAIN_IGBT_MODEL.md) - IGBT温度趋势预测

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-04-23 | 2.0.0 | 引入时间滞后特征，模拟粉尘运移延迟 |
| 2026-04-23 | 1.0.0 | 初始版本，支持煤矿粉尘浓度预测 |
