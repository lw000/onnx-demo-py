# IGBT 温度预测模型

## 模型概述

本模型用于预测变频器 IGBT (绝缘栅双极型晶体管) 模块的未来温度，帮助实现预测性维护和过热预警。

### 应用场景

- 变频器 IGBT 温度监测
- 过热风险预警
- 设备维护计划优化
- 负载优化调度

## 模型信息

| 属性 | 值 |
|------|-----|
| 模型类型 | 回归模型 |
| 算法 | Random Forest Regressor |
| 特征数量 | 5 |
| 输出 | IGBT 温度 (°C) |
| 训练样本 | 5000 |
| ONNX 大小 | ~500 KB |

## 特征说明

| 特征名 | 说明 | 数据类型 | 取值范围 |
|--------|------|----------|----------|
| current | 输出电流 | float | 10-90 A |
| frequency | 输出频率 | float | 10-50 Hz |
| ambient_temp | 环境温度 | float | 20-45 °C |
| temp_rate | 温升速率 | float | -0.5 ~ 2.0 °C/s |
| load_factor | 负载率 | float | 0.2-1.0 |

## 模型性能

| 指标 | 值 |
|------|-----|
| MSE (均方误差) | ~4.0 |
| R² (决定系数) | ~0.95 |
| MAE (平均绝对误差) | ~1.5°C |

## 使用方法

### 训练模型

```bash
python scripts/train_igbt_model.py
```

### Python 推理

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession('models/inverter_temp.onnx')

# 准备输入数据
input_data = np.array([[
    50.0,   # current: 输出电流 (A)
    30.0,   # frequency: 输出频率 (Hz)
    35.0,   # ambient_temp: 环境温度 (°C)
    0.5,    # temp_rate: 温升速率 (°C/s)
    0.7     # load_factor: 负载率
]]).astype(np.float32)

# 推理
outputs = session.run(None, {'float_input': input_data})
predicted_temp = outputs[0][0][0]

print(f"预测 IGBT 温度: {predicted_temp:.2f}°C")

# 温度预警判断
if predicted_temp > 100:
    print("⚠️ 警告: IGBT 温度过高，请检查散热系统")
elif predicted_temp > 85:
    print("⚡ 注意: IGBT 温度偏高，建议降低负载")
else:
    print("✅ IGBT 温度正常")
```

### C++ 推理

```cpp
#include <onnxruntime_cxx_api.h>

// 创建会话
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "igbt_prediction");
Ort::SessionOptions session_options;
Ort::Session session(env, L"models/inverter_temp.onnx", session_options);

// 准备输入
std::vector<float> input_values = {50.0f, 30.0f, 35.0f, 0.5f, 0.7f};
std::vector<int64_t> input_shape = {1, 5};

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

float predicted_temp = output_tensors[0].GetTensorMutableData<float>()[0];
std::cout << "预测 IGBT 温度: " << predicted_temp << "°C" << std::endl;
```

## 模型原理

### 温度预测公式

模型基于物理规律构建，考虑以下因素：

```
预测温度 = 基础温度 + 电流影响 + 负载影响 + 环境温度影响 + 温升速率影响 + 噪声

其中:
- 基础温度: 40°C (IGBT 静态温度)
- 电流影响: 0.4 × current (电流越大，温度越高)
- 负载影响: 10 × load_factor (负载率影响散热效率)
- 环境温度影响: 0.5 × ambient_temp (环境温度影响散热)
- 温升速率影响: 5 × temp_rate (温度变化趋势)
```

### 随机森林模型

使用随机森林回归器进行预测：
- 树的数量: 100
- 最大深度: 10
- 适合处理非线性关系和异常值

## 数据预处理

模型训练前会对数据进行标准化处理：

```python
from sklearn.preprocessing import StandardScaler

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 模型文件

| 文件 | 路径 | 说明 |
|------|------|------|
| ONNX 模型 | `models/inverter_temp.onnx` | 跨平台部署格式 |
| 训练脚本 | `scripts/train_igbt_model.py` | 模型训练代码 |
| 训练数据 | `data/igbt_temp_samples.csv` | IGBT 温度样本数据 |

## 扩展建议

1. **实时数据接入**: 连接变频器传感器实时数据流
2. **多步预测**: 扩展为预测未来多个时间点的温度
3. **异常检测**: 结合温度预测误差进行异常检测
4. **自适应学习**: 定期用新数据更新模型参数

## 注意事项

1. 输入数据必须为 `float32` 类型
2. 特征顺序必须与训练时一致
3. 模型预测的是未来 5 秒后的温度
4. 温度超过 100°C 时需要立即处理

## 相关模型

- [变频器健康预测模型](TRAIN_INVERTER_PREDICTION.md) - 电容寿命和温升异常预测
- [高级温度预测模型](ADVANCED_TEMP_MODEL.md) - 设备温度预测

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-03-11 | 1.1.0 | 训练数据迁移至 data 目录 |
| 2026-03-09 | 1.0.0 | 初始版本，支持 IGBT 温度预测 |
