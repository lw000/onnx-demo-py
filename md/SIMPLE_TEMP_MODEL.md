# 简单温度预测模型

## 模型概述

这是一个入门级的温度预测模型示例，使用线性回归算法演示 sklearn 模型转换为 ONNX 格式的基本流程。

### 应用场景

- 学习 sklearn 到 ONNX 转换
- 简单温度预测任务
- 模型部署入门示例

## 模型信息

| 属性 | 值 |
|------|-----|
| 模型类型 | 回归模型 |
| 算法 | Linear Regression |
| 特征数量 | 3 |
| 输出 | 预测温度 (°C) |
| ONNX 大小 | ~280 B |

## 特征说明

| 特征名 | 说明 | 数据类型 |
|--------|------|----------|
| temp | 当前温度 | float |
| vibration_x | X 轴振动 | float |
| current | 电流 | float |

## 模型原理

使用线性回归模型：

```
预测温度 = 0.5 * temp + 0.3 * vibration_x + 0.2 * current + 10
```

## 使用方法

### 训练模型

```bash
python scripts/simple_temp_model.py
```

### Python 推理

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession('models/simple_temp_model.onnx')

# 准备输入数据 [temp, vibration_x, current]
input_data = np.array([[0.5, 0.3, 0.2]]).astype(np.float32)

# 推理
outputs = session.run(None, {'float_input': input_data})
predicted_temp = outputs[0][0][0]

print(f"预测温度: {predicted_temp:.2f}°C")
```

## 模型文件

| 文件 | 路径 | 说明 |
|------|------|------|
| ONNX 模型 | `models/simple_temp_model.onnx` | 跨平台部署格式 |
| 训练脚本 | `scripts/simple_temp_model.py` | 模型训练代码 |

## 注意事项

1. 这是一个教学示例，实际应用请使用 `advanced_temp_model.py`
2. 输入数据必须为 `float32` 类型
3. 模型非常简单，仅用于演示 ONNX 转换流程

## 相关模型

- [高级温度预测模型](ADVANCED_TEMP_MODEL.md) - 生产级温度预测
