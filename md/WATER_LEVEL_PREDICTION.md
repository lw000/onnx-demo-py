# 水位测点趋势预测模型

## 模型概述

本模型用于预测水文监测点的水位高度趋势，通过滑动窗口特征工程捕捉时间序列中的历史模式。

### 应用场景

- 河流水位监测与预警
- 水库蓄水管理
- 洪水预警系统
- 城市排水监测

## 模型信息

| 属性 | 值 |
|------|-----|
| 模型类型 | 时间序列回归模型 |
| 算法 | Random Forest Regressor |
| 特征数量 | 10 (滑动窗口) |
| 滑动窗口 | 10个时间步 |
| 输出 | 水位高度预测值 |
| 训练样本 | 792 |
| ONNX 大小 | ~500 KB |

## 特征说明

### 滑动窗口特征

| 特征名 | 说明 | 取值范围 |
|--------|------|----------|
| lag_1 | 最近1小时水位 | 基准值±10 |
| lag_2 | 最近2小时水位 | 基准值±10 |
| lag_3 | 最近3小时水位 | 基准值±10 |
| lag_4 | 最近4小时水位 | 基准值±10 |
| lag_5 | 最近5小时水位 | 基准值±10 |
| lag_6 | 最近6小时水位 | 基准值±10 |
| lag_7 | 最近7小时水位 | 基准值±10 |
| lag_8 | 最近8小时水位 | 基准值±10 |
| lag_9 | 最近9小时水位 | 基准值±10 |
| lag_10 | 最近10小时水位 | 基准值±10 |

### 预测目标

| 目标 | 说明 | 单位 |
|------|------|------|
| target | 下一时刻水位预测值 | m |

## 使用方法

### 训练模型

```bash
python scripts/warter_model.py
```

### Python 推理

```python
import numpy as np
import onnxruntime as ort

# 加载模型
session = ort.InferenceSession('models/water_level_prediction_model.onnx')
input_name = session.get_inputs()[0].name

# 准备输入数据 (最近10个水位值)
input_data = np.array([[
    100.5, 101.2, 100.8, 101.5, 102.0,
    101.8, 102.3, 102.5, 102.2, 102.8
]]).astype(np.float32)

# 推理
output = session.run(None, {input_name: input_data})
predicted_level = output[0][0, 0]

print(f"预测水位: {predicted_level:.2f} m")

# 水位预警判断
if predicted_level > 105:
    print("警告: 水位过高，请注意防洪")
elif predicted_level < 95:
    print("注意: 水位偏低，请关注蓄水情况")
else:
    print("水位正常")
```

### C++ 推理

```cpp
#include <onnxruntime_cxx_api.h>

int main() {
    // 创建会话
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "water_level");
    Ort::SessionOptions session_options;
    Ort::Session session(env, L"models/water_level_prediction_model.onnx", session_options);

    // 输入数据 (10个滑动窗口特征)
    std::vector<float> input_values = {
        100.5f, 101.2f, 100.8f, 101.5f, 102.0f,
        101.8f, 102.3f, 102.5f, 102.2f, 102.8f
    };
    std::vector<int64_t> input_shape = {1, 10};

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

    float predicted_level = output_tensors[0].GetTensorMutableData<float>()[0];
    std::cout << "预测水位: " << predicted_level << " m" << std::endl;

    return 0;
}
```

## 模型原理

### 数据生成模型

水位数据由以下成分构成：

```
水位 = 基准值 + 趋势项 + 周期性项 + 随机噪声

其中:
- 基准值: 100.0 m
- 趋势项: 线性增长 0 ~ 5 m (随时间上涨)
- 周期性项: sin(2πt/50) × 3 m (50小时周期波动)
- 随机噪声: N(0, 1) m
```

### 滑动窗口特征工程

```
时间轴:
lag_10  lag_9  lag_8  ...  lag_2  lag_1   target
  │       │      │         │      │        │
  ▼       ▼      ▼         ▼      ▼        ▼
[t-10]  [t-9]  [t-8] ... [t-2] [t-1]   [t+1]
                    ↑                       ↑
              滑动窗口(10步)          预测目标
```

### 随机森林模型

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 100 | 决策树数量 |
| random_state | 42 | 随机种子 |

## 模型文件

| 文件 | 路径 | 说明 |
|------|------|------|
| ONNX 模型 | `models/water_level_prediction_model.onnx` | 跨平台部署格式 |
| 训练脚本 | `scripts/warter_model.py` | 模型训练代码 |
| 训练数据 | `data/water_level_train.csv` | 792条训练样本 |
| 测试数据 | `data/water_level_test.csv` | 198条测试样本 |

## 预警阈值建议

| 水位 (m) | 等级 | 建议措施 |
|----------|------|----------|
| < 95 | 偏低 | 关注蓄水情况 |
| 95-100 | 正常偏低 | 正常监测 |
| 100-105 | 正常 | 正常作业 |
| 105-110 | 偏高 | 加强监测，准备防洪 |
| > 110 | 危险 | 启动防洪预案 |

## 扩展建议

1. **多源融合**: 引入降雨量、蒸发量等辅助特征
2. **多步预测**: 扩展为预测未来多个时间点的水位
3. **异常检测**: 结合预测误差进行异常检测
4. **季节性建模**: 引入更复杂的周期性特征

## 相关模型

- [煤矿粉尘预测模型](TRAIN_DUST_MODEL.md) - 粉尘浓度趋势预测
- [IGBT温度预测模型](TRAIN_IGBT_MODEL.md) - 设备温度预测
- [皮带机打滑预测](BELT_CONVEYOR_SLIP_PREDICTION.md) - 皮带机状态预测

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-04-23 | 1.0.0 | 初始版本，支持水位预测 |
