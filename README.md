# 工业设备预测模型集合

基于机器学习的工业设备监测系统，包含温度预测、泵故障预测、压缩机泄漏检测、采煤机故障预测和皮带机打滑故障预测五个核心模型，均支持 ONNX 格式跨平台部署。

## 项目概述

本项目提供了六个独立的工业设备预测模型，用于不同场景的设备监测和故障诊断：

1. **温度预测模型** (`advanced_temp_model.py`) - 使用 Gradient Boosting 回归预测设备温度
2. **泵故障预测模型** (`pump_failure_prediction.py`) - 使用随机森林分类预测泵设备故障状态
3. **压缩机泄漏预测模型** (`compressor_leakage_prediction.py`) - 使用随机森林分类检测空压系统管网泄漏
4. **采煤机故障预测模型** (`shearer_cutting_unit_failure_prediction.py`) - 使用随机森林分类预测采煤机截割部故障
5. **皮带机打滑故障预测模型** (`belt_conveyor_slippage_fault_prediction.py`) - 使用随机森林分类预测皮带输送机打滑、卡阻等故障
6. **皮带机打滑预测模型** (`belt_conveyor_slip_prediction.py`) - 使用随机森林分类预测皮带机打滑状态

所有模型都采用 sklearn 管道构建，并转换为 ONNX 格式以便于跨平台部署。

## 技术栈

- **Python**: 3.14+
- **机器学习库**: scikit-learn
- **模型转换**: skl2onnx, onnx, onnxruntime
- **数据处理**: numpy, pandas
- **模型保存**: joblib

## 安装依赖

```bash
pip install numpy scikit-learn onnx skl2onnx joblib pandas matplotlib seaborn
```

如需验证 ONNX 模型，还需安装：
```bash
pip install onnxruntime
```

## 项目结构

```
onnx-demo/
├── advanced_temp_model.py                       # 温度预测模型训练脚本
├── advanced_temp_model.onnx                     # ONNX 格式温度预测模型 (386 KB)
├── advanced_temp_model_sklearn.pkl              # Sklearn 原始温度预测模型 (792 KB)
│
├── pump_failure_prediction.py                   # 泵故障预测模型训练脚本
├── pump_failure_classifier.onnx                  # ONNX 格式泵故障分类器 (307 KB)
├── pump_failure_classifier_sklearn.pkl           # Sklearn 原始泵故障分类器 (606 KB)
│
├── compressor_leakage_prediction.py            # 压缩机泄漏预测模型训练脚本
├── compressor_leakage_detector.onnx             # ONNX 格式泄漏检测器 (836 KB)
├── compressor_leakage_detector_sklearn.pkl      # Sklearn 原始泄漏检测器 (1.77 MB)
│
├── shearer_cutting_unit_failure_prediction.py   # 采煤机故障预测模型训练脚本
├── shearer_cutting_unit_failure_detector.onnx   # ONNX 格式故障检测器 (99 KB)
├── shearer_cutting_unit_failure_detector_sklearn.pkl  # Sklearn 原始故障检测器 (256 KB)
│
├── belt_conveyor_slippage_fault_prediction.py   # 皮带机打滑故障预测模型训练脚本
├── belt_conveyor_slippage_fault_detector.onnx   # ONNX 格式故障检测器
├── belt_conveyor_slippage_fault_detector_sklearn.pkl  # Sklearn 原始故障检测器
│
├── belt_conveyor_slip_prediction.py            # 皮带机打滑预测模型训练脚本
├── conveyor_slip_model.onnx                    # ONNX 格式打滑预测模型
│
├── simple_temp_model.py                        # 简单温度预测示例
├── simple_temp_model.onnx                      # 简单 ONNX 模型
│
├── TEMP_MODEL_README.md                        # 温度预测模型详细文档
├── PUMP_MODEL_README.md                        # 泵故障预测模型详细文档
├── COMPRESSOR_MODEL_README.md                  # 压缩机泄漏预测模型详细文档
├── SHEARER_MODEL_README.md                     # 采煤机故障预测模型详细文档
├── BELT_CONVEYOR_MODEL_README.md              # 皮带机故障预测模型详细文档
├── BELT_CONVEYOR_SLIP_README.md              # 皮带机打滑预测模型详细文档
└── README.md                                  # 本文档
```

## 模型概览

### 1. 温度预测模型

**文件**: `advanced_temp_model.py`

**用途**: 基于多传感器数据预测设备温度

**特征** (8 个):
- temp_current (°C): 当前温度
- vibration_x/y/z (mm/s): 三轴振动幅度
- current (A): 电流
- voltage (V): 电压
- pressure (kPa): 气压
- humidity (%): 湿度

**模型**: Gradient Boosting Regressor
- 200 棵决策树
- 多项式特征工程 (degree=2)
- R² 分数: ~0.98

**运行**:
```bash
python advanced_temp_model.py
```

**详细文档**: [TEMP_MODEL_README.md](TEMP_MODEL_README.md)

---

### 2. 泵故障预测模型

**文件**: `pump_failure_prediction.py`

**用途**: 预测主排水泵的运行状态（正常、磨损、汽蚀）

**特征** (4 个):
- flow (m³/h): 流量
- head (m): 扬程
- power (kW): 功率
- vibration (mm/s): 振动

**模型**: Random Forest Classifier
- 100 棵决策树
- 多分类（正常/磨损/汽蚀）
- 准确率: ~97%

**运行**:
```bash
python pump_failure_prediction.py
```

**详细文档**: [PUMP_MODEL_README.md](PUMP_MODEL_README.md)

---

### 3. 压缩机泄漏预测模型

**文件**: `compressor_leakage_prediction.py`

**用途**: 检测空压系统管网泄漏

**特征** (3 个):
- pressure (MPa): 管网压力
- supply_flow (m³/h): 供气流量
- demand_flow (m³/h): 用气流量

**模型**: Random Forest Classifier
- 100 棵决策树
- 二分类（正常/泄漏）
- 准确率: ~95%+

**运行**:
```bash
python compressor_leakage_prediction.py
```

---

### 4. 采煤机故障预测模型

**文件**: `shearer_cutting_unit_failure_prediction.py`

**用途**: 预测采煤机截割部故障状态

**特征** (5 个):
- vibration (mm/s): 振动值
- temperature (°C): 温度
- current (A): 电流
- pressure (bar): 液压
- oil_quality (%): 油液质量

**模型**: Random Forest Classifier
- 100 棵决策树
- 二分类（正常/故障）
- 准确率: ~99%

**运行**:
```bash
python shearer_cutting_unit_failure_prediction.py
```

**详细文档**: [SHEARER_MODEL_README.md](SHEARER_MODEL_README.md)

---

### 5. 皮带机打滑故障预测模型

**文件**: `belt_conveyor_slippage_fault_prediction.py`

**用途**: 预测皮带输送机的打滑、卡阻、轴承损坏等故障

**特征** (5 个):
- current (A): 电机电流
- speed_diff (m/s): 头尾速度差（打滑的核心指标）
- vibration_motor (mm/s): 电机振动
- temperature_bearing (°C): 轴承温度
- current_std: 电流波动率（滚动标准差）

**模型**: Random Forest Classifier
- 200 棵决策树
- 二分类（正常/故障）
- 准确率: ~98%

**运行**:
```bash
python belt_conveyor_slippage_fault_prediction.py
```

**详细文档**: [BELT_CONVEYOR_MODEL_README.md](BELT_CONVEYOR_MODEL_README.md)

---

### 6. 皮带机打滑预测模型

**文件**: `belt_conveyor_slip_prediction.py`

**用途**: 预测皮带输送机的打滑状态

**特征** (4 个):
- current (A): 电机电流
- speed_diff (m/s): 头尾速度差（打滑的核心指标）
- vibration (mm/s): 电机振动
- temperature (°C): 轴承温度

**模型**: Random Forest Classifier
- 100 棵决策树
- 二分类（正常/打滑）
- 准确率: ~98%

**运行**:
```bash
python belt_conveyor_slip_prediction.py
```

**详细文档**: [BELT_CONVEYOR_SLIP_README.md](BELT_CONVEYOR_SLIP_README.md)

---

## 快速开始

### 训练所有模型

```bash
# 训练温度预测模型
python advanced_temp_model.py

# 训练泵故障预测模型
python pump_failure_prediction.py

# 训练压缩机泄漏预测模型
python compressor_leakage_prediction.py

# 训练采煤机故障预测模型
python shearer_cutting_unit_failure_prediction.py

# 训练皮带机打滑故障预测模型
python belt_conveyor_slippage_fault_prediction.py

# 训练皮带机打滑预测模型
python belt_conveyor_slip_prediction.py
```

### 使用模型预测

#### 温度预测

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('advanced_temp_model.onnx')

input_data = np.array([[
    50.0, 3.5, 4.2, 2.8, 8.5, 235, 100, 45
]]).astype(np.float32)

outputs = session.run(None, {'float_input': input_data})
prediction = outputs[0][0]
print(f"预测温度: {prediction:.2f}°C")
```

#### 泵故障预测

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('pump_failure_classifier.onnx')

input_data = np.array([[110.0, 55.0, 50.0, 1.2]]).astype(np.float32)

outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])
probabilities = outputs[1][0]

label_names = ['cavitation', 'normal', 'wear']
print(f"预测状态: {label_names[label_idx]}")
print(f"概率: {dict(zip(label_names, probabilities))}")
```

#### 泄漏检测

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('compressor_leakage_detector.onnx')

input_data = np.array([[0.75, 500, 490]]).astype(np.float32)

outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])
probabilities = outputs[1][0]

status = "泄漏" if label_idx == 1 else "正常"
print(f"预测状态: {status}")
print(f"概率: 正常={probabilities[0]:.2%}, 泄漏={probabilities[1]:.2%}")
```

#### 采煤机故障预测

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('shearer_cutting_unit_failure_detector.onnx')

input_data = np.array([[12.5, 95.0, 180.0, 220.0, 40.0]]).astype(np.float32)

outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])
probabilities = outputs[1][0]

status = "故障" if label_idx == 1 else "正常"
print(f"预测状态: {status}")
print(f"概率: 正常={probabilities[0]:.2%}, 故障={probabilities[1]:.2%}")
```

#### 皮带机打滑故障预测

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('belt_conveyor_slippage_fault_detector.onnx')

input_data = np.array([[140, 0.8, 10, 90, 25]]).astype(np.float32)
# 依次为: current, speed_diff, vibration_motor, temperature_bearing, current_std

outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])
probabilities = outputs[1][0]

status = "故障" if label_idx == 1 else "正常"
print(f"预测状态: {status}")
print(f"概率: 正常={probabilities[0]:.2%}, 故障={probabilities[1]:.2%}")
```

#### 皮带机打滑预测

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('conveyor_slip_model.onnx')

input_data = np.array([[135, 0.25, 9.0, 88]]).astype(np.float32)
# 依次为: current, speed_diff, vibration, temperature

outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])
probabilities = outputs[1][0]

status = "打滑" if label_idx == 1 else "正常"
print(f"预测状态: {status}")
print(f"概率: 正常={probabilities[0]:.2%}, 打滑={probabilities[1]:.2%}")
```

## 模型对比

| 特性 | 温度预测 | 泵故障预测 | 泄漏检测 | 采煤机故障 | 皮带机故障 | 皮带机打滑 |
|------|---------|-----------|---------|-----------|-----------|-----------|
| 模型类型 | 回归 | 分类 | 分类 | 分类 | 分类 | 分类 |
| 任务 | 温度预测 | 故障诊断 | 泄漏检测 | 故障诊断 | 故障诊断 | 打滑预测 |
| 算法 | Gradient Boosting | Random Forest | Random Forest | Random Forest | Random Forest | Random Forest |
| 特征数量 | 8 | 4 | 3 | 5 | 5 | 4 |
| 样本数量 | 5000 | 10000 | 10000 | 10000 | 5000 | 8000 |
| 输出 | 连续数值 | 类别标签 | 类别标签 | 类别标签 | 类别标签 | 类别标签 |
| R²/准确率 | ~0.98 | ~97% | ~95% | ~99% | ~98% | ~98% |
| ONNX 大小 | 386 KB | 307 KB | 836 KB | 99 KB | ~500 KB | ~500 KB |

## 部署选项

所有模型都支持以下部署方式：

### 1. C++ 部署

使用 ONNX Runtime C++ API 部署到嵌入式设备或工业控制器。

### 2. Web 部署

使用 ONNX Runtime Web 部署到浏览器，实现实时在线监测。

### 3. 移动端部署

使用 ONNX Runtime Mobile 部署到 iOS/Android 应用。

### 4. 边缘计算

使用 ONNX Runtime 直接部署到边缘计算设备（如树莓派、Jetson 等）。

### 5. 云端部署

使用 Docker 容器化部署到云平台，支持大规模并发预测。

## 通用功能

### ONNX 模型验证

```python
import onnx

# 验证模型结构
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")

# 查看模型信息
print(onnx.helper.printable_graph(onnx_model.graph))
```

### 对比验证 (Sklearn vs ONNX)

```python
import joblib
import onnxruntime as ort
import numpy as np

# 加载两个模型
pipeline = joblib.load('model_sklearn.pkl')
session = ort.InferenceSession('model.onnx')

# 测试数据
test_data = np.random.rand(10, n_features).astype(np.float32)

# 预测对比
sklearn_pred = pipeline.predict(test_data)
onnx_pred = session.run(None, {'float_input': test_data})[0]

# 计算差异
diff = np.abs(sklearn_pred - onnx_pred).max()
print(f"最大预测差异: {diff:.10f}")
```

### 批量预测

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')

# 批量预测
batch_data = np.random.rand(100, n_features).astype(np.float32)
outputs = session.run(None, {'float_input': batch_data})
predictions = outputs[0]

print(f"批量预测 {len(predictions)} 个样本")
```

## 模型性能参考

### 温度预测模型

| 指标 | 值 |
|------|-----|
| 训练集 R² | ~0.99 |
| 测试集 R² | ~0.98 |
| 交叉验证 R² | ~0.97 ± 0.01 |
| MAE | ~0.5°C |
| RMSE | ~0.7°C |
| 特征数量 | 8 |
| 训练样本 | 4000 |
| 测试样本 | 1000 |

### 泵故障预测模型

| 指标 | 值 |
|------|-----|
| 准确率 | ~97% |
| 精确率 | ~97% |
| 召回率 | ~97% |
| F1 分数 | ~97% |
| 特征数量 | 4 |
| 训练样本 | 8000 |
| 测试样本 | 2000 |

### 压缩机泄漏预测模型

| 指标 | 值 |
|------|-----|
| 准确率 | ~95%+ |
| 精确率 | ~95%+ |
| 召回率 | ~95%+ |
| F1 分数 | ~95%+ |
| 特征数量 | 3 |
| 训练样本 | 8000 |
| 测试样本 | 2000 |

### 采煤机故障预测模型

| 指标 | 值 |
|------|-----|
| 准确率 | ~99% |
| 精确率 | ~99% |
| 召回率 | ~99% |
| F1 分数 | ~99% |
| 特征数量 | 5 |
| 训练样本 | 8000 |
| 测试样本 | 2000 |

### 皮带机打滑故障预测模型

| 指标 | 值 |
|------|-----|
| 准确率 | ~98% |
| 精确率 | ~98% |
| 召回率 | ~98% |
| F1 分数 | ~98% |
| 特征数量 | 5 |
| 训练样本 | 4000 |
| 测试样本 | 1000 |

## 实际应用场景

### 温度预测模型

- 设备温度监测与预警
- 冷却系统智能控制
- 设备维护计划优化
- 能耗分析与管理

### 泵故障预测模型

- 水泵运行状态监测
- 故障预警与诊断
- 维护决策支持
- 设备寿命评估

### 压缩机泄漏预测模型

- 空压系统管网泄漏检测
- 能耗异常监测
- 供气效率优化
- 安全风险预警

### 采煤机故障预测模型

- 采煤机截割部故障诊断
- 设备健康监测
- 预测性维护
- 生产安全保障

### 皮带机打滑故障预测模型

- 矿山皮带输送机监测
- 港口散装物料输送系统
- 电厂输煤皮带机
- 建材行业物料输送
- 预测性维护与安全预警

### 皮带机打滑预测模型

- 煤矿带式运输机打滑监测
- 港口物流输送系统
- 电厂输煤皮带机
- 建材行业输送系统
- 实时打滑预警

## 故障排查

### 常见问题

**Q: ONNX 转换失败**
```python
# 尝试降低目标算子集版本
onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=10)
```

**Q: 预测结果差异较大**
- 确保 Sklearn 和 ONNX 使用相同的输入数据类型（float32）
- 检查输入特征的顺序是否一致
- 验证模型版本是否匹配

**Q: 性能不佳**
- 增加训练样本数量
- 尝试不同的模型或参数
- 检查特征工程是否充分

**Q: onnxruntime 未安装**
```bash
pip install onnxruntime  # CPU 版本
# 或
pip install onnxruntime-gpu  # GPU 版本
```

**Q: 内存不足**
- 减少训练样本数量
- 减少模型复杂度（树的数量、深度等）

## 扩展建议

1. **实时数据接入**: 连接工业传感器实时数据流
2. **可视化界面**: 开发 Web/移动端监控界面
3. **告警系统**: 基于预测结果触发告警
4. **历史数据分析**: 保存预测结果用于长期分析
5. **模型更新**: 定期用新数据重新训练模型
6. **多设备支持**: 扩展支持多种设备类型的预测

## 相关文档

- [温度预测模型详细文档](TEMP_MODEL_README.md) - 包含完整的 API 文档、示例代码和配置说明
- [泵故障预测模型详细文档](PUMP_MODEL_README.md) - 包含完整的 API 文档、示例代码和配置说明
- [压缩机泄漏预测模型详细文档](COMPRESSOR_MODEL_README.md) - 包含完整的 API 文档、示例代码和配置说明
- [采煤机故障预测模型详细文档](SHEARER_MODEL_README.md) - 包含完整的 API 文档、示例代码和配置说明
- [皮带机故障预测模型详细文档](BELT_CONVEYOR_MODEL_README.md) - 包含完整的 API 文档、示例代码和配置说明
- [皮带机打滑预测模型详细文档](BELT_CONVEYOR_SLIP_README.md) - 包含完整的 API 文档、示例代码和配置说明

## 开发指南

### 添加新模型

1. 创建新的训练脚本 `your_model.py`
2. 定义特征和数据生成逻辑
3. 选择合适的 sklearn 模型
4. 转换为 ONNX 格式
5. 保存模型文件
6. 编写详细文档

### 代码风格

- 使用类型提示
- 添加详细的注释和文档字符串
- 遵循 PEP 8 代码风格
- 包含错误处理和日志记录

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至 [373102227@qq.com]
