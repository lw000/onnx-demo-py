# 温度预测模型

基于 Gradient Boosting 回归的工业设备温度预测系统，支持 ONNX 格式跨平台部署。

## 模型概述

温度预测模型是一个回归模型，通过分析 8 个传感器数据（温度、三轴振动、电流、电压、气压、湿度）来预测设备未来的温度变化。模型采用 sklearn 管道构建，包含特征标准化、多项式特征工程和梯度提升回归器。

## 技术栈

- **Python**: 3.14+
- **机器学习库**: scikit-learn
- **模型转换**: skl2onnx, onnx
- **数据处理**: numpy
- **模型保存**: joblib

## 安装依赖

```bash
pip install numpy scikit-learn onnx skl2onnx joblib
```

如需验证 ONNX 模型，还需安装：
```bash
pip install onnxruntime
```

## 文件说明

```
温度预测模型/
├── advanced_temp_model.py                   # 模型训练脚本
├── advanced_temp_model.onnx                 # ONNX 格式模型 (386 KB)
├── advanced_temp_model_sklearn.pkl          # Sklearn 原始模型 (792 KB)
└── TEMP_MODEL_README.md                     # 本文档
```

## 特征说明

| 特征名称 | 单位 | 范围 | 说明 |
|---------|------|------|------|
| temp_current | °C | 20-80 | 当前温度，主要影响因素 |
| vibration_x | mm/s | 0-10 | X轴振动幅度 |
| vibration_y | mm/s | 0-10 | Y轴振动幅度 |
| vibration_z | mm/s | 0-10 | Z轴振动幅度 |
| current | A | 1-15 | 工作电流 |
| voltage | V | 220-240 | 工作电压 |
| pressure | kPa | 90-110 | 环境气压 |
| humidity | % | 30-70 | 环境湿度 |

## 模型架构

### 处理流水线

```python
Pipeline([
    ('scaler', StandardScaler()),                      # 特征标准化
    ('poly', PolynomialFeatures(degree=2)),           # 二次多项式特征
    ('model', GradientBoostingRegressor(              # 梯度提升回归
        n_estimators=200,      # 200棵决策树
        max_depth=5,           # 树的最大深度
        learning_rate=0.1,     # 学习率
        min_samples_split=10,  # 最小分裂样本数
        min_samples_leaf=5,    # 最小叶子节点样本数
        random_state=42        # 随机种子
    ))
])
```

### 模型特点

- **非线性建模**: 使用多项式特征捕获非线性关系
- **特征交互**: 自动学习特征间的交互效应
- **集成学习**: 200 棵决策树的梯度提升
- **标准化处理**: 自动特征缩放
- **正则化**: 控制过拟合

## 使用方法

### 1. 训练模型

```bash
python advanced_temp_model.py
```

训练过程输出：
```
开始训练模型...

模型性能评估:
训练集 R²: 0.9902
测试集 R²: 0.9837
交叉验证 R²: 0.9745 (±0.0053)

正在转换为 ONNX 格式...
ONNX 模型验证通过

模型已保存:
  ONNX 模型: advanced_temp_model.onnx
  Sklearn 模型: advanced_temp_model_sklearn.pkl

模型信息:
  输入特征数: 8
  训练样本数: 4000
  测试样本数: 1000

测试预测:
输入: [45.2 5.3 4.1 3.8 9.2 232.5 98.3 52.1]
预测温度: 48.76°C
实际温度: 48.72°C
```

### 2. 使用 Sklearn 模型预测

```python
import joblib
import numpy as np

# 加载模型
pipeline = joblib.load('advanced_temp_model_sklearn.pkl')

# 准备输入数据 (8 个特征)
input_data = np.array([[
    50.0,    # temp_current (°C)
    3.5,     # vibration_x (mm/s)
    4.2,     # vibration_y (mm/s)
    2.8,     # vibration_z (mm/s)
    8.5,     # current (A)
    235,     # voltage (V)
    100,     # pressure (kPa)
    45       # humidity (%)
]]).astype(np.float32)

# 预测
prediction = pipeline.predict(input_data)
print(f"预测温度: {prediction[0]:.2f}°C")
```

### 3. 使用 ONNX 模型预测

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
session = ort.InferenceSession('advanced_temp_model.onnx')

# 准备输入数据
input_data = np.array([[
    50.0,    # temp_current (°C)
    3.5,     # vibration_x (mm/s)
    4.2,     # vibration_y (mm/s)
    2.8,     # vibration_z (mm/s)
    8.5,     # current (A)
    235,     # voltage (V)
    100,     # pressure (kPa)
    45       # humidity (%)
]]).astype(np.float32)

# 预测
outputs = session.run(None, {'float_input': input_data})
prediction = outputs[0][0]
print(f"预测温度: {prediction:.2f}°C")
```

### 4. 批量预测

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession('advanced_temp_model.onnx')

# 批量预测 (多个样本)
batch_data = np.random.rand(100, 8).astype(np.float32)
outputs = session.run(None, {'float_input': batch_data})
predictions = outputs[0]

print(f"批量预测 {len(predictions)} 个样本")
print(f"平均预测温度: {predictions.mean():.2f}°C")
print(f"温度范围: {predictions.min():.2f}°C - {predictions.max():.2f}°C")
```

### 5. 单样本预测函数封装

```python
import onnxruntime as ort
import numpy as np

class TemperaturePredictor:
    def __init__(self, model_path='advanced_temp_model.onnx'):
        self.session = ort.InferenceSession(model_path)
    
    def predict(self, temp_current, vibration_x, vibration_y, vibration_z, 
                current, voltage, pressure, humidity):
        """
        预测温度
        
        参数:
            temp_current: 当前温度 (°C)
            vibration_x: X轴振动 (mm/s)
            vibration_y: Y轴振动 (mm/s)
            vibration_z: Z轴振动 (mm/s)
            current: 电流 (A)
            voltage: 电压 (V)
            pressure: 气压 (kPa)
            humidity: 湿度 (%)
        
        返回:
            预测温度 (°C)
        """
        input_data = np.array([[
            temp_current, vibration_x, vibration_y, vibration_z,
            current, voltage, pressure, humidity
        ]]).astype(np.float32)
        
        outputs = self.session.run(None, {'float_input': input_data})
        return float(outputs[0][0])

# 使用示例
predictor = TemperaturePredictor()
result = predictor.predict(
    temp_current=50.0,
    vibration_x=3.5,
    vibration_y=4.2,
    vibration_z=2.8,
    current=8.5,
    voltage=235,
    pressure=100,
    humidity=45
)
print(f"预测温度: {result:.2f}°C")
```

## 模型性能

### 评估指标

- **R² 分数**: 模型解释的方差比例（越接近 1 越好）
- **交叉验证**: 5 折交叉验证确保模型泛化能力
- **预测误差**: 平均绝对误差（MAE）和均方根误差（RMSE）

### 性能参考

基于 5000 个样本的模拟数据：

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
| 模型大小 | 386 KB (ONNX) |

## 模型验证

### ONNX 模型验证

```python
import onnx

# 验证 ONNX 模型结构
onnx_model = onnx.load('advanced_temp_model.onnx')
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")

# 查看模型信息
print("\n模型信息:")
print(onnx.helper.printable_graph(onnx_model.graph))
```

### 对比验证 (Sklearn vs ONNX)

```python
import joblib
import onnxruntime as ort
import numpy as np

# 加载两个模型
pipeline = joblib.load('advanced_temp_model_sklearn.pkl')
session = ort.InferenceSession('advanced_temp_model.onnx')

# 测试数据
test_data = np.random.rand(10, 8).astype(np.float32)

# 预测对比
sklearn_pred = pipeline.predict(test_data)
onnx_pred = session.run(None, {'float_input': test_data})[0]

# 计算差异
diff = np.abs(sklearn_pred - onnx_pred).max()
print(f"最大预测差异: {diff:.10f}")

if diff < 1e-6:
    print("✅ Sklearn 和 ONNX 预测结果一致")
else:
    print("⚠️ 存在预测差异")
```

## 部署选项

### 1. C++ 部署

使用 ONNX Runtime C++ API 部署到嵌入式设备或工业控制器。

```cpp
#include <onnxruntime_cxx_api.h>

// 加载模型
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
Ort::Session session{env, L"advanced_temp_model.onnx", Ort::SessionOptions{nullptr}};

// 准备输入
float input_tensor_values[] = {50.0f, 3.5f, 4.2f, 2.8f, 8.5f, 235.0f, 100.0f, 45.0f};

// 运行推理
// ... 推理代码
```

### 2. Web 部署

使用 ONNX Runtime Web 部署到浏览器。

```javascript
import * as ort from 'onnxruntime-web';

// 加载模型
const session = await ort.InferenceSession.create('advanced_temp_model.onnx');

// 准备输入
const input = new ort.Tensor('float32', [50.0, 3.5, 4.2, 2.8, 8.5, 235.0, 100.0, 45.0], [1, 8]);

// 运行推理
const outputs = await session.run({ float_input: input });
const prediction = outputs[outputNames[0]].data[0];

console.log(`预测温度: ${prediction.toFixed(2)}°C`);
```

### 3. 移动端部署

使用 ONNX Runtime Mobile 部署到 iOS/Android 应用。

```swift
import ORTObjectives

// iOS Swift 示例
let modelPath = Bundle.main.path(forResource: "advanced_temp_model", ofType: "onnx")!
let session = try ORTSession(path: modelPath)

// 准备输入和运行推理
// ...
```

### 4. 边缘计算

使用 ONNX Runtime 直接部署到边缘计算设备（如树莓派、Jetson 等）。

### 5. 云端部署

使用 Docker 容器化部署到云平台。

```dockerfile
FROM python:3.14-slim

RUN pip install onnxruntime numpy
COPY advanced_temp_model.onnx /app/

CMD ["python", "serve.py"]
```

## 高级配置

### 调整模型参数

在 `advanced_temp_model.py` 中修改以下参数：

```python
from sklearn.ensemble import GradientBoostingRegressor

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),  # 修改多项式阶数
    ('model', GradientBoostingRegressor(
        n_estimators=200,      # 树的数量（增加可提高精度但降低速度）
        max_depth=5,           # 树的最大深度
        learning_rate=0.1,     # 学习率（降低需要增加 n_estimators）
        min_samples_split=10,  # 最小分裂样本数
        min_samples_leaf=5,    # 最小叶子节点样本数
        random_state=42
    ))
])
```

### 使用不同的回归器

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# 随机森林
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 支持向量回归
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVR(kernel='rbf'))
])

# XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(n_estimators=200, learning_rate=0.1))
])
```

### 添加新特征

```python
# 在 advanced_temp_model.py 中添加新特征
additional_features = np.random.uniform(0, 100, n_samples)  # 新特征

# 扩展特征矩阵
X = np.column_stack([
    temp_current, vibration_x, vibration_y, vibration_z,
    current, voltage, pressure, humidity, additional_features
]).astype(np.float32)
```

## 实际应用场景

1. **设备温度监测与预警**
   - 实时预测设备温度
   - 超过阈值时触发预警

2. **冷却系统智能控制**
   - 根据预测温度自动调节冷却系统
   - 优化能耗

3. **设备维护计划优化**
   - 预测温度趋势，提前安排维护
   - 避免设备过热损坏

4. **能耗分析与管理**
   - 分析温度与能耗的关系
   - 优化运行参数

## 故障排查

### 常见问题

**Q: ONNX 转换失败**
```python
# 指定较低的目标算子集版本
onnx_model = convert_sklearn(
    pipeline, 
    initial_types=initial_type, 
    target_opset=10
)
```

**Q: 预测结果差异较大**
- 确保 Sklearn 和 ONNX 使用相同的输入数据类型（float32）
- 检查输入特征的顺序是否一致
- 验证模型版本是否匹配

**Q: 性能不佳**
- 增加训练样本数量
- 尝试不同的模型或参数
- 检查特征工程是否充分
- 调整多项式特征阶数

**Q: 内存不足**
```python
# 减少训练样本数量
n_samples = 2000  # 从 5000 减少到 2000

# 减少树的数量
GradientBoostingRegressor(n_estimators=100)  # 从 200 减少到 100
```

**Q: 推理速度慢**
```python
# 减少模型复杂度
GradientBoostingRegressor(
    n_estimators=100,  # 减少树的数量
    max_depth=3         # 减少树的深度
)
```

## 扩展建议

1. **实时数据接入**
   - 连接工业传感器实时数据流
   - 实现实时温度预测

2. **历史数据分析**
   - 保存预测结果用于长期分析
   - 分析温度变化趋势

3. **模型定期更新**
   - 定期用新数据重新训练模型
   - 保持模型准确性

4. **可视化界面**
   - 开发 Web/移动端监控界面
   - 实时显示预测结果

5. **告警系统**
   - 基于预测结果触发告警
   - 支持多种告警方式

## 数据生成逻辑

模型使用以下公式生成模拟数据：

```python
y = (
    0.6 * temp_current +                          # 当前温度影响最大
    0.25 * (vibration_x * vibration_y) ** 0.5 +   # 振动交互效应
    0.15 * current ** 0.8 +                        # 电流的非线性影响
    0.08 * (voltage - 230) * 0.5 +                 # 电压偏差影响
    0.05 * np.sin(pressure * 0.1) +               # 气压周期性影响
    0.03 * (humidity - 50) ** 2 / 100 +            # 湿度二次影响
    0.02 * vibration_x * current / 10 +           # 振动与电流的交互
    15                                             # 基础偏移
) + np.random.normal(0, 0.5, size=n_samples)      # 添加噪声
```

**注意**: 以上系数（0.6, 0.25, 0.15 等）是模拟数据生成参数，用于生成训练数据。**模型微调时不需要调整这些系数**。

## 模型参数调优

### 可调参数

本模型使用 `GradientBoostingRegressor`，以下是关键可调参数：

| 参数 | 当前值 | 说明 | 调优建议 |
|------|--------|------|----------|
| n_estimators | 200 | 决策树数量 | 100-500，越多越稳定但越慢 |
| max_depth | 5 | 树的最大深度 | 3-8，控制过拟合 |
| learning_rate | 0.1 | 学习率 | 0.01-0.2，降低需增加 n_estimators |
| min_samples_split | 10 | 节点分裂最小样本数 | 5-20，防止过拟合 |
| min_samples_leaf | 5 | 叶节点最小样本数 | 2-10，防止过拟合 |
| polynomial degree | 2 | 多项式特征阶数 | 2-3，阶数过高易过拟合 |

### 调优函数

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

# 定义参数网格
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__min_samples_split': [5, 10, 20],
    'poly__degree': [2, 3]
}

# 创建管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures()),
    ('model', GradientBoostingRegressor(random_state=42))
])

# 网格搜索
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳 R²: {grid_search.best_score_:.4f}")

# 使用最佳参数
best_pipeline = grid_search.best_estimator_
```

### 调优步骤

1. **基准测试**: 使用当前参数训练，记录 R² 得分
2. **网格搜索**: 使用 `GridSearchCV` 寻找最优参数组合
3. **交叉验证**: 5折交叉验证确保稳定性
4. **性能对比**: 对比调优前后的训练集和测试集 R²
5. **部署验证**: 重新导出 ONNX 模型并验证一致性

### 注意事项

- **数据生成公式系数**（0.6, 0.25, 0.15 等）是模拟数据参数，**不需要调整**
- 只需调整机器学习模型的超参数
- 降低 `learning_rate` 时应同步增加 `n_estimators`
- 多项式阶数过高（>3）会导致特征爆炸和过拟合
- 调整参数后需重新训练并导出 ONNX 模型

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
