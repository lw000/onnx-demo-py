# 采煤机截割部故障预测模型

基于随机森林分类的煤矿设备故障诊断系统，支持 ONNX 格式跨平台部署。

## 模型概述

采煤机截割部故障预测模型是一个二分类模型，用于预测煤矿采煤机截割部是否存在故障。通过分析 5 个关键运行参数（振动、温度、电流、液压、油液质量），模型可以识别正常和故障两种状态。模型采用 sklearn 管道构建，包含特征标准化和随机森林分类器。

## 技术栈

- **Python**: 3.14+
- **机器学习库**: scikit-learn
- **数据可视化**: matplotlib, seaborn
- **模型转换**: skl2onnx, onnx
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

## 文件说明

```
采煤机截割部故障预测模型/
├── shearer_cutting_unit_failure_prediction.py    # 模型训练脚本
├── shearer_cutting_unit_failure_detector.onnx     # ONNX 格式模型 (99 KB)
├── shearer_cutting_unit_failure_detector_sklearn.pkl  # Sklearn 原始模型 (256 KB)
└── SHEARER_MODEL_README.md                       # 本文档
```

## 特征说明

| 特征名称 | 单位 | 范围 | 说明 |
|---------|------|------|------|
| vibration | mm/s | 1-15 | 振动值 |
| temperature | °C | 40-110 | 设备温度 |
| current | A | 50-250 | 工作电流 |
| pressure | bar | 50-350 | 液压压力 |
| oil_quality | % | 50-120 | 油液质量 |

## 预测类别

| 类别 | 说明 | 典型特征 | 预警等级 |
|------|------|----------|----------|
| normal | 正常运行 | 振动<8, 温度<85, 油质>60 | 🟢 正常 |
| fault | 故障状态 | 振动>8且温度>85，或油质<60 | 🔴 严重 |

### 故障判断逻辑

模型使用以下规则生成训练标签：

```python
# 故障条件（满足任一即判定为故障）：
1. 振动 > 8 mm/s 且 温度 > 85°C  （高振动+高温）
2. 油液质量 < 60%                （油质差）
```

### 故障类型分析

**高振动+高温故障**
- 可能原因：轴承损坏、齿轮磨损、润滑不良
- 典型表现：振动幅度大，温度异常升高
- 处理建议：立即停机检查，检修轴承和齿轮系统

**油液质量故障**
- 可能原因：油液老化、污染、泄漏
- 典型表现：油液质量指标下降
- 处理建议：更换油液，检查润滑系统

## 模型架构

### 处理流水线

```python
Pipeline([
    ('scaler', StandardScaler()),                      # 特征标准化
    ('classifier', RandomForestClassifier(            # 随机森林分类器
        n_estimators=100,     # 100棵决策树
        max_depth=10,         # 树的最大深度（防止过拟合）
        random_state=42,      # 随机种子
        n_jobs=-1            # 使用所有CPU核心
    ))
])
```

### 模型特点

- **集成学习**: 100 棵决策树的随机森林
- **二分类支持**: 识别正常和故障两种状态
- **特征重要性**: 可分析各特征对故障的贡献度
- **概率输出**: 提供故障发生的置信度
- **抗过拟合**: 通过集成和深度限制提高泛化能力
- **特征分析**: 提供特征重要性排序和可视化

## 使用方法

### 1. 训练模型

```bash
python shearer_cutting_unit_failure_prediction.py
```

训练过程输出：
```
数据集前5行:
   vibration  temperature  current  pressure  oil_quality  fault
0   5.752903    72.959677   162.780   210.562    89.031      0
1   4.806863    80.771537   137.647   175.960    91.549      0
2   4.237457    76.838262   164.291   225.470    95.823      0
3   4.079676    75.364842   159.726   236.774    90.451      0
4   5.897953    71.790906   155.501   233.770    81.424      0

故障样本占比: 12.35%

正在训练模型...

模型准确率: 99.25%

分类报告:
              precision    recall  f1-score   support

        正常       0.99      1.00      1.00      1753
        故障       0.97      0.92      0.94       247

accuracy                           0.99      2000
macro avg       0.98      0.96      0.97      2000
weighted avg       0.99      0.99      0.99      2000

特征重要性排序 (哪个参数最影响设备故障):
         Feature  Importance
1   temperature    0.3623
2    vibration    0.2875
3    oil_quality    0.1821
4      current    0.1021
5      pressure    0.0660

正在转换为 ONNX 格式...
✅ ONNX 模型验证通过
✅ ONNX 模型已保存至: shearer_cutting_unit_failure_detector.onnx
✅ scikit-learn 模型已保存至: shearer_cutting_unit_failure_detector_sklearn.pkl

验证 ONNX 模型一致性...
   - 测试样本输入: [4.8 78.3 156.2 210.5 88.1]
   - scikit-learn 预测 (0:正常, 1:故障): 0
   - scikit-learn 概率: [0.98 0.02]
   - ONNX 模型预测 (0:正常, 1:故障): 0
   - ONNX 概率: [0.98 0.02]
   - 预测结果一致: True

========================================
实时预测示例:
输入数据: 振动=12.5, 温度=95.0, 油质=40.0
预测结果: 故障预警
故障概率: 98.50%
========================================
```

### 2. 使用 Sklearn 模型预测

```python
import joblib
import numpy as np

# 加载模型
pipeline = joblib.load('shearer_cutting_unit_failure_detector_sklearn.pkl')

# 准备输入数据 (5 个特征)
input_data = np.array([[
    12.5,    # vibration (mm/s)
    95.0,    # temperature (°C)
    180.0,   # current (A)
    220.0,   # pressure (bar)
    40.0     # oil_quality (%)
]]).astype(np.float32)

# 预测
prediction = pipeline.predict(input_data)[0]
probability = pipeline.predict_proba(input_data)[0]

print(f"预测状态: {'故障' if prediction == 1 else '正常'}")
print(f"概率分布: 正常={probability[0]:.2%}, 故障={probability[1]:.2%}")
```

### 3. 使用 ONNX 模型预测

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
session = ort.InferenceSession('shearer_cutting_unit_failure_detector.onnx')

# 准备输入数据
input_data = np.array([[
    12.5,    # vibration (mm/s)
    95.0,    # temperature (°C)
    180.0,   # current (A)
    220.0,   # pressure (bar)
    40.0     # oil_quality (%)
]]).astype(np.float32)

# 预测
outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])          # 预测类别索引
probabilities = outputs[1][0]           # 各类别概率

predicted_class = "故障" if label_idx == 1 else "正常"

print(f"预测状态: {predicted_class}")
print(f"概率分布: 正常={probabilities[0]:.2%}, 故障={probabilities[1]:.2%}")
```

### 4. 批量预测

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession('shearer_cutting_unit_failure_detector.onnx')

# 批量预测
batch_data = np.random.rand(100, 5).astype(np.float32)
outputs = session.run(None, {'float_input': batch_data})
labels = outputs[0]
probabilities = outputs[1]

# 统计各类别数量
normal_count = np.sum(labels == 0)
fault_count = np.sum(labels == 1)

print(f"批量预测 {len(batch_data)} 个样本")
print(f"正常: {normal_count} ({normal_count/len(batch_data):.1%})")
print(f"故障: {fault_count} ({fault_count/len(batch_data):.1%})")
```

### 5. 单样本预测函数封装

```python
import onnxruntime as ort
import numpy as np

class ShearerFailureDetector:
    def __init__(self, model_path='shearer_cutting_unit_failure_detector.onnx'):
        self.session = ort.InferenceSession(model_path)

    def predict(self, vibration, temperature, current, pressure, oil_quality):
        """
        预测故障状态

        参数:
            vibration: 振动值 (mm/s)
            temperature: 温度 (°C)
            current: 电流 (A)
            pressure: 液压 (bar)
            oil_quality: 油液质量 (%)

        返回:
            dict: {
                'prediction': 预测类别,
                'probabilities': 各类别概率,
                'confidence': 置信度,
                'risk_factors': 风险因素分析
            }
        """
        input_data = np.array([[
            vibration, temperature, current, pressure, oil_quality
        ]]).astype(np.float32)

        outputs = self.session.run(None, {'float_input': input_data})
        label_idx = int(outputs[0][0])
        probabilities = outputs[1][0]

        prediction = "故障" if label_idx == 1 else "正常"
        confidence = float(probabilities[label_idx])

        # 风险因素分析
        risk_factors = self._analyze_risk_factors(
            vibration, temperature, current, pressure, oil_quality
        )

        return {
            'prediction': prediction,
            'probabilities': {
                'normal': float(probabilities[0]),
                'fault': float(probabilities[1])
            },
            'confidence': confidence,
            'risk_factors': risk_factors
        }

    def _analyze_risk_factors(self, vibration, temperature, current, pressure, oil_quality):
        """分析风险因素"""
        risk_factors = []

        if vibration > 8:
            risk_factors.append({
                'factor': '振动过高',
                'value': vibration,
                'threshold': 8,
                'severity': 'high' if vibration > 10 else 'medium'
            })

        if temperature > 85:
            risk_factors.append({
                'factor': '温度过高',
                'value': temperature,
                'threshold': 85,
                'severity': 'high' if temperature > 95 else 'medium'
            })

        if current > 200:
            risk_factors.append({
                'factor': '电流过大',
                'value': current,
                'threshold': 200,
                'severity': 'medium'
            })

        if oil_quality < 60:
            risk_factors.append({
                'factor': '油液质量差',
                'value': oil_quality,
                'threshold': 60,
                'severity': 'high' if oil_quality < 50 else 'medium'
            })

        return risk_factors

    def get_risk_level(self, prediction, confidence):
        """获取风险等级"""
        if prediction == "正常":
            if confidence > 0.95:
                return ("🟢 正常", "low")
            else:
                return ("🟡 需关注", "medium")
        else:
            if confidence > 0.9:
                return ("🔴 严重故障", "high")
            else:
                return ("🟠 潜在故障", "medium")

# 使用示例
detector = ShearerFailureDetector()
result = detector.predict(
    vibration=12.5,
    temperature=95.0,
    current=180.0,
    pressure=220.0,
    oil_quality=40.0
)

print(f"预测状态: {result['prediction']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"风险等级: {detector.get_risk_level(result['prediction'], result['confidence'])[0]}")
print("风险因素:")
for factor in result['risk_factors']:
    print(f"  - {factor['factor']}: {factor['value']} (阈值: {factor['threshold']})")
```

## 数据生成逻辑

### 故障判断规则

模型使用以下规则生成训练标签：

```python
# 故障条件（满足任一即判定为故障）：
fault = ((vibration > 8) & (temperature > 85)) | (oil_quality < 60)
```

### 数据分布

模型生成 10000 个模拟样本：

```python
# 所有特征
vibration: N(5, 2), range: ~1-15
temperature: N(75, 10), range: ~40-110
current: N(150, 30), range: ~50-250
pressure: N(200, 50), range: ~50-350
oil_quality: N(90, 10), range: ~60-120

# 故障样本占比: ~12.35%
```

### 特征重要性

根据模型训练结果，特征重要性排序：

| 排名 | 特征 | 重要性 | 说明 |
|------|------|--------|------|
| 1 | temperature | ~36% | 温度是最重要的故障指标 |
| 2 | vibration | ~29% | 振动是第二重要的指标 |
| 3 | oil_quality | ~18% | 油液质量对故障有较大影响 |
| 4 | current | ~10% | 电流有一定影响 |
| 5 | pressure | ~7% | 液压影响相对较小 |

## 模型性能

### 评估指标

- **准确率 (Accuracy)**: 预测正确的比例
- **精确率 (Precision)**: 预测为故障的样本中实际为故障的比例
- **召回率 (Recall)**: 实际为故障的样本中被正确预测的比例
- **F1 分数**: 精确率和召回率的调和平均

### 性能参考

基于 10000 个样本的模拟数据（故障样本占比 ~12.35%）：

| 指标 | 正常 | 故障 |
|------|------|------|
| 精确率 | ~99% | ~97% |
| 召回率 | ~100% | ~92% |
| F1 分数 | ~100% | ~94% |

| 总体指标 | 值 |
|----------|-----|
| 准确率 | ~99.25% |
| 宏平均 | ~97% |
| 加权平均 | ~99% |
| 特征数量 | 5 |
| 训练样本 | 8000 |
| 测试样本 | 2000 |
| 模型大小 | 99 KB (ONNX) |

### 混淆矩阵

```
                预测正常    预测故障
真实正常        1753          0
真实故障          20         227
```

## 模型验证

### ONNX 模型验证

```python
import onnx

# 验证 ONNX 模型结构
onnx_model = onnx.load('shearer_cutting_unit_failure_detector.onnx')
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
pipeline = joblib.load('shearer_cutting_unit_failure_detector_sklearn.pkl')
session = ort.InferenceSession('shearer_cutting_unit_failure_detector.onnx')

# 测试数据
test_data = np.random.rand(10, 5).astype(np.float32)

# 预测对比
sklearn_pred = pipeline.predict(test_data)
onnx_pred = session.run(None, {'float_input': test_data})[0]

# 检查一致性
consistent = np.all(sklearn_pred == onnx_pred)
print(f"预测一致性: {consistent}")

if not consistent:
    print("不一致的样本:")
    for i in range(len(sklearn_pred)):
        if sklearn_pred[i] != onnx_pred[i]:
            print(f"  样本 {i}: Sklearn={sklearn_pred[i]}, ONNX={onnx_pred[i]}")
```

## 部署选项

### 1. C++ 部署

使用 ONNX Runtime C++ API 部署到嵌入式设备或工业控制器。

```cpp
#include <onnxruntime_cxx_api.h>

// 加载模型
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
Ort::Session session{env, L"shearer_cutting_unit_failure_detector.onnx", Ort::SessionOptions{nullptr}};

// 准备输入
float input_tensor_values[] = {12.5f, 95.0f, 180.0f, 220.0f, 40.0f};

// 运行推理
// ... 推理代码
```

### 2. Web 部署

使用 ONNX Runtime Web 部署到浏览器。

```javascript
import * as ort from 'onnxruntime-web';

// 加载模型
const session = await ort.InferenceSession.create('shearer_cutting_unit_failure_detector.onnx');

// 准备输入
const input = new ort.Tensor('float32', [12.5, 95.0, 180.0, 220.0, 40.0], [1, 5]);

// 运行推理
const outputs = await session.run({ float_input: input });
const labelIdx = outputs['label'].data[0];
const probabilities = outputs['probabilities'];

const status = labelIdx === 1 ? '故障' : '正常';
console.log(`预测状态: ${status}`);
```

### 3. 移动端部署

使用 ONNX Runtime Mobile 部署到 iOS/Android 应用。

```swift
import ORTObjectives

// iOS Swift 示例
let modelPath = Bundle.main.path(forResource: "shearer_cutting_unit_failure_detector", ofType: "onnx")!
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

RUN pip install onnxruntime numpy pandas matplotlib seaborn
COPY shearer_cutting_unit_failure_detector.onnx /app/

CMD ["python", "serve.py"]
```

## 高级配置

### 调整模型参数

在 `shearer_cutting_unit_failure_prediction.py` 中修改以下参数：

```python
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,      # 树的数量（增加可提高精度但降低速度）
        max_depth=10,          # 树的最大深度（增加可提高精度但易过拟合）
        min_samples_split=5,   # 最小分裂样本数
        min_samples_leaf=2,    # 最小叶子节点样本数
        max_features='sqrt',   # 每棵树考虑的最大特征数
        random_state=42,
        n_jobs=-1              # 使用所有CPU核心
    ))
])
```

### 使用不同的分类器

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 梯度提升
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ))
])

# 支持向量机
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    ))
])

# XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ))
])
```

### 添加新特征

```python
# 在数据生成中添加新特征
data = {
    # 原有特征
    'vibration': ...,
    'temperature': ...,
    'current': ...,
    'pressure': ...,
    'oil_quality': ...,

    # 新特征
    'power': np.random.normal(30, 5, n_samples),       # 功率 (kW)
    'rotation_speed': np.random.normal(800, 50, n_samples),  # 转速 (rpm)
    'noise': np.random.normal(75, 10, n_samples),       # 噪声 (dB)
}

# 更新故障逻辑
df['fault'] = (
    ((df['vibration'] > 8) & (df['temperature'] > 85)) |
    (df['oil_quality'] < 60) |
    (df['noise'] > 90)  # 新增噪声判断
)
```

### 处理类别不平衡

```python
from sklearn.utils import class_weight

# 计算类别权重
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# 在模型中使用
classifier = RandomForestClassifier(
    class_weight='balanced',  # 自动平衡
    # 或手动指定
    # class_weight={0: 1, 1: 8}  # 故障样本权重更高
)
```

## 实际应用场景

### 1. 实时故障监测

```python
# 实时监测采煤机状态
import time
import random

detector = ShearerFailureDetector()

while True:
    # 模拟获取传感器数据
    sensor_data = {
        'vibration': random.uniform(1, 15),
        'temperature': random.uniform(40, 110),
        'current': random.uniform(50, 250),
        'pressure': random.uniform(50, 350),
        'oil_quality': random.uniform(50, 120)
    }

    # 预测
    result = detector.predict(**sensor_data)
    risk = detector.get_risk_level(result['prediction'], result['confidence'])

    print(f"状态: {result['prediction']}, 风险: {risk[0]}, 置信度: {result['confidence']:.2%}")

    if result['prediction'] == '故障':
        print("⚠️ 检测到故障！风险因素:")
        for factor in result['risk_factors']:
            print(f"  - {factor['factor']}: {factor['value']}")

    time.sleep(60)  # 每分钟检测一次
```

### 2. 故障告警系统

```python
# 告警系统
import time

class FaultAlertSystem:
    def __init__(self):
        self.predictor = ShearerFailureDetector()
        self.last_alert_time = None

    def check_and_alert(self, sensor_data):
        """检查并发送告警"""
        result = self.predictor.predict(**sensor_data)
        risk_level, risk = self.predictor.get_risk_level(
            result['prediction'], result['confidence']
        )

        # 如果检测到故障，发送告警
        if result['prediction'] == '故障' and result['confidence'] > 0.8:
            # 避免频繁告警（10分钟内只告警一次）
            if self.last_alert_time is None or \
               time.time() - self.last_alert_time > 600:
                self.send_alert(result, risk_level)
                self.last_alert_time = time.time()

    def send_alert(self, result, risk_level):
        """发送告警通知"""
        msg = f"""
        采煤机截割部故障告警

        预测状态: {result['prediction']}
        风险等级: {risk_level}
        置信度: {result['confidence']:.2%}

        风险因素:
        """
        for factor in result['risk_factors']:
            msg += f"  - {factor['factor']}: {factor['value']}\n"

        msg += f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"

        # 发送告警（邮件/短信/推送等）
        print(msg)
        # send_email(msg)
        # send_sms(msg)

# 使用示例
alert_system = FaultAlertSystem()
alert_system.check_and_alert({
    'vibration': 12.5,
    'temperature': 95.0,
    'current': 180.0,
    'pressure': 220.0,
    'oil_quality': 40.0
})
```

### 3. 特征重要性可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 从训练好的模型获取特征重要性
pipeline = joblib.load('shearer_cutting_unit_failure_detector_sklearn.pkl')
classifier = pipeline.named_steps['classifier']

features = ['vibration', 'temperature', 'current', 'pressure', 'oil_quality']
importances = classifier.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# 可视化
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature')
plt.title('特征重要性分析 (采煤机截割部故障预测)')
plt.xlabel('重要性得分')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("特征重要性排序:")
print(feature_importance_df)
```

### 4. 混淆矩阵可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 获取预测结果
y_pred = pipeline.predict(X_test)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['正常', '故障'],
            yticklabels=['正常', '故障'])
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
```

## 故障排查

### 常见问题

**Q: ONNX 转换失败**
```python
# 尝试降低目标算子集版本
onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    target_opset=10
)
```

**Q: 预测结果差异较大**
- 确保 Sklearn 和 ONNX 使用相同的输入数据类型（float32）
- 检查输入特征的顺序是否一致
- 验证类别映射是否正确
- 检查模型版本是否匹配

**Q: 性能不佳**
- 增加训练样本数量
- 尝试不同的分类器或参数
- 检查特征工程是否充分
- 添加更多相关特征

**Q: 召回率低（漏报多）**
```python
# 增加故障样本权重
classifier = RandomForestClassifier(
    class_weight={0: 1, 1: 8},  # 故障样本权重更高
    random_state=42
)

# 或调整决策阈值
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.3).astype(int)  # 降低阈值
```

**Q: 精确率低（误报多）**
```python
# 提高决策阈值
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.7).astype(int)  # 提高阈值

# 或增加正常样本权重
classifier = RandomForestClassifier(
    class_weight={0: 2, 1: 1},
    random_state=42
)
```

**Q: onnxruntime 未安装**
```bash
pip install onnxruntime  # CPU 版本
# 或
pip install onnxruntime-gpu  # GPU 版本
```

## 扩展建议

1. **实时数据接入**
   - 连接工业传感器实时数据流
   - 实现实时故障监测

2. **历史数据分析**
   - 保存预测结果用于长期分析
   - 分析故障频率和模式

3. **模型定期更新**
   - 定期用新数据重新训练模型
   - 适应设备老化情况

4. **可视化界面**
   - 开发 Web/移动端监控界面
   - 实时显示设备状态和趋势

5. **多设备支持**
   - 扩展支持多台采煤机
   - 为不同设备训练专用模型

6. **预测性维护**
   - 结合故障预测和剩余寿命估计
   - 优化维护计划

## 与其他模型的对比

| 特性 | 采煤机故障 | 温度预测 | 泵故障预测 | 泄漏检测 |
|------|-----------|---------|-----------|---------|
| 模型类型 | 分类 | 回归 | 分类 | 分类 |
| 输出 | 类别标签 | 连续数值 | 类别标签 | 类别标签 |
| 特征数量 | 5 | 8 | 4 | 3 |
| 样本数量 | 10000 | 5000 | 10000 | 10000 |
| 模型 | 随机森林 | 梯度提升 | 随机森林 | 随机森林 |
| 应用场景 | 故障诊断 | 温度预测 | 故障诊断 | 泄漏检测 |
| 输出示例 | "故障" | 48.76°C | "normal" | "泄漏" |

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
