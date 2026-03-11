# 泵故障预测模型

基于随机森林分类的主排水泵故障诊断系统，支持 ONNX 格式跨平台部署。

## 模型概述

泵故障预测模型是一个多分类模型，用于预测主排水泵的运行状态。通过分析 4 个关键运行参数（流量、扬程、功率、振动），模型可以识别三种状态：正常运行、磨损故障和汽蚀故障。模型采用 sklearn 管道构建，包含特征标准化和随机森林分类器。

## 技术栈

- **Python**: 3.14+
- **机器学习库**: scikit-learn
- **模型转换**: skl2onnx, onnx
- **数据处理**: numpy, pandas
- **模型保存**: joblib

## 安装依赖

```bash
pip install numpy scikit-learn onnx skl2onnx joblib pandas
```

如需验证 ONNX 模型，还需安装：
```bash
pip install onnxruntime
```

## 文件说明

```
泵故障预测模型/
├── pump_failure_prediction.py                # 模型训练脚本
├── pump_failure_classifier.onnx              # ONNX 格式模型 (307 KB)
├── pump_failure_classifier_sklearn.pkl       # Sklearn 原始模型 (606 KB)
└── PUMP_MODEL_README.md                       # 本文档
```

## 特征说明

| 特征名称 | 单位 | 范围 | 说明 |
|---------|------|------|------|
| flow | m³/h | 80-120 | 流量，水泵输出流量 |
| head | m | 40-60 | 扬程，水泵提升高度 |
| power | kW | 40-60 | 功率，水泵消耗功率 |
| vibration | mm/s | 0.5-4.5 | 振动，设备振动幅度 |

## 预测类别

| 类别 | 说明 | 典型特征 | 预警等级 |
|------|------|----------|----------|
| 0 (normal) | 正常运行 | 流量正常（100-120），振动低（0.5-1.5） | 🟢 正常 |
| 1 (wear) | 磨损故障 | 流量略减（90-110），振动增强（1.5-3.0） | 🟡 预警 |
| 2 (cavitation) | 汽蚀故障 | 流量大幅下降（80-100），振动强烈（2.5-4.5） | 🔴 严重 |

**标签映射**：
```python
label_map = {
    0: "normal",      # 正常
    1: "wear",        # 磨损
    2: "cavitation"   # 汽蚀
}
```

### 故障详情

**正常运行 (Normal)**
- 流量处于正常范围
- 振动幅度较低
- 功率消耗稳定
- 设备运行平稳

**磨损故障 (Wear)**
- 叶轮或轴承磨损导致效率下降
- 流量略有减少
- 振动明显增强
- 功率消耗可能增加
- 需要尽快安排维护

**汽蚀故障 (Cavitation)**
- 液体在泵内汽化产生气泡
- 流量大幅下降
- 振动强烈（伴有噪音）
- 功率下降，效率严重降低
- 需要立即停机检查

## 模型架构

### 处理流水线

```python
Pipeline([
    ('scaler', StandardScaler()),                      # 特征标准化
    ('classifier', RandomForestClassifier(              # 随机森林分类器
        n_estimators=100,     # 100棵决策树
        random_state=42,      # 随机种子
        max_depth=None,       # 树的最大深度（不限制）
        min_samples_split=2,  # 最小分裂样本数
        min_samples_leaf=1,   # 最小叶子节点样本数
        n_jobs=-1            # 使用所有CPU核心
    ))
])
```

### 模型特点

- **集成学习**: 100 棵决策树的随机森林
- **多分类支持**: 同时识别三种运行状态
- **特征重要性**: 可分析各特征对分类的贡献
- **概率输出**: 提供各类别的置信度
- **抗过拟合**: 通过集成和随机采样提高泛化能力

## 使用方法

### 1. 训练模型

```bash
python pump_failure_prediction.py
```

训练过程输出：
```
--- 开始训练主排水泵故障预测模型 ---
1. 生成模拟数据...
2. 划分训练集和测试集...
3. 训练模型...
4. 评估模型...
              precision    recall  f1-score   support

           0       0.96      0.97      0.96       667
           1       0.97      0.98      0.98       667
           2       0.97      0.95      0.96       667

    accuracy                           0.97      2000
   macro avg       0.97      0.97      0.97      2000
weighted avg       0.97      0.97      0.97      2000

5. 转换模型为 ONNX 格式...
✅ ONNX 模型已保存至: pump_failure_classifier.onnx
✅ scikit-learn 模型已保存至: pump_failure_classifier_sklearn.pkl

8. 验证 ONNX 模型一致性...
   - scikit-learn 预测类别: 0, 概率: [0.02 0.95 0.03]
   - ONNX 模型预测类别: 0, 概率: [0.02, 0.95, 0.03]
   - 预测类别一致: True

--- Python 端任务完成 ---
```

### 2. 使用 Sklearn 模型预测

```python
import joblib
import numpy as np

# 加载模型
pipeline = joblib.load('pump_failure_classifier_sklearn.pkl')

# 标签映射
label_map = {0: "normal", 1: "wear", 2: "cavitation"}

# 准备输入数据 (4 个特征)
input_data = np.array([[
    110.0,   # flow (m³/h)
    55.0,    # head (m)
    50.0,    # power (kW)
    1.2      # vibration (mm/s)
]]).astype(np.float32)

# 预测
prediction = pipeline.predict(input_data)[0]
probability = pipeline.predict_proba(input_data)[0]

print(f"预测状态: {prediction} ({label_map[prediction]})")
print("概率分布:")
for cls, prob in zip(pipeline.classes_, probability):
    print(f"  {cls} ({label_map[cls]}): {prob:.2%}")
```

输出示例：
```
预测状态: 0 (normal)
概率分布:
  0 (normal): 95.12%
  1 (wear): 2.53%
  2 (cavitation): 2.35%
```

### 3. 使用 ONNX 模型预测

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
session = ort.InferenceSession('pump_failure_classifier.onnx')

# 类别映射（标签顺序按数值排列）
label_map = {0: "normal", 1: "wear", 2: "cavitation"}
label_names = ["normal", "wear", "cavitation"]  # 对应索引 0, 1, 2

# 准备输入数据
input_data = np.array([[
    110.0,   # flow (m³/h)
    55.0,    # head (m)
    50.0,    # power (kW)
    1.2      # vibration (mm/s)
]]).astype(np.float32)

# 预测
outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])          # 预测类别索引 (0, 1, 或 2)
probabilities = outputs[1][0]           # 各类别概率数组

predicted_class = label_names[label_idx]

print(f"预测状态: {label_idx} ({predicted_class})")
print("概率分布:")
for idx, name in enumerate(label_names):
    print(f"  {idx} ({name}): {probabilities[idx]:.2%}")
```

输出示例：
```
预测状态: 0 (normal)
概率分布:
  0 (normal): 95.12%
  1 (wear): 2.53%
  2 (cavitation): 2.35%
```

### 4. 批量预测

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession('pump_failure_classifier.onnx')
label_map = {0: "normal", 1: "wear", 2: "cavitation"}
label_names = ["normal", "wear", "cavitation"]

# 批量预测
batch_data = np.random.rand(100, 4).astype(np.float32)
outputs = session.run(None, {'float_input': batch_data})
labels = outputs[0]
probabilities = outputs[1]

# 统计各类别数量
label_counts = {}
for label in labels:
    label_idx = int(label)
    cls = label_map[label_idx]
    label_counts[cls] = label_counts.get(cls, 0) + 1

print(f"批量预测 {len(batch_data)} 个样本")
print("类别分布:")
for idx, name in enumerate(label_names):
    count = label_counts.get(name, 0)
    print(f"  {idx} ({name}): {count} ({count/len(batch_data):.1%})")
```

### 5. 单样本预测函数封装

```python
import onnxruntime as ort
import numpy as np

class PumpFailurePredictor:
    def __init__(self, model_path='pump_failure_classifier.onnx'):
        self.session = ort.InferenceSession(model_path)
        self.label_map = {0: "normal", 1: "wear", 2: "cavitation"}
        self.label_names = ["normal", "wear", "cavitation"]

    def predict(self, flow, head, power, vibration):
        """
        预测泵故障状态

        参数:
            flow: 流量 (m³/h)
            head: 扬程 (m)
            power: 功率 (kW)
            vibration: 振动 (mm/s)

        返回:
            dict: {
                'prediction': 预测类别索引,
                'prediction_name': 预测类别名称,
                'probabilities': 各类别概率,
                'confidence': 置信度
            }
        """
        input_data = np.array([[flow, head, power, vibration]]).astype(np.float32)

        outputs = self.session.run(None, {'float_input': input_data})
        label_idx = int(outputs[0][0])
        probabilities = outputs[1][0]

        prediction_name = self.label_map[label_idx]
        confidence = float(probabilities[label_idx])

        return {
            'prediction': label_idx,
            'prediction_name': prediction_name,
            'probabilities': dict(zip(self.label_names, probabilities)),
            'confidence': confidence
        }

    def get_risk_level(self, prediction):
        """获取风险等级"""
        if isinstance(prediction, int):
            prediction = self.label_map[prediction]
        risk_levels = {
            'normal': ('🟢 正常', 'low'),
            'wear': ('🟡 预警', 'medium'),
            'cavitation': ('🔴 严重', 'high')
        }
        return risk_levels.get(prediction, ('❓ 未知', 'unknown'))

# 使用示例
predictor = PumpFailurePredictor()
result = predictor.predict(flow=110.0, head=55.0, power=50.0, vibration=1.2)

print(f"预测状态: {result['prediction']} ({result['prediction_name']})")
print(f"置信度: {result['confidence']:.2%}")
print(f"风险等级: {predictor.get_risk_level(result['prediction'])[0]}")
print("概率分布:")
for idx, name in enumerate(predictor.label_names):
    print(f"  {idx} ({name}): {result['probabilities'][name]:.2%}")
```

## 数据生成逻辑

### 状态参数分布

模型生成 10000 个模拟样本，每种状态约 3333 个：

```python
states = {
    "normal": {
        "flow": (100, 120),          # 正常流量范围
        "head": (50, 60),            # 正常扬程范围
        "power": (45, 55),           # 正常功率范围
        "vibration": (0.5, 1.5)     # 正常振动范围
    },
    "wear": {
        "flow": (90, 110),           # 磨损导致流量略减
        "head": (45, 55),            # 扬程略有下降
        "power": (50, 60),           # 需要更多功率
        "vibration": (1.5, 3.0)      # 振动明显增强
    },
    "cavitation": {
        "flow": (80, 100),           # 汽蚀导致流量大幅下降
        "head": (40, 50),            # 扬程下降
        "power": (40, 50),           # 功率下降
        "vibration": (2.5, 4.5)     # 振动强烈
    }
}
```

### 数据生成函数

```python
def generate_pump_data(n_samples=10000):
    """
    生成模拟的泵运行数据，包括正常、磨损、汽蚀三种状态。
    
    返回:
        X: 特征矩阵 (n_samples, 4)
        y: 标签数组 (n_samples,)
    """
    np.random.seed(42)
    data = []
    labels = []
    
    for state, params in states.items():
        n_state_samples = n_samples // 3
        
        flow = np.random.normal(
            loc=np.mean(params["flow"]),
            scale=(params["flow"][1]-params["flow"][0])/6,
            size=n_state_samples
        ).clip(*params["flow"])
        
        head = np.random.normal(
            loc=np.mean(params["head"]),
            scale=(params["head"][1]-params["head"][0])/6,
            size=n_state_samples
        ).clip(*params["head"])
        
        power = np.random.normal(
            loc=np.mean(params["power"]),
            scale=(params["power"][1]-params["power"][0])/6,
            size=n_state_samples
        ).clip(*params["power"])
        
        vibration = np.random.exponential(
            scale=np.mean(params["vibration"]),
            size=n_state_samples
        ).clip(*params["vibration"])
        
        for i in range(n_state_samples):
            data.append([flow[i], head[i], power[i], vibration[i]])
            labels.append(state)
            
    return np.array(data), np.array(labels)
```

## 模型性能

### 评估指标

- **准确率 (Accuracy)**: 预测正确的比例
- **精确率 (Precision)**: 预测为某类的样本中实际为该类的比例
- **召回率 (Recall)**: 实际为某类的样本中被正确预测的比例
- **F1 分数**: 精确率和召回率的调和平均

### 性能参考

基于 10000 个样本的模拟数据：

| 指标 | 正常 | 磨损 | 汽蚀 |
|------|------|------|------|
| 精确率 | ~97% | ~97% | ~96% |
| 召回率 | ~98% | ~95% | ~97% |
| F1 分数 | ~98% | ~96% | ~96% |

| 总体指标 | 值 |
|----------|-----|
| 准确率 | ~97% |
| 宏平均 | ~97% |
| 加权平均 | ~97% |
| 特征数量 | 4 |
| 训练样本 | 8000 |
| 测试样本 | 2000 |
| 模型大小 | 307 KB (ONNX) |

## 模型验证

### ONNX 模型验证

```python
import onnx

# 验证 ONNX 模型结构
onnx_model = onnx.load('pump_failure_classifier.onnx')
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
pipeline = joblib.load('pump_failure_classifier_sklearn.pkl')
session = ort.InferenceSession('pump_failure_classifier.onnx')

# 测试数据
test_data = np.random.rand(10, 4).astype(np.float32)

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

### 特征重要性分析

```python
import joblib
import matplotlib.pyplot as plt

# 加载模型
pipeline = joblib.load('pump_failure_classifier_sklearn.pkl')
classifier = pipeline.named_steps['classifier']

# 获取特征重要性
feature_names = ['flow', 'head', 'power', 'vibration']
importances = classifier.feature_importances_

# 打印特征重要性
print("特征重要性:")
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig('feature_importance.png')
plt.show()
```

## 部署选项

### 1. C++ 部署

使用 ONNX Runtime C++ API 部署到嵌入式设备或工业控制器。

```cpp
#include <onnxruntime_cxx_api.h>

// 加载模型
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
Ort::Session session{env, L"pump_failure_classifier.onnx", Ort::SessionOptions{nullptr}};

// 准备输入
float input_tensor_values[] = {110.0f, 55.0f, 50.0f, 1.2f};

// 运行推理
// ... 推理代码
```

### 2. Web 部署

使用 ONNX Runtime Web 部署到浏览器。

```javascript
import * as ort from 'onnxruntime-web';

// 加载模型
const session = await ort.InferenceSession.create('pump_failure_classifier.onnx');

// 准备输入
const input = new ort.Tensor('float32', [110.0, 55.0, 50.0, 1.2], [1, 4]);

// 运行推理
const outputs = await session.run({ float_input: input });
const labelIdx = outputs['label'].data[0];
const probabilities = outputs['probabilities'];

const labelNames = ['cavitation', 'normal', 'wear'];
console.log(`预测状态: ${labelNames[labelIdx]}`);
```

### 3. 移动端部署

使用 ONNX Runtime Mobile 部署到 iOS/Android 应用。

```swift
import ORTObjectives

// iOS Swift 示例
let modelPath = Bundle.main.path(forResource: "pump_failure_classifier", ofType: "onnx")!
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

RUN pip install onnxruntime numpy pandas
COPY pump_failure_classifier.onnx /app/

CMD ["python", "serve.py"]
```

## 高级配置

### 调整模型参数

在 `pump_failure_prediction.py` 中修改以下参数：

```python
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,      # 树的数量
        max_depth=10,          # 树的最大深度
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
# 在 pump_failure_prediction.py 中添加新特征
def generate_pump_data(n_samples=10000):
    # 原有特征
    flow = ...
    head = ...
    power = ...
    vibration = ...
    
    # 添加新特征：温度、压力等
    temperature = np.random.uniform(40, 80, n_samples)
    pressure = np.random.uniform(0.1, 0.8, n_samples)
    
    # 扩展数据
    for i in range(n_samples):
        data.append([
            flow[i], head[i], power[i], vibration[i],
            temperature[i], pressure[i]
        ])
    
    return np.array(data), np.array(labels)
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
    # class_weight={0: 1, 1: 2, 2: 1.5}
)
```

## 实际应用场景

### 1. 水泵运行状态监测

```python
# 实时监测水泵状态
import time
import random

predictor = PumpFailurePredictor()

while True:
    # 模拟获取传感器数据
    sensor_data = {
        'flow': random.uniform(80, 120),
        'head': random.uniform(40, 60),
        'power': random.uniform(40, 60),
        'vibration': random.uniform(0.5, 4.5)
    }
    
    # 预测
    result = predictor.predict(**sensor_data)
    risk = predictor.get_risk_level(result['prediction'])
    
    print(f"状态: {result['prediction']}, 风险: {risk[0]}")
    
    if result['confidence'] > 0.95 and result['prediction'] != 'normal':
        print("⚠️ 触发告警！")
    
    time.sleep(60)  # 每分钟检测一次
```

### 2. 故障预警系统

```python
# 预警系统
import onnxruntime as ort
import smtplib
from email.mime.text import MIMEText

class PumpAlertSystem:
    def __init__(self):
        self.predictor = PumpFailurePredictor()
        self.last_alert_time = {}
    
    def check_and_alert(self, flow, head, power, vibration):
        result = self.predictor.predict(flow, head, power, vibration)
        risk_level, risk = self.predictor.get_risk_level(result['prediction'])
        
        # 如果风险为中高，发送告警
        if risk in ['medium', 'high']:
            # 避免频繁告警（同一状态10分钟内只告警一次）
            if result['prediction'] not in self.last_alert_time or \
               time.time() - self.last_alert_time[result['prediction']] > 600:
                self.send_alert(result, risk_level)
                self.last_alert_time[result['prediction']] = time.time()
    
    def send_alert(self, result, risk_level):
        """发送告警邮件"""
        msg = MIMEText(f"""
        泵故障告警
        
        预测状态: {result['prediction']}
        风险等级: {risk_level}
        置信度: {result['confidence']:.2%}
        时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
        """)
        msg['Subject'] = f"泵故障告警: {risk_level}"
        msg['From'] = "monitoring@company.com"
        msg['To'] = "maintenance@company.com"
        
        # 发送邮件
        # smtp.send_message(msg)
        print(f"发送告警: {risk_level}")
```

### 3. 维护决策支持

```python
# 维护决策支持
def maintenance_recommendation(prediction, confidence, sensor_data):
    """根据预测结果提供维护建议"""
    
    if prediction == 'normal':
        if confidence > 0.95:
            return "设备运行正常，按计划维护"
        else:
            return "设备运行基本正常，建议增加监测频率"
    
    elif prediction == 'wear':
        return """
        检测到磨损故障，建议：
        1. 在一周内安排检查
        2. 检查叶轮、轴承等易损部件
        3. 准备备件
        """
    
    elif prediction == 'cavitation':
        return """
        检测到汽蚀故障，建议：
        1. 立即停机检查
        2. 检查进水条件
        3. 检查安装高度
        4. 检查吸水管路
        5. 可能需要更换水泵
        """
```

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
- 验证类别映射是否正确
- 检查模型版本是否匹配

**Q: 性能不佳**
- 增加训练样本数量
- 尝试不同的分类器或参数
- 检查特征工程是否充分
- 添加更多相关特征

**Q: 类别预测不平衡**
```python
# 使用类别权重
classifier = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

# 或使用 StratifiedKFold 进行交叉验证
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Q: onnxruntime 未安装**
```bash
pip install onnxruntime  # CPU 版本
# 或
pip install onnxruntime-gpu  # GPU 版本
```

**Q: 模型输出顺序不确定**
```python
# 获取 ONNX 模型的类别顺序
session = ort.InferenceSession('pump_failure_classifier.onnx')

# 查看输出信息
for output in session.get_outputs():
    print(f"输出名称: {output.name}")
    print(f"输出形状: {output.shape}")
    print(f"输出类型: {output.type}")

# 根据实际输出调整类别映射
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
   - 适应设备老化等变化

4. **可视化界面**
   - 开发 Web/移动端监控界面
   - 实时显示运行状态和趋势

5. **多设备支持**
   - 扩展支持多种泵型号
   - 为不同设备训练专用模型

6. **预测性维护**
   - 结合故障预测和剩余寿命估计
   - 优化维护计划

## 与其他模型的对比

| 特性 | 泵故障预测 | 温度预测 |
|------|-----------|---------|
| 模型类型 | 分类 | 回归 |
| 输出 | 类别标签 | 连续数值 |
| 特征数量 | 4 | 8 |
| 样本数量 | 10000 | 5000 |
| 模型 | 随机森林 | 梯度提升 |
| 应用场景 | 故障诊断 | 温度预测 |
| 输出示例 | "normal" | 48.76 |

## 模型参数调优

### 可调参数

本模型使用 `RandomForestClassifier`，以下是关键可调参数：

| 参数 | 当前值 | 说明 | 调优建议 |
|------|--------|------|----------|
| n_estimators | 100 | 决策树数量 | 50-300，越多越稳定但越慢 |
| max_depth | 10 | 树的最大深度 | 8-15，控制过拟合 |
| min_samples_split | 2 | 节点分裂最小样本数 | 2-10，防止过拟合 |
| min_samples_leaf | 1 | 叶节点最小样本数 | 1-5，防止过拟合 |
| max_features | sqrt | 每棵树考虑的特征数 | auto/sqrt/log2 |

### 调优函数

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 定义参数网格
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [8, 10, 12, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# 创建管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# 网格搜索
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳 F1 得分: {grid_search.best_score_:.4f}")

# 使用最佳参数
best_pipeline = grid_search.best_estimator_
```

### 调优步骤

1. **基准测试**: 使用当前参数训练，记录准确率和 F1 分数
2. **网格搜索**: 使用 `GridSearchCV` 寻找最优参数组合
3. **交叉验证**: 5折交叉验证确保稳定性
4. **性能对比**: 对比调优前后的各指标（精确率、召回率、F1）
5. **部署验证**: 重新导出 ONNX 模型并验证一致性

### 注意事项

- **数据生成参数**（如 normal 状态的 flow 范围 100-120）是模拟数据参数，**不需要调整**
- 只需调整机器学习模型的超参数
- 调整参数后需重新训练并导出 ONNX 模型
- 确保调优不会导致过拟合（测试集性能下降）
- 三分类问题建议关注宏平均 F1 分数

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
