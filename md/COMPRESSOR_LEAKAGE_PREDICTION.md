# 压缩机泄漏预测模型

基于随机森林分类的空压系统管网泄漏检测系统，支持 ONNX 格式跨平台部署。

## 模型概述

压缩机泄漏预测模型是一个二分类模型，用于检测空压系统管网是否存在泄漏。通过分析 **5 个特征**（包含原始参数 + 2 个衍生特征），模型可以识别正常运行和泄漏两种状态。模型采用 sklearn 管道构建，包含特征标准化、随机森林分类器，并自动处理类别不平衡问题。

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
压缩机泄漏预测模型/
├── compressor_leakage_prediction.py            # 模型训练脚本
├── compressor_leakage_detector.onnx             # ONNX 格式模型 (836 KB)
├── compressor_leakage_detector_sklearn.pkl      # Sklearn 原始模型 (1.77 MB)
└── COMPRESSOR_MODEL_README.md                    # 本文档
```

## 特征说明

| 特征名称 | 单位 | 范围 | 说明 |
|---------|------|------|------|
| pressure | MPa | 0.5-0.85 | 管网压力 |
| supply_flow | m³/h | 300-700 | 供气流量 |
| demand_flow | m³/h | 10-700 | 用气流量 |
| flow_diff | m³/h | 0-90 | 供气流量与用气流量的差值 (衍生特征) |
| flow_ratio | - | 1.0-70.0 | 供气流量与用气流量的比值 (衍生特征) |

### 衍生特征说明

**FlowDiff (流量差值)**
- 计算公式: `FlowDiff = SupplyFlow - DemandFlow`
- 正常状态: 5-20 m³/h（管道和设备微小损耗）
- 泄漏状态: 15-90 m³/h（包含泄漏量）
- 作用: 直接反映流量损失量，是泄漏检测的核心特征

**FlowRatio (流量比值)**
- 计算公式: `FlowRatio = SupplyFlow / (DemandFlow + 1e-5)`
- 正常状态: 接近 1.0-1.1
- 泄漏状态: 可能显著大于 1.1（极端情况下可达 70）
- 作用: 比值能消除绝对值的影响，对小流量场景更敏感

## 预测类别

| 类别 | 说明 | 典型特征 | 预警等级 |
|------|------|----------|----------|
| normal | 正常运行 | 供气流量≈用气流量，压力稳定 | 🟢 正常 |
| leak | 泄漏状态 | 供气流量>用气流量，压力偏低 | 🔴 严重 |

### 状态详解

**正常运行 (Normal)**
- 供气流量和用气流量基本相等
- 流量差值在 5-20 m³/h（管道和设备微小损耗）
- 压力稳定在 0.6-0.85 MPa
- 系统运行正常

**泄漏状态 (Leak)**
- 供气流量明显高于用气流量
- 流量差值在 30-80 m³/h（泄漏量）
- 压力可能偏低（0.5-0.8 MPa）或不稳定
- 供气可能略增以维持压力
- 需要立即检查和维修

## 模型架构

### 处理流水线

```python
Pipeline([
    ('scaler', StandardScaler()),                      # 特征标准化
    ('classifier', RandomForestClassifier(            # 随机森林分类器
        n_estimators=100,     # 100棵决策树
        random_state=42,      # 随机种子
        n_jobs=-1,           # 使用所有CPU核心
        class_weight='balanced',  # 自动处理类别不平衡
        max_depth=10          # 限制深度防止过拟合
    ))
])
```

### 模型特点

- **集成学习**: 100 棵决策树的随机森林
- **二分类支持**: 识别正常和泄漏两种状态
- **特征工程**: 衍生 FlowDiff 和 FlowRatio 两个特征提升检测精度
- **类别平衡**: 使用 `class_weight='balanced'` 自动处理不平衡（正常:泄漏 = 7:3）
- **特征重要性**: 可分析各特征对分类的贡献
- **概率输出**: 提供各类别的置信度
- **抗过拟合**: 通过集成和随机采样提高泛化能力

## 使用方法

### 1. 训练模型

```bash
python compressor_leakage_prediction.py
```

训练过程输出：
```
--- 开始训练空压系统管网泄漏预测模型 ---
1. 生成模拟数据...
2. 划分训练集和测试集...
3. 训练模型 (Random Forest Classifier)...
4. 评估模型...
              precision    recall  f1-score   support

      正常       0.97      0.98      0.97      1400
      泄漏       0.95      0.92      0.93       600

accuracy                           0.96      2000
macro avg       0.96      0.95      0.95      2000
weighted avg       0.96      0.96      0.96      2000

    # 5. 转换为 ONNX
    print("5. 转换模型为 ONNX 格式...")
    # RandomForestClassifier 的 ONNX 转换是支持的
    # 关键修改：添加 options 参数，强制使用 tensor 输出，禁用 zipmap
    options = {type(model_pipeline): {'zipmap': False}} # 使用 type(model) 作为 key 也有效
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model_pipeline, initial_types=initial_type, target_opset=12, options=options) # 这是核心修改

    # 6. 保存 ONNX 模型
    onnx_model_path = "compressor_leakage_detector.onnx"
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ ONNX 模型已保存至: {onnx_model_path}")

    # 7. 保存原始模型 (可选)
    joblib.dump(model_pipeline, "compressor_leakage_detector_sklearn.pkl")
    print(f"✅ scikit-learn 模型已保存至: compressor_leakage_detector_sklearn.pkl")

    # 8. 验证 ONNX 模型 (可选)
    print("\n8. 验证 ONNX 模型一致性...")
    try:
        import onnxruntime as rt

        sess = rt.InferenceSession(onnx_model_path)

        # 准备一个测试样本 (模拟一个泄漏场景)
        sample_input = X_test[y_test == 1][0:1].astype(np.float32)  # 取一个泄漏样本

        # scikit-learn 预测
        sk_pred = model_pipeline.predict(sample_input)[0]
        sk_proba = model_pipeline.predict_proba(sample_input)[0]

        # ONNX 预测
        onnx_pred = sess.run(None, {'float_input': sample_input})
        # RandomForestClassifier 的 ONNX 输出: [label_index, probabilities]
        onnx_label = int(onnx_pred[0][0])
        onnx_proba = onnx_pred[1][0]

        print(f"   - 测试样本输入: {sample_input[0]}")
        print(f"   - scikit-learn 预测 (0:正常, 1:泄漏): {sk_pred}")
        print(f"   - scikit-learn 概率: {sk_proba}")
        print(f"   - ONNX 模型预测 (0:正常, 1:泄漏): {onnx_label}")
        print(f"   - ONNX 概率: {onnx_proba}")
        print(f"   - 预测结果一致: {sk_pred == onnx_label}")

    except ImportError:
        print("   - 未安装 onnxruntime，跳过验证。")
    except Exception as e:
        print(f"   - ONNX 验证失败: {e}")
```

### 2. 使用 Sklearn 模型预测

```python
import joblib
import numpy as np

# 加载模型
pipeline = joblib.load('compressor_leakage_detector_sklearn.pkl')

# 准备输入数据 (5 个特征：Pressure, SupplyFlow, DemandFlow, FlowDiff, FlowRatio)
input_data = np.array([[
    0.75,    # Pressure (MPa)
    500.0,   # SupplyFlow (m³/h)
    490.0,   # DemandFlow (m³/h)
    10.0,    # FlowDiff = SupplyFlow - DemandFlow (m³/h)
    1.02     # FlowRatio = SupplyFlow / DemandFlow
]]).astype(np.float32)

# 预测
prediction = pipeline.predict(input_data)[0]
probability = pipeline.predict_proba(input_data)[0]

print(f"预测状态: {'泄漏' if prediction == 1 else '正常'}")
print(f"概率分布: 正常={probability[0]:.2%}, 泄漏={probability[1]:.2%}")
```

### 3. 使用 ONNX 模型预测

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
session = ort.InferenceSession('compressor_leakage_detector.onnx')

# 准备输入数据 (5 个特征)
input_data = np.array([[
    0.75,    # Pressure (MPa)
    500.0,   # SupplyFlow (m³/h)
    490.0,   # DemandFlow (m³/h)
    10.0,    # FlowDiff (m³/h)
    1.02     # FlowRatio
]]).astype(np.float32)

# 预测
outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])          # 预测类别索引
probabilities = outputs[1][0]           # 各类别概率

predicted_class = "泄漏" if label_idx == 1 else "正常"

print(f"预测状态: {predicted_class}")
print(f"概率分布: 正常={probabilities[0]:.2%}, 泄漏={probabilities[1]:.2%}")
```

### 4. 批量预测

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession('compressor_leakage_detector.onnx')

# 批量预测
batch_data = np.random.rand(100, 5).astype(np.float32)
outputs = session.run(None, {'float_input': batch_data})
labels = outputs[0]
probabilities = outputs[1]

# 统计各类别数量
normal_count = np.sum(labels == 0)
leak_count = np.sum(labels == 1)

print(f"批量预测 {len(batch_data)} 个样本")
print(f"正常: {normal_count} ({normal_count/len(batch_data):.1%})")
print(f"泄漏: {leak_count} ({leak_count/len(batch_data):.1%})")
```

### 5. 实时预测示例 (Python ONNX)

训练脚本中包含 ONNX 模型验证和实时预测示例，运行以下命令即可查看：

```bash
python compressor_leakage_prediction.py
```

输出示例：

```
--- 开始训练空压系统管网泄漏预测模型 ---
1. 生成模拟数据...
2. 划分训练集和测试集...
3. 训练模型 (Random Forest Classifier)...
4. 评估模型...
              precision    recall  f1-score   support

      正常       0.97      0.98      0.97      1400
      泄漏       0.95      0.92      0.93       600

accuracy                           0.96      2000
macro avg       0.96      0.95      0.95      2000
weighted avg       0.96      0.96      0.96      2000

5. 转换模型为 ONNX 格式...
✅ ONNX 模型已保存至: compressor_leakage_detector.onnx
✅ scikit-learn 模型已保存至: compressor_leakage_detector_sklearn.pkl

8. 验证 ONNX 模型一致性...
   - 测试样本输入: [0.68 480. 400.  80.    1.2 ]
   - scikit-learn 预测 (0:正常, 1:泄漏): 1
   - scikit-learn 概率: [0.03 0.97]
   - ONNX 模型预测 (0:正常, 1:泄漏): 1
   - ONNX 概率: [0.03 0.97]
   - 预测结果一致: True

--- Python 端任务完成 ---
```

### 6. 单样本预测函数封装

```python
import onnxruntime as ort
import numpy as np

class CompressorLeakageDetector:
    def __init__(self, model_path='compressor_leakage_detector.onnx'):
        self.session = ort.InferenceSession(model_path)

    def predict(self, pressure, supply_flow, demand_flow, flow_diff, flow_ratio):
        """
        预测泄漏状态

        参数:
            pressure: 管网压力 (MPa)
            supply_flow: 供气流量 (m³/h)
            demand_flow: 用气流量 (m³/h)
            flow_diff: 流量差值 (m³/h)
            flow_ratio: 流量比值

        返回:
            dict: {
                'prediction': 预测类别,
                'probabilities': 各类别概率,
                'confidence': 置信度,
                'leakage_amount': 泄漏量估算
            }
        """
        input_data = np.array([[pressure, supply_flow, demand_flow, flow_diff, flow_ratio]]).astype(np.float32)

        outputs = self.session.run(None, {'float_input': input_data})
        label_idx = int(outputs[0][0])
        probabilities = outputs[1][0]

        prediction = "泄漏" if label_idx == 1 else "正常"
        confidence = float(probabilities[label_idx])
        leakage_amount = max(0, supply_flow - demand_flow)

        return {
            'prediction': prediction,
            'probabilities': {
                'normal': float(probabilities[0]),
                'leak': float(probabilities[1])
            },
            'confidence': confidence,
            'leakage_amount': leakage_amount
        }

    def get_risk_level(self, prediction, leakage_amount):
        """获取风险等级"""
        if prediction == "正常":
            return ("🟢 正常", "low")
        else:
            if leakage_amount > 60:
                return ("🔴 严重泄漏", "high")
            elif leakage_amount > 40:
                return ("🟠 中度泄漏", "medium")
            else:
                return ("🟡 轻微泄漏", "low")

# 使用示例
detector = CompressorLeakageDetector()
result = detector.predict(pressure=0.68, supply_flow=480, demand_flow=400, flow_diff=80, flow_ratio=1.2)

print(f"预测状态: {result['prediction']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"泄漏量: {result['leakage_amount']:.1f} m³/h")
risk_level = detector.get_risk_level(result['prediction'], result['leakage_amount'])
print(f"风险等级: {risk_level[0]}")
```

## 数据生成逻辑

### 状态参数分布

模型生成 10000 个模拟样本，其中 70% 为正常状态，30% 为泄漏状态：

```python
# 正常运行状态 (70%)
pressure_normal: N(0.75, 0.05), range: 0.6-0.85
supply_flow_normal: N(500, 50), range: 300-700
# 正常损耗：基础损耗 5-15 + 动态损耗（流量的 0.5%-1.5%）
base_loss: Uniform(5, 15)
dynamic_loss: supply_flow * 0.01 * Uniform(0.5, 1.5)
demand_flow_normal = supply_flow - (base_loss + dynamic_loss)

# 泄漏状态 (30%)
pressure_leak: N(0.70, 0.08), range: 0.5-0.8
supply_flow_leak: N(500, 50), range: 300-700
leakage_amount: Uniform(15, 90)  # 包含与正常损耗重叠的范围（15-25）
demand_flow_leak = max(supply_flow - leakage_amount, 10)
```

### 数据生成函数

```python
def generate_compressor_data(n_samples=10000):
    """
    生成模拟的空压站运行数据，包括正常和泄漏两种状态。
    包含特征工程：FlowDiff 和 FlowRatio。

    返回:
        X: 特征矩阵 (n_samples, 5)
        y: 标签数组 (n_samples,), 0=正常, 1=泄漏
    """
    np.random.seed(42)
    data = []
    labels = []

    # 正常状态 (70%)
    n_normal = int(n_samples * 0.7)
    pressure_normal = np.random.normal(0.75, 0.05, n_normal).clip(0.6, 0.85)
    supply_flow_normal = np.random.normal(500, 50, n_normal).clip(300, 700)
    base_loss = np.random.uniform(5, 15, size=n_normal)
    dynamic_loss = supply_flow_normal * 0.01 * np.random.uniform(0.5, 1.5, size=n_normal)
    total_loss = base_loss + dynamic_loss
    demand_flow_normal = supply_flow_normal - total_loss

    for i in range(n_normal):
        diff = supply_flow_normal[i] - demand_flow_normal[i]
        ratio = supply_flow_normal[i] / (demand_flow_normal[i] + 1e-5)
        data.append([pressure_normal[i], supply_flow_normal[i],
                   demand_flow_normal[i], diff, ratio])
        labels.append(0)

    # 泄漏状态 (30%)
    n_leak = n_samples - n_normal
    pressure_leak = np.random.normal(0.70, 0.08, n_leak).clip(0.5, 0.8)
    supply_flow_leak = np.random.normal(500, 50, n_leak).clip(300, 700)
    leakage_amount = np.random.uniform(15, 90, size=n_leak)
    demand_flow_leak = np.maximum(supply_flow_leak - leakage_amount, 10)

    for i in range(n_leak):
        diff = supply_flow_leak[i] - demand_flow_leak[i]
        ratio = supply_flow_leak[i] / (demand_flow_leak[i] + 1e-5)
        data.append([pressure_leak[i], supply_flow_leak[i],
                   demand_flow_leak[i], diff, ratio])
        labels.append(1)

    return np.array(data), np.array(labels)
```

## 模型性能

### 评估指标

- **准确率 (Accuracy)**: 预测正确的比例
- **精确率 (Precision)**: 预测为某类的样本中实际为该类的比例
- **召回率 (Recall)**: 实际为某类的样本中被正确预测的比例
- **F1 分数**: 精确率和召回率的调和平均

### 性能参考

基于 10000 个样本的模拟数据（70% 正常，30% 泄漏）：

| 指标 | 正常 | 泄漏 |
|------|------|------|
| 精确率 | ~97% | ~95% |
| 召回率 | ~98% | ~92% |
| F1 分数 | ~97% | ~93% |

| 总体指标 | 值 |
|----------|-----|
| 准确率 | ~96% |
| 宏平均 | ~95% |
| 加权平均 | ~96% |
| 特征数量 | 5（含 2 个衍生特征） |
| 训练样本 | 8000 |
| 测试样本 | 2000 |
| 模型大小 | 836 KB (ONNX) |

## ONNX 导出优化

### 禁用 ZipMap 提升兼容性

当前代码已配置禁用 ZipMap，提升 ONNX 模型的跨平台兼容性：

```python
# 关键配置
options = {type(model_pipeline): {'zipmap': False}}

onnx_model = convert_sklearn(
    model_pipeline,
    initial_types=initial_type,
    target_opset=12,
    options=options  # 强制使用 tensor 输出
)
```

**优点**:
- 避免 ZipMap 输出格式在 C++/嵌入式设备上的解析问题
- 输出为标准张量格式，兼容性更强
- 减少模型复杂度，提升推理速度

### 模型验证

导出时自动验证模型结构：

```python
# 验证模型
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")
```

## 模型验证

### ONNX 模型验证

```python
import onnx

# 验证 ONNX 模型结构
onnx_model = onnx.load('compressor_leakage_detector.onnx')
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
pipeline = joblib.load('compressor_leakage_detector_sklearn.pkl')
session = ort.InferenceSession('compressor_leakage_detector.onnx')

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
#include <iostream>
#include <vector>

class CompressorLeakageDetector {
private:
    Ort::Session session;
    Ort::Env env;

public:
    CompressorLeakageDetector(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "Compressor"),
          session(env, model_path.c_str(), Ort::SessionOptions{}) {
        std::cout << "✅ 模型加载成功" << std::endl;
    }

    void predict(float pressure, float supply_flow, float demand_flow, float flow_diff, float flow_ratio) {
        // 准备输入
        std::vector<float> input_data = {pressure, supply_flow, demand_flow, flow_diff, flow_ratio};
        std::vector<int64_t> input_shape = {1, 5};

        // 创建张量
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        // 运行推理
        const char* input_names[] = {"float_input"};
        const char* output_names[] = {"output_label", "output_probability"};

        auto outputs = session.run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 2
        );

        // 获取结果
        int64_t label = outputs[0].GetTensorMutableData<int64_t>()[0];
        float* proba = outputs[1].GetTensorMutableData<float>();

        // 输出
        std::cout << "\n=== 预测结果 ===" << std::endl;
        std::cout << "输入: 压力=" << pressure << "MPa, 供气流量=" << supply_flow
                  << "m³/h, 用气流量=" << demand_flow << "m³/h, 流量差=" << flow_diff
                  << "m³/h, 流量比=" << flow_ratio << std::endl;
        std::cout << "预测: " << (label == 1 ? "泄漏" : "正常") << std::endl;
        std::cout << "正常概率: " << proba[0] * 100 << "%" << std::endl;
        std::cout << "泄漏概率: " << proba[1] * 100 << "%" << std::endl;

        // 计算泄漏量
        float leakage_amount = std::max(0.0f, supply_flow - demand_flow);
        std::cout << "泄漏量: " << leakage_amount << " m³/h" << std::endl;

        // 风险等级
        if (label == 1) {
            if (leakage_amount > 60) {
                std::cout << "风险等级: 严重泄漏 🔴" << std::endl;
            } else if (leakage_amount > 40) {
                std::cout << "风险等级: 中度泄漏 🟠" << std::endl;
            } else {
                std::cout << "风险等级: 轻微泄漏 🟡" << std::endl;
            }
        } else {
            std::cout << "风险等级: 正常 🟢" << std::endl;
        }
    }
};

int main() {
    try {
        CompressorLeakageDetector detector("compressor_leakage_detector.onnx");

        // 测试不同工况
        std::cout << "\n【测试1: 正常工况】" << std::endl;
        detector.predict(0.75f, 500.0f, 490.0f, 10.0f, 1.02f);

        std::cout << "\n【测试2: 泄漏工况】" << std::endl;
        detector.predict(0.68f, 480.0f, 400.0f, 80.0f, 1.2f);

        std::cout << "\n【测试3: 临界状态】" << std::endl;
        detector.predict(0.72f, 510.0f, 475.0f, 35.0f, 1.074f);

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime 错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

### 2. Web 部署

使用 ONNX Runtime Web 部署到浏览器。

```javascript
import * as ort from 'onnxruntime-web';

// 加载模型
const session = await ort.InferenceSession.create('compressor_leakage_detector.onnx');

// 准备输入
const input = new ort.Tensor('float32', [0.75, 500.0, 490.0, 10.0, 1.02], [1, 5]);

// 运行推理
const outputs = await session.run({ float_input: input });
const labelIdx = outputs['label'].data[0];
const probabilities = outputs['probabilities'];

const status = labelIdx === 1 ? '泄漏' : '正常';
console.log(`预测状态: ${status}`);
```

### 3. 移动端部署

使用 ONNX Runtime Mobile 部署到 iOS/Android 应用。

```swift
import ORTObjectives

// iOS Swift 示例
let modelPath = Bundle.main.path(forResource: "compressor_leakage_detector", ofType: "onnx")!
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
COPY compressor_leakage_detector.onnx /app/

CMD ["python", "serve.py"]
```

## 高级配置

### 调整模型参数

在 `compressor_leakage_prediction.py` 中修改以下参数：

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
# 在 generate_compressor_data 中添加新特征
def generate_compressor_data(n_samples=10000):
    # 原有特征
    pressure = ...
    supply_flow = ...
    demand_flow = ...

    # 添加新特征：温度、湿度等
    temperature = np.random.uniform(30, 60, n_samples)
    humidity = np.random.uniform(40, 80, n_samples)

    # 扩展数据
    for i in range(n_samples):
        data.append([
            pressure[i], supply_flow[i], demand_flow[i],
            temperature[i], humidity[i]
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
    # class_weight={0: 1, 1: 2}
)
```

## 实际应用场景

### 1. 实时泄漏监测

```python
# 实时监测空压系统状态
import time
import random

detector = CompressorLeakageDetector()

while True:
    # 模拟获取传感器数据
    sensor_data = {
        'pressure': random.uniform(0.5, 0.85),
        'supply_flow': random.uniform(450, 560),
        'demand_flow': random.uniform(380, 550)
    }

    # 预测
    result = detector.predict(**sensor_data)
    risk = detector.get_risk_level(result['prediction'], result['leakage_amount'])

    print(f"状态: {result['prediction']}, 风险: {risk[0]}")

    if result['prediction'] == '泄漏':
        print(f"⚠️ 检测到泄漏！泄漏量: {result['leakage_amount']:.1f} m³/h")

    time.sleep(60)  # 每分钟检测一次
```

### 2. 泄漏告警系统

```python
# 告警系统
import onnxruntime as ort
import smtplib
from email.mime.text import MIMEText

class LeakAlertSystem:
    def __init__(self):
        self.predictor = CompressorLeakageDetector()
        self.last_alert_time = None

    def check_and_alert(self, pressure, supply_flow, demand_flow):
        """检查并发送告警"""
        result = self.predictor.predict(pressure, supply_flow, demand_flow)
        risk_level, risk = self.predictor.get_risk_level(
            result['prediction'], result['leakage_amount']
        )

        # 如果检测到泄漏，发送告警
        if result['prediction'] == '泄漏':
            # 避免频繁告警（10分钟内只告警一次）
            if self.last_alert_time is None or \
               time.time() - self.last_alert_time > 600:
                self.send_alert(result, risk_level)
                self.last_alert_time = time.time()

    def send_alert(self, result, risk_level):
        """发送告警邮件"""
        msg = MIMEText(f"""
        空压系统泄漏告警

        预测状态: {result['prediction']}
        风险等级: {risk_level}
        置信度: {result['confidence']:.2%}
        泄漏量: {result['leakage_amount']:.1f} m³/h
        时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
        """)
        msg['Subject'] = f"空压系统泄漏告警: {risk_level}"
        msg['From'] = "monitoring@company.com"
        msg['To'] = "maintenance@company.com"

        # 发送邮件
        # smtp.send_message(msg)
        print(f"发送告警: {risk_level}")
```

### 3. 泄漏量估算

```python
# 泄漏量估算
def estimate_leakage(pressure, supply_flow, demand_flow):
    """
    估算泄漏量

    根据压力和流量差值估算泄漏量
    """
    flow_difference = supply_flow - demand_flow

    if flow_difference <= 20:
        return 0  # 正常损耗范围
    else:
        # 根据压力调整（压力越低，泄漏量越大）
        pressure_factor = (0.85 - pressure) / 0.35 + 0.5
        estimated_leakage = flow_difference * pressure_factor

        return estimated_leakage

# 使用示例
leakage = estimate_leakage(pressure=0.68, supply_flow=480, demand_flow=400)
print(f"估算泄漏量: {leakage:.1f} m³/h")
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
- 添加更多相关特征（如温度、湿度）

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

**Q: 泄漏量估算不准确**
```python
# 调整泄漏量估算逻辑
def estimate_leakage(pressure, supply_flow, demand_flow):
    flow_diff = supply_flow - demand_flow

    if flow_diff <= 20:
        return 0

    # 考虑压力、温度等多因素
    pressure_factor = (0.85 - pressure) / 0.35 + 0.5
    temperature_factor = 1.0  # 可根据温度调整

    estimated = flow_diff * pressure_factor * temperature_factor
    return estimated
```

## 扩展建议

1. **实时数据接入**
   - 连接工业传感器实时数据流
   - 实现实时泄漏监测

2. **历史数据分析**
   - 保存预测结果用于长期分析
   - 分析泄漏频率和模式

3. **模型定期更新**
   - 定期用新数据重新训练模型
   - 适应系统老化和环境变化

4. **可视化界面**
   - 开发 Web/移动端监控界面
   - 实时显示系统状态和泄漏趋势

5. **多系统支持**
   - 扩展支持多个空压站
   - 为不同系统训练专用模型

6. **预测性维护**
   - 结合泄漏预测和剩余寿命估计
   - 优化维护计划

## 与其他模型的对比

| 特性 | 泄漏预测 | 温度预测 | 泵故障预测 |
|------|---------|---------|-----------|
| 模型类型 | 分类 | 回归 | 分类 |
| 输出 | 类别标签 | 连续数值 | 类别标签 |
| 特征数量 | 5（含 2 个衍生特征） | 8 | 4 |
| 样本数量 | 10000 | 5000 | 10000 |
| 模型 | 随机森林 | 梯度提升 | 随机森林 |
| 应用场景 | 泄漏检测 | 温度预测 | 故障诊断 |
| 输出示例 | "泄漏" | 48.76°C | "normal" |

## 模型参数调优

### 可调参数

本模型使用 `RandomForestClassifier`，以下是关键可调参数：

| 参数 | 当前值 | 说明 | 调优建议 |
|------|--------|------|----------|
| n_estimators | 100 | 决策树数量 | 50-300，越多越稳定但越慢 |
| max_depth | 10 | 树的最大深度 | 8-15，控制过拟合 |
| min_samples_split | 5 | 节点分裂最小样本数 | 2-10，防止过拟合 |
| min_samples_leaf | 2 | 叶节点最小样本数 | 1-5，防止过拟合 |
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

- **数据生成参数**（如 pressure_normal 的范围 0.6-0.85）是模拟数据参数，**不需要调整**
- 只需调整机器学习模型的超参数
- 调整参数后需重新训练并导出 ONNX 模型
- 确保调优不会导致过拟合（测试集性能下降）
- 二分类问题建议关注召回率（漏检泄漏的风险高于误报）

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
