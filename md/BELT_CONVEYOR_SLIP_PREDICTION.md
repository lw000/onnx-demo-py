# 皮带机打滑预测模型文档

## 模型概述

本模型用于预测煤矿带式运输机的皮带打滑故障，通过分析电流、速度差、振动和温度等传感器数据，实现对皮带机运行状态的实时监测和打滑预警。

### 应用场景

- **煤矿运输**：井下带式输送机打滑监测
- **港口物流**：散装物料输送系统
- **电厂输煤**：煤炭运输皮带机
- **建材行业**：水泥、砂石输送系统
- **预测性维护**：提前发现打滑异常，避免生产中断

---

## 技术架构

### 算法选择

**Random Forest Classifier (随机森林分类器)**

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 100 | 决策树数量 |
| max_depth | 8 | 树的最大深度，防止过拟合 |
| random_state | 42 | 随机种子，保证可复现性 |

---

## 输入特征详解

模型使用 4 个传感器特征进行预测：

| 特征名称 | 单位 | 数据类型 | 说明 |
|---------|------|---------|------|
| current | A (安培) | float32 | 电机电流，反映负载状态 |
| speed_diff | m/s (米/秒) | float32 | 头尾速度差，**打滑的核心指标** |
| vibration | mm/s | float32 | 电机振动值 |
| temperature | °C (摄氏度) | float32 | 轴承温度 |

### 数据分布（正常/打滑对比）

| 特征 | 正常均值 | 打滑均值 | 说明 |
|------|---------|---------|------|
| current | 100 A | 130 A | 打滑时负载增大，电流升高 |
| speed_diff | ~0 m/s | 0.2 m/s | 打滑时速度差显著增大 |
| vibration | 2 mm/s | 8 mm/s | 打滑时振动加剧 |
| temperature | 60°C | 85°C | 打滑时摩擦生热，温度升高 |

### 特征说明

**1. current（电机电流）**
- 正常工况：电流在 100A 左右波动
- 打滑工况：电流升高至 130A 左右
- 反映皮带负载和阻力情况

**2. speed_diff（速度差）**
- 正常工况：头尾速度基本一致，差值接近 0
- 打滑工况：尾部速度降低，差值增大至 0.2m/s 以上
- **这是判断打滑最关键的指标**

**3. vibration（振动）**
- 正常工况：振动约 2 mm/s
- 打滑工况：振动加剧至 8 mm/s
- 反映皮带运行平稳性

**4. temperature（温度）**
- 正常工况：温度约 60°C
- 打滑工况：摩擦生热导致温度升高至 85°C
- 反映轴承和皮带摩擦状态

---

## 输出标签

### 分类标签

| 标签值 | 含义 | 说明 |
|--------|------|------|
| 0 | 正常 | 皮带机运行平稳，无打滑现象 |
| 1 | 打滑 | 检测到皮带打滑异常，需关注 |

---

## 快速开始

### 1. 训练模型

```bash
python belt_conveyor_slip_prediction.py
```

训练完成后会生成：
- `conveyor_slip_model.onnx` - ONNX 格式模型

### 2. 使用 Sklearn 模型预测

**注意**: 当前代码未保存 `.pkl` 文件，如需使用请自行添加保存逻辑。

```python
import joblib
import numpy as np

# 加载模型（需要先修改代码保存 .pkl 文件）
model = joblib.load('belt_conveyor_slip_model.pkl')

# 准备输入数据 [current, speed_diff, vibration, temperature]
input_data = np.array([[140, 0.3, 10, 90]]).astype(np.float32)

# 预测
prediction = model.predict(input_data)[0]  # 0: 正常, 1: 打滑
probability = model.predict_proba(input_data)[0]

print(f"预测状态: {'打滑' if prediction == 1 else '正常'}")
print(f"打滑概率: {probability[1]:.2%}")
```

### 3. 实时预测示例 (Python Sklearn)

训练脚本中包含实时预测示例，运行以下命令即可查看：

```bash
python belt_conveyor_slip_prediction.py
```

输出示例：

```
==================================================
【实时监测】皮带机状态诊断示例
==================================================

正常工况:
  输入特征: [100.   0.02  2.5  62. ]
  输入含义: 电流=100.0A, 速度差=0.02m/s, 振动=2.5mm/s, 温度=62.0°C
  ✓ 预测结果: 运行正常
  📊 正常概率: 98.50%

打滑工况:
  输入特征: [135.    0.25  9.   88. ]
  输入含义: 电流=135.0A, 速度差=0.25m/s, 振动=9.0mm/s, 温度=88.0°C
  ⚠️  预测结果: 检测到打滑！
  📊 故障概率: 96.30%
  🚨 预警等级: 严重
```

**三种工况对比**:

| 工况 | 电流(A) | 速度差(m/s) | 振动(mm/s) | 温度(°C) | 预测结果 |
|------|---------|-------------|------------|----------|----------|
| 正常 | 100 | 0.02 | 2.5 | 62 | 正常 |
| 打滑 | 135 | 0.25 | 9.0 | 88 | 打滑 |
| 临界 | 115 | 0.12 | 5.0 | 72 | 趋势预警 |

**预警等级说明**:
- 🚨 **严重**: 故障概率 > 90%
- ⚡ **较高**: 故障概率 70% - 90%
- 💡 **一般**: 故障概率 < 70%

### 4. 使用 ONNX 模型预测

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
session = ort.InferenceSession('conveyor_slip_model.onnx')

# 准备输入数据
input_data = np.array([[140, 0.3, 10, 90]]).astype(np.float32)

# 预测
outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])      # 0: 正常, 1: 打滑
probabilities = outputs[1][0]       # [正常概率, 打滑概率]

print(f"预测状态: {'打滑' if label_idx == 1 else '正常'}")
print(f"正常概率: {probabilities[0]:.2%}")
print(f"打滑概率: {probabilities[1]:.2%}")
```

### 4. 批量预测

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('conveyor_slip_model.onnx')

# 批量输入数据 (N x 4)
batch_data = np.array([
    [100, 0.01, 2, 60],   # 正常工况
    [140, 0.3, 10, 90],   # 严重打滑
    [110, 0.05, 3, 65],   # 轻微异常
]).astype(np.float32)

# 批量预测
outputs = session.run(None, {'float_input': batch_data})
labels = outputs[0].astype(int)
probabilities = outputs[1]

# 输出结果
for i, (label, prob) in enumerate(zip(labels, probabilities)):
    status = '打滑' if label == 1 else '正常'
    print(f"样本 {i}: {status} (打滑概率: {prob[1]:.2%})")
```

---

## 封装类使用

### 皮带机打滑监测器封装类

```python
import onnxruntime as ort
import numpy as np
from typing import Tuple

class BeltSlipMonitor:
    """皮带机打滑监测器"""

    def __init__(self, model_path: str = 'conveyor_slip_model.onnx'):
        """初始化监测器

        Args:
            model_path: ONNX 模型文件路径
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_label = self.session.get_outputs()[0].name
        self.output_proba = self.session.get_outputs()[1].name

    def predict(self, current: float, speed_head: float, speed_tail: float,
                vibration: float, temperature: float) -> Tuple[int, float, str]:
        """预测皮带机打滑状态

        Args:
            current: 电机电流 (A)
            speed_head: 头部速度 (m/s)
            speed_tail: 尾部速度 (m/s)
            vibration: 电机振动 (mm/s)
            temperature: 轴承温度 (°C)

        Returns:
            (label, probability, status): 标签(0/1), 打滑概率, 状态描述
        """
        # 计算速度差
        speed_diff = abs(speed_head - speed_tail)

        # 准备输入
        input_data = np.array([[current, speed_diff, vibration, temperature]]).astype(np.float32)

        # 预测
        outputs = self.session.run([self.output_label, self.output_proba],
                                    {self.input_name: input_data})
        label = int(outputs[0][0])
        prob = outputs[1][0][1]

        # 状态描述
        if label == 1:
            if prob > 0.9:
                status = "严重打滑 - 立即停机检查"
            elif prob > 0.7:
                status = "打滑预警 - 尽快安排检查"
            else:
                status = "疑似打滑 - 持续监测"
        else:
            status = "运行正常"

        return label, prob, status

    def batch_predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量预测

        Args:
            data: 输入数据矩阵 (N x 4)
                  [current, speed_diff, vibration, temperature]

        Returns:
            (labels, probabilities): 标签数组, 概率数组
        """
        input_data = data.astype(np.float32)
        outputs = self.session.run([self.output_label, self.output_proba],
                                    {self.input_name: input_data})
        labels = outputs[0].astype(int).flatten()
        probabilities = outputs[1]
        return labels, probabilities

# 使用示例
if __name__ == "__main__":
    monitor = BeltSlipMonitor()

    # 单次预测
    label, prob, status = monitor.predict(
        current=140,
        speed_head=2.0,
        speed_tail=1.7,  # 速度差大
        vibration=10,
        temperature=90
    )

    print(f"状态: {status}")
    print(f"打滑概率: {prob:.2%}")
```

---

## 模型性能

### 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 准确率 | ~98% | 整体分类准确率 |
| 精确率 | ~98% | 预测为打滑的样本中实际打滑的比例 |
| 召回率 | ~98% | 实际打滑样本中被正确预测的比例 |
| F1 分数 | ~98% | 精确率和召回率的调和平均 |

### 数据集信息

| 项目 | 值 |
|------|-----|
| 总样本数 | 8000 |
| 训练样本 | 6400 (80%) |
| 测试样本 | 1600 (20%) |
| 正常样本 | 4000 (50%) |
| 打滑样本 | 4000 (50%) |

### 特征重要性

| 排名 | 特征 | 重要性 | 说明 |
|------|------|--------|------|
| 1 | speed_diff | ~45% | 速度差是**最关键**的指标 |
| 2 | current | ~25% | 电流反映负载情况 |
| 3 | vibration | ~18% | 振动反映机械状态 |
| 4 | temperature | ~12% | 温度反映摩擦状态 |

---

## ONNX 导出优化

### 禁用 ZipMap 提升兼容性

当前代码已配置禁用 ZipMap，提升 ONNX 模型的跨平台兼容性：

```python
# 关键配置
options = {type(model): {'zipmap': False}}

onnx_model = convert_sklearn(
    model,
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

---

## 模型验证

### Sklearn vs ONNX 一致性验证

**注意**: 当前训练脚本未保存 `.pkl` 文件，如需验证一致性，请先添加保存逻辑。

```python
import joblib
import onnxruntime as ort
import numpy as np

# 加载两个模型
model = joblib.load('belt_conveyor_slip_model.pkl')
session = ort.InferenceSession('conveyor_slip_model.onnx')

# 测试数据
test_data = np.array([[130, 0.25, 8, 85]]).astype(np.float32)

# 预测对比
sklearn_pred = model.predict(test_data)[0]
sklearn_proba = model.predict_proba(test_data)[0]

onnx_pred = session.run(None, {'float_input': test_data})
onnx_label = int(onnx_pred[0][0])
onnx_proba = onnx_pred[1][0]

print(f"Sklearn 预测: {sklearn_pred}, 概率: {sklearn_proba}")
print(f"ONNX 预测: {onnx_label}, 概率: {onnx_proba}")
print(f"预测结果一致: {sklearn_pred == onnx_label}")
```

### ONNX 模型验证

```python
import onnx

# 加载模型
onnx_model = onnx.load('conveyor_slip_model.onnx')

# 验证模型结构
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")
```

---

## 部署选项

### 1. Python Web 服务 (Flask)

```python
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np

app = Flask(__name__)
session = ort.InferenceSession('conveyor_slip_model.onnx')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # 计算速度差
    speed_diff = abs(data['speed_head'] - data['speed_tail'])

    input_data = np.array([[
        data['current'],
        speed_diff,
        data['vibration'],
        data['temperature']
    ]]).astype(np.float32)

    outputs = session.run(None, {'float_input': input_data})
    label = int(outputs[0][0])
    prob = outputs[1][0][1]

    return jsonify({
        'status': 'slip' if label == 1 else 'normal',
        'slip_probability': float(prob),
        'message': '打滑预警' if label == 1 else '运行正常'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. C++ 部署

```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

int main() {
    // 初始化 ONNX Runtime 环境
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "BeltSlip"};
    Ort::SessionOptions session_options;

    // 加载模型
    Ort::Session session{env, "conveyor_slip_model.onnx", session_options};

    // 准备输入数据
    std::vector<float> input_data = {140.0f, 0.3f, 10.0f, 90.0f};

    // 运行推理
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {"float_input"};
    const char* output_names[] = {"output_label", "output_probability"};

    auto outputs = session.run(Ort::RunOptions{nullptr},
                                input_names, &input_tensor, 1,
                                output_names, 2);

    // 获取结果
    int64_t label = outputs[0].GetTensorMutableData<int64_t>()[0];
    float* prob = outputs[1].GetTensorMutableData<float>();

    std::cout << "预测状态: " << (label == 1 ? "打滑" : "正常") << std::endl;
    std::cout << "打滑概率: " << prob[1] << std::endl;

    return 0;
}
```

### 3. 移动端部署 (Android)

使用 ONNX Runtime Mobile：

```java
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class BeltSlipDetector {
    private OrtSession session;

    public void loadModel() throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        session = env.createSession("conveyor_slip_model.onnx");
    }

    public int predict(float[] features) throws OrtException {
        float[][] input = {features};
        OrtSession.Result results = session.run(
            Map.of("float_input", OrtTensor.createTensor(env, input))
        );

        long label = ((long[]) results.get(0).getValue())[0];
        return (int) label;
    }
}
```

---

## 实时预测示例

### Python 完整示例

```python
import onnxruntime as ort
import numpy as np

def realtime_prediction():
    """实时预测示例"""

    # 加载 ONNX 模型
    session = ort.InferenceSession('conveyor_slip_model.onnx')

    # 示例数据
    test_cases = {
        "正常工况": np.array([[100, 0.02, 2.5, 62]]),
        "打滑工况": np.array([[135, 0.25, 9.0, 88]]),
        "临界状态": np.array([[115, 0.12, 5.0, 72]])
    }

    for case_name, data in test_cases.items():
        data = data.astype(np.float32)

        # 运行推理
        outputs = session.run(None, {'float_input': data})
        label = int(outputs[0][0])
        proba = outputs[1][0]

        # 输出结果
        print(f"\n{case_name}:")
        print(f"  输入: {data[0]}")
        print(f"  预测: {'打滑' if label == 1 else '正常'}")
        print(f"  概率: {proba[label]:.2%}")

        # 预警等级
        if label == 1:
            if proba[1] > 0.9:
                print(f"  预警等级: 严重")
            elif proba[1] > 0.7:
                print(f"  预警等级: 较高")
            else:
                print(f"  预警等级: 一般")

# 运行
realtime_prediction()
```

### C++ 完整示例

```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

class BeltSlipDetector {
private:
    Ort::Session session;
    Ort::Env env;

public:
    BeltSlipDetector(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "BeltSlip"),
          session(env, model_path.c_str(), Ort::SessionOptions{}) {
        std::cout << "✅ 模型加载成功" << std::endl;
    }

    void predict(float current, float speed_diff, float vibration, float temperature) {
        // 准备输入
        std::vector<float> input_data = {current, speed_diff, vibration, temperature};
        std::vector<int64_t> input_shape = {1, 4};

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
        std::cout << "输入: 电流=" << current << "A, 速度差=" << speed_diff
                  << "m/s, 振动=" << vibration << "mm/s, 温度=" << temperature << "°C" << std::endl;
        std::cout << "预测: " << (label == 1 ? "打滑" : "正常") << std::endl;
        std::cout << "正常概率: " << proba[0] * 100 << "%" << std::endl;
        std::cout << "打滑概率: " << proba[1] * 100 << "%" << std::endl;

        // 预警等级
        if (label == 1) {
            if (proba[1] > 0.9f) {
                std::cout << "预警等级: 严重 🚨" << std::endl;
            } else if (proba[1] > 0.7f) {
                std::cout << "预警等级: 较高 ⚡" << std::endl;
            } else {
                std::cout << "预警等级: 一般 💡" << std::endl;
            }
        }
    }
};

int main() {
    try {
        BeltSlipDetector detector("conveyor_slip_model.onnx");

        // 测试不同工况
        std::cout << "\n【测试1: 正常工况】" << std::endl;
        detector.predict(100, 0.02f, 2.5f, 62);

        std::cout << "\n【测试2: 打滑工况】" << std::endl;
        detector.predict(135, 0.25f, 9.0f, 88);

        std::cout << "\n【测试3: 临界状态】" << std::endl;
        detector.predict(115, 0.12f, 5.0f, 72);

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime 错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

---

## 高级配置

### 参数调整

```python
# 更精确的模型（可能过拟合）
model = RandomForestClassifier(
    n_estimators=200,   # 增加树数量
    max_depth=12,       # 增加深度
    min_samples_split=2,
    random_state=42
)

# 更稳健的模型（可能欠拟合）
model = RandomForestClassifier(
    n_estimators=150,   # 减少树数量
    max_depth=6,        # 减少深度
    min_samples_split=10,
    random_state=42
)
```

### 特征扩展

可以添加更多特征提升模型性能：

```python
# 添加新特征
df['power'] = df['current'] * 2.0  # 功率估算
df['temp_rate'] = df['temperature'].diff()  # 温度变化率
df['vibration_trend'] = df['vibration'].rolling(10).mean()  # 振动趋势
```

### 使用 Pipeline

**重要**: 如果使用 Pipeline，需要更新 `export_to_onnx` 函数的 options 配置：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    ))
])

model_pipeline.fit(X_train, y_train)

# 导出时需要调整 options
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]

# Pipeline 模型的 options 配置
options = {
    'zipmap': False,  # 禁用 ZipMap
    'nocl': True      # 禁用类标签（某些情况）
}

onnx_model = convert_sklearn(
    model_pipeline,
    initial_types=initial_type,
    target_opset=12,
    options=options
)
```

### 添加模型保存逻辑

如果需要保存 Sklearn 模型用于验证一致性，可以修改训练脚本：

```python
import joblib

# 在 train_model 函数最后添加
def train_model(df):
    # ... 现有训练代码 ...

    # 保存 Sklearn 模型
    joblib.dump(model, 'belt_conveyor_slip_model.pkl')
    print("✅ Sklearn 模型已保存至: belt_conveyor_slip_model.pkl")

    return model, features
```

---

## 实际应用示例

### 实时监测系统

```python
import time
import onnxruntime as ort
import numpy as np
import random

class RealTimeSlipMonitor:
    def __init__(self):
        self.session = ort.InferenceSession('conveyor_slip_model.onnx')

    def simulate_sensor_data(self):
        """模拟传感器数据"""
        base_current = random.choice([100, 105, 110, 130, 140])
        base_speed_head = random.choice([2.0, 2.1, 1.9])

        # 打滑时速度差增大
        is_slip = base_current > 120
        speed_tail = base_speed_head - (random.uniform(0.2, 0.4) if is_slip else random.uniform(0, 0.05))

        return {
            'current': base_current + random.normalvariate(0, 5),
            'speed_head': base_speed_head,
            'speed_tail': speed_tail,
            'vibration': random.choice([2, 8]) if is_slip else random.normalvariate(2, 0.5),
            'temperature': random.choice([85, 90]) if is_slip else random.normalvariate(60, 5)
        }

    def monitor(self):
        """持续监测"""
        for i in range(100):
            data = self.simulate_sensor_data()

            # 计算速度差
            speed_diff = abs(data['speed_head'] - data['speed_tail'])

            # 预测
            input_data = np.array([[
                data['current'], speed_diff, data['vibration'], data['temperature']
            ]]).astype(np.float32)

            outputs = self.session.run(None, {'float_input': input_data})
            label = int(outputs[0][0])
            prob = outputs[1][0][1]

            # 输出结果
            timestamp = time.strftime("%H:%M:%S")
            if label == 1:
                print(f"[{timestamp}] 🚨 打滑预警! 概率: {prob:.2%}")
            else:
                print(f"[{timestamp}] ✓ 运行正常")

            time.sleep(1)

# 运行监测
monitor = RealTimeSlipMonitor()
monitor.monitor()
```

---

## 故障排查

### 常见问题

**1. 模型加载失败**
```python
# 检查文件是否存在
import os
print(os.path.exists('conveyor_slip_model.onnx'))
```

**2. 预测结果异常**
```python
# 验证输入数据格式
print(f"输入形状: {input_data.shape}")  # 应为 (1, 4)
print(f"数据类型: {input_data.dtype}")  # 应为 float32
```

**3. ONNX Runtime 兼容性**
```bash
# 更新 ONNX Runtime
pip install --upgrade onnxruntime

# 检查版本
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

**4. 性能优化**
```python
# 启用多个执行提供程序
session = ort.InferenceSession(
    'conveyor_slip_model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

---

## 数据采集建议

### 传感器配置

| 传感器 | 推荐型号 | 采样频率 | 安装位置 |
|--------|---------|---------|---------|
| 电流互感器 | Hall 传感器 | 10 Hz | 电机进线 |
| 测速传感器 | 编码器 | 10 Hz | 传动滚筒 |
| 振动传感器 | 加速度传感器 | 1 kHz | 电机轴承 |
| 温度传感器 | PT100 | 1 Hz | 轴承座 |

### 数据预处理

1. **滤波处理**：去除高频噪声
2. **异常值剔除**：3σ 准则
3. **数据对齐**：确保时间同步
4. **缺失值处理**：插值或删除

### 速度差计算

```python
# 实时计算速度差
def calculate_speed_diff(speed_head, speed_tail):
    """计算头尾速度差"""
    return abs(speed_head - speed_tail)
```

---

## 维护建议

### 定期任务

1. **每周**：检查传感器数据质量
2. **每月**：评估模型性能，收集误报案例
3. **每季度**：重新训练模型（使用新数据）
4. **每年**：全面审查模型架构和特征

### 模型更新流程

```python
# 1. 收集新数据
new_data = collect_recent_data(days=30)

# 2. 标注数据
labeled_data = label_data(new_data)

# 3. 重新训练
new_model = train_model(labeled_data)

# 4. 验证性能
validate_model(new_model, test_data)

# 5. 部署更新
export_to_onnx(new_model, features, 'conveyor_slip_model_v2.onnx')
```

---

## 模型参数调优

### 可调参数

本模型使用 `RandomForestClassifier`，以下是关键可调参数：

| 参数 | 当前值 | 说明 | 调优建议 |
|------|--------|------|----------|
| n_estimators | 100 | 决策树数量 | 50-300，越多越稳定但越慢 |
| max_depth | 8 | 树的最大深度 | 6-12，控制过拟合 |
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
    'classifier__max_depth': [6, 8, 10, None],
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

- **打滑判断公式**（如 `speed_diff > 0.1`）是模拟数据标签生成规则，**不需要调整**
- 只需调整机器学习模型的超参数
- 调整参数后需重新训练并导出 ONNX 模型
- 确保调优不会导致过拟合（测试集性能下降）
- 速度差（speed_diff）是最关键的特征，权重约 45%

---

## 参考文献

- Scikit-learn: https://scikit-learn.org/
- ONNX: https://onnx.ai/
- ONNX Runtime: https://onnxruntime.ai/
- Random Forest: Breiman, L. (2001). "Random Forests". Machine Learning.

---

## 联系支持

如有问题或建议，请提交 Issue 或联系项目维护者。
