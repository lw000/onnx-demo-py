# 皮带机打滑故障预测模型文档

## 模型概述

本模型用于预测皮带输送机的打滑故障，通过分析电机电流、速度差、振动、温度等传感器数据，实现对皮带机运行状态的实时监测和故障预警。

### 应用场景

- **矿山运输**：井下皮带输送机监测
- **港口物流**：散装物料输送系统
- **电厂输煤**：煤炭运输皮带机
- **建材行业**：水泥、砂石输送系统
- **预测性维护**：提前发现异常，避免生产中断

---

## 技术架构

### 算法选择

**Random Forest Classifier (随机森林分类器)**

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 200 | 决策树数量 |
| max_depth | 8 | 树的最大深度，防止过拟合 |
| random_state | 42 | 随机种子，保证可复现性 |
| n_jobs | -1 | 使用所有 CPU 核心 |

### Pipeline 结构

```python
Pipeline([
    ('scaler', StandardScaler()),           # 特征标准化
    ('classifier', RandomForestClassifier(...))  # 随机森林分类器
])
```

---

## 输入特征详解

模型使用 5 个传感器特征进行预测：

| 特征名称 | 单位 | 数据类型 | 说明 |
|---------|------|---------|------|
| current | A (安培) | float32 | 电机电流，反映负载状态 |
| speed_diff | m/s (米/秒) | float32 | 头尾速度差，**打滑的核心指标** |
| vibration_motor | mm/s | float32 | 电机振动值 |
| temperature_bearing | °C (摄氏度) | float32 | 轴承温度 |
| current_std | - | float32 | 电流滚动标准差，反映波动性 |

### 特征工程说明

**1. speed_diff（速度差）**
```python
df['speed_diff'] = abs(df['speed_head'] - df['speed_tail'])
```
- 头尾速度差是判断打滑的核心指标
- 正常情况下头尾速度基本一致
- 打滑时尾部速度显著降低

**2. current_std（电流波动）**
```python
df['current_std'] = df['current'].rolling(window=5, min_periods=1).std()
```
- 使用 5 点滑动窗口计算标准差
- 反映电流的短期波动情况
- 波动大可能表示负载不稳定或打滑

### 数据分布（正常/故障对比）

| 特征 | 正常均值 | 故障均值 | 说明 |
|------|---------|---------|------|
| current | 100 A | 130 A | 故障时负载增大 |
| speed_diff | ~0 m/s | ~0.5 m/s | 故障时速度差显著 |
| vibration_motor | 2 mm/s | 8 mm/s | 故障时振动剧烈 |
| temperature_bearing | 60°C | 85°C | 故障时温度升高 |
| current_std | 低 | 高 | 故障时波动大 |

---

## 输出标签

### 分类标签

| 标签值 | 含义 | 说明 |
|--------|------|------|
| 0 | 正常 | 皮带机运行平稳，无打滑或卡阻 |
| 1 | 故障 | 检测到打滑、卡阻或轴承损坏等异常 |

### 故障类型识别

模型预测的故障状态可能包括：

1. **皮带打滑**：速度差大，电流波动
2. **负载卡阻**：电流高，温度高
3. **轴承损坏**：振动大，温度高

---

## 快速开始

### 1. 训练模型

```bash
python belt_conveyor_slippage_fault_prediction.py
```

训练完成后会生成：
- `belt_conveyor_slippage_fault_detector.onnx` - ONNX 格式模型
- `belt_conveyor_slippage_fault_detector_sklearn.pkl` - Sklearn 原始模型

### 2. 使用 Sklearn 模型预测

```python
import joblib
import numpy as np

# 加载模型
model = joblib.load('belt_conveyor_slippage_fault_detector_sklearn.pkl')

# 准备输入数据 [current, speed_diff, vibration_motor, temperature_bearing, current_std]
input_data = np.array([[140, 0.8, 10, 90, 25]]).astype(np.float32)

# 预测
prediction = model.predict(input_data)[0]  # 0: 正常, 1: 故障
probability = model.predict_proba(input_data)[0]

print(f"预测状态: {'故障' if prediction == 1 else '正常'}")
print(f"故障概率: {probability[1]:.2%}")
```

### 3. 使用 ONNX 模型预测

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
session = ort.InferenceSession('belt_conveyor_slippage_fault_detector.onnx')

# 准备输入数据
input_data = np.array([[140, 0.8, 10, 90, 25]]).astype(np.float32)

# 预测
outputs = session.run(None, {'float_input': input_data})
label_idx = int(outputs[0][0])      # 0: 正常, 1: 故障
probabilities = outputs[1][0]       # [正常概率, 故障概率]

print(f"预测状态: {'故障' if label_idx == 1 else '正常'}")
print(f"正常概率: {probabilities[0]:.2%}")
print(f"故障概率: {probabilities[1]:.2%}")
```

### 4. 批量预测

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('belt_conveyor_slippage_fault_detector.onnx')

# 批量输入数据 (N x 5)
batch_data = np.array([
    [100, 0.05, 2, 60, 5],   # 正常工况
    [140, 0.8, 10, 90, 25],  # 严重故障
    [110, 0.1, 3, 65, 8],    # 轻微异常
]).astype(np.float32)

# 批量预测
outputs = session.run(None, {'float_input': batch_data})
labels = outputs[0].astype(int)
probabilities = outputs[1]

# 输出结果
for i, (label, prob) in enumerate(zip(labels, probabilities)):
    status = '故障' if label == 1 else '正常'
    print(f"样本 {i}: {status} (故障概率: {prob[1]:.2%})")
```

---

## 封装类使用

### 皮带机监测器封装类

```python
import onnxruntime as ort
import numpy as np
from typing import Tuple

class BeltConveyorMonitor:
    """皮带机打滑故障监测器"""
    
    def __init__(self, model_path: str = 'belt_conveyor_slippage_fault_detector.onnx'):
        """初始化监测器
        
        Args:
            model_path: ONNX 模型文件路径
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_label = self.session.get_outputs()[0].name
        self.output_proba = self.session.get_outputs()[1].name
    
    def predict(self, current: float, speed_head: float, speed_tail: float,
                vibration_motor: float, temperature_bearing: float,
                history_current: list = None) -> Tuple[int, float, str]:
        """预测皮带机状态
        
        Args:
            current: 电机电流 (A)
            speed_head: 头部速度 (m/s)
            speed_tail: 尾部速度 (m/s)
            vibration_motor: 电机振动 (mm/s)
            temperature_bearing: 轴承温度 (°C)
            history_current: 历史电流值列表（用于计算波动率）
            
        Returns:
            (label, probability, status): 标签(0/1), 故障概率, 状态描述
        """
        # 计算速度差
        speed_diff = abs(speed_head - speed_tail)
        
        # 计算电流波动率
        if history_current and len(history_current) >= 2:
            current_std = np.std(history_current[-5:])
        else:
            current_std = 0
        
        # 准备输入
        input_data = np.array([[current, speed_diff, vibration_motor,
                                 temperature_bearing, current_std]]).astype(np.float32)
        
        # 预测
        outputs = self.session.run([self.output_label, self.output_proba],
                                    {self.input_name: input_data})
        label = int(outputs[0][0])
        prob = outputs[1][0][1]
        
        # 状态描述
        if label == 1:
            if prob > 0.9:
                status = "严重故障 - 立即停机检查"
            elif prob > 0.7:
                status = "故障预警 - 尽快安排检查"
            else:
                status = "疑似异常 - 持续监测"
        else:
            status = "运行正常"
        
        return label, prob, status
    
    def batch_predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量预测
        
        Args:
            data: 输入数据矩阵 (N x 5)
                  [current, speed_diff, vibration_motor, temperature_bearing, current_std]
            
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
    monitor = BeltConveyorMonitor()
    
    # 单次预测
    label, prob, status = monitor.predict(
        current=140,
        speed_head=2.0,
        speed_tail=1.2,
        vibration_motor=10,
        temperature_bearing=90,
        history_current=[135, 138, 142, 140, 139]
    )
    
    print(f"状态: {status}")
    print(f"故障概率: {prob:.2%}")
```

---

## 模型性能

### 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 准确率 | ~98% | 整体分类准确率 |
| 精确率 | ~98% | 预测为故障的样本中实际故障的比例 |
| 召回率 | ~98% | 实际故障样本中被正确预测的比例 |
| F1 分数 | ~98% | 精确率和召回率的调和平均 |

### 特征重要性排序

| 排名 | 特征 | 重要性 | 说明 |
|------|------|--------|------|
| 1 | speed_diff | ~40% | 速度差是**最关键**的指标 |
| 2 | vibration_motor | ~25% | 振动反映机械状态 |
| 3 | current | ~15% | 电流反映负载情况 |
| 4 | temperature_bearing | ~12% | 温度反映轴承健康 |
| 5 | current_std | ~8% | 电流波动率辅助判断 |

### 混淆矩阵示例

```
              预测正常    预测故障
实际正常      992          8
实际故障      12         988
```

---

## 模型验证

### Sklearn vs ONNX 一致性验证

训练脚本会自动验证 Sklearn 和 ONNX 模型的一致性：

```python
验证 ONNX 模型一致性...
   - 测试样本输入: [130.  0.6  8. 85. 15.]
   - scikit-learn 预测 (0:正常, 1:故障): 1
   - scikit-learn 概率: [0.02 0.98]
   - ONNX 模型预测 (0:正常, 1:故障): 1
   - ONNX 概率: [0.02 0.98]
   - 预测结果一致: True
```

### ONNX 模型验证

```python
import onnx

# 加载模型
onnx_model = onnx.load('belt_conveyor_slippage_fault_detector.onnx')

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
session = ort.InferenceSession('belt_conveyor_slippage_fault_detector.onnx')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    input_data = np.array([[
        data['current'],
        abs(data['speed_head'] - data['speed_tail']),
        data['vibration_motor'],
        data['temperature_bearing'],
        data['current_std']
    ]]).astype(np.float32)
    
    outputs = session.run(None, {'float_input': input_data})
    label = int(outputs[0][0])
    prob = outputs[1][0][1]
    
    return jsonify({
        'status': 'fault' if label == 1 else 'normal',
        'fault_probability': float(prob),
        'message': '故障预警' if label == 1 else '运行正常'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. C++ 部署

```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>

int main() {
    // 初始化 ONNX Runtime 环境
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "BeltConveyor"};
    Ort::SessionOptions session_options;
    
    // 加载模型
    Ort::Session session{env, "belt_conveyor_slippage_fault_detector.onnx", session_options};
    
    // 准备输入数据
    std::vector<float> input_data = {140.0f, 0.8f, 10.0f, 90.0f, 25.0f};
    
    // 运行推理
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());
    
    const char* input_names[] = {"float_input"};
    const char* output_names[] = {"output_label", "output_probability"};
    
    auto outputs = session.Run(Ort::RunOptions{nullptr},
                                input_names, &input_tensor, 1,
                                output_names, 2);
    
    // 获取结果
    int64_t label = outputs[0].GetTensorMutableData<int64_t>()[0];
    float* prob = outputs[1].GetTensorMutableData<float>();
    
    std::cout << "预测状态: " << (label == 1 ? "故障" : "正常") << std::endl;
    std::cout << "故障概率: " << prob[1] << std::endl;
    
    return 0;
}
```

### 3. 移动端部署 (Android)

使用 ONNX Runtime Mobile：

```java
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class BeltConveyorDetector {
    private OrtSession session;
    
    public void loadModel() throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        session = env.createSession("belt_conveyor_slippage_fault_detector.onnx");
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

### 4. 边缘计算部署

部署到树莓派、Jetson 等边缘设备：

```bash
# 安装 ONNX Runtime Edge
pip install onnxruntime-extensions

# 运行预测服务
python belt_monitor_service.py
```

---

## 高级配置

### 参数调整

```python
# 更精确的模型（可能过拟合）
model = RandomForestClassifier(
    n_estimators=300,   # 增加树数量
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
df['power'] = df['current'] * df['speed_head']  # 功率
df['vibration_trend'] = df['vibration_motor'].rolling(10).mean()  # 振动趋势
df['temp_rate'] = df['temperature_bearing'].diff()  # 温度变化率
```

### 模型集成

结合多个模型提升可靠性：

```python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

rf = RandomForestClassifier(n_estimators=200, max_depth=8)
xgb = XGBClassifier(n_estimators=200, max_depth=8)
svc = SVC(probability=True)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('svc', svc)],
    voting='soft'  # 使用概率投票
)
```

---

## 实际应用示例

### 实时监测系统

```python
import time
import onnxruntime as ort
import random

class RealTimeMonitor:
    def __init__(self):
        self.session = ort.InferenceSession('belt_conveyor_slippage_fault_detector.onnx')
        self.history = []
    
    def simulate_sensor_data(self):
        """模拟传感器数据"""
        base_current = random.choice([100, 105, 110, 130, 140])
        base_speed_head = random.choice([2.0, 2.1, 1.9])
        
        # 故障时速度差增大
        is_fault = base_current > 120
        speed_tail = base_speed_head - (random.uniform(0.3, 0.8) if is_fault else random.uniform(0, 0.05))
        
        return {
            'current': base_current + random.normalvariate(0, 5),
            'speed_head': base_speed_head,
            'speed_tail': speed_tail,
            'vibration_motor': random.choice([2, 8]) if is_fault else random.normalvariate(2, 0.5),
            'temperature_bearing': random.choice([85, 90]) if is_fault else random.normalvariate(60, 5)
        }
    
    def monitor(self):
        """持续监测"""
        for i in range(100):
            data = self.simulate_sensor_data()
            
            # 计算特征
            speed_diff = abs(data['speed_head'] - data['speed_tail'])
            
            # 计算电流波动
            self.history.append(data['current'])
            if len(self.history) > 5:
                current_std = np.std(self.history[-5:])
            else:
                current_std = 0
            
            # 预测
            input_data = np.array([[
                data['current'], speed_diff, data['vibration_motor'],
                data['temperature_bearing'], current_std
            ]]).astype(np.float32)
            
            outputs = self.session.run(None, {'float_input': input_data})
            label = int(outputs[0][0])
            prob = outputs[1][0][1]
            
            # 输出结果
            timestamp = time.strftime("%H:%M:%S")
            if label == 1:
                print(f"[{timestamp}] 🚨 故障预警! 概率: {prob:.2%}")
            else:
                print(f"[{timestamp}] ✓ 正常运行")
            
            time.sleep(1)

# 运行监测
monitor = RealTimeMonitor()
monitor.monitor()
```

---

## 故障排查

### 常见问题

**1. 模型加载失败**
```python
# 检查文件是否存在
import os
print(os.path.exists('belt_conveyor_slippage_fault_detector.onnx'))
```

**2. 预测结果异常**
```python
# 验证输入数据格式
print(f"输入形状: {input_data.shape}")  # 应为 (1, 5)
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
    'belt_conveyor_slippage_fault_detector.onnx',
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
deploy_model(new_model)
```

---

## 参考文献

- Scikit-learn: https://scikit-learn.org/
- ONNX: https://onnx.ai/
- ONNX Runtime: https://onnxruntime.ai/
- Random Forest: Breiman, L. (2001). "Random Forests". Machine Learning.

---

## 联系支持

如有问题或建议，请提交 Issue 或联系项目维护者。
