# 变频器健康预测模型

基于多输出随机森林的变频器健康监测系统，支持 ONNX 格式跨平台部署。

## 模型概述

变频器健康预测模型是一个多输出回归模型，用于预测变频器的两个关键健康指标：
1. **电容寿命百分比** (0-100%) - 基于电容纹波数据预测剩余寿命
2. **温升异常概率** (0-1) - 基于温度数据检测散热故障

模型通过分析 6 个运行特征（平均纹波、纹波波动、平均温度、温升趋势、平均负载、温度波动），实现同时预测两个目标值。

## 技术栈

- **Python**: 3.14+
- **机器学习库**: scikit-learn
- **模型转换**: skl2onnx, onnx
- **数据处理**: numpy, pandas
- **模型推理**: onnxruntime

## 安装依赖

```bash
pip install numpy scikit-learn onnx skl2onnx onnxruntime pandas
```

## 文件说明

```
变频器健康预测模型/
├── train_inverter_prediction.py              # 模型训练脚本
├── inverter_health_multi.onnx                # ONNX 格式模型
└── INVERTER_MODEL_README.md                   # 本文档
```

## 特征说明

| 特征名称 | 单位 | 范围 | 说明 |
|---------|------|------|------|
| mean_ripple | V | 2-17 | 平均纹波 RMS，电容老化关键指标 |
| std_ripple | V | 0-5 | 纹波波动，反映纹波稳定性 |
| mean_temp | °C | 60-100 | 平均温度 |
| temp_rise | °C | 0-30 | 窗口内温差，温升趋势指标 |
| mean_load | - | 0.3-1.0 | 平均负载率 |
| temp_range | °C | 0-20 | 温度波动范围 |

**特征计算方法**（滑动窗口，window_size=20）：

```python
# 在时间窗口内计算统计特征
features = [
    np.mean(ripple_window),           # 0: 平均纹波
    np.std(ripple_window),            # 1: 纹波标准差
    np.mean(temp_window),             # 2: 平均温度
    temp_window[-1] - temp_window[0], # 3: 温升趋势
    np.mean(load_window),             # 4: 平均负载
    np.max(temp_window) - np.min(temp_window) # 5: 温度范围
]
```

## 预测输出

| 输出名称 | 范围 | 类型 | 说明 |
|---------|------|------|------|
| life_pct | 0-100 | float32 | 电容剩余寿命百分比 |
| thermal_risk | 0-1 | float32 | 温升异常概率 |

**输出含义**：

1. **电容寿命百分比**
   - `>80%`: 健康，电容状态良好
   - `50-80%`: 警告，开始老化
   - `<50%`: 严重老化，需要更换

2. **温升异常概率**
   - `<0.3`: 正常，散热良好
   - `0.3-0.7`: 警告，温升偏高
   - `>0.7`: 异常，散热故障风险高

## 模型架构

### 处理流水线

```python
MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=100,    # 100 棵决策树
        max_depth=10,       # 最大深度 10
        random_state=42
    )
)
```

### 模型特点

- **多输出回归**: 同时预测 2 个目标值
- **集成学习**: 100 棵决策树的随机森林
- **固定批次**: Batch Size = 1，适合实时推理
- **张量模式**: ONNX 输入形状 [1, 6]，输出 [1, 2]

## 使用方法

### 1. 训练模型

```bash
python train_inverter_prediction.py
```

训练过程输出：
```
数据集形状: X=(4980, 6), Y=(4980, 2)

模型评估:
- 电容寿命预测 MAE: 6.45%
- 温升异常检测准确率: 100.00%

模型已导出 (张量模式 [1, 6]): inverter_health_multi.onnx

=== ONNX 模型信息 ===
输入节点: float_input, 形状: [1, 6], 类型: tensor(float)
输出节点[0]: variable, 形状: [1, 2], 类型: tensor(float)

=== ONNX 推理测试 ===
测试输入: [2.18 0.41 72.43 2.86 0.62 14.02]
ONNX 预测结果 - [寿命%, 热故障]: [96.15  0.00]

=== Sklearn vs ONNX 对比 ===
Sklearn 预测: [96.15  0.00]
ONNX 预测:   [96.15  0.00]
一致: True
```

### 2. 使用 Sklearn 模型预测

```python
import numpy as np
import joblib

# 假设已保存模型
# model = joblib.load('inverter_health_model_sklearn.pkl')

# 或直接使用训练好的模型
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# 加载或创建模型（这里需要重新训练）
# model = ...

# 准备输入数据 (6 个特征)
input_data = np.array([[
    2.18,    # mean_ripple (平均纹波)
    0.41,    # std_ripple (纹波波动)
    72.43,   # mean_temp (平均温度)
    2.86,    # temp_rise (温升趋势)
    0.62,    # mean_load (平均负载)
    14.02    # temp_range (温度范围)
]], dtype=np.float32)

# 预测
prediction = model.predict(input_data)[0]
life_pct = prediction[0]
thermal_risk = prediction[1]

print(f"电容寿命: {life_pct:.2f}%")
print(f"温升异常概率: {thermal_risk:.2%}")

# 健康状态评估
if life_pct > 80 and thermal_risk < 0.3:
    print("状态: 健康")
elif life_pct > 50 or thermal_risk < 0.7:
    print("状态: 警告")
else:
    print("状态: 异常")
```

### 3. 使用 ONNX 模型预测

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
session = ort.InferenceSession('inverter_health_multi.onnx')
input_name = session.get_inputs()[0].name

# 准备输入数据 (注意形状必须是 [1, 6])
input_data = np.array([[
    2.18,    # mean_ripple (平均纹波)
    0.41,    # std_ripple (纹波波动)
    72.43,   # mean_temp (平均温度)
    2.86,    # temp_rise (温升趋势)
    0.62,    # mean_load (平均负载)
    14.02    # temp_range (温度范围)
]], dtype=np.float32)

# 预测
outputs = session.run(None, {input_name: input_data})
prediction = outputs[0][0]

life_pct = prediction[0]
thermal_risk = prediction[1]

print(f"电容寿命: {life_pct:.2f}%")
print(f"温升异常概率: {thermal_risk:.2%}")

# 健康状态评估
if life_pct > 80 and thermal_risk < 0.3:
    print("状态: 健康")
elif life_pct > 50 or thermal_risk < 0.7:
    print("状态: 警告")
else:
    print("状态: 异常")
```

### 4. 实时监测类封装

```python
import onnxruntime as ort
import numpy as np
from collections import deque

class InverterHealthMonitor:
    def __init__(self, model_path='inverter_health_multi.onnx', window_size=20):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.window_size = window_size

        # 初始化滑动窗口
        self.temp_window = deque(maxlen=window_size)
        self.ripple_window = deque(maxlen=window_size)
        self.load_window = deque(maxlen=window_size)

    def add_sensor_data(self, temp, ripple, load):
        """
        添加传感器数据

        参数:
            temp: 温度 (°C)
            ripple: 纹波 (V)
            load: 负载率 (0-1)
        """
        self.temp_window.append(temp)
        self.ripple_window.append(ripple)
        self.load_window.append(load)

    def predict(self):
        """
        预测健康状态

        返回:
            dict: {
                'life_pct': 电容寿命,
                'thermal_risk': 温升异常概率,
                'health_status': 健康状态
            }
        """
        if len(self.temp_window) < self.window_size:
            return {'error': '数据不足，需要更多采样'}

        # 计算特征
        features = [
            np.mean(self.ripple_window),
            np.std(self.ripple_window),
            np.mean(self.temp_window),
            self.temp_window[-1] - self.temp_window[0],
            np.mean(self.load_window),
            np.max(self.temp_window) - np.min(self.temp_window)
        ]

        # 预测
        input_data = np.array([features], dtype=np.float32)
        prediction = self.session.run(None, {self.input_name: input_data})[0][0]

        life_pct = prediction[0]
        thermal_risk = prediction[1]

        # 评估健康状态
        if life_pct > 80 and thermal_risk < 0.3:
            health_status = "健康"
        elif life_pct > 50 or thermal_risk < 0.7:
            health_status = "警告"
        else:
            health_status = "异常"

        return {
            'life_pct': life_pct,
            'thermal_risk': thermal_risk,
            'health_status': health_status
        }

    def get_risk_recommendation(self, result):
        """获取风险建议"""
        life_pct = result['life_pct']
        thermal_risk = result['thermal_risk']

        recommendations = []

        if life_pct < 80:
            recommendations.append("电容寿命低于 80%，建议检查电容状态")
        if life_pct < 50:
            recommendations.append("电容寿命严重不足，建议准备更换电容")

        if thermal_risk > 0.3:
            recommendations.append("温升异常，检查散热系统")
        if thermal_risk > 0.7:
            recommendations.append("散热故障风险高，建议立即停机检查")

        if not recommendations:
            recommendations.append("设备运行正常，保持监测")

        return recommendations

# 使用示例
monitor = InverterHealthMonitor()

# 模拟实时数据流
import time
import random

for i in range(30):
    # 模拟传感器数据
    temp = 60 + random.uniform(0, 10)
    ripple = 2.0 + random.uniform(0, 1)
    load = random.uniform(0.3, 0.8)

    monitor.add_sensor_data(temp, ripple, load)

    # 每 5 个采样预测一次
    if len(monitor.temp_window) == monitor.window_size:
        result = monitor.predict()
        if 'error' not in result:
            print(f"采样 {i}: 寿命={result['life_pct']:.1f}%, "
                  f"温升风险={result['thermal_risk']:.2%}, "
                  f"状态={result['health_status']}")

            # 获取建议
            if result['health_status'] != "健康":
                for rec in monitor.get_risk_recommendation(result):
                    print(f"  - {rec}")

        time.sleep(0.1)
```

### 5. 批量预测

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession('inverter_health_multi.onnx')
input_name = session.get_inputs()[0].name

# 批量数据
batch_data = np.random.rand(10, 6).astype(np.float32)

# 逐条预测（模型固定 batch_size=1）
predictions = []
for i in range(len(batch_data)):
    single_input = batch_data[i:i+1]
    pred = session.run(None, {input_name: single_input})[0][0]
    predictions.append(pred)

predictions = np.array(predictions)

print(f"批量预测 {len(batch_data)} 个样本")
print(f"平均电容寿命: {np.mean(predictions[:, 0]):.2f}%")
print(f"平均温升风险: {np.mean(predictions[:, 1]):.2%}")
```

## C++ 部署

### C++ 推理代码

```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

class InverterHealthPredictor {
private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    std::string input_name_;
    std::string output_name_;

public:
    InverterHealthPredictor(const std::wstring& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "test")
        , session_(env_, model_path.c_str(), Ort::SessionOptions{nullptr})
        , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        // 获取输入输出名称
        Ort::AllocatorWithDefaultOptions allocator;
        char* input_name = session_.GetInputName(0, allocator);
        char* output_name = session_.GetOutputName(0, allocator);

        input_name_ = input_name;
        output_name_ = output_name;

        std::cout << "输入节点: " << input_name_ << std::endl;
        std::cout << "输出节点: " << output_name_ << std::endl;

        allocator.Free(input_name);
        allocator.Free(output_name);
    }

    struct PredictionResult {
        float life_pct;      // 电容寿命
        float thermal_risk;  // 温升异常概率
    };

    PredictionResult predict(float mean_ripple, float std_ripple, float mean_temp,
                            float temp_rise, float mean_load, float temp_range) {
        // 准备输入数据
        std::vector<float> input_values = {
            mean_ripple, std_ripple, mean_temp,
            temp_rise, mean_load, temp_range
        };
        std::vector<int64_t> input_shape = {1, 6};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            input_values.data(),
            input_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        // 运行推理
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // 获取输出
        auto& output_tensor = output_tensors[0];
        float* output_data = output_tensor.GetTensorMutableData<float>();

        PredictionResult result;
        result.life_pct = output_data[0];
        result.thermal_risk = output_data[1];

        return result;
    }

    void printResult(const PredictionResult& result) {
        std::cout << "预测结果:" << std::endl;
        std::cout << "  电容寿命: " << result.life_pct << "%" << std::endl;
        std::cout << "  温升风险: " << result.thermal_risk << std::endl;

        if (result.life_pct > 80 && result.thermal_risk < 0.3) {
            std::cout << "  状态: 健康" << std::endl;
        } else if (result.life_pct > 50 || result.thermal_risk < 0.7) {
            std::cout << "  状态: 警告" << std::endl;
        } else {
            std::cout << "  状态: 异常" << std::endl;
        }
    }
};

int main() {
    try {
        InverterHealthPredictor predictor(L"inverter_health_multi.onnx");

        // 测试预测
        auto result = predictor.predict(
            2.18f,   // mean_ripple
            0.41f,   // std_ripple
            72.43f,  // mean_temp
            2.86f,   // temp_rise
            0.62f,   // mean_load
            14.02f   // temp_range
        );

        predictor.printResult(result);

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

## 数据生成逻辑

### 模拟数据特征

模型生成 5000 个时间步的模拟数据，窗口大小 20：

```python
# 正常状态参数
base_temp = 60.0
base_ripple = 2.0  # 正常纹波 RMS

# 电容老化过程
aging_factor = min(1.0, i / (n_samples * 0.8))
ripple = base_ripple + (aging_factor * 15.0 * load_factor) + noise

# 温升故障
if i > n_samples * 0.7:
    thermal_fault = (i - n_samples * 0.7) * 0.5
temp = base_temp + (load_factor * 20) + thermal_fault + noise
```

## 模型性能

### 评估指标

| 指标 | 值 |
|------|-----|
| 电容寿命 MAE | ~6.45% |
| 温升异常准确率 | ~100% |
| 训练样本 | 4980 |
| 特征数量 | 6 |
| 模型大小 | ONNX 格式 |

### 性能特点

- **电容寿命预测**: 平均绝对误差约 6.45%
- **温升异常检测**: 准确率接近 100%
- **实时推理**: 单次推理时间 < 1ms（CPU）

## 模型验证

### ONNX 模型验证

```python
import onnx

# 验证 ONNX 模型结构
onnx_model = onnx.load('inverter_health_multi.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX 模型验证通过")

# 查看模型信息
print(onnx.helper.printable_graph(onnx_model.graph))
```

### 对比验证 (Sklearn vs ONNX)

```python
# 测试 10 个样本的一致性
batch_size = 10
batch_input = X[:batch_size]

sklearn_pred = model.predict(batch_input)
onnx_pred_list = []

for i in range(batch_size):
    single_input = batch_input[i:i+1]
    single_pred = session.run(None, {input_name: single_input})[0][0]
    onnx_pred_list.append(single_pred)

onnx_pred = np.array(onnx_pred_list)

# 计算差异
diff = np.abs(sklearn_pred - onnx_pred)
print(f"平均差异: {np.mean(diff):.6f}")
print(f"最大差异: {np.max(diff):.6f}")
```

## 常见问题

**Q: 模型固定 batch_size=1，如何批量预测？**

```python
# 逐条推理
predictions = []
for i in range(batch_size):
    single_input = batch_data[i:i+1]
    pred = session.run(None, {input_name: single_input})[0][0]
    predictions.append(pred)
```

**Q: 特征计算必须使用滑动窗口吗？**

是的，C++ 端必须完全复现 Python 的特征计算逻辑：
```python
features = [
    np.mean(ripple_window),
    np.std(ripple_window),
    np.mean(temp_window),
    temp_window[-1] - temp_window[0],
    np.mean(load_window),
    np.max(temp_window) - np.min(temp_window)
]
```

**Q: 如何判断设备是否需要维护？**

```python
def get_maintenance_action(life_pct, thermal_risk):
    if life_pct < 50:
        return "立即更换电容"
    elif thermal_risk > 0.7:
        return "立即检查散热系统"
    elif life_pct < 80 or thermal_risk > 0.3:
        return "安排预防性维护"
    else:
        return "正常监测"
```

**Q: 预测结果如何解读？**

- **电容寿命**: 
  - `>80%`: 健康，无需干预
  - `50-80%`: 老化，关注
  - `<50%`: 严重老化，需要更换

- **温升异常概率**:
  - `<0.3`: 正常
  - `0.3-0.7`: 警告
  - `>0.7`: 异常

## 扩展建议

1. **增加特征**
   - 添加更多传感器数据（电压、电流、频率等）
   - 引入时域频域特征

2. **模型优化**
   - 尝试其他算法（XGBoost、LightGBM）
   - 调整超参数

3. **实时部署**
   - 集成到 SCADA 系统
   - 实现 Web 监控界面

4. **预测性维护**
   - 结合历史数据预测剩余使用寿命
   - 优化维护计划

## 与其他模型的对比

| 特性 | 变频器预测 | 泵故障预测 |
|------|-----------|-----------|
| 模型类型 | 多输出回归 | 分类 |
| 输出 | [寿命%, 风险概率] | 类别索引 |
| 特征数量 | 6 | 4 |
| 样本数量 | 4980 | 10000 |
| Batch Size | 固定 1 | 动态 |
| 应用场景 | 健康监测 | 故障诊断 |
| 输出示例 | [96.15, 0.00] | 0 |

## 模型参数调优

### 可调参数

本模型使用 `RandomForestRegressor`，以下是关键可调参数：

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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error

# 定义参数网格
param_grid = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_depth': [8, 10, 12, None],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4]
}

# 创建模型
base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
model = MultiOutputRegressor(base_model)

# 自定义评分函数（针对电容寿命预测）
def life_score(y_true, y_pred):
    return -mean_absolute_error(y_true[:, 0], y_pred[:, 0])

# 网格搜索
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring={'life': make_scorer(life_score)},
    refit='life',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {-grid_search.best_score_:.4f}")

# 使用最佳参数
best_model = grid_search.best_estimator_
```

### 调优步骤

1. **基准测试**: 使用当前参数训练，记录性能指标
2. **网格搜索**: 使用 `GridSearchCV` 寻找最优参数组合
3. **交叉验证**: 5折交叉验证确保稳定性
4. **性能对比**: 对比调优前后的 MAE 和准确率
5. **部署验证**: 重新导出 ONNX 模型并验证一致性

### 注意事项

- **数据生成公式系数**（如 `base_temp=60`, `base_ripple=2`）是模拟数据参数，**不需要调整**
- 只需调整机器学习模型的超参数
- 调整参数后需重新训练并导出 ONNX 模型
- 确保调优不会导致过拟合（测试集性能下降）

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
