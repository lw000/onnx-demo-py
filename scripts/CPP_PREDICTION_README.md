# C++ ONNX 推理指南

## 问题描述

在 C++ 中加载 ONNX 模型进行推理时，获取不到标签对应的概率数据。

## 根本原因

### 1. 输出数量错误
**问题**: 只获取了第一个输出(标签)，没有获取第二个输出(概率)

```cpp
// ❌ 错误：只获取 1 个输出
auto output = session.Run(..., 1, output_names, 1);

// ✅ 正确：必须获取 2 个输出
auto outputs = session.Run(..., 1, output_names, 2);
```

### 2. 数据类型错误
**问题**: 用错误的数据类型读取概率数据

| 输出 | 数据类型 | C++ 类型 |
|------|----------|----------|
| label | int64 | `int64_t*` |
| probabilities | float32 | `float*` |

```cpp
// ❌ 错误：用 int64 读取 float 数据
int64_t* proba = proba_tensor.GetTensorData<int64_t>();

// ✅ 正确：用 float 读取概率数据
float* proba = proba_tensor.GetTensorMutableData<float>();
```

### 3. 输出节点名称未正确获取
**问题**: 硬编码输出节点名称可能失败

```cpp
// ❌ 错误：硬编码节点名称
const char* output_names[] = {"label", "probabilities"};

// ✅ 正确：动态获取节点名称
char* output_label_name = session.GetOutputName(0, allocator);
char* output_proba_name = session.GetOutputName(1, allocator);
```

## 模型输出结构

```
输入节点: float_input (tensor[float], [None, 4])
输出[0]: label (tensor[int64], [None])
输出[1]: probabilities (tensor[float], [None, 3])
```

- **label**: 预测的类别索引 (0=normal, 1=wear, 2=cavitation)
- **probabilities**: 各类别的概率数组 [prob_class_0, prob_class_1, prob_class_2]

## 完整解决方案

### 关键代码片段

```cpp
// 1. 获取输出节点名称
Ort::AllocatorWithDefaultOptions allocator;
char* output_label_name = session.GetOutputName(0, allocator);  // 第1个输出: label
char* output_proba_name = session.GetOutputName(1, allocator); // 第2个输出: probabilities

// 2. 运行推理 - 关键：指定 2 个输出名称
const char* input_names[] = {input_name};
const char* output_names[] = {output_label_name, output_proba_name};

auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names,
    &input_tensor,
    1,          // 输入数量
    output_names,
    2           // 输出数量 - 必须=2
);

// 3. 获取标签 (int64)
auto& label_tensor = output_tensors[0];
int64_t label_value = *label_tensor.GetTensorData<int64_t>();

// 4. 获取概率 (float32) - 关键步骤
auto& proba_tensor = output_tensors[1];
float* proba_data = proba_tensor.GetTensorMutableData<float>();

// 5. 读取概率值
int num_classes = 3;  // 3 个类别
for (int i = 0; i < num_classes; i++) {
    float prob = proba_data[i];
    std::cout << "Class " << i << ": " << prob << std::endl;
}

// 释放内存
allocator.Free(output_label_name);
allocator.Free(output_proba_name);
```

## 编译与运行

### Windows (Visual Studio)

1. 安装 ONNX Runtime NuGet 包
2. 配置项目属性：
   - 包含目录: `[ONNX Runtime 安装路径]/include`
   - 库目录: `[ONNX Runtime 安装路径]/lib`
   - 链接器输入: `onnxruntime.lib`

3. 编译并运行

### Windows (MSVC 命令行)

```bash
# 编译
cl /EHsc /std:c++17 /I"onnxruntime/include" pump_prediction_cpp.cpp /link onnxruntime.lib

# 运行
pump_prediction_cpp.exe
```

### Linux (GCC)

```bash
# 编译
g++ -std=c++17 -I/usr/include/onnxruntime pump_prediction_cpp.cpp -L/usr/lib -lonnxruntime -o pump_prediction_cpp

# 运行
./pump_prediction_cpp
```

## 输出示例

```
=== 模型加载成功 ===
输入节点: float_input
输出[0](标签): label
输出[1](概率): probabilities

=== 测试 1: 正常状态 ===
预测结果:
  标签: 0 (normal)
  置信度: 0.999997
  概率分布:
    0 (normal): 0.999997
    1 (wear): 0.000001
    2 (cavitation): 0.000001

=== 测试 2: 磨损状态 ===
预测结果:
  标签: 1 (wear)
  置信度: 0.854321
  概率分布:
    0 (normal): 0.054321
    1 (wear): 0.854321
    2 (cavitation): 0.091358

=== 测试 3: 汽蚀状态 ===
预测结果:
  标签: 2 (cavitation)
  置信度: 0.923456
  概率分布:
    0 (normal): 0.023456
    1 (wear): 0.053088
    2 (cavitation): 0.923456
```

## 常见问题排查

### 问题 1: 输出数量只有 1 个

**症状**: `output_tensors.size() == 1`

**原因**: `Run()` 方法的输出数量参数设为 1

**解决**: 将输出数量参数改为 2

```cpp
auto outputs = session.Run(..., 1, output_names, 2);  // 最后一个参数必须是 2
```

### 问题 2: 概率值全是 0 或异常值

**症状**: `proba_data[i]` 的值不正确

**原因**: 用错误的数据类型读取概率数据

**解决**: 使用 `float*` 而不是 `int64_t*`

```cpp
// ✅ 正确
float* proba_data = proba_tensor.GetTensorMutableData<float>();

// ❌ 错误
int64_t* proba_data = proba_tensor.GetTensorData<int64_t>();
```

### 问题 3: 程序崩溃

**症状**: 访问 `proba_data[i]` 时崩溃

**原因**: 概率数组形状不匹配，访问越界

**解决**: 先获取数组形状，再访问

```cpp
auto shape = proba_tensor.GetTensorTypeAndShapeInfo().GetShape();
int64_t num_classes = shape[1];  // 第 2 维是类别数量

for (int i = 0; i < num_classes; i++) {
    float prob = proba_data[i];
    // ...
}
```

### 问题 4: 输出节点名称为空

**症状**: `output_name` 为空字符串

**原因**: 硬编码节点名称与实际不匹配

**解决**: 动态获取节点名称

```cpp
char* output_label_name = session.GetOutputName(0, allocator);
char* output_proba_name = session.GetOutputName(1, allocator);

std::cout << "输出[0]名称: " << output_label_name << std::endl;
std::cout << "输出[1]名称: " << output_proba_name << std::endl;
```

## 性能优化

### 1. 复用 MemoryInfo

```cpp
class Predictor {
    Ort::MemoryInfo memory_info_;
public:
    Predictor() : memory_info_(Ort::MemoryInfo::CreateCpu(...)) {}
    // 使用 memory_info_ 创建所有张量
};
```

### 2. 批量预测

```cpp
std::vector<int64_t> batch_input_shape = {batch_size, 4};
Orrt::Value batch_input_tensor = Ort::Value::CreateTensor<float>(
    memory_info_, batch_data.data(), batch_data.size(),
    batch_input_shape.data(), batch_input_shape.size()
);

// 一次推理处理多个样本
auto outputs = session.Run(..., &batch_input_tensor, 1, ...);
```

### 3. 多线程推理

```cpp
// 每个线程使用独立的 Session 对象
std::vector<std::thread> threads;
for (int i = 0; i < num_threads; i++) {
    threads.emplace_back([&, i]() {
        Ort::Session session(env, model_path, session_options);
        // 每个线程独立推理
    });
}
```

## 调试技巧

### 1. 打印输入输出信息

```cpp
std::cout << "输入: " << input_name_ << std::endl;
std::cout << "  形状: [" << input_shape[0] << ", " << input_shape[1] << "]" << std::endl;
std::cout << "  数据: ";
for (float v : input_values) std::cout << v << " ";
std::cout << std::endl;

std::cout << "输出标签: " << label_value << std::endl;
std::cout << "输出概率: ";
for (int i = 0; i < 3; i++) std::cout << proba_data[i] << " ";
std::cout << std::endl;
```

### 2. 验证模型结构

使用 Python 脚本验证模型输出结构：

```bash
python check_model_output.py
```

### 3. 检查内存泄漏

```cpp
// 确保释放动态分配的内存
allocator.Free(input_name);
allocator.Free(output_label_name);
allocator.Free(output_proba_name);
```

## 参考资料

- [ONNX Runtime C++ API 文档](https://onnxruntime.ai/docs/api/c/struct_ort-api.html)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [ONNX 规范](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
