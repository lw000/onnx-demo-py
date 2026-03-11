# ONNX 模型输出检查工具

## 工具概述

用于检查和验证 ONNX 模型的输入输出结构，帮助开发者了解模型接口信息，便于 C++ 部署时获取正确的节点名称。

## 功能说明

- 查看模型输入节点信息
- 查看模型输出节点信息
- 测试推理功能
- 输出 C++ 调用所需的节点名称

## 使用方法

```bash
python scripts/check_model_output.py
```

## 输出示例

```
=== 检查 ONNX 模型输出结构 ===

1. 模型图输入节点:
   - 名称: float_input
   - 类型: tensor_type(float)

2. 模型图输出节点:
   - 名称: output_label
   - 类型: tensor_type(int64)

3. ONNX Runtime 推理测试:

   输入信息:
   - 名称: float_input
   - 形状: [None, 4]
   - 类型: tensor(float)

   输出信息:
   - 名称: output_label
   - 形状: [None]
   - 类型: tensor(int64)

4. 执行预测:
   - 输出数量: 2
   - 输出[0]: 标签 (int64)
   - 输出[1]: 概率数组 (float32)

5. C++ 需要使用的输出节点名称:
   - 输出[0] 名称: 'output_label'
   - 输出[1] 名称: 'output_probability'
```

## 关键提示

1. **确保获取正确的输出数量**：分类模型通常有 2 个输出（标签和概率）
2. **概率数据类型**：概率数组是 float32 类型，不是 int64
3. **C++ 调用**：必须传入正确的输出节点名称

## 文件位置

| 文件 | 路径 |
|------|------|
| 脚本 | `scripts/check_model_output.py` |
