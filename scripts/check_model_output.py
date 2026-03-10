import onnx
import numpy as np
import onnxruntime as rt

print("=== 检查 ONNX 模型输出结构 ===\n")

# 1. 加载模型并查看输入输出信息
model = onnx.load('pump_failure_classifier.onnx')

print("1. 模型图输入节点:")
for inp in model.graph.input:
    print(f"   - 名称: {inp.name}")
    print(f"   - 类型: {inp.type}")
    print()

print("2. 模型图输出节点:")
for out in model.graph.output:
    print(f"   - 名称: {out.name}")
    print(f"   - 类型: {out.type}")
    print()

# 2. 使用 ONNX Runtime 测试推理
print("3. ONNX Runtime 推理测试:")
session = rt.InferenceSession('pump_failure_classifier.onnx')

print("\n   输入信息:")
for inp in session.get_inputs():
    print(f"   - 名称: {inp.name}")
    print(f"   - 形状: {inp.shape}")
    print(f"   - 类型: {inp.type}")

print("\n   输出信息:")
for out in session.get_outputs():
    print(f"   - 名称: {out.name}")
    print(f"   - 形状: {out.shape}")
    print(f"   - 类型: {out.type}")

# 3. 进行一次预测
print("\n4. 执行预测:")
test_input = np.array([[110.0, 55.0, 50.0, 1.2]], dtype=np.float32)
outputs = session.run(None, {'float_input': test_input})

print(f"\n   输出数量: {len(outputs)}")
for i, output in enumerate(outputs):
    print(f"\n   输出 [{i}]:")
    print(f"   - 数据类型: {type(output)}")
    print(f"   - 形状: {output.shape if hasattr(output, 'shape') else 'N/A'}")
    print(f"   - 值: {output}")

# 4. 获取输出节点名称
print("\n5. C++ 需要使用的输出节点名称:")
for i, out in enumerate(session.get_outputs()):
    print(f"   - 输出[{i}] 名称: '{out.name}'")

print("\n" + "="*60)
print("C++ 获取概率数据的关键点:")
print("="*60)
print("""
1. 确保获取 2 个输出：
   - 输出[0]: 标签 (int64)
   - 输出[1]: 概率数组 (float32)

2. 概率数据是 float32 类型，不是 int64

3. 概率数组形状为 [1, num_classes]

4. C++ 代码中必须传入 2 个输出节点名称
""")
