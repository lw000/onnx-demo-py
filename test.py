import onnx

model = onnx.load("models/train_igbt_model.onnx")
params = 0
for initializer in model.graph.initializer:
    # 计算每个张量的元素个数并累加
    params += onnx.numpy_helper.to_array(initializer).size
print(f"参数量: {params:,}")