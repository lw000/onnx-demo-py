# create_simple_model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 假设我们的模型很简单：预测温度 = 0.5 * 当前温度 + 0.3 * 振动X + 0.2 * 电流 + 10
# 我们创建一些虚拟数据来训练它
X = np.random.rand(100, 3).astype(np.float32) # 3个特征: temp, vibration_x, current
y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + 10 + np.random.normal(0, 0.01, size=X.shape[0])

model = LinearRegression()
model.fit(X, y)

# 转换为 ONNX
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# 保存模型
with open("simple_temp_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model saved as simple_temp_model.onnx")