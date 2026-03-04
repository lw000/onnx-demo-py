# 复杂温度预测模型
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

# 设置随机种子保证可复现性
np.random.seed(42)

# 生成更真实的工业数据 (5000 个样本)
n_samples = 5000

# 基础特征
temp_current = np.random.uniform(20, 80, n_samples)  # 当前温度 (20-80°C)
vibration_x = np.random.uniform(0, 10, n_samples)    # X轴振动
vibration_y = np.random.uniform(0, 10, n_samples)    # Y轴振动
vibration_z = np.random.uniform(0, 10, n_samples)    # Z轴振动
current = np.random.uniform(1, 15, n_samples)        # 电流 (A)
voltage = np.random.uniform(220, 240, n_samples)     # 电压 (V)
pressure = np.random.uniform(90, 110, n_samples)     # 气压
humidity = np.random.uniform(30, 70, n_samples)      # 湿度

# 构建特征矩阵
X = np.column_stack([
    temp_current, vibration_x, vibration_y, vibration_z,
    current, voltage, pressure, humidity
]).astype(np.float32)

# 构建更复杂的温度预测目标 (非线性关系 + 交互效应)
# 温度受多种因素影响，包含非线性关系和特征交互
y = (
    0.6 * temp_current +                          # 当前温度影响最大
    0.25 * (vibration_x * vibration_y) ** 0.5 +   # 振动交互效应
    0.15 * current ** 0.8 +                        # 电流的非线性影响
    0.08 * (voltage - 230) * 0.5 +                 # 电压偏差影响
    0.05 * np.sin(pressure * 0.1) +               # 气压周期性影响
    0.03 * (humidity - 50) ** 2 / 100 +            # 湿度二次影响
    0.02 * vibration_x * current / 10 +           # 振动与电流的交互
    15                                             # 基础偏移
) + np.random.normal(0, 0.5, size=n_samples)      # 添加噪声

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 构建复杂的模型流水线
# 1. 标准化特征
# 2. 添加多项式特征 (捕获非线性关系)
# 3. 使用随机森林回归器
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

# 训练模型
print("开始训练模型...")
pipeline.fit(X_train, y_train)

# 评估模型
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print(f"\n模型性能评估:")
print(f"训练集 R²: {train_score:.4f}")
print(f"测试集 R²: {test_score:.4f}")
print(f"交叉验证 R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 转换为 ONNX
print("\n正在转换为 ONNX 格式...")
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    target_opset=12,  # 使用较新的 ONNX 算子集
    name='advanced_temp_model'
)

# 验证 ONNX 模型
onnx.checker.check_model(onnx_model)
print("ONNX 模型验证通过")

# 保存模型
onnx_path = "advanced_temp_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# 同时保存 sklearn 模型用于对比
joblib.dump(pipeline, "advanced_temp_model_sklearn.pkl")

print(f"\n模型已保存:")
print(f"  ONNX 模型: {onnx_path}")
print(f"  Sklearn 模型: advanced_temp_model_sklearn.pkl")

# 打印模型信息
print(f"\n模型信息:")
print(f"  输入特征数: {X.shape[1]}")
print(f"  训练样本数: {len(X_train)}")
print(f"  测试样本数: {len(X_test)}")
print(f"  特征名称: temp_current, vibration_x, vibration_y, vibration_z, current, voltage, pressure, humidity")

# 测试预测
print("\n测试预测:")
test_sample = X_test[:5].astype(np.float32)
predictions = pipeline.predict(test_sample)
print(f"输入: {test_sample[0]}")
print(f"预测温度: {predictions[0]:.2f}°C")
print(f"实际温度: {y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0]:.2f}°C")