import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# 模型说明：预测变频器 IGBT (绝缘栅双极型晶体管) 模块的未来温度，帮助实现预测性维护和过热预警

# 目录配置
base_dir = os.path.dirname(os.path.dirname(__file__))
model_dir = os.path.join(base_dir, "models")
samples_dir = os.path.join(base_dir, "samples")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

# 数据文件路径
data_file = os.path.join(samples_dir, "igbt_temp_samples.csv")


def generate_training_data(n_samples=5000):
    """生成 IGBT 温度预测训练数据"""
    np.random.seed(42)
    
    # 特征：
    # current: 输出电流 (0-100A)
    # frequency: 输出频率 (0-50Hz)
    # ambient_temp: 环境温度 (20-40°C)
    # temp_rate: 温升速率 (°C/s, 过去几秒的平均变化)
    # load_factor: 负载率 (0-1.0)
    data = {
        'current': np.random.uniform(10, 90, n_samples),
        'frequency': np.random.uniform(10, 50, n_samples),
        'ambient_temp': np.random.uniform(20, 45, n_samples),
        'temp_rate': np.random.uniform(-0.5, 2.0, n_samples), 
        'load_factor': np.random.uniform(0.2, 1.0, n_samples)
    }
    df = pd.DataFrame(data)

    # 标签：未来 5 秒后的 IGBT 温度 (构造一个物理逻辑公式 + 噪声)
    # 假设基础温度 40，电流和负载影响大，温升速率影响直接
    df['target_temp'] = (
        40 + 
        0.4 * df['current'] + 
        10 * df['load_factor'] + 
        0.5 * df['ambient_temp'] + 
        5 * df['temp_rate'] + 
        np.random.normal(0, 2, n_samples) # 噪声
    )
    
    # 保存到 CSV
    df.to_csv(data_file, index=False)
    print(f"📁 训练数据已生成并保存到: {data_file}")
    return df


# ==========================================
# 1. 加载或生成训练数据
# ==========================================
if os.path.exists(data_file):
    print(f"📂 从 {data_file} 加载训练数据...")
    df = pd.read_csv(data_file)
    print(f"✅ 成功加载数据，共 {len(df)} 条记录")
else:
    print("⚠️ 训练数据不存在，正在生成模拟数据...")
    df = generate_training_data()

X = df[['current', 'frequency', 'ambient_temp', 'temp_rate', 'load_factor']].values
y = df['target_temp'].values

# ==========================================
# 2. 数据集划分与模型训练
# ==========================================
print("🚀 正在训练模型...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建管道：标准化 + 随机森林回归
# 随机森林对异常值鲁棒，适合工业场景
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    ))
])

model_pipeline.fit(X_train, y_train)

# 评估
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ 模型评估 - MSE: {mse:.4f}, R²: {r2:.4f}")

# ==========================================
# 3. 导出为 ONNX 格式
# ==========================================
print("💾 正在导出 ONNX 模型...")

# 定义输入类型：名称为 'input', 形状为 [None, 5] (批量大小可变，特征数为5)
# 注意：Skl2onnx 处理 Pipeline 时，需要确保所有步骤都支持 ONNX 转换
# StandardScaler 和 RandomForest 均支持良好
initial_type = [('float_input', FloatTensorType([None, 5]))]

# 转换
onnx_model = convert_sklearn(
    model_pipeline, 
    initial_types=initial_type, 
    target_opset=14 # 使用较新的算子集
)

# 保存
model_filename = os.path.join(model_dir, "inverter_temp.onnx")
with open(model_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✨ 成功导出模型：{model_filename}")
print(f"   输入节点名：{onnx_model.graph.input[0].name}")
print(f"   输出节点名：{onnx_model.graph.output[0].name}")

# 验证 ONNX 模型 (可选)
import onnxruntime as ort
sess = ort.InferenceSession(model_filename)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# 测试一次推理
test_input = X_test[:1].astype(np.float32)
ort_pred = sess.run(None, {input_name: test_input})[0]
skl_pred = model_pipeline.predict(X_test[:1])
print(f"🔍 验证推理 - Sklearn: {skl_pred[0]:.4f}, ONNX: {ort_pred[0][0]:.4f}")