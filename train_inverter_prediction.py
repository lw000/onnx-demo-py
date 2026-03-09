import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import os

# 变频器，实现电容寿命预测和温升率检测

# 数据目录和文件路径
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
raw_data_file = os.path.join(data_dir, "inverter_raw_data.csv")

# 1. 检查或生成原始训练数据
def generate_raw_data():
    """生成模拟的原始数据并保存到CSV"""
    np.random.seed(42)
    n_samples = 5000

    # 基础信号
    time = np.arange(n_samples)
    load_factor = np.random.uniform(0.3, 1.0, n_samples)

    # 正常状态参数
    base_temp = 60.0
    base_ripple = 2.0

    # 初始化数组
    temps = np.zeros(n_samples)
    ripples = np.zeros(n_samples)
    labels_life = np.zeros(n_samples)
    labels_thermal = np.zeros(n_samples)

    # 模拟退化过程
    for i in range(n_samples):
        # 模拟电容老化
        aging_factor = min(1.0, i / (n_samples * 0.8))
        ripple_noise = np.random.normal(0, 0.5)
        ripples[i] = base_ripple + (aging_factor * 15.0 * load_factor[i]) + ripple_noise

        # 模拟温度
        thermal_fault = 0.0
        if i > n_samples * 0.7:
            thermal_fault = (i - n_samples * 0.7) * 0.5

        temps[i] = base_temp + (load_factor[i] * 20) + thermal_fault + np.random.normal(0, 1)

        # 构建标签
        life_pct = max(0, 100 - (ripples[i] - base_ripple) * 5.0)
        labels_life[i] = life_pct

        is_thermal_risk = 1.0 if thermal_fault > 5.0 else 0.0
        labels_thermal[i] = is_thermal_risk

    # 保存原始数据到CSV
    df = pd.DataFrame({
        'time': time,
        'load_factor': load_factor,
        'temperature': temps,
        'ripple': ripples,
        'label_life': labels_life,
        'label_thermal': labels_thermal
    })
    df.to_csv(raw_data_file, index=False)
    print(f"原始数据已生成并保存到: {raw_data_file}")
    return df

# 检查数据文件是否存在
if os.path.exists(raw_data_file):
    print(f"从 {raw_data_file} 加载训练数据...")
    df = pd.read_csv(raw_data_file)
    print(f"成功加载数据，共 {len(df)} 条记录")
else:
    print(f"训练数据不存在，正在生成模拟数据...")
    df = generate_raw_data()

# 2. 特征工程 (滑动窗口)
# 注意：C++ 端必须完全复现此逻辑
window_size = 20
X = []
y = []

temps = df['temperature'].values
ripples = df['ripple'].values
load_factor = df['load_factor'].values
labels_life = df['label_life'].values
labels_thermal = df['label_thermal'].values

for i in range(window_size, len(df)):
    t_win = temps[i-window_size:i]
    r_win = ripples[i-window_size:i]
    l_win = load_factor[i-window_size:i]

    # 特征向量 (6维)
    features = [
        np.mean(r_win),             # 0: 平均纹波 (关键)
        np.std(r_win),              # 1: 纹波波动
        np.mean(t_win),             # 2: 平均温度
        (t_win[-1] - t_win[0]),     # 3: 窗口内温差 (代表温升趋势)
        np.mean(l_win),             # 4: 平均负载
        np.max(t_win) - np.min(t_win) # 5: 温度波动范围
    ]

    X.append(features)
    # 多输出目标：[寿命百分比, 热故障概率]
    y.append([labels_life[i], labels_thermal[i]])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"数据集形状: X={X.shape}, Y={y.shape}")

# 3. 训练模型
# 使用随机森林回归器同时预测两个值
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
model.fit(X, y)

# 评估
y_pred = model.predict(X)
mae_life = mean_absolute_error(y[:, 0], y_pred[:, 0])
acc_thermal = accuracy_score((y[:, 1] > 0.5).astype(int), (y_pred[:, 1] > 0.5).astype(int))

print(f"\n模型评估:")
print(f"- 电容寿命预测 MAE: {mae_life:.2f}%")
print(f"- 温升异常检测准确率: {acc_thermal:.2%}")

# 4. 导出为 ONNX (张量模式)
# 关键点：指定初始类型为固定形状 [1, 6]，即 Batch Size=1, Features=6
# 这强制模型进入"张量模式"，适合 C++ 单次推理
initial_type = [('float_input', FloatTensorType([1, 6]))]

onnx_model = convert_sklearn(
    model, 
    initial_types=initial_type,
    target_opset=14
)

# 保存
model_path = "inverter_health_multi.onnx"
with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\n模型已导出 (张量模式 [1, 6]): {model_path}")

# 5. 验证导出模型
import onnxruntime as ort
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name

# 查看模型输入输出信息
print("\n=== ONNX 模型信息 ===")
print(f"输入节点: {input_name}, 形状: {sess.get_inputs()[0].shape}, 类型: {sess.get_inputs()[0].type}")
for i, out in enumerate(sess.get_outputs()):
    print(f"输出节点[{i}]: {out.name}, 形状: {out.shape}, 类型: {out.type}")

# 测试单条数据
test_input = X[:1]
pred = sess.run(None, {input_name: test_input})
print(f"\n=== ONNX 推理测试 ===")
print(f"测试输入: {test_input[0]}")
print(f"ONNX 预测结果 - [寿命%, 热故障]: {pred[0][0]}")

# 对比 sklearn 和 ONNX 推理结果
print("\n=== Sklearn vs ONNX 对比 ===")
sklearn_pred = model.predict(test_input)[0]
onnx_pred = pred[0][0]
print(f"Sklearn 预测: {sklearn_pred}")
print(f"ONNX 预测:   {onnx_pred}")
print(f"差异: {np.abs(sklearn_pred - onnx_pred)}")
print(f"一致: {np.allclose(sklearn_pred, onnx_pred, atol=1e-5)}")

# 批量测试 (逐条推理，因为模型固定 batch_size=1)
print("\n=== 批量推理测试 (逐条) ===")
batch_size = 10
batch_input = X[:batch_size]
batch_pred_sklearn = model.predict(batch_input)
batch_pred_onnx_list = []

for i in range(batch_size):
    single_input = batch_input[i:i+1]
    single_pred = sess.run(None, {input_name: single_input})[0][0]
    batch_pred_onnx_list.append(single_pred)

batch_pred_onnx = np.array(batch_pred_onnx_list)

print(f"批量预测 {batch_size} 个样本:")
print(f"Sklearn 平均寿命: {np.mean(batch_pred_sklearn[:, 0]):.2f}%")
print(f"ONNX 平均寿命:   {np.mean(batch_pred_onnx[:, 0]):.2f}%")
print(f"平均差异: {np.mean(np.abs(batch_pred_sklearn - batch_pred_onnx)):.6f}")

# 验证模型结构
print("\n=== ONNX 模型验证 ===")
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
print("ONNX 模型结构验证通过")