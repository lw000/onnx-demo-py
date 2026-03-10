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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import warnings
warnings.filterwarnings('ignore')

# 变频器，实现电容寿命预测和温升率检测

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

# 1.5 数据清洗函数
def clean_data(df):
    """数据清洗和验证"""
    print("\n=== 数据清洗开始 ===")
    df_clean = df.copy()

    # 1. 检查缺失值
    missing_count = df_clean.isnull().sum()
    if missing_count.sum() > 0:
        print(f"发现缺失值:\n{missing_count}")
        df_clean = df_clean.dropna()
        print(f"已删除缺失值行，剩余 {len(df_clean)} 条记录")
    else:
        print("✓ 无缺失值")

    # 2. 检查重复值
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        print(f"发现 {dup_count} 条重复记录，已删除")
        df_clean = df_clean.drop_duplicates()
    else:
        print("✓ 无重复记录")

    # 3. 数据范围验证
    print("\n数据范围验证:")
    ranges = {
        'load_factor': (0.0, 1.0),
        'temperature': (-50, 150),
        'ripple': (0, 50),
        'label_life': (0, 100),
        'label_thermal': (0, 1)
    }

    for col, (min_val, max_val) in ranges.items():
        if col in df_clean.columns:
            out_of_range = ((df_clean[col] < min_val) | (df_clean[col] > max_val)).sum()
            if out_of_range > 0:
                print(f"  ⚠ {col}: {out_of_range} 条超出范围 [{min_val}, {max_val}]")
                # 删除异常值
                df_clean = df_clean[(df_clean[col] >= min_val) & (df_clean[col] <= max_val)]
            else:
                print(f"  ✓ {col}: 全部在范围内 [{min_val}, {max_val}]")

    print(f"\n清洗后数据: {len(df_clean)} 条记录 (原始: {len(df)} 条)")
    print("=== 数据清洗完成 ===\n")
    return df_clean


def visualize_data(df, df_clean):
    """数据可视化"""
    print("=== 生成数据可视化图表 ===")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('变频器健康数据可视化 (清洗前后对比)', fontsize=14, fontweight='bold')

    # 样本数（用于归一化显示）
    n_samples = len(df)
    n_clean = len(df_clean)

    # 1. 温度变化曲线
    axes[0, 0].plot(df.index, df['temperature'], 'b-', alpha=0.3, label='原始')
    axes[0, 0].plot(df_clean.index, df_clean['temperature'], 'r-', linewidth=1, label='清洗后')
    axes[0, 0].set_title('温度变化曲线')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('温度 (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 纹波变化曲线
    axes[0, 1].plot(df.index, df['ripple'], 'b-', alpha=0.3, label='原始')
    axes[0, 1].plot(df_clean.index, df_clean['ripple'], 'r-', linewidth=1, label='清洗后')
    axes[0, 1].set_title('纹波变化曲线')
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('纹波 (V)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 负载率分布直方图
    axes[1, 0].hist(df['load_factor'], bins=50, alpha=0.5, label=f'原始 (n={n_samples})')
    axes[1, 0].hist(df_clean['load_factor'], bins=50, alpha=0.5, label=f'清洗后 (n={n_clean})')
    axes[1, 0].set_title('负载率分布')
    axes[1, 0].set_xlabel('负载率')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 电容寿命分布直方图
    axes[1, 1].hist(df['label_life'], bins=50, alpha=0.5, label=f'原始 (n={n_samples})')
    axes[1, 1].hist(df_clean['label_life'], bins=50, alpha=0.5, label=f'清洗后 (n={n_clean})')
    axes[1, 1].set_title('电容寿命分布')
    axes[1, 1].set_xlabel('寿命 (%)')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 5. 温度 vs 纹波散点图
    scatter = axes[2, 0].scatter(df_clean['temperature'], df_clean['ripple'],
                                 c=df_clean['label_thermal'], cmap='RdYlGn_r',
                                 alpha=0.6, s=10)
    axes[2, 0].set_title('温度 vs 纹波 (颜色=热故障)')
    axes[2, 0].set_xlabel('温度 (°C)')
    axes[2, 0].set_ylabel('纹波 (V)')
    axes[2, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2, 0], label='热故障概率')

    # 6. 温升异常计数
    thermal_count_raw = (df['label_thermal'] > 0.5).sum()
    thermal_count_clean = (df_clean['label_thermal'] > 0.5).sum()
    bars = axes[2, 1].bar(['原始', '清洗后'], [thermal_count_raw, thermal_count_clean],
                          color=['gray', 'red'])
    axes[2, 1].set_title('温升异常样本数')
    axes[2, 1].set_ylabel('样本数')
    for bar, count in zip(bars, [thermal_count_raw, thermal_count_clean]):
        height = bar.get_height()
        axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    viz_path = os.path.join(data_dir, "data_visualization.svg")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化图表已保存: {viz_path}\n")


# 检查数据文件是否存在
if os.path.exists(raw_data_file):
    print(f"从 {raw_data_file} 加载训练数据...")
    df = pd.read_csv(raw_data_file)
    print(f"成功加载数据，共 {len(df)} 条记录")

    # 数据清洗
    df = clean_data(df)

    # 数据可视化
    visualize_data(pd.read_csv(raw_data_file), df)
else:
    print(f"训练数据不存在，正在生成模拟数据...")
    df = generate_raw_data()
    # 生成数据后也需要清洗和可视化
    df = clean_data(df)
    visualize_data(df, df)  # 生成的数据没有异常，清洗前后相同

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