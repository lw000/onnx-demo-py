import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于无显示环境

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 目录配置
base_dir = os.path.dirname(os.path.dirname(__file__))
model_dir = os.path.join(base_dir, "models")
samples_dir = os.path.join(base_dir, "data")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)


# 重要特征说明:
    # Flow 流量
    # Head 扬程
    # Power 功率
    # Vibration 振动
    
# 衍生特征:
    # Efficiency_Index 估算效率 (Q * H) / P
    # Specific_Power 比功率 P / Q


def generate_pump_data(n_samples=10000):
    """
    生成模拟的泵运行数据，包括正常、磨损、汽蚀三种状态。
    """
    np.random.seed(42)

    data = []
    labels = []

    # 定义三种状态的参数范围
    states = {
        "normal": {
            "flow": (100, 120),
            "head": (50, 60),
            "power": (45, 55),
            "vibration": (0.5, 1.5),
            "eff_factor": 1.0  # 效率因子，正常状态为1.0
        },
        "wear": {
            "flow": (90, 110), # 磨损导致效率下降，流量略减
            "head": (45, 55),  # 扬程略有下降
            "power": (50, 60), # 可能需要更多功率维持运转
            "vibration": (1.5, 3.0), # 磨损导致振动加剧
            "eff_factor": 0.85 # 磨损状态效率降低，设为0.85
        },
        "cavitation": {
            "flow": (80, 100), # 汽蚀导致流量大幅下降
            "head": (40, 50),  # 扬程下降
            "power": (40, 50), # 功率可能下降（效率降低）
            "vibration": (2.5, 4.5), # 汽蚀产生强烈振动和噪音
            "eff_factor": 0.65 # 汽蚀状态效率降低，设为0.65
        }
    }

    # 创建标签映射: 字符串 -> 数字
    label_map = {"normal": 0, "wear": 1, "cavitation": 2}

    for state, params in states.items():
        n_state_samples = n_samples // 3

        flow = np.random.normal(loc=np.mean(params["flow"]), scale=(params["flow"][1]-params["flow"][0])/6, size=n_state_samples).clip(*params["flow"])
        head = np.random.normal(loc=np.mean(params["head"]), scale=(params["head"][1]-params["head"][0])/6, size=n_state_samples).clip(*params["head"])
        power = np.random.normal(loc=np.mean(params["power"]), scale=(params["power"][1]-params["power"][0])/6, size=n_state_samples).clip(*params["power"])
        vibration = np.random.exponential(scale=np.mean(params["vibration"]), size=n_state_samples).clip(*params["vibration"])

        # --- 核心改进：计算衍生特征 ---
        
        # 1. 理论水力功率 (Proportional to Q * H)
        # 注意：实际工程中需乘以 rho*g，但分类任务中比例常数不影响，可省略
        hydraulic_power = flow * head 
        
        # 2. 估算效率 (Efficiency Index) = (Q * H) / P
        # 添加微小值防止除以零
        efficiency = hydraulic_power / (power + 1e-6)
        
        # 为了模拟真实传感器的噪声，给效率特征加一点随机扰动
        noise = np.random.normal(0, 0.05 * efficiency, size=n_state_samples)
        efficiency_noisy = efficiency + noise

        # 3. 比功率 (Specific Power) = P / Q (辅助特征，反映单位流量能耗)
        specific_power = power / (flow + 1e-6)

        for i in range(n_state_samples):
            # 特征向量: [Flow, Head, Power, Vibration, Efficiency_Index, Specific_Power]
            data.append([flow[i], head[i], power[i], vibration[i], efficiency_noisy[i], specific_power[i]])
            labels.append(label_map[state])  # 使用数字标签

    return np.array(data), np.array(labels)

def save_data_to_csv(X, y, feature_names, filepath):
    """将生成的数据保存到 CSV 文件"""
    # 创建标签映射
    label_map_reverse = {0: "normal", 1: "wear", 2: "cavitation"}
    
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df['label'] = df['label'].map(label_map_reverse)
    df.to_csv(filepath, index=False)
    print(f"[OK] 数据已保存至: {filepath}")

def load_and_clean_data(filepath, feature_names):
    """从 CSV 文件加载数据并进行清洗"""
    print(f"从 CSV 加载数据: {filepath}")

    # 加载数据
    df = pd.read_csv(filepath)

    print(f"原始数据形状: {df.shape}")

    # 数据清洗
    # 1. 检查缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"发现 {missing_count} 个缺失值，已删除")
        df = df.dropna()

    # 2. 检查异常值（流量、扬程、功率必须为正值）
    df = df[(df['Flow'] > 0) & (df['Head'] > 0) & (df['Power'] > 0)]

    # 3. 检查振动值范围
    df = df[(df['Vibration'] >= 0) & (df['Vibration'] <= 10)]

    # 4. 检查标签
    df = df[df['label'].isin(['normal', 'wear', 'cavitation'])]

    # 去重
    before_dedup = len(df)
    df = df.drop_duplicates()
    if len(df) < before_dedup:
        print(f"删除了 {before_dedup - len(df)} 条重复记录")

    print(f"清洗后数据形状: {df.shape}")

    # 提取特征和标签
    X = df[feature_names].values
    y = df['label'].map({'normal': 0, 'wear': 1, 'cavitation': 2}).values

    return X, y

def visualize_data(X, y, feature_names, label_names, save_dir):
    """可视化清洗后的数据分布"""
    print("\n生成数据可视化图表...")

    # 创建DataFrame便于绘图
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = [label_names[i] for i in y]

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('泵故障预测数据分布分析', fontsize=16, fontweight='bold')

    label_colors = {'normal': '#2ecc71', 'wear': '#f39c12', 'cavitation': '#e74c3c'}

    # 1. 流量分布
    ax = axes[0, 0]
    for label in label_names:
        data = df[df['label'] == label]['Flow']
        ax.hist(data, bins=30, alpha=0.6, label=label, color=label_colors[label], edgecolor='black')
    ax.set_xlabel('Flow (m³/h)', fontsize=11)
    ax.set_ylabel('样本数', fontsize=11)
    ax.set_title('流量分布', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. 扬程分布
    ax = axes[0, 1]
    for label in label_names:
        data = df[df['label'] == label]['Head']
        ax.hist(data, bins=30, alpha=0.6, label=label, color=label_colors[label], edgecolor='black')
    ax.set_xlabel('Head (m)', fontsize=11)
    ax.set_ylabel('样本数', fontsize=11)
    ax.set_title('扬程分布', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. 功率分布
    ax = axes[1, 0]
    for label in label_names:
        data = df[df['label'] == label]['Power']
        ax.hist(data, bins=30, alpha=0.6, label=label, color=label_colors[label], edgecolor='black')
    ax.set_xlabel('Power (kW)', fontsize=11)
    ax.set_ylabel('样本数', fontsize=11)
    ax.set_title('功率分布', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. 振动分布
    ax = axes[1, 1]
    for label in label_names:
        data = df[df['label'] == label]['Vibration']
        ax.hist(data, bins=30, alpha=0.6, label=label, color=label_colors[label], edgecolor='black')
    ax.set_xlabel('Vibration (mm/s)', fontsize=11)
    ax.set_ylabel('样本数', fontsize=11)
    ax.set_title('振动分布', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # 保存图表
    hist_path = os.path.join(save_dir, "pump_failure_data_distribution.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 直方图已保存至: {hist_path}")
    plt.close()

    # 创建散点图矩阵（流量 vs 扬程）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('泵故障预测特征散点图', fontsize=16, fontweight='bold')

    # 流量 vs 扬程
    ax = axes[0, 0]
    for label in label_names:
        data = df[df['label'] == label]
        ax.scatter(data['Flow'], data['Head'], alpha=0.4, label=label, color=label_colors[label], s=10)
    ax.set_xlabel('Flow (m³/h)', fontsize=11)
    ax.set_ylabel('Head (m)', fontsize=11)
    ax.set_title('流量 vs 扬程', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 功率 vs 振动
    ax = axes[0, 1]
    for label in label_names:
        data = df[df['label'] == label]
        ax.scatter(data['Power'], data['Vibration'], alpha=0.4, label=label, color=label_colors[label], s=10)
    ax.set_xlabel('Power (kW)', fontsize=11)
    ax.set_ylabel('Vibration (mm/s)', fontsize=11)
    ax.set_title('功率 vs 振动', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 流量 vs 功率
    ax = axes[1, 0]
    for label in label_names:
        data = df[df['label'] == label]
        ax.scatter(data['Flow'], data['Power'], alpha=0.4, label=label, color=label_colors[label], s=10)
    ax.set_xlabel('Flow (m³/h)', fontsize=11)
    ax.set_ylabel('Power (kW)', fontsize=11)
    ax.set_title('流量 vs 功率', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 扬程 vs 振动
    ax = axes[1, 1]
    for label in label_names:
        data = df[df['label'] == label]
        ax.scatter(data['Head'], data['Vibration'], alpha=0.4, label=label, color=label_colors[label], s=10)
    ax.set_xlabel('Head (m)', fontsize=11)
    ax.set_ylabel('Vibration (mm/s)', fontsize=11)
    ax.set_title('扬程 vs 振动', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # 保存图表
    scatter_path = os.path.join(save_dir, "pump_failure_feature_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 散点图已保存至: {scatter_path}")
    plt.close()

    # 创建箱线图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('泵故障预测特征箱线图', fontsize=16, fontweight='bold')

    # 准备箱线图数据
    features = ['Flow', 'Head', 'Power', 'Vibration']
    feature_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (feature, pos) in enumerate(zip(features, feature_positions)):
        ax = axes[pos]
        data_to_plot = []
        labels_to_plot = []
        colors_to_plot = []

        for label in label_names:
            data = df[df['label'] == label][feature].values
            data_to_plot.append(data)
            labels_to_plot.append(label)
            colors_to_plot.append(label_colors[label])

        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                        showmeans=True, meanline=True, showfliers=False)

        for patch, color in zip(bp['boxes'], colors_to_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)

        ax.set_ylabel(feature, fontsize=11)
        ax.set_title(f'{feature} 箱线图', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图表
    boxplot_path = os.path.join(save_dir, "pump_failure_feature_boxplot.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 箱线图已保存至: {boxplot_path}")
    plt.close()

    # 创建标签分布饼图
    fig, ax = plt.subplots(figsize=(8, 8))
    label_counts = df['label'].value_counts()
    labels = label_counts.index
    sizes = label_counts.values
    colors = [label_colors[label] for label in labels]

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90, explode=(0.05, 0.05, 0.05),
                                       shadow=True, textprops={'fontsize': 12})

    ax.set_title('标签分布', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    # 保存图表
    pie_path = os.path.join(save_dir, "pump_failure_label_distribution.png")
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 饼图已保存至: {pie_path}")
    plt.close()

def main():
    print("--- 开始训练主排水泵故障预测模型 ---")

    # 特征名称定义
    feature_names = ["Flow", "Head", "Power", "Vibration", "Efficiency_Index", "Specific_Power"]
    print(f"特征列表: {feature_names}")

    # CSV 文件路径
    csv_path = os.path.join(samples_dir, "pump_failure_train_data.csv")

    # 检查是否已存在 CSV 文件
    regenerate_data = False

    if os.path.exists(csv_path):
        print(f"\n发现已存在的数据文件: {csv_path}")
        choice = input("是否重新生成数据？(y/n, 默认 n): ").strip().lower()
        regenerate_data = (choice == 'y')
    else:
        regenerate_data = True

    if regenerate_data:
        # 1. 生成数据
        print("\n1. 生成模拟数据...")
        X, y = generate_pump_data()
        print(f"生成数据完成: {len(X)} 条样本")

        # 2. 保存到 CSV
        save_data_to_csv(X, y, feature_names, csv_path)
    else:
        # 3. 从 CSV 加载数据并清洗
        print("\n从 CSV 文件加载并清洗数据...")
        X, y = load_and_clean_data(csv_path, feature_names)

    print(f"最终数据集: {len(X)} 条样本")
    print(f"正常样本: {np.sum(y == 0)} 条 ({np.sum(y == 0) / len(y):.1%})")
    print(f"磨损样本: {np.sum(y == 1)} 条 ({np.sum(y == 1) / len(y):.1%})")
    print(f"汽蚀样本: {np.sum(y == 2)} 条 ({np.sum(y == 2) / len(y):.1%})")

    # 可视化数据
    label_names = ["normal", "wear", "cavitation"]
    visualize_data(X, y, feature_names, label_names, samples_dir)

    # 2. 划分数据集
    print("\n2. 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. 训练模型
    print("3. 训练模型...")
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model_pipeline.fit(X_train, y_train)

    # 4. 模型评估
    print("4. 评估模型...")
    y_pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 5. 转换为 ONNX
    print("5. 转换模型为 ONNX 格式...")
    # 关键修改：添加 options 参数，强制使用 tensor 输出，禁用 zipmap
    options = {type(model_pipeline): {'zipmap': False}} # 使用 type(model_pipeline) 作为 key 也有效

    # 定义输入类型: [批大小, 特征数]。批大小设为 None 表示可以动态变化。
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model_pipeline, initial_types=initial_type, target_opset=12, options=options)

    # 6. 保存 ONNX 模型
    onnx_model_path = os.path.join(model_dir, "pump_failure_classifier.onnx")
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[OK] ONNX 模型已保存至: {onnx_model_path}")

    # 7. 保存原始模型 (可选，用于调试)
    pkl_model_path = os.path.join(model_dir, "pump_failure_classifier_sklearn.pkl")
    joblib.dump(model_pipeline, pkl_model_path)
    print(f"[OK] scikit-learn 模型已保存至: {pkl_model_path}")

    # 8. 验证 ONNX 模型 (可选)
    print("\n8. 验证 ONNX 模型一致性...")
    try:
        import onnxruntime as rt

        sess = rt.InferenceSession(onnx_model_path)

        # 获取输入输出信息
        input_name = sess.get_inputs()[0].name
        output_names = [output.name for output in sess.get_outputs()]
        print(f"   - 输入节点: {input_name}")
        print(f"   - 输出节点: {output_names}")

        # 准备一个测试样本
        sample_input = X_test[0:1].astype(np.float32)

        # scikit-learn 预测
        sk_pred = model_pipeline.predict(sample_input)
        sk_proba = model_pipeline.predict_proba(sample_input)

        # ONNX 预测
        onnx_outputs = sess.run(None, {input_name: sample_input})
        # ONNX 输出: [label, probabilities]
        onnx_label = onnx_outputs[0][0]
        onnx_proba = onnx_outputs[1][0]  # 概率数组

        print(f"   - scikit-learn 预测类别: {sk_pred[0]}, 概率: {sk_proba[0]}")
        print(f"   - ONNX 模型预测类别: {onnx_label}, 概率: {onnx_proba}")

        # 检查标签类型
        if isinstance(onnx_label, (bytes, str)):
            print(f"   - ONNX 标签类型: 字符串")
            onnx_label_str = onnx_label.decode('utf-8') if isinstance(onnx_label, bytes) else onnx_label
            onnx_label_int = onnx_label.get(onnx_label_str, -1)
            print(f"   - 预测类别一致: {sk_pred[0] == onnx_label_int}")
        else:
            print(f"   - ONNX 标签类型: {type(onnx_label)}")
            print(f"   - 预测类别一致: {sk_pred[0] == onnx_label}")

    except ImportError:
        print("   - 未安装 onnxruntime，跳过验证。")
    except Exception as e:
        print(f"   - ONNX 验证失败: {e}")

    print("\n--- Python 端任务完成 ---")

if __name__ == "__main__":
    main()
