# compressor_leakage_prediction.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
import os

# 目录配置
base_dir = os.path.dirname(os.path.dirname(__file__))
model_dir = os.path.join(base_dir, "models")
samples_dir = os.path.join(base_dir, "data")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

def generate_compressor_data(n_samples=10000):
    np.random.seed(42)
    data = []
    labels = []

    # --- 1. 正常运行状态 (70%) ---
    n_normal = int(n_samples * 0.7)
    
    # 模拟不同压力工况
    pressure_normal = np.random.normal(loc=0.75, scale=0.05, size=n_normal).clip(0.6, 0.85)
    supply_flow_normal = np.random.normal(loc=500, scale=50, size=n_normal).clip(300, 700)
    
    # 正常损耗随流量增大而略微增大，且存在随机噪声
    # 基础损耗 5-15，加上流量的 1% 作为动态损耗
    base_loss = np.random.uniform(5, 15, size=n_normal)
    dynamic_loss = supply_flow_normal * 0.01 * np.random.uniform(0.5, 1.5, size=n_normal)
    total_loss = base_loss + dynamic_loss
    
    demand_flow_normal = supply_flow_normal - total_loss

    for i in range(n_normal):
        # 构造特征：[P, Q_sup, Q_dem, Diff, Ratio]
        diff = supply_flow_normal[i] - demand_flow_normal[i]
        ratio = supply_flow_normal[i] / (demand_flow_normal[i] + 1e-5)
        data.append([pressure_normal[i], supply_flow_normal[i], demand_flow_normal[i], diff, ratio])
        labels.append(0)

    # --- 2. 泄漏状态 (30%) ---
    n_leak = n_samples - n_normal
    
    pressure_leak = np.random.normal(loc=0.70, scale=0.08, size=n_leak).clip(0.5, 0.8)
    supply_flow_leak = np.random.normal(loc=500, scale=50, size=n_leak).clip(300, 700)
    
    # 泄漏量模拟：包含微小泄漏 (难检测) 到巨大泄漏
    # 让部分泄漏量与正常损耗重叠 (15-25)，考验模型结合压力判断的能力
    leakage_amount = np.random.uniform(15, 90, size=n_leak) 
    
    demand_flow_leak = supply_flow_leak - leakage_amount
    
    # 确保 demand 不为负
    demand_flow_leak = np.maximum(demand_flow_leak, 10)

    for i in range(n_leak):
        diff = supply_flow_leak[i] - demand_flow_leak[i]
        ratio = supply_flow_leak[i] / (demand_flow_leak[i] + 1e-5)
        data.append([pressure_leak[i], supply_flow_leak[i], demand_flow_leak[i], diff, ratio])
        labels.append(1)

    return np.array(data), np.array(labels)

def save_data_to_csv(X, y, feature_names, filepath):
    """将生成的数据保存到 CSV 文件"""
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df['label'] = df['label'].map({0: 'normal', 1: 'leak'})
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

    # 2. 检查异常值（压力应在合理范围内）
    df = df[(df['Pressure'] >= 0.4) & (df['Pressure'] <= 0.9)]

    # 3. 检查流量异常（不应为负值）
    df = df[(df['SupplyFlow'] > 0) & (df['DemandFlow'] > 0)]

    # 4. 检查衍生特征的合理性
    # FlowDiff 应该非负
    df = df[df['FlowDiff'] >= 0]

    # FlowRatio 应该合理（1.0 - 100.0）
    df = df[(df['FlowRatio'] >= 1.0) & (df['FlowRatio'] <= 100.0)]

    # 5. 检查标签
    df = df[df['label'].isin(['normal', 'leak'])]

    # 去重
    before_dedup = len(df)
    df = df.drop_duplicates()
    if len(df) < before_dedup:
        print(f"删除了 {before_dedup - len(df)} 条重复记录")

    print(f"清洗后数据形状: {df.shape}")

    # 提取特征和标签
    X = df[feature_names].values
    y = df['label'].map({'normal': 0, 'leak': 1}).values

    return X, y

def main():
    print("--- 开始训练优化的空压系统泄漏预测模型 ---")

    # 特征名称定义
    feature_names = ["Pressure", "SupplyFlow", "DemandFlow", "FlowDiff", "FlowRatio"]
    print(f"特征列表: {feature_names}")

    # CSV 文件路径
    csv_path = os.path.join(samples_dir, "compressor_leakage_train_data.csv")

    # 检查是否已存在 CSV 文件
    regenerate_data = False

    if os.path.exists(csv_path):
        print(f"\n发现已存在的数据文件: {csv_path}")
        choice = input("是否重新生成数据？(y/n, 默认 n): ").strip().lower()
        regenerate_data = (choice == 'y')
    else:
        regenerate_data = True

    if regenerate_data:
        # 1. 生成数据 (包含 5 个特征)
        X, y = generate_compressor_data()
        print(f"\n生成数据完成: {len(X)} 条样本")

        # 2. 保存到 CSV
        save_data_to_csv(X, y, feature_names, csv_path)
    else:
        # 3. 从 CSV 加载数据并清洗
        X, y = load_and_clean_data(csv_path, feature_names)

    print(f"最终数据集: {len(X)} 条样本")
    print(f"正常样本: {np.sum(y == 0)} 条 ({np.sum(y == 0) / len(y):.1%})")
    print(f"泄漏样本: {np.sum(y == 1)} 条 ({np.sum(y == 1) / len(y):.1%})")

    # 2. 划分数据集 (保持分层采样)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. 构建模型管道
    # class_weight='balanced' 非常重要，防止模型忽略少数类 (泄漏)
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced', # 自动处理不平衡
            max_depth=10             # 限制深度防止过拟合
        ))
    ])
    
    model_pipeline.fit(X_train, y_train)

    # 4. 模型评估
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)

    print("\n--- 模型评估报告 ---")
    # target_names 对应标签 0 和 1
    print(classification_report(y_test, y_pred, target_names=["正常", "泄漏"]))
    
    # 特别关注混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)
    print(f"漏报数 (实际泄漏但预测为正常): {cm[1][0]}")
    print(f"误报数 (实际正常但预测为泄漏): {cm[0][1]}")

    # 5. 转换为 ONNX
    print("\n5. 转换模型为 ONNX 格式...")
    
    # 输入维度现在是 5 (因为加了特征工程)
    initial_type = [('float_input', FloatTensorType([None, 5]))]
    
    # 关键：禁用 zipmap，确保输出是 Tensor [N, 2]
    options = {type(model_pipeline.named_steps['classifier']): {'zipmap': False}}
    
    onnx_model = convert_sklearn(
        model_pipeline, 
        initial_types=initial_type, 
        target_opset=12, 
        options=options
    )

    # 验证 ONNX 输出类型
    for output in onnx_model.graph.output:
        print(f"ONNX 输出节点 '{output.name}' 类型: {output.type}")
    # 6. 保存 ONNX 模型
    onnx_model_path = os.path.join(model_dir, "compressor_leakage_detector.onnx")
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[OK] ONNX 模型已保存至: {onnx_model_path}")

    # 7. 保存原始模型 (可选)
    pkl_model_path = os.path.join(model_dir, "compressor_leakage_detector_sklearn.pkl")
    joblib.dump(model_pipeline, pkl_model_path)
    print(f"[OK] scikit-learn 模型已保存至: {pkl_model_path}")

    # 8. 验证 ONNX 模型 (可选)
    print("\n8. 验证 ONNX 模型一致性...")
    try:
        import onnxruntime as rt

        sess = rt.InferenceSession(onnx_model_path)

        # 准备一个测试样本 (模拟一个泄漏场景)
        sample_input = X_test[y_test == 1][0:1].astype(np.float32)  # 取一个泄漏样本

        # scikit-learn 预测
        sk_pred = model_pipeline.predict(sample_input)[0]
        sk_proba = model_pipeline.predict_proba(sample_input)[0]

        # ONNX 预测
        onnx_pred = sess.run(None, {'float_input': sample_input})
        # RandomForestClassifier 的 ONNX 输出: [label_index, probabilities]
        onnx_label = int(onnx_pred[0][0])
        onnx_proba = onnx_pred[1][0]

        print(f"   - 测试样本输入: {sample_input[0]}")
        print(f"   - scikit-learn 预测 (0:正常, 1:泄漏): {sk_pred}")
        print(f"   - scikit-learn 概率: {sk_proba}")
        print(f"   - ONNX 模型预测 (0:正常, 1:泄漏): {onnx_label}")
        print(f"   - ONNX 概率: {onnx_proba}")
        print(f"   - 预测结果一致: {sk_pred == onnx_label}")

    except ImportError:
        print("   - 未安装 onnxruntime，跳过验证。")
    except Exception as e:
        print(f"   - ONNX 验证失败: {e}")

    print("\n--- Python 端任务完成 ---")

if __name__ == "__main__":
    main()