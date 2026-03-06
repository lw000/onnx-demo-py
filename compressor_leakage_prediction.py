# compressor_leakage_prediction.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

def generate_compressor_data(n_samples=10000):
    """
    生成模拟的空压站运行数据，包括正常和泄漏两种状态。
    """
    np.random.seed(42)
    
    data = []
    labels = []

    # --- 1. 正常运行状态 ---
    # 在正常状态下，供气流量和用气流量基本相等，压力稳定
    n_normal_samples = int(n_samples * 0.7) # 70% 的数据为正常
    
    pressure_normal = np.random.normal(loc=0.75, scale=0.05, size=n_normal_samples).clip(0.6, 0.85)
    supply_flow_normal = np.random.normal(loc=500, scale=20, size=n_normal_samples).clip(450, 550)
    # 用气流量略小于供气流量，模拟管道和设备的微小损耗
    demand_flow_normal = supply_flow_normal - np.random.normal(loc=10, scale=5, size=n_normal_samples).clip(5, 20)

    for i in range(n_normal_samples):
        data.append([pressure_normal[i], supply_flow_normal[i], demand_flow_normal[i]])
        labels.append(0) # 0 代表 "正常"

    # --- 2. 泄漏状态 ---
    # 在泄漏状态下，供气流量会高于用气流量，压力可能偏低或不稳定
    n_leak_samples = n_samples - n_normal_samples
    
    pressure_leak = np.random.normal(loc=0.70, scale=0.08, size=n_leak_samples).clip(0.5, 0.8) # 压力可能更低
    supply_flow_leak = np.random.normal(loc=510, scale=25, size=n_leak_samples).clip(460, 560) # 为了维持压力，供气可能略增
    # 用气流量远低于供气流量，差值即为泄漏量
    leakage_amount = np.random.uniform(low=30, high=80, size=n_leak_samples)
    demand_flow_leak = supply_flow_leak - leakage_amount

    for i in range(n_leak_samples):
        data.append([pressure_leak[i], supply_flow_leak[i], demand_flow_leak[i]])
        labels.append(1) # 1 代表 "泄漏"

    return np.array(data), np.array(labels)

def main():
    print("--- 开始训练空压系统管网泄漏预测模型 ---")

    # 1. 生成数据
    print("1. 生成模拟数据...")
    X, y = generate_compressor_data()

    # 2. 划分数据集
    print("2. 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. 训练模型
    print("3. 训练模型 (Random Forest Classifier)...")
    # 使用有监督的随机森林分类器，更适合已知标签的分类问题
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    model_pipeline.fit(X_train, y_train)  # 随机森林需要标签 y 进行训练

    # 4. 模型评估
    print("4. 评估模型...")
    y_pred = model_pipeline.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=["正常", "泄漏"]))

    # 5. 转换为 ONNX
    print("5. 转换模型为 ONNX 格式...")
    # RandomForestClassifier 的 ONNX 转换是支持的
    # 关键修改：添加 options 参数，强制使用 tensor 输出，禁用 zipmap
    options = {type(model_pipeline): {'zipmap': False}} # 使用 type(model_pipeline) 作为 key 也有效
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model_pipeline, initial_types=initial_type, target_opset=12, options=options) # 这是核心修改)

    # 6. 保存 ONNX 模型
    onnx_model_path = "compressor_leakage_detector.onnx"
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ ONNX 模型已保存至: {onnx_model_path}")

    # 7. 保存原始模型 (可选)
    joblib.dump(model_pipeline, "compressor_leakage_detector_sklearn.pkl")
    print(f"✅ scikit-learn 模型已保存至: compressor_leakage_detector_sklearn.pkl")

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