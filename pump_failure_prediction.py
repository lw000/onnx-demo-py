# pump_failure_prediction.py
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
            "vibration": (0.5, 1.5)
        },
        "wear": {
            "flow": (90, 110), # 磨损导致效率下降，流量略减
            "head": (45, 55),  # 扬程略有下降
            "power": (50, 60), # 可能需要更多功率维持运转
            "vibration": (1.5, 3.0) # 磨损导致振动加剧
        },
        "cavitation": {
            "flow": (80, 100), # 汽蚀导致流量大幅下降
            "head": (40, 50),  # 扬程下降
            "power": (40, 50), # 功率可能下降（效率降低）
            "vibration": (2.5, 4.5) # 汽蚀产生强烈振动和噪音
        }
    }

    for state, params in states.items():
        n_state_samples = n_samples // 3
        
        flow = np.random.normal(loc=np.mean(params["flow"]), scale=(params["flow"][1]-params["flow"][0])/6, size=n_state_samples).clip(*params["flow"])
        head = np.random.normal(loc=np.mean(params["head"]), scale=(params["head"][1]-params["head"][0])/6, size=n_state_samples).clip(*params["head"])
        power = np.random.normal(loc=np.mean(params["power"]), scale=(params["power"][1]-params["power"][0])/6, size=n_state_samples).clip(*params["power"])
        vibration = np.random.exponential(scale=np.mean(params["vibration"]), size=n_state_samples).clip(*params["vibration"])

        for i in range(n_state_samples):
            data.append([flow[i], head[i], power[i], vibration[i]])
            labels.append(state)
            
    return np.array(data), np.array(labels)

def main():
    print("--- 开始训练主排水泵故障预测模型 ---")

    # 1. 生成数据
    print("1. 生成模拟数据...")
    X, y = generate_pump_data()

    # 2. 划分数据集
    print("2. 划分训练集和测试集...")
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
    # 定义输入类型: [批大小, 特征数]。批大小设为 None 表示可以动态变化。
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model_pipeline, initial_types=initial_type, target_opset=12)

    # 6. 保存 ONNX 模型
    onnx_model_path = "pump_failure_classifier.onnx"
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ ONNX 模型已保存至: {onnx_model_path}")

    # 7. 保存原始模型 (可选，用于调试)
    joblib.dump(model_pipeline, "pump_failure_classifier_sklearn.pkl")
    print(f"✅ scikit-learn 模型已保存至: pump_failure_classifier_sklearn.pkl")

    # 8. 验证 ONNX 模型 (可选)
    print("\n8. 验证 ONNX 模型一致性...")
    try:
        import onnxruntime as rt
        
        sess = rt.InferenceSession(onnx_model_path)
        
        # 准备一个测试样本
        sample_input = X_test[0:1].astype(np.float32)
        
        # scikit-learn 预测
        sk_pred = model_pipeline.predict(sample_input)
        sk_proba = model_pipeline.predict_proba(sample_input)
        
        # ONNX 预测
        onnx_pred = sess.run(None, {'float_input': sample_input})
        # ONNX 输出通常是 (label, probabilities)
        onnx_label = onnx_pred[0][0]
        onnx_proba = onnx_pred[1][0] # 概率字典
        
        print(f"   - scikit-learn 预测类别: {sk_pred[0]}, 概率: {sk_proba[0]}")
        print(f"   - ONNX 模型预测类别: {onnx_label}, 概率: {list(onnx_proba.values())}")
        print(f"   - 预测类别一致: {sk_pred[0] == onnx_label}")
        
    except ImportError:
        print("   - 未安装 onnxruntime，跳过验证。")
    except Exception as e:
        print(f"   - ONNX 验证失败: {e}")

    print("\n--- Python 端任务完成 ---")

if __name__ == "__main__":
    main()