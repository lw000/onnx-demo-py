import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import warnings
warnings.filterwarnings('ignore')

# 煤矿带式运输机皮带机打滑预测

def create_dataset():
    """
    模拟皮带机传感器数据集
    特征: [电流, 速度差, 振动, 温度]
    标签: [0: 正常, 1: 打滑]
    """
    print("正在生成模拟数据集...")
    np.random.seed(42)
    n_samples = 8000
    
    # 正常工况数据
    normal_data = {
        'current': np.random.normal(100, 10, n_samples // 2),
        'speed_diff': np.random.normal(0, 0.02, n_samples // 2), # 速度差接近0
        'vibration': np.random.normal(2, 0.5, n_samples // 2),
        'temperature': np.random.normal(60, 5, n_samples // 2),
        'label': np.zeros(n_samples // 2, dtype=int)
    }
    
    # 打滑工况数据 (速度差大, 电流高, 振动加剧)
    slip_data = {
        'current': np.random.normal(130, 15, n_samples // 2),
        'speed_diff': np.random.normal(0.2, 0.05, n_samples // 2), # 速度差明显增大
        'vibration': np.random.normal(8, 2, n_samples // 2),
        'temperature': np.random.normal(85, 8, n_samples // 2),
        'label': np.ones(n_samples // 2, dtype=int)
    }
    
    # 合并数据
    all_data = {k: np.concatenate([v, slip_data[k]]) for k, v in normal_data.items()}
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1).reset_index(drop=True) # 打乱数据
    
    print(f"数据集生成完成。样本总数: {len(df)}, 打滑样本占比: {df['label'].mean():.2%}")
    return df

def train_model(df):
    """训练模型"""
    print("\n正在训练模型...")
    features = ['current', 'speed_diff', 'vibration', 'temperature']
    X = df[features]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用随机森林
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"模型训练完成。测试集准确率: {acc:.2%}")
    print("\n分类报告:\n", classification_report(y_test, y_pred))
    
    return model, features

def export_to_onnx(model, features, filename="conveyor_slip_model.onnx"):
    """将模型导出为 ONNX 格式"""
    print(f"\n正在将模型导出为 ONNX 格式 ({filename})...")
    
    # 定义输入类型
    initial_type = [('float_input', FloatTensorType([None, len(features)]))]
    
    # 转换模型
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    
    # 保存模型
    with open(filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    # 验证模型
    onnx_model_check = onnx.load(filename)
    onnx.checker.check_model(onnx_model_check)
    print(f"模型已成功导出为 ONNX 格式: {filename}")

if __name__ == "__main__":
    dataset = create_dataset()
    model, features = train_model(dataset)
    export_to_onnx(model, features)
    
    print("\n--- Python 模型训练与导出完成 ---")
    print("接下来请运行 C++ 代码进行推理测试。")