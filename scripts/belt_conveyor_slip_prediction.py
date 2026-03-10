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

def create_dataset():
    print("正在生成模拟数据集...")
    np.random.seed(42)
    n_samples = 8000
    
    normal_data = {
        'current': np.random.normal(100, 10, n_samples // 2),
        'speed_diff': np.random.normal(0, 0.02, n_samples // 2),
        'vibration': np.random.normal(2, 0.5, n_samples // 2),
        'temperature': np.random.normal(60, 5, n_samples // 2),
        'label': np.zeros(n_samples // 2, dtype=int)
    }
    
    slip_data = {
        'current': np.random.normal(130, 15, n_samples // 2),
        'speed_diff': np.random.normal(0.2, 0.05, n_samples // 2),
        'vibration': np.random.normal(8, 2, n_samples // 2),
        'temperature': np.random.normal(85, 8, n_samples // 2),
        'label': np.ones(n_samples // 2, dtype=int)
    }
    
    all_data = {k: np.concatenate([v, slip_data[k]]) for k, v in normal_data.items()}
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"数据集生成完成。样本总数: {len(df)}, 打滑样本占比: {df['label'].mean():.2%}")
    return df

def train_model(df):
    print("\n正在训练模型...")
    features = ['current', 'speed_diff', 'vibration', 'temperature']
    X = df[features]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"模型训练完成。测试集准确率: {acc:.2%}")
    print("\n分类报告:\n", classification_report(y_test, y_pred))
    
    return model, features

def export_to_onnx(model, features, filename="conveyor_slip_model.onnx"):
    print(f"\n正在将模型导出为 ONNX 格式 ({filename})...")
    
    initial_type = [('float_input', FloatTensorType([None, len(features)]))] # 输入类型
    
    # 关键修改：添加 options 参数，强制使用 tensor 输出，禁用 zipmap
    options = {type(model): {'zipmap': False}} # 使用 type(model) 作为 key 也有效
    
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_type, 
        target_opset=12,
        options=options # 这是核心修改
    )
    
    # 验证模型
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过。")
    except onnx.checker.ValidationError as e:
        print(f"ONNX 模型验证失败: {e}")

    with open(filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"模型已成功导出为 ONNX 格式: {filename}")

if __name__ == "__main__":
    dataset = create_dataset()
    model, features = train_model(dataset)
    export_to_onnx(model, features)
    
    print("\n--- Python 模型训练与导出完成 ---")
    print("请重新运行 C++ 代码进行推理测试。")
