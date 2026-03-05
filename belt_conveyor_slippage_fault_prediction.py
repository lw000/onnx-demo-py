import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# -------------------------------------------------
# 1. 模拟皮带机数据 (Belt Conveyor Simulation)
# -------------------------------------------------
np.random.seed(42)
n_samples = 5000

# 正常工况数据
data_normal = {
    'current': np.random.normal(100, 10, n_samples//2),   # 电流 (A)
    'speed_head': np.random.normal(2.0, 0.1, n_samples//2), # 头部速度 (m/s)
    'speed_tail': np.random.normal(2.0, 0.1, n_samples//2), # 尾部速度 (m/s)
    'vibration_motor': np.random.normal(2, 0.5, n_samples//2), # 电机振动 (mm/s)
    'temperature_bearing': np.random.normal(60, 5, n_samples//2), # 轴承温度 (C)
    'slip_rate': np.random.normal(0, 0.02, n_samples//2) # 理论打滑率
}

# 故障工况数据 (模拟打滑、卡阻、轴承损坏)
data_fault = {
    'current': np.random.normal(130, 15, n_samples//2), # 负载增大或卡阻
    'speed_head': np.random.normal(2.0, 0.1, n_samples//2),
    'speed_tail': np.random.normal(1.5, 0.3, n_samples//2), # 尾部速度降低 (打滑/卡住)
    'vibration_motor': np.random.normal(8, 3, n_samples//2), # 振动剧烈
    'temperature_bearing': np.random.normal(85, 10, n_samples//2), # 温度升高
    'slip_rate': np.random.normal(0.15, 0.05, n_samples//2) # 打滑率高
}

# 合并数据
df_normal = pd.DataFrame(data_normal)
df_fault = pd.DataFrame(data_fault)
df_normal['fault'] = 0 # 0 = 正常
df_fault['fault'] = 1  # 1 = 故障

df = pd.concat([df_normal, df_fault], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True) # 打乱数据

# -------------------------------------------------
# 2. 特征工程 (关键步骤)
# -------------------------------------------------
# 计算头尾部速度差，这是判断打滑的核心指标
df['speed_diff'] = abs(df['speed_head'] - df['speed_tail'])

# 计算电流波动率
df['current_std'] = df['current'].rolling(window=5, min_periods=1).std()

# 选取最终特征
features = ['current', 'speed_diff', 'vibration_motor', 'temperature_bearing', 'current_std']
X = df[features]
y = df['fault']

# 处理可能存在的NaN (由滚动计算产生)
X = X.fillna(X.mean())

# -------------------------------------------------
# 3. 训练模型 (使用 Pipeline 便于 ONNX 转换)
# -------------------------------------------------
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 Pipeline (包含标准化和分类器)
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=200,      # 树的数量
        max_depth=8,          # 树的最大深度
        random_state=42,
        n_jobs=-1            # 使用所有 CPU 核心
    ))
])
model_pipeline.fit(X_train, y_train)

# -------------------------------------------------
# 4. 评估与预测
# -------------------------------------------------
y_pred = model_pipeline.predict(X_test)
print("皮带机故障预测模型准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['正常', '故障']))

# --- 特征重要性分析 ---
classifier = model_pipeline.named_steps['classifier']
importances = classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\n特征重要性排序 (皮带机):")
print(feature_importance_df)

# --- 混淆矩阵 ---
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['正常', '故障'], yticklabels=['正常', '故障'])
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.title('混淆矩阵 - 皮带机故障预测')
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 5. 保存模型为 ONNX 格式
# -------------------------------------------------
print("\n正在转换为 ONNX 格式...")
# 定义输入类型
initial_type = [('float_input', FloatTensorType([None, len(features)]))]

# 转换模型
onnx_model = convert_sklearn(
    model_pipeline,
    initial_types=initial_type,
    target_opset=12,
    name='belt_conveyor_slippage_fault_detector'
)

# 验证 ONNX 模型
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")

# 保存 ONNX 模型
onnx_model_path = "belt_conveyor_slippage_fault_detector.onnx"
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"✅ ONNX 模型已保存至: {onnx_model_path}")

# 保存原始 Sklearn 模型
joblib.dump(model_pipeline, "belt_conveyor_slippage_fault_detector_sklearn.pkl")
print(f"✅ scikit-learn 模型已保存至: belt_conveyor_slippage_fault_detector_sklearn.pkl")

# -------------------------------------------------
# 6. 验证 ONNX 模型一致性
# -------------------------------------------------
print("\n验证 ONNX 模型一致性...")
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_model_path)

    # 准备测试样本
    sample_input = X_test[y_test == 1][0:1].astype(np.float32)

    # Sklearn 预测
    sk_pred = model_pipeline.predict(sample_input)[0]
    sk_proba = model_pipeline.predict_proba(sample_input)[0]

    # ONNX 预测
    onnx_pred = sess.run(None, {'float_input': sample_input})
    onnx_label = int(onnx_pred[0][0])
    onnx_proba = onnx_pred[1][0]

    print(f"   - 测试样本输入: {sample_input[0]}")
    print(f"   - scikit-learn 预测 (0:正常, 1:故障): {sk_pred}")
    print(f"   - scikit-learn 概率: {sk_proba}")
    print(f"   - ONNX 模型预测 (0:正常, 1:故障): {onnx_label}")
    print(f"   - ONNX 概率: {onnx_proba}")
    print(f"   - 预测结果一致: {sk_pred == onnx_label}")

except ImportError:
    print("   - 未安装 onnxruntime，跳过验证。")
except Exception as e:
    print(f"   - ONNX 验证失败: {e}")

# -------------------------------------------------
# 7. 实时预测示例 (模拟)
# -------------------------------------------------
# 假设现在传感器传回一组新数据：电流高、速度差大、振动大
new_data = np.array([[140,      # 电流高 -> 可能卡阻
                      0.8,      # 速度差大 -> 严重打滑
                      10,       # 振动剧烈 -> 机械故障
                      90,       # 温度高
                      25]])     # 电流波动大

prediction = model_pipeline.predict(new_data)
prob = model_pipeline.predict_proba(new_data)

print("\n" + "="*40)
print("【实时监测】皮带机状态诊断:")
if prediction[0] == 1:
    print("🚨 预警：检测到皮带机异常！")
    print(f"故障概率: {prob[0][1]:.2%}")
else:
    print(" 状态：皮带机运行正常")
print("="*40)