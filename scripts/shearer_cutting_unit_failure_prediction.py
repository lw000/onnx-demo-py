# 1. 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# -----------------------------
# 2. 模拟煤矿设备（采煤机截割部）数据集
# 在实际应用中，这里应该是从数据库或传感器读取的真实数据
# -----------------------------
np.random.seed(42)
n_samples = 10000

data = {
    # 模拟特征 (Features)
    'vibration': np.random.normal(5, 2, n_samples), # 振动值 (mm/s)
    'temperature': np.random.normal(75, 10, n_samples), # 温度 (°C)
    'current': np.random.normal(150, 30, n_samples), # 电流 (A)
    'pressure': np.random.normal(200, 50, n_samples), # 液压 (bar)
    'oil_quality': np.random.normal(90, 10, n_samples), # 油液质量 (%)
    
    # 模拟目标变量 (Target): 0=正常, 1=故障
    # 这里加入一些逻辑：振动大、温度高、油质差时，故障概率增加
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 制造故障逻辑 (基于规则生成标签，模拟真实情况)
# 如果振动 > 8 且 温度 > 85，或者 油质 < 60，则标记为故障
df['fault'] = ((df['vibration'] > 8) & (df['temperature'] > 85)) | (df['oil_quality'] < 60)
df['fault'] = df['fault'].astype(int)

# 查看数据分布
print("数据集前5行:")
print(df.head())

print(f"\n故障样本占比: {df['fault'].mean():.2%}")

# -----------------------------
# 3. 数据预处理
# -----------------------------
# 定义特征 (X) 和目标 (y)
features = ['vibration', 'temperature', 'current', 'pressure', 'oil_quality']
X = df[features].values
y = df['fault'].values

# 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. 构建随机森林模型 (使用 Pipeline 便于 ONNX 转换)
# -----------------------------
# 创建 Pipeline (包含标准化和分类器)
# 虽然随机森林不需要标准化，但使用 Pipeline 便于 ONNX 转换和后续扩展
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,     # 树的数量
        max_depth=10,         # 树的最大深度 (防止过拟合)
        random_state=42,       # 保证结果可复现
        n_jobs=-1            # 使用所有 CPU 核心
    ))
])

# 训练模型
print("\n正在训练模型...")
model_pipeline.fit(X_train, y_train)

# -----------------------------
# 5. 预测与评估
# -----------------------------
# 在测试集上进行预测
y_pred = model_pipeline.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.2%}")

# 详细分类报告 (精确率、召回率、F1值)
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['正常', '故障']))

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常', '故障'], yticklabels=['正常', '故障'])
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.title('混淆矩阵')
plt.show()

# -----------------------------
# 6. 特征重要性分析 (关键步骤)
# -----------------------------
# 查看哪些传感器数据对预测故障最重要
# 从 Pipeline 中获取分类器
classifier = model_pipeline.named_steps['classifier']
importances = classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n特征重要性排序 (哪个参数最影响设备故障):")
print(feature_importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature')
plt.title('特征重要性分析 (采煤机截割部故障预测)')
plt.xlabel('重要性得分')
plt.tight_layout()
plt.show()

# -----------------------------
# 7. 保存模型为 ONNX 格式
# -----------------------------
print("\n正在转换为 ONNX 格式...")
# 定义输入类型
initial_type = [('float_input', FloatTensorType([None, len(features)]))]

# 转换模型
onnx_model = convert_sklearn(
    model_pipeline,
    initial_types=initial_type,
    target_opset=12,
    name='shearer_cutting_unit_failure_detector'
)

# 验证 ONNX 模型
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")

# 保存 ONNX 模型
onnx_model_path = "shearer_cutting_unit_failure_detector.onnx"
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"✅ ONNX 模型已保存至: {onnx_model_path}")

# 保存原始 Sklearn 模型
joblib.dump(model_pipeline, "shearer_cutting_unit_failure_detector_sklearn.pkl")
print(f"✅ scikit-learn 模型已保存至: shearer_cutting_unit_failure_detector_sklearn.pkl")

# -----------------------------
# 8. 验证 ONNX 模型一致性
# -----------------------------
print("\n验证 ONNX 模型一致性...")
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_model_path)

    # 准备测试样本
    sample_input = X_test[0:1].astype(np.float32)

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

# -----------------------------
# 9. 实时预测示例 (模拟)
# -----------------------------
# 假设现在有一组新的传感器读数
new_data = np.array([[12.5,  # 振动
                      95.0,  # 温度
                      180.0, # 电流
                      220.0, # 液压
                      40.0]]) # 油质

prediction = model_pipeline.predict(new_data)
probability = model_pipeline.predict_proba(new_data)

print("\n" + "="*40)
print("实时预测示例:")
print(f"输入数据: 振动={new_data[0][0]}, 温度={new_data[0][1]}, 油质={new_data[0][4]}")
print(f"预测结果: {'故障预警 ' if prediction[0] == 1 else '设备正常 '}")

# 显示故障概率
if prediction[0] == 1:
    print(f"故障概率: {probability[0][1]:.2%}")
else:
    print(f"正常概率: {probability[0][0]:.2%}")
print("="*40)