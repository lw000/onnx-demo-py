# 复杂温度预测模型
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
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

# 设置随机种子保证可复现性
np.random.seed(42)

def generate_temp_data(n_samples=5000):
    """生成温度预测模拟数据"""
    # 基础特征
    temp_current = np.random.uniform(20, 80, n_samples)  # 当前温度 (20-80°C)
    vibration_x = np.random.uniform(0, 10, n_samples)    # X轴振动
    vibration_y = np.random.uniform(0, 10, n_samples)    # Y轴振动
    vibration_z = np.random.uniform(0, 10, n_samples)    # Z轴振动
    current = np.random.uniform(1, 15, n_samples)        # 电流 (A)
    voltage = np.random.uniform(220, 240, n_samples)     # 电压 (V)
    pressure = np.random.uniform(90, 110, n_samples)     # 气压
    humidity = np.random.uniform(30, 70, n_samples)      # 湿度

    # 构建特征矩阵
    X = np.column_stack([
        temp_current, vibration_x, vibration_y, vibration_z,
        current, voltage, pressure, humidity
    ]).astype(np.float32)

    # 构建更复杂的温度预测目标 (非线性关系 + 交互效应)
    # 温度受多种因素影响，包含非线性关系和特征交互
    y = (
        0.6 * temp_current +                          # 当前温度影响最大
        0.25 * (vibration_x * vibration_y) ** 0.5 +   # 振动交互效应
        0.15 * current ** 0.8 +                        # 电流的非线性影响
        0.08 * (voltage - 230) * 0.5 +                 # 电压偏差影响
        0.05 * np.sin(pressure * 0.1) +               # 气压周期性影响
        0.03 * (humidity - 50) ** 2 / 100 +            # 湿度二次影响
        0.02 * vibration_x * current / 10 +           # 振动与电流的交互
        15                                             # 基础偏移
    ) + np.random.normal(0, 0.5, size=n_samples)      # 添加噪声

    return X, y

def save_data_to_csv(X, y, feature_names, filepath):
    """将生成的数据保存到 CSV 文件"""
    df = pd.DataFrame(X, columns=feature_names)
    df['temperature'] = y
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

    # 2. 检查异常值（温度应在合理范围内）
    df = df[(df['temperature'] >= 0) & (df['temperature'] <= 100)]

    # 3. 检查振动值范围
    df = df[(df['vibration_x'] >= 0) & (df['vibration_x'] <= 20)]
    df = df[(df['vibration_y'] >= 0) & (df['vibration_y'] <= 20)]
    df = df[(df['vibration_z'] >= 0) & (df['vibration_z'] <= 20)]

    # 4. 检查电压范围
    df = df[(df['voltage'] >= 200) & (df['voltage'] <= 250)]

    # 5. 检查电流范围
    df = df[(df['current'] > 0) & (df['current'] <= 20)]

    # 去重
    before_dedup = len(df)
    df = df.drop_duplicates()
    if len(df) < before_dedup:
        print(f"删除了 {before_dedup - len(df)} 条重复记录")

    print(f"清洗后数据形状: {df.shape}")

    # 提取特征和标签
    X = df[feature_names].values
    y = df['temperature'].values

    return X, y

def main():
    """主函数"""
    n_samples = 5000
    feature_names = ["temp_current", "vibration_x", "vibration_y", "vibration_z",
                     "current", "voltage", "pressure", "humidity"]
    csv_path = os.path.join(samples_dir, "temp_prediction_train_data.csv")

    print("--- 开始训练复杂温度预测模型 ---")
    print(f"特征列表: {feature_names}")

    # 检查是否已存在 CSV 文件
    regenerate_data = False

    if os.path.exists(csv_path):
        print(f"\n发现已存在的数据文件: {csv_path}")
        choice = input("是否重新生成数据？(y/n, 默认 n): ").strip().lower()
        regenerate_data = (choice == 'y')
    else:
        regenerate_data = True

    if regenerate_data:
        # 生成数据
        print("\n1. 生成模拟数据...")
        X, y = generate_temp_data(n_samples)
        print(f"生成数据完成: {len(X)} 条样本")

        # 保存到 CSV
        save_data_to_csv(X, y, feature_names, csv_path)
    else:
        # 从 CSV 加载数据并清洗
        print("\n从 CSV 文件加载并清洗数据...")
        X, y = load_and_clean_data(csv_path, feature_names)

    print(f"最终数据集: {len(X)} 条样本")
    print(f"温度范围: {y.min():.2f}°C - {y.max():.2f}°C")
    print(f"温度均值: {y.mean():.2f}°C")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 构建复杂的模型流水线
    # 1. 标准化特征
    # 2. 添加多项式特征 (捕获非线性关系)
    # 3. 使用随机森林回归器
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ))
    ])

    # 训练模型
    print("\n2. 训练模型...")
    pipeline.fit(X_train, y_train)

    # 评估模型
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    print(f"\n3. 模型性能评估:")
    print(f"  训练集 R2: {train_score:.4f}")
    print(f"  测试集 R2: {test_score:.4f}")
    print(f"  交叉验证 R2: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 转换为 ONNX
    print("\n4. 转换为 ONNX 格式...")
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=12,  # 使用较新的 ONNX 算子集
        name='advanced_temp_model'
    )

    # 验证 ONNX 模型
    onnx.checker.check_model(onnx_model)
    print("  ONNX 模型验证通过")

    # 保存模型
    onnx_path = os.path.join(model_dir, "advanced_temp_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 同时保存 sklearn 模型用于对比
    pkl_path = os.path.join(model_dir, "advanced_temp_model_sklearn.pkl")
    joblib.dump(pipeline, pkl_path)

    print(f"  ONNX 模型: {onnx_path}")
    print(f"  Sklearn 模型: {pkl_path}")

    # 打印模型信息
    print(f"\n5. 模型信息:")
    print(f"  输入特征数: {X.shape[1]}")
    print(f"  训练样本数: {len(X_train)}")
    print(f"  测试样本数: {len(X_test)}")
    print(f"  特征名称: {', '.join(feature_names)}")

    # 测试预测
    print("\n6. 测试预测:")
    test_sample = X_test[:5].astype(np.float32)
    predictions = pipeline.predict(test_sample)
    print(f"  输入: {test_sample[0]}")
    print(f"  预测温度: {predictions[0]:.2f}°C")
    print(f"  实际温度: {y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0]:.2f}°C")

    # ONNX 推理验证
    print("\n7. ONNX 推理验证")
    print("="*50)

    try:
        import onnxruntime as ort

        # 加载 ONNX 模型
        session = ort.InferenceSession(onnx_path)

        # 准备测试数据
        test_input = test_sample[0:1].astype(np.float32)

        # Sklearn 预测
        sklearn_pred = pipeline.predict(test_input)[0]

        # ONNX 预测
        onnx_result = session.run(None, {'float_input': test_input})
        # ONNX 输出可能是 shape (1,) 或 (1, 1)
        onnx_output = onnx_result[0].flatten()
        onnx_pred = float(onnx_output[0])

        print(f"\n【对比结果】")
        print(f"  测试输入: {test_input[0]}")
        print(f"  Sklearn 预测温度: {sklearn_pred:.4f}°C")
        print(f"  ONNX 预测温度: {onnx_pred:.4f}°C")
        print(f"  差异: {abs(sklearn_pred - onnx_pred):.6f}°C")

        if abs(sklearn_pred - onnx_pred) < 0.001:
            print("  [OK] Sklearn 和 ONNX 预测结果一致")
        else:
            print("  [WARNING] 预测结果存在轻微差异（可能由浮点精度导致）")

        # 批量推理测试
        print(f"\n【批量推理测试】")
        batch_input = test_sample[:10].astype(np.float32)

        # Sklearn 批量预测
        sklearn_batch = pipeline.predict(batch_input)

        # ONNX 批量预测
        onnx_batch_result = session.run(None, {'float_input': batch_input})
        # 确保 ONNX 输出是一维数组
        onnx_batch = onnx_batch_result[0].flatten()

        print(f"  批量预测样本数: {len(batch_input)}")
        print(f"  平均差异: {np.mean(np.abs(sklearn_batch - onnx_batch)):.6f}°C")
        print(f"  最大差异: {np.max(np.abs(sklearn_batch - onnx_batch)):.6f}°C")

        # 特定样本展示
        print(f"\n【前3个样本预测结果】")
        for i in range(min(3, len(batch_input))):
            print(f"  样本 {i+1}: Sklearn={sklearn_batch[i]:.2f}°C, ONNX={onnx_batch[i]:.2f}°C, 差异={abs(sklearn_batch[i] - onnx_batch[i]):.6f}°C")

    except ImportError:
        print("\n  [WARNING] 未安装 onnxruntime，跳过 ONNX 推理测试")
        print("  安装命令: pip install onnxruntime")
    except Exception as e:
        print(f"\n  [ERROR] ONNX 推理测试失败: {e}")

    # 实时预测示例
    print("\n8. 实时预测示例")
    print("="*50)

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)

        # 模拟不同工况的实时数据
        scenarios = {
            "正常工况": np.array([[25.0, 2.0, 2.5, 1.8, 5.0, 230.0, 100.0, 50.0]], dtype=np.float32),
            "高负荷": np.array([[60.0, 5.0, 6.0, 4.5, 12.0, 225.0, 105.0, 55.0]], dtype=np.float32),
            "低负荷": np.array([[20.0, 1.0, 1.5, 1.2, 2.0, 235.0, 95.0, 45.0]], dtype=np.float32),
            "振动异常": np.array([[40.0, 8.0, 9.0, 7.5, 8.0, 228.0, 102.0, 60.0]], dtype=np.float32),
        }

        print("\n工况预测结果:")
        for scenario_name, input_data in scenarios.items():
            result = session.run(None, {'float_input': input_data})
            # 确保 ONNX 输出是一维数组
            result_flat = result[0].flatten()
            predicted_temp = float(result_flat[0])

            print(f"\n  {scenario_name}:")
            print(f"    输入: [温度={input_data[0][0]}°C, 振动_X={input_data[0][1]}, 振动_Y={input_data[0][2]}, "
                  f"振动_Z={input_data[0][3]}, 电流={input_data[0][4]}A, 电压={input_data[0][5]}V, "
                  f"气压={input_data[0][6]}, 湿度={input_data[0][7]}]")
            print(f"    预测温度: {predicted_temp:.2f}°C")

            # 温度预警
            if predicted_temp > 70:
                print(f"    [ALERT] 预警: 温度过高！")
            elif predicted_temp > 50:
                print(f"    [INFO] 提示: 温度偏高")
            else:
                print(f"    [OK] 温度正常")

    except ImportError:
        pass
    except Exception as e:
        print(f"  实时预测示例失败: {e}")

    print("\n--- Python 端任务完成 ---")

if __name__ == "__main__":
    main()