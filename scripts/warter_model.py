import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os
warnings.filterwarnings("ignore")


# 目录配置
base_dir = os.path.dirname(os.path.dirname(__file__))
model_dir = os.path.join(base_dir, "models")
samples_dir = os.path.join(base_dir, "data")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)


# --- 1. 数据准备 (Simulate Water Level Data) ---
# 模拟一段时间内的水文测点值，例如水位高度
np.random.seed(42)
time_steps = 1000
# 模拟一个带有趋势和噪声的基础水位序列
base_level = 100.0
trend = np.linspace(0, 5, time_steps) # 慢慢上涨的趋势
seasonality = 3 * np.sin(np.arange(time_steps) * 2 * np.pi / 50) # 周期性变化
noise = np.random.normal(0, 1, time_steps) # 随机噪声
water_levels = base_level + trend + seasonality + noise

df = pd.DataFrame({'timestamp': pd.date_range(start='2023-01-01', periods=time_steps, freq='h'),
                   'water_level': water_levels})

print(f"Generated synthetic water level data shape: {df.shape}")
print(df.head())

# --- 2. 特征工程 (Feature Engineering with Sliding Window) ---
def create_features_and_target(data, window_size):
    """
    Creates features and target for supervised learning using a sliding window.
    Each sample will have `window_size` previous values as features to predict the next value.
    
    Args:
        data (array-like): The input time series data.
        window_size (int): Number of past time steps to use as features.

    Returns:
        tuple: (features, targets) as numpy arrays.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        # Features: previous `window_size` values
        X.append(data[i:(i + window_size)])
        # Target: the next value after the window
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Define the window size (e.g., use last 10 hours to predict the next hour)
WINDOW_SIZE = 10

X, y = create_features_and_target(df['water_level'].values, window_size=WINDOW_SIZE)

if len(X) == 0 or len(y) == 0:
    raise ValueError("Not enough data points after creating features with the given window size.")

print(f"Features shape after sliding window (X): {X.shape}")
print(f"Target shape after sliding window (y): {y.shape}")

# --- 3. 数据分割 (Train-Test Split) ---
# It's crucial to maintain temporal order for time series data
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training set shape (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Test set shape (X_test, y_test): {X_test.shape}, {y_test.shape}")

# 保存训练数据到 samples_dir
train_df = pd.DataFrame(X_train, columns=[f'lag_{i+1}' for i in range(WINDOW_SIZE)])
train_df['target'] = y_train
train_csv_path = os.path.join(samples_dir, "water_level_train.csv")
train_df.to_csv(train_csv_path, index=False)
print(f"训练数据已保存: {train_csv_path}")

test_df = pd.DataFrame(X_test, columns=[f'lag_{i+1}' for i in range(WINDOW_SIZE)])
test_df['target'] = y_test
test_csv_path = os.path.join(samples_dir, "water_level_test.csv")
test_df.to_csv(test_csv_path, index=False)
print(f"测试数据已保存: {test_csv_path}")


# --- 4. 模型训练 (Model Training with Sklearn) ---
# Choose a regressor from sklearn
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred_sklearn = model.predict(X_test)

# Evaluate the sklearn model
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print("\n--- Sklearn Model Evaluation ---")
print(f"Mean Squared Error (Sklearn): {mse_sklearn:.4f}")
print(f"R^2 Score (Sklearn): {r2_sklearn:.4f}")


# --- 5. 导出模型为 ONNX 格式 ---
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

    # Convert the sklearn model to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model to a file
    onnx_filename = os.path.join(model_dir, "water_level_prediction_model.onnx")
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"\n--- Model Exported ---")
    print(f"ONNX model saved as: {onnx_filename}")

except ImportError:
    print("\n--- ONNX Export Failed ---")
    print("The 'skl2onnx' package is not installed. Please install it using 'pip install skl2onnx'.")


# --- 6. 验证 ONNX 模型 (Validate ONNX Model with ONNX Runtime) ---
try:
    import onnxruntime as rt

    # Load the ONNX model
    sess = rt.InferenceSession(onnx_filename, providers=['CPUExecutionProvider']) # Use CPU provider

    # Get the name of the input
    input_name = sess.get_inputs()[0].name
    print(f"ONNX Model Input Name: {input_name}")

    # Run inference using ONNX Runtime
    # Ensure the input data type is float32
    y_pred_onnx = sess.run(None, {input_name: X_test.astype(np.float32)})[0]

    # Compare predictions from sklearn and ONNX Runtime
    # They should be very close
    mse_comparison = mean_squared_error(y_pred_sklearn, y_pred_onnx.flatten()) # Flatten if needed

    print("\n--- ONNX Model Validation ---")
    print(f"MSE between Sklearn and ONNX predictions: {mse_comparison:.8f} (should be ~0.0)")
    
    if np.allclose(y_pred_sklearn, y_pred_onnx.flatten(), atol=1e-5):
        print("SUCCESS: Sklearn and ONNX predictions match closely!")
    else:
        print("WARNING: Sklearn and ONNX predictions differ significantly!")

    # Example prediction with a single sample for C++ usage context
    sample_input = X_test[0:1].astype(np.float32) # Shape: (1, WINDOW_SIZE)
    sample_prediction_sklearn = model.predict(sample_input)
    sample_prediction_onnx = sess.run(None, {input_name: sample_input})[0][0] # Extract scalar

    print(f"\nExample Single Prediction:")
    print(f"Input (last {WINDOW_SIZE} levels): {sample_input.flatten()}")
    print(f"Sklearn Prediction: {sample_prediction_sklearn[0]:.4f}")
    print(f"ONNX Runtime Prediction: {sample_prediction_onnx[0]:.4f}")

except ImportError:
    print("\n--- ONNX Runtime Validation Failed ---")
    print("The 'onnxruntime' package is not installed. Please install it using 'pip install onnxruntime'.")