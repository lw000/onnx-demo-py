"""
IGBT 温度预测模型测试脚本

基于 ONNX 模型实现温度预测 API，支持单样本和批量预测。
模型输入: 8 个特征
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
import os
import time
from typing import Dict, List, Any, Optional

# 目录配置
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# 模型配置
MODEL_CODE = 101000
MODEL_NAME = "igbt_temperature"
MODEL_CODE_NAME = "IGBT温度预测模型"
MODEL_PATH = os.path.join(MODEL_DIR, "advanced_temp_model.onnx")

# 输入特征顺序（与模型训练时一致）
# 模型实际需要 8 个特征: temp_current, vibration_x, vibration_y, vibration_z, current, voltage, pressure, humidity
FEATURE_NAMES = ["temp_current", "vibration_x", "vibration_y", "vibration_z",
                  "current", "voltage", "pressure", "humidity"]

# 特征范围验证
FEATURE_RANGES = {
    "temp_current": (20, 80),     # °C - 当前温度
    "vibration_x": (0, 10),       # mm/s - X轴振动
    "vibration_y": (0, 10),       # mm/s - Y轴振动
    "vibration_z": (0, 10),       # mm/s - Z轴振动
    "current": (1, 15),           # A - 电流
    "voltage": (220, 240),        # V - 电压
    "pressure": (90, 110),        # kPa - 气压
    "humidity": (30, 70),         # % - 湿度
}


class IGBTemperaturePredictor:
    """IGBT 温度预测器"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径，默认使用项目中的模型
        """
        self.model_path = model_path or MODEL_PATH
        self.session = None
        self._load_model()

    def _load_model(self):
        """加载 ONNX 模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.session = ort.InferenceSession(self.model_path)
        print(f"[OK] 模型加载成功: {self.model_path}")

    def _validate_input(self, input_data: Dict[str, float]) -> Dict[str, float]:
        """
        验证输入数据

        Args:
            input_data: 输入特征字典

        Returns:
            验证后的输入数据

        Raises:
            ValueError: 输入数据无效
        """
        validated = {}
        for feature, value in input_data.items():
            if feature not in FEATURE_RANGES:
                raise ValueError(f"未知特征: {feature}")

            min_val, max_val = FEATURE_RANGES[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{feature} 值 {value} 超出范围 [{min_val}, {max_val}]")

            validated[feature] = float(value)

        # 检查是否有缺失的特征
        for feature in FEATURE_NAMES:
            if feature not in validated:
                raise ValueError(f"缺少必需特征: {feature}")

        return validated

    def _convert_to_model_input(self, input_data: Dict[str, float]) -> np.ndarray:
        """
        将输入字典转换为模型输入数组

        Args:
            input_data: 输入特征字典

        Returns:
            模型输入数组 (1, 8)
        """
        features = [input_data[name] for name in FEATURE_NAMES]
        return np.array([features], dtype=np.float32)

    def predict(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        单样本预测

        Args:
            input_data: 输入特征字典

        Returns:
            预测结果
        """
        start_time = time.time()

        # 验证输入
        validated_input = self._validate_input(input_data)

        # 转换为模型输入
        model_input = self._convert_to_model_input(validated_input)

        # 推理
        outputs = self.session.run(None, {'float_input': model_input})
        predicted_temp = float(outputs[0].flatten()[0])

        inference_time = (time.time() - start_time) * 1000

        return {
            "predicted_temp": round(predicted_temp, 2),
            "input_features": validated_input,
            "inference_time_ms": round(inference_time, 2)
        }

    def predict_batch(self, input_data_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        批量预测

        Args:
            input_data_list: 输入特征字典列表

        Returns:
            批量预测结果
        """
        start_time = time.time()

        # 验证所有输入
        validated_inputs = []
        for i, input_data in enumerate(input_data_list):
            validated = self._validate_input(input_data)
            validated_inputs.append(validated)

        # 转换为批量模型输入
        model_input = np.array([
            [validated[name] for name in FEATURE_NAMES]
            for validated in validated_inputs
        ], dtype=np.float32)

        # 批量推理
        outputs = self.session.run(None, {'float_input': model_input})
        predicted_temps = outputs[0].flatten()

        inference_time = (time.time() - start_time) * 1000

        # 构建结果
        predictions = []
        for i, temp in enumerate(predicted_temps):
            predictions.append({
                "index": i + 1,
                "predicted_temp": round(float(temp), 2)
            })

        batch_size = len(input_data_list)
        throughput = batch_size / (inference_time / 1000) if inference_time > 0 else 0

        return {
            "batch_size": batch_size,
            "predictions": predictions,
            "inference_time_ms": round(inference_time, 2),
            "throughput": round(throughput, 2)
        }


def build_response(
    status: str,
    result: Any,
    batch_size: Optional[int] = None,
    inference_time_ms: Optional[float] = None,
    throughput: Optional[float] = None
) -> Dict[str, Any]:
    """构建 API 响应"""
    response = {
        "status": status,
        "model_code": MODEL_CODE,
        "model_name": MODEL_NAME,
        "model_code_name": MODEL_CODE_NAME,
        "result": result
    }

    if batch_size is not None:
        response["batch_size"] = batch_size
    if inference_time_ms is not None:
        response["inference_time_ms"] = inference_time_ms
    if throughput is not None:
        response["throughput"] = throughput

    return response


def test_single_prediction(predictor: IGBTemperaturePredictor):
    """测试单样本预测"""
    print("\n" + "="*60)
    print("测试 1: 单样本预测")
    print("="*60)

    # 使用 8 个特征
    test_data = {
        "temp_current": 50.0,      # 当前温度 (°C)
        "vibration_x": 3.0,         # X轴振动 (mm/s)
        "vibration_y": 4.0,        # Y轴振动 (mm/s)
        "vibration_z": 2.5,         # Z轴振动 (mm/s)
        "current": 8.0,            # 电流 (A)
        "voltage": 230.0,           # 电压 (V)
        "pressure": 100.0,          # 气压 (kPa)
        "humidity": 50.0            # 湿度 (%)
    }

    print(f"\n输入数据:")
    for key, value in test_data.items():
        print(f"  {key}: {value}")

    try:
        result = predictor.predict(test_data)
        response = build_response(
            status="success",
            result=result,
            inference_time_ms=result["inference_time_ms"]
        )
        # 删除内部时间字段
        del result["inference_time_ms"]
        response["result"] = result

        print(f"\n预测结果:")
        print(f"  预测温度: {result['predicted_temp']}°C")
        print(f"\nAPI 响应:")
        print(f"  {response}")

    except Exception as e:
        response = build_response(status="error", result={"error": str(e)})
        print(f"\n预测失败: {e}")

    return response


def test_batch_prediction(predictor: IGBTemperaturePredictor):
    """测试批量预测"""
    print("\n" + "="*60)
    print("测试 2: 批量预测")
    print("="*60)

    # 使用 8 个特征
    test_data_list = [
        {
            "temp_current": 50.0,
            "vibration_x": 3.0,
            "vibration_y": 4.0,
            "vibration_z": 2.5,
            "current": 8.0,
            "voltage": 230.0,
            "pressure": 100.0,
            "humidity": 50.0
        },
        {
            "temp_current": 60.0,
            "vibration_x": 5.0,
            "vibration_y": 6.0,
            "vibration_z": 4.5,
            "current": 12.0,
            "voltage": 225.0,
            "pressure": 105.0,
            "humidity": 55.0
        },
        {
            "temp_current": 45.0,
            "vibration_x": 2.0,
            "vibration_y": 3.0,
            "vibration_z": 1.8,
            "current": 5.0,
            "voltage": 235.0,
            "pressure": 95.0,
            "humidity": 45.0
        }
    ]

    print(f"\n输入数据 ({len(test_data_list)} 条):")
    for i, data in enumerate(test_data_list):
        print(f"  样本 {i+1}: {data}")

    try:
        result = predictor.predict_batch(test_data_list)
        response = build_response(
            status="success",
            result={"predictions": result["predictions"]},
            batch_size=result["batch_size"],
            inference_time_ms=result["inference_time_ms"],
            throughput=result["throughput"]
        )

        print(f"\n预测结果:")
        for pred in result["predictions"]:
            print(f"  样本 {pred['index']}: 预测温度 = {pred['predicted_temp']}°C")
        print(f"\n统计信息:")
        print(f"  批量大小: {result['batch_size']}")
        print(f"  推理时间: {result['inference_time_ms']}ms")
        print(f"  吞吐量: {result['throughput']} samples/s")
        print(f"\nAPI 响应:")
        print(f"  {response}")

    except Exception as e:
        response = build_response(status="error", result={"error": str(e)})
        print(f"\n预测失败: {e}")

    return response


def test_validation(predictor: IGBTemperaturePredictor):
    """测试输入验证"""
    print("\n" + "="*60)
    print("测试 3: 输入验证")
    print("="*60)

    test_cases = [
        # 有效输入
        {
            "name": "正常输入",
            "data": {
                "temp_current": 50.0,
                "vibration_x": 3.0,
                "vibration_y": 4.0,
                "vibration_z": 2.5,
                "current": 8.0,
                "voltage": 230.0,
                "pressure": 100.0,
                "humidity": 50.0
            },
            "expected": "success"
        },
        # 无效输入 - 超出范围
        {
            "name": "温度超出范围",
            "data": {
                "temp_current": 150.0,  # 超出 20-80 范围
                "vibration_x": 3.0,
                "vibration_y": 4.0,
                "vibration_z": 2.5,
                "current": 8.0,
                "voltage": 230.0,
                "pressure": 100.0,
                "humidity": 50.0
            },
            "expected": "error"
        },
        # 无效输入 - 缺少特征
        {
            "name": "缺少特征",
            "data": {
                "temp_current": 50.0,
                "current": 8.0,
                "voltage": 230.0
                # 缺少其他特征
            },
            "expected": "error"
        },
        # 无效输入 - 未知特征
        {
            "name": "未知特征",
            "data": {
                "temp_current": 50.0,
                "vibration_x": 3.0,
                "vibration_y": 4.0,
                "vibration_z": 2.5,
                "current": 8.0,
                "voltage": 230.0,
                "pressure": 100.0,
                "humidity": 50.0,
                "unknown_field": 1.0  # 未知字段
            },
            "expected": "error"
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n  测试 {i+1}: {test_case['name']}")
        print(f"    输入: {test_case['data']}")

        try:
            result = predictor.predict(test_case['data'])
            status = "success"
            print(f"    结果: 预测温度 = {result['predicted_temp']}°C")
        except ValueError as e:
            status = "error"
            print(f"    结果: 验证错误 - {e}")
        except Exception as e:
            status = "error"
            print(f"    结果: 错误 - {e}")

        expected = test_case['expected']
        if status == expected:
            print(f"    [OK] 符合预期")
        else:
            print(f"    [FAIL] 预期 {expected}, 实际 {status}")


def test_edge_cases(predictor: IGBTemperaturePredictor):
    """测试边界情况"""
    print("\n" + "="*60)
    print("测试 4: 边界情况")
    print("="*60)

    test_cases = [
        # 最小值
        {
            "name": "最小值输入",
            "data": {
                "temp_current": 20.0,
                "vibration_x": 0.0,
                "vibration_y": 0.0,
                "vibration_z": 0.0,
                "current": 1.0,
                "voltage": 220.0,
                "pressure": 90.0,
                "humidity": 30.0
            }
        },
        # 最大值
        {
            "name": "最大值输入",
            "data": {
                "temp_current": 80.0,
                "vibration_x": 10.0,
                "vibration_y": 10.0,
                "vibration_z": 10.0,
                "current": 15.0,
                "voltage": 240.0,
                "pressure": 110.0,
                "humidity": 70.0
            }
        },
        # 典型工业值
        {
            "name": "典型工业值",
            "data": {
                "temp_current": 50.0,
                "vibration_x": 5.0,
                "vibration_y": 5.0,
                "vibration_z": 4.0,
                "current": 10.0,
                "voltage": 230.0,
                "pressure": 100.0,
                "humidity": 50.0
            }
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n  测试 {i+1}: {test_case['name']}")
        print(f"    输入: {test_case['data']}")

        try:
            result = predictor.predict(test_case['data'])
            print(f"    结果: 预测温度 = {result['predicted_temp']}°C")
        except Exception as e:
            print(f"    错误: {e}")


def test_scenario_predictions(predictor: IGBTemperaturePredictor):
    """测试不同工况场景"""
    print("\n" + "="*60)
    print("测试 5: 不同工况场景")
    print("="*60)

    scenarios = {
        "正常工况": {
            "temp_current": 45.0,
            "vibration_x": 2.0,
            "vibration_y": 2.5,
            "vibration_z": 1.8,
            "current": 5.0,
            "voltage": 230.0,
            "pressure": 100.0,
            "humidity": 50.0
        },
        "高负荷": {
            "temp_current": 65.0,
            "vibration_x": 6.0,
            "vibration_y": 7.0,
            "vibration_z": 5.5,
            "current": 13.0,
            "voltage": 225.0,
            "pressure": 105.0,
            "humidity": 60.0
        },
        "低负荷": {
            "temp_current": 35.0,
            "vibration_x": 1.0,
            "vibration_y": 1.5,
            "vibration_z": 1.0,
            "current": 2.0,
            "voltage": 235.0,
            "pressure": 95.0,
            "humidity": 40.0
        },
        "振动异常": {
            "temp_current": 55.0,
            "vibration_x": 8.0,
            "vibration_y": 9.0,
            "vibration_z": 7.5,
            "current": 10.0,
            "voltage": 228.0,
            "pressure": 102.0,
            "humidity": 55.0
        }
    }

    print(f"\n工况预测:")
    for scenario_name, data in scenarios.items():
        try:
            result = predictor.predict(data)
            predicted_temp = result['predicted_temp']

            # 温度预警
            if predicted_temp > 70:
                alert = "[ALERT] 预警: 温度过高！"
            elif predicted_temp > 50:
                alert = "[INFO] 提示: 温度偏高"
            else:
                alert = "[OK] 温度正常"

            print(f"\n  {scenario_name}:")
            print(f"    预测温度: {predicted_temp}°C")
            print(f"    {alert}")
        except Exception as e:
            print(f"\n  {scenario_name}: 错误 - {e}")


def main():
    """主函数"""
    print("="*60)
    print("IGBT 温度预测模型测试")
    print("="*60)
    print(f"模型代码: {MODEL_CODE}")
    print(f"模型名称: {MODEL_NAME}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"\n输入特征 ({len(FEATURE_NAMES)} 个):")
    for name in FEATURE_NAMES:
        range_info = FEATURE_RANGES[name]
        print(f"  - {name}: {range_info}")

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] 模型文件不存在: {MODEL_PATH}")
        print("请先运行 scripts/advanced_temp_model.py 生成模型")
        return

    # 创建预测器
    try:
        predictor = IGBTemperaturePredictor(MODEL_PATH)
    except Exception as e:
        print(f"\n[ERROR] 初始化预测器失败: {e}")
        return

    # 运行测试
    test_single_prediction(predictor)
    test_batch_prediction(predictor)
    test_validation(predictor)
    test_edge_cases(predictor)
    test_scenario_predictions(predictor)

    print("\n" + "="*60)
    print("所有测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
