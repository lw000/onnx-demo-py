"""
高级温度预测模型 API 测试脚本

测试接口:
- POST 127.0.0.1:9080/predict (单样本预测)
- POST 127.0.0.1:9080/predict/batch (批量预测)

模型输入: 8 个特征
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List

# API 配置
API_BASE_URL = "http://127.0.0.1:9080"
API_ENDPOINT = f"{API_BASE_URL}/predict"
API_BATCH_ENDPOINT = f"{API_BASE_URL}/predict/batch"

# 模型配置
MODEL_CODE = 103000
MODEL_NAME = "advanced_temperature"
MODEL_CODE_NAME = "高级温度模型"

# 输入特征顺序
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


def validate_input(input_data: Dict[str, float]) -> Dict[str, float]:
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

    for feature in FEATURE_NAMES:
        if feature not in validated:
            raise ValueError(f"缺少必需特征: {feature}")

    return validated


def build_single_request(input_data: Dict[str, float]) -> Dict[str, Any]:
    """构建单样本请求体"""
    return {
        "model_code": MODEL_CODE,
        "input_data": input_data
    }


def build_batch_request(input_data_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """构建批量请求体"""
    return {
        "model_code": MODEL_CODE,
        "input_data": input_data_list
    }


def test_single_prediction() -> Optional[Dict[str, Any]]:
    """
    测试 1: 单样本预测

    Returns:
        API 响应或 None
    """
    print("\n" + "="*60)
    print("测试 1: 单样本预测")
    print("="*60)

    test_data = {
        "temp_current": 50.0,   # 当前温度 (°C)
        "vibration_x": 5.0,     # X轴振动 (mm/s)
        "vibration_y": 4.5,      # Y轴振动 (mm/s)
        "vibration_z": 3.8,      # Z轴振动 (mm/s)
        "current": 8.0,          # 电流 (A)
        "voltage": 230.0,        # 电压 (V)
        "pressure": 100.0,       # 气压 (kPa)
        "humidity": 50.0         # 湿度 (%)
    }

    print(f"\nPOST {API_ENDPOINT}")
    print(f"请求体:")
    print(json.dumps(build_single_request(test_data), indent=2, ensure_ascii=False))

    try:
        start_time = time.time()
        response = requests.post(API_ENDPOINT, json=build_single_request(test_data))
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"\n响应状态: {response.status_code}")
        print(f"响应时间: {elapsed_ms:.2f}ms")

        if response.status_code == 200:
            result = response.json()
            print(f"\n响应体:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # 适配 code/data/msg 响应格式
            code = result.get("code", -1)
            data = result.get("data", {})
            predicted_temp = data.get("result", {}).get("predicted_temperature", "N/A")
            inference_time = data.get("inference_time_ms", "N/A")

            if code == 0 and predicted_temp != "N/A":
                print(f"\n[OK] 预测成功 - 预测温度: {predicted_temp}°C, 推理时间: {inference_time}ms")
                return result
            else:
                print(f"\n[ERROR] 预测失败: {result.get('msg', '未知错误')}")
                return result
        else:
            print(f"[ERROR] 请求失败: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] 连接失败: 无法连接到 {API_ENDPOINT}")
        return None
    except Exception as e:
        print(f"[ERROR] 请求异常: {e}")
        return None


def test_batch_prediction() -> Optional[Dict[str, Any]]:
    """
    测试 2: 批量预测

    Returns:
        API 响应或 None
    """
    print("\n" + "="*60)
    print("测试 2: 批量预测")
    print("="*60)

    test_data_list = [
        # 样本 1
        {
            "temp_current": 50.0,   # 当前温度 (°C)
            "vibration_x": 5.0,     # X轴振动 (mm/s)
            "vibration_y": 4.5,      # Y轴振动 (mm/s)
            "vibration_z": 3.8,      # Z轴振动 (mm/s)
            "current": 8.0,          # 电流 (A)
            "voltage": 230.0,        # 电压 (V)
            "pressure": 100.0,       # 气压 (kPa)
            "humidity": 50.0         # 湿度 (%)
        },
        # 样本 2
        {
            "temp_current": 55.0,
            "vibration_x": 6.0,
            "vibration_y": 5.2,
            "vibration_z": 4.1,
            "current": 9.0,
            "voltage": 235.0,
            "pressure": 102.0,
            "humidity": 55.0
        }
    ]

    print(f"\nPOST {API_BATCH_ENDPOINT}")
    print(f"请求体 ({len(test_data_list)} 条数据):")
    print(json.dumps(build_batch_request(test_data_list), indent=2, ensure_ascii=False))

    try:
        start_time = time.time()
        response = requests.post(API_BATCH_ENDPOINT, json=build_batch_request(test_data_list))
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"\n响应状态: {response.status_code}")
        print(f"响应时间: {elapsed_ms:.2f}ms")

        if response.status_code == 200:
            result = response.json()
            print(f"\n响应体:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # 适配 code/data/msg 响应格式
            code = result.get("code", -1)
            data = result.get("data", {})
            predictions = data.get("result", {}).get("predictions", [])

            if code == 0 and predictions:
                batch_size = len(predictions)
                print(f"\n[OK] 批量预测成功")
                print(f"  批量大小: {batch_size}")
                print(f"  预测结果:")
                for pred in predictions:
                    print(f"    样本 {pred['index']}: {pred['predicted_temperature']}°C")
                return result
            else:
                print(f"\n[ERROR] 批量预测失败: {result.get('msg', '未知错误')}")
                return result
        else:
            print(f"[ERROR] 请求失败: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] 连接失败: 无法连接到 {API_BATCH_ENDPOINT}")
        return None
    except Exception as e:
        print(f"[ERROR] 请求异常: {e}")
        return None


def test_validation() -> None:
    """
    测试 3: 输入验证
    """
    print("\n" + "="*60)
    print("测试 3: 输入验证")
    print("="*60)

    test_cases = [
        {
            "name": "正常输入",
            "data": {
                "temp_current": 50.0,   # 当前温度 (°C)
                "vibration_x": 5.0,     # X轴振动 (mm/s)
                "vibration_y": 4.5,      # Y轴振动 (mm/s)
                "vibration_z": 3.8,      # Z轴振动 (mm/s)
                "current": 8.0,          # 电流 (A)
                "voltage": 230.0,        # 电压 (V)
                "pressure": 100.0,       # 气压 (kPa)
                "humidity": 50.0          # 湿度 (%)
            },
            "expected": "success"
        },
        {
            "name": "温度超出范围",
            "data": {
                "temp_current": 150.0,  # 当前温度 (°C) - 超出 20-80 范围
                "vibration_x": 5.0,     # X轴振动 (mm/s)
                "vibration_y": 4.5,      # Y轴振动 (mm/s)
                "vibration_z": 3.8,      # Z轴振动 (mm/s)
                "current": 8.0,          # 电流 (A)
                "voltage": 230.0,        # 电压 (V)
                "pressure": 100.0,       # 气压 (kPa)
                "humidity": 50.0         # 湿度 (%)
            },
            "expected": "error"
        },
        {
            "name": "缺少特征",
            "data": {
                "temp_current": 50.0,
                "vibration_x": 5.0,
                "current": 8.0
                # 缺少其他特征
            },
            "expected": "error"
        },
        {
            "name": "振动值异常",
            "data": {
                "temp_current": 50.0,   # 当前温度 (°C)
                "vibration_x": 15.0,   # X轴振动 (mm/s) - 超出 0-10 范围
                "vibration_y": 4.5,     # Y轴振动 (mm/s)
                "vibration_z": 3.8,     # Z轴振动 (mm/s)
                "current": 8.0,         # 电流 (A)
                "voltage": 230.0,       # 电压 (V)
                "pressure": 100.0,      # 气压 (kPa)
                "humidity": 50.0        # 湿度 (%)
            },
            "expected": "error"
        },
        {
            "name": "电压异常",
            "data": {
                "temp_current": 50.0,   # 当前温度 (°C)
                "vibration_x": 5.0,     # X轴振动 (mm/s)
                "vibration_y": 4.5,      # Y轴振动 (mm/s)
                "vibration_z": 3.8,      # Z轴振动 (mm/s)
                "current": 8.0,          # 电流 (A)
                "voltage": 300.0,      # 电压 (V) - 超出 220-240 范围
                "pressure": 100.0,       # 气压 (kPa)
                "humidity": 50.0         # 湿度 (%)
            },
            "expected": "error"
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n  测试 {i+1}: {test_case['name']}")
        print(f"    输入: {test_case['data']}")

        try:
            validated = validate_input(test_case['data'])
            print(f"    本地验证: [OK] 通过")

            response = requests.post(API_ENDPOINT, json=build_single_request(validated))
            if response.status_code == 200:
                result = response.json()
                code = result.get("code", -1)
                data = result.get("data", {})
                # API返回code=0表示成功，无需检查status字段
                actual_status = "success" if code == 0 else "error"
            else:
                actual_status = "error"

            print(f"    API 状态: {actual_status}")

        except ValueError as e:
            print(f"    本地验证: [ERROR] {e}")
            actual_status = "error"
        except requests.exceptions.ConnectionError:
            print(f"    [ERROR] 无法连接到 API 服务")
            continue
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        expected = test_case['expected']
        if actual_status == expected:
            print(f"    [OK] 符合预期")
        else:
            print(f"    [FAIL] 预期 {expected}, 实际 {actual_status}")


def test_scenario_predictions() -> None:
    """
    测试 4: 不同工况场景
    """
    print("\n" + "="*60)
    print("测试 4: 不同工况场景")
    print("="*60)

    scenarios = {
        "正常工况": {
            "temp_current": 50.0,   # 当前温度 (°C)
            "vibration_x": 5.0,     # X轴振动 (mm/s)
            "vibration_y": 4.5,      # Y轴振动 (mm/s)
            "vibration_z": 3.8,      # Z轴振动 (mm/s)
            "current": 8.0,          # 电流 (A)
            "voltage": 230.0,        # 电压 (V)
            "pressure": 100.0,       # 气压 (kPa)
            "humidity": 50.0         # 湿度 (%)
        },
        "高负荷运行": {
            "temp_current": 65.0,   # 当前温度 (°C)
            "vibration_x": 8.0,     # X轴振动 (mm/s)
            "vibration_y": 7.5,      # Y轴振动 (mm/s)
            "vibration_z": 6.5,      # Z轴振动 (mm/s)
            "current": 13.0,         # 电流 (A)
            "voltage": 225.0,        # 电压 (V)
            "pressure": 105.0,       # 气压 (kPa)
            "humidity": 60.0         # 湿度 (%)
        },
        "低负荷运行": {
            "temp_current": 35.0,   # 当前温度 (°C)
            "vibration_x": 2.0,     # X轴振动 (mm/s)
            "vibration_y": 1.8,      # Y轴振动 (mm/s)
            "vibration_z": 1.5,      # Z轴振动 (mm/s)
            "current": 3.0,          # 电流 (A)
            "voltage": 235.0,        # 电压 (V)
            "pressure": 95.0,        # 气压 (kPa)
            "humidity": 40.0         # 湿度 (%)
        },
        "振动异常": {
            "temp_current": 55.0,   # 当前温度 (°C)
            "vibration_x": 9.5,    # X轴振动 (mm/s)
            "vibration_y": 9.0,      # Y轴振动 (mm/s)
            "vibration_z": 8.5,      # Z轴振动 (mm/s)
            "current": 10.0,         # 电流 (A)
            "voltage": 228.0,         # 电压 (V)
            "pressure": 102.0,       # 气压 (kPa)
            "humidity": 55.0         # 湿度 (%)
        },
        "温升较快": {
            "temp_current": 70.0,   # 当前温度 (°C)
            "vibration_x": 6.0,     # X轴振动 (mm/s)
            "vibration_y": 5.5,      # Y轴振动 (mm/s)
            "vibration_z": 5.0,       # Z轴振动 (mm/s)
            "current": 12.0,         # 电流 (A)
            "voltage": 232.0,        # 电压 (V)
            "pressure": 103.0,       # 气压 (kPa)
            "humidity": 58.0         # 湿度 (%)
        }
    }

    print(f"\nPOST {API_ENDPOINT}")
    print(f"场景预测:")

    for scenario_name, scenario_data in scenarios.items():
        try:
            response = requests.post(API_ENDPOINT, json=build_single_request(scenario_data))

            if response.status_code == 200:
                result = response.json()
                code = result.get("code", -1)
                data_obj = result.get("data", {})

                if code == 0:
                    predicted_temp = data_obj.get("result", {}).get("predicted_temperature", "N/A")
                    inference_time = data_obj.get("inference_time_ms", "N/A")

                    if predicted_temp != "N/A":
                        if predicted_temp > 75:
                            alert = "[ALERT] 预警: 温度过高!"
                        elif predicted_temp > 60:
                            alert = "[WARNING] 提示: 温度偏高"
                        else:
                            alert = "[OK] 温度正常"

                        print(f"\n  {scenario_name}:")
                        print(f"    预测温度: {predicted_temp}°C")
                        print(f"    推理时间: {inference_time}ms")
                        print(f"    {alert}")
                    else:
                        print(f"\n  {scenario_name}: [ERROR] 响应格式异常")
                else:
                    print(f"\n  {scenario_name}: [ERROR] {result.get('msg', '未知错误')}")
            else:
                print(f"\n  {scenario_name}: [ERROR] HTTP {response.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"\n  {scenario_name}: [ERROR] 无法连接到 API 服务")
            break
        except Exception as e:
            print(f"\n  {scenario_name}: [ERROR] {e}")


def test_batch_performance() -> None:
    """
    测试 5: 批量性能测试
    """
    print("\n" + "="*60)
    print("测试 5: 批量性能测试")
    print("="*60)

    # 生成不同大小的批量数据
    batch_sizes = [5, 10, 20]

    for batch_size in batch_sizes:
        test_data_list = []
        for i in range(batch_size):
            test_data_list.append({
                "temp_current": 40.0 + i % 20,
                "vibration_x": 3.0 + (i % 5),
                "vibration_y": 2.5 + (i % 5),
                "vibration_z": 2.0 + (i % 4),
                "current": 5.0 + (i % 8),
                "voltage": 225.0 + (i % 10),
                "pressure": 95.0 + (i % 10),
                "humidity": 40.0 + (i % 20)
            })

        print(f"\n  批量大小: {batch_size}")
        print(f"  POST {API_BATCH_ENDPOINT}")

        try:
            start_time = time.time()
            response = requests.post(API_BATCH_ENDPOINT, json=build_batch_request(test_data_list))
            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                code = result.get("code", -1)
                data = result.get("data", {})

                if code == 0:
                    predictions = data.get("result", {}).get("predictions", [])
                    api_inference_time = data.get("inference_time_ms", 0)
                    throughput = batch_size / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
                    print(f"    响应时间: {elapsed_ms:.2f}ms")
                    print(f"    吞吐量: {throughput:.2f} samples/s")
                    print(f"    成功预测: {len(predictions)}/{batch_size}")
                else:
                    print(f"    [ERROR] {result.get('msg', '未知错误')}")
            else:
                print(f"    [ERROR] HTTP {response.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"    [ERROR] 无法连接到 API 服务")
            break
        except Exception as e:
            print(f"    [ERROR] {e}")


def test_high_concurrency() -> None:
    """
    测试 6: 高并发测试 (串行模拟)
    """
    print("\n" + "="*60)
    print("测试 6: 高并发测试 (100 次连续请求)")
    print("="*60)

    test_data = {
        "temp_current": 50.0,   # 当前温度 (°C)
        "vibration_x": 5.0,     # X轴振动 (mm/s)
        "vibration_y": 4.5,      # Y轴振动 (mm/s)
        "vibration_z": 3.8,      # Z轴振动 (mm/s)
        "current": 8.0,          # 电流 (A)
        "voltage": 230.0,        # 电压 (V)
        "pressure": 100.0,       # 气压 (kPa)
        "humidity": 50.0         # 湿度 (%)
    }

    num_requests = 100
    times = []
    success_count = 0

    print(f"\n执行 {num_requests} 次请求...")

    try:
        for i in range(num_requests):
            start_time = time.time()
            response = requests.post(API_ENDPOINT, json=build_single_request(test_data))
            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                code = result.get("code", -1)
                if code == 0:
                    success_count += 1

            times.append(elapsed_ms)

            if (i + 1) % 25 == 0:
                print(f"  已完成: {i + 1}/{num_requests}")

        # 统计
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput = success_count / (sum(times) / 1000) if sum(times) > 0 else 0

        print(f"\n性能统计:")
        print(f"  总请求数: {num_requests}")
        print(f"  成功请求: {success_count}")
        print(f"  平均响应时间: {avg_time:.2f}ms")
        print(f"  最短响应时间: {min_time:.2f}ms")
        print(f"  最长响应时间: {max_time:.2f}ms")
        print(f"  吞吐量: {throughput:.2f} req/s")

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] 无法连接到 API 服务")
    except Exception as e:
        print(f"[ERROR] {e}")


def check_api_health() -> bool:
    """
    检查 API 服务是否可用
    """
    try:
        test_data = {
            "temp_current": 50.0,   # 当前温度 (°C)
            "vibration_x": 5.0,     # X轴振动 (mm/s)
            "vibration_y": 4.5,      # Y轴振动 (mm/s)
            "vibration_z": 3.8,      # Z轴振动 (mm/s)
            "current": 8.0,          # 电流 (A)
            "voltage": 230.0,        # 电压 (V)
            "pressure": 100.0,       # 气压 (kPa)
            "humidity": 50.0         # 湿度 (%)
        }
        response = requests.post(API_ENDPOINT, json=build_single_request(test_data), timeout=2)
        return True
    except:
        return False


def main():
    """主函数"""
    print("="*60)
    print("高级温度预测模型 API 测试")
    print("="*60)
    print(f"单样本接口: {API_ENDPOINT}")
    print(f"批量接口: {API_BATCH_ENDPOINT}")
    print(f"模型代码: {MODEL_CODE}")
    print(f"模型名称: {MODEL_NAME}")
    print(f"模型描述: {MODEL_CODE_NAME}")
    print(f"\n输入特征 ({len(FEATURE_NAMES)} 个):")
    for name in FEATURE_NAMES:
        range_info = FEATURE_RANGES[name]
        print(f"  - {name}: {range_info}")

    # 检查 API 服务
    print("\n检查 API 服务...")
    if not check_api_health():
        print(f"[WARNING] 无法连接到 {API_BASE_URL}")
        print("请确保 API 服务正在运行:")
        print("  python -m uvicorn main:app --host 127.0.0.1 --port 9080")
        print("\n继续执行本地验证测试...")
    else:
        print("[OK] API 服务正常")

    # 运行测试
    print("\n" + "="*60)
    print("开始测试")
    print("="*60)

    test_single_prediction()
    test_batch_prediction()
    test_validation()
    test_scenario_predictions()
    test_batch_performance()
    test_high_concurrency()

    print("\n" + "="*60)
    print("所有测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
