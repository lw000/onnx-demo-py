"""
IGBT 温度预测模型 API 测试脚本

测试 POST 127.0.0.1:9080/predict 接口

模型输入: 5 个特征 (current, frequency, ambient_temp, temp_rate, load_factor)
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List

# API 配置
API_BASE_URL = "http://127.0.0.1:9080"
API_ENDPOINT = f"{API_BASE_URL}/predict"

# 模型配置
MODEL_CODE = 101000
MODEL_NAME = "igbt_temperature"
MODEL_CODE_NAME = "IGBT温度预测模型"

# 特征范围验证
FEATURE_RANGES = {
    "current": (10, 100),        # A - 电流
    "frequency": (10, 60),        # Hz - 频率
    "ambient_temp": (15, 40),     # °C - 环境温度
    "temp_rate": (0.1, 2.0),      # °C/min - 温升速率
    "load_factor": (0.1, 1.0)     # 负载因子 (0-1)
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

    # 检查是否有缺失的特征
    required_features = list(FEATURE_RANGES.keys())
    for feature in required_features:
        if feature not in validated:
            raise ValueError(f"缺少必需特征: {feature}")

    return validated


def build_request_body(input_data: Dict[str, float]) -> Dict[str, Any]:
    """构建请求体"""
    return {
        "model_code": MODEL_CODE,
        "input_data": input_data
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
        "current": 50.0,       # 电流 (A)
        "frequency": 30.0,     # 频率 (Hz)
        "ambient_temp": 25.0,   # 环境温度 (°C)
        "temp_rate": 0.5,       # 温升速率 (°C/min)
        "load_factor": 0.6     # 负载因子 (0-1)
    }

    print(f"\nPOST {API_ENDPOINT}")
    print(f"请求体:")
    print(json.dumps(build_request_body(test_data), indent=2, ensure_ascii=False))

    try:
        start_time = time.time()
        response = requests.post(API_ENDPOINT, json=build_request_body(test_data))
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"\n响应状态: {response.status_code}")
        print(f"响应时间: {elapsed_ms:.2f}ms")

        if response.status_code == 200:
            result = response.json()
            print(f"\n响应体:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # 检查业务状态
            if result.get("code") == 0:
                data = result.get("data", {})
                status = data.get("status", "unknown")
                predicted_temp = data.get("result", {}).get("predicted_temp", "N/A")
                print(f"\n[OK] 预测成功 - 预测温度: {predicted_temp}°C")
            else:
                status = "error"
                print(f"\n[ERROR] 业务错误: {result.get('msg', '未知错误')}")

            return result
        else:
            print(f"[ERROR] 请求失败: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] 连接失败: 无法连接到 {API_ENDPOINT}")
        print("请确保 API 服务正在运行 (python -m uvicorn main:app --host 127.0.0.1 --port 9080)")
        return None
    except Exception as e:
        print(f"[ERROR] 请求异常: {e}")
        return None


def test_batch_prediction() -> Optional[Dict[str, Any]]:
    """
    测试 2: 批量预测 (如果 API 支持)

    Returns:
        API 响应或 None
    """
    print("\n" + "="*60)
    print("测试 2: 批量预测")
    print("="*60)

    # 批量测试数据 - 覆盖正常、高负荷、低负荷、高温、低温等多种工况
    test_data_list = [
        # 正常工况样本
        {
            "current": 50.0,       # 电流 (A) - 样本1: 中等负荷正常
            "frequency": 30.0,     # 频率 (Hz)
            "ambient_temp": 25.0,  # 环境温度 (°C)
            "temp_rate": 0.5,      # 温升速率 (°C/min)
            "load_factor": 0.6     # 负载因子 (0-1)
        },
        {
            "current": 45.0,       # 电流 (A) - 样本2: 低负荷正常
            "frequency": 25.0,     # 频率 (Hz)
            "ambient_temp": 22.0,  # 环境温度 (°C)
            "temp_rate": 0.3,      # 温升速率 (°C/min)
            "load_factor": 0.5     # 负载因子 (0-1)
        },
        {
            "current": 65.0,       # 电流 (A) - 样本3: 偏高负荷
            "frequency": 40.0,     # 频率 (Hz)
            "ambient_temp": 28.0,  # 环境温度 (°C)
            "temp_rate": 0.8,      # 温升速率 (°C/min)
            "load_factor": 0.75    # 负载因子 (0-1)
        },
        # 高负荷运行样本
        {
            "current": 85.0,       # 电流 (A) - 样本4: 重负荷
            "frequency": 55.0,     # 频率 (Hz)
            "ambient_temp": 35.0,  # 环境温度 (°C)
            "temp_rate": 1.5,      # 温升速率 (°C/min)
            "load_factor": 0.95    # 负载因子 (0-1)
        },
        {
            "current": 90.0,       # 电流 (A) - 样本5: 极限负荷
            "frequency": 58.0,     # 频率 (Hz)
            "ambient_temp": 38.0,  # 环境温度 (°C)
            "temp_rate": 1.8,      # 温升速率 (°C/min)
            "load_factor": 1.0     # 负载因子 (0-1)
        },
        # 低负荷运行样本
        {
            "current": 20.0,       # 电流 (A) - 样本6: 轻负荷
            "frequency": 15.0,     # 频率 (Hz)
            "ambient_temp": 18.0,  # 环境温度 (°C)
            "temp_rate": 0.15,     # 温升速率 (°C/min)
            "load_factor": 0.2     # 负载因子 (0-1)
        },
        {
            "current": 15.0,       # 电流 (A) - 样本7: 最低负荷
            "frequency": 12.0,     # 频率 (Hz)
            "ambient_temp": 16.0,  # 环境温度 (°C)
            "temp_rate": 0.1,      # 温升速率 (°C/min)
            "load_factor": 0.15    # 负载因子 (0-1)
        },
        # 高温环境样本
        {
            "current": 70.0,       # 电流 (A) - 样本8: 高温环境
            "frequency": 45.0,     # 频率 (Hz)
            "ambient_temp": 38.0,  # 环境温度 (°C)
            "temp_rate": 1.2,      # 温升速率 (°C/min)
            "load_factor": 0.85    # 负载因子 (0-1)
        },
        {
            "current": 75.0,       # 电流 (A) - 样本9: 极端高温
            "frequency": 50.0,     # 频率 (Hz)
            "ambient_temp": 40.0,  # 环境温度 (°C)
            "temp_rate": 1.6,      # 温升速率 (°C/min)
            "load_factor": 0.9     # 负载因子 (0-1)
        },
        # 低温环境样本
        {
            "current": 30.0,       # 电流 (A) - 样本10: 低温环境
            "frequency": 20.0,     # 频率 (Hz)
            "ambient_temp": 16.0,  # 环境温度 (°C)
            "temp_rate": 0.2,      # 温升速率 (°C/min)
            "load_factor": 0.4     # 负载因子 (0-1)
        }
    ]

    # 测试逐个请求并汇总时间
    print(f"\nPOST {API_ENDPOINT} (批量测试: {len(test_data_list)} 条)")
    print(f"逐个请求测试批量性能...")

    try:
        start_time = time.time()
        results = []
        for data in test_data_list:
            response = requests.post(API_ENDPOINT, json=build_request_body(data))
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    predicted_temp = result.get("data", {}).get("result", {}).get("predicted_temp", "N/A")
                    results.append(predicted_temp)

        elapsed_ms = (time.time() - start_time) * 1000
        throughput = len(test_data_list) / (elapsed_ms / 1000)

        print(f"\n响应时间: {elapsed_ms:.2f}ms")
        print(f"吞吐量: {throughput:.2f} req/s")
        print(f"\n预测结果:")
        for i, temp in enumerate(results):
            print(f"  样本 {i+1}: {temp}°C")

        return {"predictions": results, "elapsed_ms": elapsed_ms, "throughput": throughput}

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] 连接失败: 无法连接到 {API_ENDPOINT}")
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
                "current": 50.0,       # 电流 (A)
                "frequency": 30.0,     # 频率 (Hz)
                "ambient_temp": 25.0,   # 环境温度 (°C)
                "temp_rate": 0.5,       # 温升速率 (°C/min)
                "load_factor": 0.6     # 负载因子 (0-1)
            },
            "expected_status": "success"
        },
        {
            "name": "电流超出范围",
            "data": {
                "current": 150.0,    # 电流 (A) - 超出 10-100 范围
                "frequency": 30.0,     # 频率 (Hz)
                "ambient_temp": 25.0,   # 环境温度 (°C)
                "temp_rate": 0.5,       # 温升速率 (°C/min)
                "load_factor": 0.6     # 负载因子 (0-1)
            },
            "expected_status": "error"
        },
        {
            "name": "缺少特征",
            "data": {
                "current": 50.0,
                "frequency": 30.0
                # 缺少其他特征
            },
            "expected_status": "error"
        },
        {
            "name": "未知特征",
            "data": {
                "current": 50.0,       # 电流 (A)
                "frequency": 30.0,     # 频率 (Hz)
                "ambient_temp": 25.0,   # 环境温度 (°C)
                "temp_rate": 0.5,       # 温升速率 (°C/min)
                "load_factor": 0.6,    # 负载因子 (0-1)
                "unknown_field": 1.0  # 未知字段
            },
            "expected_status": "error"
        },
        {
            "name": "负载因子超出范围",
            "data": {
                "current": 50.0,       # 电流 (A)
                "frequency": 30.0,     # 频率 (Hz)
                "ambient_temp": 25.0,   # 环境温度 (°C)
                "temp_rate": 0.5,       # 温升速率 (°C/min)
                "load_factor": 1.5    # 负载因子 (0-1) - 超出 0.1-1.0 范围
            },
            "expected_status": "error"
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n  测试 {i+1}: {test_case['name']}")
        print(f"    输入: {test_case['data']}")

        try:
            # 本地验证
            validated = validate_input(test_case['data'])
            print(f"    本地验证: [OK] 通过")

            # 尝试 API 调用
            response = requests.post(API_ENDPOINT, json=build_request_body(validated))
            if response.status_code == 200:
                result = response.json()
                actual_status = "success" if result.get("code") == 0 else "error"
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

        expected = test_case['expected_status']
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
            "current": 50.0,       # 电流 (A)
            "frequency": 30.0,     # 频率 (Hz)
            "ambient_temp": 25.0,   # 环境温度 (°C)
            "temp_rate": 0.5,       # 温升速率 (°C/min)
            "load_factor": 0.6     # 负载因子 (0-1)
        },
        "高负荷运行": {
            "current": 85.0,       # 电流 (A)
            "frequency": 55.0,     # 频率 (Hz)
            "ambient_temp": 35.0,   # 环境温度 (°C)
            "temp_rate": 1.5,       # 温升速率 (°C/min)
            "load_factor": 0.95    # 负载因子 (0-1)
        },
        "低负荷运行": {
            "current": 20.0,       # 电流 (A)
            "frequency": 15.0,     # 频率 (Hz)
            "ambient_temp": 18.0,   # 环境温度 (°C)
            "temp_rate": 0.15,      # 温升速率 (°C/min)
            "load_factor": 0.2     # 负载因子 (0-1)
        },
        "高温环境": {
            "current": 70.0,       # 电流 (A)
            "frequency": 45.0,     # 频率 (Hz)
            "ambient_temp": 38.0,   # 环境温度 (°C)
            "temp_rate": 1.2,       # 温升速率 (°C/min)
            "load_factor": 0.85     # 负载因子 (0-1)
        },
        "低温环境": {
            "current": 30.0,       # 电流 (A)
            "frequency": 20.0,     # 频率 (Hz)
            "ambient_temp": 16.0,   # 环境温度 (°C)
            "temp_rate": 0.2,       # 温升速率 (°C/min)
            "load_factor": 0.4     # 负载因子 (0-1)
        }
    }

    print(f"\nPOST {API_ENDPOINT}")
    print(f"场景预测:")

    for scenario_name, data in scenarios.items():
        try:
            response = requests.post(API_ENDPOINT, json=build_request_body(data))

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    data_obj = result.get("data", {})
                    predicted_temp = data_obj.get("result", {}).get("predicted_temp", "N/A")
                    inference_time = data_obj.get("inference_time_ms", "N/A")

                    # 温度预警
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


def test_performance() -> None:
    """
    测试 5: 性能测试
    """
    print("\n" + "="*60)
    print("测试 5: 性能测试")
    print("="*60)

    test_data = {
        "current": 50.0,       # 电流 (A)
        "frequency": 30.0,     # 频率 (Hz)
        "ambient_temp": 25.0,   # 环境温度 (°C)
        "temp_rate": 0.5,       # 温升速率 (°C/min)
        "load_factor": 0.6     # 负载因子 (0-1)
    }

    # 测试 100 次请求
    num_requests = 100
    times = []
    success_count = 0

    print(f"\n执行 {num_requests} 次请求...")

    try:
        for i in range(num_requests):
            start_time = time.time()
            response = requests.post(API_ENDPOINT, json=build_request_body(test_data))
            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    success_count += 1

            times.append(elapsed_ms)

            if (i + 1) % 20 == 0:
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

    Returns:
        True 如果服务可用, 否则 False
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        # 尝试直接发送预测请求检查服务
        try:
            test_data = {
                "current": 50.0,       # 电流 (A)
                "frequency": 30.0,     # 频率 (Hz)
                "ambient_temp": 25.0,   # 环境温度 (°C)
                "temp_rate": 0.5,       # 温升速率 (°C/min)
                "load_factor": 0.6     # 负载因子 (0-1)
            }
            response = requests.post(API_ENDPOINT, json=build_request_body(test_data), timeout=2)
            return True
        except:
            return False


def main():
    """主函数"""
    print("="*60)
    print("IGBT 温度预测模型 API 测试")
    print("="*60)
    print(f"API 地址: {API_ENDPOINT}")
    print(f"模型代码: {MODEL_CODE}")
    print(f"模型名称: {MODEL_NAME}")
    print(f"模型描述: {MODEL_CODE_NAME}")
    print(f"\n输入特征 ({len(FEATURE_RANGES)} 个):")
    for name, range_info in FEATURE_RANGES.items():
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

    # 单样本预测
    test_single_prediction()

    # 批量预测
    test_batch_prediction()

    # 输入验证
    test_validation()

    # 工况场景
    test_scenario_predictions()

    # 性能测试
    test_performance()

    print("\n" + "="*60)
    print("所有测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
