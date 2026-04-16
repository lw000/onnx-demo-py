"""
皮带机打滑预测模型 API 测试脚本
模型代码: 203000
"""
import requests
import time
from typing import Dict, List, Any


BASE_URL = "http://127.0.0.1:9080"
TIMEOUT = 10


def check_server():
    """检查服务器是否可用"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def test_single_prediction():
    """测试单样本预测"""
    print("\n" + "=" * 60)
    print("测试1: 单样本预测")
    print("=" * 60)
    
    # 测试数据 - 皮带机打滑预测参数
    test_data = {
        "model_code": 203000,
        "input_data": {
            "current": 120.0,         # 电流 (A)
            "speed_diff": 0.5,        # 速度差 (m/s)
            "vibration": 3.2,          # 振动值 (mm/s)
            "temperature": 45.0       # 温度 (°C)
        }
    }
    
    print(f"请求数据: {test_data}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        result = response.json()
        print(f"响应状态: {response.status_code}")
        print(f"响应内容: {result}")
        
        # 解析响应
        status = result.get("status", "unknown")
        if status == "success":
            data = result.get("result", {})
            is_slip = data.get("is_slip", None)
            probabilities = data.get("probabilities", {})
            print(f"\n✓ 预测成功!")
            print(f"  是否打滑: {'是' if is_slip else '否'}")
            print(f"  各类别概率:")
            for cls, prob in probabilities.items():
                print(f"    - {cls}: {prob:.4f} ({prob*100:.2f}%)")
            return True
        else:
            print(f"\n✗ 预测失败: {result.get('message', '未知错误')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")
        return False


def test_batch_prediction():
    """测试批量预测"""
    print("\n" + "=" * 60)
    print("测试2: 批量预测")
    print("=" * 60)
    
    # 批量测试数据
    batch_data = {
        "model_code": 203000,
        "input_data": [
            {
                "current": 120.0,         # 电流 (A) - 样本1: 正常工况
                "speed_diff": 0.5,         # 速度差 (m/s)
                "vibration": 3.2,          # 振动值 (mm/s)
                "temperature": 45.0        # 温度 (°C)
            },
            {
                "current": 180.0,         # 电流 (A) - 样本2: 打滑工况
                "speed_diff": 2.8,         # 速度差 (m/s)
                "vibration": 7.5,          # 振动值 (mm/s)
                "temperature": 68.0        # 温度 (°C)
            },
            {
                "current": 150.0,         # 电流 (A) - 样本3: 边缘工况
                "speed_diff": 1.5,         # 速度差 (m/s)
                "vibration": 4.5,          # 振动值 (mm/s)
                "temperature": 55.0        # 温度 (°C)
            }
        ]
    }
    
    print(f"批量大小: {len(batch_data['input_data'])}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        result = response.json()
        print(f"响应状态: {response.status_code}")
        print(f"响应内容: {result}")
        
        status = result.get("status", "unknown")
        if status == "success":
            data = result.get("result", {})
            predictions = data.get("predictions", [])
            print(f"\n✓ 批量预测成功! 共 {len(predictions)} 条结果")
            for pred in predictions:
                idx = pred.get("index", "?")
                is_slip = pred.get("is_slip", None)
                probs = pred.get("probabilities", {})
                slip_prob = probs.get("slip", 0)
                print(f"\n  样本{idx}:")
                print(f"    是否打滑: {'是' if is_slip else '否'} (打滑概率: {slip_prob*100:.2f}%)")
                print(f"    概率分布: ", end="")
                print(", ".join([f"{k}={v:.3f}" for k, v in probs.items()]))
            print(f"\n性能指标:")
            print(f"  推理时间: {result.get('inference_time_ms', 'N/A')} ms")
            print(f"  吞吐量: {result.get('throughput', 'N/A')} req/s")
            return True
        else:
            print(f"\n✗ 批量预测失败: {result.get('message', '未知错误')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")
        return False


def test_input_validation():
    """测试输入参数验证"""
    print("\n" + "=" * 60)
    print("测试3: 输入参数验证")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "缺失参数",
            "data": {
                "model_code": 203000,
                "input_data": {
                    "current": 120.0,     # 电流 (A)
                    "speed_diff": 0.5      # 速度差 (m/s)
                }
            }
        },
        {
            "name": "参数类型错误",
            "data": {
                "model_code": 203000,
                "input_data": {
                    "current": "high",    # 电流 (A) - 类型错误
                    "speed_diff": 0.5,    # 速度差 (m/s)
                    "vibration": 3.2,     # 振动值 (mm/s)
                    "temperature": 45.0  # 温度 (°C)
                }
            }
        },
        {
            "name": "参数越界(负电流)",
            "data": {
                "model_code": 203000,
                "input_data": {
                    "current": -10.0,    # 电流 (A) - 越界
                    "speed_diff": 0.5,    # 速度差 (m/s)
                    "vibration": 3.2,    # 振动值 (mm/s)
                    "temperature": 45.0  # 温度 (°C)
                }
            }
        },
        {
            "name": "参数越界(速度差过大)",
            "data": {
                "model_code": 203000,
                "input_data": {
                    "current": 120.0,     # 电流 (A)
                    "speed_diff": 10.0,   # 速度差 (m/s) - 越界
                    "vibration": 3.2,     # 振动值 (mm/s)
                    "temperature": 45.0  # 温度 (°C)
                }
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {case['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=case["data"],
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            result = response.json()
            status = result.get("status", "unknown")
            if status == "success":
                print(f"  ⚠ 未检测到异常 (可能需要后端验证)")
            else:
                print(f"  ✓ 正确拒绝: {result.get('message', result)}")
        except requests.exceptions.RequestException as e:
            print(f"  ✗ 请求异常: {e}")
    
    return True


def test_slip_scenarios():
    """测试不同打滑风险场景"""
    print("\n" + "=" * 60)
    print("测试4: 打滑风险场景测试")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "理想工况",
            "data": {
                "current": 100.0,      # 电流 (A)
                "speed_diff": 0.2,     # 速度差 (m/s)
                "vibration": 2.0,      # 振动值 (mm/s)
                "temperature": 35.0    # 温度 (°C)
            }
        },
        {
            "name": "正常工况",
            "data": {
                "current": 120.0,      # 电流 (A)
                "speed_diff": 0.5,     # 速度差 (m/s)
                "vibration": 3.2,     # 振动值 (mm/s)
                "temperature": 45.0    # 温度 (°C)
            }
        },
        {
            "name": "边缘风险",
            "data": {
                "current": 160.0,      # 电流 (A)
                "speed_diff": 1.8,     # 速度差 (m/s)
                "vibration": 5.0,     # 振动值 (mm/s)
                "temperature": 58.0    # 温度 (°C)
            }
        },
        {
            "name": "高风险打滑",
            "data": {
                "current": 200.0,      # 电流 (A)
                "speed_diff": 3.5,     # 速度差 (m/s)
                "vibration": 8.0,      # 振动值 (mm/s)
                "temperature": 75.0    # 温度 (°C)
            }
        },
        {
            "name": "严重打滑",
            "data": {
                "current": 220.0,      # 电流 (A)
                "speed_diff": 5.0,     # 速度差 (m/s)
                "vibration": 10.0,     # 振动值 (mm/s)
                "temperature": 85.0    # 温度 (°C)
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        request_data = {
            "model_code": 203000,
            "input_data": scenario["data"]
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            result = response.json()
            status = result.get("status", "unknown")
            
            if status == "success":
                data = result.get("result", {})
                is_slip = data.get("is_slip", None)
                probs = data.get("probabilities", {})
                slip_prob = probs.get("slip", 0)
                risk_level = "低" if slip_prob < 0.2 else ("中" if slip_prob < 0.5 else ("高" if slip_prob < 0.8 else "严重"))
                print(f"  是否打滑: {'是' if is_slip else '否'}, 打滑概率: {slip_prob*100:.2f}%, 风险等级: {risk_level}")
            else:
                print(f"  响应: {result}")
        except requests.exceptions.RequestException as e:
            print(f"  请求失败: {e}")
    
    return True


def test_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试5: 性能测试")
    print("=" * 60)
    
    test_data = {
        "model_code": 203000,
        "input_data": {
            "current": 120.0,       # 电流 (A)
            "speed_diff": 0.5,      # 速度差 (m/s)
            "vibration": 3.2,       # 振动值 (mm/s)
            "temperature": 45.0      # 温度 (°C)
        }
    }
    
    # 单次延迟测试
    print("\n单次推理延迟测试 (10次):")
    latencies = []
    for i in range(10):
        start = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            print(f"  第{i+1}次: {latency:.2f} ms")
        except:
            print(f"  第{i+1}次: 请求失败")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print(f"\n延迟统计:")
        print(f"  平均: {avg_latency:.2f} ms")
        print(f"  最小: {min_latency:.2f} ms")
        print(f"  最大: {max_latency:.2f} ms")
    
    # 批量吞吐量测试
    print("\n批量吞吐量测试 (100次请求):")
    batch_data = {
        "model_code": 203000,
        "input_data": [test_data["input_data"]] * 10
    }
    
    start = time.time()
    success_count = 0
    for _ in range(10):
        try:
            response = requests.post(
                f"{BASE_URL}/predict/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    total_time = time.time() - start
    if total_time > 0:
        throughput = (success_count * 10) / total_time
        print(f"  成功: {success_count}/10 批次")
        print(f"  总时间: {total_time:.2f} s")
        print(f"  吞吐量: {throughput:.2f} samples/s")
    
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("皮带机打滑预测模型 API 测试")
    print("模型代码: 203000")
    print("=" * 60)
    
    # 检查服务器
    if not check_server():
        print("\n⚠ 服务器不可用，请确保服务已启动 (http://127.0.0.1:9080)")
        print("按 Enter 键继续测试...")
        input()
    
    # 执行测试
    results = []
    results.append(("单样本预测", test_single_prediction()))
    results.append(("批量预测", test_batch_prediction()))
    results.append(("输入验证", test_input_validation()))
    results.append(("打滑场景", test_slip_scenarios()))
    results.append(("性能测试", test_performance()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n总计: {passed}/{total} 项测试通过")


if __name__ == "__main__":
    main()
