"""
压缩机泄漏预测模型 API 测试脚本
模型代码: 202000
"""
import requests
import time
from typing import Dict, List, Any


BASE_URL = "http://127.0.0.1:9080"
TIMEOUT = 10

# 模型代码常量
MODEL_CODE = 202000


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
    
    # 测试数据 - 压缩机泄漏预测参数
    test_data = {
        "model_code": MODEL_CODE,
        "input_data": {
            "pressure": 0.75,           # 压力 (MPa)
            "supply_flow": 320.0,       # 供给流量 (L/min)
            "demand_flow": 315.0        # 需求流量 (L/min)
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
        
        # 解析响应 - API返回 code: 0 表示成功，数据在 data.result 中
        code = result.get("code", -1)
        if code == 0:
            resp_data = result.get("data", {})
            result_data = resp_data.get("result", {})
            is_leak = result_data.get("is_leak", None)
            probabilities = result_data.get("probabilities", {})
            print(f"\n[PASS] 预测成功!")
            print(f"  是否泄漏: {'是' if is_leak else '否'}")
            print(f"  各类别概率:")
            for cls, prob in probabilities.items():
                print(f"    - {cls}: {prob:.4f} ({prob*100:.2f}%)")
            return True
        else:
            print(f"\n[FAIL] 预测失败: {result.get('msg', '未知错误')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] 请求失败: {e}")
        return False


def test_batch_prediction():
    """测试批量预测"""
    print("\n" + "=" * 60)
    print("测试2: 批量预测")
    print("=" * 60)
    
    # 批量测试数据 - 覆盖正常、泄漏、边缘等多种工况
    batch_data = {
        "model_code": MODEL_CODE,
        "input_data": [
            # 正常工况样本
            {
                "pressure": 0.82,           # 压力 (MPa) - 样本1: 理想工况
                "supply_flow": 360.0,       # 供给流量 (L/min)
                "demand_flow": 358.0        # 需求流量 (L/min)
            },
            {
                "pressure": 0.78,           # 压力 (MPa) - 样本2: 正常工况
                "supply_flow": 340.0,       # 供给流量 (L/min)
                "demand_flow": 335.0        # 需求流量 (L/min)
            },
            {
                "pressure": 0.75,           # 压力 (MPa) - 样本3: 正常偏低
                "supply_flow": 320.0,       # 供给流量 (L/min)
                "demand_flow": 318.0        # 需求流量 (L/min)
            },
            # 边缘工况样本
            {
                "pressure": 0.70,           # 压力 (MPa) - 样本4: 边缘工况-压力低
                "supply_flow": 300.0,       # 供给流量 (L/min)
                "demand_flow": 310.0        # 需求流量 (L/min)
            },
            {
                "pressure": 0.72,           # 压力 (MPa) - 样本5: 边缘工况-流量差
                "supply_flow": 310.0,       # 供给流量 (L/min)
                "demand_flow": 325.0        # 需求流量 (L/min)
            },
            {
                "pressure": 0.68,           # 压力 (MPa) - 样本6: 边缘工况-流量倒灌
                "supply_flow": 290.0,       # 供给流量 (L/min)
                "demand_flow": 305.0        # 需求流量 (L/min)
            },
            # 泄漏工况样本
            {
                "pressure": 0.65,           # 压力 (MPa) - 样本7: 轻微泄漏
                "supply_flow": 280.0,       # 供给流量 (L/min)
                "demand_flow": 320.0        # 需求流量 (L/min)
            },
            {
                "pressure": 0.60,           # 压力 (MPa) - 样本8: 中度泄漏
                "supply_flow": 260.0,       # 供给流量 (L/min)
                "demand_flow": 340.0        # 需求流量 (L/min)
            },
            {
                "pressure": 0.55,           # 压力 (MPa) - 样本9: 明显泄漏
                "supply_flow": 240.0,       # 供给流量 (L/min)
                "demand_flow": 360.0        # 需求流量 (L/min)
            },
            {
                "pressure": 0.50,           # 压力 (MPa) - 样本10: 严重泄漏
                "supply_flow": 220.0,       # 供给流量 (L/min)
                "demand_flow": 400.0        # 需求流量 (L/min)
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
        
        # API返回 code: 0 表示成功
        code = result.get("code", -1)
        if code == 0:
            resp_data = result.get("data", {})
            result_data = resp_data.get("result", {})
            predictions = result_data.get("predictions", [])
            inference_time = resp_data.get("inference_time_ms", 0)
            batch_size = len(predictions)
            throughput = (batch_size / inference_time * 1000) if inference_time > 0 else 0
            print(f"\n[PASS] 批量预测成功! 共 {len(predictions)} 条结果")
            for pred in predictions:
                idx = pred.get("index", "?")
                is_leak = pred.get("is_leak", None)
                probs = pred.get("probabilities", {})
                leak_prob = probs.get("leak", 0)
                print(f"\n  样本{idx}:")
                print(f"    是否泄漏: {'是' if is_leak else '否'} (泄漏概率: {leak_prob*100:.2f}%)")
                print(f"    概率分布: ", end="")
                print(", ".join([f"{k}={v:.3f}" for k, v in probs.items()]))
            print(f"\n性能指标:")
            print(f"  推理时间: {inference_time} ms")
            print(f"  吞吐量: {throughput:.2f} samples/s")
            return True
        else:
            print(f"\n[FAIL] 批量预测失败: {result.get('msg', '未知错误')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] 请求失败: {e}")
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
                "model_code": MODEL_CODE,
                "input_data": {
                    "pressure": 0.75,           # 压力 (MPa)
                    "supply_flow": 320.0         # 供给流量 (L/min)
                }
            }
        },
        {
            "name": "参数类型错误",
            "data": {
                "model_code": MODEL_CODE,
                "input_data": {
                    "pressure": "high",         # 压力 (MPa) - 类型错误
                    "supply_flow": 320.0,       # 供给流量 (L/min)
                    "demand_flow": 315.0         # 需求流量 (L/min)
                }
            }
        },
        {
            "name": "参数越界(压力<0)",
            "data": {
                "model_code": MODEL_CODE,
                "input_data": {
                    "pressure": -0.1,           # 压力 (MPa) - 越界
                    "supply_flow": 320.0,       # 供给流量 (L/min)
                    "demand_flow": 315.0         # 需求流量 (L/min)
                }
            }
        },
        {
            "name": "参数越界(压力>1)",
            "data": {
                "model_code": MODEL_CODE,
                "input_data": {
                    "pressure": 1.5,            # 压力 (MPa) - 越界
                    "supply_flow": 320.0,       # 供给流量 (L/min)
                    "demand_flow": 315.0         # 需求流量 (L/min)
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
            code = result.get("code", -1)
            if code == 0:
                print(f"  [WARN] 未检测到异常 (可能需要后端验证)")
            else:
                print(f"  [PASS] 正确拒绝: {result.get('msg', result)}")
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] 请求异常: {e}")
    
    return True


def test_leak_scenarios():
    """测试不同泄漏风险场景"""
    print("\n" + "=" * 60)
    print("测试4: 泄漏风险场景测试")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "理想工况",
            "data": {
                "pressure": 0.80,           # 压力 (MPa)
                "supply_flow": 350.0,       # 供给流量 (L/min)
                "demand_flow": 348.0        # 需求流量 (L/min)
            }
        },
        {
            "name": "正常工况",
            "data": {
                "pressure": 0.75,           # 压力 (MPa)
                "supply_flow": 320.0,       # 供给流量 (L/min)
                "demand_flow": 315.0        # 需求流量 (L/min)
            }
        },
        {
            "name": "轻微泄漏",
            "data": {
                "pressure": 0.72,           # 压力 (MPa)
                "supply_flow": 310.0,       # 供给流量 (L/min)
                "demand_flow": 340.0        # 需求流量 (L/min)
            }
        },
        {
            "name": "明显泄漏",
            "data": {
                "pressure": 0.65,           # 压力 (MPa)
                "supply_flow": 280.0,       # 供给流量 (L/min)
                "demand_flow": 360.0        # 需求流量 (L/min)
            }
        },
        {
            "name": "严重泄漏",
            "data": {
                "pressure": 0.55,           # 压力 (MPa)
                "supply_flow": 250.0,       # 供给流量 (L/min)
                "demand_flow": 400.0        # 需求流量 (L/min)
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        request_data = {
            "model_code": MODEL_CODE,
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
            # API返回 code: 0 表示成功
            code = result.get("code", -1)
            
            if code == 0:
                resp_data = result.get("data", {})
                result_data = resp_data.get("result", {})
                is_leak = result_data.get("is_leak", None)
                probs = result_data.get("probabilities", {})
                leak_prob = probs.get("leak", 0)
                risk_level = "正常" if leak_prob < 0.2 else ("轻微" if leak_prob < 0.5 else ("明显" if leak_prob < 0.8 else "严重"))
                print(f"  是否泄漏: {'是' if is_leak else '否'}, 泄漏概率: {leak_prob*100:.2f}%, 风险等级: {risk_level}")
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
        "model_code": MODEL_CODE,
        "input_data": {
            "pressure": 0.75,           # 压力 (MPa)
            "supply_flow": 320.0,       # 供给流量 (L/min)
            "demand_flow": 315.0        # 需求流量 (L/min)
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
        "model_code": MODEL_CODE,
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
    print("压缩机泄漏预测模型 API 测试")
    print("模型代码: 202000")
    print("=" * 60)
    
    # 检查服务器
    if not check_server():
        print("\n[!] 服务器不可用，请确保服务已启动 (http://127.0.0.1:9080)")
        print("按 Enter 键继续测试...")
        input()
    
    # 执行测试
    results = []
    results.append(("单样本预测", test_single_prediction()))
    results.append(("批量预测", test_batch_prediction()))
    results.append(("输入验证", test_input_validation()))
    results.append(("泄漏场景", test_leak_scenarios()))
    results.append(("性能测试", test_performance()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n总计: {passed}/{total} 项测试通过")


if __name__ == "__main__":
    main()
