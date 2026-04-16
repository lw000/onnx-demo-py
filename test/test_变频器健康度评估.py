"""
变频器健康度评估模型 API 测试脚本
模型代码: 301000
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
    
    # 测试数据 - 变频器健康评估参数
    test_data = {
        "model_code": 301000,
        "input_data": {
            "mean_ripple": 2.5,      # 平均纹波 (V)
            "std_ripple": 0.8,        # 纹波标准差 (V)
            "mean_temp": 65.0,        # 平均温度 (°C)
            "temp_rise": 5.2,         # 温升 (°C)
            "mean_load": 0.75,        # 平均负载率 (0-1)
            "temp_range": 15.0        # 温度范围 (°C)
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
            print(f"\n✓ 预测成功!")
            print(f"  电容寿命: {data.get('life_pct', 'N/A')}%")
            print(f"  热风险指数: {data.get('thermal_risk', 'N/A')}")
            print(f"  健康状态: {data.get('status_name', 'N/A')}")
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
        "model_code": 301000,
        "input_data": [
            {
                "mean_ripple": 2.5,      # 样本1: 正常工况
                "std_ripple": 0.8,
                "mean_temp": 65.0,
                "temp_rise": 5.2,
                "mean_load": 0.75,
                "temp_range": 15.0
            },
            {
                "mean_ripple": 4.8,      # 样本2: 高风险工况
                "std_ripple": 1.5,
                "mean_temp": 85.0,
                "temp_rise": 12.5,
                "mean_load": 0.92,
                "temp_range": 32.0
            },
            {
                "mean_ripple": 3.2,      # 样本3: 警告工况
                "std_ripple": 1.0,
                "mean_temp": 75.0,
                "temp_rise": 8.5,
                "mean_load": 0.85,
                "temp_range": 20.0
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
                life = pred.get("life_pct", "N/A")
                risk = pred.get("thermal_risk", "N/A")
                status_name = pred.get("status_name", "N/A")
                print(f"  样本{idx}: 寿命={life}%, 热风险={risk}, 状态={status_name}")
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
                "model_code": 301000,
                "input_data": {
                    "mean_ripple": 2.5,
                    "std_ripple": 0.8
                }
            }
        },
        {
            "name": "参数类型错误",
            "data": {
                "model_code": 301000,
                "input_data": {
                    "mean_ripple": "high",
                    "std_ripple": 0.8,
                    "mean_temp": 65.0,
                    "temp_rise": 5.2,
                    "mean_load": 0.75,
                    "temp_range": 15.0
                }
            }
        },
        {
            "name": "参数越界(负载>1)",
            "data": {
                "model_code": 301000,
                "input_data": {
                    "mean_ripple": 2.5,
                    "std_ripple": 0.8,
                    "mean_temp": 65.0,
                    "temp_rise": 5.2,
                    "mean_load": 1.5,
                    "temp_range": 15.0
                }
            }
        },
        {
            "name": "参数越界(负温度)",
            "data": {
                "model_code": 301000,
                "input_data": {
                    "mean_ripple": 2.5,
                    "std_ripple": 0.8,
                    "mean_temp": -50.0,
                    "temp_rise": 5.2,
                    "mean_load": 0.75,
                    "temp_range": 15.0
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


def test_working_conditions():
    """测试不同工况场景"""
    print("\n" + "=" * 60)
    print("测试4: 工况场景测试")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "理想工况",
            "data": {
                "mean_ripple": 1.5,
                "std_ripple": 0.3,
                "mean_temp": 45.0,
                "temp_rise": 2.0,
                "mean_load": 0.5,
                "temp_range": 8.0
            }
        },
        {
            "name": "正常工况",
            "data": {
                "mean_ripple": 2.5,
                "std_ripple": 0.8,
                "mean_temp": 65.0,
                "temp_rise": 5.2,
                "mean_load": 0.75,
                "temp_range": 15.0
            }
        },
        {
            "name": "高温警告",
            "data": {
                "mean_ripple": 3.5,
                "std_ripple": 1.2,
                "mean_temp": 78.0,
                "temp_rise": 10.0,
                "mean_load": 0.88,
                "temp_range": 25.0
            }
        },
        {
            "name": "高风险异常",
            "data": {
                "mean_ripple": 5.0,
                "std_ripple": 2.0,
                "mean_temp": 90.0,
                "temp_rise": 15.0,
                "mean_load": 0.95,
                "temp_range": 35.0
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        request_data = {
            "model_code": 301000,
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
                life = data.get("life_pct", "N/A")
                risk = data.get("thermal_risk", "N/A")
                status_name = data.get("status_name", "N/A")
                print(f"  寿命: {life}%, 热风险: {risk}, 状态: {status_name}")
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
        "model_code": 301000,
        "input_data": {
            "mean_ripple": 2.5,
            "std_ripple": 0.8,
            "mean_temp": 65.0,
            "temp_rise": 5.2,
            "mean_load": 0.75,
            "temp_range": 15.0
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
        "model_code": 301000,
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
    print("变频器健康度评估模型 API 测试")
    print("模型代码: 301000")
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
    results.append(("工况场景", test_working_conditions()))
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
