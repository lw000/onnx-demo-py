"""
测试脚本批量执行器

功能: 依次执行 test 目录下所有以 test_ 开头的 Python 脚本
用法: python test/test.py
"""

import os
import subprocess
import sys
from pathlib import Path


def get_test_files():
    """获取所有测试文件"""
    # 获取当前文件所在目录
    script_dir = Path(__file__).parent.resolve()
    
    # 匹配所有 test_*.py 文件
    pattern = "test_*.py"
    files = sorted(script_dir.glob(pattern))
    
    # 过滤掉此脚本自身
    current_script = Path(__file__).name
    files = [f for f in files if f.name != current_script]
    
    return files


def run_test_file(file_path):
    """执行单个测试文件"""
    print(f"\n{'='*70}")
    print(f"执行: {file_path.name}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            cwd=str(file_path.parent),
            capture_output=False  # 直接输出到终端
        )
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False


def main():
    print("="*70)
    print("测试脚本批量执行器")
    print("="*70)
    
    # 获取测试文件
    test_files = get_test_files()
    
    if not test_files:
        print("\n未找到任何测试脚本!")
        return
    
    print(f"\n找到 {len(test_files)} 个测试脚本:")
    for i, f in enumerate(test_files, 1):
        print(f"  {i}. {f.name}")
    
    # 检查API服务
    print("\n" + "-"*70)
    print("提示: 请确保 API 服务已启动 (http://127.0.0.1:9080)")
    print("      启动命令: python -m uvicorn main:app --host 127.0.0.1 --port 9080")
    print("-"*70)
    
    # 执行测试
    results = []
    for i, file_path in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}]")
        success = run_test_file(file_path)
        results.append((file_path.name, success))
    
    # 汇总结果
    print("\n" + "="*70)
    print("测试汇总")
    print("="*70)
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    # 返回退出码
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
