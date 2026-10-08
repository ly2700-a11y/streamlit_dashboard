#!/usr/bin/env python3
"""
Verify Streamlit app setup and dependencies
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_imports():
    """Check all required imports."""
    print("🔍 检查依赖导入...")
    
    dependencies = {
        "streamlit": "Streamlit 框架",
        "pandas": "数据处理",
        "numpy": "数值计算",
        "matplotlib": "图表绘制",
        "seaborn": "统计可视化",
    }
    
    missing = []
    for pkg, desc in dependencies.items():
        try:
            __import__(pkg)
            print(f"  ✅ {pkg:20} ({desc})")
        except ImportError:
            print(f"  ❌ {pkg:20} ({desc}) - 缺失!")
            missing.append(pkg)
    
    return len(missing) == 0

def check_project_files():
    """Check required project files."""
    print("\n🗂️  检查项目文件...")
    
    required_files = [
        "config.py",
        "data_loader.py",
        "baseline_run.py",
        "strategies.py",
        "metrics.py",
        "src/app/main.py",
        "data/CLC",
    ]
    
    missing = []
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - 缺失!")
            missing.append(file_path)
    
    return len(missing) == 0

def check_data():
    """Check data files."""
    print("\n📊 检查数据文件...")
    
    data_dir = PROJECT_ROOT / "data" / "CLC"
    if not data_dir.exists():
        print(f"  ❌ {data_dir} 不存在!")
        return False
    
    # Check for both .csv and .CSV files
    csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.CSV"))
    if not csv_files:
        print(f"  ❌ {data_dir} 中没有 CSV 文件!")
        return False
    
    print(f"  ✅ 找到 {len(csv_files)} 个 CSV 文件")
    print(f"     示例: {csv_files[0].name}")
    
    return True

def main():
    """Run all checks."""
    print("=" * 60)
    print("🔧 Streamlit 应用配置检查")
    print("=" * 60)
    
    checks = [
        ("依赖导入", check_imports),
        ("项目文件", check_project_files),
        ("数据文件", check_data),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 检查 {name} 时出错: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 检查总结")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有检查通过！可以启动应用:")
        print("   $ streamlit run src/app/main.py")
        print("   或使用: $ ./run_app.sh")
    else:
        print("❌ 有些检查未通过，请修复后重试")
        print("\n解决方案:")
        print("1️⃣  缺少依赖? 运行: pip install -r streamlit_requirements.txt")
        print("2️⃣  缺少数据? 检查 data/CLC/ 目录是否存在")
        print("3️⃣  缺少文件? 检查项目根目录文件完整性")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
