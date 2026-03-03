#!/usr/bin/env python3
"""
PA形态测试与回测框架 - 主入口

用法：
    cd /Users/guozhi/Agent/pa_dashboard
    python -m tests.backtest.run_backtest              # 完整回测（12只股票，30天）
    python -m tests.backtest.run_backtest --quick      # 快速回测（4只股票，20天）
    python -m tests.backtest.run_backtest --unit-only  # 仅运行单元测试
    python -m tests.backtest.run_backtest --no-unit    # 跳过单元测试，直接回测
"""

import sys
import os
import argparse

# 确保 pa_dashboard 在 Python 路径中
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import io


def run_unit_tests() -> dict:
    """运行单元测试，返回结果统计"""
    try:
        import pytest
    except ImportError:
        print("  ⚠ pytest未安装，跳过单元测试")
        print("  安装：pip install pytest")
        return None

    print("\n" + "=" * 60)
    print("阶段1：运行单元测试")
    print("=" * 60)

    # 运行测试，捕获输出
    test_dir = os.path.join(_ROOT, 'tests', 'unit')

    # 收集结果
    class ResultCollector:
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.xfailed = 0
            self.errors = []

    collector = ResultCollector()

    class ResultPlugin:
        def pytest_runtest_logreport(self, report):
            if report.when == 'call':
                if hasattr(report, 'wasxfail'):
                    # XPASS or XFAIL
                    collector.xfailed += 1
                elif report.passed:
                    collector.passed += 1
                elif report.failed:
                    collector.failed += 1
                    collector.errors.append(str(report.nodeid))

    plugin = ResultPlugin()

    exit_code = pytest.main(
        [test_dir, '-v', '--tb=short', '--no-header', '-x'],
        plugins=[plugin]
    )

    total = collector.passed + collector.failed + collector.xfailed
    print(f"\n单元测试结果: {collector.passed}/{total} 通过, "
          f"{collector.xfailed} XFAIL(已知偏差), "
          f"{collector.failed} 失败")

    if collector.errors:
        print(f"\n失败的测试:")
        for err in collector.errors:
            print(f"  ✗ {err}")

    return {
        'passed': collector.passed,
        'failed': collector.failed,
        'xfailed': collector.xfailed,
        'total': total,
        'exit_code': exit_code,
    }


def run_backtest(quick=False) -> dict:
    """运行历史回测"""
    from tests.backtest.backtester import Backtester, BACKTEST_SYMBOLS

    print("\n" + "=" * 60)
    print("阶段2：历史回测")
    print("=" * 60)

    if quick:
        symbols = ['AAPL', 'NVDA', 'SPY', 'TSLA']
        lookback_days = 15
        print(f"快速模式：{symbols}，回看{lookback_days}天")
    else:
        symbols = BACKTEST_SYMBOLS
        lookback_days = 30
        print(f"完整模式：{len(symbols)}只股票，回看{lookback_days}天")

    bt = Backtester(
        symbols=symbols,
        interval='5m',
        lookback_days=lookback_days,
        forward_bars=78,
    )

    print("开始回测（每只股票逐日截断分析）...")
    data = bt.run()

    return data


def apply_known_fixes():
    """应用pa_engine.py中的已知阈值修复"""
    from core import pa_engine

    # Fix A：双顶价格差阈值 0.8% → 3%
    # Fix B：双顶回调深度阈值 0.8% → 3%
    # Fix C：头肩顶头部深度 1% → 2%
    # 这些修复通过直接修改pa_engine.py实现（见Task #9）
    # 本函数作为占位符，提醒用户还需要手动修复
    print("\n注意：已知阈值问题需要修改 pa_engine.py 代码（见报告第四节）")
    print("修复后重新运行此脚本以验证改善效果。")


def main():
    parser = argparse.ArgumentParser(description='PA形态测试与回测框架')
    parser.add_argument('--quick', action='store_true', help='快速回测（4只股票，15天）')
    parser.add_argument('--unit-only', action='store_true', help='仅运行单元测试')
    parser.add_argument('--no-unit', action='store_true', help='跳过单元测试')
    args = parser.parse_args()

    print("PA形态测试与回测框架")
    print("=" * 60)

    unit_results = None
    backtest_data = None

    # 阶段1：单元测试
    if not args.no_unit:
        unit_results = run_unit_tests()
        if args.unit_only:
            print("\n（--unit-only 模式，跳过回测）")
            return

    # 阶段2：历史回测
    backtest_data = run_backtest(quick=args.quick)

    # 阶段3：生成报告
    from tests.backtest.report_generator import generate_report
    report = generate_report(backtest_data, unit_test_results=unit_results)

    print("\n" + "=" * 60)
    print("阶段3：回测报告")
    print("=" * 60)
    print(report)

    # 保存报告
    report_path = os.path.join(_ROOT, 'backtest_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存到: {report_path}")

    # 阶段4：提示已知修复
    apply_known_fixes()

    # 返回码：单元测试有失败则返回1
    if unit_results and unit_results.get('failed', 0) > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
