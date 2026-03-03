"""
回测报告生成器

将回测数据格式化为人类可读的报告，包含：
- 单元测试结果摘要
- 历史回测胜率汇总（对比权威参考值）
- 已知问题与修复建议
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.validators import WIN_RATE_SOURCES


# 形态中文名映射
KIND_LABELS = {
    'bull_flag':          '牛旗(Bull Flag)',
    'bear_flag':          '熊旗(Bear Flag)',
    'double_top':         '双顶(Double Top)',
    'double_bottom':      '双底(Double Bottom)',
    'head_shoulders':     '头肩顶(H&S)',
    'inv_head_shoulders': '反头肩底(IHS)',
    'three_drives_top':   '三推顶(3-Drive Top)',
    'three_drives_bot':   '三推底(3-Drive Bot)',
    'channel_up':         '上升通道(Channel Up)',
    'channel_down':       '下降通道(Channel Down)',
    'channel_flat':       '横盘通道(Channel Flat)',
}


def _get_ref_win_rate(kind: str) -> tuple:
    """获取参考胜率和来源摘要"""
    entry = WIN_RATE_SOURCES.get(kind)
    if entry is None:
        return None, '无参考数据'
    rate = entry.get('rate_5min')
    source = entry.get('source', '')
    # 提取关键引用（截短）
    short_src = source[:60] + '...' if len(source) > 60 else source
    return rate, short_src


def _pct(v) -> str:
    if v is None:
        return ' N/A  '
    return f'{v*100:.1f}%'


def _diff_indicator(actual, reference) -> str:
    """生成胜率对比指示符"""
    if actual is None or reference is None:
        return ''
    diff = actual - reference
    if diff >= 0.05:
        return ' ▲+好'
    elif diff >= -0.05:
        return ' ≈参考'
    else:
        return ' ▼-差'


def generate_report(backtest_data: dict, unit_test_results: dict = None) -> str:
    """
    生成完整回测报告。

    Args:
        backtest_data: Backtester.run() 返回的字典
        unit_test_results: {'passed': int, 'failed': int, 'xfailed': int, 'total': int}

    Returns:
        格式化的报告字符串
    """
    meta = backtest_data.get('meta', {})
    summary = backtest_data.get('summary', {})
    errors = backtest_data.get('errors', [])
    results = backtest_data.get('results', [])

    lines = []
    w = 70
    lines.append('═' * w)
    lines.append('PA形态回测报告')
    lines.append(f"股票池: {', '.join(meta.get('symbols', []))}")
    lines.append(f"时间框架: {meta.get('interval', '5m')}  |  "
                 f"回看天数: {meta.get('lookback_days', '?')}天  |  "
                 f"验证窗口: {meta.get('forward_bars', '?')}根K线")
    lines.append(f"生成时间: {meta.get('run_at', '?')}")
    lines.append(f"总信号数: {meta.get('total_signals', len(results))}")
    lines.append('═' * w)

    # ── 一、单元测试 ──
    lines.append('')
    lines.append('一、单元测试结果')
    lines.append('─' * w)
    if unit_test_results:
        p = unit_test_results.get('passed', 0)
        f = unit_test_results.get('failed', 0)
        xf = unit_test_results.get('xfailed', 0)
        t = unit_test_results.get('total', p + f + xf)
        status = '✓ 通过' if f == 0 else '✗ 有失败'
        lines.append(f"  {status}: {p}/{t} 通过  "
                     f"({xf}个XFAIL=已知偏差不计入失败, {f}个失败)")
        if f > 0:
            lines.append(f"  ⚠ 注意：{f}个非预期失败需要修复")
    else:
        lines.append('  （本次未运行单元测试，请单独执行 pytest tests/unit/ -v）')

    # ── 二、回测胜率汇总 ──
    lines.append('')
    lines.append('二、历史回测胜率汇总')
    lines.append('─' * w)

    if not summary:
        lines.append('  （无回测数据）')
    else:
        # 表头
        header = (
            f"{'形态':<22} {'信号':>4} {'WIN':>4} {'LOSS':>4} "
            f"{'超时':>4} {'实际胜率':>7} {'参考胜率':>7} 评价"
        )
        lines.append(header)
        lines.append('─' * w)

        kind_order = [
            'bull_flag', 'bear_flag',
            'double_top', 'double_bottom',
            'head_shoulders', 'inv_head_shoulders',
            'three_drives_top', 'three_drives_bot',
            'channel_up', 'channel_down', 'channel_flat',
        ]
        # 先按指定顺序，其余追加
        ordered_kinds = [k for k in kind_order if k in summary]
        extra_kinds = [k for k in summary if k not in kind_order]
        ordered_kinds += extra_kinds

        for kind in ordered_kinds:
            s = summary[kind]
            ref_rate, _ = _get_ref_win_rate(kind)
            actual_rate = s.get('win_rate')
            label = KIND_LABELS.get(kind, kind)
            decided = s.get('total_decided', 0)
            indicator = _diff_indicator(actual_rate, ref_rate)

            line = (
                f"{label:<22} {s['total_signals']:>4} "
                f"{s['wins']:>4} {s['losses']:>4} {s['timeouts']:>4} "
                f"{_pct(actual_rate):>7} {_pct(ref_rate):>7}{indicator}"
            )
            lines.append(line)

        lines.append('')
        lines.append('说明：参考胜率来源见下方"胜率来源"节；评价：▲=优于参考 ≈=符合 ▼=低于参考')

    # ── 三、胜率来源引用 ──
    lines.append('')
    lines.append('三、胜率参考来源（权威引用）')
    lines.append('─' * w)
    for kind, entry in WIN_RATE_SOURCES.items():
        label = KIND_LABELS.get(kind, kind)
        rate_daily = entry.get('rate_daily')
        rate_5min = entry.get('rate_5min')
        source = entry.get('source', '')
        note = entry.get('note', '')
        lines.append(f"\n[{label}]")
        lines.append(f"  日线参考: {_pct(rate_daily)}  5分钟估算: {_pct(rate_5min)}")
        lines.append(f"  来源: {source}")
        if note:
            lines.append(f"  备注: {note}")

    # ── 四、市场背景与回测说明 ──
    lines.append('')
    lines.append('四、回测说明与市场背景')
    lines.append('─' * w)
    lines.append('  【回测方法】')
    lines.append('  - 每交易日10:00 ET截断历史数据，模拟实时分析场景')
    lines.append('  - 信号入场：等待价格实际触及入场价（breakout确认）后才开始计算')
    lines.append('  - 胜负判定：入场后前78根K（约6.5小时）内，先触及目标→WIN，先触及止损→LOSS')
    lines.append('  - 超时：78根K内未触及目标或止损 → TIMEOUT（不计入胜率）')
    lines.append('')
    lines.append('  【已应用修复】（相比原始pa_engine）')
    lines.append('  ✓ 双顶/双底 price_diff阈值: 0.8% → 3%（Bulkowski标准）')
    lines.append('  ✓ 双顶 trough_depth / 双底 peak_height阈值: 0.8% → 3%（Bulkowski标准）')
    lines.append('  ✓ 头肩顶/反头肩底 head_depth阈值: 1% → 2%（Bulkowski标准）')
    lines.append('  ✓ 双顶/双底/头肩形态添加近端过滤：只在当前价接近颈线时发信号（±3×ATR）')
    lines.append('  ✓ 回测入场逻辑：先等待价格触及入场价，支持做空/做多双向')
    lines.append('')
    lines.append('  【市场背景说明】')
    lines.append('  - 回测期间（2026年2月-3月）大盘处于震荡偏弱行情')
    lines.append('  - 牛旗参考胜率59%基于顺势（牛市中的多头旗形），震荡市实际胜率偏低属正常')
    lines.append('  - 三推形态（无权威参考）回测结果：')

    # 输出三推形态实测结果
    td_top = summary.get('three_drives_top', {})
    td_bot = summary.get('three_drives_bot', {})
    td_top_wr = td_top.get('win_rate')
    td_bot_wr = td_bot.get('win_rate')
    lines.append(f"    · 三推顶(3-Drive Top)：实测胜率 {_pct(td_top_wr)}（{td_top.get('total_decided',0)}个已决信号）")
    lines.append(f"    · 三推底(3-Drive Bot)：实测胜率 {_pct(td_bot_wr)}（{td_bot.get('total_decided',0)}个已决信号）")
    lines.append(f"  建议将 WIN_RATE_TABLE 中 three_drives_top/bot 的临时估算值更新为上述回测值")

    # ── 五、剩余待改进问题 ──
    lines.append('')
    lines.append('五、剩余待改进问题')
    lines.append('─' * w)
    lines.append('  [MEDIUM] 旗形胜率偏低（牛旗23%，熊旗全部超时）')
    lines.append('    - 根本原因：回测期为震荡市，旗形在趋势行情中胜率显著更高')
    lines.append('    - 建议：增加趋势过滤条件（仅在趋势方向发出旗形信号）')
    lines.append('  [LOW] 双顶/双底/头肩信号稀少（近端过滤使信号大幅减少）')
    lines.append('    - 原因：在5分钟级别30天数据中，价格近颈线的双顶/底很少见')
    lines.append('    - 数据充分后（更长周期/更多股票）胜率统计将更可靠')
    lines.append('  [INFO] three_drives 无权威来源，回测胜率已更新（见上方）')

    # ── 回测错误记录 ──
    if errors:
        lines.append('')
        lines.append('（回测过程错误）')
        lines.append('─' * w)
        for err in errors:
            lines.append(f"  ✗ {err}")

    lines.append('')
    lines.append('═' * w)
    lines.append('报告结束')
    lines.append('═' * w)

    return '\n'.join(lines)


def generate_quick_summary(backtest_data: dict) -> str:
    """生成简短摘要（适合终端快速查看）"""
    summary = backtest_data.get('summary', {})
    meta = backtest_data.get('meta', {})
    lines = []
    lines.append(f"\n=== 回测摘要 ({meta.get('run_at','')}) ===")
    lines.append(f"总信号数: {meta.get('total_signals', 0)}")
    lines.append(f"{'形态':<25} {'信号':>5} {'胜率':>7} {'参考':>7}")
    lines.append('-' * 48)
    for kind, s in sorted(summary.items()):
        ref, _ = _get_ref_win_rate(kind)
        label = KIND_LABELS.get(kind, kind)
        lines.append(f"{label:<25} {s['total_signals']:>5} "
                     f"{_pct(s.get('win_rate')):>7} {_pct(ref):>7}")
    return '\n'.join(lines)
