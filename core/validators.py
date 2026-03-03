"""
PA形态权威参考定义与胜率来源

引用文献：
- Al Brooks, "Trading Price Action Trends" (2012), Wiley
- Al Brooks, "Trading Price Action Reversals" (2012), Wiley
- Thomas N. Bulkowski, "Encyclopedia of Chart Patterns" 3rd ed. (2021), Wiley
- 5分钟级别折减系数：参考 Bulkowski 对日内短时框架的噪音折减，约 ×0.88

说明：
- rate_daily  = 权威文献中日线级别统计胜率（颈线/目标有效突破后衡量）
- rate_5min   = 对5分钟级别的估算折减值
- rate_5min_bt = 由本框架回测得出的实测值（回测完成后填入，初始为None）
"""

WIN_RATE_SOURCES: dict = {
    'bull_flag': {
        'rate_daily':    0.67,
        'rate_5min':     0.59,   # 0.67 × 0.88 ≈ 0.59
        'rate_5min_bt':  None,
        'source': (
            'Al Brooks "Trading Price Action Trends" (2012) p.253: '
            '"A bull flag succeeds about two thirds of the time in a bull trend." '
            '顺势旗形成功率约67%（以突破旗面高点为入场信号）'
        ),
        'note': '5分钟级别因噪音增加估算折减至59%；逆势环境胜率下降至约50%',
    },
    'bear_flag': {
        'rate_daily':    0.67,
        'rate_5min':     0.59,
        'rate_5min_bt':  None,
        'source': (
            'Al Brooks "Trading Price Action Trends" (2012) p.253: '
            '熊旗与牛旗对称，顺势成功率约67%'
        ),
        'note': '同牛旗，旗面斜率向上为顺势（空头趋势中的回调整理）',
    },
    'double_top': {
        'rate_daily':    0.73,
        'rate_5min':     0.64,
        'rate_5min_bt':  None,
        'source': (
            'Thomas Bulkowski "Encyclopedia of Chart Patterns" 3rd ed. (2021) p.271: '
            '"Double tops: 73% of those with downward breakouts reach the target." '
            '双顶形成后向下突破颈线，73%的概率达到目标（跌幅=顶到颈线距离）'
        ),
        'note': '需颈线有效跌破（收盘价低于颈线）；仅统计下行突破案例',
    },
    'double_bottom': {
        'rate_daily':    0.72,
        'rate_5min':     0.63,
        'rate_5min_bt':  None,
        'source': (
            'Bulkowski (2021) p.243: '
            '"Double bottoms: 72% reach the target after upward breakout." '
            '双底向上突破颈线后，72%概率达到目标'
        ),
        'note': '需颈线有效突破（收盘价高于颈线）',
    },
    'head_shoulders': {
        'rate_daily':    0.83,
        'rate_5min':     0.73,
        'rate_5min_bt':  None,
        'source': (
            'Bulkowski (2021) p.357: '
            '"Head-and-shoulders tops with downward breakouts show an 83% success rate." '
            '头肩顶向下突破颈线后，83%概率达到目标（日线级别）'
        ),
        'note': (
            '5分钟折减后约73%；'
            '成功率定义：价格下跌至少（头到颈线距离）；'
            '需颈线有效跌破'
        ),
    },
    'inv_head_shoulders': {
        'rate_daily':    0.83,
        'rate_5min':     0.73,
        'rate_5min_bt':  None,
        'source': (
            'Bulkowski (2021) p.368: '
            '"Inverse head-and-shoulders: 83% success rate after upward breakout." '
            '反头肩底向上突破颈线后，83%概率达到目标'
        ),
        'note': '与头肩顶对称',
    },
    'three_drives_top': {
        'rate_daily':    None,
        'rate_5min':     None,
        'rate_5min_bt':  None,
        'source': (
            '无权威统计来源。'
            'Al Brooks "Trading Price Action Reversals" (2012) 第9章描述"楔形/三推反转"(Three-push reversal/Wedge)概念，'
            '认为第三推动能衰竭后是做空机会，但未给出统计胜率。'
            '当前代码使用0.58为临时估算值，应由本框架回测结果替换。'
        ),
        'note': 'Brooks将其归类为Wedge(楔形)的一种变体；识别关键：三次推进动能依次递减',
    },
    'three_drives_bot': {
        'rate_daily':    None,
        'rate_5min':     None,
        'rate_5min_bt':  None,
        'source': (
            '无权威统计来源。同三推顶，为三推底部结构（三次下探，动能递减）。'
            '当前代码使用0.57为临时估算值，应由本框架回测结果替换。'
        ),
        'note': '三推底与三推顶结构对称',
    },
    'channel_up': {
        'rate_daily':    0.54,
        'rate_5min':     0.52,
        'rate_5min_bt':  None,
        'source': (
            'Bulkowski (2021) p.145: '
            '"Rising price channels: 54% of trades from the lower trendline reach the upper trendline." '
            '上升通道从下轨买入触及上轨的成功率54%（中性偏弱形态）'
        ),
        'note': '通道内做波段交易；突破上轨方向继续的概率不在此统计中',
    },
    'channel_down': {
        'rate_daily':    0.54,
        'rate_5min':     0.52,
        'rate_5min_bt':  None,
        'source': (
            'Bulkowski (2021) p.145 对称：'
            '下降通道从上轨做空触及下轨成功率54%'
        ),
        'note': '同上升通道对称',
    },
}


# ─────────────────────────────────────────────
# 权威几何约束
# 单元测试用此表验证 pa_engine 的实现是否符合定义
# ─────────────────────────────────────────────

PATTERN_CONSTRAINTS: dict = {

    'double_top': {
        # 两顶价格差（百分比）: 允许最大误差
        # 标准：两顶价格应"基本相等"，Bulkowski允许3%以内的差异
        'price_diff_max':     0.03,   # 当前pa_engine: 0.008 (过严，导致漏报)
        # 颈线回调深度: 两顶之间的回调必须足够深
        # 标准：Bulkowski要求至少3%的回调才视为有效颈线
        'trough_depth_min':   0.03,   # 当前pa_engine: 0.008 (过宽，导致误报)
        # 两顶之间最小间距（根K线数）
        'min_bar_gap':        10,
        # 注：pa_engine 当前 price_diff > 0.008 且 trough_depth >= 0.008 都不符合权威标准
        'pa_engine_issues': [
            'price_diff阈值0.8%过严（权威允许3%），导致两顶价格只要差异>0.8%就被拒绝',
            'trough_depth阈值0.8%过宽（权威要求3%），导致浅回调也被识别为双顶',
        ],
    },

    'double_bottom': {
        'price_diff_max':     0.03,
        'peak_height_min':    0.03,   # 两底之间的反弹幅度 ≥ 3%
        'min_bar_gap':        10,
        'pa_engine_issues': [
            '与double_top相同的阈值问题（0.8% vs 3%）',
        ],
    },

    'head_shoulders': {
        # 头比两肩高出的最小幅度（相对颈线）
        # 标准：头部应明显高于肩部，至少高出颈线2%
        'head_depth_min':       0.02,  # 当前pa_engine: 0.01 (过低，误报小头肩)
        # 两肩价格差异最大允许值
        'shoulder_diff_max':    0.05,  # 两肩相差不超过5% —— pa_engine已符合
        # 颈线斜率（每根K线的价格变化/颈线价格），基本水平
        'neckline_slope_max':   0.002,
        # 右肩通常不高于左肩（可选增强条件）
        'right_shoulder_lower': True,  # 当前pa_engine未检查此条件
        'pa_engine_issues': [
            'head_depth阈值1%过低（权威要求2%），导致浅头肩被误识别',
            '未检查右肩是否低于左肩（Brooks认为右肩更低是更强确认）',
        ],
    },

    'inv_head_shoulders': {
        'head_depth_min':       0.02,
        'shoulder_diff_max':    0.05,
        'neckline_slope_max':   0.002,
        'right_shoulder_higher': True,
        'pa_engine_issues': [
            '同head_shoulders，head_depth阈值1%过低',
        ],
    },

    'bull_flag': {
        # 旗杆最小幅度（ATR倍数）
        'pole_atr_min':           2.5,   # pa_engine已符合（使用2.5）
        # 旗杆实体比（净涨幅/总幅度）—— 过滤震荡型假旗杆
        'pole_body_ratio_min':    0.6,   # pa_engine已符合
        # 旗面宽度（高低差）不超过旗杆高度的此比例
        'flag_width_max_ratio':   0.4,   # pa_engine已符合
        # 旗面斜率方向：牛旗必须向下倾斜（逆势整理）
        'flag_slope_negative':    True,  # pa_engine已符合
        # 旗面线性回归R²（整理的有序程度）
        'flag_r2_min':            0.3,   # pa_engine已符合
        'pa_engine_issues': [],          # 牛旗实现基本符合标准
    },

    'bear_flag': {
        'pole_atr_min':           2.5,
        'pole_body_ratio_min':    0.6,
        'flag_width_max_ratio':   0.4,
        'flag_slope_positive':    True,  # 熊旗整理斜率向上
        'flag_r2_min':            0.3,
        'pa_engine_issues': [],
    },

    'three_drives': {
        # 第三推动能衰竭最小幅度（相对第一推）
        # Brooks: 第三推应明显弱于第一推，通常>15%的动能衰减
        'momentum_decay_min':     0.15,  # pa_engine使用0.15（符合）
        # 每两推之间必须有明确的回调（摆动低点/高点）
        'intermediate_pullback':  True,  # pa_engine已检查
        'pa_engine_issues': [
            '无权威统计来源，胜率0.58/0.57为临时估算值需回测验证',
        ],
    },

    'channel': {
        # 上下轨线性回归R²最小值
        'r2_min':                 0.65,  # pa_engine已符合
        # 上下轨斜率差异最大比例（平行通道条件）
        'slope_diff_ratio_max':   0.35,  # pa_engine已符合
        # 最少摆动点数
        'min_swing_points':       3,     # pa_engine已符合
        'pa_engine_issues': [],
    },
}


def get_reference_win_rate(kind: str, timeframe: str = '5min') -> tuple:
    """
    获取形态的参考胜率及来源说明。

    Returns:
        (rate, source_text) 其中 rate 可能为 None（无权威来源）
    """
    entry = WIN_RATE_SOURCES.get(kind)
    if entry is None:
        return None, f'未知形态: {kind}'
    field = f'rate_{timeframe}'
    rate = entry.get(field)
    source = entry.get('source', '')
    note = entry.get('note', '')
    return rate, f'{source}' + (f'  [备注: {note}]' if note else '')


def get_constraints(kind: str) -> dict:
    """获取形态的权威几何约束字典"""
    # 通道类统一到 'channel'
    if kind.startswith('channel'):
        return PATTERN_CONSTRAINTS.get('channel', {})
    if kind.startswith('three_drives'):
        return PATTERN_CONSTRAINTS.get('three_drives', {})
    return PATTERN_CONSTRAINTS.get(kind, {})


def list_pa_engine_issues() -> list:
    """汇总 pa_engine 中所有已知的阈值问题"""
    issues = []
    for kind, constraints in PATTERN_CONSTRAINTS.items():
        for issue in constraints.get('pa_engine_issues', []):
            issues.append({'pattern': kind, 'issue': issue})
    return issues
