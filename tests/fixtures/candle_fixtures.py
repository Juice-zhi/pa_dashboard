"""
合成K线数据工厂

用于单元测试的精确构造K线序列，确保形态可预测性。
所有函数均返回带 America/New_York 时区的 DataFrame。
"""

import pandas as pd
import numpy as np


def make_df(ohlcv_list, start='2024-01-02 09:30', freq='5min'):
    """
    将 [(open, high, low, close, volume), ...] 转为带时间索引的DataFrame。
    自动补足 OHLCV 约束：High >= max(Open,Close), Low <= min(Open,Close)
    """
    rows = []
    for row in ohlcv_list:
        o, h, l, c, v = row
        h = max(h, o, c)
        l = min(l, o, c)
        rows.append((o, h, l, c, v))

    idx = pd.date_range(start, periods=len(rows), freq=freq,
                        tz='America/New_York')
    df = pd.DataFrame(rows, index=idx,
                      columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    return df


# ─────────────────────────────────────────────
# 蜡烛图形态
# ─────────────────────────────────────────────

def make_pin_bar_bull(n_base=10, atr=1.0):
    """
    构造看涨Pin Bar（锤子线）：
    - 前 n_base 根普通K线，之后一根长下影线K
    - 下影线 > 实体3倍，下影线 > 总幅60%
    """
    rows = []
    price = 100.0
    for _ in range(n_base):
        rows.append((price, price + atr * 0.3, price - atr * 0.1, price + 0.2, 50000))
        price += 0.1
    # Pin Bar：下影很长
    body = atr * 0.2
    low = price - atr * 3.5
    high = price + atr * 0.1
    close = price + body
    rows.append((price, high, low, close, 120000))
    return make_df(rows)


def make_pin_bar_bear(n_base=10, atr=1.0):
    """
    构造看跌Pin Bar（墓碑线）：
    - 前 n_base 根普通K线，之后一根长上影线K
    """
    rows = []
    price = 100.0
    for _ in range(n_base):
        rows.append((price, price + atr * 0.3, price - atr * 0.1, price + 0.2, 50000))
        price += 0.1
    body = atr * 0.2
    high = price + atr * 3.5
    low = price - atr * 0.1
    close = price - body
    rows.append((price, high, low, close, 120000))
    return make_df(rows)


def make_engulfing_bull(n_base=10, atr=1.0):
    """
    构造看涨吞没：前一根大阴线被后一根大阳线完全包裹
    """
    rows = []
    price = 100.0
    for _ in range(n_base):
        rows.append((price, price + atr * 0.3, price - atr * 0.1, price + 0.2, 50000))
        price += 0.1
    # 大阴线
    o1 = price; c1 = price - atr * 1.2
    rows.append((o1, o1 + atr * 0.1, c1 - atr * 0.1, c1, 80000))
    # 吞没阳线：低于c1开，高于o1收
    o2 = c1 - atr * 0.1; c2 = o1 + atr * 0.1
    rows.append((o2, c2 + atr * 0.05, o2 - atr * 0.05, c2, 150000))
    return make_df(rows)


def make_engulfing_bear(n_base=10, atr=1.0):
    """
    构造看跌吞没：前一根大阳线被后一根大阴线完全包裹
    """
    rows = []
    price = 100.0
    for _ in range(n_base):
        rows.append((price, price + atr * 0.3, price - atr * 0.1, price + 0.2, 50000))
        price += 0.1
    # 大阳线
    o1 = price; c1 = price + atr * 1.2
    rows.append((o1, c1 + atr * 0.1, o1 - atr * 0.1, c1, 80000))
    # 吞没阴线
    o2 = c1 + atr * 0.1; c2 = o1 - atr * 0.1
    rows.append((o2, o2 + atr * 0.05, c2 - atr * 0.05, c2, 150000))
    return make_df(rows)


def make_doji(n_base=10, atr=1.0):
    """
    构造十字星：实体/总幅 < 10%
    """
    rows = []
    price = 100.0
    for _ in range(n_base):
        rows.append((price, price + atr * 0.3, price - atr * 0.1, price + 0.2, 50000))
        price += 0.1
    # 十字星：开收几乎相同
    rows.append((price, price + atr * 0.8, price - atr * 0.8, price + atr * 0.02, 60000))
    return make_df(rows)


def make_inside_bar(n_base=10, atr=1.0):
    """
    构造内包线：当前K High <= 前K High 且 Low >= 前K Low
    使用明确的非doji形态避免干扰
    """
    rows = []
    price = 100.0
    # 普通阳线基础
    for _ in range(n_base):
        rows.append((price, price + atr * 0.5, price - atr * 0.2, price + atr * 0.4, 50000))
        price += atr * 0.4
    # 大K线（母K）— 宽幅阳线，有大实体
    mother_h = price + atr * 2.0
    mother_l = price - atr * 2.0
    rows.append((price, mother_h, mother_l, price + atr * 1.5, 100000))
    price += atr * 1.5
    # 内包K（子K）— 比母K窄，有一定实体（避免触发doji）
    ib_h = mother_h - atr * 0.5   # 低于母K高点
    ib_l = mother_l + atr * 0.5   # 高于母K低点
    ib_o = (ib_h + ib_l) / 2 - atr * 0.2
    ib_c = (ib_h + ib_l) / 2 + atr * 0.3  # 有明确实体（约0.5 ATR），不是doji
    rows.append((ib_o, ib_h, ib_l, ib_c, 40000))
    return make_df(rows)


def make_outside_bar(n_base=10, atr=1.0):
    """
    构造外包线：当前K High > 前K High 且 Low < 前K Low
    """
    rows = []
    price = 100.0
    for _ in range(n_base):
        rows.append((price, price + atr * 0.3, price - atr * 0.1, price + 0.2, 50000))
        price += 0.1
    # 小K线
    rows.append((price, price + atr * 0.5, price - atr * 0.5, price + 0.1, 50000))
    price += 0.1
    # 外包K
    rows.append((price, price + atr * 1.5, price - atr * 1.5, price + atr * 0.5, 120000))
    return make_df(rows)


def make_morning_star(n_base=8, atr=1.0):
    """
    构造早晨之星：大阴 + 小实体 + 大阳（收盘超过大阴中点）
    """
    rows = []
    price = 105.0
    for _ in range(n_base):
        rows.append((price, price + atr * 0.3, price - atr * 0.1, price - 0.2, 50000))
        price -= 0.2
    # 大阴线
    o1 = price; c1 = price - atr * 1.5
    rows.append((o1, o1 + atr * 0.1, c1 - atr * 0.1, c1, 100000))
    # 小实体（doji/spinning）
    o2 = c1 - atr * 0.2; c2 = c1 - atr * 0.1
    rows.append((o2, o2 + atr * 0.4, o2 - atr * 0.4, c2, 50000))
    # 大阳线（收盘 > 大阴中点）
    midpoint = (o1 + c1) / 2
    o3 = c2; c3 = midpoint + atr * 0.3
    rows.append((o3, c3 + atr * 0.1, o3 - atr * 0.1, c3, 130000))
    return make_df(rows)


def make_evening_star(n_base=8, atr=1.0):
    """
    构造黄昏之星：大阳 + 小实体 + 大阴（收盘低于大阳中点）
    """
    rows = []
    price = 100.0
    for _ in range(n_base):
        rows.append((price, price + atr * 0.3, price - atr * 0.1, price + 0.2, 50000))
        price += 0.2
    # 大阳线
    o1 = price; c1 = price + atr * 1.5
    rows.append((o1, c1 + atr * 0.1, o1 - atr * 0.1, c1, 100000))
    # 小实体
    o2 = c1 + atr * 0.2; c2 = c1 + atr * 0.1
    rows.append((o2, o2 + atr * 0.4, o2 - atr * 0.4, c2, 50000))
    # 大阴线（收盘 < 大阳中点）
    midpoint = (o1 + c1) / 2
    o3 = c2; c3 = midpoint - atr * 0.3
    rows.append((o3, o3 + atr * 0.1, c3 - atr * 0.1, c3, 130000))
    return make_df(rows)


# ─────────────────────────────────────────────
# 结构形态
# ─────────────────────────────────────────────

def make_bull_flag(atr=1.0, n_flat=10):
    """
    构造标准牛旗：
    - 25根热身K（确保ATR有效 + 满足 n-15 > 0 约束）
    - 旗杆：5根强阳，净涨幅 > 2.5 ATR，实体比 > 60%
    - 旗面：n_flat 根窄幅小阴（总跌幅 < 旗杆的40%），斜率向下
    """
    rows = []
    price = 100.0

    # 热身K线（25根）：让ATR和swing点数量满足算法需求
    for _ in range(25):
        rows.append((price, price + atr * 0.4, price - atr * 0.3, price + atr * 0.2, 60000))
        price += atr * 0.1

    # 旗杆：5根强阳
    for _ in range(5):
        o = price
        c = price + atr * 0.85
        rows.append((o, c + atr * 0.08, o - atr * 0.04, c, 120000))
        price = c

    pole_move = 5 * atr * 0.85   # 旗杆总涨幅
    max_flag_range = pole_move * 0.38

    # 旗面：小阴K，斜率向下
    flag_drop = max_flag_range * 0.7
    drop_per_bar = flag_drop / n_flat
    for i in range(n_flat):
        o = price
        c = price - drop_per_bar
        rows.append((o, o + atr * 0.05, c - atr * 0.03, c, 30000))
        price = c

    return make_df(rows)


def make_bull_flag_wide(atr=1.0, n_flat=10):
    """
    构造旗面过宽的牛旗（旗面宽度 > 旗杆的40%，应被拒绝）
    """
    rows = []
    price = 100.0
    # 热身K（同make_bull_flag）
    for _ in range(25):
        rows.append((price, price + atr * 0.4, price - atr * 0.3, price + atr * 0.2, 60000))
        price += atr * 0.1

    for _ in range(5):
        o = price
        c = price + atr * 0.85
        rows.append((o, c + atr * 0.08, o - atr * 0.04, c, 120000))
        price = c

    pole_move = 5 * atr * 0.85
    # 旗面宽度设为旗杆的50%（超过40%阈值）
    flag_range = pole_move * 0.50
    for i in range(n_flat):
        o = price
        c = price - flag_range / n_flat
        h = max(o, c) + flag_range * 0.25
        l = min(o, c) - flag_range * 0.25
        rows.append((o, h, l, c, 30000))
        price = c

    return make_df(rows)


def make_bear_flag(atr=1.0, n_flat=10):
    """
    构造标准熊旗：
    - 25根热身K
    - 旗杆：5根强阴，净跌幅 > 2.5 ATR，实体比 > 60%
    - 旗面：n_flat 根窄幅小阳（总涨幅 < 旗杆的40%），斜率向上
    """
    rows = []
    price = 120.0

    # 热身K线
    for _ in range(25):
        rows.append((price, price + atr * 0.4, price - atr * 0.3, price - atr * 0.1, 60000))
        price -= atr * 0.1

    for _ in range(5):
        o = price
        c = price - atr * 0.85
        rows.append((o, o + atr * 0.04, c - atr * 0.08, c, 120000))
        price = c

    pole_move = 5 * atr * 0.85
    max_flag_range = pole_move * 0.38
    rise_per_bar = (max_flag_range * 0.7) / n_flat

    for i in range(n_flat):
        o = price
        c = price + rise_per_bar
        rows.append((o, c + atr * 0.03, o - atr * 0.05, c, 30000))
        price = c

    return make_df(rows)


def make_double_top(price_diff_pct=0.005, trough_depth_pct=0.05, gap=20):
    """
    构造双顶：
    - 20根热身K（确保ATR有效）
    - price_diff_pct: 两顶价格差（百分比），默认0.5%（两顶几乎相等）
    - trough_depth_pct: 颈线回调深度，默认5%
    - gap: 两顶间隔K线数
    """
    rows = []
    price = 100.0
    atr = 0.5

    # 热身K线
    for _ in range(20):
        rows.append((price, price + atr * 0.4, price - atr * 0.3, price + atr * 0.2, 60000))
        price += atr * 0.1

    # 上涨到第一顶
    for i in range(12):
        o = price
        c = price + atr * 0.6
        rows.append((o, c + atr * 0.1, o - atr * 0.05, c, 80000))
        price = c
    top1 = price

    # 回调到颈线
    trough = top1 * (1 - trough_depth_pct)
    n_down = gap // 2
    step_down = (top1 - trough) / n_down
    for i in range(n_down):
        o = price
        c = price - step_down
        rows.append((o, o + atr * 0.05, c - atr * 0.05, c, 60000))
        price = c

    # 反弹到第二顶（与第一顶相差 price_diff_pct）
    top2 = top1 * (1 + price_diff_pct)
    n_up = gap - n_down
    step_up = (top2 - price) / n_up
    for i in range(n_up):
        o = price
        c = price + step_up
        rows.append((o, c + atr * 0.05, o - atr * 0.05, c, 70000))
        price = c

    # 从第二顶回落至颈线附近（使当前价格处于颈线 ±3×ATR 内，满足近端过滤）
    neckline = top1 * (1 - trough_depth_pct)
    decline_target = neckline + atr * 0.5  # 停在颈线上方0.5 ATR（触发区附近）
    n_decline = 8
    step_decline = (top2 - decline_target) / n_decline
    for i in range(n_decline):
        o = price
        c = price - step_decline
        rows.append((o, o + atr * 0.05, c - atr * 0.05, c, 60000))
        price = c

    return make_df(rows)


def make_double_top_shallow(trough_depth_pct=0.008):
    """
    构造回调仅0.8%的双顶（低于权威3%标准，但高于pa_engine的0.8%阈值边界）
    预期：pa_engine会接受此形态，但权威标准应拒绝
    """
    return make_double_top(price_diff_pct=0.003, trough_depth_pct=trough_depth_pct)


def make_double_bottom(price_diff_pct=0.005, peak_height_pct=0.05, gap=20):
    """
    构造双底：
    - 20根热身K
    - price_diff_pct: 两底价格差（百分比）
    - peak_height_pct: 中间反弹幅度
    - gap: 两底间隔K线数
    """
    rows = []
    price = 120.0
    atr = 0.5

    # 热身K线
    for _ in range(20):
        rows.append((price, price + atr * 0.4, price - atr * 0.3, price - atr * 0.1, 60000))
        price -= atr * 0.1

    # 下跌到第一底
    for i in range(12):
        o = price
        c = price - atr * 0.6
        rows.append((o, o + atr * 0.05, c - atr * 0.1, c, 80000))
        price = c
    bot1 = price

    # 反弹到颈线
    peak = bot1 * (1 + peak_height_pct)
    n_up = gap // 2
    step_up = (peak - bot1) / n_up
    for i in range(n_up):
        o = price
        c = price + step_up
        rows.append((o, c + atr * 0.05, o - atr * 0.05, c, 60000))
        price = c

    # 回落到第二底（与第一底相差 price_diff_pct）
    bot2 = bot1 * (1 - price_diff_pct)
    n_down = gap - n_up
    step_down = (price - bot2) / n_down
    for i in range(n_down):
        o = price
        c = price - step_down
        rows.append((o, o + atr * 0.05, c - atr * 0.05, c, 70000))
        price = c

    # 从第二底反弹至颈线附近（使当前价格处于颈线 ±3×ATR 内，满足近端过滤）
    neckline = peak  # 颈线=中间反弹高点
    rally_target = neckline - atr * 0.5  # 停在颈线下方0.5 ATR（突破区附近）
    n_rally = 8
    step_rally = (rally_target - bot2) / n_rally
    for i in range(n_rally):
        o = price
        c = price + step_rally
        rows.append((o, c + atr * 0.05, o - atr * 0.05, c, 60000))
        price = c

    return make_df(rows)


def make_head_shoulders(head_depth_pct=0.04, shoulder_diff_pct=0.01):
    """
    构造标准头肩顶：左肩-头-右肩，颈线水平
    头肩顶使用足够大的摆动幅度确保每个峰谷都是唯一清晰的摆动点。
    """
    rows = []
    atr = 0.4
    neckline = 100.0

    # 热身K线（20根）：单调上涨到颈线，提供ATR和方向
    price = 90.0
    for i in range(20):
        o = price; c = price + 0.5
        rows.append((o, c + atr * 0.05, o - atr * 0.02, c, 60000))
        price = c

    # === 左肩 ===
    # 上涨到左肩高点（清晰的摆动高）
    shoulder_h = neckline * (1 + head_depth_pct * 0.6)
    step = (shoulder_h - price) / 6
    for _ in range(6):
        o = price; c = price + step
        rows.append((o, c + atr * 0.05, o - atr * 0.02, c, 80000))
        price = c
    # 单根K标注左肩峰顶（High明显高于两侧）
    rows.append((price, price + atr * 0.5, price - atr * 0.1, price - atr * 0.05, 90000))
    price -= atr * 0.05

    # 下跌回颈线
    step_down = (price - neckline) / 5
    for _ in range(5):
        o = price; c = price - step_down
        rows.append((o, o + atr * 0.02, c - atr * 0.05, c, 60000))
        price = c
    # 颈线低点（单根K标注）
    rows.append((price, price + atr * 0.1, price - atr * 0.5, price + atr * 0.05, 70000))
    price += atr * 0.05

    # === 头部 ===
    head_h = neckline * (1 + head_depth_pct)
    step = (head_h - price) / 8
    for _ in range(8):
        o = price; c = price + step
        rows.append((o, c + atr * 0.05, o - atr * 0.02, c, 100000))
        price = c
    # 单根K标注头部峰顶
    rows.append((price, price + atr * 0.6, price - atr * 0.1, price - atr * 0.05, 110000))
    price -= atr * 0.05

    # 下跌回颈线
    step_down = (price - neckline) / 6
    for _ in range(6):
        o = price; c = price - step_down
        rows.append((o, o + atr * 0.02, c - atr * 0.05, c, 70000))
        price = c
    # 颈线低点（单根K标注）
    rows.append((price, price + atr * 0.1, price - atr * 0.5, price + atr * 0.05, 65000))
    price += atr * 0.05

    # === 右肩 ===
    rs_h = shoulder_h * (1 - shoulder_diff_pct)
    step = (rs_h - price) / 6
    for _ in range(6):
        o = price; c = price + step
        rows.append((o, c + atr * 0.05, o - atr * 0.02, c, 75000))
        price = c
    # 单根K标注右肩峰顶
    rows.append((price, price + atr * 0.5, price - atr * 0.1, price - atr * 0.05, 70000))
    price -= atr * 0.05

    # 跌破颈线
    step_down = (price - (neckline - atr)) / 4
    for _ in range(4):
        o = price; c = price - step_down
        rows.append((o, o + atr * 0.05, c - atr * 0.1, c, 55000))
        price = c

    return make_df(rows)


def make_head_shoulders_shallow(head_depth_pct=0.008):
    """
    构造头部深度不足的头肩顶（0.8%，低于权威2%标准，但高于pa_engine的1%阈值边界）
    预期：pa_engine会接受（≥1%），但权威标准应拒绝（<2%）
    """
    return make_head_shoulders(head_depth_pct=head_depth_pct)


def make_three_drives_top(decay_pct=0.25):
    """
    构造三推顶：三次上推，高点依次升高，但每次涨幅依次减小（decay_pct）
    前15根热身K确保swing算法有足够数据
    """
    rows = []
    price = 100.0
    atr = 0.5

    # 热身K线（15根）
    for _ in range(15):
        rows.append((price, price + atr * 0.4, price - atr * 0.3, price + atr * 0.1, 50000))

    # 第一推
    amp1 = atr * 8
    for _ in range(8):
        o = price; c = price + amp1 / 8
        rows.append((o, c + atr * 0.1, o - atr * 0.05, c, 90000))
        price = c

    # 回调
    pullback = amp1 * 0.4
    for _ in range(5):
        o = price; c = price - pullback / 5
        rows.append((o, o + atr * 0.05, c - atr * 0.05, c, 60000))
        price = c

    # 第二推（涨幅比第一推略小）
    amp2 = amp1 * 0.85
    for _ in range(7):
        o = price; c = price + amp2 / 7
        rows.append((o, c + atr * 0.1, o - atr * 0.05, c, 80000))
        price = c

    # 回调
    for _ in range(5):
        o = price; c = price - pullback / 5
        rows.append((o, o + atr * 0.05, c - atr * 0.05, c, 55000))
        price = c

    # 第三推（动能明显衰竭）
    amp3 = amp1 * (1 - decay_pct)
    for _ in range(6):
        o = price; c = price + amp3 / 6
        rows.append((o, c + atr * 0.1, o - atr * 0.05, c, 65000))
        price = c

    # 几根确认K
    for _ in range(4):
        rows.append((price, price + atr * 0.2, price - atr * 0.2, price - 0.1, 45000))
        price -= 0.1

    return make_df(rows)


def make_channel_up(n_bars=60, atr=0.5, slope=0.2):
    """
    构造上升通道：价格在两条平行上升直线之间确定性波动（sin波形）
    - slope: 每根K线的价格上升量
    - atr: 通道宽度参数
    """
    rows = []
    import math
    price_base = 100.0
    channel_width = atr * 3

    for i in range(n_bars):
        # 用sin波形在通道内波动，确定性不随机
        phase = math.sin(i * 2 * math.pi / 12) * channel_width * 0.4
        trend = price_base + slope * i
        mid = trend + phase
        o = mid
        c = mid + atr * 0.1 * math.sin(i * 0.5)
        h = max(o, c) + atr * 0.2
        l = min(o, c) - atr * 0.2
        rows.append((o, h, l, c, 60000))

    return make_df(rows)


def make_channel_down(n_bars=60, atr=0.5, slope=-0.2):
    """构造下降通道（确定性sin波形）"""
    rows = []
    import math
    price_base = 120.0
    channel_width = atr * 3

    for i in range(n_bars):
        phase = math.sin(i * 2 * math.pi / 12) * channel_width * 0.4
        trend = price_base + slope * i
        mid = trend + phase
        o = mid
        c = mid + atr * 0.1 * math.sin(i * 0.5)
        h = max(o, c) + atr * 0.2
        l = min(o, c) - atr * 0.2
        rows.append((o, h, l, c, 60000))

    return make_df(rows)


def make_inv_head_shoulders(head_depth_pct=0.04, shoulder_diff_pct=0.01):
    """
    构造标准反头肩底：左肩-头-右肩（底部结构），对称于头肩顶
    """
    rows = []
    atr = 0.4
    neckline = 110.0

    # 热身K线（20根）：单调下跌到颈线
    price = 120.0
    for i in range(20):
        o = price; c = price - 0.5
        rows.append((o, o + atr * 0.02, c - atr * 0.05, c, 60000))
        price = c

    # === 左肩 ===
    shoulder_l = neckline * (1 - head_depth_pct * 0.6)
    step_down = (price - shoulder_l) / 6
    for _ in range(6):
        o = price; c = price - step_down
        rows.append((o, o + atr * 0.02, c - atr * 0.05, c, 80000))
        price = c
    # 单根K标注左肩底部
    rows.append((price, price + atr * 0.1, price - atr * 0.5, price + atr * 0.05, 90000))
    price += atr * 0.05

    # 反弹到颈线
    step_up = (neckline - price) / 5
    for _ in range(5):
        o = price; c = price + step_up
        rows.append((o, c + atr * 0.05, o - atr * 0.02, c, 60000))
        price = c
    # 颈线高点（单根K标注）
    rows.append((price, price + atr * 0.5, price - atr * 0.1, price - atr * 0.05, 70000))
    price -= atr * 0.05

    # === 头部 ===
    head_l = neckline * (1 - head_depth_pct)
    step_down = (price - head_l) / 8
    for _ in range(8):
        o = price; c = price - step_down
        rows.append((o, o + atr * 0.02, c - atr * 0.05, c, 100000))
        price = c
    # 单根K标注头部底
    rows.append((price, price + atr * 0.1, price - atr * 0.6, price + atr * 0.05, 110000))
    price += atr * 0.05

    # 反弹到颈线
    step_up = (neckline - price) / 6
    for _ in range(6):
        o = price; c = price + step_up
        rows.append((o, c + atr * 0.05, o - atr * 0.02, c, 70000))
        price = c
    # 颈线高点（单根K标注）
    rows.append((price, price + atr * 0.5, price - atr * 0.1, price - atr * 0.05, 65000))
    price -= atr * 0.05

    # === 右肩 ===
    rs_l = shoulder_l * (1 + shoulder_diff_pct)
    step_down = (price - rs_l) / 6
    for _ in range(6):
        o = price; c = price - step_down
        rows.append((o, o + atr * 0.02, c - atr * 0.05, c, 75000))
        price = c
    # 单根K标注右肩底
    rows.append((price, price + atr * 0.1, price - atr * 0.5, price + atr * 0.05, 70000))
    price += atr * 0.05

    # 突破颈线向上
    step_up = (neckline + atr - price) / 4
    for _ in range(4):
        o = price; c = price + step_up
        rows.append((o, c + atr * 0.1, o - atr * 0.05, c, 55000))
        price = c

    return make_df(rows)
