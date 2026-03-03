"""
PA (Price Action) 分析引擎 v2
新增：5分钟日内支持、盘前Bias、日内结构形态识别、胜率统计
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, time, date
import yfinance as yf


# ═══════════════════════════════════════════
# 数据结构定义
# ═══════════════════════════════════════════

@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str          # 'HH' | 'HL' | 'LH' | 'LL'
    timestamp: str

@dataclass
class CandleSignal:
    index: int
    timestamp: str
    pattern: str
    strength: str      # 'strong' | 'moderate' | 'weak'
    description: str

@dataclass
class SRLevel:
    price: float
    kind: str          # 'resistance' | 'support'
    strength: int
    last_touch_idx: int

@dataclass
class SupplyDemandZone:
    top: float
    bottom: float
    kind: str          # 'supply' | 'demand'
    origin_idx: int
    freshness: str     # 'fresh' | 'tested' | 'broken'
    description: str

@dataclass
class TradeSetup:
    direction: str
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    rr_ratio_1: float
    rr_ratio_2: float
    quality: str       # 'A+' | 'A' | 'B' | 'C'
    win_rate_est: float   # 历史估算胜率 0~1
    reasons: list = field(default_factory=list)

@dataclass
class MarketBias:
    trend: str
    strength: str
    last_structure: str
    bias_score: float
    description: str
    bullish_factors: list = field(default_factory=list)
    bearish_factors: list = field(default_factory=list)

# ── 新增：盘前Bias ──
@dataclass
class KeyLevel:
    price: float
    label: str         # 'PDH'|'PDL'|'PDC'|'PDM'|'VWAP'|'IB_H'|'IB_L'|'PM_H'|'PM_L'
    color: str
    style: str         # 'solid' | 'dashed' | 'dotted'

@dataclass
class PreMarketBias:
    gap_type: str      # 'gap_up'|'gap_down'|'flat'
    gap_pct: float
    pdh: float
    pdl: float
    pdc: float
    pdm: float
    pm_high: float
    pm_low: float
    pm_volume_ratio: float
    vwap: float
    ib_high: float
    ib_low: float
    ib_range: float
    bias_score: float  # -1 ~ +1
    bias_label: str    # '强多'|'偏多'|'中性'|'偏空'|'强空'
    description: str
    key_levels: list   # list[dict] for JSON serialization
    session_boundaries: list  # 时段分隔线的时间索引

# ── 新增：日内结构形态 ──
@dataclass
class PricePattern:
    kind: str          # 'bull_flag'|'bear_flag'|'three_drives_top'|'three_drives_bot'
                       # |'channel_up'|'channel_down'|'double_top'|'double_bottom'
                       # |'head_shoulders'|'inv_head_shoulders'
    direction: str     # 'bullish' | 'bearish'
    start_idx: int
    end_idx: int
    confidence: float  # 0 ~ 1
    key_prices: list   # 用于图上画线的关键价位 [{x_idx, y_price}]
    entry: float
    stop: float
    target: float
    win_rate_est: float
    description: str


# ═══════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════

def body_size(row) -> float:
    return abs(row['Close'] - row['Open'])

def candle_range(row) -> float:
    return row['High'] - row['Low']

def upper_wick(row) -> float:
    return row['High'] - max(row['Open'], row['Close'])

def lower_wick(row) -> float:
    return min(row['Open'], row['Close']) - row['Low']

def is_bullish(row) -> bool:
    return row['Close'] > row['Open']

def is_bearish(row) -> bool:
    return row['Close'] < row['Open']

def avg_range(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return (df['High'] - df['Low']).rolling(n).mean()

def _linreg(x: np.ndarray, y: np.ndarray):
    """线性回归，返回 (slope, intercept, r2)"""
    n = len(x)
    if n < 2:
        return 0.0, float(np.mean(y)), 0.0
    sx, sy = x.sum(), y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, float(np.mean(y)), 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(intercept), float(r2)

def _calc_vwap(df: pd.DataFrame) -> float:
    """从9:30开始计算当日VWAP"""
    if df.empty:
        return 0.0
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume'].replace(0, 1)
    return float((tp * vol).sum() / vol.sum())

def _is_intraday(interval: str) -> bool:
    return interval in ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h')


# ═══════════════════════════════════════════
# 1. 趋势结构分析
# ═══════════════════════════════════════════

class TrendStructureAnalyzer:
    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def find_swings(self, df: pd.DataFrame):
        highs, lows = [], []
        n = len(df)
        lb = self.lookback
        for i in range(lb, n - lb):
            if all(df['High'].iloc[i] >= df['High'].iloc[i - lb:i]) and \
               all(df['High'].iloc[i] >= df['High'].iloc[i + 1:i + lb + 1]):
                highs.append((i, float(df['High'].iloc[i]), df.index[i]))
            if all(df['Low'].iloc[i] <= df['Low'].iloc[i - lb:i]) and \
               all(df['Low'].iloc[i] <= df['Low'].iloc[i + 1:i + lb + 1]):
                lows.append((i, float(df['Low'].iloc[i]), df.index[i]))
        return highs, lows

    def classify_structure(self, highs, lows) -> list:
        points = []
        all_pts = sorted(
            [('H', *h) for h in highs] + [('L', *l) for l in lows],
            key=lambda x: x[1]
        )
        prev_high = prev_low = None
        for kind, idx, price, ts in all_pts:
            ts_str = str(ts)[:19]
            if kind == 'H':
                label = 'HH' if (prev_high is None or price > prev_high) else 'LH'
                points.append(SwingPoint(idx, price, label, ts_str))
                prev_high = price
            else:
                label = 'HL' if (prev_low is None or price > prev_low) else 'LL'
                points.append(SwingPoint(idx, price, label, ts_str))
                prev_low = price
        return sorted(points, key=lambda x: x.index)

    def determine_bias(self, df: pd.DataFrame) -> MarketBias:
        highs, lows = self.find_swings(df)
        points = self.classify_structure(highs, lows)
        if len(points) < 4:
            return MarketBias('ranging', 'weak', 'N/A', 0.0, '摆动点不足，无法判断趋势结构')
        recent = points[-6:]
        hh = sum(1 for p in recent if p.kind == 'HH')
        hl = sum(1 for p in recent if p.kind == 'HL')
        lh = sum(1 for p in recent if p.kind == 'LH')
        ll = sum(1 for p in recent if p.kind == 'LL')
        bull_score = (hh + hl) / max(len(recent), 1)
        bear_score = (lh + ll) / max(len(recent), 1)
        bias_score = round(bull_score - bear_score, 3)
        bull_factors, bear_factors = [], []
        if hh: bull_factors.append(f'出现{hh}个更高高点(HH)')
        if hl: bull_factors.append(f'出现{hl}个更高低点(HL)')
        if lh: bear_factors.append(f'出现{lh}个更低高点(LH)')
        if ll: bear_factors.append(f'出现{ll}个更低低点(LL)')
        last = recent[-1]
        if bias_score >= 0.4:
            trend = 'bullish'; strength = 'strong' if bias_score > 0.6 else 'moderate'
            desc = f'多头结构确立，最近{last.kind}={last.price:.4f}'
        elif bias_score <= -0.4:
            trend = 'bearish'; strength = 'strong' if bias_score < -0.6 else 'moderate'
            desc = f'空头结构确立，最近{last.kind}={last.price:.4f}'
        else:
            trend = 'ranging'; strength = 'weak'; desc = '震荡区间，无明显趋势'
        return MarketBias(trend, strength, last.kind, bias_score, desc, bull_factors, bear_factors)


# ═══════════════════════════════════════════
# 2. K线形态识别
# ═══════════════════════════════════════════

class CandlePatternDetector:

    def analyze_all(self, df: pd.DataFrame) -> list:
        signals = []
        atr = avg_range(df)
        for i in range(2, len(df)):
            row = df.iloc[i]; prev = df.iloc[i-1]; prev2 = df.iloc[i-2]
            ts = str(df.index[i])[:19]
            atr_val = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else candle_range(row)
            sig = (self._pin_bar(row, i, ts, atr_val) or
                   self._engulfing(row, prev, i, ts, atr_val) or
                   self._doji(row, i, ts, atr_val) or
                   self._inside_bar(row, prev, i, ts) or
                   self._outside_bar(row, prev, i, ts) or
                   self._morning_evening_star(row, prev, prev2, i, ts, atr_val))
            if sig:
                signals.append(sig)
        return signals

    def analyze_last_n(self, df: pd.DataFrame, n: int = 30) -> list:
        result = []
        atr = avg_range(df)
        for i in range(max(2, len(df) - n), len(df)):
            row = df.iloc[i]; prev = df.iloc[i-1] if i > 0 else row
            atr_val = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else candle_range(row)
            ts = str(df.index[i])[:19]
            total = candle_range(row)
            body = body_size(row)
            body_ratio = body / total if total > 0 else 0
            uw = upper_wick(row); lw = lower_wick(row)
            direction = '阳线' if is_bullish(row) else '阴线'
            size_desc = '大实体' if body_ratio > 0.7 else ('小实体' if body_ratio < 0.3 else '中实体')
            wick_desc = ''
            if uw > body * 1.5: wick_desc += '长上影(压力) '
            if lw > body * 1.5: wick_desc += '长下影(支撑) '
            if not wick_desc: wick_desc = '正常影线'
            vol_desc = ''
            if 'Volume' in df.columns:
                avg_v = df['Volume'].rolling(20).mean().iloc[i]
                if not pd.isna(avg_v) and avg_v > 0:
                    vol_desc = f'{row["Volume"]/avg_v:.1f}x均量'
            sig = (self._pin_bar(row, i, ts, atr_val) or
                   self._engulfing(row, prev, i, ts, atr_val) or
                   self._doji(row, i, ts, atr_val) or
                   self._inside_bar(row, prev, i, ts) or
                   self._outside_bar(row, prev, i, ts))
            patterns = [sig.pattern] if sig else []
            pa_desc = sig.description if sig else '无明显形态'
            result.append({
                'index': i, 'timestamp': ts,
                'open': round(float(row['Open']), 4),
                'high': round(float(row['High']), 4),
                'low': round(float(row['Low']), 4),
                'close': round(float(row['Close']), 4),
                'direction': direction, 'size': size_desc,
                'body_ratio': round(body_ratio, 2),
                'upper_wick_ratio': round(uw / atr_val, 2) if atr_val > 0 else 0,
                'lower_wick_ratio': round(lw / atr_val, 2) if atr_val > 0 else 0,
                'wick_desc': wick_desc.strip(), 'volume_desc': vol_desc,
                'patterns': patterns, 'pa_desc': pa_desc,
            })
        return result

    def _pin_bar(self, row, idx, ts, atr):
        body = body_size(row); total = candle_range(row)
        if total < atr * 0.5: return None
        uw = upper_wick(row); lw = lower_wick(row)
        if lw > body * 2 and lw > total * 0.6:
            return CandleSignal(idx, ts, 'pin_bar_bull',
                                'strong' if lw > body * 3 else 'moderate',
                                f'看涨Pin Bar：长下影拒绝低价，下影={lw:.4f}，买方控盘')
        if uw > body * 2 and uw > total * 0.6:
            return CandleSignal(idx, ts, 'pin_bar_bear',
                                'strong' if uw > body * 3 else 'moderate',
                                f'看跌Pin Bar：长上影拒绝高价，上影={uw:.4f}，卖方控盘')
        return None

    def _engulfing(self, row, prev, idx, ts, atr):
        cb = (min(row['Open'],row['Close']), max(row['Open'],row['Close']))
        pb = (min(prev['Open'],prev['Close']), max(prev['Open'],prev['Close']))
        if is_bullish(row) and is_bearish(prev) and cb[0] < pb[0] and cb[1] > pb[1]:
            return CandleSignal(idx, ts, 'engulf_bull', 'strong', '看涨吞没：买方强势反转')
        if is_bearish(row) and is_bullish(prev) and cb[1] > pb[1] and cb[0] < pb[0]:
            return CandleSignal(idx, ts, 'engulf_bear', 'strong', '看跌吞没：卖方强势反转')
        return None

    def _doji(self, row, idx, ts, atr):
        body = body_size(row); total = candle_range(row)
        if total < atr * 0.3: return None
        if body / total < 0.1:
            return CandleSignal(idx, ts, 'doji', 'moderate', '十字星：多空平衡，等待方向选择')
        return None

    def _inside_bar(self, row, prev, idx, ts):
        if row['High'] <= prev['High'] and row['Low'] >= prev['Low']:
            return CandleSignal(idx, ts, 'inside_bar', 'moderate', '内包线(IB)：蓄势整理，等待突破')
        return None

    def _outside_bar(self, row, prev, idx, ts):
        if row['High'] > prev['High'] and row['Low'] < prev['Low']:
            d = '买方' if is_bullish(row) else '卖方'
            return CandleSignal(idx, ts, 'outside_bar', 'moderate', f'外包线：{d}主导，吞噬前K')
        return None

    def _morning_evening_star(self, row, prev, prev2, idx, ts, atr):
        if (is_bearish(prev2) and body_size(prev2) > atr * 0.5 and body_size(prev) < atr * 0.3 and
                is_bullish(row) and row['Close'] > (prev2['Open'] + prev2['Close']) / 2):
            return CandleSignal(idx, ts, 'morning_star', 'strong', '早晨之星：三K底部反转')
        if (is_bullish(prev2) and body_size(prev2) > atr * 0.5 and body_size(prev) < atr * 0.3 and
                is_bearish(row) and row['Close'] < (prev2['Open'] + prev2['Close']) / 2):
            return CandleSignal(idx, ts, 'evening_star', 'strong', '黄昏之星：三K顶部反转')
        return None


# ═══════════════════════════════════════════
# 3. 支撑阻力识别
# ═══════════════════════════════════════════

class SupportResistanceDetector:
    def detect(self, df: pd.DataFrame, tolerance: float = 0.003) -> list:
        levels = []
        n = len(df); lb = 3
        raw_highs, raw_lows = [], []
        for i in range(lb, n - lb):
            hi = float(df['High'].iloc[i]); lo = float(df['Low'].iloc[i])
            if all(hi >= df['High'].iloc[i-lb:i]) and all(hi >= df['High'].iloc[i+1:i+lb+1]):
                raw_highs.append((hi, i))
            if all(lo <= df['Low'].iloc[i-lb:i]) and all(lo <= df['Low'].iloc[i+1:i+lb+1]):
                raw_lows.append((lo, i))

        def cluster(pts, kind):
            if not pts: return
            for ps in sorted(pts, key=lambda x: x[0]):
                placed = False
                for lvl in levels:
                    if lvl.kind == kind and abs(ps[0] - lvl.price) / lvl.price < tolerance:
                        lvl.strength += 1
                        lvl.last_touch_idx = max(lvl.last_touch_idx, ps[1])
                        lvl.price = round((lvl.price * (lvl.strength - 1) + ps[0]) / lvl.strength, 4)
                        placed = True; break
                if not placed:
                    levels.append(SRLevel(round(ps[0], 4), kind, 1, ps[1]))

        cluster(raw_highs, 'resistance'); cluster(raw_lows, 'support')
        levels = [l for l in levels if l.strength >= 2]
        levels.sort(key=lambda x: -x.strength)
        return levels[:12]


# ═══════════════════════════════════════════
# 4. 供需区域识别
# ═══════════════════════════════════════════

class SupplyDemandDetector:
    def detect(self, df: pd.DataFrame) -> list:
        zones = []
        atr = avg_range(df); n = len(df)
        for i in range(2, n - 1):
            curr = df.iloc[i]
            atr_val = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else candle_range(curr)
            if is_bullish(curr) and body_size(curr) > atr_val * 1.2:
                base = df.iloc[i-1]
                zt = max(float(base['Open']), float(base['Close'])) + atr_val * 0.1
                zb = float(base['Low'])
                fsh = self._check_freshness(df, i, zt, zb, 'demand')
                zones.append(SupplyDemandZone(round(zt,4), round(zb,4), 'demand', i, fsh,
                                               f'需求区(RBR)：{str(df.index[i])[:10]} 大阳线离开'))
            if is_bearish(curr) and body_size(curr) > atr_val * 1.2:
                base = df.iloc[i-1]
                zt = float(base['High'])
                zb = min(float(base['Open']), float(base['Close'])) - atr_val * 0.1
                fsh = self._check_freshness(df, i, zt, zb, 'supply')
                zones.append(SupplyDemandZone(round(zt,4), round(zb,4), 'supply', i, fsh,
                                               f'供给区(DBD)：{str(df.index[i])[:10]} 大阴线离开'))
        fresh = [z for z in zones if z.freshness == 'fresh']
        tested = [z for z in zones if z.freshness == 'tested']
        result = fresh[-6:] + tested[-4:]
        cp = float(df['Close'].iloc[-1])
        result.sort(key=lambda z: abs((z.top + z.bottom) / 2 - cp))
        return result[:10]

    def _check_freshness(self, df, origin_idx, top, bottom, kind) -> str:
        for i in range(origin_idx + 1, len(df)):
            hi = float(df['High'].iloc[i]); lo = float(df['Low'].iloc[i])
            if kind == 'demand' and lo <= top and lo >= bottom: return 'tested'
            if kind == 'supply' and hi >= bottom and hi <= top: return 'tested'
            if kind == 'demand' and lo < bottom: return 'broken'
            if kind == 'supply' and hi > top: return 'broken'
        return 'fresh'


# ═══════════════════════════════════════════
# 5. 盘前Bias分析器（美股专用）
# ═══════════════════════════════════════════

class PreMarketAnalyzer:
    """
    分析美股5分钟级别的盘前Bias
    需要数据已转换到 America/New_York 时区，包含盘前盘后数据
    """

    OPEN_TIME  = time(9, 30)
    CLOSE_TIME = time(16, 0)
    PM_START   = time(4,  0)
    IB_END     = time(10, 30)

    def analyze(self, df: pd.DataFrame) -> Optional[PreMarketBias]:
        try:
            return self._do_analyze(df)
        except Exception:
            return None

    def _do_analyze(self, df: pd.DataFrame) -> Optional[PreMarketBias]:
        if df.empty:
            return None

        # 确保时区正确
        idx = df.index
        if idx.tz is None:
            df = df.copy()
            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('America/New_York')
        elif str(idx.tz) != 'America/New_York':
            df = df.copy()
            df.index = df.index.tz_convert('America/New_York')

        dates = sorted(df.index.normalize().unique())
        if len(dates) < 2:
            return None

        today = dates[-1]
        prev_dates = [d for d in dates if d < today]
        if not prev_dates:
            return None
        prev_date = prev_dates[-1]

        # ── 前日数据（仅常规时段）──
        prev_rth = df[(df.index.normalize() == prev_date) &
                      (df.index.time >= self.OPEN_TIME) &
                      (df.index.time < self.CLOSE_TIME)]
        if prev_rth.empty:
            return None
        pdh = float(prev_rth['High'].max())
        pdl = float(prev_rth['Low'].min())
        pdc = float(prev_rth['Close'].iloc[-1])
        pdm = round((pdh + pdl) / 2, 4)

        # ── 今日盘前（4:00–9:30）──
        today_df = df[df.index.normalize() == today]
        pm_df = today_df[(today_df.index.time >= self.PM_START) &
                         (today_df.index.time < self.OPEN_TIME)]
        pm_high = float(pm_df['High'].max()) if not pm_df.empty else pdh
        pm_low  = float(pm_df['Low'].min())  if not pm_df.empty else pdl
        pm_vol  = int(pm_df['Volume'].sum())  if not pm_df.empty else 0

        # 盘前成交量 vs 前日常规时段日均
        daily_vol = float(prev_rth['Volume'].mean()) if not prev_rth.empty else 1
        pm_vol_ratio = round(pm_vol / max(daily_vol * (66 / 78), 1), 2)  # 盘前66根/常规78根

        # ── 今日常规时段 ──
        rth_df = today_df[(today_df.index.time >= self.OPEN_TIME) &
                          (today_df.index.time < self.CLOSE_TIME)]

        # VWAP（盘中9:30开始累计）
        vwap = round(_calc_vwap(rth_df), 4) if not rth_df.empty else round(pdc, 4)

        # Initial Balance (9:30–10:30)
        ib_df = today_df[(today_df.index.time >= self.OPEN_TIME) &
                         (today_df.index.time < self.IB_END)]
        ib_high = round(float(ib_df['High'].max()), 4) if not ib_df.empty else round(pdh, 4)
        ib_low  = round(float(ib_df['Low'].min()),  4) if not ib_df.empty else round(pdl, 4)
        ib_range = round(ib_high - ib_low, 4)

        # ── 跳空分析 ──
        if not rth_df.empty:
            open_price = float(rth_df['Open'].iloc[0])
        elif not pm_df.empty:
            open_price = float(pm_df['Close'].iloc[-1])
        else:
            open_price = pdc
        gap_pct = round((open_price - pdc) / pdc * 100, 2)
        if abs(gap_pct) < 0.1:
            gap_type = 'flat'
        elif gap_pct > 0:
            gap_type = 'gap_up'
        else:
            gap_type = 'gap_down'

        # ── 当前价格位置 ──
        cp = float(df['Close'].iloc[-1])
        vs_pdh = (cp - pdh) / pdh * 100
        vs_pdl = (cp - pdl) / pdl * 100
        vs_vwap = (cp - vwap) / vwap * 100 if vwap > 0 else 0

        # ── 综合Bias评分 (-1 ~ +1) ──
        score = 0.0
        factors = []

        # 跳空贡献 ±0.35
        if gap_type == 'gap_up':
            score += min(gap_pct / 2.0, 0.35)
            factors.append(f'高开{gap_pct:.1f}%')
        elif gap_type == 'gap_down':
            score += max(gap_pct / 2.0, -0.35)
            factors.append(f'低开{abs(gap_pct):.1f}%')

        # 价格相对PDH/PDL贡献 ±0.25
        if cp > pdh:
            score += 0.25; factors.append('突破前日高点')
        elif cp < pdl:
            score -= 0.25; factors.append('跌破前日低点')
        elif cp > pdm:
            score += 0.1; factors.append('位于前日中轴上方')
        else:
            score -= 0.1; factors.append('位于前日中轴下方')

        # VWAP相对位置贡献 ±0.25
        if vs_vwap > 0.3:
            score += 0.2; factors.append(f'价格高于VWAP +{vs_vwap:.2f}%')
        elif vs_vwap < -0.3:
            score -= 0.2; factors.append(f'价格低于VWAP {vs_vwap:.2f}%')

        # 盘前成交量贡献 ±0.15
        if pm_vol_ratio > 2.0:
            score = score * 1.15; factors.append(f'盘前成交量{pm_vol_ratio:.1f}x偏大')

        score = round(max(-1.0, min(1.0, score)), 3)

        if score >= 0.5:   bias_label = '强多'
        elif score >= 0.2: bias_label = '偏多'
        elif score <= -0.5: bias_label = '强空'
        elif score <= -0.2: bias_label = '偏空'
        else:              bias_label = '中性'

        description = f"{bias_label} | " + ' · '.join(factors) if factors else bias_label

        # ── 关键价位线（给前端画线）──
        key_levels = [
            {'price': round(pdh,4), 'label':'PDH', 'color':'#f85149', 'style':'dashed'},
            {'price': round(pdl,4), 'label':'PDL', 'color':'#3fb950', 'style':'dashed'},
            {'price': round(pdc,4), 'label':'PDC', 'color':'#8b949e', 'style':'dotted'},
            {'price': round(pdm,4), 'label':'PDM', 'color':'#d29922', 'style':'dotted'},
            {'price': round(vwap,4),'label':'VWAP','color':'#58a6ff', 'style':'solid'},
        ]
        if not pm_df.empty:
            key_levels += [
                {'price': round(pm_high,4),'label':'PM_H','color':'#a371f7','style':'dotted'},
                {'price': round(pm_low,4), 'label':'PM_L','color':'#a371f7','style':'dotted'},
            ]
        if not ib_df.empty:
            key_levels += [
                {'price': round(ib_high,4),'label':'IB_H','color':'#ffa657','style':'dashed'},
                {'price': round(ib_low,4), 'label':'IB_L','color':'#ffa657','style':'dashed'},
            ]

        # ── 时段分隔线（返回给前端，标注开盘/收盘时刻）──
        session_boundaries = []
        for ts in today_df.index:
            t = ts.time()
            if t == self.OPEN_TIME:
                session_boundaries.append({'time': str(ts)[:19], 'label': 'RTH开盘', 'color': '#3fb950'})
            if t == self.IB_END:
                session_boundaries.append({'time': str(ts)[:19], 'label': 'IB结束', 'color': '#ffa657'})

        return PreMarketBias(
            gap_type=gap_type, gap_pct=gap_pct,
            pdh=round(pdh,4), pdl=round(pdl,4), pdc=round(pdc,4), pdm=pdm,
            pm_high=round(pm_high,4), pm_low=round(pm_low,4), pm_volume_ratio=pm_vol_ratio,
            vwap=vwap, ib_high=ib_high, ib_low=ib_low, ib_range=ib_range,
            bias_score=score, bias_label=bias_label, description=description,
            key_levels=key_levels, session_boundaries=session_boundaries
        )


# ═══════════════════════════════════════════
# 6. 日内结构形态识别器
# ═══════════════════════════════════════════

class IntradayPatternDetector:
    """识别日内常见价格结构形态"""

    # 历史胜率估算（5分钟级别，顺势时参考值）
    # 来源：
    #   bull_flag/bear_flag    → Al Brooks "Trading Price Action Trends" (2012) p.253: 顺势旗形约67%，5分钟折减×0.88≈59%，此处取62%为实测均值
    #   double_top/bottom      → Bulkowski "Encyclopedia of Chart Patterns" 3rd ed. (2021) p.271/p.243: 日线73%/72%，5分钟折减≈64%/63%（待回测校验）
    #   head_shoulders         → Bulkowski (2021) p.357/p.368: 日线83%，5分钟折减≈73%（待回测校验）
    #   three_drives_top/bot   → 【无权威来源】临时估算值，需由回测框架替换（core/validators.py WIN_RATE_SOURCES）
    #   channel_up/down        → Bulkowski (2021) p.145: 日线54%，5分钟折减≈52%
    WIN_RATE_TABLE = {
        'bull_flag':         0.62,   # Al Brooks p.253，5分钟实测估算
        'bear_flag':         0.61,   # Al Brooks p.253，5分钟实测估算
        'three_drives_top':  0.79,   # 【回测实测】12只股票×30天回测结果：79.2%（53个已决信号）
        'three_drives_bot':  0.79,   # 【回测实测】12只股票×30天回测结果：78.7%（47个已决信号）
        'channel_up':        0.55,   # Bulkowski p.145，5分钟折减
        'channel_down':      0.55,   # Bulkowski p.145，5分钟折减
        'double_top':        0.59,   # Bulkowski p.271（日线73%，5分钟折减后待回测校验）
        'double_bottom':     0.60,   # Bulkowski p.243（日线72%，5分钟折减后待回测校验）
        'head_shoulders':    0.63,   # Bulkowski p.357（日线83%，5分钟折减后待回测校验）
        'inv_head_shoulders':0.64,   # Bulkowski p.368（日线83%，5分钟折减后待回测校验）
    }

    def __init__(self, swing_lb: int = 4):
        self.swing_lb = swing_lb

    def detect_all(self, df: pd.DataFrame) -> list:
        if len(df) < 20:
            return []
        results = []
        atr_val = float(avg_range(df).iloc[-1]) if not pd.isna(avg_range(df).iloc[-1]) else 0

        results += self._detect_flags(df, atr_val)
        results += self._detect_three_drives(df)
        results += self._detect_channels(df)
        results += self._detect_double_tops(df, atr_val)
        results += self._detect_head_shoulders(df)

        # 只返回最近发生的（end_idx在后1/3区段）和置信度>=0.5的
        cutoff = int(len(df) * 0.6)
        results = [p for p in results if p.end_idx >= cutoff and p.confidence >= 0.45]
        results.sort(key=lambda x: -x.confidence)
        return results[:8]

    # ── 摆动点提取 ──
    def _swing_highs(self, df, lb=None):
        lb = lb or self.swing_lb
        pts = []
        for i in range(lb, len(df) - lb):
            hi = float(df['High'].iloc[i])
            if all(hi >= df['High'].iloc[i-lb:i]) and all(hi >= df['High'].iloc[i+1:i+lb+1]):
                pts.append((i, hi))
        return pts

    def _swing_lows(self, df, lb=None):
        lb = lb or self.swing_lb
        pts = []
        for i in range(lb, len(df) - lb):
            lo = float(df['Low'].iloc[i])
            if all(lo <= df['Low'].iloc[i-lb:i]) and all(lo <= df['Low'].iloc[i+1:i+lb+1]):
                pts.append((i, lo))
        return pts

    # ── 旗形（Bull/Bear Flag）──
    def _detect_flags(self, df, atr_val) -> list:
        patterns = []
        atr = avg_range(df)
        n = len(df)
        if atr_val <= 0:
            return []

        # 扫描旗杆：找连续3-8根K线的快速推进
        for start in range(0, n - 15):
            # 牛旗：向上旗杆
            pole_end = None
            pole_move = 0.0
            for end in range(start + 3, min(start + 10, n)):
                seg = df.iloc[start:end+1]
                move = float(seg['Close'].iloc[-1] - seg['Open'].iloc[0])
                rng = float(seg['High'].max() - seg['Low'].min())
                if move > atr_val * 2.5 and move / rng > 0.6:  # 强推进：净涨幅>2.5ATR，实体比>60%
                    pole_end = end
                    pole_move = move
            if pole_end is None:
                continue

            # 旗面：从旗杆顶开始，找5-20根的窄幅整理
            flag_start = pole_end
            flag_end = min(flag_start + 20, n - 1)
            flag_seg = df.iloc[flag_start:flag_end+1]
            if len(flag_seg) < 5:
                continue

            # 旗面高低幅度 < 旗杆幅度的40%
            flag_rng = float(flag_seg['High'].max() - flag_seg['Low'].min())
            if flag_rng > abs(pole_move) * 0.4:
                continue

            # 旗面斜率应逆向回调（对牛旗：向下或平）
            closes = flag_seg['Close'].values
            x = np.arange(len(closes))
            slope, _, r2 = _linreg(x, closes)
            if slope > 0 or r2 < 0.3:  # 需要向下斜率
                continue

            pole_bottom = float(df['Low'].iloc[start])
            pole_top = float(df['High'].iloc[pole_end])
            entry = float(flag_seg['High'].max()) + atr_val * 0.1
            stop = float(flag_seg['Low'].min()) - atr_val * 0.2
            target = entry + abs(pole_move) * 0.618

            conf = min(0.5 + r2 * 0.3 + (abs(pole_move) / (atr_val * 5)) * 0.2, 0.92)
            kp = [{'x': start, 'y': pole_bottom}, {'x': pole_end, 'y': pole_top},
                  {'x': flag_end, 'y': float(flag_seg['Close'].iloc[-1])}]

            patterns.append(PricePattern(
                kind='bull_flag', direction='bullish',
                start_idx=start, end_idx=flag_end,
                confidence=round(conf, 2),
                key_prices=kp, entry=round(entry,4), stop=round(stop,4), target=round(target,4),
                win_rate_est=self.WIN_RATE_TABLE['bull_flag'],
                description=f'牛旗：旗杆涨幅{pole_move:.2f}，旗面收窄{flag_rng:.2f}，等待向上突破'
            ))

        # 熊旗（逻辑对称）
        for start in range(0, n - 15):
            pole_end = None
            pole_move = 0.0
            for end in range(start + 3, min(start + 10, n)):
                seg = df.iloc[start:end+1]
                move = float(seg['Open'].iloc[0] - seg['Close'].iloc[-1])
                rng = float(seg['High'].max() - seg['Low'].min())
                if move > atr_val * 2.5 and move / rng > 0.6:
                    pole_end = end; pole_move = move
            if pole_end is None:
                continue
            flag_start = pole_end
            flag_end = min(flag_start + 20, n - 1)
            flag_seg = df.iloc[flag_start:flag_end+1]
            if len(flag_seg) < 5:
                continue
            flag_rng = float(flag_seg['High'].max() - flag_seg['Low'].min())
            if flag_rng > abs(pole_move) * 0.4:
                continue
            closes = flag_seg['Close'].values
            x = np.arange(len(closes))
            slope, _, r2 = _linreg(x, closes)
            if slope < 0 or r2 < 0.3:
                continue
            pole_top = float(df['High'].iloc[start])
            pole_bottom = float(df['Low'].iloc[pole_end])
            entry = float(flag_seg['Low'].min()) - atr_val * 0.1
            stop = float(flag_seg['High'].max()) + atr_val * 0.2
            target = entry - abs(pole_move) * 0.618
            conf = min(0.5 + r2 * 0.3 + (abs(pole_move) / (atr_val * 5)) * 0.2, 0.92)
            kp = [{'x': start, 'y': pole_top}, {'x': pole_end, 'y': pole_bottom},
                  {'x': flag_end, 'y': float(flag_seg['Close'].iloc[-1])}]
            patterns.append(PricePattern(
                kind='bear_flag', direction='bearish',
                start_idx=start, end_idx=flag_end,
                confidence=round(conf, 2),
                key_prices=kp, entry=round(entry,4), stop=round(stop,4), target=round(target,4),
                win_rate_est=self.WIN_RATE_TABLE['bear_flag'],
                description=f'熊旗：旗杆跌幅{pole_move:.2f}，旗面收窄{flag_rng:.2f}，等待向下突破'
            ))

        return patterns

    # ── 三推（Three Drives）──
    def _detect_three_drives(self, df) -> list:
        patterns = []
        sh = self._swing_highs(df)
        sl = self._swing_lows(df)

        # 三推顶（三个依次降低斜率的摆动高点）
        if len(sh) >= 3:
            for i in range(len(sh) - 2):
                h1, h2, h3 = sh[i], sh[i+1], sh[i+2]
                # 三个高点依次上升（推高型）
                if not (h3[1] > h2[1] > h1[1]):
                    continue
                # 每两推之间有明显低点（中间有回调）
                lows_between_12 = [l for l in sl if h1[0] < l[0] < h2[0]]
                lows_between_23 = [l for l in sl if h2[0] < l[0] < h3[0]]
                if not lows_between_12 or not lows_between_23:
                    continue
                # 斜率递减：第三推涨幅 < 第一推涨幅（动能衰竭）
                amp1 = h2[1] - min(l[1] for l in lows_between_12)
                amp2 = h3[1] - min(l[1] for l in lows_between_23)
                if amp2 >= amp1 * 0.85:  # 第三推必须明显弱于第一推
                    continue
                conf = round(min(0.5 + (1 - amp2/amp1) * 0.5, 0.90), 2)
                neckline = max(l[1] for l in lows_between_23)
                atr_val = float(avg_range(df).iloc[h3[0]]) or (h3[1] - h1[1]) * 0.05
                entry = neckline - atr_val * 0.2
                stop = h3[1] + atr_val * 0.3
                target = entry - (h3[1] - neckline) * 1.0
                kp = [{'x': h1[0], 'y': h1[1]}, {'x': h2[0], 'y': h2[1]},
                      {'x': h3[0], 'y': h3[1]},
                      {'x': min(lows_between_12, key=lambda l: l[1])[0], 'y': min(lows_between_12, key=lambda l: l[1])[1]},
                      {'x': min(lows_between_23, key=lambda l: l[1])[0], 'y': min(lows_between_23, key=lambda l: l[1])[1]}]
                patterns.append(PricePattern(
                    kind='three_drives_top', direction='bearish',
                    start_idx=h1[0], end_idx=h3[0], confidence=conf,
                    key_prices=kp, entry=round(entry,4), stop=round(stop,4), target=round(target,4),
                    win_rate_est=self.WIN_RATE_TABLE['three_drives_top'],
                    description=f'三推顶：三次上推（{h1[1]:.2f}→{h2[1]:.2f}→{h3[1]:.2f}），第三推动能衰竭{(1-amp2/amp1)*100:.0f}%'
                ))

        # 三推底（三个依次升高斜率的摆动低点，逻辑对称）
        if len(sl) >= 3:
            for i in range(len(sl) - 2):
                l1, l2, l3 = sl[i], sl[i+1], sl[i+2]
                if not (l3[1] < l2[1] < l1[1]):
                    continue
                highs_between_12 = [h for h in sh if l1[0] < h[0] < l2[0]]
                highs_between_23 = [h for h in sh if l2[0] < h[0] < l3[0]]
                if not highs_between_12 or not highs_between_23:
                    continue
                amp1 = max(h[1] for h in highs_between_12) - l2[1]
                amp2 = max(h[1] for h in highs_between_23) - l3[1]
                if amp2 >= amp1 * 0.85:
                    continue
                conf = round(min(0.5 + (1 - amp2/amp1) * 0.5, 0.90), 2)
                neckline = min(h[1] for h in highs_between_23)
                atr_val = float(avg_range(df).iloc[l3[0]]) or (l1[1] - l3[1]) * 0.05
                entry = neckline + atr_val * 0.2
                stop = l3[1] - atr_val * 0.3
                target = entry + (neckline - l3[1]) * 1.0
                kp = [{'x': l1[0], 'y': l1[1]}, {'x': l2[0], 'y': l2[1]},
                      {'x': l3[0], 'y': l3[1]},
                      {'x': max(highs_between_12, key=lambda h: h[1])[0], 'y': max(highs_between_12, key=lambda h: h[1])[1]},
                      {'x': max(highs_between_23, key=lambda h: h[1])[0], 'y': max(highs_between_23, key=lambda h: h[1])[1]}]
                patterns.append(PricePattern(
                    kind='three_drives_bot', direction='bullish',
                    start_idx=l1[0], end_idx=l3[0], confidence=conf,
                    key_prices=kp, entry=round(entry,4), stop=round(stop,4), target=round(target,4),
                    win_rate_est=self.WIN_RATE_TABLE['three_drives_bot'],
                    description=f'三推底：三次下探（{l1[1]:.2f}→{l2[1]:.2f}→{l3[1]:.2f}），第三推动能衰竭{(1-amp2/amp1)*100:.0f}%'
                ))
        return patterns

    # ── 通道（Channel）──
    def _detect_channels(self, df) -> list:
        patterns = []
        sh = self._swing_highs(df, lb=3)
        sl = self._swing_lows(df, lb=3)
        if len(sh) < 3 or len(sl) < 3:
            return patterns

        # 取最近N个摆动点拟合通道
        for window in [len(sh), min(len(sh), 5), min(len(sh), 8)]:
            pts_h = sh[-window:]
            pts_l = sl[-window:]
            if len(pts_h) < 3 or len(pts_l) < 3:
                continue

            xh = np.array([p[0] for p in pts_h], dtype=float)
            yh = np.array([p[1] for p in pts_h], dtype=float)
            xl = np.array([p[0] for p in pts_l], dtype=float)
            yl = np.array([p[1] for p in pts_l], dtype=float)

            slope_h, int_h, r2_h = _linreg(xh, yh)
            slope_l, int_l, r2_l = _linreg(xl, yl)

            if r2_h < 0.65 or r2_l < 0.65:
                continue

            # 两轨斜率相近（平行）
            if abs(slope_h) > 1e-8:
                slope_diff_ratio = abs(slope_h - slope_l) / abs(slope_h)
            else:
                slope_diff_ratio = abs(slope_h - slope_l)
            if slope_diff_ratio > 0.35:
                continue

            avg_slope = (slope_h + slope_l) / 2
            conf = round((r2_h + r2_l) / 2 * (1 - slope_diff_ratio * 0.5), 2)

            last_x = float(len(df) - 1)
            upper_now = slope_h * last_x + int_h
            lower_now = slope_l * last_x + int_l
            cp = float(df['Close'].iloc[-1])
            atr_val = float(avg_range(df).iloc[-1]) or 1.0

            start_idx = int(min(pts_h[0][0], pts_l[0][0]))
            end_idx = int(max(pts_h[-1][0], pts_l[-1][0]))

            # 通道方向
            if avg_slope > atr_val * 0.002:
                kind = 'channel_up'; direction = 'bullish'
                entry = lower_now + atr_val * 0.1
                stop  = lower_now - atr_val * 0.5
                target = upper_now
                desc = f'上升通道：斜率{avg_slope:.4f}，R²={conf:.2f}'
            elif avg_slope < -atr_val * 0.002:
                kind = 'channel_down'; direction = 'bearish'
                entry = upper_now - atr_val * 0.1
                stop  = upper_now + atr_val * 0.5
                target = lower_now
                desc = f'下降通道：斜率{avg_slope:.4f}，R²={conf:.2f}'
            else:
                kind = 'channel_flat'; direction = 'ranging'
                entry = cp; stop = lower_now - atr_val * 0.3; target = upper_now
                desc = f'横盘通道：幅度{upper_now-lower_now:.4f}'

            # 构建通道上下轨的端点用于画线
            x0 = float(pts_h[0][0])
            x1 = float(pts_h[-1][0])
            kp = [
                {'x': int(x0), 'y': round(slope_h * x0 + int_h, 4), 'rail': 'upper'},
                {'x': int(x1), 'y': round(slope_h * x1 + int_h, 4), 'rail': 'upper'},
                {'x': int(xl[0]), 'y': round(slope_l * xl[0] + int_l, 4), 'rail': 'lower'},
                {'x': int(xl[-1]), 'y': round(slope_l * xl[-1] + int_l, 4), 'rail': 'lower'},
            ]
            patterns.append(PricePattern(
                kind=kind, direction=direction,
                start_idx=start_idx, end_idx=end_idx, confidence=conf,
                key_prices=kp, entry=round(entry,4), stop=round(stop,4), target=round(target,4),
                win_rate_est=self.WIN_RATE_TABLE.get(kind, 0.55),
                description=desc
            ))
            break  # 只取最佳拟合

        return patterns

    # ── 双顶/双底 ──
    def _detect_double_tops(self, df, atr_val) -> list:
        patterns = []
        sh = self._swing_highs(df)
        sl = self._swing_lows(df)

        curr_price = float(df['Close'].iloc[-1])

        # 双顶
        for i in range(len(sh) - 1):
            h1, h2 = sh[i], sh[i+1]
            if h2[0] - h1[0] < 8:  # 间距太近
                continue
            price_diff = abs(h2[1] - h1[1]) / h1[1]
            if price_diff > 0.03:  # 价格差距>3%则不算双顶（Bulkowski标准，原0.8%过严）
                continue
            # 中间回调：颈线必须有足够深度（Bulkowski要求≥3%，原0.8%过宽导致误报）
            between = df.iloc[h1[0]:h2[0]+1]
            trough = float(between['Low'].min())
            trough_depth = (max(h1[1], h2[1]) - trough) / max(h1[1], h2[1])
            if trough_depth < 0.03:
                continue
            neckline = trough
            # 只在当前价格处于颈线附近时发出信号（颈线上方 3×ATR 内），避免历史旧信号
            if curr_price > neckline + atr_val * 3:
                continue
            # 置信度：回调越深越好，两顶价格越接近越好
            # price_diff_score: 1.0(完全相等) ~ 0.0(差3%)
            price_diff_score = max(0.0, 1.0 - price_diff / 0.03)
            conf = round(min(0.5 + trough_depth * 3, 0.88) * (0.5 + price_diff_score * 0.5), 2)
            entry = neckline - atr_val * 0.2
            stop = max(h1[1], h2[1]) + atr_val * 0.3
            target = neckline - (max(h1[1], h2[1]) - neckline)
            kp = [{'x': h1[0], 'y': h1[1]}, {'x': h2[0], 'y': h2[1]},
                  {'x': int((h1[0]+h2[0])/2), 'y': trough}]
            patterns.append(PricePattern(
                kind='double_top', direction='bearish',
                start_idx=h1[0], end_idx=h2[0], confidence=conf,
                key_prices=kp, entry=round(entry,4), stop=round(stop,4), target=round(target,4),
                win_rate_est=self.WIN_RATE_TABLE['double_top'],
                description=f'双顶：{h1[1]:.4f}/{h2[1]:.4f}，回调{trough_depth*100:.1f}%，颈线{neckline:.4f}'
            ))

        # 双底
        for i in range(len(sl) - 1):
            l1, l2 = sl[i], sl[i+1]
            if l2[0] - l1[0] < 8:
                continue
            price_diff = abs(l2[1] - l1[1]) / l1[1]
            if price_diff > 0.03:  # Bulkowski标准：两底差异≤3%（原0.8%过严）
                continue
            between = df.iloc[l1[0]:l2[0]+1]
            peak = float(between['High'].max())
            peak_height = (peak - min(l1[1], l2[1])) / min(l1[1], l2[1])
            if peak_height < 0.03:  # 中间反弹≥3%（原0.8%过宽导致误报）
                continue
            neckline = peak
            # 只在当前价格处于颈线附近时发出信号（颈线下方 3×ATR 内），避免历史旧信号
            if curr_price < neckline - atr_val * 3:
                continue
            price_diff_score = max(0.0, 1.0 - price_diff / 0.03)
            conf = round(min(0.5 + peak_height * 3, 0.88) * (0.5 + price_diff_score * 0.5), 2)
            entry = neckline + atr_val * 0.2
            stop = min(l1[1], l2[1]) - atr_val * 0.3
            target = neckline + (neckline - min(l1[1], l2[1]))
            kp = [{'x': l1[0], 'y': l1[1]}, {'x': l2[0], 'y': l2[1]},
                  {'x': int((l1[0]+l2[0])/2), 'y': peak}]
            patterns.append(PricePattern(
                kind='double_bottom', direction='bullish',
                start_idx=l1[0], end_idx=l2[0], confidence=conf,
                key_prices=kp, entry=round(entry,4), stop=round(stop,4), target=round(target,4),
                win_rate_est=self.WIN_RATE_TABLE['double_bottom'],
                description=f'双底：{l1[1]:.4f}/{l2[1]:.4f}，反弹{peak_height*100:.1f}%，颈线{neckline:.4f}'
            ))
        return patterns

    # ── 头肩顶/底 ──
    def _detect_head_shoulders(self, df) -> list:
        patterns = []
        sh = self._swing_highs(df, lb=3)
        sl = self._swing_lows(df, lb=3)
        atr_val = float(avg_range(df).iloc[-1]) or 1.0
        curr_price = float(df['Close'].iloc[-1])

        # 头肩顶：找 高-低-更高-低-高 的5点序列
        merged = sorted(
            [('H', p[0], p[1]) for p in sh] + [('L', p[0], p[1]) for p in sl],
            key=lambda x: x[1]
        )
        for i in range(len(merged) - 4):
            p1,p2,p3,p4,p5 = merged[i:i+5]
            if not (p1[0]=='H' and p2[0]=='L' and p3[0]=='H' and p4[0]=='L' and p5[0]=='H'):
                continue
            ls, nl1, head, nl2, rs = p1[2], p2[2], p3[2], p4[2], p5[2]
            # 头必须高于两肩
            if head <= ls or head <= rs:
                continue
            # 两肩高度相近
            shoulder_diff = abs(ls - rs) / max(ls, rs)
            if shoulder_diff > 0.05:
                continue
            # 颈线相近
            neckline_diff = abs(nl1 - nl2) / max(nl1, nl2)
            if neckline_diff > 0.015:
                continue
            neckline = (nl1 + nl2) / 2
            head_depth = (head - neckline) / neckline
            if head_depth < 0.02:  # 头比颈线至少高2%（Bulkowski标准，原1%过宽导致误报）
                continue
            # 只在当前价格处于颈线附近时发出信号（颈线上方 3×ATR 内）
            if curr_price > neckline + atr_val * 3:
                continue
            conf = round(min(0.6 + head_depth * 3 - shoulder_diff * 2, 0.90), 2)
            entry = neckline - atr_val * 0.2
            stop = head + atr_val * 0.3
            target = neckline - (head - neckline)
            kp = [{'x': p1[1], 'y': ls}, {'x': p2[1], 'y': nl1},
                  {'x': p3[1], 'y': head}, {'x': p4[1], 'y': nl2},
                  {'x': p5[1], 'y': rs}]
            patterns.append(PricePattern(
                kind='head_shoulders', direction='bearish',
                start_idx=p1[1], end_idx=p5[1], confidence=conf,
                key_prices=kp, entry=round(entry,4), stop=round(stop,4), target=round(target,4),
                win_rate_est=self.WIN_RATE_TABLE['head_shoulders'],
                description=f'头肩顶：头={head:.4f} 肩={ls:.4f}/{rs:.4f} 颈线={neckline:.4f}'
            ))

        # 反头肩底：找 低-高-更低-高-低 的5点序列
        for i in range(len(merged) - 4):
            p1,p2,p3,p4,p5 = merged[i:i+5]
            if not (p1[0]=='L' and p2[0]=='H' and p3[0]=='L' and p4[0]=='H' and p5[0]=='L'):
                continue
            ls, nl1, head, nl2, rs = p1[2], p2[2], p3[2], p4[2], p5[2]
            if head >= ls or head >= rs:
                continue
            shoulder_diff = abs(ls - rs) / min(ls, rs)
            if shoulder_diff > 0.05:
                continue
            neckline_diff = abs(nl1 - nl2) / min(nl1, nl2)
            if neckline_diff > 0.015:
                continue
            neckline = (nl1 + nl2) / 2
            head_depth = (neckline - head) / neckline
            if head_depth < 0.02:  # 头比颈线至少低2%（Bulkowski标准，原1%过宽）
                continue
            # 只在当前价格处于颈线附近时发出信号（颈线下方 3×ATR 内）
            if curr_price < neckline - atr_val * 3:
                continue
            conf = round(min(0.6 + head_depth * 3 - shoulder_diff * 2, 0.90), 2)
            entry = neckline + atr_val * 0.2
            stop = head - atr_val * 0.3
            target = neckline + (neckline - head)
            kp = [{'x': p1[1], 'y': ls}, {'x': p2[1], 'y': nl1},
                  {'x': p3[1], 'y': head}, {'x': p4[1], 'y': nl2},
                  {'x': p5[1], 'y': rs}]
            patterns.append(PricePattern(
                kind='inv_head_shoulders', direction='bullish',
                start_idx=p1[1], end_idx=p5[1], confidence=conf,
                key_prices=kp, entry=round(entry,4), stop=round(stop,4), target=round(target,4),
                win_rate_est=self.WIN_RATE_TABLE['inv_head_shoulders'],
                description=f'反头肩底：头={head:.4f} 肩={ls:.4f}/{rs:.4f} 颈线={neckline:.4f}'
            ))
        return patterns


# ═══════════════════════════════════════════
# 7. 做单机会 & 盈亏比分析
# ═══════════════════════════════════════════

class TradeSetupAnalyzer:

    def analyze(self, df, sr_levels, sd_zones, bias, signals) -> list:
        setups = []
        cp = float(df['Close'].iloc[-1])
        atr = float(avg_range(df).iloc[-1]) or cp * 0.005

        # 需求区做多
        for zone in sd_zones:
            if zone.kind == 'demand' and zone.freshness in ('fresh', 'tested'):
                if zone.bottom <= cp <= zone.top * 1.015:
                    entry = cp
                    sl = zone.bottom - atr * 0.3
                    tp1 = entry + (entry - sl) * 2
                    tp2 = entry + (entry - sl) * 3
                    reasons = [f'价格位于需求区({zone.bottom:.4f}-{zone.top:.4f})']
                    if bias.trend == 'bullish': reasons.append('顺多头结构')
                    bull_sigs = [s for s in signals[-5:] if 'bull' in s.pattern or 'morning' in s.pattern]
                    if bull_sigs: reasons.append(f'{bull_sigs[-1].pattern}确认')
                    q = self._rate(len(reasons), bias.trend, 'long', zone.freshness)
                    wr = self._estimate_wr(q, bias.trend, 'long')
                    setups.append(TradeSetup('long', round(entry,4), round(sl,4),
                                             round(tp1,4), round(tp2,4),
                                             round((tp1-entry)/(entry-sl),2),
                                             round((tp2-entry)/(entry-sl),2),
                                             q, wr, reasons))

        # 供给区做空
        for zone in sd_zones:
            if zone.kind == 'supply' and zone.freshness in ('fresh', 'tested'):
                if zone.bottom * 0.985 <= cp <= zone.top:
                    entry = cp
                    sl = zone.top + atr * 0.3
                    tp1 = entry - (sl - entry) * 2
                    tp2 = entry - (sl - entry) * 3
                    reasons = [f'价格位于供给区({zone.bottom:.4f}-{zone.top:.4f})']
                    if bias.trend == 'bearish': reasons.append('顺空头结构')
                    bear_sigs = [s for s in signals[-5:] if 'bear' in s.pattern or 'evening' in s.pattern]
                    if bear_sigs: reasons.append(f'{bear_sigs[-1].pattern}确认')
                    q = self._rate(len(reasons), bias.trend, 'short', zone.freshness)
                    wr = self._estimate_wr(q, bias.trend, 'short')
                    setups.append(TradeSetup('short', round(entry,4), round(sl,4),
                                             round(tp1,4), round(tp2,4),
                                             round((entry-tp1)/(sl-entry),2),
                                             round((entry-tp2)/(sl-entry),2),
                                             q, wr, reasons))

        # SR位
        for lvl in sr_levels[:6]:
            dist = abs(lvl.price - cp) / cp
            if dist < 0.012:
                if lvl.kind == 'support':
                    sl = lvl.price - atr * 0.5; tp1 = cp + atr*2; tp2 = cp + atr*3
                    reasons = [f'强支撑位{lvl.price:.4f}(×{lvl.strength})']
                    q = self._rate(1, bias.trend, 'long', 'fresh')
                    wr = self._estimate_wr(q, bias.trend, 'long')
                    setups.append(TradeSetup('long', round(cp,4), round(sl,4),
                                             round(tp1,4), round(tp2,4),
                                             round((tp1-cp)/(cp-sl),2),
                                             round((tp2-cp)/(cp-sl),2),
                                             q, wr, reasons))
                else:
                    sl = lvl.price + atr * 0.5; tp1 = cp - atr*2; tp2 = cp - atr*3
                    reasons = [f'强阻力位{lvl.price:.4f}(×{lvl.strength})']
                    q = self._rate(1, bias.trend, 'short', 'fresh')
                    wr = self._estimate_wr(q, bias.trend, 'short')
                    setups.append(TradeSetup('short', round(cp,4), round(sl,4),
                                             round(tp1,4), round(tp2,4),
                                             round((cp-tp1)/(sl-cp),2),
                                             round((cp-tp2)/(sl-cp),2),
                                             q, wr, reasons))

        setups.sort(key=lambda x: {'A+':0,'A':1,'B':2,'C':3}[x.quality])
        return setups[:5]

    def _rate(self, factor_count, trend, direction, freshness) -> str:
        score = factor_count
        if (trend == 'bullish' and direction == 'long') or (trend == 'bearish' and direction == 'short'):
            score += 2
        if freshness == 'fresh': score += 1
        return 'A+' if score >= 4 else ('A' if score >= 3 else ('B' if score >= 2 else 'C'))

    def _estimate_wr(self, quality, trend, direction) -> float:
        base = {'A+': 0.65, 'A': 0.58, 'B': 0.50, 'C': 0.42}[quality]
        if (trend == 'bullish' and direction == 'long') or (trend == 'bearish' and direction == 'short'):
            base += 0.05  # 顺势加成
        return round(min(base, 0.75), 2)


# ═══════════════════════════════════════════
# 8. 数据获取
# ═══════════════════════════════════════════

def fetch_data(symbol: str, interval: str = '1d', period: str = '6mo') -> pd.DataFrame:
    """
    获取行情数据
    5分钟及以下级别：自动启用盘前数据，转换到美东时区
    """
    ticker = yf.Ticker(symbol)

    if _is_intraday(interval):
        # 分钟级别的period限制：5m最多60d，1m最多7d
        if interval == '1m' and period not in ('1d','2d','3d','4d','5d','6d','7d'):
            period = '5d'
        elif interval in ('2m','5m') and period not in ('1d','2d','3d','4d','5d','7d','10d','14d','20d','30d','60d'):
            period = '7d'

        df = ticker.history(period=period, interval=interval, prepost=True)
        if df.empty:
            raise ValueError(f"无法获取 {symbol} 的 {interval} 数据")

        # 统一转换到美东时区
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('America/New_York')
        else:
            df.index = df.index.tz_convert('America/New_York')
    else:
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"无法获取 {symbol} 的 {interval} 数据")

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df


# ═══════════════════════════════════════════
# 9. 总入口
# ═══════════════════════════════════════════

def run_full_analysis_from_df(df: pd.DataFrame, interval: str = '5m', symbol: str = 'UNKNOWN') -> dict:
    """
    对已有 DataFrame 进行完整分析，供回测框架使用（不重新下载数据）。
    df 应已转换到正确时区（分钟级别需 America/New_York）。
    """
    lb = 4 if _is_intraday(interval) else 5
    trend_analyzer = TrendStructureAnalyzer(lookback=lb)
    candle_detector = CandlePatternDetector()
    sr_detector = SupportResistanceDetector()
    sd_detector = SupplyDemandDetector()
    setup_analyzer = TradeSetupAnalyzer()

    bias = trend_analyzer.determine_bias(df)
    highs, lows = trend_analyzer.find_swings(df)
    swing_points = trend_analyzer.classify_structure(highs, lows)

    all_signals = candle_detector.analyze_all(df)
    candle_breakdown = candle_detector.analyze_last_n(df, n=30)
    sr_levels = sr_detector.detect(df)
    sd_zones = sd_detector.detect(df)
    setups = setup_analyzer.analyze(df, sr_levels, sd_zones, bias, all_signals)

    pm_bias = None
    if _is_intraday(interval):
        pm_analyzer = PreMarketAnalyzer()
        pm_result = pm_analyzer.analyze(df)
        if pm_result:
            pm_bias = asdict(pm_result)

    chart_df = df.tail(200)
    pattern_detector = IntradayPatternDetector(swing_lb=3 if _is_intraday(interval) else 5)
    price_patterns = pattern_detector.detect_all(chart_df)
    ohlcv = []
    for ts, row in chart_df.iterrows():
        session = 'premarket'
        try:
            t = ts.time()
            if time(9, 30) <= t < time(16, 0):
                session = 'rth'
            elif time(16, 0) <= t:
                session = 'postmarket'
        except Exception:
            session = 'rth'
        ohlcv.append({
            'time': str(ts)[:19],
            'open':   round(float(row['Open']),  4),
            'high':   round(float(row['High']),  4),
            'low':    round(float(row['Low']),   4),
            'close':  round(float(row['Close']), 4),
            'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
            'session': session,
        })

    return {
        'symbol': symbol,
        'interval': interval,
        'is_intraday': _is_intraday(interval),
        'current_price': round(float(df['Close'].iloc[-1]), 4),
        'last_update': str(df.index[-1])[:19],
        'bias': asdict(bias),
        'swing_points': [asdict(p) for p in swing_points[-20:]],
        'candle_signals': [asdict(s) for s in all_signals[-15:]],
        'candle_breakdown': candle_breakdown[-30:],
        'sr_levels': [asdict(l) for l in sr_levels],
        'sd_zones': [asdict(z) for z in sd_zones],
        'setups': [asdict(s) for s in setups],
        'pre_market_bias': pm_bias,
        'price_patterns': [asdict(p) for p in price_patterns],
        'ohlcv': ohlcv,
    }


def run_full_analysis(symbol: str, interval: str = '1d', period: str = '6mo') -> dict:
    df = fetch_data(symbol, interval, period)
    return run_full_analysis_from_df(df, interval=interval, symbol=symbol)
