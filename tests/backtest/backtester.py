"""
历史回测引擎

对12只美股拉取历史5分钟数据，逐日截断分析，
统计每种形态信号出现后的实际胜率（WIN/LOSS/TIMEOUT）。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import traceback
from collections import defaultdict
from datetime import datetime

from core.pa_engine import fetch_data, run_full_analysis_from_df


BACKTEST_SYMBOLS = [
    'AAPL', 'NVDA', 'TSLA', 'MSFT', 'META',
    'SPY', 'QQQ', 'AMZN', 'GOOGL', 'AMD',
    'NFLX', 'JPM',
]


class Backtester:
    """
    滑动窗口回测器：
    - 对每只股票，拉取 lookback_days 天的5分钟历史数据
    - 对每个交易日10:00 ET截断数据，运行形态分析
    - 向前最多 forward_bars 根K线验证信号结果
    """

    def __init__(
        self,
        symbols=None,
        interval='5m',
        lookback_days=30,
        forward_bars=78,   # 最多78根5分钟K（约一个完整交易日）
        cutoff_hour=10,    # 每天截断时间（ET小时，9:30开盘后半小时有充足历史数据）
    ):
        self.symbols = symbols or BACKTEST_SYMBOLS
        self.interval = interval
        self.lookback_days = lookback_days
        self.forward_bars = forward_bars
        self.cutoff_hour = cutoff_hour

    def run(self) -> dict:
        """运行全部股票回测，返回汇总结果"""
        all_results = []
        errors = []

        for i, sym in enumerate(self.symbols):
            print(f"  [{i+1}/{len(self.symbols)}] 回测 {sym}...", end=' ', flush=True)
            try:
                results = self._backtest_symbol(sym)
                all_results.extend(results)
                print(f"找到 {len(results)} 个信号")
            except Exception as e:
                err_msg = f"{sym}: {e}"
                errors.append(err_msg)
                print(f"失败: {e}")

        summary = self._aggregate(all_results)
        return {
            'results': all_results,
            'summary': summary,
            'errors': errors,
            'meta': {
                'symbols': self.symbols,
                'interval': self.interval,
                'lookback_days': self.lookback_days,
                'forward_bars': self.forward_bars,
                'total_signals': len(all_results),
                'run_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        }

    def _backtest_symbol(self, symbol: str) -> list:
        """对单只股票逐日截断分析"""
        period_str = f'{self.lookback_days}d'
        df = fetch_data(symbol, interval=self.interval, period=period_str)

        if df.empty:
            return []

        # 获取所有交易日
        dates = sorted(df.index.normalize().unique())
        results = []

        for day in dates[2:]:  # 至少需要前2天的历史数据
            try:
                day_date = day.date() if hasattr(day, 'date') else day
                # 截断点：当天 cutoff_hour:00 ET
                cutoff = pd.Timestamp(day_date, tz='America/New_York').replace(
                    hour=self.cutoff_hour, minute=0, second=0
                )
                hist = df[df.index <= cutoff]

                if len(hist) < 50:
                    continue

                # 运行分析
                analysis = run_full_analysis_from_df(hist, interval=self.interval, symbol=symbol)
                patterns = analysis.get('price_patterns', [])

                if not patterns:
                    continue

                # 后续数据用于验证
                future = df[df.index > cutoff]
                if future.empty:
                    continue

                for pat in patterns:
                    outcome = self._check_outcome(
                        future,
                        entry=pat['entry'],
                        stop=pat['stop'],
                        target=pat['target'],
                        direction=pat.get('direction', 'bullish'),
                    )
                    results.append({
                        'symbol': symbol,
                        'date': str(day_date),
                        'kind': pat['kind'],
                        'direction': pat['direction'],
                        'confidence': round(pat['confidence'], 3),
                        'win_rate_est': pat.get('win_rate_est', 0),
                        'entry': pat['entry'],
                        'stop': pat['stop'],
                        'target': pat['target'],
                        'outcome': outcome,
                    })

            except Exception:
                continue

        return results

    def _check_outcome(self, future_df, entry, stop, target, direction='bullish') -> str:
        """
        向前最多 forward_bars 根K，判断先触及目标(WIN)还是止损(LOSS)。
        都未触及则为 TIMEOUT。

        逻辑：
        1. 先等待价格触及 entry（突破确认），若超时则 TIMEOUT
        2. 入场后，先触及 target → WIN；先触及 stop → LOSS

        支持方向：
        - bullish: entry < stop？INVALID；价格向上突破 entry 后，High>=target(WIN) Low<=stop(LOSS)
        - bearish: entry > stop？INVALID；价格向下跌破 entry 后，Low<=target(WIN) High>=stop(LOSS)
        """
        is_bullish = direction == 'bullish'

        # 验证 entry/stop/target 方向一致性
        if is_bullish:
            if not (target > entry > stop):
                return 'INVALID'
        else:
            if not (target < entry < stop):
                return 'INVALID'

        rows = list(future_df.head(self.forward_bars).itertuples())
        # 阶段1：等待价格触及入场价
        entry_bar = None
        for i, row in enumerate(rows):
            if is_bullish:
                if row.High >= entry:
                    entry_bar = i
                    break
            else:
                if row.Low <= entry:
                    entry_bar = i
                    break

        if entry_bar is None:
            return 'TIMEOUT'

        # 阶段2：从入场后检测 target / stop
        for row in rows[entry_bar:]:
            if is_bullish:
                if row.High >= target:
                    return 'WIN'
                if row.Low <= stop:
                    return 'LOSS'
            else:
                if row.Low <= target:
                    return 'WIN'
                if row.High >= stop:
                    return 'LOSS'

        return 'TIMEOUT'

    def _aggregate(self, results: list) -> dict:
        """按形态汇总胜率统计"""
        stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'timeouts': 0, 'invalid': 0,
            'total_signals': 0,
        })

        for r in results:
            k = r['kind']
            stats[k]['total_signals'] += 1
            outcome = r['outcome']
            if outcome == 'WIN':
                stats[k]['wins'] += 1
            elif outcome == 'LOSS':
                stats[k]['losses'] += 1
            elif outcome == 'TIMEOUT':
                stats[k]['timeouts'] += 1
            else:
                stats[k]['invalid'] += 1

        summary = {}
        for kind, s in stats.items():
            total_decided = s['wins'] + s['losses']
            summary[kind] = {
                'total_signals': s['total_signals'],
                'wins': s['wins'],
                'losses': s['losses'],
                'timeouts': s['timeouts'],
                'invalid': s['invalid'],
                'total_decided': total_decided,
                'win_rate': round(s['wins'] / total_decided, 3) if total_decided > 0 else None,
            }

        return summary


def run_quick_backtest(symbols=None, lookback_days=20) -> dict:
    """快速回测（减少股票数量和天数，用于调试）"""
    quick_symbols = symbols or ['AAPL', 'NVDA', 'SPY', 'TSLA']
    bt = Backtester(
        symbols=quick_symbols,
        lookback_days=lookback_days,
        forward_bars=52,  # 约4小时
    )
    return bt.run()
