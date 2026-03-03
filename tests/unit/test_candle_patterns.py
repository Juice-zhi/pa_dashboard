"""
蜡烛图形态单元测试

测试 CandlePatternDetector 的6种形态识别：
- pin_bar_bull / pin_bar_bear
- engulf_bull / engulf_bear
- doji
- inside_bar
- outside_bar
- morning_star / evening_star
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from core.pa_engine import CandlePatternDetector
from tests.fixtures.candle_fixtures import (
    make_pin_bar_bull, make_pin_bar_bear,
    make_engulfing_bull, make_engulfing_bear,
    make_doji, make_inside_bar, make_outside_bar,
    make_morning_star, make_evening_star,
)


@pytest.fixture
def detector():
    return CandlePatternDetector()


class TestPinBar:

    def test_detects_bull_pin_bar(self, detector):
        """标准看涨Pin Bar应被识别（长下影线）"""
        df = make_pin_bar_bull(atr=1.0)
        signals = detector.analyze_all(df)
        patterns = [s.pattern for s in signals]
        assert 'pin_bar_bull' in patterns, f"未检测到pin_bar_bull，检测到: {patterns}"

    def test_bull_pin_bar_is_strong(self, detector):
        """下影线 > 实体3倍时，强度应为 'strong'"""
        df = make_pin_bar_bull(atr=1.0)
        signals = detector.analyze_all(df)
        pin_sigs = [s for s in signals if s.pattern == 'pin_bar_bull']
        assert any(s.strength == 'strong' for s in pin_sigs), \
            "长下影Pin Bar应为strong强度"

    def test_detects_bear_pin_bar(self, detector):
        """标准看跌Pin Bar应被识别（长上影线）"""
        df = make_pin_bar_bear(atr=1.0)
        signals = detector.analyze_all(df)
        patterns = [s.pattern for s in signals]
        assert 'pin_bar_bear' in patterns, f"未检测到pin_bar_bear，检测到: {patterns}"

    def test_no_pin_bar_for_normal_candle(self, detector):
        """普通小实体K线（无长影线）不应触发Pin Bar"""
        from tests.fixtures.candle_fixtures import make_df
        rows = []
        price = 100.0
        for _ in range(15):
            rows.append((price, price + 0.3, price - 0.3, price + 0.1, 50000))
            price += 0.1
        df = make_df(rows)
        signals = detector.analyze_all(df)
        pin_sigs = [s for s in signals if 'pin_bar' in s.pattern]
        assert len(pin_sigs) == 0, f"普通K线不应触发Pin Bar，但检测到: {[s.pattern for s in pin_sigs]}"


class TestEngulfing:

    def test_detects_bull_engulfing(self, detector):
        """标准看涨吞没应被识别"""
        df = make_engulfing_bull(atr=1.0)
        signals = detector.analyze_all(df)
        patterns = [s.pattern for s in signals]
        assert 'engulf_bull' in patterns, f"未检测到engulf_bull，检测到: {patterns}"

    def test_bull_engulfing_is_strong(self, detector):
        """吞没形态应标注为 strong"""
        df = make_engulfing_bull(atr=1.0)
        signals = detector.analyze_all(df)
        eng = [s for s in signals if s.pattern == 'engulf_bull']
        assert any(s.strength == 'strong' for s in eng), "吞没形态应为strong"

    def test_detects_bear_engulfing(self, detector):
        """标准看跌吞没应被识别"""
        df = make_engulfing_bear(atr=1.0)
        signals = detector.analyze_all(df)
        patterns = [s.pattern for s in signals]
        assert 'engulf_bear' in patterns, f"未检测到engulf_bear，检测到: {patterns}"

    def test_no_engulfing_same_direction(self, detector):
        """同方向两根K线不应触发吞没"""
        from tests.fixtures.candle_fixtures import make_df
        rows = []
        price = 100.0
        # 两根连续阳线（前根小，后根大）但同方向
        rows += [(100, 101.5, 99.5, 101, 50000)] * 12
        rows.append((100.5, 103, 100, 102.8, 80000))  # 大阳，但前根也是阳
        df = make_df(rows)
        signals = detector.analyze_all(df)
        # 方向相同不应触发吞没
        eng_bear = [s for s in signals if s.pattern == 'engulf_bear']
        assert len(eng_bear) == 0, "同向K线不应触发看跌吞没"


class TestDoji:

    def test_detects_doji(self, detector):
        """标准十字星应被识别"""
        df = make_doji(atr=1.0)
        signals = detector.analyze_all(df)
        patterns = [s.pattern for s in signals]
        assert 'doji' in patterns, f"未检测到doji，检测到: {patterns}"

    def test_no_doji_large_body(self, detector):
        """大实体K线（实体/总幅=80%）不应触发十字星"""
        from tests.fixtures.candle_fixtures import make_df
        rows = [(100, 102, 99.5, 101.9, 60000)] * 15  # 实体≈80%
        df = make_df(rows)
        signals = detector.analyze_all(df)
        doji_sigs = [s for s in signals if s.pattern == 'doji']
        assert len(doji_sigs) == 0, f"大实体K不应触发十字星，但检测到{len(doji_sigs)}个"


class TestInsideBar:

    def test_detects_inside_bar(self, detector):
        """标准内包线应被识别"""
        df = make_inside_bar(atr=1.0)
        signals = detector.analyze_all(df)
        patterns = [s.pattern for s in signals]
        assert 'inside_bar' in patterns, f"未检测到inside_bar，检测到: {patterns}"

    def test_no_inside_bar_when_breakout(self, detector):
        """突破前K高点的K线不应触发内包线"""
        from tests.fixtures.candle_fixtures import make_df
        rows = [(100, 102, 98, 101, 50000)] * 12
        rows.append((101, 103, 97, 102, 60000))  # 突破高低，应为外包
        df = make_df(rows)
        signals = detector.analyze_all(df)
        ib_sigs = [s for s in signals[-1:] if s.pattern == 'inside_bar']
        # 最后一根不是内包（它是外包）
        for sig in signals:
            if sig.index == len(df) - 1:
                assert sig.pattern != 'inside_bar', "突破外包K线不应识别为内包线"


class TestOutsideBar:

    def test_detects_outside_bar(self, detector):
        """标准外包线应被识别"""
        df = make_outside_bar(atr=1.0)
        signals = detector.analyze_all(df)
        patterns = [s.pattern for s in signals]
        assert 'outside_bar' in patterns, f"未检测到outside_bar，检测到: {patterns}"


class TestMorningEveningStar:

    def test_detects_morning_star(self, detector):
        """早晨之星应被识别"""
        df = make_morning_star(atr=1.0)
        signals = detector.analyze_all(df)
        patterns = [s.pattern for s in signals]
        assert 'morning_star' in patterns, f"未检测到morning_star，检测到: {patterns}"

    def test_morning_star_is_strong(self, detector):
        """早晨之星应为strong"""
        df = make_morning_star(atr=1.0)
        signals = detector.analyze_all(df)
        ms = [s for s in signals if s.pattern == 'morning_star']
        assert any(s.strength == 'strong' for s in ms), "早晨之星应为strong"

    def test_detects_evening_star(self, detector):
        """黄昏之星应被识别"""
        df = make_evening_star(atr=1.0)
        signals = detector.analyze_all(df)
        patterns = [s.pattern for s in signals]
        assert 'evening_star' in patterns, f"未检测到evening_star，检测到: {patterns}"


class TestAnalyzeLastN:

    def test_returns_correct_count(self, detector):
        """analyze_last_n 应返回最多 n 条记录"""
        df = make_bull_flag_df()
        result = detector.analyze_last_n(df, n=10)
        assert len(result) <= 10

    def test_contains_required_fields(self, detector):
        """每条记录应包含必要字段"""
        from tests.fixtures.candle_fixtures import make_df
        rows = [(100, 101, 99, 100.5, 50000)] * 20
        df = make_df(rows)
        result = detector.analyze_last_n(df, n=5)
        required_fields = {'index', 'timestamp', 'open', 'high', 'low', 'close',
                           'direction', 'size', 'body_ratio', 'patterns', 'pa_desc'}
        for record in result:
            missing = required_fields - set(record.keys())
            assert not missing, f"记录缺少字段: {missing}"


def make_bull_flag_df():
    """辅助函数：生成用于analyze_last_n测试的数据"""
    from tests.fixtures.candle_fixtures import make_bull_flag
    return make_bull_flag()
