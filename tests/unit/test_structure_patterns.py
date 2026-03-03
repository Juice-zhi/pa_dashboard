"""
结构形态单元测试

测试 IntradayPatternDetector 的5类形态识别：
- bull_flag / bear_flag
- three_drives_top / three_drives_bot
- channel_up / channel_down
- double_top / double_bottom
- head_shoulders / inv_head_shoulders

每类包含：
1. 几何正确性测试：合成完美形态 → 应被检测到
2. 阈值边界测试：超出约束的形态 → 应被拒绝
3. XFAIL测试：已知 pa_engine 与权威标准不符的情况（标记为已知偏差）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from core.pa_engine import IntradayPatternDetector
from core.validators import PATTERN_CONSTRAINTS
from tests.fixtures.candle_fixtures import (
    make_bull_flag, make_bull_flag_wide,
    make_bear_flag,
    make_double_top, make_double_top_shallow, make_double_bottom,
    make_head_shoulders, make_head_shoulders_shallow, make_inv_head_shoulders,
    make_three_drives_top,
    make_channel_up, make_channel_down,
)


@pytest.fixture
def detector():
    return IntradayPatternDetector(swing_lb=2)


# ─────────────────────────────────────────────
# 旗形
# ─────────────────────────────────────────────

class TestBullFlag:

    def test_detects_standard_bull_flag(self, detector):
        """标准牛旗（旗杆5根强阳 + 10根窄幅下倾）应被检测到"""
        df = make_bull_flag(atr=1.0, n_flat=10)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns]
        assert 'bull_flag' in kinds, f"未检测到bull_flag，检测到: {kinds}"

    def test_bull_flag_confidence_above_threshold(self, detector):
        """检测到的牛旗置信度应 >= 0.5"""
        df = make_bull_flag(atr=1.0, n_flat=10)
        patterns = detector.detect_all(df)
        flags = [p for p in patterns if p.kind == 'bull_flag']
        assert flags, "未检测到牛旗"
        assert any(p.confidence >= 0.5 for p in flags), \
            f"牛旗置信度过低: {[p.confidence for p in flags]}"

    def test_bull_flag_has_valid_rr(self, detector):
        """牛旗的 target > entry > stop（合理的RR结构）"""
        df = make_bull_flag(atr=1.0, n_flat=10)
        patterns = detector.detect_all(df)
        flags = [p for p in patterns if p.kind == 'bull_flag']
        for p in flags:
            assert p.target > p.entry, f"牛旗target({p.target}) <= entry({p.entry})"
            assert p.entry > p.stop, f"牛旗entry({p.entry}) <= stop({p.stop})"

    def test_no_bull_flag_when_consolidation_too_wide(self, detector):
        """旗面宽度超过旗杆40%时，应不识别为牛旗"""
        df = make_bull_flag_wide(atr=1.0, n_flat=10)
        patterns = detector.detect_all(df)
        flag_kinds = [p.kind for p in patterns if p.kind == 'bull_flag']
        assert len(flag_kinds) == 0, \
            f"宽旗面不应识别为牛旗，但检测到{len(flag_kinds)}个"

    def test_pole_atr_threshold(self):
        """验证 pa_engine 牛旗旗杆最小幅度阈值符合权威约束"""
        constraints = PATTERN_CONSTRAINTS['bull_flag']
        # pa_engine 使用 2.5 ATR，权威要求 2.5 ATR —— 符合
        assert constraints['pole_atr_min'] == 2.5, \
            f"期望旗杆ATR阈值2.5，实际: {constraints['pole_atr_min']}"


class TestBearFlag:

    def test_detects_standard_bear_flag(self, detector):
        """标准熊旗应被检测到"""
        df = make_bear_flag(atr=1.0, n_flat=10)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns]
        assert 'bear_flag' in kinds, f"未检测到bear_flag，检测到: {kinds}"

    def test_bear_flag_has_valid_rr(self, detector):
        """熊旗的 target < entry < stop（合理的RR结构）"""
        df = make_bear_flag(atr=1.0, n_flat=10)
        patterns = detector.detect_all(df)
        flags = [p for p in patterns if p.kind == 'bear_flag']
        for p in flags:
            assert p.target < p.entry, f"熊旗target({p.target}) >= entry({p.entry})"
            assert p.entry < p.stop, f"熊旗entry({p.entry}) >= stop({p.stop})"


# ─────────────────────────────────────────────
# 双顶/双底
# ─────────────────────────────────────────────

class TestDoubleTop:

    def test_detects_standard_double_top(self, detector):
        """标准双顶（两顶相差0.5%，回调5%）应被检测到"""
        df = make_double_top(price_diff_pct=0.005, trough_depth_pct=0.05)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns]
        assert 'double_top' in kinds, f"未检测到double_top，检测到: {kinds}"

    def test_double_top_neckline_exists(self, detector):
        """双顶应包含颈线信息（3个关键价格点）"""
        df = make_double_top(price_diff_pct=0.005, trough_depth_pct=0.05)
        patterns = detector.detect_all(df)
        tops = [p for p in patterns if p.kind == 'double_top']
        for p in tops:
            assert len(p.key_prices) >= 3, \
                f"双顶应有至少3个关键价格点（两顶+颈线），实际: {len(p.key_prices)}"

    def test_no_double_top_when_tops_too_different(self, detector):
        """两顶价格差过大（5%）时，不应识别为双顶"""
        df = make_double_top(price_diff_pct=0.05, trough_depth_pct=0.05)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns if p.kind == 'double_top']
        assert len(kinds) == 0, \
            f"两顶差5%不应识别为双顶，但检测到{len(kinds)}个"

    def test_rejects_shallow_trough_double_top(self, detector):
        """
        修复后验证：回调深度1.5%（<权威3%标准）的双顶应被拒绝。
        pa_engine已修复：trough_depth < 0.03（原0.8%已改为3%）
        """
        df = make_double_top_shallow(trough_depth_pct=0.015)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns if p.kind == 'double_top']
        assert len(kinds) == 0, \
            f"回调1.5%（<3%标准）不应识别为双顶，但检测到{len(kinds)}个"

    def test_accepts_two_percent_price_diff(self, detector):
        """
        修复后验证：两顶相差2%（在权威3%范围内），应被识别为双顶。
        pa_engine已修复：price_diff > 0.03（原0.8%已改为3%）
        """
        df = make_double_top(price_diff_pct=0.02, trough_depth_pct=0.05)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns if p.kind == 'double_top']
        assert len(kinds) >= 1, \
            f"两顶相差2%（<权威3%标准）应识别为双顶，但未检测到"


class TestDoubleBottom:

    def test_detects_standard_double_bottom(self, detector):
        """标准双底（两底相差0.5%，反弹5%）应被检测到"""
        df = make_double_bottom(price_diff_pct=0.005, peak_height_pct=0.05)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns]
        assert 'double_bottom' in kinds, f"未检测到double_bottom，检测到: {kinds}"

    def test_double_bottom_has_valid_rr(self, detector):
        """双底的 target > entry > stop"""
        df = make_double_bottom(price_diff_pct=0.005, peak_height_pct=0.05)
        patterns = detector.detect_all(df)
        bots = [p for p in patterns if p.kind == 'double_bottom']
        for p in bots:
            assert p.target > p.entry, f"双底target({p.target}) <= entry({p.entry})"
            assert p.entry > p.stop, f"双底entry({p.entry}) <= stop({p.stop})"


# ─────────────────────────────────────────────
# 头肩形态
# ─────────────────────────────────────────────

class TestHeadShoulders:

    def test_detects_standard_head_shoulders(self, detector):
        """标准头肩顶（头比颈线高4%，两肩差1%）应被检测到"""
        df = make_head_shoulders(head_depth_pct=0.04, shoulder_diff_pct=0.01)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns]
        assert 'head_shoulders' in kinds, f"未检测到head_shoulders，检测到: {kinds}"

    def test_head_shoulders_has_5_key_prices(self, detector):
        """头肩顶应有5个关键价格点（左肩-颈1-头-颈2-右肩）"""
        df = make_head_shoulders(head_depth_pct=0.04, shoulder_diff_pct=0.01)
        patterns = detector.detect_all(df)
        hs = [p for p in patterns if p.kind == 'head_shoulders']
        for p in hs:
            assert len(p.key_prices) == 5, \
                f"头肩顶应有5个关键点，实际: {len(p.key_prices)}"

    def test_no_head_shoulders_when_shoulders_too_different(self, detector):
        """两肩差异超过5%时，不应识别为头肩顶"""
        df = make_head_shoulders(head_depth_pct=0.04, shoulder_diff_pct=0.07)
        patterns = detector.detect_all(df)
        hs = [p for p in patterns if p.kind == 'head_shoulders']
        assert len(hs) == 0, \
            f"两肩差7%（>5%限制）不应识别为头肩，但检测到{len(hs)}个"

    def test_rejects_shallow_head(self, detector):
        """
        修复后验证：头部深度1.5%（<权威2%标准）的头肩顶应被拒绝。
        pa_engine已修复：head_depth < 0.02（原1%已改为2%）
        """
        df = make_head_shoulders_shallow(head_depth_pct=0.015)
        patterns = detector.detect_all(df)
        hs = [p for p in patterns if p.kind == 'head_shoulders']
        assert len(hs) == 0, \
            f"头部深度1.5%（<权威2%标准）不应识别为头肩，但检测到{len(hs)}个"


class TestInvHeadShoulders:

    def test_detects_standard_inv_head_shoulders(self, detector):
        """标准反头肩底应被检测到"""
        df = make_inv_head_shoulders(head_depth_pct=0.04, shoulder_diff_pct=0.01)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns]
        assert 'inv_head_shoulders' in kinds, \
            f"未检测到inv_head_shoulders，检测到: {kinds}"

    def test_inv_head_shoulders_has_valid_rr(self, detector):
        """反头肩底的 target > entry > stop"""
        df = make_inv_head_shoulders(head_depth_pct=0.04, shoulder_diff_pct=0.01)
        patterns = detector.detect_all(df)
        ihs = [p for p in patterns if p.kind == 'inv_head_shoulders']
        for p in ihs:
            assert p.target > p.entry, \
                f"反头肩底target({p.target}) <= entry({p.entry})"
            assert p.entry > p.stop, \
                f"反头肩底entry({p.entry}) <= stop({p.stop})"


# ─────────────────────────────────────────────
# 三推形态
# ─────────────────────────────────────────────

class TestThreeDrives:

    def test_detects_three_drives_top(self, detector):
        """标准三推顶（第三推动能衰竭25%）应被检测到"""
        df = make_three_drives_top(decay_pct=0.25)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns]
        assert 'three_drives_top' in kinds, \
            f"未检测到three_drives_top，检测到: {kinds}"

    def test_three_drives_confidence(self, detector):
        """三推顶置信度应 >= 0.5"""
        df = make_three_drives_top(decay_pct=0.25)
        patterns = detector.detect_all(df)
        tdt = [p for p in patterns if p.kind == 'three_drives_top']
        assert tdt, "未检测到三推顶"
        assert any(p.confidence >= 0.5 for p in tdt), \
            f"三推顶置信度过低: {[p.confidence for p in tdt]}"

    def test_three_drives_win_rate_is_backtest_pending(self):
        """
        三推形态的胜率来源说明应包含"回测"或"无权威来源"的标注
        验证 validators.py 中的标注是否正确
        """
        from core.validators import WIN_RATE_SOURCES
        for kind in ['three_drives_top', 'three_drives_bot']:
            entry = WIN_RATE_SOURCES[kind]
            assert entry['rate_daily'] is None, \
                f"{kind} 应标记为无权威来源(None)，但当前: {entry['rate_daily']}"
            assert '无权威' in entry['source'] or '回测' in entry['source'], \
                f"{kind} 来源说明应包含'无权威'或'回测'"


# ─────────────────────────────────────────────
# 通道形态
# ─────────────────────────────────────────────

class TestChannel:

    def test_detects_channel_up(self, detector):
        """上升通道应被检测到"""
        df = make_channel_up(n_bars=80, atr=0.5, slope=0.15)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns]
        assert 'channel_up' in kinds, f"未检测到channel_up，检测到: {kinds}"

    def test_detects_channel_down(self, detector):
        """下降通道应被检测到"""
        df = make_channel_down(n_bars=80, atr=0.5, slope=-0.15)
        patterns = detector.detect_all(df)
        kinds = [p.kind for p in patterns]
        assert 'channel_down' in kinds, f"未检测到channel_down，检测到: {kinds}"

    def test_channel_has_4_key_prices(self, detector):
        """通道应有4个关键点（上轨两端点 + 下轨两端点）"""
        df = make_channel_up(n_bars=80, atr=0.5, slope=0.15)
        patterns = detector.detect_all(df)
        channels = [p for p in patterns if 'channel' in p.kind]
        for p in channels:
            assert len(p.key_prices) >= 4, \
                f"通道应有4个关键点，实际: {len(p.key_prices)}"


# ─────────────────────────────────────────────
# 通用属性验证
# ─────────────────────────────────────────────

class TestPatternAttributes:

    def test_pattern_win_rate_from_table(self, detector):
        """所有形态的win_rate_est应从WIN_RATE_TABLE获取，不为0"""
        df = make_bull_flag(atr=1.0, n_flat=10)
        patterns = detector.detect_all(df)
        for p in patterns:
            assert p.win_rate_est > 0, \
                f"形态 {p.kind} 的胜率不应为0"
            assert p.win_rate_est <= 1.0, \
                f"形态 {p.kind} 的胜率超过1.0: {p.win_rate_est}"

    def test_confidence_range(self, detector):
        """所有形态置信度应在 [0, 1] 区间"""
        df = make_bull_flag(atr=1.0, n_flat=10)
        patterns = detector.detect_all(df)
        for p in patterns:
            assert 0 <= p.confidence <= 1.0, \
                f"形态 {p.kind} 置信度越界: {p.confidence}"

    def test_detect_all_returns_max_8(self, detector):
        """detect_all 最多返回8个形态"""
        df = make_bull_flag(atr=1.0, n_flat=10)
        patterns = detector.detect_all(df)
        assert len(patterns) <= 8, f"返回形态数超过8: {len(patterns)}"


# ─────────────────────────────────────────────
# 权威约束对比测试
# ─────────────────────────────────────────────

class TestConstraintDocumentation:
    """验证 validators.py 中的约束文档是否完整且合理"""

    def test_all_pattern_kinds_have_constraints(self):
        """主要形态类型应在 PATTERN_CONSTRAINTS 中有约束定义"""
        required_kinds = ['double_top', 'double_bottom', 'head_shoulders',
                          'inv_head_shoulders', 'bull_flag', 'bear_flag',
                          'three_drives', 'channel']
        for kind in required_kinds:
            assert kind in PATTERN_CONSTRAINTS, \
                f"缺少形态约束: {kind}"

    def test_all_win_rate_sources_have_citation(self):
        """所有形态应有胜率来源说明"""
        from core.validators import WIN_RATE_SOURCES
        for kind, entry in WIN_RATE_SOURCES.items():
            assert 'source' in entry and len(entry['source']) > 10, \
                f"形态 {kind} 缺少胜率来源说明"

    def test_double_top_trough_depth_standard(self):
        """双顶的权威回调深度标准应为3%（不是0.8%）"""
        constraints = PATTERN_CONSTRAINTS['double_top']
        assert constraints['trough_depth_min'] == 0.03, \
            f"双顶权威回调深度标准应为3%，实际: {constraints['trough_depth_min']}"

    def test_head_shoulders_depth_standard(self):
        """头肩顶的权威头部深度标准应为2%（不是1%）"""
        constraints = PATTERN_CONSTRAINTS['head_shoulders']
        assert constraints['head_depth_min'] == 0.02, \
            f"头肩顶权威深度标准应为2%，实际: {constraints['head_depth_min']}"

    def test_pa_engine_issues_are_documented(self):
        """validators.py 应记录 pa_engine 的已知问题"""
        from core.validators import list_pa_engine_issues
        issues = list_pa_engine_issues()
        # 至少应有3个已知问题（double_top × 2 + head_shoulders × 1）
        assert len(issues) >= 3, \
            f"应记录至少3个pa_engine已知问题，实际: {len(issues)}"
        # 应包含double_top相关问题
        dt_issues = [i for i in issues if i['pattern'] == 'double_top']
        assert len(dt_issues) >= 1, "应记录double_top的已知问题"
