
# -*- coding: utf-8 -*-
"""
backtest_contract_v5.py

V5 backtest / tune contract for the full single-tier stack.

핵심 방향
---------
- entry: multi-horizon `dir + hyb + utility + retcls` composite
- thesis / exit: `path10 / utility_10 / first_hit_10 / tth_10`
- runtime richness: dynamic / regime / tp_window / entry_episode / rearm / same-side-hold
- execution skeleton: v4 live-aligned maker-first / IOC fallback
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

BACKTEST_CONTRACT_VERSION = "backtest_v5_dirhyb_path10_single_tier_full_r1"


def _table_ext() -> str:
    try:
        import pyarrow  # type: ignore
        _ = pyarrow
        return ".parquet"
    except Exception:
        try:
            import fastparquet  # type: ignore
            _ = fastparquet
            return ".parquet"
        except Exception:
            return ".csv.gz"


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class ExecStageV5(str, Enum):
    IDLE = "IDLE"
    ENTRY_SIGNAL_ACCEPTED = "ENTRY_SIGNAL_ACCEPTED"
    ENTRY_PRECHECK = "ENTRY_PRECHECK"
    ENTRY_POSTONLY_SUBMIT = "ENTRY_POSTONLY_SUBMIT"
    ENTRY_POSTONLY_REST = "ENTRY_POSTONLY_REST"
    ENTRY_IOC_SUBMIT = "ENTRY_IOC_SUBMIT"
    ENTRY_SKIPPED = "ENTRY_SKIPPED"
    ENTRY_FILLED = "ENTRY_FILLED"
    PRE_BEP = "PRE_BEP"
    BEP_ARMED = "BEP_ARMED"
    TRAIL_ACTIVE = "TRAIL_ACTIVE"
    THESIS_REVIEW = "THESIS_REVIEW"
    EXIT_TRIGGERED = "EXIT_TRIGGERED"
    EXIT_POSTONLY_SUBMIT = "EXIT_POSTONLY_SUBMIT"
    EXIT_POSTONLY_REST = "EXIT_POSTONLY_REST"
    EXIT_IOC_SUBMIT = "EXIT_IOC_SUBMIT"
    EXIT_MARKET_SUBMIT = "EXIT_MARKET_SUBMIT"
    EXIT_FILLED = "EXIT_FILLED"


class FallbackPathV5(str, Enum):
    NONE = "NONE"
    POST_ONLY = "POST_ONLY"
    IOC_LIMIT = "IOC_LIMIT"
    MARKET = "MARKET"
    TAKER_NEXT_OPEN = "TAKER_NEXT_OPEN"


class ThesisStateV5(str, Enum):
    NONE = "NONE"
    STRONG_SAME = "STRONG_SAME"
    WEAK_SAME = "WEAK_SAME"
    NEUTRAL_DRIFT = "NEUTRAL_DRIFT"
    WEAK_OPPOSITE = "WEAK_OPPOSITE"
    STRONG_OPPOSITE = "STRONG_OPPOSITE"
    LIQUIDITY_SHOCK = "LIQUIDITY_SHOCK"


class ExitReasonV5(str, Enum):
    TP = "TP"
    SL = "SL"
    TRAIL = "TRAIL"
    SOFT_SL = "SOFT_SL"
    MAX_HOLD = "MAX_HOLD"
    PRE_BEP_TIMEOUT = "PRE_BEP_TIMEOUT"
    THESIS_FAIL_SOFT = "THESIS_FAIL_SOFT"
    THESIS_FAIL_HARD = "THESIS_FAIL_HARD"
    GAP_FLATTEN = "GAP_FLATTEN"
    SHOCK_EXIT = "SHOCK_EXIT"
    FORCE_CLOSE = "FORCE_CLOSE"
    ENTRY_UNFILLED = "ENTRY_UNFILLED"
    SKIP = "SKIP"


class EntryFillModeV5(str, Enum):
    MAKER_REST_THEN_IOC = "maker_rest_then_ioc"
    MAKER_ONLY = "maker_only"
    TAKER_NEXT_OPEN = "taker_next_open"


class ExitFillModeV5(str, Enum):
    MAKER_TOUCH = "maker_touch"
    MAKER_TOUCH_THEN_IOC = "maker_touch_then_ioc"
    IOC_LIMIT = "ioc_limit"
    TAKER_TOUCH = "taker_touch"
    TAKER_CLOSE = "taker_close"
    GAP_OPEN = "gap_open"
    MARKET = "market"


class TouchRuleV5(str, Enum):
    TOUCH = "touch"
    PENETRATE_OFFSET = "penetrate_offset"
    PENETRATE_FRAC_RANGE = "penetrate_frac_range"


class IntrabarModeV5(str, Enum):
    ADVERSE_FIRST = "adverse_first"
    FAVORABLE_FIRST = "favorable_first"
    GAP_AWARE_ADVERSE_FIRST = "gap_aware_adverse_first"
    MAKER_CONSERVATIVE = "maker_conservative"


class BepStopModeV5(str, Enum):
    MAKER_BE = "maker_be"
    TAKER_BE = "taker_be"


class DynamicModeV5(str, Enum):
    ENTRY_LATCHED = "entry_latched"
    BAR_LIVE = "bar_live"


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

HORIZON_KEYS: Tuple[str, ...] = ("w1", "w3", "w5", "w8", "w10")


def _default_gate_weights() -> Dict[str, float]:
    return {"w1": 0.28, "w3": 0.28, "w5": 0.20, "w8": 0.14, "w10": 0.10}


def _default_dir_weights() -> Dict[str, float]:
    return {"w1": 0.30, "w3": 0.25, "w5": 0.20, "w8": 0.15, "w10": 0.10}


def _default_gate_calm_anchor() -> Dict[str, float]:
    return {"w1": 0.45, "w3": 0.40, "w5": 0.12, "w8": 0.03, "w10": 0.00}


def _default_gate_active_anchor() -> Dict[str, float]:
    return {"w1": 0.08, "w3": 0.28, "w5": 0.32, "w8": 0.22, "w10": 0.10}


def _default_dir_calm_anchor() -> Dict[str, float]:
    return {"w1": 0.30, "w3": 0.24, "w5": 0.24, "w8": 0.16, "w10": 0.06}


def _default_dir_active_anchor() -> Dict[str, float]:
    return {"w1": 0.08, "w3": 0.12, "w5": 0.32, "w8": 0.32, "w10": 0.16}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyConfigV5:
    gate_weights: Dict[str, float] = field(default_factory=_default_gate_weights)
    dir_weights: Dict[str, float] = field(default_factory=_default_dir_weights)

    hyb_weight: float = 0.55
    util_weight: float = 1.00
    cls_weight: float = 0.15
    dirprob_weight: float = 0.10

    entry_q: float = 0.85
    entry_th_floor: float = 0.00
    entry_min_score: float = 0.00
    entry_min_gap: float = 0.00
    entry_min_hyb_abs: float = 0.00
    entry_min_cls_abs: float = 0.00
    min_side_agreement_frac: float = 0.00

    entry_min_utility_10: float = 0.05
    entry_min_utility_gap_10: float = 0.00

    confirm_main_barrier: float = 1.0
    confirm_main_prob: float = 0.50
    timing_barrier: float = 1.0
    timing_first_hit_prob: float = 0.34
    timing_max_expected_bars: float = 8.0
    timing_max_censored_prob: float = 0.60
    require_retcls_alignment: bool = False
    min_retcls_align_score: float = 0.00

    dir_score_gate_floor: float = 0.00
    gate_score_floor: float = 0.00

    profit_floor_enabled: bool = False
    profit_floor_frac_of_mfe: float = 0.35

    thesis_grace_bars_after_entry: int = 2
    thesis_state_confirm_bars: int = 2
    thesis_strong_flip_margin: float = 0.35
    thesis_weak_flip_margin: float = 0.15
    thesis_progress_protect_frac: float = 0.40
    thesis_pre_bep_ioc_enable: bool = True
    thesis_post_bep_trail_tighten: bool = True
    thesis_market_exit_margin: float = 0.60
    thesis_ioc_exit_margin: float = 0.30
    thesis_neutral_tighten_after_bars: int = 4
    thesis_oppose_near_maxhold_aggr: bool = True

    TP: float = 1.20
    SL: float = 1.10
    BEP_ARM: float = 0.12
    trailing: float = 0.18
    fee_tp_mult: float = 0.70
    bep_arm_fee_mult: float = 1.00
    bep_stop_fee_mult: float = 2.20
    bep_stop_mode: str = BepStopModeV5.TAKER_BE.value

    min_hold_tp_bars: int = 2
    min_hold_trail_bars: int = 1
    min_hold_soft_sl_bars: int = 2
    max_hold_bars: int = 10
    hard_max_hold_bars: int = 10
    hard_sl_mult_pre_unlock: float = 1.00
    trail_grace_after_bep: int = 0
    trail_grace_after_unlock: int = 0

    cooldown_bars: int = 0
    gap_skip_n: float = 1.50
    gap_force_exit_enabled: bool = True
    shock_bar_range_n: float = 2.50
    maker_taker_penalty_cap: float = 0.25
    max_unfilled_entry_rate: float = 0.40

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicConfigV5:
    enabled: bool = False
    mode: str = DynamicModeV5.ENTRY_LATCHED.value

    use_dyn_lev: bool = True
    use_dyn_gate: bool = True
    use_dyn_bep: bool = True
    use_dyn_trail: bool = True
    use_dyn_sl: bool = True
    use_dyn_soft_sl: bool = True

    allow_soft_sl_before_trail: bool = False
    softsl_hold_floor: int = 0
    post_bep_shield_ignore_softsl_hold: bool = False

    lev_scale_min: float = 0.90
    lev_scale_max: float = 1.10
    gate_mult_min: float = 0.90
    gate_mult_max: float = 1.20
    bep_scale_min: float = 0.90
    bep_scale_max: float = 1.15
    trail_scale_min: float = 0.90
    trail_scale_max: float = 1.20
    sl_scale_min: float = 0.90
    sl_scale_max: float = 1.20
    softsl_relax_mid: int = 1
    softsl_relax_hi: int = 2

    stress_lo: float = 0.25
    stress_hi: float = 0.65
    alpha_ema: float = 0.15
    alpha_hysteresis: float = 0.03

    w_atr: float = 0.30
    w_rng: float = 0.20
    w_vol: float = 0.20
    w_stretch: float = 0.15
    w_band: float = 0.15

    use_pre_bep_timeout: bool = True
    pre_bep_timeout_bars: int = 5
    pre_bep_stress_th: float = 0.55
    pre_bep_progress_frac: float = 0.12
    pre_bep_degrade_sl_scale: float = 0.90
    pre_bep_softsl_delta: int = 1
    pre_bep_force_close_bars: int = 0
    pre_bep_force_close_red_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgressProtectConfigV5:
    early_softsl_enabled: bool = True
    early_softsl_min_hold: int = 2
    early_softsl_progress_frac: float = 0.55
    early_trail_enabled: bool = True
    early_trail_min_hold: int = 2
    early_trail_progress_frac: float = 0.70
    early_trail_ref_updates_min: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TPWindowConfigV5:
    enabled: bool = False
    progress_frac_arm: float = 0.75
    extend_bars: int = 2
    block_early_trail: bool = True
    block_early_soft_sl: bool = True
    floor_trail_hold_to_tp: bool = True
    floor_soft_sl_hold_to_tp: bool = True
    suspend_post_bep_shield_before_tp: bool = False
    expire_on_pullback_frac: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntryEpisodeConfigV5:
    entry_episode_enabled: bool = False
    rearm_enabled: bool = False
    run_gap_reset_bars: int = 3
    episode_max_entries_per_run: int = 1
    rearm_same_side_only: bool = True
    rearm_cooldown_bars: int = 1
    rearm_max_bars_after_exit: int = 6
    rearm_gate_reset_frac: float = 0.85
    rearm_gate_refresh_frac: float = 0.90
    rearm_price_reset_frac: float = 0.60
    rearm_after_trail: bool = True
    rearm_after_tp: bool = True
    rearm_after_sl: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SameSideHoldConfigV5:
    enabled: bool = False
    weak_enabled: bool = False
    strong_ratio: float = 0.80
    weak_ratio: float = 0.60
    weak_min_progress_frac: float = 0.40
    allow_pre_bep_weak: bool = False
    pre_bep_max_bonus_bars: int = 1
    bonus_bars_strong: int = 2
    bonus_bars_weak: int = 1
    max_extra_bars: int = 4
    grace_after_bep_strong: int = 1
    grace_after_bep_weak: int = 0
    grace_after_unlock_strong: int = 1
    grace_after_unlock_weak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeDetectConfigV5:
    enabled: bool = False
    stress_lo: float = 0.25
    stress_hi: float = 0.65
    alpha_ema: float = 0.15
    alpha_hysteresis: float = 0.03
    w_atr: float = 0.30
    w_rng: float = 0.20
    w_vol: float = 0.20
    w_stretch: float = 0.15
    w_band: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeWeightConfigV5:
    enabled: bool = False
    gate_calm_mix: float = 0.60
    gate_active_mix: float = 0.55
    dir_calm_mix: float = 0.35
    dir_active_mix: float = 0.50
    gate_calm_anchor: Dict[str, float] = field(default_factory=_default_gate_calm_anchor)
    gate_active_anchor: Dict[str, float] = field(default_factory=_default_gate_active_anchor)
    dir_calm_anchor: Dict[str, float] = field(default_factory=_default_dir_calm_anchor)
    dir_active_anchor: Dict[str, float] = field(default_factory=_default_dir_active_anchor)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeThresholdConfigV5:
    enabled: bool = False
    q_entry_calm: float = 0.78
    q_entry_mid: float = 0.84
    q_entry_active: float = 0.90
    entry_th_calm: float = 0.00
    entry_th_mid: float = 0.00
    entry_th_active: float = 0.00
    bucket_min_ready: int = 64
    bucket_fallback_global: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeFilterConfigV5:
    enabled: bool = False
    use_vol_split: bool = True
    use_entry_mult_split: bool = True
    mid_interp_mode: str = "linear"
    vol_low_th_calm: float = -999.0
    vol_low_th_mid: float = -999.0
    vol_low_th_active: float = -999.0
    atr_entry_mult_calm: float = 0.00
    atr_entry_mult_active: float = 0.00
    range_entry_mult_calm: float = 0.00
    range_entry_mult_active: float = 0.00

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeLaneConfigV5:
    enabled: bool = False
    active_sparse_enabled: bool = False
    active_sparse_min_ready: int = 96
    sparse_gate_q: float = 0.94
    sparse_gate_floor_q: float = 0.90
    sparse_atr_q: float = 0.85
    sparse_range_q: float = 0.85
    sparse_vol_q: float = 0.70
    sparse_require_high_vol: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionConfigV5:
    leverage: float = 11.0
    cost_per_side: float = 0.00070
    slip_per_side: float = 0.00015
    maker_fee_per_side: float = 0.00020

    entry_fill_mode: str = EntryFillModeV5.MAKER_REST_THEN_IOC.value
    entry_postonly_offset_bps: float = 0.50
    entry_fill_max_bars: int = 2
    entry_touch_rule: str = TouchRuleV5.PENETRATE_OFFSET.value
    entry_touch_offset_bps: float = 0.25
    entry_touch_frac_range: float = 0.15
    entry_ioc_enabled: bool = True
    entry_ioc_offset_bps: float = 1.50
    entry_ioc_impact_cap_bps: float = 4.00

    emergency_market_enabled: bool = False
    emergency_market_offset_bps: float = 3.00

    tp_fill_mode: str = ExitFillModeV5.MAKER_TOUCH.value
    tp_postonly_offset_bps: float = 0.10
    trail_fill_mode: str = ExitFillModeV5.MAKER_TOUCH_THEN_IOC.value
    trail_postonly_offset_bps: float = 0.10
    softsl_fill_mode: str = ExitFillModeV5.IOC_LIMIT.value
    sl_fill_mode: str = ExitFillModeV5.IOC_LIMIT.value
    sl_ioc_offset_bps: float = 1.50

    exit_touch_rule: str = TouchRuleV5.PENETRATE_OFFSET.value
    exit_touch_offset_bps: float = 0.10
    exit_touch_frac_range: float = 0.10
    market_exit_impact_cap_bps: float = 8.00

    integer_leverage: bool = False
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestConfigV5:
    entry_on_next_open: bool = True
    intrabar_mode: str = IntrabarModeV5.GAP_AWARE_ADVERSE_FIRST.value
    segments: int = 10
    annualization_days: float = 365.0
    prefer_fast: bool = False

    threshold_lookback_bars: int = 480
    threshold_refresh_bars: int = 1
    threshold_min_ready: int = 64

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveConfigV5:
    mean_seg_weight: float = 1.00
    worst2_seg_weight: float = 1.25
    mdd_penalty: float = 1.00
    tail_penalty: float = 0.35
    trade_band_penalty: float = 0.20
    min_trades_penalty: float = 0.35
    maker_ratio_penalty: float = 0.35
    overhold_penalty: float = 0.25
    unfilled_entry_penalty: float = 0.25
    shock_exit_penalty: float = 0.20
    trade_dispersion_penalty: float = 0.00
    bottom_k_penalty: float = 0.00
    regime_extreme_max_frac: float = 1.00
    regime_extreme_penalty_k: float = 0.00
    side_balance_penalty_k: float = 0.00
    min_short_trades: int = 0
    min_short_share: float = 0.00
    target_trade_min: int = 80
    target_trade_max: int = 4000
    target_maker_entry_ratio: float = 0.60
    target_maker_exit_ratio: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunArtifactConfigV5:
    out_dir: str = "./bt_runs"
    log_out: str = "v5_run"
    save_pred: bool = False
    save_fast_cache: bool = False
    save_debug_sqlite: bool = False

    def decision_path(self) -> str:
        return str(Path(self.out_dir) / f"decision_{self.log_out}{_table_ext()}")

    def trade_path(self) -> str:
        return str(Path(self.out_dir) / f"tradelog_{self.log_out}{_table_ext()}")

    def summary_path(self) -> str:
        return str(Path(self.out_dir) / f"summary_{self.log_out}.json")

    def pred_path(self) -> str:
        return str(Path(self.out_dir) / f"pred_cache_{self.log_out}{_table_ext()}")

    def fast_path(self) -> str:
        return str(Path(self.out_dir) / f"pred_fast_{self.log_out}.npz")

    def best_policy_path(self) -> str:
        return str(Path(self.out_dir) / f"best_policy_{self.log_out}.json")

    def trials_path(self) -> str:
        return str(Path(self.out_dir) / f"trials_{self.log_out}{_table_ext()}")

    def tune_summary_path(self) -> str:
        return str(Path(self.out_dir) / f"summary_best_{self.log_out}.json")


DEFAULT_POLICY_V5 = PolicyConfigV5()
DEFAULT_DYNAMIC_V5 = DynamicConfigV5()
DEFAULT_PROGRESS_PROTECT_V5 = ProgressProtectConfigV5()
DEFAULT_TPWINDOW_V5 = TPWindowConfigV5()
DEFAULT_ENTRY_EPISODE_V5 = EntryEpisodeConfigV5()
DEFAULT_SAME_SIDE_HOLD_V5 = SameSideHoldConfigV5()
DEFAULT_REGIME_DETECT_V5 = RegimeDetectConfigV5()
DEFAULT_REGIME_WEIGHT_V5 = RegimeWeightConfigV5()
DEFAULT_REGIME_THRESHOLD_V5 = RegimeThresholdConfigV5()
DEFAULT_REGIME_FILTER_V5 = RegimeFilterConfigV5()
DEFAULT_REGIME_LANE_V5 = RegimeLaneConfigV5()
DEFAULT_EXECUTION_V5 = ExecutionConfigV5()
DEFAULT_BACKTEST_V5 = BacktestConfigV5()
DEFAULT_OBJECTIVE_V5 = ObjectiveConfigV5()


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v) != 0)
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(v)


def normalize_horizon_weights(raw: Mapping[str, Any] | None, *, fallback: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    fb = dict(fallback or _default_gate_weights())
    vals: Dict[str, float] = {}
    for k in HORIZON_KEYS:
        vals[k] = max(0.0, float(raw.get(k, fb.get(k, 0.0)) if raw is not None else fb.get(k, 0.0)))
    s = float(sum(vals.values()))
    if s <= 0.0:
        vals = {str(k): float(v) for k, v in fb.items()}
        s = float(sum(vals.values()))
        if s <= 0.0:
            vals = {"w1": 0.0, "w3": 0.0, "w5": 1.0, "w8": 0.0, "w10": 0.0}
            s = 1.0
    return {k: float(v / max(s, 1.0e-12)) for k, v in vals.items()}


def _dataclass_from_mapping(cls, payload: Optional[Mapping[str, Any]] = None):
    payload = dict(payload or {})
    default_obj = cls()
    kwargs: Dict[str, Any] = {}
    for field_name, field_def in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if field_name not in payload:
            continue
        val = payload[field_name]
        default_val = getattr(default_obj, field_name)
        if isinstance(default_val, bool):
            val = _coerce_bool(val)
        elif isinstance(default_val, int) and not isinstance(default_val, bool):
            val = int(val)
        elif isinstance(default_val, float):
            val = float(val)
        elif isinstance(default_val, str):
            val = str(val)
        elif isinstance(default_val, dict):
            if "weights" in field_name or field_name.endswith("_anchor"):
                fb = default_val
                val = normalize_horizon_weights(val if isinstance(val, Mapping) else None, fallback=fb)
            elif isinstance(val, Mapping):
                val = dict(val)
            else:
                val = dict(default_val)
        kwargs[field_name] = val
    obj = cls(**kwargs)
    # post-normalize weight dicts
    if hasattr(obj, "gate_weights"):
        object.__setattr__(obj, "gate_weights", normalize_horizon_weights(getattr(obj, "gate_weights"), fallback=_default_gate_weights()))  # type: ignore[misc]
    if hasattr(obj, "dir_weights"):
        object.__setattr__(obj, "dir_weights", normalize_horizon_weights(getattr(obj, "dir_weights"), fallback=_default_dir_weights()))  # type: ignore[misc]
    if hasattr(obj, "gate_calm_anchor"):
        object.__setattr__(obj, "gate_calm_anchor", normalize_horizon_weights(getattr(obj, "gate_calm_anchor"), fallback=_default_gate_calm_anchor()))  # type: ignore[misc]
        object.__setattr__(obj, "gate_active_anchor", normalize_horizon_weights(getattr(obj, "gate_active_anchor"), fallback=_default_gate_active_anchor()))  # type: ignore[misc]
        object.__setattr__(obj, "dir_calm_anchor", normalize_horizon_weights(getattr(obj, "dir_calm_anchor"), fallback=_default_dir_calm_anchor()))  # type: ignore[misc]
        object.__setattr__(obj, "dir_active_anchor", normalize_horizon_weights(getattr(obj, "dir_active_anchor"), fallback=_default_dir_active_anchor()))  # type: ignore[misc]
    return obj


def _read_json_file(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return json.loads(text)


def load_config_dataclass(cls, json_path: str = "", inline_json: str = ""):
    payload: Dict[str, Any] = {}
    if str(json_path).strip():
        raw = _read_json_file(json_path)
        if isinstance(raw, Mapping):
            payload.update(dict(raw))
    if str(inline_json).strip():
        raw = json.loads(str(inline_json).lstrip("\ufeff"))
        if isinstance(raw, Mapping):
            payload.update(dict(raw))

    nested_key = {
        "PolicyConfigV5": "policy",
        "DynamicConfigV5": "dynamic",
        "ProgressProtectConfigV5": "progress_protect",
        "TPWindowConfigV5": "tp_window",
        "EntryEpisodeConfigV5": "entry_episode",
        "SameSideHoldConfigV5": "same_side_hold",
        "RegimeDetectConfigV5": "regime_detect",
        "RegimeWeightConfigV5": "regime_weight",
        "RegimeThresholdConfigV5": "regime_threshold",
        "RegimeFilterConfigV5": "regime_filter",
        "RegimeLaneConfigV5": "regime_lane",
        "ExecutionConfigV5": "execution",
        "BacktestConfigV5": "backtest",
        "ObjectiveConfigV5": "objective",
    }.get(getattr(cls, "__name__", ""), "")
    if nested_key and isinstance(payload.get(nested_key), Mapping):
        payload = dict(payload[nested_key])
    return _dataclass_from_mapping(cls, payload)


def derive_artifact_paths(out_dir: str | Path, log_out: str) -> Dict[str, str]:
    cfg = RunArtifactConfigV5(out_dir=str(out_dir), log_out=str(log_out))
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    return {
        "decision": cfg.decision_path(),
        "tradelog": cfg.trade_path(),
        "summary": cfg.summary_path(),
        "pred_cache": cfg.pred_path(),
        "pred_fast": cfg.fast_path(),
        "best_policy": cfg.best_policy_path(),
        "trials": cfg.trials_path(),
        "best_summary": cfg.tune_summary_path(),
    }


__all__ = [
    "BACKTEST_CONTRACT_VERSION",
    "ExecStageV5",
    "FallbackPathV5",
    "ThesisStateV5",
    "ExitReasonV5",
    "EntryFillModeV5",
    "ExitFillModeV5",
    "TouchRuleV5",
    "IntrabarModeV5",
    "BepStopModeV5",
    "DynamicModeV5",
    "PolicyConfigV5",
    "DynamicConfigV5",
    "ProgressProtectConfigV5",
    "TPWindowConfigV5",
    "EntryEpisodeConfigV5",
    "SameSideHoldConfigV5",
    "RegimeDetectConfigV5",
    "RegimeWeightConfigV5",
    "RegimeThresholdConfigV5",
    "RegimeFilterConfigV5",
    "RegimeLaneConfigV5",
    "ExecutionConfigV5",
    "BacktestConfigV5",
    "ObjectiveConfigV5",
    "RunArtifactConfigV5",
    "DEFAULT_POLICY_V5",
    "DEFAULT_DYNAMIC_V5",
    "DEFAULT_PROGRESS_PROTECT_V5",
    "DEFAULT_TPWINDOW_V5",
    "DEFAULT_ENTRY_EPISODE_V5",
    "DEFAULT_SAME_SIDE_HOLD_V5",
    "DEFAULT_REGIME_DETECT_V5",
    "DEFAULT_REGIME_WEIGHT_V5",
    "DEFAULT_REGIME_THRESHOLD_V5",
    "DEFAULT_REGIME_FILTER_V5",
    "DEFAULT_REGIME_LANE_V5",
    "DEFAULT_EXECUTION_V5",
    "DEFAULT_BACKTEST_V5",
    "DEFAULT_OBJECTIVE_V5",
    "normalize_horizon_weights",
    "load_config_dataclass",
    "derive_artifact_paths",
]
