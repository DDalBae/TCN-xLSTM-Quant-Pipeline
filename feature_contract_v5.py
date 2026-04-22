# -*- coding: utf-8 -*-
"""
feature_contract_v5.py

V5 reactive feature contract (single source of truth).

핵심 원칙
---------
- v5는 v4 하위호환을 고려하지 않는다.
- raw schema는 최소 OHLCV + taker_buy_base 만 유지한다.
- 모델 입력은 reactive core 26 feature로 고정한다.
- 30-window / funding / quote/trade_count 기반 의존성은 제거한다.
- gap audit + feature_ready 는 downstream 전 구간의 공통 계약이다.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Final, Iterable, Mapping, Sequence, Tuple

# -----------------------------------------------------------------------------
# Version / source contract
# -----------------------------------------------------------------------------

RAW_SOURCE_NAME: Final[str] = "binance_um_futures"
RAW_SCHEMA_VERSION: Final[str] = "raw_v5_binance_um_1m_minimal_r1"
FEATURE_CONTRACT_VERSION: Final[str] = "feat_v5_reactive26_r1"
EXPECTED_INTERVAL: Final[str] = "1m"
EXPECTED_BAR_SECONDS: Final[int] = 60
EXPECTED_BAR_NS: Final[int] = EXPECTED_BAR_SECONDS * 1_000_000_000

# -----------------------------------------------------------------------------
# Raw market contract
# -----------------------------------------------------------------------------

RAW_COLUMNS: Final[Tuple[str, ...]] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "taker_buy_base",
)
RAW_NUMERIC_COLUMNS: Final[Tuple[str, ...]] = tuple(c for c in RAW_COLUMNS if c != "timestamp")
RAW_INT_COLUMNS: Final[Tuple[str, ...]] = tuple()
RAW_NONNEGATIVE_COLUMNS: Final[Tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "taker_buy_base",
)

# -----------------------------------------------------------------------------
# Audit / readiness columns
# -----------------------------------------------------------------------------

GAP_AUDIT_COLUMNS: Final[Tuple[str, ...]] = (
    "gap_prev_min",
    "gap_prev_bad",
    "contig_run_id",
    "contig_pos",
)
FEATURE_READY_COL: Final[str] = "feature_ready"
AUX_AUDIT_COLUMNS: Final[Tuple[str, ...]] = (
    *GAP_AUDIT_COLUMNS,
    FEATURE_READY_COL,
)

# -----------------------------------------------------------------------------
# Core 26 feature contract
# -----------------------------------------------------------------------------

CANDLE_MICRO_FEATURES: Final[Tuple[str, ...]] = (
    "gap_open",
    "high_ext",
    "low_ext",
    "body_ret",
    "upper_wick_rel",
    "lower_wick_rel",
    "wick_ratio",
)

RETURN_FEATURES: Final[Tuple[str, ...]] = (
    "r1",
    "r3",
    "r5",
    "r8",
    "r10",
)

VOLATILITY_FEATURES: Final[Tuple[str, ...]] = (
    "atr1_rel",
    "atr3_rel",
    "atr5_rel",
    "atr8_rel",
    "atr10_rel",
)

ACTIVITY_FEATURES: Final[Tuple[str, ...]] = (
    "vol_z_3",
    "vol_z_5",
    "vol_z_8",
    "vol_z_10",
)

FLOW_STATE_FEATURES: Final[Tuple[str, ...]] = (
    "taker_buy_ratio",
    "rolling_vwap_dist_5",
    "rolling_vwap_dist_10",
    "efficiency_ratio_10",
    "bb_pctb_20",
)

CORE_FEATURE_GROUPS: Final["OrderedDict[str, Tuple[str, ...]]"] = OrderedDict(
    [
        ("candle_micro", CANDLE_MICRO_FEATURES),
        ("returns", RETURN_FEATURES),
        ("volatility", VOLATILITY_FEATURES),
        ("activity", ACTIVITY_FEATURES),
        ("flow_state", FLOW_STATE_FEATURES),
    ]
)

GROUP_DESCRIPTIONS_KO: Final[dict[str, str]] = {
    "candle_micro": "갭/캔들 바디/진짜 wick 기반 마이크로 구조",
    "returns": "1/3/5/8/10 close-to-close log return ladder",
    "volatility": "1/3/5/8/10 ATR-relative volatility ladder",
    "activity": "3/5/8/10 volume z-score ladder",
    "flow_state": "taker ratio + rolling VWAP stretch + efficiency + BB state",
}

REQUIRED_HISTORY_BARS: Final[int] = 20

# -----------------------------------------------------------------------------
# Flattened lists / helpers
# -----------------------------------------------------------------------------


def _flatten(groups: Mapping[str, Sequence[str]]) -> Tuple[str, ...]:
    out: list[str] = []
    for cols in groups.values():
        out.extend(str(c) for c in cols)
    return tuple(dict.fromkeys(out))


REQUIRED_FEATURE_COLUMNS: Final[Tuple[str, ...]] = _flatten(CORE_FEATURE_GROUPS)
ALL_DERIVED_FEATURE_COLUMNS: Final[Tuple[str, ...]] = REQUIRED_FEATURE_COLUMNS
DEFAULT_MODEL_INPUT_COLUMNS: Final[Tuple[str, ...]] = REQUIRED_FEATURE_COLUMNS

FEATURE_BUILD_OUTPUT_COLUMNS: Final[Tuple[str, ...]] = tuple(
    dict.fromkeys(
        (
            *RAW_COLUMNS,
            *AUX_AUDIT_COLUMNS,
            *ALL_DERIVED_FEATURE_COLUMNS,
        )
    )
)

FEATURE_TO_GROUP: Final[dict[str, str]] = {
    feature: group_name
    for group_name, group_cols in CORE_FEATURE_GROUPS.items()
    for feature in group_cols
}


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------


def build_feature_list() -> Tuple[str, ...]:
    return DEFAULT_MODEL_INPUT_COLUMNS


def iter_grouped_features() -> Iterable[tuple[str, Tuple[str, ...]]]:
    yield from CORE_FEATURE_GROUPS.items()


def get_required_history_bars() -> int:
    return int(REQUIRED_HISTORY_BARS)


def get_sequence_history_extra(seq_len: int) -> int:
    seq_len = int(seq_len)
    return int(seq_len + get_required_history_bars() - 1)


# -----------------------------------------------------------------------------
# Contract validation
# -----------------------------------------------------------------------------


def _assert_unique(items: Iterable[str], *, label: str) -> None:
    seen: set[str] = set()
    dup: list[str] = []
    for item in items:
        if item in seen:
            dup.append(item)
        seen.add(item)
    if dup:
        raise ValueError(f"duplicate {label}: {dup}")


_assert_unique(RAW_COLUMNS, label="raw columns")
_assert_unique(RAW_NUMERIC_COLUMNS, label="raw numeric columns")
_assert_unique(RAW_NONNEGATIVE_COLUMNS, label="raw nonnegative columns")
_assert_unique(GAP_AUDIT_COLUMNS, label="gap audit columns")
_assert_unique(AUX_AUDIT_COLUMNS, label="aux audit columns")
_assert_unique(REQUIRED_FEATURE_COLUMNS, label="required feature columns")
_assert_unique(ALL_DERIVED_FEATURE_COLUMNS, label="all derived feature columns")
_assert_unique(FEATURE_BUILD_OUTPUT_COLUMNS, label="feature build output columns")

__all__ = [
    "RAW_SOURCE_NAME",
    "RAW_SCHEMA_VERSION",
    "FEATURE_CONTRACT_VERSION",
    "EXPECTED_INTERVAL",
    "EXPECTED_BAR_SECONDS",
    "EXPECTED_BAR_NS",
    "RAW_COLUMNS",
    "RAW_NUMERIC_COLUMNS",
    "RAW_INT_COLUMNS",
    "RAW_NONNEGATIVE_COLUMNS",
    "GAP_AUDIT_COLUMNS",
    "FEATURE_READY_COL",
    "AUX_AUDIT_COLUMNS",
    "CANDLE_MICRO_FEATURES",
    "RETURN_FEATURES",
    "VOLATILITY_FEATURES",
    "ACTIVITY_FEATURES",
    "FLOW_STATE_FEATURES",
    "CORE_FEATURE_GROUPS",
    "GROUP_DESCRIPTIONS_KO",
    "REQUIRED_HISTORY_BARS",
    "REQUIRED_FEATURE_COLUMNS",
    "ALL_DERIVED_FEATURE_COLUMNS",
    "DEFAULT_MODEL_INPUT_COLUMNS",
    "FEATURE_BUILD_OUTPUT_COLUMNS",
    "FEATURE_TO_GROUP",
    "build_feature_list",
    "iter_grouped_features",
    "get_required_history_bars",
    "get_sequence_history_extra",
]
