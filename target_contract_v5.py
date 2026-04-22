# -*- coding: utf-8 -*-
"""
target_contract_v5.py

V5 target / label contract (single source of truth).

핵심 방향
---------
- signal 은 close[t] 시점에 계산되지만, target reference 는 **open[t+1]** 기준이다.
- main return horizons: 1 / 3 / 5 / 8 / 10
- legacy 의 **dir + hyb** 구조를 정식 이식한다.
  - return regression target 은 그대로 유지한다.
  - binary direction target 을 `1/3/5/8/10` 전부 둔다.
  - utility 도 `1/3/5/8/10` 전부 둔다.
  - retcls 도 `1/3/5/8/10` 전부 둔다.
- main path horizon 은 10 이고, extension horizon 은 제거한다.
- short-hold entry 는 multi-horizon 으로 보고, thesis / exit geometry 는 계속 path10 중심으로 유지한다.
- scale_ref 는 v5 feature contract naming 에 맞춰 `atr10_rel` 을 기준으로 고정한다.

설계 의도
---------
- entry timing 은 legacy 의 fast multi-horizon signal 철학을 복구한다.
- exit geometry / thesis monitoring 은 v4 의 `path10 / no-extension / open[t+1]` 정렬을 유지한다.
- 따라서 v5 의 핵심 조합은
  `legacy_entry(1/3/5/8/10 dir+hyb)` + `v4_exit(path10)` 이다.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence, Tuple

# -----------------------------------------------------------------------------
# Version / horizon policy
# -----------------------------------------------------------------------------

TARGET_CONTRACT_VERSION: Final[str] = "target_v5_openref_1_3_5_8_10_dirhyb_path10_noext_r1"

TARGET_REFERENCE_MODE: Final[str] = "next_open"
TARGET_ENTRY_REF_EXPR: Final[str] = "entry_ref_t = open[t+1]"
TARGET_TIMING_NOTE: Final[str] = "signal at close[t], label reference aligned to next bar open"

APPROVED_RETURN_HORIZONS: Final[Tuple[int, ...]] = (1, 3, 5, 8, 10)
RETURN_HORIZONS_MAIN: Final[Tuple[int, ...]] = (1, 3, 5, 8, 10)
DIR_HORIZONS: Final[Tuple[int, ...]] = (1, 3, 5, 8, 10)
CLASS_HORIZONS: Final[Tuple[int, ...]] = (1, 3, 5, 8, 10)
UTILITY_HORIZONS: Final[Tuple[int, ...]] = (1, 3, 5, 8, 10)
PATH_HORIZON_MAIN: Final[int] = 10
PATH_HORIZON_EXTENSION: Final[int] = 0  # removed in v5 as in v4

ENTRY_HORIZONS: Final[Tuple[int, ...]] = (1, 3, 5, 8, 10)
THESIS_ANCHOR_HORIZON: Final[int] = 10

BARRIERS_ATR_MAIN: Final[Tuple[float, ...]] = (1.0, 1.5, 2.0)
FIRST_HIT_BARRIERS_ATR: Final[Tuple[float, ...]] = (1.0, 1.5)
TTH_BARRIERS_ATR: Final[Tuple[float, ...]] = (1.0,)
QUANTILES: Final[Tuple[float, ...]] = (0.50, 0.75, 0.90)

# tighter 7-bin return classes for short-hold scalp regime
CLASS_BINS_ATR: Final[Tuple[float, ...]] = (-999.0, -1.5, -0.75, -0.20, 0.20, 0.75, 1.5, 999.0)

# -----------------------------------------------------------------------------
# Scale / normalization policy
# -----------------------------------------------------------------------------

TARGET_SCALE_REF_FEATURE: Final[str] = "atr10_rel"
TARGET_SCALE_REF_EXPR: Final[str] = "scale_ref_t = max(atr10_rel_t, fee_floor_ret_t, 1e-6)"
TARGET_FEE_FLOOR_EXPR: Final[str] = "fee_floor_ret_t = 2 * (cost_per_side + slip_per_side)"
TARGET_FEE_N_EXPR: Final[str] = "fee_n_t = fee_floor_ret_t / scale_ref_t"

# -----------------------------------------------------------------------------
# Direction / hybrid policy
# -----------------------------------------------------------------------------

DIR_TARGET_MODE: Final[str] = "binary_with_neutral_band"
DIR_POSITIVE_VALUE: Final[float] = 1.0
DIR_NEGATIVE_VALUE: Final[float] = 0.0
DIR_NEUTRAL_VALUE: Final[float] = float("nan")
DIR_NEUTRAL_NOTE: Final[str] = "1=UP_OR_LONG, 0=DOWN_OR_SHORT, NaN=neutral band"

# legacy-friendly default epsilon schedule.
# fee_n_t already lives in normalized return space, so eps_h is also normalized.
DIR_EPS_FEE_MULT_BASE: Final[float] = 1.0
DIR_EPS_BASE_EXPR: Final[str] = "dir_eps_base_t = fee_n_t * dir_eps_fee_mult_base"
DIR_EPS_MULT_BY_HORIZON: Final[dict[int, float]] = {
    1: 0.60,
    3: 1.00,
    5: 1.00,
    8: 1.00,
    10: 1.00,
}
DIR_EPS_EXPR_TEMPLATE: Final[str] = "eps_h_t = fee_n_t * dir_eps_fee_mult_base * dir_eps_mult[h]"
HYBRID_SIGNAL_EXPR: Final[str] = "predhy_h = decode_reg_h(pred_tgt_ret_h_n_raw) * tanh(predlogit_tgt_dir_h / 2.0)"
HYBRID_TARGET_NOTE: Final[str] = "hyb target is the signed normalized return tgt_ret_h_n; no separate tgt_hyb_h column is stored"

DIR_LABEL_TO_NAME: Final[dict[int, str]] = {
    0: "DOWN_OR_SHORT",
    1: "UP_OR_LONG",
}

# -----------------------------------------------------------------------------
# Utility policy
# -----------------------------------------------------------------------------

UTILITY_LAMBDA_ADVERSE: Final[float] = 1.20
UTILITY_CAP_FAV_BY_HORIZON: Final[dict[int, float]] = {h: 3.0 for h in UTILITY_HORIZONS}
UTILITY_CAP_ADV_BY_HORIZON: Final[dict[int, float]] = {h: 2.5 for h in UTILITY_HORIZONS}

# keep v4 constant names as compatibility-free v5 aliases for easier migration
UTILITY_CAP_FAV_MAIN: Final[float] = UTILITY_CAP_FAV_BY_HORIZON[THESIS_ANCHOR_HORIZON]
UTILITY_CAP_ADV_MAIN: Final[float] = UTILITY_CAP_ADV_BY_HORIZON[THESIS_ANCHOR_HORIZON]
UTILITY_CAP_FAV_EXTENSION: Final[float] = 0.0
UTILITY_CAP_ADV_EXTENSION: Final[float] = 0.0

# -----------------------------------------------------------------------------
# Classification / censoring names
# -----------------------------------------------------------------------------

RET_CLASS_ID_TO_NAME: Final[dict[int, str]] = {
    0: "strong_dn",
    1: "mod_dn",
    2: "weak_dn",
    3: "neutral",
    4: "weak_up",
    5: "mod_up",
    6: "strong_up",
}

FIRST_HIT_ID_TO_NAME: Final[dict[int, str]] = {
    0: "NONE_OR_TIE",
    1: "UP_FIRST",
    2: "DOWN_FIRST",
}

TTH_CENSORED_VALUE_MAIN: Final[int] = PATH_HORIZON_MAIN + 1
TTH_CENSORED_VALUE_EXTENSION: Final[int] = 0

# -----------------------------------------------------------------------------
# Canonical target lists
# -----------------------------------------------------------------------------

RETURN_TARGETS_MAIN: Final[Tuple[str, ...]] = tuple(f"tgt_ret_{h}_n" for h in RETURN_HORIZONS_MAIN)
PATH_TARGETS_MAIN: Final[Tuple[str, ...]] = (
    f"tgt_up_excur_{PATH_HORIZON_MAIN}_n",
    f"tgt_down_excur_{PATH_HORIZON_MAIN}_n",
)
UTILITY_TARGETS_MAIN: Final[Tuple[str, ...]] = tuple(
    tgt
    for h in UTILITY_HORIZONS
    for tgt in (f"tgt_long_utility_{h}", f"tgt_short_utility_{h}")
)

DIR_TARGETS_MAIN: Final[Tuple[str, ...]] = tuple(f"tgt_dir_{h}" for h in DIR_HORIZONS)
RETCLS_TARGETS_MAIN: Final[Tuple[str, ...]] = tuple(f"tgt_retcls_{h}" for h in CLASS_HORIZONS)
BARRIER_HIT_TARGETS_MAIN: Final[Tuple[str, ...]] = tuple(
    [f"tgt_up_hit_{int(round(b * 100.0)):03d}_{PATH_HORIZON_MAIN}" for b in BARRIERS_ATR_MAIN]
    + [f"tgt_down_hit_{int(round(b * 100.0)):03d}_{PATH_HORIZON_MAIN}" for b in BARRIERS_ATR_MAIN]
)
FIRST_HIT_TARGETS_MAIN: Final[Tuple[str, ...]] = tuple(
    f"tgt_first_hit_{int(round(b * 100.0)):03d}_{PATH_HORIZON_MAIN}" for b in FIRST_HIT_BARRIERS_ATR
)
TTH_TARGETS_MAIN: Final[Tuple[str, ...]] = tuple(
    [f"tgt_tth_up_{int(round(b * 100.0)):03d}_{PATH_HORIZON_MAIN}" for b in TTH_BARRIERS_ATR]
    + [f"tgt_tth_down_{int(round(b * 100.0)):03d}_{PATH_HORIZON_MAIN}" for b in TTH_BARRIERS_ATR]
)

REGRESSION_TARGETS_MAIN: Final[Tuple[str, ...]] = (
    *RETURN_TARGETS_MAIN,
    *PATH_TARGETS_MAIN,
    *UTILITY_TARGETS_MAIN,
)

CLASSIFICATION_TARGETS_MAIN: Final[Tuple[str, ...]] = (
    *DIR_TARGETS_MAIN,
    *RETCLS_TARGETS_MAIN,
    *BARRIER_HIT_TARGETS_MAIN,
    *FIRST_HIT_TARGETS_MAIN,
    *TTH_TARGETS_MAIN,
)

OPTIONAL_EXTENSION_TARGETS: Final[Tuple[str, ...]] = tuple()

# -----------------------------------------------------------------------------
# Grouped registries
# -----------------------------------------------------------------------------

REGRESSION_TARGET_GROUPS: Final["OrderedDict[str, Tuple[str, ...]]"] = OrderedDict(
    [
        ("main_future_returns", RETURN_TARGETS_MAIN),
        ("main_path_extremes", PATH_TARGETS_MAIN),
        ("main_utilities", UTILITY_TARGETS_MAIN),
    ]
)

CLASSIFICATION_TARGET_GROUPS: Final["OrderedDict[str, Tuple[str, ...]]"] = OrderedDict(
    [
        ("direction_binaries", DIR_TARGETS_MAIN),
        ("return_classes", RETCLS_TARGETS_MAIN),
        ("main_up_down_barrier_hits", BARRIER_HIT_TARGETS_MAIN),
        ("main_first_hit", FIRST_HIT_TARGETS_MAIN),
        ("time_to_hit_main", TTH_TARGETS_MAIN),
    ]
)

# -----------------------------------------------------------------------------
# Rich doc registry
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSpec:
    name: str
    family: str
    dtype: str
    description: str
    formula: str
    optional: bool = False


def _barrier_token(barrier_atr: float) -> str:
    return f"{int(round(float(barrier_atr) * 100.0)):03d}"


def _flatten(groups: Mapping[str, Sequence[str]]) -> Tuple[str, ...]:
    out: list[str] = []
    for cols in groups.values():
        out.extend(str(c) for c in cols)
    return tuple(dict.fromkeys(out))


def _assert_unique(items: Iterable[str], *, label: str) -> None:
    seen: set[str] = set()
    dup: list[str] = []
    for item in items:
        if item in seen:
            dup.append(item)
        seen.add(item)
    if dup:
        raise ValueError(f"duplicate {label}: {dup}")


TARGET_DOCS: dict[str, str] = {}

for h in RETURN_HORIZONS_MAIN:
    TARGET_DOCS[f"tgt_ret_{h}_n"] = f"log(close[t+{h}] / open[t+1]) / scale_ref_t"

TARGET_DOCS[f"tgt_up_excur_{PATH_HORIZON_MAIN}_n"] = (
    f"max_{{k=1..{PATH_HORIZON_MAIN}}} log(high[t+k] / open[t+1]) / scale_ref_t"
)
TARGET_DOCS[f"tgt_down_excur_{PATH_HORIZON_MAIN}_n"] = (
    f"max_{{k=1..{PATH_HORIZON_MAIN}}} -log(low[t+k] / open[t+1]) / scale_ref_t"
)

for h in UTILITY_HORIZONS:
    cap_f = UTILITY_CAP_FAV_BY_HORIZON[h]
    cap_a = UTILITY_CAP_ADV_BY_HORIZON[h]
    TARGET_DOCS[f"tgt_long_utility_{h}"] = (
        f"min(up_{h}, {cap_f:.1f}) - lambda_adv*min(down_{h}, {cap_a:.1f}) - fee_n_t"
    )
    TARGET_DOCS[f"tgt_short_utility_{h}"] = (
        f"min(down_{h}, {cap_f:.1f}) - lambda_adv*min(up_{h}, {cap_a:.1f}) - fee_n_t"
    )

for h in DIR_HORIZONS:
    TARGET_DOCS[f"tgt_dir_{h}"] = (
        f"1 if tgt_ret_{h}_n > eps_{h}, 0 if tgt_ret_{h}_n < -eps_{h}, NaN otherwise; "
        f"eps_{h}=fee_n_t*{DIR_EPS_FEE_MULT_BASE:.1f}*{DIR_EPS_MULT_BY_HORIZON[h]:.2f}"
    )

for h in CLASS_HORIZONS:
    TARGET_DOCS[f"tgt_retcls_{h}"] = f"7-bin class of tgt_ret_{h}_n using CLASS_BINS_ATR"

for barrier in BARRIERS_ATR_MAIN:
    tok = _barrier_token(barrier)
    TARGET_DOCS[f"tgt_up_hit_{tok}_{PATH_HORIZON_MAIN}"] = (
        f"1 if tgt_up_excur_{PATH_HORIZON_MAIN}_n >= {barrier:.1f} else 0"
    )
    TARGET_DOCS[f"tgt_down_hit_{tok}_{PATH_HORIZON_MAIN}"] = (
        f"1 if tgt_down_excur_{PATH_HORIZON_MAIN}_n >= {barrier:.1f} else 0"
    )

for barrier in FIRST_HIT_BARRIERS_ATR:
    tok = _barrier_token(barrier)
    TARGET_DOCS[f"tgt_first_hit_{tok}_{PATH_HORIZON_MAIN}"] = (
        f"0=NONE_OR_TIE, 1=UP_FIRST, 2=DOWN_FIRST for {barrier:.1f} barrier within {PATH_HORIZON_MAIN} bars"
    )

for barrier in TTH_BARRIERS_ATR:
    tok = _barrier_token(barrier)
    TARGET_DOCS[f"tgt_tth_up_{tok}_{PATH_HORIZON_MAIN}"] = (
        f"first k in [1,{PATH_HORIZON_MAIN}] where up excursion >= {barrier:.1f}, else censored={TTH_CENSORED_VALUE_MAIN}"
    )
    TARGET_DOCS[f"tgt_tth_down_{tok}_{PATH_HORIZON_MAIN}"] = (
        f"first k in [1,{PATH_HORIZON_MAIN}] where down excursion >= {barrier:.1f}, else censored={TTH_CENSORED_VALUE_MAIN}"
    )


def _dtype_for_target(name: str) -> str:
    if name in RETURN_TARGETS_MAIN or name in PATH_TARGETS_MAIN or name in UTILITY_TARGETS_MAIN:
        return "float32"
    if name in DIR_TARGETS_MAIN:
        return "float32(nullable_binary)"
    if name in RETCLS_TARGETS_MAIN:
        return "int8"
    if name in BARRIER_HIT_TARGETS_MAIN:
        return "int8"
    if name in FIRST_HIT_TARGETS_MAIN:
        return "int8"
    if name in TTH_TARGETS_MAIN:
        return "int16"
    return "unknown"


def _family_for_target(name: str) -> str:
    if name in RETURN_TARGETS_MAIN:
        return "regression_return"
    if name in PATH_TARGETS_MAIN:
        return "regression_path"
    if name in UTILITY_TARGETS_MAIN:
        return "regression_utility"
    if name in DIR_TARGETS_MAIN:
        return "classification_direction"
    if name in RETCLS_TARGETS_MAIN:
        return "classification_return_class"
    if name in BARRIER_HIT_TARGETS_MAIN:
        return "classification_barrier_hit"
    if name in FIRST_HIT_TARGETS_MAIN:
        return "classification_first_hit"
    if name in TTH_TARGETS_MAIN:
        return "classification_tth"
    return "unknown"


TARGET_SPECS: Final[Tuple[TargetSpec, ...]] = tuple(
    TargetSpec(
        name=name,
        family=_family_for_target(name),
        dtype=_dtype_for_target(name),
        description=TARGET_DOCS[name],
        formula=TARGET_DOCS[name],
        optional=False,
    )
    for name in (
        *REGRESSION_TARGETS_MAIN,
        *CLASSIFICATION_TARGETS_MAIN,
    )
)

# -----------------------------------------------------------------------------
# Flattened convenience lists
# -----------------------------------------------------------------------------

REGRESSION_TARGET_COLUMNS: Final[Tuple[str, ...]] = REGRESSION_TARGETS_MAIN
CLASSIFICATION_TARGET_COLUMNS: Final[Tuple[str, ...]] = CLASSIFICATION_TARGETS_MAIN

ALL_TARGET_COLUMNS: Final[Tuple[str, ...]] = (
    *REGRESSION_TARGETS_MAIN,
    *CLASSIFICATION_TARGETS_MAIN,
)
DEFAULT_MAIN_TARGET_COLUMNS: Final[Tuple[str, ...]] = ALL_TARGET_COLUMNS

TARGET_GROUPS: Final["OrderedDict[str, Tuple[str, ...]]"] = OrderedDict(
    [
        *REGRESSION_TARGET_GROUPS.items(),
        *CLASSIFICATION_TARGET_GROUPS.items(),
    ]
)


def build_target_list(*, include_extension: bool = False) -> Tuple[str, ...]:
    # extension 은 v5 에서도 제거되었으므로 include_extension 인자는 호환용으로만 남긴다.
    return ALL_TARGET_COLUMNS


# -----------------------------------------------------------------------------
# Contract validation
# -----------------------------------------------------------------------------

_assert_unique(REGRESSION_TARGETS_MAIN, label="regression targets main")
_assert_unique(CLASSIFICATION_TARGETS_MAIN, label="classification targets main")
_assert_unique(OPTIONAL_EXTENSION_TARGETS, label="optional extension targets")
_assert_unique(ALL_TARGET_COLUMNS, label="all target columns")

__all__ = [
    "TargetSpec",
    "TARGET_CONTRACT_VERSION",
    "TARGET_REFERENCE_MODE",
    "TARGET_ENTRY_REF_EXPR",
    "TARGET_TIMING_NOTE",
    "APPROVED_RETURN_HORIZONS",
    "RETURN_HORIZONS_MAIN",
    "DIR_HORIZONS",
    "CLASS_HORIZONS",
    "UTILITY_HORIZONS",
    "ENTRY_HORIZONS",
    "THESIS_ANCHOR_HORIZON",
    "PATH_HORIZON_MAIN",
    "PATH_HORIZON_EXTENSION",
    "BARRIERS_ATR_MAIN",
    "FIRST_HIT_BARRIERS_ATR",
    "TTH_BARRIERS_ATR",
    "QUANTILES",
    "CLASS_BINS_ATR",
    "RET_CLASS_ID_TO_NAME",
    "FIRST_HIT_ID_TO_NAME",
    "DIR_LABEL_TO_NAME",
    "DIR_TARGET_MODE",
    "DIR_POSITIVE_VALUE",
    "DIR_NEGATIVE_VALUE",
    "DIR_NEUTRAL_VALUE",
    "DIR_NEUTRAL_NOTE",
    "DIR_EPS_FEE_MULT_BASE",
    "DIR_EPS_BASE_EXPR",
    "DIR_EPS_MULT_BY_HORIZON",
    "DIR_EPS_EXPR_TEMPLATE",
    "HYBRID_SIGNAL_EXPR",
    "HYBRID_TARGET_NOTE",
    "TARGET_SCALE_REF_FEATURE",
    "TARGET_SCALE_REF_EXPR",
    "TARGET_FEE_FLOOR_EXPR",
    "TARGET_FEE_N_EXPR",
    "UTILITY_LAMBDA_ADVERSE",
    "UTILITY_CAP_FAV_BY_HORIZON",
    "UTILITY_CAP_ADV_BY_HORIZON",
    "UTILITY_CAP_FAV_MAIN",
    "UTILITY_CAP_ADV_MAIN",
    "UTILITY_CAP_FAV_EXTENSION",
    "UTILITY_CAP_ADV_EXTENSION",
    "TTH_CENSORED_VALUE_MAIN",
    "TTH_CENSORED_VALUE_EXTENSION",
    "RETURN_TARGETS_MAIN",
    "PATH_TARGETS_MAIN",
    "UTILITY_TARGETS_MAIN",
    "DIR_TARGETS_MAIN",
    "RETCLS_TARGETS_MAIN",
    "BARRIER_HIT_TARGETS_MAIN",
    "FIRST_HIT_TARGETS_MAIN",
    "TTH_TARGETS_MAIN",
    "REGRESSION_TARGETS_MAIN",
    "CLASSIFICATION_TARGETS_MAIN",
    "OPTIONAL_EXTENSION_TARGETS",
    "REGRESSION_TARGET_GROUPS",
    "CLASSIFICATION_TARGET_GROUPS",
    "TARGET_GROUPS",
    "TARGET_DOCS",
    "TARGET_SPECS",
    "REGRESSION_TARGET_COLUMNS",
    "CLASSIFICATION_TARGET_COLUMNS",
    "ALL_TARGET_COLUMNS",
    "DEFAULT_MAIN_TARGET_COLUMNS",
    "build_target_list",
]
