# -*- coding: utf-8 -*-
"""
label_builder_v5_1.py

V5.1 offline label / target builder.

핵심 원칙
---------
- 이 모듈은 **오프라인 학습용 타겟 생성기**다.
- 입력 피처는 `feature_ops_v5.py` 가 만들고, 본 모듈은 그 위에 미래 구간을 사용해 라벨을 만든다.
- 따라서 본 모듈의 미래 참조는 **의도적이며 타겟 생성 전용**이다.
- live / backtest / feature 생성 경로에는 이 미래참조 로직이 절대 들어가면 안 된다.
- row-wise Python for-loop 금지. 허용되는 반복은 horizon / barrier 같은 작은 메타데이터 반복뿐이다.
- gap 을 가로질러 미래 라벨을 만들지 않도록 contiguity(run) 기준 valid mask를 만든다.

v5.1 방향
---------
- feature contract 는 `feat_v5_reactive26_r1` 을 사용한다.
- raw schema 는 minimal OHLCV + taker_buy_base 로 축소되었다.
- target semantics 는 **target_contract_v5.py** 를 따른다.
  - reference = `open[t+1]`
  - return horizons = `1 / 3 / 5 / 8 / 10`
  - direction targets = `1 / 3 / 5 / 8 / 10`
  - utility targets = `1 / 3 / 5 / 8 / 10`
  - retcls targets = `1 / 3 / 5 / 8 / 10`
  - path / thesis anchor = `10`
  - no-extension
- scale_ref 입력은 v5 reactive feature naming 의 `atr10_rel` 을 사용한다.

권장 흐름
---------
raw parquet/csv -> sanitize_v5_1(raw) -> split raw -> feature_ops_v5 -> label_builder_v5_1

출력
----
- feature frame + v5.1 target columns + label audit columns
- meta json (선택)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from feature_contract_v5 import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_READY_COL,
    GAP_AUDIT_COLUMNS,
    RAW_COLUMNS,
    REQUIRED_FEATURE_COLUMNS,
)
from feature_ops_v5 import (
    FeatureBuildConfig,
    build_features,
    read_frame,
    summarize_feature_frame,
    write_frame,
)
from target_contract_v5 import (
    ALL_TARGET_COLUMNS,
    APPROVED_RETURN_HORIZONS,
    BARRIERS_ATR_MAIN,
    CLASS_BINS_ATR,
    CLASS_HORIZONS,
    CLASSIFICATION_TARGETS_MAIN,
    DIR_EPS_FEE_MULT_BASE,
    DIR_EPS_MULT_BY_HORIZON,
    DIR_HORIZONS,
    DIR_LABEL_TO_NAME,
    DIR_NEGATIVE_VALUE,
    DIR_NEUTRAL_NOTE,
    DIR_NEUTRAL_VALUE,
    DIR_POSITIVE_VALUE,
    DIR_TARGET_MODE,
    FIRST_HIT_BARRIERS_ATR,
    FIRST_HIT_ID_TO_NAME,
    HYBRID_SIGNAL_EXPR,
    HYBRID_TARGET_NOTE,
    OPTIONAL_EXTENSION_TARGETS,
    PATH_HORIZON_EXTENSION,
    PATH_HORIZON_MAIN,
    REGRESSION_TARGETS_MAIN,
    RET_CLASS_ID_TO_NAME,
    RETURN_HORIZONS_MAIN,
    TARGET_CONTRACT_VERSION,
    TARGET_DOCS,
    TARGET_ENTRY_REF_EXPR,
    TARGET_FEE_FLOOR_EXPR,
    TARGET_FEE_N_EXPR,
    TARGET_REFERENCE_MODE,
    TARGET_SCALE_REF_EXPR,
    TARGET_SCALE_REF_FEATURE,
    TARGET_TIMING_NOTE,
    TTH_BARRIERS_ATR,
    TTH_CENSORED_VALUE_EXTENSION,
    TTH_CENSORED_VALUE_MAIN,
    UTILITY_CAP_ADV_BY_HORIZON,
    UTILITY_CAP_FAV_BY_HORIZON,
    UTILITY_HORIZONS,
    UTILITY_LAMBDA_ADVERSE,
    build_target_list,
)

EPS = 1.0e-12
LABEL_BUILDER_VERSION = "label_builder_v5_1_r1"
INVALID_CLASS_VALUE = np.int8(-1)

TARGET_SCALE_REF_FEATURE_V5 = TARGET_SCALE_REF_FEATURE
TARGET_SCALE_REF_FEATURE_CONTRACT_ALIAS = TARGET_SCALE_REF_FEATURE
TARGET_SCALE_REF_EXPR_V5 = TARGET_SCALE_REF_EXPR

# label audit columns are not model inputs. They help trainer/backtest filter safely.
LABEL_AUDIT_COLUMNS: Tuple[str, ...] = (
    "scale_ref_t",
    "fee_floor_ret_t",
    "fee_n_t",
    "target_ready_main",
    "dataset_ready_main",
)

# Defensive cleanup list for frames that may already contain old labels.
DEPRECATED_LABEL_COLUMNS: Tuple[str, ...] = (
    "target_ready_extension",
    "dataset_ready_extension",
)


@dataclass(frozen=True)
class LabelBuildConfig:
    """Configuration for offline label building.

    Parameters
    ----------
    input_kind:
        - "auto": feature columns가 이미 있으면 그대로 사용, 아니면 feature_ops_v5로 생성
        - "raw": 항상 raw -> feature 재생성
        - "features": 이미 v5 feature frame이라고 가정
        - "dataset": 기존 dataset frame이어도 raw/feature 컬럼만 읽고 타겟을 다시 생성
    cost_per_side / slip_per_side:
        fee floor = 2 * (cost_per_side + slip_per_side) 를 만드는 입력.
        target contract 수식과 동일하다.
    cast_float32:
        파생 regression target / audit float를 float32로 내려 메모리 사용을 줄인다.
    preserve_extra_columns:
        input frame에 contract 외 추가 컬럼이 있으면 유지할지 여부.
    feature_mask_not_ready:
        raw 입력에서 feature_ops_v5를 호출할 때 초기/gap 영향 row를 마스킹할지 여부.
    """

    input_kind: str = "auto"
    cost_per_side: float = 0.00070
    slip_per_side: float = 0.00015
    cast_float32: bool = True
    preserve_extra_columns: bool = True
    feature_mask_not_ready: bool = True


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _assert_has_columns(df: pd.DataFrame, cols: Iterable[str], *, label: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"missing {label} columns: {miss}")



def _to_bool_np(s: pd.Series, *, default: bool = False) -> np.ndarray:
    if s is None:
        return np.zeros(0, dtype=bool)
    try:
        return s.fillna(default).astype(bool).to_numpy(copy=False)
    except Exception:
        return pd.Series(s).fillna(default).astype(bool).to_numpy(copy=False)



def _as_float64_np(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").astype("float64").to_numpy(copy=False)



def _as_int64_np(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").to_numpy(copy=False)



def _needs_feature_build(df: pd.DataFrame) -> bool:
    required = set(RAW_COLUMNS) | set(REQUIRED_FEATURE_COLUMNS) | {FEATURE_READY_COL, *GAP_AUDIT_COLUMNS}
    return not required.issubset(df.columns)



def _strip_existing_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c
        for c in df.columns
        if c.startswith("tgt_") or c in LABEL_AUDIT_COLUMNS or c in DEPRECATED_LABEL_COLUMNS
    ]
    if not drop_cols:
        return df.copy()
    return df.drop(columns=drop_cols, errors="ignore").copy()



def _ensure_feature_frame(df: pd.DataFrame, cfg: LabelBuildConfig) -> Tuple[pd.DataFrame, bool]:
    """Return (feature_frame, built_now)."""
    input_kind = str(cfg.input_kind).strip().lower()
    if input_kind not in {"auto", "raw", "features", "dataset"}:
        raise ValueError(f"unsupported input_kind: {cfg.input_kind}")

    if input_kind in {"features", "dataset"}:
        feat = _strip_existing_label_columns(df)
        _assert_has_columns(feat, RAW_COLUMNS, label="raw")
        _assert_has_columns(feat, REQUIRED_FEATURE_COLUMNS, label="required feature")
        _assert_has_columns(feat, GAP_AUDIT_COLUMNS, label="gap audit")
        _assert_has_columns(feat, (FEATURE_READY_COL,), label="feature ready")
        return feat, False

    if input_kind == "auto" and not _needs_feature_build(df):
        feat = _strip_existing_label_columns(df)
        return feat, False

    feat = build_features(
        df,
        config=FeatureBuildConfig(
            mask_not_ready=bool(cfg.feature_mask_not_ready),
            cast_float32=bool(cfg.cast_float32),
            preserve_extra_columns=bool(cfg.preserve_extra_columns),
        ),
    )
    feat = _strip_existing_label_columns(feat)
    return feat, True


# NOTE:
# label_builder는 오프라인 타겟 생성기이므로 아래 helper들은 미래 구간을 읽는다.
# 이는 input feature 생성과 분리된 학습용 처리이며, live/runtime에는 절대 재사용하면 안 된다.


def _future_valid_mask(run_id: np.ndarray, horizon: int) -> np.ndarray:
    """Return mask where signal bar t and future horizon t+H stay inside one contiguous run.

    Notes
    -----
    - signal 시점은 close[t] 이고 entry reference 는 open[t+1] 이다.
    - 따라서 run_id[t] == run_id[t+H] 이면 t+1..t+H 구간에 gap 이 없고,
      signal bar 와 entry bar 사이(t -> t+1)도 같은 contiguous run 안에 있다.
    """
    horizon = int(horizon)
    n = int(len(run_id))
    out = np.zeros(n, dtype=bool)
    if horizon <= 0 or n <= horizon:
        return out
    idx = np.arange(n - horizon, dtype=np.int64)
    out[idx] = run_id[idx] == run_id[idx + horizon]
    return out



def _entry_ref_next_open(open_: np.ndarray) -> np.ndarray:
    n = int(len(open_))
    out = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        out[:-1] = open_[1:]
    return out



def _future_return_n(
    close: np.ndarray,
    open_: np.ndarray,
    scale_ref: np.ndarray,
    run_id: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized future return using entry reference open[t+1].

    tgt_ret_h_n[t] = log(close[t+h] / open[t+1]) / scale_ref_t
    """
    horizon = int(horizon)
    n = int(len(close))
    out = np.full(n, np.nan, dtype=np.float64)
    valid = _future_valid_mask(run_id, horizon)
    if n <= horizon:
        return out, valid

    idx = np.arange(n - horizon, dtype=np.int64)
    entry_ref = open_[idx + 1]
    future_close = close[idx + horizon]

    base_ok = valid[idx] & np.isfinite(scale_ref[idx]) & (scale_ref[idx] > 0.0)
    base_ok &= np.isfinite(entry_ref) & np.isfinite(future_close)
    base_ok &= (entry_ref > 0.0) & (future_close > 0.0)

    vals = np.full(n - horizon, np.nan, dtype=np.float64)
    if np.any(base_ok):
        vals[base_ok] = np.log(future_close[base_ok] / entry_ref[base_ok]) / scale_ref[idx[base_ok]]
    out[idx] = vals
    return out, valid



def _future_path_extremes_n(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    scale_ref: np.ndarray,
    run_id: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized favorable/adverse path extremes over next `horizon` bars.

    semantics
    ---------
    up_excur_n[t]   = max_{k=1..H}  log(high[t+k] / open[t+1]) / scale_ref_t
    down_excur_n[t] = max_{k=1..H} -log(low[t+k]  / open[t+1]) / scale_ref_t
    """
    horizon = int(horizon)
    n = int(len(open_))
    up = np.full(n, np.nan, dtype=np.float64)
    down = np.full(n, np.nan, dtype=np.float64)
    valid = _future_valid_mask(run_id, horizon)
    if n <= horizon:
        return up, down, valid

    high_fwd = sliding_window_view(high[1:], horizon)
    low_fwd = sliding_window_view(low[1:], horizon)
    future_high_max = np.nanmax(high_fwd, axis=1)
    future_low_min = np.nanmin(low_fwd, axis=1)

    idx = np.arange(n - horizon, dtype=np.int64)
    entry_ref = open_[idx + 1]

    base_ok = valid[idx] & np.isfinite(scale_ref[idx]) & (scale_ref[idx] > 0.0)
    base_ok &= np.isfinite(entry_ref) & (entry_ref > 0.0)
    base_ok &= np.isfinite(future_high_max) & np.isfinite(future_low_min)
    base_ok &= (future_high_max > 0.0) & (future_low_min > 0.0)

    if np.any(base_ok):
        pos = idx[base_ok]
        ref = entry_ref[base_ok]
        scale = scale_ref[pos]
        up[pos] = np.log(future_high_max[base_ok] / ref) / scale
        down[pos] = -np.log(future_low_min[base_ok] / ref) / scale
    return up, down, valid



def _utility_from_extremes(
    favorable_n: np.ndarray,
    adverse_n: np.ndarray,
    fee_n: np.ndarray,
    *,
    cap_fav: float,
    cap_adv: float,
    lambda_adv: float,
) -> np.ndarray:
    out = np.full_like(favorable_n, np.nan, dtype=np.float64)
    ok = np.isfinite(favorable_n) & np.isfinite(adverse_n) & np.isfinite(fee_n)
    if np.any(ok):
        fav_clip = np.minimum(favorable_n[ok], float(cap_fav))
        adv_clip = np.minimum(adverse_n[ok], float(cap_adv))
        out[ok] = fav_clip - float(lambda_adv) * adv_clip - fee_n[ok]
    return out



def _direction_target_from_return(
    ret_n: np.ndarray,
    fee_n: np.ndarray,
    *,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dir_target, valid_mask, eps_h).

    Direction target semantics
    --------------------------
    - 1.0 if ret_h > +eps_h
    - 0.0 if ret_h < -eps_h
    - NaN otherwise (neutral band)
    """
    horizon = int(horizon)
    out = np.full(len(ret_n), DIR_NEUTRAL_VALUE, dtype=np.float64)
    eps_h = np.full(len(ret_n), np.nan, dtype=np.float64)
    valid = np.isfinite(ret_n) & np.isfinite(fee_n)
    if np.any(valid):
        eps_h[valid] = fee_n[valid] * float(DIR_EPS_FEE_MULT_BASE) * float(DIR_EPS_MULT_BY_HORIZON[horizon])
        long_mask = valid & (ret_n > eps_h)
        short_mask = valid & (ret_n < -eps_h)
        out[long_mask] = float(DIR_POSITIVE_VALUE)
        out[short_mask] = float(DIR_NEGATIVE_VALUE)
    return out, valid, eps_h



def _class_from_bins(x: np.ndarray, bins: Tuple[float, ...], *, invalid_value: np.int8 = INVALID_CLASS_VALUE) -> np.ndarray:
    out = np.full(len(x), int(invalid_value), dtype=np.int8)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out
    inner = np.asarray(tuple(bins)[1:-1], dtype=np.float64)
    out[finite] = np.digitize(x[finite], inner, right=False).astype(np.int8)
    return out



def _binary_hit_from_extreme(x: np.ndarray, barrier: float, *, invalid_value: np.int8 = INVALID_CLASS_VALUE) -> np.ndarray:
    out = np.full(len(x), int(invalid_value), dtype=np.int8)
    finite = np.isfinite(x)
    if np.any(finite):
        out[finite] = (x[finite] >= float(barrier)).astype(np.int8)
    return out



def _first_hit_and_tth(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    scale_ref: np.ndarray,
    run_id: np.ndarray,
    horizon: int,
    barrier: float,
    *,
    invalid_value: np.int8 = INVALID_CLASS_VALUE,
    censored_value: int = TTH_CENSORED_VALUE_MAIN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (first_hit_class, tth_up, tth_down) using next-open entry reference.

    first_hit_class meaning:
    - -1: invalid (tail / gap / bad scale / bad entry_ref)
    -  0: NONE_OR_TIE
    -  1: UP_FIRST
    -  2: DOWN_FIRST

    tth values:
    - -1: invalid
    -  1..H: first hit bar offset
    - censored_value (= H+1): valid window but no hit within horizon
    """
    horizon = int(horizon)
    n = int(len(open_))
    first_hit = np.full(n, int(invalid_value), dtype=np.int8)
    tth_up = np.full(n, int(invalid_value), dtype=np.int16)
    tth_down = np.full(n, int(invalid_value), dtype=np.int16)
    valid = _future_valid_mask(run_id, horizon)
    if n <= horizon:
        return first_hit, tth_up, tth_down

    high_fwd = sliding_window_view(high[1:], horizon)
    low_fwd = sliding_window_view(low[1:], horizon)
    idx = np.arange(n - horizon, dtype=np.int64)
    entry_ref = open_[idx + 1]

    base_ok = valid[idx] & np.isfinite(scale_ref[idx]) & (scale_ref[idx] > 0.0)
    base_ok &= np.isfinite(entry_ref) & (entry_ref > 0.0)

    if np.any(base_ok):
        ref = entry_ref[base_ok]
        scale = scale_ref[idx[base_ok]]
        up_thr = ref * np.exp(float(barrier) * scale)
        down_thr = ref * np.exp(-float(barrier) * scale)

        up_hit_mat = high_fwd[base_ok] >= up_thr[:, None]
        down_hit_mat = low_fwd[base_ok] <= down_thr[:, None]

        up_any = up_hit_mat.any(axis=1)
        down_any = down_hit_mat.any(axis=1)

        up_first = np.where(up_any, np.argmax(up_hit_mat, axis=1) + 1, int(censored_value))
        down_first = np.where(down_any, np.argmax(down_hit_mat, axis=1) + 1, int(censored_value))

        local_first = np.zeros(len(ref), dtype=np.int8)
        local_first[(up_first < down_first)] = 1
        local_first[(down_first < up_first)] = 2

        pos = idx[base_ok]
        first_hit[pos] = local_first
        tth_up[pos] = up_first.astype(np.int16)
        tth_down[pos] = down_first.astype(np.int16)

    return first_hit, tth_up, tth_down


# -----------------------------------------------------------------------------
# Public build logic
# -----------------------------------------------------------------------------


def build_labels(df: pd.DataFrame, config: LabelBuildConfig | None = None) -> pd.DataFrame:
    cfg = config or LabelBuildConfig()
    feat_df, _built_now = _ensure_feature_frame(df, cfg)

    _assert_has_columns(feat_df, RAW_COLUMNS, label="raw")
    _assert_has_columns(feat_df, REQUIRED_FEATURE_COLUMNS, label="required feature")
    _assert_has_columns(feat_df, GAP_AUDIT_COLUMNS, label="gap audit")
    _assert_has_columns(feat_df, (FEATURE_READY_COL,), label="feature ready")
    _assert_has_columns(feat_df, (TARGET_SCALE_REF_FEATURE_V5,), label="v5 scale ref feature")

    open_ = _as_float64_np(feat_df["open"])
    close = _as_float64_np(feat_df["close"])
    high = _as_float64_np(feat_df["high"])
    low = _as_float64_np(feat_df["low"])
    run_id = _as_int64_np(feat_df["contig_run_id"])
    feature_ready = _to_bool_np(feat_df[FEATURE_READY_COL])

    scale_base = _as_float64_np(feat_df[TARGET_SCALE_REF_FEATURE_V5])
    fee_floor_ret = np.full(len(feat_df), 2.0 * (float(cfg.cost_per_side) + float(cfg.slip_per_side)), dtype=np.float64)
    scale_ref = np.maximum.reduce(
        [
            np.nan_to_num(scale_base, nan=0.0, posinf=0.0, neginf=0.0),
            fee_floor_ret,
            np.full(len(feat_df), 1e-6, dtype=np.float64),
        ]
    )
    fee_n = fee_floor_ret / scale_ref

    base = _strip_existing_label_columns(feat_df)
    out = base.copy()
    out["scale_ref_t"] = scale_ref.astype("float32") if cfg.cast_float32 else scale_ref
    out["fee_floor_ret_t"] = fee_floor_ret.astype("float32") if cfg.cast_float32 else fee_floor_ret
    out["fee_n_t"] = fee_n.astype("float32") if cfg.cast_float32 else fee_n

    # --- main future return targets (1/3/5/8/10) ---
    ret_by_h: Dict[int, np.ndarray] = {}
    for h in RETURN_HORIZONS_MAIN:
        ret_n, _valid_h = _future_return_n(close, open_, scale_ref, run_id, int(h))
        ret_by_h[int(h)] = ret_n
        col = f"tgt_ret_{int(h)}_n"
        out[col] = ret_n.astype("float32") if cfg.cast_float32 else ret_n

    # --- path extremes by horizon for utilities + thesis anchor path10 ---
    path_up_by_h: Dict[int, np.ndarray] = {}
    path_down_by_h: Dict[int, np.ndarray] = {}
    needed_path_horizons = sorted(set(int(h) for h in UTILITY_HORIZONS) | {int(PATH_HORIZON_MAIN)})
    for h in needed_path_horizons:
        up_h, down_h, _valid_h = _future_path_extremes_n(high, low, open_, scale_ref, run_id, int(h))
        path_up_by_h[int(h)] = up_h
        path_down_by_h[int(h)] = down_h
        if int(h) == int(PATH_HORIZON_MAIN):
            out[f"tgt_up_excur_{int(PATH_HORIZON_MAIN)}_n"] = up_h.astype("float32") if cfg.cast_float32 else up_h
            out[f"tgt_down_excur_{int(PATH_HORIZON_MAIN)}_n"] = down_h.astype("float32") if cfg.cast_float32 else down_h

    # --- utilities (1/3/5/8/10) ---
    for h in UTILITY_HORIZONS:
        h = int(h)
        cap_f = float(UTILITY_CAP_FAV_BY_HORIZON[h])
        cap_a = float(UTILITY_CAP_ADV_BY_HORIZON[h])
        up_h = path_up_by_h[h]
        down_h = path_down_by_h[h]
        long_u = _utility_from_extremes(
            favorable_n=up_h,
            adverse_n=down_h,
            fee_n=fee_n,
            cap_fav=cap_f,
            cap_adv=cap_a,
            lambda_adv=float(UTILITY_LAMBDA_ADVERSE),
        )
        short_u = _utility_from_extremes(
            favorable_n=down_h,
            adverse_n=up_h,
            fee_n=fee_n,
            cap_fav=cap_f,
            cap_adv=cap_a,
            lambda_adv=float(UTILITY_LAMBDA_ADVERSE),
        )
        out[f"tgt_long_utility_{h}"] = long_u.astype("float32") if cfg.cast_float32 else long_u
        out[f"tgt_short_utility_{h}"] = short_u.astype("float32") if cfg.cast_float32 else short_u

    # --- direction targets (1/3/5/8/10) ---
    dir_valid_by_h: Dict[int, np.ndarray] = {}
    dir_eps_by_h: Dict[int, np.ndarray] = {}
    for h in DIR_HORIZONS:
        h = int(h)
        dir_tgt, dir_valid, dir_eps = _direction_target_from_return(ret_by_h[h], fee_n, horizon=h)
        dir_valid_by_h[h] = dir_valid
        dir_eps_by_h[h] = dir_eps
        out[f"tgt_dir_{h}"] = dir_tgt.astype("float32") if cfg.cast_float32 else dir_tgt

    # --- return classes (1/3/5/8/10) ---
    for h in CLASS_HORIZONS:
        h = int(h)
        cls = _class_from_bins(ret_by_h[h], CLASS_BINS_ATR)
        out[f"tgt_retcls_{h}"] = cls

    # --- main barrier hits (1.0/1.5/2.0 on path10) ---
    up10 = path_up_by_h[int(PATH_HORIZON_MAIN)]
    down10 = path_down_by_h[int(PATH_HORIZON_MAIN)]
    for b in BARRIERS_ATR_MAIN:
        tok = f"{int(round(float(b) * 100.0)):03d}"
        out[f"tgt_up_hit_{tok}_10"] = _binary_hit_from_extreme(up10, float(b))
        out[f"tgt_down_hit_{tok}_10"] = _binary_hit_from_extreme(down10, float(b))

    # --- first-hit classes + tth (main only) ---
    tth_barriers = set(float(x) for x in TTH_BARRIERS_ATR)
    for b in FIRST_HIT_BARRIERS_ATR:
        tok = f"{int(round(float(b) * 100.0)):03d}"
        first_hit, tth_up, tth_down = _first_hit_and_tth(
            high=high,
            low=low,
            open_=open_,
            scale_ref=scale_ref,
            run_id=run_id,
            horizon=PATH_HORIZON_MAIN,
            barrier=float(b),
            invalid_value=INVALID_CLASS_VALUE,
            censored_value=TTH_CENSORED_VALUE_MAIN,
        )
        out[f"tgt_first_hit_{tok}_10"] = first_hit
        if float(b) in tth_barriers:
            out[f"tgt_tth_up_{tok}_10"] = tth_up.astype(np.int16)
            out[f"tgt_tth_down_{tok}_10"] = tth_down.astype(np.int16)

    # --- readiness / audit columns ---
    reg_cols = [c for c in REGRESSION_TARGETS_MAIN if c in out.columns]
    reg_ready = np.ones(len(out), dtype=bool)
    if reg_cols:
        reg_ready = out.loc[:, reg_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy(copy=False)

    dir_ready = np.ones(len(out), dtype=bool)
    for h in DIR_HORIZONS:
        if int(h) in dir_valid_by_h:
            dir_ready &= dir_valid_by_h[int(h)]

    discrete_cls_cols = [c for c in CLASSIFICATION_TARGETS_MAIN if c not in {f"tgt_dir_{int(h)}" for h in DIR_HORIZONS} and c in out.columns]
    discrete_cls_ready = np.ones(len(out), dtype=bool)
    if discrete_cls_cols:
        disc = out.loc[:, discrete_cls_cols].apply(pd.to_numeric, errors="coerce")
        discrete_cls_ready = (disc.notna() & (disc != int(INVALID_CLASS_VALUE))).all(axis=1).to_numpy(copy=False)

    target_ready_main = reg_ready & dir_ready & discrete_cls_ready & np.isfinite(scale_ref) & (scale_ref > 0.0)
    dataset_ready_main = feature_ready & target_ready_main

    out["target_ready_main"] = target_ready_main
    out["dataset_ready_main"] = dataset_ready_main

    known_cols = list(
        dict.fromkeys(
            [
                *base.columns,
                *LABEL_AUDIT_COLUMNS,
                *build_target_list(include_extension=False),
            ]
        )
    )
    extra_cols = [
        c
        for c in out.columns
        if c not in known_cols and not c.startswith("tgt_") and c not in DEPRECATED_LABEL_COLUMNS
    ]
    if cfg.preserve_extra_columns:
        out = out.loc[:, known_cols + extra_cols]
    else:
        out = out.loc[:, known_cols]

    return out


# -----------------------------------------------------------------------------
# Summary / meta helpers
# -----------------------------------------------------------------------------


def summarize_labeled_frame(df: pd.DataFrame) -> Dict[str, Any]:
    feat_summary = summarize_feature_frame(df)
    out: Dict[str, Any] = dict(feat_summary)
    out["label_builder_version"] = LABEL_BUILDER_VERSION
    out["target_contract_version"] = TARGET_CONTRACT_VERSION
    out["target_reference_mode"] = TARGET_REFERENCE_MODE
    out["target_scale_ref_feature_runtime"] = TARGET_SCALE_REF_FEATURE_V5
    out["target_scale_ref_feature_contract_alias"] = TARGET_SCALE_REF_FEATURE_CONTRACT_ALIAS
    out["rows"] = int(len(df))
    out["target_ready_main_rows"] = int(pd.Series(df.get("target_ready_main", False)).fillna(False).astype(bool).sum())
    out["dataset_ready_main_rows"] = int(pd.Series(df.get("dataset_ready_main", False)).fillna(False).astype(bool).sum())
    out["invalid_class_value"] = int(INVALID_CLASS_VALUE)
    out["first_hit_id_to_name"] = dict(FIRST_HIT_ID_TO_NAME)
    out["ret_class_id_to_name"] = dict(RET_CLASS_ID_TO_NAME)
    out["dir_label_to_name"] = dict(DIR_LABEL_TO_NAME)
    out["dir_target_mode"] = DIR_TARGET_MODE
    out["dir_neutral_note"] = DIR_NEUTRAL_NOTE
    out["regression_target_count"] = int(len(REGRESSION_TARGETS_MAIN))
    out["classification_target_count"] = int(len(CLASSIFICATION_TARGETS_MAIN))
    out["optional_extension_target_count"] = int(len(OPTIONAL_EXTENSION_TARGETS))

    for h in DIR_HORIZONS:
        col = f"tgt_dir_{int(h)}"
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_rates"] = {
                "up": float((s == float(DIR_POSITIVE_VALUE)).mean()),
                "down": float((s == float(DIR_NEGATIVE_VALUE)).mean()),
                "neutral": float(s.isna().mean()),
            }

    cls_col = "tgt_retcls_10"
    if cls_col in df.columns:
        s = pd.to_numeric(df[cls_col], errors="coerce")
        s = s[s >= 0]
        out["tgt_retcls_10_dist"] = {str(int(k)): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}

    fh_col = "tgt_first_hit_100_10"
    if fh_col in df.columns:
        s = pd.to_numeric(df[fh_col], errors="coerce")
        s = s[s >= 0]
        out["tgt_first_hit_100_10_dist"] = {str(int(k)): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}

    hit_col = "tgt_up_hit_100_10"
    if hit_col in df.columns:
        s = pd.to_numeric(df[hit_col], errors="coerce")
        s = s[s >= 0]
        out["tgt_up_hit_100_10_pos_rate"] = float(s.mean()) if len(s) else float("nan")
    return out



def build_meta(df: pd.DataFrame, cfg: LabelBuildConfig, *, input_path: str = "", output_path: str = "") -> Dict[str, Any]:
    summary = summarize_labeled_frame(df)
    meta: Dict[str, Any] = {
        "label_builder_version": LABEL_BUILDER_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "input_path": input_path,
        "output_path": output_path,
        "input_kind": str(cfg.input_kind),
        "include_context": False,
        "has_optional_context_block": False,
        "has_30_anchor_features": False,
        "include_extension": False,
        "cost_per_side": float(cfg.cost_per_side),
        "slip_per_side": float(cfg.slip_per_side),
        "invalid_class_value": int(INVALID_CLASS_VALUE),
        "target_reference_mode": TARGET_REFERENCE_MODE,
        "target_entry_ref_expr": TARGET_ENTRY_REF_EXPR,
        "target_timing_note": TARGET_TIMING_NOTE,
        "target_scale_ref_feature": TARGET_SCALE_REF_FEATURE_V5,
        "target_scale_ref_feature_contract_alias": TARGET_SCALE_REF_FEATURE_CONTRACT_ALIAS,
        "target_scale_ref_expr": TARGET_SCALE_REF_EXPR_V5,
        "target_fee_floor_expr": TARGET_FEE_FLOOR_EXPR,
        "target_fee_n_expr": TARGET_FEE_N_EXPR,
        "main_path_horizon": int(PATH_HORIZON_MAIN),
        "extension_path_horizon": int(PATH_HORIZON_EXTENSION),
        "return_horizons_main": [int(x) for x in RETURN_HORIZONS_MAIN],
        "dir_horizons": [int(x) for x in DIR_HORIZONS],
        "class_horizons": [int(x) for x in CLASS_HORIZONS],
        "utility_horizons": [int(x) for x in UTILITY_HORIZONS],
        "approved_return_horizons": [int(x) for x in APPROVED_RETURN_HORIZONS],
        "barriers_atr_main": [float(x) for x in BARRIERS_ATR_MAIN],
        "first_hit_barriers_atr": [float(x) for x in FIRST_HIT_BARRIERS_ATR],
        "tth_barriers_atr": [float(x) for x in TTH_BARRIERS_ATR],
        "tth_censored_value_main": int(TTH_CENSORED_VALUE_MAIN),
        "tth_censored_value_extension": int(TTH_CENSORED_VALUE_EXTENSION),
        "utility_lambda_adverse": float(UTILITY_LAMBDA_ADVERSE),
        "utility_cap_fav_by_horizon": {str(int(k)): float(v) for k, v in UTILITY_CAP_FAV_BY_HORIZON.items()},
        "utility_cap_adv_by_horizon": {str(int(k)): float(v) for k, v in UTILITY_CAP_ADV_BY_HORIZON.items()},
        "dir_target_mode": DIR_TARGET_MODE,
        "dir_positive_value": float(DIR_POSITIVE_VALUE),
        "dir_negative_value": float(DIR_NEGATIVE_VALUE),
        "dir_neutral_value": None,
        "dir_neutral_note": DIR_NEUTRAL_NOTE,
        "dir_eps_fee_mult_base": float(DIR_EPS_FEE_MULT_BASE),
        "dir_eps_mult_by_horizon": {str(int(k)): float(v) for k, v in DIR_EPS_MULT_BY_HORIZON.items()},
        "dir_label_to_name": dict(DIR_LABEL_TO_NAME),
        "hybrid_signal_expr": HYBRID_SIGNAL_EXPR,
        "hybrid_target_note": HYBRID_TARGET_NOTE,
        "target_docs": dict(TARGET_DOCS),
        "summary": summary,
    }
    return meta


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build v5.1 labels / targets from raw or feature frame")
    ap.add_argument("--input", required=True, help="input file (.parquet/.csv/.pkl), raw or feature frame")
    ap.add_argument("--output", required=True, help="output dataset file (.parquet/.csv/.pkl)")
    ap.add_argument(
        "--input-kind",
        default="auto",
        choices=["auto", "raw", "features", "dataset"],
        help="how to treat input",
    )
    ap.add_argument("--cost-per-side", type=float, default=0.00070, help="taker fee per side used for fee floor")
    ap.add_argument("--slip-per-side", type=float, default=0.00015, help="slippage per side used for fee floor")
    ap.add_argument("--cast-float32", type=int, default=1, help="1=cast regression targets / audit floats to float32")
    ap.add_argument("--preserve-extra-columns", type=int, default=1, help="1=keep non-contract extra columns")
    ap.add_argument("--feature-mask-not-ready", type=int, default=1, help="1=mask early/gap affected feature rows when raw input is built")
    ap.add_argument("--meta-json", default="", help="optional dataset meta json output path")
    ap.add_argument("--report-json", default="", help="optional summary json output path")
    return ap.parse_args()



def main() -> None:
    args = _parse_args()
    inp = read_frame(args.input)
    cfg = LabelBuildConfig(
        input_kind=str(args.input_kind),
        cost_per_side=float(args.cost_per_side),
        slip_per_side=float(args.slip_per_side),
        cast_float32=bool(int(args.cast_float32)),
        preserve_extra_columns=bool(int(args.preserve_extra_columns)),
        feature_mask_not_ready=bool(int(args.feature_mask_not_ready)),
    )
    labeled = build_labels(inp, config=cfg)
    write_frame(labeled, args.output)

    summary = summarize_labeled_frame(labeled)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.meta_json:
        meta = build_meta(
            labeled,
            cfg,
            input_path=str(args.input),
            output_path=str(args.output),
        )
        Path(args.meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
