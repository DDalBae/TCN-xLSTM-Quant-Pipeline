
# -*- coding: utf-8 -*-
"""
backtest_core_v5.py

V5 backtest core aligned with:
- backtest_contract_v5.py
- inference_v5.py
- target_contract_v5.py

핵심 방향
---------
- entry: multi-horizon `dir + hyb + utility + retcls` composite
- thesis / exit: `path10 / utility_10 / first_hit_10 / tth_10`
- runtime richness: dynamic / regime / tp_window / entry_episode / rearm / same-side-hold
- rich path + fast(array) path 를 동일한 코어 로직 위에 둔다.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator

from backtest_contract_v5 import (
    BACKTEST_CONTRACT_VERSION,
    BacktestConfigV5,
    DynamicConfigV5,
    EntryEpisodeConfigV5,
    EntryFillModeV5,
    ExecStageV5,
    ExecutionConfigV5,
    ExitFillModeV5,
    ExitReasonV5,
    FallbackPathV5,
    IntrabarModeV5,
    ObjectiveConfigV5,
    PolicyConfigV5,
    ProgressProtectConfigV5,
    RegimeDetectConfigV5,
    RegimeFilterConfigV5,
    RegimeLaneConfigV5,
    RegimeThresholdConfigV5,
    RegimeWeightConfigV5,
    SameSideHoldConfigV5,
    TPWindowConfigV5,
    ThesisStateV5,
    TouchRuleV5,
    normalize_horizon_weights,
)
from feature_ops_v5 import read_frame, write_frame
from target_contract_v5 import (
    BARRIERS_ATR_MAIN,
    FIRST_HIT_BARRIERS_ATR,
    PATH_HORIZON_MAIN,
    RETURN_HORIZONS_MAIN,
    TARGET_SCALE_REF_FEATURE,
    TTH_BARRIERS_ATR,
)

EPS = 1.0e-12
JSON_NAN = float("nan")

REQUIRED_RAW_SIM_COLUMNS: Tuple[str, ...] = ("timestamp", "open", "high", "low", "close")
OPTIONAL_RUNTIME_FEATURE_COLUMNS: Tuple[str, ...] = (
    "feature_ready",
    "atr10_rel",
    "vol_z_10",
    "rolling_vwap_dist_10",
    "bb_pctb_20",
    "efficiency_ratio_10",
)

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if not np.isfinite(out):
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return bool(int(v) != 0)
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _stable_json(data: Any) -> str:
    def _default(x: Any) -> Any:
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (pd.Timestamp, pd.Timedelta)):
            return str(x)
        return str(x)

    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_default)


def _barrier_token(barrier: float) -> str:
    return f"{int(round(float(barrier) * 100.0)):03d}"


def _bps_factor(bps: float) -> float:
    return float(bps) * 1.0e-4


def _log_ret(side: int, entry: float, exit_price: float) -> float:
    if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0.0 or exit_price <= 0.0:
        return float("nan")
    raw = math.log(float(exit_price) / float(entry))
    return float(raw if int(side) > 0 else -raw)


def _long_price_from_n(entry: float, n_val: float, scale_ref: float) -> float:
    return float(entry) * float(math.exp(float(n_val) * float(scale_ref)))


def _short_price_from_n(entry: float, n_val: float, scale_ref: float) -> float:
    return float(entry) * float(math.exp(-float(n_val) * float(scale_ref)))


def _favorable_n(side: int, entry: float, high: float, low: float, scale_ref: float) -> float:
    if side > 0:
        if not np.isfinite(high) or high <= 0.0:
            return float("nan")
        return float(math.log(high / entry) / max(scale_ref, EPS))
    if not np.isfinite(low) or low <= 0.0:
        return float("nan")
    return float(-math.log(low / entry) / max(scale_ref, EPS))


def _adverse_n(side: int, entry: float, high: float, low: float, scale_ref: float) -> float:
    if side > 0:
        if not np.isfinite(low) or low <= 0.0:
            return float("nan")
        return float(-math.log(low / entry) / max(scale_ref, EPS))
    if not np.isfinite(high) or high <= 0.0:
        return float("nan")
    return float(math.log(high / entry) / max(scale_ref, EPS))


def _maker_price_from_open(open_px: float, side: int, offset_bps: float, *, is_entry: bool) -> float:
    bps = _bps_factor(offset_bps)
    if is_entry:
        mult = (1.0 - bps) if int(side) > 0 else (1.0 + bps)
    else:
        mult = (1.0 + bps) if int(side) > 0 else (1.0 - bps)
    return float(open_px) * float(mult)


def _ioc_price_from_open(open_px: float, side: int, offset_bps: float, *, is_entry: bool) -> float:
    bps = _bps_factor(offset_bps)
    if is_entry:
        mult = (1.0 + bps) if int(side) > 0 else (1.0 - bps)
    else:
        mult = (1.0 - bps) if int(side) > 0 else (1.0 + bps)
    return float(open_px) * float(mult)


def _touch_hit(
    side: int,
    limit_price: float,
    high: float,
    low: float,
    *,
    rule: str,
    offset_bps: float,
    frac_range: float,
    is_entry: bool,
) -> bool:
    if not np.isfinite(limit_price) or limit_price <= 0.0:
        return False
    rule = str(rule).strip().lower()
    if int(side) > 0:
        touch_value = float(low) if is_entry else float(high)
        ref_sign = -1.0 if is_entry else 1.0
        compare_le = is_entry
    else:
        touch_value = float(high) if is_entry else float(low)
        ref_sign = 1.0 if is_entry else -1.0
        compare_le = not is_entry
    if not np.isfinite(touch_value):
        return False

    if rule == TouchRuleV5.TOUCH.value:
        return bool(touch_value <= limit_price) if compare_le else bool(touch_value >= limit_price)

    if rule == TouchRuleV5.PENETRATE_OFFSET.value:
        thr = float(limit_price) + ref_sign * float(limit_price) * _bps_factor(offset_bps)
        return bool(touch_value <= thr) if compare_le else bool(touch_value >= thr)

    if rule == TouchRuleV5.PENETRATE_FRAC_RANGE.value:
        rng = max(float(high) - float(low), 0.0)
        thr = float(limit_price) + ref_sign * float(frac_range) * rng
        return bool(touch_value <= thr) if compare_le else bool(touch_value >= thr)

    return False


def _same_bar_resolution(mode: str, *, tp_hit: bool, stop_hits: bool) -> str:
    mode = str(mode).strip().lower()
    if tp_hit and stop_hits:
        if mode in {IntrabarModeV5.FAVORABLE_FIRST.value}:
            return "favorable_first"
        return "adverse_first"
    return "single"


def _to_float_array(values: Any, *, fill: float = np.nan) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.asarray([float(arr)], dtype=np.float64)
    if not np.isfinite(fill):
        return arr
    out = arr.copy()
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = float(fill)
    return out


def _rolling_quantile_shifted(values: np.ndarray, q: float, window: int, min_ready: int) -> np.ndarray:
    ser = pd.Series(np.asarray(values, dtype=np.float64), copy=False)
    return ser.shift(1).rolling(window=max(1, int(window)), min_periods=max(1, int(min_ready))).quantile(float(q)).to_numpy(dtype=np.float64)


@njit(cache=True)
def _cum_equity_numba(pnl: np.ndarray) -> np.ndarray:
    out = np.empty_like(pnl)
    s = 0.0
    for i in range(pnl.shape[0]):
        s += float(pnl[i])
        out[i] = s
    return out


@njit(cache=True)
def _max_drawdown_numba(equity: np.ndarray) -> float:
    peak = 0.0
    worst = 0.0
    for i in range(equity.shape[0]):
        v = float(equity[i])
        if v > peak:
            peak = v
        dd = peak - v
        if dd > worst:
            worst = dd
    return float(worst)


@njit(cache=True)
def _ewm_hysteresis_numba(raw_alpha: np.ndarray, alpha_ema: float, alpha_hysteresis: float) -> Tuple[np.ndarray, np.ndarray]:
    n = int(raw_alpha.shape[0])
    alpha = np.zeros(n, dtype=np.float64)
    bucket = np.zeros(n, dtype=np.int8)
    if n <= 0:
        return alpha, bucket
    prev = float(raw_alpha[0]) if np.isfinite(raw_alpha[0]) else 0.0
    if prev < 0.0:
        prev = 0.0
    elif prev > 1.0:
        prev = 1.0
    alpha[0] = prev
    dead = float(max(alpha_hysteresis, 0.0))
    k = float(alpha_ema)
    if k < 0.0:
        k = 0.0
    elif k > 1.0:
        k = 1.0

    for i in range(1, n):
        x = float(raw_alpha[i]) if np.isfinite(raw_alpha[i]) else prev
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        ema = x if k <= 0.0 else (prev + k * (x - prev))
        if abs(ema - prev) < dead:
            alpha[i] = prev
        else:
            alpha[i] = ema
        prev = alpha[i]

    calm_cut = 0.5 - dead
    if calm_cut < 0.0:
        calm_cut = 0.0
    active_cut = 0.5 + dead
    if active_cut > 1.0:
        active_cut = 1.0

    for i in range(n):
        a = float(alpha[i])
        if a <= calm_cut:
            bucket[i] = 0
        elif a >= active_cut:
            bucket[i] = 2
        else:
            bucket[i] = 1
    return alpha, bucket


# -----------------------------------------------------------------------------
# Prediction frame helpers
# -----------------------------------------------------------------------------

def _required_pred_columns_v5() -> Tuple[str, ...]:
    cols: List[str] = ["pred_ready", "pred_scale_ref_t"]
    for h in RETURN_HORIZONS_MAIN:
        cols.extend(
            [
                f"pred_tgt_ret_{h}_n",
                f"predscore_tgt_retcls_{h}",
                f"pred_tgt_long_utility_{h}",
                f"pred_tgt_short_utility_{h}",
                f"predprob_tgt_dir_{h}",
                f"predscore_tgt_dir_{h}",
            ]
        )
    cols.extend(
        [
            f"pred_tgt_up_excur_{PATH_HORIZON_MAIN}_n",
            f"pred_tgt_down_excur_{PATH_HORIZON_MAIN}_n",
        ]
    )
    for b in BARRIERS_ATR_MAIN:
        tok = _barrier_token(b)
        cols.extend(
            [
                f"predprob_tgt_up_hit_{tok}_{PATH_HORIZON_MAIN}",
                f"predprob_tgt_down_hit_{tok}_{PATH_HORIZON_MAIN}",
            ]
        )
    for b in FIRST_HIT_BARRIERS_ATR:
        tok = _barrier_token(b)
        cols.extend(
            [
                f"predprob_up_tgt_first_hit_{tok}_{PATH_HORIZON_MAIN}",
                f"predprob_down_tgt_first_hit_{tok}_{PATH_HORIZON_MAIN}",
            ]
        )
    for b in TTH_BARRIERS_ATR:
        tok = _barrier_token(b)
        cols.extend(
            [
                f"predexp_tgt_tth_up_{tok}_{PATH_HORIZON_MAIN}",
                f"predexp_tgt_tth_down_{tok}_{PATH_HORIZON_MAIN}",
                f"predprob_censored_tgt_tth_up_{tok}_{PATH_HORIZON_MAIN}",
                f"predprob_censored_tgt_tth_down_{tok}_{PATH_HORIZON_MAIN}",
            ]
        )
    return tuple(cols)


def ensure_prediction_frame_v5(pred: pd.DataFrame) -> pd.DataFrame:
    df = pred.copy()
    missing_raw = [c for c in REQUIRED_RAW_SIM_COLUMNS if c not in df.columns]
    if missing_raw:
        raise ValueError(f"prediction frame missing required raw columns: {missing_raw}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    missing_pred = [c for c in _required_pred_columns_v5() if c not in df.columns]
    if missing_pred:
        raise ValueError(f"prediction frame missing required v5 prediction columns: {missing_pred}")

    if "pred_scale_ref_t" not in df.columns:
        df["pred_scale_ref_t"] = pd.to_numeric(df[f"pred_{TARGET_SCALE_REF_FEATURE}"], errors="coerce")
    df["pred_ready"] = pd.Series(df["pred_ready"]).fillna(False).astype(bool)
    if "feature_ready" in df.columns:
        df["feature_ready"] = pd.Series(df["feature_ready"]).fillna(False).astype(bool)
    else:
        df["feature_ready"] = df["pred_ready"].astype(bool)

    for c in REQUIRED_RAW_SIM_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def prepare_fast_eval_cache_v5(pred_eval: pd.DataFrame) -> Dict[str, Any]:
    df = ensure_prediction_frame_v5(pred_eval)
    out: Dict[str, Any] = {}
    out["timestamp_ns"] = df["timestamp"].astype("int64", copy=False).to_numpy(copy=False)
    out["open"] = _to_float_array(df["open"])
    out["high"] = _to_float_array(df["high"])
    out["low"] = _to_float_array(df["low"])
    out["close"] = _to_float_array(df["close"])
    out["pred_ready"] = pd.Series(df["pred_ready"]).fillna(False).astype(bool).to_numpy(copy=False)
    out["feature_ready"] = pd.Series(df.get("feature_ready", df["pred_ready"])).fillna(False).astype(bool).to_numpy(copy=False)
    out["scale_ref_t"] = _to_float_array(df["pred_scale_ref_t"])
    out["atr10_rel"] = _to_float_array(df.get("atr10_rel"))
    out["vol_z_10"] = _to_float_array(df.get("vol_z_10"))
    out["rolling_vwap_dist_10"] = _to_float_array(df.get("rolling_vwap_dist_10"))
    out["bb_pctb_20"] = _to_float_array(df.get("bb_pctb_20"))
    out["efficiency_ratio_10"] = _to_float_array(df.get("efficiency_ratio_10"))
    out["bar_range_rel"] = np.where(
        (np.isfinite(out["close"]) & (out["close"] > 0.0)),
        np.maximum(0.0, out["high"] - out["low"]) / np.maximum(out["close"], EPS),
        np.nan,
    )
    for h in RETURN_HORIZONS_MAIN:
        out[f"ret_h{h}"] = _to_float_array(df[f"pred_tgt_ret_{h}_n"])
        out[f"retcls_score_h{h}"] = _to_float_array(df[f"predscore_tgt_retcls_{h}"])
        out[f"util_long_h{h}"] = _to_float_array(df[f"pred_tgt_long_utility_{h}"])
        out[f"util_short_h{h}"] = _to_float_array(df[f"pred_tgt_short_utility_{h}"])
        out[f"dir_prob_h{h}"] = _to_float_array(df[f"predprob_tgt_dir_{h}"])
        out[f"dir_score_h{h}"] = _to_float_array(df[f"predscore_tgt_dir_{h}"])
    out["path_up_n"] = _to_float_array(df[f"pred_tgt_up_excur_{PATH_HORIZON_MAIN}_n"])
    out["path_down_n"] = _to_float_array(df[f"pred_tgt_down_excur_{PATH_HORIZON_MAIN}_n"])
    for b in BARRIERS_ATR_MAIN:
        tok = _barrier_token(b)
        out[f"up_hit_{tok}"] = _to_float_array(df[f"predprob_tgt_up_hit_{tok}_{PATH_HORIZON_MAIN}"])
        out[f"down_hit_{tok}"] = _to_float_array(df[f"predprob_tgt_down_hit_{tok}_{PATH_HORIZON_MAIN}"])
    for b in FIRST_HIT_BARRIERS_ATR:
        tok = _barrier_token(b)
        out[f"first_up_{tok}"] = _to_float_array(df[f"predprob_up_tgt_first_hit_{tok}_{PATH_HORIZON_MAIN}"])
        out[f"first_down_{tok}"] = _to_float_array(df[f"predprob_down_tgt_first_hit_{tok}_{PATH_HORIZON_MAIN}"])
    for b in TTH_BARRIERS_ATR:
        tok = _barrier_token(b)
        out[f"tth_up_{tok}_exp"] = _to_float_array(df[f"predexp_tgt_tth_up_{tok}_{PATH_HORIZON_MAIN}"])
        out[f"tth_down_{tok}_exp"] = _to_float_array(df[f"predexp_tgt_tth_down_{tok}_{PATH_HORIZON_MAIN}"])
        out[f"tth_up_{tok}_cens"] = _to_float_array(df[f"predprob_censored_tgt_tth_up_{tok}_{PATH_HORIZON_MAIN}"])
        out[f"tth_down_{tok}_cens"] = _to_float_array(df[f"predprob_censored_tgt_tth_down_{tok}_{PATH_HORIZON_MAIN}"])
    out["n_rows"] = int(len(df))
    return out


def save_fast_eval_cache_v5(cache: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {str(k): v for k, v in cache.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(p, **arrays, meta=np.array([json.dumps({"keys": list(cache.keys())})], dtype=object))


def load_fast_eval_cache_v5(path: str | Path) -> Dict[str, Any]:
    z = np.load(Path(path), allow_pickle=True)
    out: Dict[str, Any] = {str(k): z[k] for k in z.files if k != "meta"}
    out["n_rows"] = int(len(out.get("open", [])))
    return out


# -----------------------------------------------------------------------------
# Signal / regime / dynamic helpers
# -----------------------------------------------------------------------------

def _normalize_stress_component(values: np.ndarray, *, q50: float, q90: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    denom = max(float(q90) - float(q50), 1.0e-9)
    return np.clip((arr - float(q50)) / denom, 0.0, 1.0)


def _prepare_regime_dynamic_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    dynamic: DynamicConfigV5,
    regime_detect: RegimeDetectConfigV5,
) -> Dict[str, np.ndarray]:
    atr = _to_float_array(arrays.get("atr10_rel"))
    rng = _to_float_array(arrays.get("bar_range_rel"))
    vol = np.abs(_to_float_array(arrays.get("vol_z_10")))
    stretch = np.abs(_to_float_array(arrays.get("rolling_vwap_dist_10")))
    band = np.abs(_to_float_array(arrays.get("bb_pctb_20")) - 0.5) * 2.0

    def _q(v: np.ndarray, p: float, default: float) -> float:
        good = v[np.isfinite(v)]
        if good.size == 0:
            return float(default)
        return float(np.quantile(good, p))

    w_src = dynamic if bool(dynamic.enabled) else regime_detect
    atr50, atr90 = _q(atr, 0.50, 0.0), _q(atr, 0.90, 1.0)
    rng50, rng90 = _q(rng, 0.50, 0.0), _q(rng, 0.90, 1.0)
    vol50, vol90 = _q(vol, 0.50, 0.0), _q(vol, 0.90, 1.0)
    stretch50, stretch90 = _q(stretch, 0.50, 0.0), _q(stretch, 0.90, 1.0)
    band50, band90 = _q(band, 0.50, 0.0), _q(band, 0.90, 1.0)

    c_atr = _normalize_stress_component(atr, q50=atr50, q90=atr90)
    c_rng = _normalize_stress_component(rng, q50=rng50, q90=rng90)
    c_vol = _normalize_stress_component(vol, q50=vol50, q90=vol90)
    c_stretch = _normalize_stress_component(stretch, q50=stretch50, q90=stretch90)
    c_band = _normalize_stress_component(band, q50=band50, q90=band90)

    ws = float(w_src.w_atr + w_src.w_rng + w_src.w_vol + w_src.w_stretch + w_src.w_band)
    if ws <= 0.0:
        ws = 1.0
    stress_raw = (
        float(w_src.w_atr) * c_atr
        + float(w_src.w_rng) * c_rng
        + float(w_src.w_vol) * c_vol
        + float(w_src.w_stretch) * c_stretch
        + float(w_src.w_band) * c_band
    ) / ws

    lo = float(getattr(w_src, "stress_lo", 0.25))
    hi = float(getattr(w_src, "stress_hi", 0.65))
    denom = max(hi - lo, 1.0e-6)
    raw_alpha = np.clip((stress_raw - lo) / denom, 0.0, 1.0).astype(np.float64, copy=False)
    alpha, bucket = _ewm_hysteresis_numba(raw_alpha, float(getattr(w_src, "alpha_ema", 0.15)), float(getattr(w_src, "alpha_hysteresis", 0.03)))
    return {
        "stress_raw": stress_raw.astype(np.float64, copy=False),
        "regime_alpha": alpha.astype(np.float64, copy=False),
        "regime_bucket": bucket.astype(np.int8, copy=False),
        "component_atr": c_atr.astype(np.float64, copy=False),
        "component_rng": c_rng.astype(np.float64, copy=False),
        "component_vol": c_vol.astype(np.float64, copy=False),
        "component_stretch": c_stretch.astype(np.float64, copy=False),
        "component_band": c_band.astype(np.float64, copy=False),
    }


def _bucket_name_from_code(code: int) -> str:
    if int(code) <= 0:
        return "calm"
    if int(code) >= 2:
        return "active"
    return "mid"


def _blend_weight_matrix(
    *,
    base: Mapping[str, float],
    calm_anchor: Mapping[str, float],
    active_anchor: Mapping[str, float],
    calm_mix: float,
    active_mix: float,
    alpha: np.ndarray,
    bucket: np.ndarray,
) -> Dict[int, np.ndarray]:
    hz = list(RETURN_HORIZONS_MAIN)
    raw_cols: List[np.ndarray] = []
    for h in hz:
        k = f"w{int(h)}"
        base_w = float(base.get(k, 0.0))
        calm_w = float(calm_anchor.get(k, 0.0))
        active_w = float(active_anchor.get(k, 0.0))
        anchor = np.where(bucket <= 0, calm_w, np.where(bucket >= 2, active_w, (1.0 - alpha) * calm_w + alpha * active_w))
        mix = np.where(bucket <= 0, float(calm_mix), np.where(bucket >= 2, float(active_mix), (1.0 - alpha) * float(calm_mix) + alpha * float(active_mix)))
        raw_cols.append((1.0 - mix) * base_w + mix * anchor)
    mat = np.stack(raw_cols, axis=1).astype(np.float64, copy=False)
    mat = np.maximum(mat, 0.0)
    row_sum = np.maximum(mat.sum(axis=1, keepdims=True), EPS)
    mat = mat / row_sum
    return {int(h): mat[:, j] for j, h in enumerate(hz)}


def _build_signal_bundle_v5(
    arrays: Mapping[str, np.ndarray],
    *,
    policy: PolicyConfigV5,
    dynamic: DynamicConfigV5,
    regime_detect: RegimeDetectConfigV5,
    regime_weight: RegimeWeightConfigV5,
    regime_threshold: RegimeThresholdConfigV5,
    regime_filter: RegimeFilterConfigV5,
    regime_lane: RegimeLaneConfigV5,
    backtest: BacktestConfigV5,
) -> Dict[str, np.ndarray]:
    n = int(len(arrays["open"]))
    regime = _prepare_regime_dynamic_arrays(arrays, dynamic=dynamic, regime_detect=regime_detect)
    alpha = regime["regime_alpha"]
    bucket = regime["regime_bucket"]
    stress = regime["stress_raw"]

    if bool(regime_weight.enabled):
        gate_w = _blend_weight_matrix(
            base=policy.gate_weights,
            calm_anchor=regime_weight.gate_calm_anchor,
            active_anchor=regime_weight.gate_active_anchor,
            calm_mix=float(regime_weight.gate_calm_mix),
            active_mix=float(regime_weight.gate_active_mix),
            alpha=alpha,
            bucket=bucket,
        )
        dir_w = _blend_weight_matrix(
            base=policy.dir_weights,
            calm_anchor=regime_weight.dir_calm_anchor,
            active_anchor=regime_weight.dir_active_anchor,
            calm_mix=float(regime_weight.dir_calm_mix),
            active_mix=float(regime_weight.dir_active_mix),
            alpha=alpha,
            bucket=bucket,
        )
    else:
        gate_w = {int(h): np.full(n, float(policy.gate_weights.get(f"w{h}", 0.0)), dtype=np.float64) for h in RETURN_HORIZONS_MAIN}
        dir_w = {int(h): np.full(n, float(policy.dir_weights.get(f"w{h}", 0.0)), dtype=np.float64) for h in RETURN_HORIZONS_MAIN}

    util_long_mix = np.zeros(n, dtype=np.float64)
    util_short_mix = np.zeros(n, dtype=np.float64)
    hyb_mix = np.zeros(n, dtype=np.float64)
    cls_mix = np.zeros(n, dtype=np.float64)
    dirprob_mix = np.zeros(n, dtype=np.float64)
    dir_mix = np.zeros(n, dtype=np.float64)
    long_core = np.zeros(n, dtype=np.float64)
    short_core = np.zeros(n, dtype=np.float64)

    dir_signal_by_h: Dict[int, np.ndarray] = {}
    for h in RETURN_HORIZONS_MAIN:
        ret_h = _to_float_array(arrays[f"ret_h{h}"])
        cls_h = _to_float_array(arrays[f"retcls_score_h{h}"])
        dirprob_h = _to_float_array(arrays[f"dir_score_h{h}"])
        util_long_h = _to_float_array(arrays[f"util_long_h{h}"])
        util_short_h = _to_float_array(arrays[f"util_short_h{h}"])

        gw = gate_w[int(h)]
        dw = dir_w[int(h)]

        util_long_mix += gw * util_long_h
        util_short_mix += gw * util_short_h
        hyb_mix += dw * ret_h
        cls_mix += dw * cls_h
        dirprob_mix += dw * dirprob_h

        dir_signal_h = float(policy.hyb_weight) * ret_h + float(policy.cls_weight) * cls_h + float(policy.dirprob_weight) * dirprob_h
        dir_signal_by_h[int(h)] = dir_signal_h.astype(np.float64, copy=False)
        dir_mix += dw * dir_signal_h

    long_core = float(policy.util_weight) * np.maximum(util_long_mix, 0.0) + np.maximum(dir_mix, 0.0)
    short_core = float(policy.util_weight) * np.maximum(util_short_mix, 0.0) + np.maximum(-dir_mix, 0.0)
    side = np.where(long_core >= short_core, 1, -1).astype(np.int8)
    entry_core = np.where(side > 0, long_core, short_core)
    entry_gap = np.where(side > 0, long_core - short_core, short_core - long_core)
    util_10_side = np.where(side > 0, arrays["util_long_h10"], arrays["util_short_h10"]).astype(np.float64)
    util_10_gap = np.where(side > 0, arrays["util_long_h10"] - arrays["util_short_h10"], arrays["util_short_h10"] - arrays["util_long_h10"]).astype(np.float64)

    side_agreement = np.zeros(n, dtype=np.float64)
    for h in RETURN_HORIZONS_MAIN:
        w = dir_w[int(h)]
        sig = dir_signal_by_h[int(h)]
        side_agreement += w * np.where(((sig >= 0.0) & (side > 0)) | ((sig <= 0.0) & (side < 0)), 1.0, 0.0)

    # dynamic arrays
    dyn_gate_mult = np.ones(n, dtype=np.float64)
    dyn_lev_scale = np.ones(n, dtype=np.float64)
    dyn_bep_scale = np.ones(n, dtype=np.float64)
    dyn_trail_scale = np.ones(n, dtype=np.float64)
    dyn_sl_scale = np.ones(n, dtype=np.float64)
    dyn_softsl_relax = np.zeros(n, dtype=np.int64)

    if bool(dynamic.enabled):
        a = alpha
        if bool(dynamic.use_dyn_gate):
            dyn_gate_mult = float(dynamic.gate_mult_min) + (float(dynamic.gate_mult_max) - float(dynamic.gate_mult_min)) * a
        if bool(dynamic.use_dyn_lev):
            dyn_lev_scale = float(dynamic.lev_scale_max) + (float(dynamic.lev_scale_min) - float(dynamic.lev_scale_max)) * a
        if bool(dynamic.use_dyn_bep):
            dyn_bep_scale = float(dynamic.bep_scale_max) + (float(dynamic.bep_scale_min) - float(dynamic.bep_scale_max)) * a
        if bool(dynamic.use_dyn_trail):
            dyn_trail_scale = float(dynamic.trail_scale_max) + (float(dynamic.trail_scale_min) - float(dynamic.trail_scale_max)) * a
        if bool(dynamic.use_dyn_sl):
            dyn_sl_scale = float(dynamic.sl_scale_max) + (float(dynamic.sl_scale_min) - float(dynamic.sl_scale_max)) * a
        if bool(dynamic.use_dyn_soft_sl):
            dyn_softsl_relax = np.where(a >= 0.66, int(dynamic.softsl_relax_hi), np.where(a >= 0.33, int(dynamic.softsl_relax_mid), 0)).astype(np.int64)

    # filters
    filter_pass = np.ones(n, dtype=bool)
    atr = _to_float_array(arrays["atr10_rel"])
    rng = _to_float_array(arrays["bar_range_rel"])
    vol_z = _to_float_array(arrays["vol_z_10"])
    atr_med = float(np.nanmedian(atr)) if np.isfinite(np.nanmedian(atr)) else 1.0
    rng_med = float(np.nanmedian(rng)) if np.isfinite(np.nanmedian(rng)) else 1.0

    def _bucket_float(calm: float, mid: float, active: float) -> np.ndarray:
        return np.where(bucket <= 0, float(calm), np.where(bucket >= 2, float(active), float(mid))).astype(np.float64)

    if bool(regime_filter.enabled):
        vol_floor_arr = _bucket_float(regime_filter.vol_low_th_calm, regime_filter.vol_low_th_mid, regime_filter.vol_low_th_active)
        atr_mult_arr = _bucket_float(regime_filter.atr_entry_mult_calm, 0.5 * (regime_filter.atr_entry_mult_calm + regime_filter.atr_entry_mult_active), regime_filter.atr_entry_mult_active)
        rng_mult_arr = _bucket_float(regime_filter.range_entry_mult_calm, 0.5 * (regime_filter.range_entry_mult_calm + regime_filter.range_entry_mult_active), regime_filter.range_entry_mult_active)

        if bool(regime_filter.use_vol_split):
            filter_pass &= (vol_floor_arr <= -1.0e8) | (vol_z >= vol_floor_arr)
        if bool(regime_filter.use_entry_mult_split):
            filter_pass &= (atr_mult_arr <= 0.0) | (atr <= atr_med * atr_mult_arr)
            filter_pass &= (rng_mult_arr <= 0.0) | (rng <= rng_med * rng_mult_arr)

    # threshold arrays
    q_base = np.full(n, float(policy.entry_q), dtype=np.float64)
    floor_base = np.full(n, float(policy.entry_th_floor), dtype=np.float64)
    if bool(regime_threshold.enabled):
        q_base = np.where(bucket <= 0, float(regime_threshold.q_entry_calm), np.where(bucket >= 2, float(regime_threshold.q_entry_active), float(regime_threshold.q_entry_mid)))
        floor_base = np.where(bucket <= 0, float(regime_threshold.entry_th_calm), np.where(bucket >= 2, float(regime_threshold.entry_th_active), float(regime_threshold.entry_th_mid)))

    # sparse lane
    active_sparse_flag = np.zeros(n, dtype=bool)
    sparse_filter_pass = np.ones(n, dtype=bool)
    if bool(regime_lane.enabled) and bool(regime_lane.active_sparse_enabled):
        lookback = max(8, int(backtest.threshold_lookback_bars))
        atr_q = _rolling_quantile_shifted(atr, float(regime_lane.sparse_atr_q), lookback, int(regime_lane.active_sparse_min_ready))
        rng_q = _rolling_quantile_shifted(rng, float(regime_lane.sparse_range_q), lookback, int(regime_lane.active_sparse_min_ready))
        vol_q = _rolling_quantile_shifted(vol_z, float(regime_lane.sparse_vol_q), lookback, int(regime_lane.active_sparse_min_ready))
        sparse_filter_pass &= np.where(np.isfinite(atr_q), atr >= atr_q, False)
        sparse_filter_pass &= np.where(np.isfinite(rng_q), rng >= rng_q, False)
        if bool(regime_lane.sparse_require_high_vol):
            sparse_filter_pass &= np.where(np.isfinite(vol_q), vol_z >= vol_q, False)
        active_sparse_flag = (bucket >= 2) & sparse_filter_pass
        q_base = np.where(active_sparse_flag, np.maximum(q_base, float(regime_lane.sparse_gate_q)), q_base)
        floor_roll_sparse = _rolling_quantile_shifted(entry_core, float(regime_lane.sparse_gate_floor_q), lookback, int(regime_lane.active_sparse_min_ready))
    else:
        floor_roll_sparse = np.full(n, np.nan, dtype=np.float64)

    lookback = max(8, int(backtest.threshold_lookback_bars))
    min_ready = max(1, int(backtest.threshold_min_ready))
    q_unique = sorted(set([float(x) for x in np.unique(q_base[np.isfinite(q_base)])])) or [float(policy.entry_q)]
    quantile_cache: Dict[float, np.ndarray] = {
        float(q): _rolling_quantile_shifted(entry_core, float(q), lookback, min_ready) for q in q_unique
    }
    quant_base = np.full(n, np.nan, dtype=np.float64)
    for qv, arr in quantile_cache.items():
        mask = np.isclose(q_base, qv)
        if np.any(mask):
            quant_base[mask] = arr[mask]
    gate_threshold = np.maximum(np.where(np.isfinite(quant_base), quant_base, floor_base), floor_base)
    gate_threshold = np.maximum(gate_threshold, np.where(np.isfinite(floor_roll_sparse), floor_roll_sparse, -np.inf))

    # gate pass diagnostics
    chosen_gate_score = np.where(side > 0, util_long_mix, util_short_mix)
    hyb_abs = np.abs(hyb_mix)
    cls_abs = np.abs(cls_mix)
    dir_abs = np.abs(dir_mix)

    gate_pass = (
        arrays["pred_ready"].astype(bool)
        & arrays["feature_ready"].astype(bool)
        & filter_pass
        & np.isfinite(entry_core)
        & (entry_core >= gate_threshold)
        & (entry_core >= float(policy.entry_min_score))
        & (entry_gap >= float(policy.entry_min_gap))
        & (np.maximum(hyb_abs, 0.0) >= float(policy.entry_min_hyb_abs))
        & (np.maximum(cls_abs, 0.0) >= float(policy.entry_min_cls_abs))
        & (side_agreement >= float(policy.min_side_agreement_frac))
        & (util_10_side >= float(policy.entry_min_utility_10))
        & (util_10_gap >= float(policy.entry_min_utility_gap_10))
        & (chosen_gate_score >= float(policy.gate_score_floor))
        & (dir_abs >= float(policy.dir_score_gate_floor))
    )

    return {
        "stress_raw": stress.astype(np.float64, copy=False),
        "regime_alpha": alpha.astype(np.float64, copy=False),
        "regime_bucket": bucket.astype(np.int8, copy=False),
        "gate_weights_eff": np.stack([gate_w[int(h)] for h in RETURN_HORIZONS_MAIN], axis=1).astype(np.float64, copy=False),
        "dir_weights_eff": np.stack([dir_w[int(h)] for h in RETURN_HORIZONS_MAIN], axis=1).astype(np.float64, copy=False),
        "entry_long_core": long_core.astype(np.float64, copy=False),
        "entry_short_core": short_core.astype(np.float64, copy=False),
        "entry_side": side.astype(np.int8, copy=False),
        "entry_core": entry_core.astype(np.float64, copy=False),
        "entry_gap": entry_gap.astype(np.float64, copy=False),
        "hyb_mix": hyb_mix.astype(np.float64, copy=False),
        "cls_mix": cls_mix.astype(np.float64, copy=False),
        "dirprob_mix": dirprob_mix.astype(np.float64, copy=False),
        "dir_mix": dir_mix.astype(np.float64, copy=False),
        "util_long_mix": util_long_mix.astype(np.float64, copy=False),
        "util_short_mix": util_short_mix.astype(np.float64, copy=False),
        "utility_10_side": util_10_side.astype(np.float64, copy=False),
        "utility_10_gap": util_10_gap.astype(np.float64, copy=False),
        "side_agreement_frac": side_agreement.astype(np.float64, copy=False),
        "dyn_gate_mult": dyn_gate_mult.astype(np.float64, copy=False),
        "dyn_lev_scale": dyn_lev_scale.astype(np.float64, copy=False),
        "dyn_bep_scale": dyn_bep_scale.astype(np.float64, copy=False),
        "dyn_trail_scale": dyn_trail_scale.astype(np.float64, copy=False),
        "dyn_sl_scale": dyn_sl_scale.astype(np.float64, copy=False),
        "dyn_softsl_relax": dyn_softsl_relax.astype(np.int64, copy=False),
        "filter_pass": filter_pass.astype(bool, copy=False),
        "active_sparse_flag": active_sparse_flag.astype(bool, copy=False),
        "gate_threshold": gate_threshold.astype(np.float64, copy=False),
        "q_entry_used": q_base.astype(np.float64, copy=False),
        "entry_floor_used": floor_base.astype(np.float64, copy=False),
        "gate_pass": gate_pass.astype(bool, copy=False),
        "chosen_gate_score": chosen_gate_score.astype(np.float64, copy=False),
        "hyb_abs": hyb_abs.astype(np.float64, copy=False),
        "cls_abs": cls_abs.astype(np.float64, copy=False),
        "dir_abs": dir_abs.astype(np.float64, copy=False),
    }


def prepare_trial_context_v5(
    arrays: Mapping[str, np.ndarray],
    *,
    policy: PolicyConfigV5,
    dynamic: Optional[DynamicConfigV5] = None,
    progress_protect: Optional[ProgressProtectConfigV5] = None,
    tp_window: Optional[TPWindowConfigV5] = None,
    entry_episode: Optional[EntryEpisodeConfigV5] = None,
    same_side_hold: Optional[SameSideHoldConfigV5] = None,
    regime_detect: Optional[RegimeDetectConfigV5] = None,
    regime_weight: Optional[RegimeWeightConfigV5] = None,
    regime_threshold: Optional[RegimeThresholdConfigV5] = None,
    regime_filter: Optional[RegimeFilterConfigV5] = None,
    regime_lane: Optional[RegimeLaneConfigV5] = None,
    execution: Optional[ExecutionConfigV5] = None,
    backtest: Optional[BacktestConfigV5] = None,
) -> Dict[str, Any]:
    dynamic = dynamic or DynamicConfigV5()
    progress_protect = progress_protect or ProgressProtectConfigV5()
    tp_window = tp_window or TPWindowConfigV5()
    entry_episode = entry_episode or EntryEpisodeConfigV5()
    same_side_hold = same_side_hold or SameSideHoldConfigV5()
    regime_detect = regime_detect or RegimeDetectConfigV5()
    regime_weight = regime_weight or RegimeWeightConfigV5()
    regime_threshold = regime_threshold or RegimeThresholdConfigV5()
    regime_filter = regime_filter or RegimeFilterConfigV5()
    regime_lane = regime_lane or RegimeLaneConfigV5()
    execution = execution or ExecutionConfigV5()
    backtest = backtest or BacktestConfigV5()

    bundle = _build_signal_bundle_v5(
        arrays,
        policy=policy,
        dynamic=dynamic,
        regime_detect=regime_detect,
        regime_weight=regime_weight,
        regime_threshold=regime_threshold,
        regime_filter=regime_filter,
        regime_lane=regime_lane,
        backtest=backtest,
    )
    ctx: Dict[str, Any] = {
        "arrays": dict(arrays),
        "bundle": bundle,
        "policy": policy,
        "dynamic": dynamic,
        "progress_protect": progress_protect,
        "tp_window": tp_window,
        "entry_episode": entry_episode,
        "same_side_hold": same_side_hold,
        "regime_detect": regime_detect,
        "regime_weight": regime_weight,
        "regime_threshold": regime_threshold,
        "regime_filter": regime_filter,
        "regime_lane": regime_lane,
        "execution": execution,
        "backtest": backtest,
    }
    return ctx


def prepare_single_segment_fast_inputs_from_context_v5(
    context: Mapping[str, Any],
    *,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Dict[str, Any]:
    arrays = context["arrays"]
    bundle = context["bundle"]
    n = int(len(arrays["open"]))
    lo = max(0, int(start_idx))
    hi = n if end_idx is None else min(n, int(end_idx))
    if hi <= lo:
        hi = lo
    sub_arrays: Dict[str, Any] = {}
    for k, v in arrays.items():
        sub_arrays[k] = v[lo:hi].copy() if isinstance(v, np.ndarray) else v
    sub_bundle: Dict[str, Any] = {}
    for k, v in bundle.items():
        sub_bundle[k] = v[lo:hi].copy() if isinstance(v, np.ndarray) else v
    out = dict(context)
    out["arrays"] = sub_arrays
    out["bundle"] = sub_bundle
    out["segment_range"] = (int(lo), int(hi))
    return out


# -----------------------------------------------------------------------------
# Trade-plan / thesis helpers
# -----------------------------------------------------------------------------

def _candidate_util10_side(bundle: Mapping[str, np.ndarray], side: int, i: int) -> float:
    return float(bundle["utility_10_side"][i]) if int(side) != 0 else float("nan")


def _candidate_util10_gap(bundle: Mapping[str, np.ndarray], side: int, i: int) -> float:
    return float(bundle["utility_10_gap"][i]) if int(side) != 0 else float("nan")


def _main_confirm_prob(arrays: Mapping[str, np.ndarray], side: int, barrier: float, i: int) -> float:
    tok = _barrier_token(barrier)
    key = f"up_hit_{tok}" if int(side) > 0 else f"down_hit_{tok}"
    return _safe_float(arrays.get(key, np.array([np.nan]))[i], JSON_NAN)


def _timing_first_hit_prob(arrays: Mapping[str, np.ndarray], side: int, barrier: float, i: int) -> float:
    tok = _barrier_token(barrier)
    key = f"first_up_{tok}" if int(side) > 0 else f"first_down_{tok}"
    return _safe_float(arrays.get(key, np.array([np.nan]))[i], JSON_NAN)


def _timing_expected_bars(arrays: Mapping[str, np.ndarray], side: int, barrier: float, i: int) -> float:
    tok = _barrier_token(barrier)
    key = f"tth_up_{tok}_exp" if int(side) > 0 else f"tth_down_{tok}_exp"
    return _safe_float(arrays.get(key, np.array([np.nan]))[i], JSON_NAN)


def _timing_censored_prob(arrays: Mapping[str, np.ndarray], side: int, barrier: float, i: int) -> float:
    tok = _barrier_token(barrier)
    key = f"tth_up_{tok}_cens" if int(side) > 0 else f"tth_down_{tok}_cens"
    return _safe_float(arrays.get(key, np.array([np.nan]))[i], JSON_NAN)


def _retcls_alignment_state(bundle: Mapping[str, np.ndarray], side: int, i: int) -> str:
    score = float(bundle["cls_mix"][i])
    if not np.isfinite(score):
        return "invalid"
    if side > 0:
        return "aligned" if score >= 0.0 else "opposed"
    return "aligned" if score <= 0.0 else "opposed"


def _classify_thesis_v5(
    arrays: Mapping[str, np.ndarray],
    bundle: Mapping[str, np.ndarray],
    position: Mapping[str, Any],
    policy: PolicyConfigV5,
    i: int,
    *,
    shock_flag: bool = False,
) -> Dict[str, Any]:
    side = int(position["side"])
    same_core = float(bundle["entry_long_core"][i]) if side > 0 else float(bundle["entry_short_core"][i])
    opp_core = float(bundle["entry_short_core"][i]) if side > 0 else float(bundle["entry_long_core"][i])
    dir_mix_signed = float(bundle["dir_mix"][i]) * float(side)
    util10_same = float(bundle["utility_10_side"][i])
    util10_gap = float(bundle["utility_10_gap"][i])
    support_ratio = same_core / max(float(position.get("entry_core", 0.0)), EPS)
    same_hyb = float(bundle["hyb_mix"][i]) * float(side)
    progress_frac = float(position.get("peak_fav_n", 0.0)) / max(float(position.get("tp_n", 1.0)), EPS)

    if shock_flag:
        state = ThesisStateV5.LIQUIDITY_SHOCK.value
    elif dir_mix_signed <= -float(policy.thesis_strong_flip_margin) and opp_core >= same_core + float(policy.thesis_strong_flip_margin):
        state = ThesisStateV5.STRONG_OPPOSITE.value
    elif dir_mix_signed <= -float(policy.thesis_weak_flip_margin) or opp_core >= same_core + float(policy.thesis_weak_flip_margin):
        state = ThesisStateV5.WEAK_OPPOSITE.value
    elif abs(dir_mix_signed) <= float(policy.thesis_weak_flip_margin) and abs(util10_gap) <= float(policy.thesis_weak_flip_margin):
        state = ThesisStateV5.NEUTRAL_DRIFT.value
    elif dir_mix_signed >= float(policy.thesis_strong_flip_margin) and same_core >= opp_core + float(policy.thesis_strong_flip_margin):
        state = ThesisStateV5.STRONG_SAME.value
    else:
        state = ThesisStateV5.WEAK_SAME.value

    return {
        "state": state,
        "same_core": same_core,
        "opp_core": opp_core,
        "dir_mix_signed": dir_mix_signed,
        "utility_10_same": util10_same,
        "utility_10_gap": util10_gap,
        "support_strength_ratio": support_ratio,
        "same_hyb_signed": same_hyb,
        "progress_frac": progress_frac,
    }


def _build_candidate_plan_v5(
    arrays: Mapping[str, np.ndarray],
    bundle: Mapping[str, np.ndarray],
    policy: PolicyConfigV5,
    i: int,
) -> Dict[str, Any]:
    side = int(bundle["entry_side"][i])
    if side == 0:
        return {"side": 0}
    scale_ref_t = max(_safe_float(arrays["scale_ref_t"][i], 0.0), 1.0e-6)
    fav_pred = max(_safe_float(arrays["path_up_n"][i], 0.0), 0.0)
    adv_pred = max(_safe_float(arrays["path_down_n"][i], 0.0), 0.0)

    tp_n = max(float(policy.TP), fav_pred * float(policy.fee_tp_mult))
    sl_n = max(float(policy.SL), adv_pred)
    bep_arm_n = max(float(policy.BEP_ARM), float(policy.bep_arm_fee_mult) * (2.0 * 0.00085 / max(scale_ref_t, EPS)))
    trail_n = float(policy.trailing)

    util10_side = _candidate_util10_side(bundle, side, i)
    util10_gap = _candidate_util10_gap(bundle, side, i)
    main_confirm = _main_confirm_prob(arrays, side, float(policy.confirm_main_barrier), i)
    timing_first = _timing_first_hit_prob(arrays, side, float(policy.timing_barrier), i)
    timing_exp = _timing_expected_bars(arrays, side, float(policy.timing_barrier), i)
    timing_cens = _timing_censored_prob(arrays, side, float(policy.timing_barrier), i)
    retcls_align_state = _retcls_alignment_state(bundle, side, i)
    retcls_align_score = float(bundle["cls_mix"][i]) * float(side)

    gate_pass = bool(bundle["gate_pass"][i])
    stage = ""
    detail = ""
    fail_value = JSON_NAN
    fail_threshold = JSON_NAN
    if not bool(arrays["pred_ready"][i]):
        stage, detail, fail_value, fail_threshold = "pred_ready", "pred not ready", 0.0, 1.0
        gate_pass = False
    elif not bool(arrays["feature_ready"][i]):
        stage, detail, fail_value, fail_threshold = "feature_ready", "feature not ready", 0.0, 1.0
        gate_pass = False
    elif not bool(bundle["filter_pass"][i]):
        stage, detail, fail_value, fail_threshold = "regime_filter", "regime filter blocked", 0.0, 1.0
        gate_pass = False
    elif float(bundle["entry_core"][i]) < float(bundle["gate_threshold"][i]):
        stage, detail, fail_value, fail_threshold = "entry_quantile", "entry core below threshold", float(bundle["entry_core"][i]), float(bundle["gate_threshold"][i])
        gate_pass = False
    elif float(bundle["entry_core"][i]) < float(policy.entry_min_score):
        stage, detail, fail_value, fail_threshold = "entry_min_score", "entry core below minimum score", float(bundle["entry_core"][i]), float(policy.entry_min_score)
        gate_pass = False
    elif float(bundle["entry_gap"][i]) < float(policy.entry_min_gap):
        stage, detail, fail_value, fail_threshold = "entry_gap", "entry gap below minimum", float(bundle["entry_gap"][i]), float(policy.entry_min_gap)
        gate_pass = False
    elif abs(float(bundle["hyb_mix"][i])) < float(policy.entry_min_hyb_abs):
        stage, detail, fail_value, fail_threshold = "entry_hyb_abs", "hyb abs below minimum", abs(float(bundle["hyb_mix"][i])), float(policy.entry_min_hyb_abs)
        gate_pass = False
    elif abs(float(bundle["cls_mix"][i])) < float(policy.entry_min_cls_abs):
        stage, detail, fail_value, fail_threshold = "entry_cls_abs", "retcls abs below minimum", abs(float(bundle["cls_mix"][i])), float(policy.entry_min_cls_abs)
        gate_pass = False
    elif float(bundle["side_agreement_frac"][i]) < float(policy.min_side_agreement_frac):
        stage, detail, fail_value, fail_threshold = "side_agreement", "side agreement below minimum", float(bundle["side_agreement_frac"][i]), float(policy.min_side_agreement_frac)
        gate_pass = False
    elif util10_side < float(policy.entry_min_utility_10):
        stage, detail, fail_value, fail_threshold = "utility10_side", "utility_10 side below minimum", util10_side, float(policy.entry_min_utility_10)
        gate_pass = False
    elif util10_gap < float(policy.entry_min_utility_gap_10):
        stage, detail, fail_value, fail_threshold = "utility10_gap", "utility_10 gap below minimum", util10_gap, float(policy.entry_min_utility_gap_10)
        gate_pass = False
    elif float(bundle["chosen_gate_score"][i]) < float(policy.gate_score_floor):
        stage, detail, fail_value, fail_threshold = "gate_score_floor", "chosen gate score below floor", float(bundle["chosen_gate_score"][i]), float(policy.gate_score_floor)
        gate_pass = False
    elif abs(float(bundle["dir_mix"][i])) < float(policy.dir_score_gate_floor):
        stage, detail, fail_value, fail_threshold = "dir_score_floor", "dir score below floor", abs(float(bundle["dir_mix"][i])), float(policy.dir_score_gate_floor)
        gate_pass = False
    elif main_confirm < float(policy.confirm_main_prob):
        stage, detail, fail_value, fail_threshold = "main_confirm_prob", "main confirm prob below threshold", main_confirm, float(policy.confirm_main_prob)
        gate_pass = False
    elif timing_first < float(policy.timing_first_hit_prob):
        stage, detail, fail_value, fail_threshold = "timing_first_hit_prob", "timing first-hit prob below threshold", timing_first, float(policy.timing_first_hit_prob)
        gate_pass = False
    elif timing_exp > float(policy.timing_max_expected_bars):
        stage, detail, fail_value, fail_threshold = "timing_expected_bars", "timing expected bars too high", timing_exp, float(policy.timing_max_expected_bars)
        gate_pass = False
    elif timing_cens > float(policy.timing_max_censored_prob):
        stage, detail, fail_value, fail_threshold = "timing_censored_prob", "timing censored prob too high", timing_cens, float(policy.timing_max_censored_prob)
        gate_pass = False
    elif bool(policy.require_retcls_alignment) and (retcls_align_state != "aligned" or retcls_align_score < float(policy.min_retcls_align_score)):
        stage, detail, fail_value, fail_threshold = "retcls_alignment", retcls_align_state, retcls_align_score, float(policy.min_retcls_align_score)
        gate_pass = False

    entry_price_ref = float(arrays["open"][min(i + 1, len(arrays["open"]) - 1)]) if (i + 1) < len(arrays["open"]) else float(arrays["close"][i])

    plan = {
        "side": int(side),
        "tp_n": float(tp_n),
        "sl_n": float(sl_n),
        "bep_arm_n": float(bep_arm_n),
        "trail_n": float(trail_n),
        "tp_ret": float(tp_n * scale_ref_t),
        "sl_ret": float(sl_n * scale_ref_t),
        "scale_ref_t": float(scale_ref_t),
        "utility_10": float(util10_side),
        "utility_gap": float(util10_gap),
        "entry_price_ref": float(entry_price_ref),
        "entry_core": float(bundle["entry_core"][i]),
        "entry_gap": float(bundle["entry_gap"][i]),
        "hyb_mix": float(bundle["hyb_mix"][i]),
        "cls_mix": float(bundle["cls_mix"][i]),
        "dirprob_mix": float(bundle["dirprob_mix"][i]),
        "dir_mix": float(bundle["dir_mix"][i]),
        "util_long_mix": float(bundle["util_long_mix"][i]),
        "util_short_mix": float(bundle["util_short_mix"][i]),
        "side_agreement_frac": float(bundle["side_agreement_frac"][i]),
        "main_confirm_prob": float(main_confirm),
        "timing_first_hit_prob": float(timing_first),
        "timing_expected_bars": float(timing_exp),
        "timing_censored_prob": float(timing_cens),
        "retcls_align_state": str(retcls_align_state),
        "retcls_align_score": float(retcls_align_score),
        "gate_threshold": float(bundle["gate_threshold"][i]),
        "q_entry_used": float(bundle["q_entry_used"][i]),
        "entry_floor_used": float(bundle["entry_floor_used"][i]),
        "active_sparse_flag": bool(bundle["active_sparse_flag"][i]),
    }
    return {
        "plan": plan if gate_pass else None,
        "policy_gate_passed": bool(gate_pass),
        "gate_fail_stage": str(stage),
        "gate_fail_detail": str(detail),
        "gate_fail_value": float(fail_value),
        "gate_fail_threshold": float(fail_threshold),
        "side": int(side),
        "entry_core": float(bundle["entry_core"][i]),
        "entry_gap": float(bundle["entry_gap"][i]),
        "entry_long_core": float(bundle["entry_long_core"][i]),
        "entry_short_core": float(bundle["entry_short_core"][i]),
        "hyb_mix": float(bundle["hyb_mix"][i]),
        "cls_mix": float(bundle["cls_mix"][i]),
        "dirprob_mix": float(bundle["dirprob_mix"][i]),
        "dir_mix": float(bundle["dir_mix"][i]),
        "util_long_mix": float(bundle["util_long_mix"][i]),
        "util_short_mix": float(bundle["util_short_mix"][i]),
        "side_agreement_frac": float(bundle["side_agreement_frac"][i]),
        "utility_10_side": float(util10_side),
        "utility_10_gap": float(util10_gap),
        "main_confirm_prob": float(main_confirm),
        "timing_first_hit_prob": float(timing_first),
        "timing_expected_bars": float(timing_exp),
        "timing_censored_prob": float(timing_cens),
        "retcls_alignment_state": str(retcls_align_state),
        "retcls_alignment_score": float(retcls_align_score),
        "gate_threshold": float(bundle["gate_threshold"][i]),
        "q_entry_used": float(bundle["q_entry_used"][i]),
        "entry_floor_used": float(bundle["entry_floor_used"][i]),
        "active_sparse_flag": bool(bundle["active_sparse_flag"][i]),
    }


def _set_pending_exit(position: MutableMapping[str, Any], *, reason: str, fallback: str, activate_idx: int, thesis_state: str = "", note: str = "") -> None:
    position["pending_exit_reason"] = str(reason)
    position["pending_exit_fallback"] = str(fallback)
    position["pending_exit_activate_idx"] = int(activate_idx)
    position["pending_exit_note"] = str(note)
    position["pending_exit_thesis_state"] = str(thesis_state)
    position["exec_stage"] = ExecStageV5.EXIT_TRIGGERED.value


def _close_trade_from_position_v5(
    pos: Mapping[str, Any],
    *,
    exit_idx: int,
    exit_ts: pd.Timestamp,
    exit_price_fill: float,
    exit_price_planned: float,
    exit_fill_mode: str,
    exit_fee_kind: str,
    exit_maker_flag: bool,
    exit_fallback_path: str,
    exit_reason: str,
    same_bar_resolution: str,
    same_bar_both_hit: bool,
) -> Dict[str, Any]:
    side = int(pos["side"])
    lev = float(pos["lev"])
    entry_fill = float(pos["entry_price_fill"])
    scale_ref_t = float(pos["scale_ref_t"])
    gross_pnl_lev = float(_log_ret(side, entry_fill, float(exit_price_fill)) * lev)
    maker_fee_side = float(pos["maker_fee_side"])
    taker_fee_side = float(pos["taker_fee_side"])
    fee_entry_lev = float(maker_fee_side * lev) if bool(pos.get("entry_maker_flag", False)) else float(taker_fee_side * lev)
    fee_exit_lev = float(maker_fee_side * lev) if bool(exit_maker_flag) else float(taker_fee_side * lev)
    fee_total_lev = float(fee_entry_lev + fee_exit_lev)
    net_pnl_lev = float(gross_pnl_lev - fee_total_lev)
    row = {
        "trade_id": str(pos["trade_id"]),
        "run_id": int(pos.get("run_id", 0)),
        "is_rearm_entry": int(bool(pos.get("is_rearm_entry", False))),
        "decision_idx": int(pos["decision_idx"]),
        "decision_ts": str(pos["decision_ts"]),
        "entry_signal_idx": int(pos["entry_signal_idx"]),
        "entry_idx": int(pos["entry_idx"]),
        "exit_idx": int(exit_idx),
        "entry_ts": str(pos["entry_ts"]),
        "exit_ts": str(exit_ts.isoformat()),
        "entry_price_planned": float(pos.get("entry_price_planned", entry_fill)),
        "entry_price_fill": float(entry_fill),
        "exit_price_planned": float(exit_price_planned),
        "exit_price_fill": float(exit_price_fill),
        "entry_fill_mode": str(pos.get("entry_fill_mode", "")),
        "exit_fill_mode": str(exit_fill_mode),
        "entry_fee_kind": str(pos.get("entry_fee_kind", "")),
        "exit_fee_kind": str(exit_fee_kind),
        "side": int(side),
        "entry_core": float(pos.get("entry_core", JSON_NAN)),
        "entry_gap": float(pos.get("entry_gap", JSON_NAN)),
        "hyb_mix": float(pos.get("hyb_mix", JSON_NAN)),
        "cls_mix": float(pos.get("cls_mix", JSON_NAN)),
        "dirprob_mix": float(pos.get("dirprob_mix", JSON_NAN)),
        "dir_mix": float(pos.get("dir_mix", JSON_NAN)),
        "util_long_mix": float(pos.get("util_long_mix", JSON_NAN)),
        "util_short_mix": float(pos.get("util_short_mix", JSON_NAN)),
        "utility_10": float(pos.get("utility_10", JSON_NAN)),
        "utility_gap": float(pos.get("utility_gap", JSON_NAN)),
        "tp_n": float(pos.get("tp_n", JSON_NAN)),
        "sl_n": float(pos.get("sl_n", JSON_NAN)),
        "trail_n": float(pos.get("trail_n", JSON_NAN)),
        "bep_arm_n": float(pos.get("bep_arm_n", JSON_NAN)),
        "scale_ref_t": float(scale_ref_t),
        "hold_bars": int(pos.get("bars_completed", 0)),
        "gross_pnl_lev": float(gross_pnl_lev),
        "gross_pnl": float(gross_pnl_lev),
        "fee_entry_lev": float(fee_entry_lev),
        "fee_exit_lev": float(fee_exit_lev),
        "fee_total_lev": float(fee_total_lev),
        "net_pnl_lev": float(net_pnl_lev),
        "net_pnl": float(net_pnl_lev),
        "mfe_n": float(pos.get("peak_fav_n", 0.0)),
        "mae_n": float(pos.get("peak_adv_n", 0.0)),
        "mfe": float(pos.get("peak_fav_n", 0.0) * scale_ref_t * lev),
        "mae": float(pos.get("peak_adv_n", 0.0) * scale_ref_t * lev),
        "tp_touched": bool(pos.get("tp_touched", False)),
        "sl_touched": bool(pos.get("sl_touched", False)),
        "trail_touched": bool(pos.get("trail_touched", False)),
        "entry_unfilled": False,
        "skip_reason": "",
        "leverage": float(lev),
        "lev": float(lev),
        "entry_maker_flag": bool(pos.get("entry_maker_flag", False)),
        "exit_maker_flag": bool(exit_maker_flag),
        "entry_fallback_path": str(pos.get("entry_fallback_path", FallbackPathV5.NONE.value)),
        "exit_fallback_path": str(exit_fallback_path),
        "pre_bep_timeout_hit": int(bool(pos.get("pre_bep_timeout_hit", False))),
        "thesis_fail_soft": int(bool(pos.get("thesis_fail_soft", False))),
        "thesis_fail_hard": int(bool(pos.get("thesis_fail_hard", False))),
        "profit_floor_hit": int(bool(pos.get("profit_floor_hit", False))),
        "gap_flatten_hit": int(bool(pos.get("gap_flatten_hit", False))),
        "shock_exit_hit": int(bool(pos.get("shock_exit_hit", False))),
        "thesis_state_last": str(pos.get("thesis_state_last", ThesisStateV5.NONE.value)),
        "same_bar_resolution": str(same_bar_resolution),
        "same_bar_both_hit": bool(same_bar_both_hit),
        "exit_reason": str(exit_reason),
        "regime_bucket_entry": int(pos.get("regime_bucket_entry", 1)),
        "regime_alpha_entry": float(pos.get("regime_alpha_entry", 0.0)),
        "active_sparse_entry": int(bool(pos.get("active_sparse_entry", False))),
        "dyn_stress_entry": float(pos.get("dyn_stress_entry", 0.0)),
        "dyn_gate_mult_entry": float(pos.get("dyn_gate_mult_entry", 1.0)),
        "dyn_lev_scale_entry": float(pos.get("dyn_lev_scale_entry", 1.0)),
        "dyn_bep_scale_entry": float(pos.get("dyn_bep_scale_entry", 1.0)),
        "dyn_trail_scale_entry": float(pos.get("dyn_trail_scale_entry", 1.0)),
        "dyn_sl_scale_entry": float(pos.get("dyn_sl_scale_entry", 1.0)),
        "same_side_bonus_bars": int(pos.get("same_side_bonus_bars", 0)),
        "support_strength_ratio_exit": float(pos.get("support_strength_ratio_last", 0.0)),
        "entry_events_json": str(pos.get("entry_events_json", "[]")),
        "exit_events_json": str(pos.get("exit_events_json", "[]")),
    }
    return row


# -----------------------------------------------------------------------------
# Core simulation loop
# -----------------------------------------------------------------------------

def _simulate_core_arrays_v5(
    context: Mapping[str, Any],
    *,
    collect_decisions: bool,
    collect_trades: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    arrays = context["arrays"]
    bundle = context["bundle"]
    policy: PolicyConfigV5 = context["policy"]
    dynamic: DynamicConfigV5 = context["dynamic"]
    progress_protect: ProgressProtectConfigV5 = context["progress_protect"]
    tp_window: TPWindowConfigV5 = context["tp_window"]
    entry_episode: EntryEpisodeConfigV5 = context["entry_episode"]
    same_side_hold: SameSideHoldConfigV5 = context["same_side_hold"]
    execution: ExecutionConfigV5 = context["execution"]
    backtest: BacktestConfigV5 = context["backtest"]

    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    ts_ns = arrays["timestamp_ns"]
    n = int(len(open_))

    decisions: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    sim_meta: Dict[str, Any] = {
        "n_rows": n,
        "n_pred_ready": int(np.sum(arrays["pred_ready"])),
        "n_gate_passed": int(np.sum(bundle["gate_pass"])),
        "n_entry_signals": 0,
        "n_entry_filled": 0,
        "n_entry_unfilled": 0,
        "n_gap_skip": 0,
        "n_cooldown_skip": 0,
        "n_pending_soft_exit": 0,
        "n_pending_hard_exit": 0,
        "n_rearm_entry_signals": 0,
        "n_rearm_entry_filled": 0,
        "maker_entry_count": 0,
        "maker_exit_count": 0,
        "ioc_exit_count": 0,
        "market_exit_count": 0,
        "shock_count": 0,
        "regime_bucket_counts": {"calm": int(np.sum(bundle["regime_bucket"] <= 0)), "mid": int(np.sum(bundle["regime_bucket"] == 1)), "active": int(np.sum(bundle["regime_bucket"] >= 2))},
        "thesis_state_counts": {s.value: 0 for s in ThesisStateV5},
        "exit_reason_counts": {r.value: 0 for r in ExitReasonV5},
        "decision_actions": {},
        "entry_core_mean": float(np.nanmean(bundle["entry_core"])),
        "entry_core_q90": float(np.nanquantile(bundle["entry_core"], 0.90)) if n else float("nan"),
    }

    cooldown_remaining = 0
    pending_entry: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, Any]] = None
    trade_id_seq = 1
    run_id = 0
    run_side = 0
    run_entries = 0
    last_entry_signal_idx = -10**9
    last_exit_idx = -10**9
    last_exit_reason = ""
    last_exit_side = 0
    last_exit_price = float("nan")
    last_exit_entry_core = 0.0

    def _count_action(action: str) -> None:
        sim_meta["decision_actions"][str(action)] = int(sim_meta["decision_actions"].get(str(action), 0)) + 1

    for i in range(n):
        ts = pd.Timestamp(int(ts_ns[i]), tz="UTC")
        prev_close = float(close[i - 1]) if i > 0 else float(close[i])
        scale_ref_cur = max(_safe_float(arrays["scale_ref_t"][i], 0.0), 1.0e-6)
        gap_n = abs(math.log(max(float(open_[i]), EPS) / max(float(prev_close), EPS))) / scale_ref_cur if i > 0 else 0.0
        range_n = abs(math.log(max(float(high[i]), EPS) / max(float(low[i]), EPS))) / scale_ref_cur if np.isfinite(high[i]) and np.isfinite(low[i]) and high[i] > 0.0 and low[i] > 0.0 else 0.0
        shock_flag = bool(np.isfinite(range_n) and range_n >= float(policy.shock_bar_range_n))
        gap_flag = bool(np.isfinite(gap_n) and gap_n >= float(policy.gap_skip_n))

        decision_diag = _build_candidate_plan_v5(arrays, bundle, policy, i)
        policy_gate_passed = bool(decision_diag.get("policy_gate_passed", False))

        action = "NO_ACTION"
        exec_stage = ExecStageV5.IDLE.value if position is None else str(position.get("exec_stage", ExecStageV5.IDLE.value))
        open_trade_id = str(position.get("trade_id")) if position is not None else ""
        entry_block_reason = ""

        # ------------------------------------------------------------------
        # 1) execute pending entry at current bar
        # ------------------------------------------------------------------
        if pending_entry is not None and position is None and i >= int(pending_entry["activate_idx"]):
            entry_side = int(pending_entry["side"])
            entry_limit_price = _maker_price_from_open(float(open_[i]), entry_side, float(execution.entry_postonly_offset_bps), is_entry=True)
            entry_filled = False
            entry_price_fill = float("nan")
            entry_fee_kind = ""
            entry_maker_flag = False
            entry_fallback_path = FallbackPathV5.NONE.value

            maker_hit = _touch_hit(
                entry_side,
                entry_limit_price,
                float(high[i]),
                float(low[i]),
                rule=str(execution.entry_touch_rule),
                offset_bps=float(execution.entry_touch_offset_bps),
                frac_range=float(execution.entry_touch_frac_range),
                is_entry=True,
            )
            if str(execution.entry_fill_mode) == EntryFillModeV5.TAKER_NEXT_OPEN.value:
                entry_filled = True
                entry_price_fill = float(open_[i])
                entry_fee_kind = "taker"
                entry_maker_flag = False
                entry_fallback_path = FallbackPathV5.TAKER_NEXT_OPEN.value
            elif maker_hit:
                entry_filled = True
                entry_price_fill = float(entry_limit_price)
                entry_fee_kind = "maker"
                entry_maker_flag = True
                entry_fallback_path = FallbackPathV5.POST_ONLY.value
            elif i >= int(pending_entry["expire_idx"]):
                if str(execution.entry_fill_mode) == EntryFillModeV5.MAKER_REST_THEN_IOC.value and bool(execution.entry_ioc_enabled):
                    entry_price_fill = _ioc_price_from_open(float(open_[i]), entry_side, float(execution.entry_ioc_offset_bps), is_entry=True)
                    impact_bps = abs(_log_ret(1, float(open_[i]), float(entry_price_fill))) * 1.0e4
                    if impact_bps <= float(execution.entry_ioc_impact_cap_bps):
                        entry_filled = True
                        entry_fee_kind = "taker"
                        entry_maker_flag = False
                        entry_fallback_path = FallbackPathV5.IOC_LIMIT.value
                else:
                    sim_meta["n_entry_unfilled"] += 1
                    action = "ENTRY_UNFILLED"
                    exec_stage = ExecStageV5.ENTRY_SKIPPED.value
                    entry_block_reason = "entry_not_filled"
                    pending_entry = None

            if entry_filled:
                trade_id = f"T{trade_id_seq:09d}"
                trade_id_seq += 1
                plan = dict(pending_entry["plan"])
                tp_n = float(plan["tp_n"]) * float(bundle["dyn_gate_mult"][pending_entry["decision_idx"]] * 0 + 1.0)  # keep explicit
                sl_n = float(plan["sl_n"]) * float(bundle["dyn_sl_scale"][pending_entry["decision_idx"]])
                bep_arm_n = float(plan["bep_arm_n"]) * float(bundle["dyn_bep_scale"][pending_entry["decision_idx"]])
                trail_n = float(plan["trail_n"]) * float(bundle["dyn_trail_scale"][pending_entry["decision_idx"]])
                lev = float(execution.leverage) * float(bundle["dyn_lev_scale"][pending_entry["decision_idx"]])
                if bool(execution.integer_leverage):
                    lev = float(max(1, int(round(lev))))

                position = {
                    "trade_id": trade_id,
                    "run_id": int(pending_entry.get("run_id", 0)),
                    "is_rearm_entry": bool(pending_entry.get("is_rearm_entry", False)),
                    "decision_idx": int(pending_entry["decision_idx"]),
                    "decision_ts": str(pending_entry["decision_ts"]),
                    "entry_signal_idx": int(pending_entry["decision_idx"]),
                    "entry_idx": int(i),
                    "entry_ts": str(ts.isoformat()),
                    "entry_price_planned": float(pending_entry.get("entry_price_planned", open_[i])),
                    "entry_price_fill": float(entry_price_fill),
                    "entry_fill_mode": str(execution.entry_fill_mode),
                    "entry_fee_kind": str(entry_fee_kind),
                    "entry_maker_flag": bool(entry_maker_flag),
                    "entry_fallback_path": str(entry_fallback_path),
                    "entry_limit_price": float(entry_limit_price),
                    "side": int(entry_side),
                    "utility_10": float(plan["utility_10"]),
                    "utility_gap": float(plan["utility_gap"]),
                    "entry_core": float(plan["entry_core"]),
                    "entry_gap": float(plan["entry_gap"]),
                    "hyb_mix": float(plan["hyb_mix"]),
                    "cls_mix": float(plan["cls_mix"]),
                    "dirprob_mix": float(plan["dirprob_mix"]),
                    "dir_mix": float(plan["dir_mix"]),
                    "util_long_mix": float(plan["util_long_mix"]),
                    "util_short_mix": float(plan["util_short_mix"]),
                    "side_agreement_frac": float(plan["side_agreement_frac"]),
                    "main_confirm_prob": float(plan["main_confirm_prob"]),
                    "timing_first_hit_prob": float(plan["timing_first_hit_prob"]),
                    "timing_expected_bars": float(plan["timing_expected_bars"]),
                    "timing_censored_prob": float(plan["timing_censored_prob"]),
                    "retcls_align_state": str(plan["retcls_align_state"]),
                    "retcls_align_score": float(plan["retcls_align_score"]),
                    "tp_n": float(tp_n),
                    "sl_n": float(sl_n),
                    "bep_arm_n": float(bep_arm_n),
                    "trail_n": float(trail_n),
                    "min_hold_tp_bars": int(policy.min_hold_tp_bars),
                    "min_hold_trail_bars": int(policy.min_hold_trail_bars),
                    "min_hold_soft_sl_bars": int(policy.min_hold_soft_sl_bars),
                    "max_hold_bars": int(policy.max_hold_bars),
                    "hard_max_hold_bars": int(policy.hard_max_hold_bars),
                    "scale_ref_t": float(plan["scale_ref_t"]),
                    "lev": float(lev),
                    "taker_fee_side": float(execution.cost_per_side + execution.slip_per_side),
                    "maker_fee_side": float(execution.maker_fee_per_side),
                    "peak_fav_n": 0.0,
                    "peak_adv_n": 0.0,
                    "bars_completed": 0,
                    "bep_armed": False,
                    "bep_armed_idx": -1,
                    "trail_stop_n": float("nan"),
                    "tp_touched": False,
                    "sl_touched": False,
                    "trail_touched": False,
                    "thesis_state_last": ThesisStateV5.NONE.value,
                    "thesis_state_candidate": ThesisStateV5.NONE.value,
                    "thesis_state_count": 0,
                    "exec_stage": ExecStageV5.PRE_BEP.value,
                    "regime_bucket_entry": int(bundle["regime_bucket"][pending_entry["decision_idx"]]),
                    "regime_alpha_entry": float(bundle["regime_alpha"][pending_entry["decision_idx"]]),
                    "active_sparse_entry": bool(bundle["active_sparse_flag"][pending_entry["decision_idx"]]),
                    "dyn_stress_entry": float(bundle["stress_raw"][pending_entry["decision_idx"]]),
                    "dyn_gate_mult_entry": float(bundle["dyn_gate_mult"][pending_entry["decision_idx"]]),
                    "dyn_lev_scale_entry": float(bundle["dyn_lev_scale"][pending_entry["decision_idx"]]),
                    "dyn_bep_scale_entry": float(bundle["dyn_bep_scale"][pending_entry["decision_idx"]]),
                    "dyn_trail_scale_entry": float(bundle["dyn_trail_scale"][pending_entry["decision_idx"]]),
                    "dyn_sl_scale_entry": float(bundle["dyn_sl_scale"][pending_entry["decision_idx"]]),
                    "same_side_bonus_bars": 0,
                    "support_strength_ratio_last": 0.0,
                    "entry_events_json": _stable_json([{"ts": str(ts.isoformat()), "kind": "ENTRY_FILLED", "path": entry_fallback_path}]),
                    "exit_events_json": "[]",
                    "pre_bep_timeout_hit": False,
                    "thesis_fail_soft": False,
                    "thesis_fail_hard": False,
                    "profit_floor_hit": False,
                    "gap_flatten_hit": False,
                    "shock_exit_hit": False,
                }
                sim_meta["n_entry_filled"] += 1
                if bool(entry_maker_flag):
                    sim_meta["maker_entry_count"] += 1
                if bool(pending_entry.get("is_rearm_entry", False)):
                    sim_meta["n_rearm_entry_filled"] += 1
                action = "ENTRY_FILLED"
                exec_stage = ExecStageV5.ENTRY_FILLED.value
                open_trade_id = trade_id
                pending_entry = None

        # ------------------------------------------------------------------
        # 2) execute pending exit on current bar open
        # ------------------------------------------------------------------
        if position is not None and position.get("pending_exit_reason") and i >= int(position.get("pending_exit_activate_idx", n + 1)):
            reason = str(position.get("pending_exit_reason"))
            fallback = str(position.get("pending_exit_fallback", FallbackPathV5.IOC_LIMIT.value))
            exit_price_fill = float(open_[i])
            exit_price_planned = float(open_[i])
            exit_fill_mode = ExitFillModeV5.IOC_LIMIT.value
            exit_fee_kind = "taker"
            exit_maker_flag = False
            if fallback == FallbackPathV5.POST_ONLY.value:
                px = _maker_price_from_open(float(open_[i]), int(position["side"]), float(execution.exit_touch_offset_bps), is_entry=False)
                touched = _touch_hit(
                    int(position["side"]),
                    px,
                    float(high[i]),
                    float(low[i]),
                    rule=str(execution.exit_touch_rule),
                    offset_bps=float(execution.exit_touch_offset_bps),
                    frac_range=float(execution.exit_touch_frac_range),
                    is_entry=False,
                )
                if touched:
                    exit_price_fill = px
                    exit_price_planned = px
                    exit_fill_mode = ExitFillModeV5.MAKER_TOUCH.value
                    exit_fee_kind = "maker"
                    exit_maker_flag = True
                    sim_meta["maker_exit_count"] += 1
                else:
                    exit_price_fill = _ioc_price_from_open(float(open_[i]), int(position["side"]), float(execution.sl_ioc_offset_bps), is_entry=False)
                    exit_fill_mode = ExitFillModeV5.IOC_LIMIT.value
                    sim_meta["ioc_exit_count"] += 1
            elif fallback == FallbackPathV5.MARKET.value:
                off = _bps_factor(float(execution.market_exit_impact_cap_bps if bool(execution.emergency_market_enabled) else execution.emergency_market_offset_bps))
                exit_price_fill = float(open_[i]) * (1.0 - off if int(position["side"]) > 0 else 1.0 + off)
                exit_fill_mode = ExitFillModeV5.MARKET.value
                exit_fee_kind = "taker"
                sim_meta["market_exit_count"] += 1
            else:
                exit_price_fill = _ioc_price_from_open(float(open_[i]), int(position["side"]), float(execution.sl_ioc_offset_bps), is_entry=False)
                exit_fill_mode = ExitFillModeV5.IOC_LIMIT.value
                sim_meta["ioc_exit_count"] += 1

            row = _close_trade_from_position_v5(
                position,
                exit_idx=int(i),
                exit_ts=ts,
                exit_price_fill=float(exit_price_fill),
                exit_price_planned=float(exit_price_planned),
                exit_fill_mode=str(exit_fill_mode),
                exit_fee_kind=str(exit_fee_kind),
                exit_maker_flag=bool(exit_maker_flag),
                exit_fallback_path=str(fallback),
                exit_reason=str(reason),
                same_bar_resolution="single",
                same_bar_both_hit=False,
            )
            if collect_trades:
                trades.append(row)
            sim_meta["exit_reason_counts"][str(reason)] = int(sim_meta["exit_reason_counts"].get(str(reason), 0)) + 1
            last_exit_idx = int(i)
            last_exit_reason = str(reason)
            last_exit_side = int(position["side"])
            last_exit_price = float(exit_price_fill)
            last_exit_entry_core = float(position.get("entry_core", 0.0))
            cooldown_remaining = max(int(policy.cooldown_bars), 0)
            position = None
            action = f"EXIT_{reason}"
            exec_stage = ExecStageV5.EXIT_FILLED.value
            open_trade_id = ""
            continue  # do not open new entry on same bar after a pending exit open fill

        # ------------------------------------------------------------------
        # 3) manage open position intrabar
        # ------------------------------------------------------------------
        if position is not None:
            side = int(position["side"])
            fav_n = _favorable_n(side, float(position["entry_price_fill"]), float(high[i]), float(low[i]), float(position["scale_ref_t"]))
            adv_n = _adverse_n(side, float(position["entry_price_fill"]), float(high[i]), float(low[i]), float(position["scale_ref_t"]))
            if np.isfinite(fav_n):
                position["peak_fav_n"] = max(float(position["peak_fav_n"]), float(fav_n))
            if np.isfinite(adv_n):
                position["peak_adv_n"] = max(float(position["peak_adv_n"]), float(adv_n))
            bars_completed = int(i - int(position["entry_idx"]))
            position["bars_completed"] = bars_completed

            progress_frac = float(position["peak_fav_n"]) / max(float(position["tp_n"]), EPS)
            unlocked = bool(progress_frac >= float(position["bep_arm_n"]))
            if unlocked and not bool(position["bep_armed"]):
                position["bep_armed"] = True
                position["bep_armed_idx"] = int(i)
                position["exec_stage"] = ExecStageV5.BEP_ARMED.value

            # progress protect
            soft_sl_n = float(position["sl_n"])
            trail_n_eff = float(position["trail_n"])
            local_soft_hold = int(position.get("min_hold_soft_sl_bars", policy.min_hold_soft_sl_bars))
            if bool(dynamic.enabled):
                local_soft_hold = max(int(dynamic.softsl_hold_floor), int(local_soft_hold) - int(bundle["dyn_softsl_relax"][i]))
            if bool(progress_protect.early_softsl_enabled) and bars_completed >= int(progress_protect.early_softsl_min_hold) and progress_frac >= float(progress_protect.early_softsl_progress_frac):
                soft_sl_n = min(soft_sl_n, max(float(position["sl_n"]) * 0.80, float(position["bep_arm_n"]) * 0.85))
            if bool(progress_protect.early_trail_enabled) and bars_completed >= int(progress_protect.early_trail_min_hold) and progress_frac >= float(progress_protect.early_trail_progress_frac):
                trail_n_eff = min(trail_n_eff, max(float(position["trail_n"]) * 0.85, 0.05))

            # pre-BEP timeout
            if bool(dynamic.enabled) and bool(dynamic.use_pre_bep_timeout) and (not bool(position["bep_armed"])) and bars_completed >= int(dynamic.pre_bep_timeout_bars):
                if progress_frac < float(dynamic.pre_bep_progress_frac) and float(bundle["stress_raw"][i]) >= float(dynamic.pre_bep_stress_th):
                    position["pre_bep_timeout_hit"] = True
                    soft_sl_n = soft_sl_n * float(dynamic.pre_bep_degrade_sl_scale)
                    local_soft_hold = max(0, local_soft_hold - int(dynamic.pre_bep_softsl_delta))
                    if int(dynamic.pre_bep_force_close_bars) > 0 and bars_completed >= int(dynamic.pre_bep_force_close_bars):
                        position["thesis_fail_hard"] = True
                        _set_pending_exit(
                            position,
                            reason=ExitReasonV5.PRE_BEP_TIMEOUT.value,
                            fallback=FallbackPathV5.IOC_LIMIT.value if bool(policy.thesis_pre_bep_ioc_enable) else FallbackPathV5.POST_ONLY.value,
                            activate_idx=min(i + 1, n - 1),
                            thesis_state=ThesisStateV5.WEAK_OPPOSITE.value,
                            note="pre_bep_timeout_force_close",
                        )
                        sim_meta["n_pending_hard_exit"] += 1

            # tp window
            tpw_live = bool(position.get("tpw_live", False))
            if bool(tp_window.enabled) and (not tpw_live) and progress_frac >= float(tp_window.progress_frac_arm):
                position["tpw_live"] = True
                position["tpw_armed_idx"] = int(i)
                tpw_live = True
            if tpw_live:
                position["tpw_peak_progress"] = max(float(position.get("tpw_peak_progress", 0.0)), progress_frac)
                if bool(tp_window.block_early_soft_sl):
                    soft_sl_n = max(soft_sl_n, float(position["sl_n"]))
                if bool(tp_window.block_early_trail):
                    trail_n_eff = max(trail_n_eff, float(position["trail_n"]))
                pullback_frac = 0.0
                peak_fav = max(float(position.get("peak_fav_n", 0.0)), EPS)
                if np.isfinite(fav_n):
                    pullback_frac = max(0.0, peak_fav - float(fav_n)) / peak_fav
                if pullback_frac >= float(tp_window.expire_on_pullback_frac):
                    position["tpw_live"] = False
                    tpw_live = False

            # trailing logic
            trail_stop_n = float("nan")
            if bool(position["bep_armed"]):
                peak_fav = float(position["peak_fav_n"])
                trail_stop_n = max(float(position["bep_arm_n"]), peak_fav - float(trail_n_eff))
                position["trail_stop_n"] = trail_stop_n
                position["exec_stage"] = ExecStageV5.TRAIL_ACTIVE.value
            else:
                position["trail_stop_n"] = float("nan")

            # profit floor
            if bool(policy.profit_floor_enabled) and float(position["peak_fav_n"]) > 0.0:
                floor_n = float(position["peak_fav_n"]) * float(policy.profit_floor_frac_of_mfe)
                if np.isfinite(fav_n) and float(fav_n) < floor_n and bars_completed >= 1:
                    position["profit_floor_hit"] = True
                    _set_pending_exit(
                        position,
                        reason=ExitReasonV5.PROFIT_FLOOR.value,
                        fallback=FallbackPathV5.IOC_LIMIT.value,
                        activate_idx=min(i + 1, n - 1),
                        thesis_state=str(position.get("thesis_state_last", ThesisStateV5.NONE.value)),
                        note="profit_floor",
                    )
                    sim_meta["n_pending_soft_exit"] += 1

            # same-side-hold support
            same_state = _classify_thesis_v5(arrays, bundle, position, policy, i, shock_flag=shock_flag)
            position["support_strength_ratio_last"] = float(same_state["support_strength_ratio"])
            if bool(same_side_hold.enabled):
                strong_ok = same_state["support_strength_ratio"] >= float(same_side_hold.strong_ratio) and same_state["same_hyb_signed"] > 0.0
                weak_ok = bool(same_side_hold.weak_enabled) and same_state["support_strength_ratio"] >= float(same_side_hold.weak_ratio) and same_state["progress_frac"] >= float(same_side_hold.weak_min_progress_frac)
                if bars_completed >= int(position.get("effective_max_hold", position["max_hold_bars"])) - 1:
                    if strong_ok:
                        position["same_side_bonus_bars"] = min(int(same_side_hold.max_extra_bars), int(position.get("same_side_bonus_bars", 0)) + int(same_side_hold.bonus_bars_strong))
                    elif weak_ok:
                        position["same_side_bonus_bars"] = min(int(same_side_hold.max_extra_bars), int(position.get("same_side_bonus_bars", 0)) + int(same_side_hold.bonus_bars_weak))

            # thesis monitoring
            state = str(same_state["state"])
            prev_cand = str(position.get("thesis_state_candidate", ThesisStateV5.NONE.value))
            if state == prev_cand:
                position["thesis_state_count"] = int(position.get("thesis_state_count", 0)) + 1
            else:
                position["thesis_state_candidate"] = state
                position["thesis_state_count"] = 1
            if int(position["thesis_state_count"]) >= int(policy.thesis_state_confirm_bars):
                position["thesis_state_last"] = state
            sim_meta["thesis_state_counts"][str(position.get("thesis_state_last", ThesisStateV5.NONE.value))] = int(sim_meta["thesis_state_counts"].get(str(position.get("thesis_state_last", ThesisStateV5.NONE.value)), 0)) + 1

            progress_protected = same_state["progress_frac"] >= float(policy.thesis_progress_protect_frac)
            near_maxhold = bars_completed >= max(0, int(position["max_hold_bars"]) - 1)

            if shock_flag:
                position["shock_exit_hit"] = True
                sim_meta["shock_count"] += 1
                _set_pending_exit(
                    position,
                    reason=ExitReasonV5.SHOCK_EXIT.value,
                    fallback=FallbackPathV5.IOC_LIMIT.value,
                    activate_idx=min(i + 1, n - 1),
                    thesis_state=ThesisStateV5.LIQUIDITY_SHOCK.value,
                    note="shock_exit",
                )
                sim_meta["n_pending_hard_exit"] += 1
            else:
                if bars_completed >= int(policy.thesis_grace_bars_after_entry):
                    if str(position.get("thesis_state_last")) == ThesisStateV5.STRONG_OPPOSITE.value:
                        if progress_protected and bool(policy.thesis_post_bep_trail_tighten):
                            position["trail_n"] = min(float(position["trail_n"]), max(float(position["trail_n"]) * 0.80, 0.05))
                        else:
                            position["thesis_fail_hard"] = True
                            opp_strength = max(0.0, same_state["opp_core"] - same_state["same_core"], -same_state["dir_mix_signed"])
                            if opp_strength >= float(policy.thesis_market_exit_margin) and bool(execution.emergency_market_enabled):
                                fb = FallbackPathV5.MARKET.value
                            elif opp_strength >= float(policy.thesis_ioc_exit_margin) or (not bool(position["bep_armed"]) and bool(policy.thesis_pre_bep_ioc_enable)):
                                fb = FallbackPathV5.IOC_LIMIT.value
                            else:
                                fb = FallbackPathV5.POST_ONLY.value
                            _set_pending_exit(position, reason=ExitReasonV5.THESIS_FAIL_HARD.value, fallback=fb, activate_idx=min(i + 1, n - 1), thesis_state=str(position.get("thesis_state_last")), note=_stable_json(same_state))
                            sim_meta["n_pending_hard_exit"] += 1
                    elif str(position.get("thesis_state_last")) == ThesisStateV5.WEAK_OPPOSITE.value:
                        if progress_protected and bool(policy.thesis_post_bep_trail_tighten):
                            position["trail_n"] = min(float(position["trail_n"]), max(float(position["trail_n"]) * 0.85, 0.05))
                        else:
                            position["thesis_fail_soft"] = True
                            fb = FallbackPathV5.IOC_LIMIT.value if near_maxhold else FallbackPathV5.POST_ONLY.value
                            _set_pending_exit(position, reason=ExitReasonV5.THESIS_FAIL_SOFT.value, fallback=fb, activate_idx=min(i + 1, n - 1), thesis_state=str(position.get("thesis_state_last")), note=_stable_json(same_state))
                            sim_meta["n_pending_soft_exit"] += 1
                    elif str(position.get("thesis_state_last")) == ThesisStateV5.NEUTRAL_DRIFT.value and bars_completed >= int(policy.thesis_neutral_tighten_after_bars):
                        position["trail_n"] = min(float(position["trail_n"]), max(float(position["trail_n"]) * 0.90, 0.05))

            # intrabar gap flatten at current open already observed
            if bool(policy.gap_force_exit_enabled) and gap_flag and bars_completed > 0 and position is not None:
                position["gap_flatten_hit"] = True
                row = _close_trade_from_position_v5(
                    position,
                    exit_idx=int(i),
                    exit_ts=ts,
                    exit_price_fill=float(open_[i]),
                    exit_price_planned=float(open_[i]),
                    exit_fill_mode=ExitFillModeV5.GAP_OPEN.value,
                    exit_fee_kind="taker",
                    exit_maker_flag=False,
                    exit_fallback_path=FallbackPathV5.TAKER_NEXT_OPEN.value,
                    exit_reason=ExitReasonV5.GAP_FLATTEN.value,
                    same_bar_resolution="single",
                    same_bar_both_hit=False,
                )
                if collect_trades:
                    trades.append(row)
                sim_meta["exit_reason_counts"][ExitReasonV5.GAP_FLATTEN.value] = int(sim_meta["exit_reason_counts"].get(ExitReasonV5.GAP_FLATTEN.value, 0)) + 1
                cooldown_remaining = max(int(policy.cooldown_bars), 0)
                last_exit_idx = int(i)
                last_exit_reason = ExitReasonV5.GAP_FLATTEN.value
                last_exit_side = int(position["side"])
                last_exit_price = float(open_[i])
                last_exit_entry_core = float(position.get("entry_core", 0.0))
                position = None
                action = "EXIT_GAP"
                exec_stage = ExecStageV5.EXIT_FILLED.value
                open_trade_id = ""
            else:
                # intrabar TP/SL/TRAIL / soft SL
                reason = ""
                exit_price_fill = float("nan")
                exit_fill_mode = ""
                exit_fee_kind = ""
                exit_maker_flag = False
                exit_fallback_path = FallbackPathV5.NONE.value
                tp_n = float(position["tp_n"])
                sl_n = max(float(position["sl_n"]) * (float(policy.hard_sl_mult_pre_unlock) if not bool(position["bep_armed"]) else 1.0), 0.01)
                tp_hit = np.isfinite(fav_n) and float(fav_n) >= tp_n
                sl_hit = np.isfinite(adv_n) and float(adv_n) >= sl_n
                softsl_hit = np.isfinite(adv_n) and float(adv_n) >= soft_sl_n and bars_completed >= int(local_soft_hold)
                trail_hit = np.isfinite(trail_stop_n) and np.isfinite(adv_n) and float(adv_n) >= float(trail_stop_n) and bars_completed >= int(position.get("min_hold_trail_bars", policy.min_hold_trail_bars) if position["bep_armed"] else 0)

                same_bar_res = _same_bar_resolution(str(backtest.intrabar_mode), tp_hit=bool(tp_hit), stop_hits=bool(sl_hit or trail_hit or softsl_hit))
                if tp_hit and (not (sl_hit or trail_hit or softsl_hit) or same_bar_res == "favorable_first"):
                    position["tp_touched"] = True
                    px = _long_price_from_n(float(position["entry_price_fill"]), tp_n, float(position["scale_ref_t"])) if side > 0 else _short_price_from_n(float(position["entry_price_fill"]), tp_n, float(position["scale_ref_t"]))
                    exit_price_fill = float(px)
                    exit_fill_mode = str(execution.tp_fill_mode)
                    reason = ExitReasonV5.TP.value
                    if str(execution.tp_fill_mode) in {ExitFillModeV5.MAKER_TOUCH.value, ExitFillModeV5.MAKER_TOUCH_THEN_IOC.value}:
                        exit_fee_kind = "maker"
                        exit_maker_flag = True
                        exit_fallback_path = FallbackPathV5.POST_ONLY.value
                        sim_meta["maker_exit_count"] += 1
                    else:
                        exit_fee_kind = "taker"
                        exit_maker_flag = False
                        exit_fallback_path = FallbackPathV5.IOC_LIMIT.value
                        sim_meta["ioc_exit_count"] += 1
                elif trail_hit:
                    position["trail_touched"] = True
                    px = _long_price_from_n(float(position["entry_price_fill"]), float(trail_stop_n), float(position["scale_ref_t"])) if side > 0 else _short_price_from_n(float(position["entry_price_fill"]), float(trail_stop_n), float(position["scale_ref_t"]))
                    exit_price_fill = float(px)
                    exit_fill_mode = str(execution.trail_fill_mode)
                    reason = ExitReasonV5.TRAIL.value
                    if str(execution.trail_fill_mode) == ExitFillModeV5.MAKER_TOUCH.value:
                        exit_fee_kind = "maker"
                        exit_maker_flag = True
                        exit_fallback_path = FallbackPathV5.POST_ONLY.value
                        sim_meta["maker_exit_count"] += 1
                    else:
                        exit_fee_kind = "taker"
                        exit_maker_flag = False
                        exit_fallback_path = FallbackPathV5.IOC_LIMIT.value
                        sim_meta["ioc_exit_count"] += 1
                elif sl_hit or softsl_hit:
                    position["sl_touched"] = True
                    stop_n = float(soft_sl_n if softsl_hit else sl_n)
                    px = _long_price_from_n(float(position["entry_price_fill"]), -stop_n, float(position["scale_ref_t"])) if side > 0 else _short_price_from_n(float(position["entry_price_fill"]), -stop_n, float(position["scale_ref_t"]))
                    exit_price_fill = float(px)
                    exit_fill_mode = str(execution.softsl_fill_mode if softsl_hit else execution.sl_fill_mode)
                    reason = ExitReasonV5.SOFT_SL.value if softsl_hit else ExitReasonV5.SL.value
                    exit_fee_kind = "taker"
                    exit_maker_flag = False
                    exit_fallback_path = FallbackPathV5.IOC_LIMIT.value
                    sim_meta["ioc_exit_count"] += 1

                if reason:
                    row = _close_trade_from_position_v5(
                        position,
                        exit_idx=int(i),
                        exit_ts=ts,
                        exit_price_fill=float(exit_price_fill),
                        exit_price_planned=float(exit_price_fill),
                        exit_fill_mode=str(exit_fill_mode),
                        exit_fee_kind=str(exit_fee_kind),
                        exit_maker_flag=bool(exit_maker_flag),
                        exit_fallback_path=str(exit_fallback_path),
                        exit_reason=str(reason),
                        same_bar_resolution=str(same_bar_res),
                        same_bar_both_hit=bool(tp_hit and (sl_hit or trail_hit or softsl_hit)),
                    )
                    if collect_trades:
                        trades.append(row)
                    sim_meta["exit_reason_counts"][str(reason)] = int(sim_meta["exit_reason_counts"].get(str(reason), 0)) + 1
                    cooldown_remaining = max(int(policy.cooldown_bars), 0)
                    last_exit_idx = int(i)
                    last_exit_reason = str(reason)
                    last_exit_side = int(side)
                    last_exit_price = float(exit_price_fill)
                    last_exit_entry_core = float(position.get("entry_core", 0.0))
                    position = None
                    action = f"EXIT_{reason}"
                    exec_stage = ExecStageV5.EXIT_FILLED.value
                    open_trade_id = ""
                else:
                    effective_max_hold = int(position.get("same_side_bonus_bars", 0)) + int(policy.max_hold_bars)
                    position["effective_max_hold"] = effective_max_hold
                    if bars_completed >= int(policy.hard_max_hold_bars):
                        _set_pending_exit(
                            position,
                            reason=ExitReasonV5.MAX_HOLD.value,
                            fallback=FallbackPathV5.IOC_LIMIT.value,
                            activate_idx=min(i + 1, n - 1),
                            thesis_state=str(position.get("thesis_state_last", ThesisStateV5.NONE.value)),
                            note="hard_max_hold",
                        )
                        sim_meta["n_pending_hard_exit"] += 1
                    elif bars_completed >= int(effective_max_hold):
                        _set_pending_exit(
                            position,
                            reason=ExitReasonV5.MAX_HOLD.value,
                            fallback=FallbackPathV5.POST_ONLY.value,
                            activate_idx=min(i + 1, n - 1),
                            thesis_state=str(position.get("thesis_state_last", ThesisStateV5.NONE.value)),
                            note="max_hold",
                        )
                        sim_meta["n_pending_hard_exit"] += 1

            if position is not None:
                exec_stage = str(position.get("exec_stage", ExecStageV5.PRE_BEP.value))
                open_trade_id = str(position.get("trade_id", ""))

        # ------------------------------------------------------------------
        # 4) new entry signal if flat
        # ------------------------------------------------------------------
        if position is None:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            can_new_entry = (pending_entry is None) and (cooldown_remaining <= 0)
            if can_new_entry and policy_gate_passed and i < (n - 1):
                side = int(decision_diag["side"])
                if gap_flag:
                    sim_meta["n_gap_skip"] += 1
                    action = "SKIP_GAP"
                    exec_stage = ExecStageV5.ENTRY_SKIPPED.value
                    entry_block_reason = "gap_skip"
                else:
                    # episode / rearm control
                    if side != run_side or (i - last_entry_signal_idx) > max(1, int(entry_episode.run_gap_reset_bars)):
                        run_id += 1
                        run_side = int(side)
                        run_entries = 0
                    allow_entry = run_entries < max(1, int(entry_episode.episode_max_entries_per_run)) if bool(entry_episode.entry_episode_enabled) else True
                    is_rearm_entry = False
                    if (not allow_entry) and bool(entry_episode.rearm_enabled):
                        allowed_reason = (
                            (last_exit_reason == ExitReasonV5.TRAIL.value and bool(entry_episode.rearm_after_trail))
                            or (last_exit_reason == ExitReasonV5.TP.value and bool(entry_episode.rearm_after_tp))
                            or (last_exit_reason == ExitReasonV5.SL.value and bool(entry_episode.rearm_after_sl))
                        )
                        price_reset_ok = True
                        if np.isfinite(last_exit_price) and last_exit_side == side:
                            price_reset_ok = abs(math.log(max(float(open_[i]), EPS) / max(float(last_exit_price), EPS))) / max(scale_ref_cur, EPS) >= float(entry_episode.rearm_price_reset_frac)
                        allow_entry = (
                            allowed_reason
                            and (side == last_exit_side if bool(entry_episode.rearm_same_side_only) else True)
                            and (i - last_exit_idx) <= int(entry_episode.rearm_max_bars_after_exit)
                            and (i - last_exit_idx) >= int(entry_episode.rearm_cooldown_bars)
                            and float(decision_diag["entry_core"]) >= float(last_exit_entry_core) * float(entry_episode.rearm_gate_refresh_frac)
                            and price_reset_ok
                        )
                        is_rearm_entry = bool(allow_entry)
                    if allow_entry:
                        plan = dict(decision_diag["plan"])
                        pending_entry = {
                            "decision_idx": int(i),
                            "decision_ts": str(ts.isoformat()),
                            "activate_idx": int(i + 1),
                            "expire_idx": int(min(i + max(1, int(execution.entry_fill_max_bars)), n - 1)),
                            "side": int(plan["side"]),
                            "plan": plan,
                            "entry_price_planned": float(open_[min(i + 1, n - 1)]),
                            "run_id": int(run_id),
                            "is_rearm_entry": bool(is_rearm_entry),
                        }
                        last_entry_signal_idx = int(i)
                        run_entries += 1
                        sim_meta["n_entry_signals"] += 1
                        if bool(is_rearm_entry):
                            sim_meta["n_rearm_entry_signals"] += 1
                        action = "ENTRY_SIGNAL_ACCEPTED"
                        exec_stage = ExecStageV5.ENTRY_SIGNAL_ACCEPTED.value
                    else:
                        action = "SKIP_EPISODE"
                        exec_stage = ExecStageV5.ENTRY_SKIPPED.value
                        entry_block_reason = "entry_episode_limit"
            elif can_new_entry and (not policy_gate_passed):
                action = "SKIP_POLICY"
                exec_stage = ExecStageV5.ENTRY_SKIPPED.value
                entry_block_reason = str(decision_diag.get("gate_fail_stage", "policy_gate"))
            elif (pending_entry is None) and (cooldown_remaining > 0):
                sim_meta["n_cooldown_skip"] += 1
                action = "SKIP_COOLDOWN"
                exec_stage = ExecStageV5.ENTRY_SKIPPED.value
                entry_block_reason = "cooldown"
        else:
            if action == "NO_ACTION":
                action = "HOLD"

        if collect_decisions:
            decision_row = {
                "timestamp": str(ts.isoformat()),
                "row_idx": int(i),
                "pred_ready": bool(arrays["pred_ready"][i]),
                "feature_ready": bool(arrays["feature_ready"][i]),
                "eval_in_range": True,
                "cooldown_active": bool(cooldown_remaining > 0),
                "chosen_side": int(decision_diag.get("side", 0) or 0),
                "entry_long_core": float(bundle["entry_long_core"][i]),
                "entry_short_core": float(bundle["entry_short_core"][i]),
                "entry_core": float(bundle["entry_core"][i]),
                "entry_gap": float(bundle["entry_gap"][i]),
                "hyb_mix": float(bundle["hyb_mix"][i]),
                "cls_mix": float(bundle["cls_mix"][i]),
                "dirprob_mix": float(bundle["dirprob_mix"][i]),
                "dir_mix": float(bundle["dir_mix"][i]),
                "util_long_mix": float(bundle["util_long_mix"][i]),
                "util_short_mix": float(bundle["util_short_mix"][i]),
                "side_agreement_frac": float(bundle["side_agreement_frac"][i]),
                "utility_10_side": float(decision_diag.get("utility_10_side", JSON_NAN)),
                "utility_10_gap": float(decision_diag.get("utility_10_gap", JSON_NAN)),
                "main_confirm_prob": float(decision_diag.get("main_confirm_prob", JSON_NAN)),
                "timing_first_hit_prob": float(decision_diag.get("timing_first_hit_prob", JSON_NAN)),
                "timing_expected_bars": float(decision_diag.get("timing_expected_bars", JSON_NAN)),
                "timing_censored_prob": float(decision_diag.get("timing_censored_prob", JSON_NAN)),
                "gate_threshold": float(decision_diag.get("gate_threshold", JSON_NAN)),
                "q_entry_used": float(decision_diag.get("q_entry_used", JSON_NAN)),
                "entry_floor_used": float(decision_diag.get("entry_floor_used", JSON_NAN)),
                "policy_gate_passed": bool(policy_gate_passed),
                "gate_fail_stage": str(decision_diag.get("gate_fail_stage", "")),
                "gate_fail_detail": str(decision_diag.get("gate_fail_detail", "")),
                "gate_fail_value": float(decision_diag.get("gate_fail_value", JSON_NAN)),
                "gate_fail_threshold": float(decision_diag.get("gate_fail_threshold", JSON_NAN)),
                "retcls_alignment_state": str(decision_diag.get("retcls_alignment_state", "")),
                "retcls_alignment_score": float(decision_diag.get("retcls_alignment_score", JSON_NAN)),
                "regime_bucket": int(bundle["regime_bucket"][i]),
                "regime_bucket_name": _bucket_name_from_code(int(bundle["regime_bucket"][i])),
                "regime_alpha": float(bundle["regime_alpha"][i]),
                "active_sparse_flag": bool(bundle["active_sparse_flag"][i]),
                "filter_pass": bool(bundle["filter_pass"][i]),
                "dyn_stress": float(bundle["stress_raw"][i]),
                "dyn_gate_mult": float(bundle["dyn_gate_mult"][i]),
                "dyn_lev_scale": float(bundle["dyn_lev_scale"][i]),
                "dyn_bep_scale": float(bundle["dyn_bep_scale"][i]),
                "dyn_trail_scale": float(bundle["dyn_trail_scale"][i]),
                "dyn_sl_scale": float(bundle["dyn_sl_scale"][i]),
                "planned_tp_n": float(decision_diag["plan"]["tp_n"]) if decision_diag.get("plan") else float("nan"),
                "planned_sl_n": float(decision_diag["plan"]["sl_n"]) if decision_diag.get("plan") else float("nan"),
                "planned_bep_arm_n": float(decision_diag["plan"]["bep_arm_n"]) if decision_diag.get("plan") else float("nan"),
                "planned_trail_n": float(decision_diag["plan"]["trail_n"]) if decision_diag.get("plan") else float("nan"),
                "gap_flag": bool(gap_flag),
                "gap_n": float(gap_n),
                "shock_flag": bool(shock_flag),
                "shock_range_n": float(range_n),
                "thesis_state": str(position.get("thesis_state_last", ThesisStateV5.NONE.value) if position is not None else ThesisStateV5.NONE.value),
                "action": str(action),
                "entry_block_reason": str(entry_block_reason),
                "open_trade_id": str(open_trade_id),
                "exec_stage": str(exec_stage),
            }
            decisions.append(decision_row)
            _count_action(action)

    if position is not None:
        final_idx = n - 1
        final_ts = pd.Timestamp(int(ts_ns[final_idx]), tz="UTC")
        row = _close_trade_from_position_v5(
            position,
            exit_idx=int(final_idx),
            exit_ts=final_ts,
            exit_price_fill=float(close[final_idx]),
            exit_price_planned=float(close[final_idx]),
            exit_fill_mode=ExitFillModeV5.TAKER_CLOSE.value,
            exit_fee_kind="taker",
            exit_maker_flag=False,
            exit_fallback_path=FallbackPathV5.IOC_LIMIT.value,
            exit_reason=ExitReasonV5.FORCE_CLOSE.value,
            same_bar_resolution="single",
            same_bar_both_hit=False,
        )
        if collect_trades:
            trades.append(row)
        sim_meta["exit_reason_counts"][ExitReasonV5.FORCE_CLOSE.value] = int(sim_meta["exit_reason_counts"].get(ExitReasonV5.FORCE_CLOSE.value, 0)) + 1
        position = None

    return trades, decisions, sim_meta


# -----------------------------------------------------------------------------
# Public simulation APIs
# -----------------------------------------------------------------------------

def simulate_from_prediction_frame_v5(
    pred_eval: pd.DataFrame,
    *,
    policy: PolicyConfigV5,
    dynamic: Optional[DynamicConfigV5] = None,
    progress_protect: Optional[ProgressProtectConfigV5] = None,
    tp_window: Optional[TPWindowConfigV5] = None,
    entry_episode: Optional[EntryEpisodeConfigV5] = None,
    same_side_hold: Optional[SameSideHoldConfigV5] = None,
    regime_detect: Optional[RegimeDetectConfigV5] = None,
    regime_weight: Optional[RegimeWeightConfigV5] = None,
    regime_threshold: Optional[RegimeThresholdConfigV5] = None,
    regime_filter: Optional[RegimeFilterConfigV5] = None,
    regime_lane: Optional[RegimeLaneConfigV5] = None,
    execution: Optional[ExecutionConfigV5] = None,
    backtest: Optional[BacktestConfigV5] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cache = prepare_fast_eval_cache_v5(pred_eval)
    ctx = prepare_trial_context_v5(
        cache,
        policy=policy,
        dynamic=dynamic,
        progress_protect=progress_protect,
        tp_window=tp_window,
        entry_episode=entry_episode,
        same_side_hold=same_side_hold,
        regime_detect=regime_detect,
        regime_weight=regime_weight,
        regime_threshold=regime_threshold,
        regime_filter=regime_filter,
        regime_lane=regime_lane,
        execution=execution,
        backtest=backtest,
    )
    trade_rows, decision_rows, sim_meta = _simulate_core_arrays_v5(
        ctx,
        collect_decisions=True,
        collect_trades=True,
    )
    return pd.DataFrame(trade_rows), pd.DataFrame(decision_rows), sim_meta


def evaluate_prediction_frame_fast_v5(
    cache: Mapping[str, Any],
    *,
    policy: PolicyConfigV5,
    dynamic: Optional[DynamicConfigV5] = None,
    progress_protect: Optional[ProgressProtectConfigV5] = None,
    tp_window: Optional[TPWindowConfigV5] = None,
    entry_episode: Optional[EntryEpisodeConfigV5] = None,
    same_side_hold: Optional[SameSideHoldConfigV5] = None,
    regime_detect: Optional[RegimeDetectConfigV5] = None,
    regime_weight: Optional[RegimeWeightConfigV5] = None,
    regime_threshold: Optional[RegimeThresholdConfigV5] = None,
    regime_filter: Optional[RegimeFilterConfigV5] = None,
    regime_lane: Optional[RegimeLaneConfigV5] = None,
    execution: Optional[ExecutionConfigV5] = None,
    backtest: Optional[BacktestConfigV5] = None,
    objective: Optional[ObjectiveConfigV5] = None,
) -> Dict[str, Any]:
    execution = execution or ExecutionConfigV5()
    backtest = backtest or BacktestConfigV5()
    objective = objective or ObjectiveConfigV5()
    ctx = prepare_trial_context_v5(
        cache,
        policy=policy,
        dynamic=dynamic,
        progress_protect=progress_protect,
        tp_window=tp_window,
        entry_episode=entry_episode,
        same_side_hold=same_side_hold,
        regime_detect=regime_detect,
        regime_weight=regime_weight,
        regime_threshold=regime_threshold,
        regime_filter=regime_filter,
        regime_lane=regime_lane,
        execution=execution,
        backtest=backtest,
    )
    trade_rows, _decision_rows, sim_meta = _simulate_core_arrays_v5(ctx, collect_decisions=False, collect_trades=True)
    trades = pd.DataFrame(trade_rows)
    fake_ts = pd.to_datetime(pd.Series(cache["timestamp_ns"], copy=False), utc=True, errors="coerce")
    equity = build_equity_curve_v5(trades, timestamps=fake_ts, annualization_days=float(backtest.annualization_days))
    segments = summarize_segments_v5(trades, timestamps=fake_ts, segments=int(backtest.segments))
    reason_stats = summarize_reason_stats_v5(trades)
    overall = summarize_trades_v5(trades, equity=equity, segments=segments, sim_meta=sim_meta)
    objective_payload = assemble_objective_v5(overall=overall, segments=segments, sim_meta=sim_meta, objective=objective)
    return {
        "trades": trades,
        "equity": equity,
        "segments": segments,
        "reason_stats": reason_stats,
        "overall": overall,
        "objective": objective_payload,
        "sim_meta": sim_meta,
        "context": ctx,
    }


def evaluate_prepared_single_segment_fast_v5(
    context: Mapping[str, Any],
    *,
    objective: Optional[ObjectiveConfigV5] = None,
) -> Dict[str, Any]:
    policy: PolicyConfigV5 = context["policy"]
    execution: ExecutionConfigV5 = context["execution"]
    backtest: BacktestConfigV5 = context["backtest"]
    objective = objective or context.get("objective") or ObjectiveConfigV5()
    trade_rows, _decision_rows, sim_meta = _simulate_core_arrays_v5(context, collect_decisions=False, collect_trades=True)
    trades = pd.DataFrame(trade_rows)
    fake_ts = pd.to_datetime(pd.Series(context["arrays"]["timestamp_ns"], copy=False), utc=True, errors="coerce")
    equity = build_equity_curve_v5(trades, timestamps=fake_ts, annualization_days=float(backtest.annualization_days))
    segments = summarize_segments_v5(trades, timestamps=fake_ts, segments=int(backtest.segments))
    reason_stats = summarize_reason_stats_v5(trades)
    overall = summarize_trades_v5(trades, equity=equity, segments=segments, sim_meta=sim_meta)
    objective_payload = assemble_objective_v5(overall=overall, segments=segments, sim_meta=sim_meta, objective=objective)
    return {
        "trades": trades,
        "equity": equity,
        "segments": segments,
        "reason_stats": reason_stats,
        "overall": overall,
        "objective": objective_payload,
        "sim_meta": sim_meta,
    }


def warmup_single_fast_core_v5() -> bool:
    return True


# -----------------------------------------------------------------------------
# Summaries / objective
# -----------------------------------------------------------------------------

def build_equity_curve_v5(trades: pd.DataFrame, *, timestamps: Optional[pd.Series] = None, annualization_days: float = 365.0) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["timestamp", "trade_id", "net_pnl_lev", "equity", "drawdown"])
    tr = trades.sort_values(["exit_idx", "trade_id"], kind="mergesort").reset_index(drop=True).copy()
    pnl = tr["net_pnl_lev"].astype(float).to_numpy(copy=False)
    equity = _cum_equity_numba(pnl.astype(np.float64)).astype(np.float64)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(tr["exit_ts"], utc=True, errors="coerce"),
            "trade_id": tr["trade_id"].astype(str),
            "net_pnl_lev": pnl,
            "equity": equity,
            "drawdown": dd,
        }
    )


def summarize_segments_v5(trades: pd.DataFrame, *, timestamps: pd.Series, segments: int = 10) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        cols = ["seg", "start_ts", "end_ts", "trade_count", "net_sum", "gross_sum", "win_rate", "maker_entry_ratio", "maker_exit_ratio", "short_share"]
        return pd.DataFrame(columns=cols)
    n_rows = int(len(timestamps))
    boundaries = np.linspace(0, n_rows, num=max(1, int(segments)) + 1, dtype=np.int64)
    rows: List[Dict[str, Any]] = []
    tr = trades.copy()
    tr["decision_idx"] = pd.to_numeric(tr["decision_idx"], errors="coerce").fillna(-1).astype(int)
    for seg in range(len(boundaries) - 1):
        lo = int(boundaries[seg])
        hi = int(boundaries[seg + 1])
        mask = (tr["decision_idx"] >= lo) & (tr["decision_idx"] < hi)
        sub = tr.loc[mask].copy()
        short_share = float(np.mean(pd.to_numeric(sub["side"], errors="coerce").to_numpy(dtype=float) < 0.0)) if len(sub) else float("nan")
        rows.append(
            {
                "seg": int(seg),
                "start_ts": str(pd.Timestamp(timestamps.iloc[lo]).isoformat()) if lo < len(timestamps) else "",
                "end_ts": str(pd.Timestamp(timestamps.iloc[max(lo, hi - 1)]).isoformat()) if hi > 0 and (hi - 1) < len(timestamps) else "",
                "trade_count": int(len(sub)),
                "net_sum": float(sub["net_pnl_lev"].sum()) if len(sub) else 0.0,
                "gross_sum": float(sub["gross_pnl_lev"].sum()) if len(sub) else 0.0,
                "win_rate": float(np.mean(sub["net_pnl_lev"].to_numpy(dtype=float) > 0.0)) if len(sub) else float("nan"),
                "maker_entry_ratio": float(np.mean(sub["entry_maker_flag"].astype(bool))) if len(sub) else float("nan"),
                "maker_exit_ratio": float(np.mean(sub["exit_maker_flag"].astype(bool))) if len(sub) else float("nan"),
                "short_share": short_share,
            }
        )
    return pd.DataFrame(rows)


def summarize_reason_stats_v5(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        cols = ["exit_reason", "trade_count", "net_sum", "avg_net", "win_rate", "hold_p50", "mfe_mean", "mae_mean"]
        return pd.DataFrame(columns=cols)
    rows: List[Dict[str, Any]] = []
    for reason, sub in trades.groupby("exit_reason", dropna=False):
        s = sub.copy()
        rows.append(
            {
                "exit_reason": str(reason),
                "trade_count": int(len(s)),
                "net_sum": float(s["net_pnl_lev"].sum()),
                "avg_net": float(s["net_pnl_lev"].mean()),
                "win_rate": float(np.mean(s["net_pnl_lev"].to_numpy(dtype=float) > 0.0)),
                "hold_p50": float(np.nanmedian(pd.to_numeric(s["hold_bars"], errors="coerce").to_numpy(dtype=float))),
                "mfe_mean": float(pd.to_numeric(s["mfe_n"], errors="coerce").mean()),
                "mae_mean": float(pd.to_numeric(s["mae_n"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["trade_count", "net_sum"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def summarize_trades_v5(
    trades: pd.DataFrame,
    *,
    equity: pd.DataFrame,
    segments: pd.DataFrame,
    sim_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    if trades is None or len(trades) == 0:
        return {
            "trade_count": 0,
            "gross_sum": 0.0,
            "net_sum": 0.0,
            "avg_net": 0.0,
            "win_rate": 0.0,
            "global_mdd": 0.0,
            "maker_entry_ratio": 0.0,
            "maker_exit_ratio": 0.0,
            "unexpected_taker_ratio": 0.0,
            "unfilled_entry_rate": float(sim_meta.get("n_entry_unfilled", 0)) / max(float(sim_meta.get("n_entry_signals", 0)), 1.0),
            "shock_exit_rate": 0.0,
            "max_hold_rate": 0.0,
            "hold_p50": 0.0,
            "hold_p90": 0.0,
            "tail_q05": 0.0,
            "regime_extreme_frac": 0.0,
            "short_share": 0.0,
        }
    pnl = pd.to_numeric(trades["net_pnl_lev"], errors="coerce").to_numpy(dtype=float)
    gross = pd.to_numeric(trades["gross_pnl_lev"], errors="coerce").to_numpy(dtype=float)
    dd = pd.to_numeric(equity["drawdown"], errors="coerce").to_numpy(dtype=float) if len(equity) else np.zeros(0, dtype=float)
    maker_entry_ratio = float(np.mean(trades["entry_maker_flag"].astype(bool)))
    maker_exit_ratio = float(np.mean(trades["exit_maker_flag"].astype(bool)))
    exit_reason = trades["exit_reason"].astype(str)
    hold = pd.to_numeric(trades["hold_bars"], errors="coerce").to_numpy(dtype=float)
    short_share = float(np.mean(pd.to_numeric(trades["side"], errors="coerce").to_numpy(dtype=float) < 0.0))
    regime_extreme_frac = float(np.mean(pd.to_numeric(trades["active_sparse_entry"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0.0))
    unexpected_taker_ratio = float(np.mean((~trades["entry_maker_flag"].astype(bool)) & (~trades["exit_maker_flag"].astype(bool))))
    return {
        "trade_count": int(len(trades)),
        "gross_sum": float(np.nansum(gross)),
        "net_sum": float(np.nansum(pnl)),
        "avg_net": float(np.nanmean(pnl)),
        "win_rate": float(np.nanmean(pnl > 0.0)),
        "global_mdd": float(np.nanmax(dd)) if len(dd) else 0.0,
        "maker_entry_ratio": maker_entry_ratio,
        "maker_exit_ratio": maker_exit_ratio,
        "unexpected_taker_ratio": unexpected_taker_ratio,
        "unfilled_entry_rate": float(sim_meta.get("n_entry_unfilled", 0)) / max(float(sim_meta.get("n_entry_signals", 0)), 1.0),
        "shock_exit_rate": float(np.mean(exit_reason == ExitReasonV5.SHOCK_EXIT.value)) if len(trades) else 0.0,
        "max_hold_rate": float(np.mean(exit_reason == ExitReasonV5.MAX_HOLD.value)) if len(trades) else 0.0,
        "hold_p50": float(np.nanquantile(hold, 0.50)) if len(hold) else float("nan"),
        "hold_p90": float(np.nanquantile(hold, 0.90)) if len(hold) else float("nan"),
        "tail_q05": float(np.nanquantile(pnl, 0.05)) if len(trades) else float("nan"),
        "seg_mean_net": float(pd.to_numeric(segments["net_sum"], errors="coerce").mean()) if len(segments) else float("nan"),
        "seg_worst2_mean": float(pd.to_numeric(segments["net_sum"], errors="coerce").nsmallest(2).mean()) if len(segments) else float("nan"),
        "regime_extreme_frac": regime_extreme_frac,
        "short_share": short_share,
    }


def assemble_objective_v5(
    *,
    overall: Mapping[str, Any],
    segments: pd.DataFrame,
    sim_meta: Mapping[str, Any],
    objective: ObjectiveConfigV5,
) -> Dict[str, Any]:
    seg_net = pd.to_numeric(segments.get("net_sum", pd.Series(dtype=float)), errors="coerce").astype(float)
    seg_trades = pd.to_numeric(segments.get("trade_count", pd.Series(dtype=float)), errors="coerce").astype(float)
    mean_seg_net = float(seg_net.mean()) if len(seg_net) else 0.0
    worst2_seg_net = float(seg_net.nsmallest(2).mean()) if len(seg_net) else 0.0
    global_mdd = _safe_float(overall.get("global_mdd", 0.0), 0.0)
    tail_q05 = _safe_float(overall.get("tail_q05", 0.0), 0.0)
    trade_count = int(max(0, _safe_int(overall.get("trade_count", 0), 0)))
    maker_entry_ratio = _safe_float(overall.get("maker_entry_ratio", 0.0), 0.0)
    maker_exit_ratio = _safe_float(overall.get("maker_exit_ratio", 0.0), 0.0)
    max_hold_rate = _safe_float(overall.get("max_hold_rate", 0.0), 0.0)
    unfilled_entry_rate = _safe_float(overall.get("unfilled_entry_rate", 0.0), 0.0)
    shock_exit_rate = _safe_float(overall.get("shock_exit_rate", 0.0), 0.0)
    regime_extreme_frac = _safe_float(overall.get("regime_extreme_frac", 0.0), 0.0)
    short_share = _safe_float(overall.get("short_share", 0.0), 0.0)

    trade_band_penalty = 0.0
    if trade_count < int(objective.target_trade_min):
        trade_band_penalty = float(objective.target_trade_min - trade_count) / max(float(objective.target_trade_min), 1.0)
    elif trade_count > int(objective.target_trade_max):
        trade_band_penalty = float(trade_count - objective.target_trade_max) / max(float(objective.target_trade_max), 1.0)

    min_trades_penalty = 1.0 if trade_count <= 0 else max(0.0, (float(objective.target_trade_min) - float(trade_count)) / max(float(objective.target_trade_min), 1.0))
    maker_ratio_penalty = max(0.0, float(objective.target_maker_entry_ratio) - maker_entry_ratio) + max(0.0, float(objective.target_maker_exit_ratio) - maker_exit_ratio)
    overhold_penalty = max_hold_rate
    tail_penalty = max(0.0, -tail_q05)

    trade_dispersion_penalty = 0.0
    bottom_k_penalty = 0.0
    if len(seg_trades) > 0:
        mean_tr = float(seg_trades.mean())
        std_tr = float(seg_trades.std(ddof=0))
        cv = std_tr / max(mean_tr, 1.0e-9) if np.isfinite(mean_tr) else 0.0
        trade_dispersion_penalty = max(0.0, cv)
        target_seg_floor = float(objective.target_trade_min) / max(float(max(len(seg_trades), 1)), 1.0)
        bottom2_mean_trades = float(seg_trades.nsmallest(2).mean()) if len(seg_trades) else 0.0
        if target_seg_floor > 0.0:
            bottom_k_penalty = max(0.0, target_seg_floor - bottom2_mean_trades) / max(target_seg_floor, 1.0)

    regime_extreme_penalty = max(0.0, regime_extreme_frac - float(objective.regime_extreme_max_frac))

    side_balance_penalty = 0.0
    short_trades = int(round(float(trade_count) * float(short_share)))
    if int(objective.min_short_trades) > 0 and short_trades < int(objective.min_short_trades):
        side_balance_penalty += float(objective.min_short_trades - short_trades) / max(float(objective.min_short_trades), 1.0)
    if float(objective.min_short_share) > 0.0 and short_share < float(objective.min_short_share):
        side_balance_penalty += max(0.0, float(objective.min_short_share) - short_share) / max(float(objective.min_short_share), 1.0e-9)

    score = (
        float(objective.mean_seg_weight) * mean_seg_net
        + float(objective.worst2_seg_weight) * worst2_seg_net
        - float(objective.mdd_penalty) * global_mdd
        - float(objective.tail_penalty) * tail_penalty
        - float(objective.trade_band_penalty) * trade_band_penalty
        - float(objective.min_trades_penalty) * min_trades_penalty
        - float(objective.maker_ratio_penalty) * maker_ratio_penalty
        - float(objective.overhold_penalty) * overhold_penalty
        - float(objective.unfilled_entry_penalty) * unfilled_entry_rate
        - float(objective.shock_exit_penalty) * shock_exit_rate
        - float(objective.trade_dispersion_penalty) * trade_dispersion_penalty
        - float(objective.bottom_k_penalty) * bottom_k_penalty
        - float(objective.regime_extreme_penalty_k) * regime_extreme_penalty
        - float(objective.side_balance_penalty_k) * side_balance_penalty
    )
    return {
        "score": float(score),
        "mean_seg_net": float(mean_seg_net),
        "worst2_seg_net": float(worst2_seg_net),
        "global_mdd": float(global_mdd),
        "tail_penalty_value": float(tail_penalty),
        "trade_band_penalty_value": float(trade_band_penalty),
        "min_trades_penalty_value": float(min_trades_penalty),
        "maker_ratio_penalty_value": float(maker_ratio_penalty),
        "overhold_penalty_value": float(overhold_penalty),
        "unfilled_entry_penalty_value": float(unfilled_entry_rate),
        "shock_exit_penalty_value": float(shock_exit_rate),
        "trade_dispersion_penalty_value": float(trade_dispersion_penalty),
        "bottom_k_penalty_value": float(bottom_k_penalty),
        "regime_extreme_penalty_value": float(regime_extreme_penalty),
        "side_balance_penalty_value": float(side_balance_penalty),
    }


def apply_cost_scenarios_v5(trades: pd.DataFrame, cost_scenarios: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if trades is None or len(trades) == 0 or not cost_scenarios:
        return {}
    out: Dict[str, Any] = {}
    tr = trades.copy()
    for idx, raw in enumerate(cost_scenarios):
        name = str(raw.get("name") or raw.get("label") or f"scenario_{idx}")
        taker = float(raw.get("taker", raw.get("cost_per_side", 0.00070)))
        maker = float(raw.get("maker", raw.get("maker_fee_per_side", 0.00020)))
        slip = float(raw.get("slip", raw.get("slip_per_side", 0.00015)))
        lev = pd.to_numeric(tr["lev"], errors="coerce").astype(float).to_numpy(copy=False)
        entry_maker = tr["entry_maker_flag"].astype(bool).to_numpy(copy=False)
        exit_maker = tr["exit_maker_flag"].astype(bool).to_numpy(copy=False)
        entry_side_fee = np.where(entry_maker, maker, taker + slip)
        exit_side_fee = np.where(exit_maker, maker, taker + slip)
        fee_total = (entry_side_fee + exit_side_fee) * lev
        gross = pd.to_numeric(tr["gross_pnl_lev"], errors="coerce").astype(float).to_numpy(copy=False)
        net = gross - fee_total
        out[name] = {
            "trade_count": int(len(tr)),
            "gross_sum": float(np.nansum(gross)),
            "net_sum": float(np.nansum(net)),
            "avg_net": float(np.nanmean(net)),
            "win_rate": float(np.nanmean(net > 0.0)),
            "taker_fee_side": float(taker + slip),
            "maker_fee_side": float(maker),
        }
    return out


def load_cost_scenarios_json(path: str = "") -> List[Dict[str, Any]]:
    if not str(path).strip():
        return []
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, Mapping)]
    if isinstance(payload, Mapping):
        return [dict(payload)]
    raise ValueError("cost scenarios json must be a dict or list[dict]")


__all__ = [
    "HAS_NUMBA",
    "ensure_prediction_frame_v5",
    "prepare_fast_eval_cache_v5",
    "save_fast_eval_cache_v5",
    "load_fast_eval_cache_v5",
    "prepare_trial_context_v5",
    "prepare_single_segment_fast_inputs_from_context_v5",
    "evaluate_prepared_single_segment_fast_v5",
    "warmup_single_fast_core_v5",
    "simulate_from_prediction_frame_v5",
    "evaluate_prediction_frame_fast_v5",
    "build_equity_curve_v5",
    "summarize_segments_v5",
    "summarize_reason_stats_v5",
    "summarize_trades_v5",
    "assemble_objective_v5",
    "apply_cost_scenarios_v5",
    "load_cost_scenarios_json",
]
