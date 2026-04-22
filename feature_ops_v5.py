# -*- coding: utf-8 -*-
"""
feature_ops_v5.py

V5 shared feature builder.

중요 원칙
---------
- raw -> feature only. 라벨/타겟을 만들지 않는다.
- row-wise Python for-loop 금지.
- 허용되는 반복은 작은 window tuple / feature group 메타데이터 반복뿐이다.
- 미래참조 방지: negative shift(-k)를 사용하지 않는다.
- true wick / rolling_vwap_dist_5,10 / strict taker_buy_ratio 규칙을 반영한다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from feature_contract_v5 import (
    ACTIVITY_FEATURES,
    ALL_DERIVED_FEATURE_COLUMNS,
    AUX_AUDIT_COLUMNS,
    CANDLE_MICRO_FEATURES,
    EXPECTED_BAR_NS,
    FEATURE_BUILD_OUTPUT_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    FEATURE_READY_COL,
    FLOW_STATE_FEATURES,
    GAP_AUDIT_COLUMNS,
    RAW_COLUMNS,
    RAW_INT_COLUMNS,
    RAW_NONNEGATIVE_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    REQUIRED_FEATURE_COLUMNS,
    REQUIRED_HISTORY_BARS,
    RETURN_FEATURES,
    VOLATILITY_FEATURES,
)

EPS = 1.0e-12
TAKER_RATIO_TOL = 1.0e-6


@dataclass(frozen=True)
class FeatureBuildConfig:
    mask_not_ready: bool = True
    cast_float32: bool = True
    preserve_extra_columns: bool = False
    compact_raw_float32: bool = False


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".csv", ".gz", ".bz2", ".xz"} or path.name.lower().endswith(".csv.gz"):
        return "csv"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    raise ValueError(f"unsupported file extension: {path}")


def read_frame(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    fmt = _detect_format(p)
    if fmt == "parquet":
        return pd.read_parquet(p)
    if fmt == "csv":
        return pd.read_csv(p, low_memory=False)
    return pd.read_pickle(p)


def write_frame(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = _detect_format(p)
    if fmt == "parquet":
        df.to_parquet(p, index=False)
        return
    if fmt == "csv":
        df.to_csv(p, index=False)
        return
    df.to_pickle(p)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _to_datetime_utc(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    return ts


def coerce_raw_frame(raw_df: pd.DataFrame, *, preserve_extra_columns: bool = False) -> pd.DataFrame:
    missing = [c for c in RAW_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"raw frame missing required columns: {missing}")

    if preserve_extra_columns:
        out = raw_df.copy()
    else:
        out = raw_df.loc[:, list(RAW_COLUMNS)].copy()

    out["timestamp"] = _to_datetime_utc(out["timestamp"])
    bad_ts = int(out["timestamp"].isna().sum())
    if bad_ts > 0:
        raise ValueError(f"raw frame contains {bad_ts} invalid timestamps; sanitize/split before feature build")

    for col in RAW_NUMERIC_COLUMNS:
        if col in RAW_INT_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out = (
        out.sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    return out


def build_gap_audit(timestamp: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    n = int(len(ts))
    if n == 0:
        return pd.DataFrame(
            {
                "gap_prev_min": pd.Series(dtype="float32"),
                "gap_prev_bad": pd.Series(dtype="bool"),
                "contig_run_id": pd.Series(dtype="int32"),
                "contig_pos": pd.Series(dtype="int32"),
            }
        )

    gap_prev_min = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        delta_min = ts.diff().dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
        gap_prev_min[1:] = delta_min[1:]

    gap_prev_bad = np.ones(n, dtype=bool)
    if n > 1:
        gap_prev_bad[1:] = ~np.isclose(gap_prev_min[1:], 1.0, rtol=0.0, atol=1e-9)

    contig_run_id = np.cumsum(gap_prev_bad.astype(np.int64)) - 1
    last_reset_idx = np.maximum.accumulate(np.where(gap_prev_bad, np.arange(n, dtype=np.int64), 0))
    contig_pos = np.arange(n, dtype=np.int64) - last_reset_idx

    return pd.DataFrame(
        {
            "gap_prev_min": gap_prev_min.astype("float32"),
            "gap_prev_bad": gap_prev_bad,
            "contig_run_id": contig_run_id.astype("int32"),
            "contig_pos": contig_pos.astype("int32"),
        },
        index=ts.index,
    )


def _safe_log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_safe = pd.Series(den, index=num.index, dtype="float64").replace(0.0, np.nan)
    num_safe = pd.Series(num, index=num.index, dtype="float64").replace(0.0, np.nan)
    return np.log(num_safe / den_safe)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den_safe = pd.Series(den, index=num.index, dtype="float64").replace(0.0, np.nan)
    return pd.Series(num, index=num.index, dtype="float64") / den_safe


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    window = int(window)
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr_rel(
    high: pd.Series,
    low: pd.Series,
    prev_close: pd.Series,
    close: pd.Series,
    *,
    gap_mask: pd.Series,
    window: int,
) -> pd.Series:
    tr = _true_range(high, low, prev_close).mask(gap_mask, np.nan)
    atr = tr.rolling(int(window), min_periods=int(window)).mean()
    return _safe_div(atr, close)


def _rolling_vwap_dist(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    window = int(window)
    typical = (high + low + close) / 3.0
    vol = volume.fillna(0.0)
    pv = typical * vol
    vol_sum = vol.rolling(window, min_periods=window).sum()
    pv_sum = pv.rolling(window, min_periods=window).sum()
    vwap = pv_sum / vol_sum.replace(0.0, np.nan)
    return _safe_log_ratio(close, vwap)


def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    window = int(window)
    change = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return change / path.replace(0.0, np.nan)


def _bb_pctb(close: pd.Series, window: int, *, num_std: float = 2.0) -> pd.Series:
    window = int(window)
    mid = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    low = mid - float(num_std) * sd
    high = mid + float(num_std) * sd
    width = (high - low).replace(0.0, np.nan)
    return (close - low) / width


# -----------------------------------------------------------------------------
# Feature build
# -----------------------------------------------------------------------------


def build_features(raw_df: pd.DataFrame, *, config: FeatureBuildConfig | None = None) -> pd.DataFrame:
    cfg = config or FeatureBuildConfig()
    raw = coerce_raw_frame(raw_df, preserve_extra_columns=cfg.preserve_extra_columns)
    audit = build_gap_audit(raw["timestamp"])

    open_ = raw["open"].astype("float64")
    high = raw["high"].astype("float64")
    low = raw["low"].astype("float64")
    close = raw["close"].astype("float64")
    volume_base = raw["volume_base"].astype("float64")
    taker_buy_base = raw["taker_buy_base"].astype("float64")

    close_safe = close.replace(0.0, np.nan)
    prev_close = close.shift(1)
    gap_mask = audit["gap_prev_bad"].astype(bool)
    contig_pos = audit["contig_pos"].astype("int64")

    feats: Dict[str, pd.Series] = {}

    # Candle / micro structure
    feats["gap_open"] = _safe_log_ratio(open_, prev_close)
    feats["high_ext"] = _safe_log_ratio(high, prev_close)
    feats["low_ext"] = _safe_log_ratio(low, prev_close)
    feats["body_ret"] = _safe_log_ratio(close, open_)

    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)).clip(lower=0.0)
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low).clip(lower=0.0)
    range_safe = (high - low).clip(lower=0.0).replace(0.0, np.nan)
    feats["upper_wick_rel"] = _safe_div(upper_wick, close_safe)
    feats["lower_wick_rel"] = _safe_div(lower_wick, close_safe)
    feats["wick_ratio"] = (upper_wick - lower_wick) / range_safe

    # Returns
    for w in (1, 3, 5, 8, 10):
        s = _safe_log_ratio(close, close.shift(w))
        feats[f"r{w}"] = s.mask(contig_pos < int(w), np.nan)

    # Volatility
    atr_cols: list[str] = []
    for w in (1, 3, 5, 8, 10):
        col = f"atr{w}_rel"
        atr_cols.append(col)
        feats[col] = _atr_rel(high, low, prev_close, close, gap_mask=gap_mask, window=w)

    # Activity
    for w in (3, 5, 8, 10):
        feats[f"vol_z_{w}"] = _rolling_zscore(volume_base, w).mask(contig_pos < (int(w) - 1), np.nan)

    # Flow / state
    invalid_ratio = (
        ~np.isfinite(volume_base)
        | ~np.isfinite(taker_buy_base)
        | (volume_base <= 0.0)
        | (taker_buy_base < 0.0)
        | (taker_buy_base > volume_base * (1.0 + TAKER_RATIO_TOL))
    )
    taker_buy_ratio = (taker_buy_base / volume_base.replace(0.0, np.nan)).mask(invalid_ratio, np.nan).clip(lower=0.0, upper=1.0)
    feats["taker_buy_ratio"] = taker_buy_ratio

    for w in (5, 10):
        feats[f"rolling_vwap_dist_{w}"] = _rolling_vwap_dist(high, low, close, volume_base, w).mask(
            contig_pos < (int(w) - 1), np.nan
        )

    feats["efficiency_ratio_10"] = _efficiency_ratio(close, 10).mask(contig_pos < 10, np.nan).clip(lower=0.0, upper=1.0)
    feats["bb_pctb_20"] = _bb_pctb(close, 20).mask(contig_pos < 19, np.nan)

    feature_df = pd.DataFrame(feats, index=raw.index).replace([np.inf, -np.inf], np.nan)

    # Explicit gap neutralization for prev_close-dependent features.
    prev_close_dep_cols = ["gap_open", "high_ext", "low_ext", *atr_cols]
    if prev_close_dep_cols:
        feature_df.loc[gap_mask.to_numpy(copy=False), prev_close_dep_cols] = np.nan

    # Defensive rolling-window neutralization by contiguous run length.
    for w in (3, 5, 8, 10):
        feature_df.loc[contig_pos.to_numpy(copy=False) < (w - 1), f"vol_z_{w}"] = np.nan
    for w in (5, 10):
        feature_df.loc[contig_pos.to_numpy(copy=False) < (w - 1), f"rolling_vwap_dist_{w}"] = np.nan
    feature_df.loc[contig_pos.to_numpy(copy=False) < 10, "efficiency_ratio_10"] = np.nan
    feature_df.loc[contig_pos.to_numpy(copy=False) < 19, "bb_pctb_20"] = np.nan

    mask_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c in feature_df.columns]
    finite_ready = feature_df.loc[:, mask_cols].notna().all(axis=1).to_numpy(copy=False)
    ready = (
        (contig_pos.to_numpy(copy=False) >= int(REQUIRED_HISTORY_BARS - 1))
        & (~gap_mask.to_numpy(copy=False))
        & np.isfinite(volume_base.to_numpy(copy=False))
        & (volume_base.to_numpy(copy=False) > 0.0)
        & np.isfinite(feature_df["taker_buy_ratio"].to_numpy(copy=False))
        & finite_ready
    )

    if cfg.mask_not_ready and mask_cols:
        feature_df.loc[~ready, mask_cols] = np.nan
        finite_ready = feature_df.loc[:, mask_cols].notna().all(axis=1).to_numpy(copy=False)
        ready = (
            (contig_pos.to_numpy(copy=False) >= int(REQUIRED_HISTORY_BARS - 1))
            & (~gap_mask.to_numpy(copy=False))
            & np.isfinite(volume_base.to_numpy(copy=False))
            & (volume_base.to_numpy(copy=False) > 0.0)
            & np.isfinite(feature_df["taker_buy_ratio"].to_numpy(copy=False))
            & finite_ready
        )

    if cfg.cast_float32:
        feature_df = feature_df.astype("float32", copy=False)

    if cfg.preserve_extra_columns:
        out = raw.copy()
        regenerated_cols = set(GAP_AUDIT_COLUMNS) | {FEATURE_READY_COL} | set(feature_df.columns)
        drop_cols = [c for c in out.columns if c in regenerated_cols]
        if drop_cols:
            out = out.drop(columns=drop_cols, errors="ignore")
    else:
        out = raw.loc[:, list(RAW_COLUMNS)].copy()

    if cfg.compact_raw_float32:
        for c in RAW_NUMERIC_COLUMNS:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    out = pd.concat([out, audit], axis=1)
    out[FEATURE_READY_COL] = ready
    out = pd.concat([out, feature_df], axis=1)

    cols = pd.Index(out.columns)
    if cols.has_duplicates:
        out = out.loc[:, ~cols.duplicated(keep="last")].copy()

    expected_cols = [c for c in FEATURE_BUILD_OUTPUT_COLUMNS if c in out.columns]
    if cfg.preserve_extra_columns:
        extra_cols = [c for c in out.columns if c not in expected_cols]
        out = out.loc[:, expected_cols + extra_cols]
    else:
        out = out.loc[:, expected_cols]
    return out


# -----------------------------------------------------------------------------
# Summary / CLI
# -----------------------------------------------------------------------------


def summarize_feature_frame(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["feature_contract_version"] = FEATURE_CONTRACT_VERSION
    out["rows"] = int(len(df))
    if len(df):
        out["time_start_utc"] = pd.Timestamp(df["timestamp"].iloc[0]).isoformat()
        out["time_end_utc"] = pd.Timestamp(df["timestamp"].iloc[-1]).isoformat()
    else:
        out["time_start_utc"] = None
        out["time_end_utc"] = None

    gap_mask = df["gap_prev_bad"].astype(bool) if "gap_prev_bad" in df.columns else pd.Series(dtype=bool)
    gap_prev_min = pd.to_numeric(df.get("gap_prev_min"), errors="coerce") if "gap_prev_min" in df.columns else pd.Series(dtype="float64")
    real_gap_mask = gap_mask.copy()
    if len(real_gap_mask):
        real_gap_mask.iloc[0] = False

    out["gap_count_excluding_first"] = int(real_gap_mask.sum())
    out["max_gap_min"] = float(gap_prev_min[real_gap_mask].max()) if len(gap_prev_min[real_gap_mask]) else 0.0

    ready_s = df.get(FEATURE_READY_COL, pd.Series(dtype=bool))
    out["ready_rows"] = int(ready_s.sum())
    out["ready_frac"] = float(ready_s.mean()) if len(df) else 0.0

    feat_nan: Dict[str, float] = {}
    for c in [col for col in REQUIRED_FEATURE_COLUMNS if col in df.columns]:
        feat_nan[c] = float(pd.to_numeric(df[c], errors="coerce").isna().mean())
    out["top_feature_nan_rates"] = dict(sorted(feat_nan.items(), key=lambda kv: kv[1], reverse=True)[:10])
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build v5 reactive26 features from raw Binance USD-M futures bars")
    ap.add_argument("--input", required=True, help="raw input file (.parquet/.csv/.pkl)")
    ap.add_argument("--output", required=True, help="feature output file (.parquet/.csv/.pkl)")
    ap.add_argument("--mask-not-ready", type=int, default=1, help="1=mask early/gap-affected rows to NaN")
    ap.add_argument("--cast-float32", type=int, default=1, help="1=cast derived features to float32")
    ap.add_argument("--preserve-extra-columns", type=int, default=0, help="1=keep non-contract extra raw columns")
    ap.add_argument("--compact-raw-float32", type=int, default=0, help="1=downcast raw numeric columns too")
    ap.add_argument("--report-json", default="", help="optional summary json output path")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    raw = read_frame(args.input)
    built = build_features(
        raw,
        config=FeatureBuildConfig(
            mask_not_ready=bool(int(args.mask_not_ready)),
            cast_float32=bool(int(args.cast_float32)),
            preserve_extra_columns=bool(int(args.preserve_extra_columns)),
            compact_raw_float32=bool(int(args.compact_raw_float32)),
        ),
    )
    write_frame(built, args.output)
    summary = summarize_feature_frame(built)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
