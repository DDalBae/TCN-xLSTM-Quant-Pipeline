
# -*- coding: utf-8 -*-
"""
inference_v5.py

V5 inference aligned with:
- feature_contract_v5.py
- feature_ops_v5.py
- target_contract_v5.py
- label_builder_v5_1.py
- trainer_v5.py
- model_v5_1.py

Role
----
- checkpoint + scaler load
- raw/features/dataset 입력에서 reactive26 feature frame 확보
- sequence batch inference
- 예측값을 행 단위 DataFrame으로 복원
- multi-horizon `dir + hyb + utility + retcls` 기반 entry composite 복원
- path10 / utility_10 / first_hit / tth 기반 thesis gate helper 제공
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from feature_contract_v5 import (
    DEFAULT_MODEL_INPUT_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    RAW_COLUMNS,
)
from feature_ops_v5 import FeatureBuildConfig, build_features, read_frame, write_frame
from model_v5_1 import (
    BIN_MAIN_TARGETS,
    DIR_MAIN_TARGETS,
    FIRST_HIT_NUM_CLASSES,
    FIRST_HIT_TARGETS,
    PATH_MAIN_TARGETS,
    PathDistMultiHeadV5_1,
    RETCLS_NUM_CLASSES,
    RETCLS_TARGETS,
    RET_MAIN_TARGETS,
    TTH_NUM_CLASSES,
    TTH_TARGETS,
    UTILITY_MAIN_TARGETS,
    build_model_from_meta,
)
from target_contract_v5 import TARGET_SCALE_REF_FEATURE
from trainer_v5 import (
    BIN_TARGET_TO_IDX,
    DIR_HORIZON_TO_IDX,
    EntryScoringConfig,
    FIRST_TARGET_TO_IDX,
    FIRST_HIT_TARGET_100,
    INVALID_CLASS_VALUE,
    MAIN_CONFIRM_TARGET_DOWN,
    MAIN_CONFIRM_TARGET_UP,
    RETCLS_HORIZON_TO_IDX,
    RETCLS_SUPPORT,
    RET_HORIZON_TO_IDX,
    TTH_TARGET_DOWN_100,
    TTH_TARGET_TO_IDX,
    TTH_TARGET_UP_100,
    UTILITY_LONG_HORIZON_TO_IDX,
    UTILITY_SHORT_HORIZON_TO_IDX,
    _barrier_token,
    build_entry_composites_numpy,
    build_sequence_starts,
    normalize_horizon_weights,
    transform_numpy_with_scaler,
)

REQUIRED_RAW_SIM_COLUMNS: Tuple[str, ...] = ("timestamp", "open", "high", "low", "close")

PRED_COLS_REQUIRED_BASE: Tuple[str, ...] = (
    "pred_ready",
    "pred_scale_ref_t",
    "predentry_core",
    "predentry_gap",
    "predentry_side",
    "pred_tgt_long_utility_10",
    "pred_tgt_short_utility_10",
    "pred_tgt_up_excur_10_n",
    "pred_tgt_down_excur_10_n",
    "predprob_tgt_up_hit_100_10",
    "predprob_tgt_down_hit_100_10",
    "predprob_up_tgt_first_hit_100_10",
    "predprob_down_tgt_first_hit_100_10",
    "predexp_tgt_tth_up_100_10",
    "predexp_tgt_tth_down_100_10",
    "predprob_censored_tgt_tth_up_100_10",
    "predprob_censored_tgt_tth_down_100_10",
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceConfig:
    input_kind: str = "auto"
    batch_size: int = 512
    device: str = "auto"
    preserve_input_columns: bool = True


@dataclass(frozen=True)
class PolicyConfig:
    # entry composite weights
    w1: float = 0.28
    w3: float = 0.28
    w5: float = 0.20
    w8: float = 0.14
    w10: float = 0.10

    hyb_weight: float = 0.55
    util_weight: float = 0.30
    cls_weight: float = 0.15

    entry_min_score: float = 0.00
    entry_min_gap: float = 0.00

    entry_min_utility_10: float = 0.05
    entry_min_utility_gap_10: float = 0.00

    confirm_main_barrier: float = 1.0
    confirm_main_prob: float = 0.50

    timing_barrier: float = 1.0
    timing_first_hit_prob: float = 0.34
    timing_max_expected_bars: float = 8.0
    timing_max_censored_prob: float = 0.60

    require_retcls_alignment: bool = False

    min_tp_n: float = 0.60
    min_sl_n: float = 0.50
    max_tp_n: float = 2.50
    max_sl_n: float = 2.00
    hard_max_hold_bars: int = 10


def _policy_to_entry_config(policy: PolicyConfig) -> EntryScoringConfig:
    return EntryScoringConfig(
        w1=float(policy.w1),
        w3=float(policy.w3),
        w5=float(policy.w5),
        w8=float(policy.w8),
        w10=float(policy.w10),
        hyb_weight=float(policy.hyb_weight),
        util_weight=float(policy.util_weight),
        cls_weight=float(policy.cls_weight),
        entry_min_score=float(policy.entry_min_score),
        entry_min_gap=float(policy.entry_min_gap),
        entry_min_utility_10=float(policy.entry_min_utility_10),
        entry_min_utility_gap_10=float(policy.entry_min_utility_gap_10),
        confirm_main_prob=float(policy.confirm_main_prob),
        timing_first_hit_prob=float(policy.timing_first_hit_prob),
        timing_max_expected_bars=float(policy.timing_max_expected_bars),
        timing_max_censored_prob=float(policy.timing_max_censored_prob),
        require_retcls_alignment=bool(policy.require_retcls_alignment),
    )


def _resolve_policy_config(
    *,
    json_path: str = "",
    inline_json: str = "",
    overrides: Optional[Mapping[str, Any]] = None,
) -> PolicyConfig:
    raw = asdict(PolicyConfig())
    if str(json_path).strip():
        raw.update(json.loads(Path(json_path).read_text(encoding="utf-8")))
    if str(inline_json).strip():
        raw.update(json.loads(str(inline_json)))
    if overrides:
        for k, v in overrides.items():
            if v is not None and k in raw:
                raw[k] = v
    kwargs: Dict[str, Any] = {}
    for f in fields(PolicyConfig):
        v = raw.get(f.name, getattr(PolicyConfig(), f.name))
        if f.type is bool:
            kwargs[f.name] = bool(v)
        elif f.type is int:
            kwargs[f.name] = int(v)
        else:
            kwargs[f.name] = float(v)
    return PolicyConfig(**kwargs)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if not np.isfinite(out):
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _load_checkpoint(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(p, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"unsupported checkpoint payload: {type(ckpt)}")
    return ckpt


def _resolve_input_kind(df: pd.DataFrame) -> str:
    has_raw = all(c in df.columns for c in RAW_COLUMNS)
    has_feat = all(c in df.columns for c in DEFAULT_MODEL_INPUT_COLUMNS)
    if has_raw and has_feat:
        return "features"
    if has_raw:
        return "raw"
    raise ValueError("input frame is neither raw nor feature/dataset frame")


def _dedupe_columns_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Index(df.columns)
    if not cols.has_duplicates:
        return df
    return df.loc[:, ~cols.duplicated(keep="last")].copy()


def compute_scale_ref_from_frame(df: pd.DataFrame, *, cost_per_side: float, slip_per_side: float) -> np.ndarray:
    fee_floor = 2.0 * (float(cost_per_side) + float(slip_per_side))
    atr = pd.to_numeric(df.get(TARGET_SCALE_REF_FEATURE), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    out = np.maximum.reduce(
        [
            np.nan_to_num(atr, nan=0.0, posinf=0.0, neginf=0.0),
            np.full(len(df), fee_floor, dtype=np.float64),
            np.full(len(df), 1.0e-6, dtype=np.float64),
        ]
    )
    return out.astype(np.float32)


def ensure_prediction_frame_v5(pred: pd.DataFrame) -> pd.DataFrame:
    df = pred.copy()
    cols = pd.Index(df.columns)
    if cols.has_duplicates:
        df = df.loc[:, ~cols.duplicated(keep="last")].copy()
    missing_raw = [c for c in REQUIRED_RAW_SIM_COLUMNS if c not in df.columns]
    missing_pred = [c for c in PRED_COLS_REQUIRED_BASE if c not in df.columns]
    if missing_raw:
        raise ValueError(f"prediction frame missing required raw columns: {missing_raw}")
    if missing_pred:
        raise ValueError(f"prediction frame missing required prediction columns: {missing_pred}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------


class PathDistInferenceV5:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        scaler_json: str | Path | None = None,
        device: str = "auto",
    ):
        ckpt = _load_checkpoint(checkpoint_path)
        meta = ckpt.get("meta") if isinstance(ckpt.get("meta"), Mapping) else ckpt.get("model_meta", {})
        if not isinstance(meta, Mapping):
            raise TypeError("checkpoint missing model meta mapping")
        self.meta = dict(meta)

        features = list(self.meta.get("features") or DEFAULT_MODEL_INPUT_COLUMNS)
        self.features = list(features)
        self.seq_len = int(self.meta.get("seq_len", 160))
        self.cost_per_side = float(self.meta.get("cost_per_side", 0.00070))
        self.slip_per_side = float(self.meta.get("slip_per_side", 0.00015))

        if scaler_json:
            self.scaler = json.loads(Path(scaler_json).read_text(encoding="utf-8"))
        else:
            scaler_ckpt = ckpt.get("scaler")
            if not isinstance(scaler_ckpt, Mapping):
                raise ValueError("scaler is required: pass --scaler-json or use a checkpoint containing 'scaler'")
            self.scaler = dict(scaler_ckpt)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(str(device))

        self.model: PathDistMultiHeadV5_1 = build_model_from_meta(self.meta).to(self.device)
        state = ckpt.get("state_dict", ckpt)
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint does not contain a valid state_dict")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        sel_cfg_raw = self.meta.get("selection_config")
        if isinstance(sel_cfg_raw, Mapping):
            self.default_policy = _resolve_policy_config(inline_json=json.dumps(sel_cfg_raw))
        else:
            self.default_policy = PolicyConfig()

    def _ensure_feature_frame(self, df: pd.DataFrame, *, input_kind: str = "auto") -> pd.DataFrame:
        kind = str(input_kind or "auto").strip().lower()
        if kind == "auto":
            kind = _resolve_input_kind(df)
        if kind == "raw":
            feat = build_features(
                df,
                config=FeatureBuildConfig(
                    mask_not_ready=True,
                    cast_float32=True,
                    preserve_extra_columns=True,
                ),
            )
            return _dedupe_columns_keep_last(feat)
        if kind in {"features", "dataset"}:
            return _dedupe_columns_keep_last(df.copy())
        raise ValueError(f"unsupported input_kind={input_kind}")

    def _init_prediction_frame(self, feat_df: pd.DataFrame, *, preserve_input_columns: bool) -> pd.DataFrame:
        out = feat_df.copy() if preserve_input_columns else pd.DataFrame(index=feat_df.index)
        n = int(len(feat_df))
        nanf = np.full(n, np.nan, dtype=np.float32)
        new_cols: Dict[str, Any] = {"pred_ready": np.zeros(n, dtype=bool)}

        for col in RET_MAIN_TARGETS:
            new_cols[f"predraw_{col}"] = nanf.copy()
            new_cols[f"predmag_{col}"] = nanf.copy()
            new_cols[f"predhy_{col}"] = nanf.copy()
            new_cols[f"pred_{col}"] = nanf.copy()  # primary signed return proxy = hyb
        for col in DIR_MAIN_TARGETS:
            new_cols[f"predlogit_{col}"] = nanf.copy()
            new_cols[f"predprob_{col}"] = nanf.copy()
            new_cols[f"predscore_{col}"] = nanf.copy()
        for col in PATH_MAIN_TARGETS:
            new_cols[f"pred_{col}"] = nanf.copy()
        for col in UTILITY_MAIN_TARGETS:
            new_cols[f"pred_{col}"] = nanf.copy()
        for col in BIN_MAIN_TARGETS:
            new_cols[f"predprob_{col}"] = nanf.copy()
        for col in RETCLS_TARGETS:
            new_cols[f"predcls_{col}"] = nanf.copy()
            new_cols[f"predscore_{col}"] = nanf.copy()
            new_cols[f"predprobmax_{col}"] = nanf.copy()
        for col in FIRST_HIT_TARGETS:
            new_cols[f"predcls_{col}"] = nanf.copy()
            new_cols[f"predprob_none_{col}"] = nanf.copy()
            new_cols[f"predprob_up_{col}"] = nanf.copy()
            new_cols[f"predprob_down_{col}"] = nanf.copy()
        for col in TTH_TARGETS:
            new_cols[f"predcls_{col}"] = nanf.copy()
            new_cols[f"predexp_{col}"] = nanf.copy()
            new_cols[f"predprob_censored_{col}"] = nanf.copy()

        scale_ref = compute_scale_ref_from_frame(
            feat_df,
            cost_per_side=float(self.cost_per_side),
            slip_per_side=float(self.slip_per_side),
        )
        fee_floor = np.full(n, 2.0 * (float(self.cost_per_side) + float(self.slip_per_side)), dtype=np.float32)
        fee_n = fee_floor / np.where(scale_ref > 0.0, scale_ref, np.nan)
        new_cols["pred_scale_ref_t"] = scale_ref.astype(np.float32, copy=False)
        new_cols["pred_fee_floor_ret_t"] = fee_floor
        new_cols["pred_fee_n_t"] = fee_n.astype(np.float32, copy=False)

        for col in (
            "predentry_long_core",
            "predentry_short_core",
            "predentry_core",
            "predentry_gap",
            "predentry_side",
            "predhy_mix",
            "predcls_mix",
            "predutil_long_mix",
            "predutil_short_mix",
            "predutility_10_side",
            "predutility_10_gap",
        ):
            new_cols[col] = nanf.copy()

        return pd.concat([out, pd.DataFrame(new_cols, index=feat_df.index)], axis=1)

    def _apply_entry_derivations(self, out: pd.DataFrame, *, policy: PolicyConfig) -> pd.DataFrame:
        ready = pd.Series(out.get("pred_ready", False)).fillna(False).astype(bool).to_numpy(copy=False)
        if not np.any(ready):
            return out

        hyb = out.loc[:, [f"pred_{name}" for name in RET_MAIN_TARGETS]].to_numpy(dtype=np.float32, copy=False)
        util = out.loc[:, [f"pred_{name}" for name in UTILITY_MAIN_TARGETS]].to_numpy(dtype=np.float32, copy=False)
        cls = out.loc[:, [f"predscore_{name}" for name in RETCLS_TARGETS]].to_numpy(dtype=np.float32, copy=False)

        entry = build_entry_composites_numpy(hyb, util, cls, config=_policy_to_entry_config(policy))
        out["predentry_long_core"] = entry["entry_long_core"]
        out["predentry_short_core"] = entry["entry_short_core"]
        out["predentry_core"] = entry["entry_core"]
        out["predentry_gap"] = entry["entry_gap"]
        out["predentry_side"] = entry["entry_side"]
        out["predhy_mix"] = entry["hyb_mix"]
        out["predcls_mix"] = entry["cls_mix"]
        out["predutil_long_mix"] = entry["util_long_mix"]
        out["predutil_short_mix"] = entry["util_short_mix"]

        util10_long = out[f"pred_tgt_long_utility_10"].to_numpy(dtype=np.float32, copy=False)
        util10_short = out[f"pred_tgt_short_utility_10"].to_numpy(dtype=np.float32, copy=False)
        side = out["predentry_side"].to_numpy(dtype=np.float32, copy=False)
        out["predutility_10_side"] = np.where(side > 0, util10_long, util10_short)
        out["predutility_10_gap"] = np.where(side > 0, util10_long - util10_short, util10_short - util10_long)
        return out

    @torch.inference_mode()
    def predict_frame(
        self,
        df: pd.DataFrame,
        *,
        input_kind: str = "auto",
        batch_size: int = 512,
        preserve_input_columns: bool = True,
        policy: Optional[PolicyConfig] = None,
    ) -> pd.DataFrame:
        feat_df = self._ensure_feature_frame(df, input_kind=input_kind)
        missing = [c for c in self.features if c not in feat_df.columns]
        if missing:
            raise ValueError(f"feature frame missing required model inputs: {missing}")

        x_raw = feat_df.loc[:, self.features].to_numpy(dtype=np.float32, copy=False)
        x_scaled = transform_numpy_with_scaler(x_raw, self.scaler, self.features)

        scaled_frame = feat_df.copy()
        for j, c in enumerate(self.features):
            scaled_frame[c] = x_scaled[:, j]
        starts, ends = build_sequence_starts(scaled_frame, self.features, seq_len=self.seq_len, require_ready_col="__row_valid__")

        out = self._init_prediction_frame(feat_df, preserve_input_columns=bool(preserve_input_columns))
        if len(starts) == 0:
            return out

        tth_support = np.arange(1, TTH_NUM_CLASSES + 1, dtype=np.float32)

        for i in range(0, len(starts), int(batch_size)):
            s_batch = starts[i : i + int(batch_size)]
            e_batch = ends[i : i + int(batch_size)]
            seq_batch = np.stack([x_scaled[s : s + self.seq_len] for s in s_batch], axis=0).astype(np.float32, copy=False)
            xb = torch.from_numpy(seq_batch).to(self.device)
            outputs = self.model(xb)

            ret_raw = outputs["ret_main"].detach().cpu().numpy()
            ret_mag = outputs["ret_main_decoded"].detach().cpu().numpy()
            hyb = outputs["hyb_main"].detach().cpu().numpy()
            dir_logits = outputs["dir_main"].detach().cpu().numpy()
            dir_prob = torch.sigmoid(outputs["dir_main"]).detach().cpu().numpy()
            dir_score = outputs["dir_score"].detach().cpu().numpy()
            path_main = outputs["path_main"].detach().cpu().numpy()
            util_main = outputs["util_main"].detach().cpu().numpy()

            for j, col in enumerate(RET_MAIN_TARGETS):
                out.loc[e_batch, f"predraw_{col}"] = ret_raw[:, j]
                out.loc[e_batch, f"predmag_{col}"] = ret_mag[:, j]
                out.loc[e_batch, f"predhy_{col}"] = hyb[:, j]
                out.loc[e_batch, f"pred_{col}"] = hyb[:, j]
            for j, col in enumerate(DIR_MAIN_TARGETS):
                out.loc[e_batch, f"predlogit_{col}"] = dir_logits[:, j]
                out.loc[e_batch, f"predprob_{col}"] = dir_prob[:, j]
                out.loc[e_batch, f"predscore_{col}"] = dir_score[:, j]
            for j, col in enumerate(PATH_MAIN_TARGETS):
                out.loc[e_batch, f"pred_{col}"] = path_main[:, j]
            for j, col in enumerate(UTILITY_MAIN_TARGETS):
                out.loc[e_batch, f"pred_{col}"] = util_main[:, j]

            bin_main = torch.sigmoid(outputs["bin_main"]).detach().cpu().numpy()
            for j, col in enumerate(BIN_MAIN_TARGETS):
                out.loc[e_batch, f"predprob_{col}"] = bin_main[:, j]

            retcls_logits = outputs["retcls"].detach().cpu()
            retcls_prob = torch.softmax(retcls_logits, dim=-1).numpy()
            retcls_cls = retcls_prob.argmax(axis=-1)
            retcls_score = (retcls_prob * RETCLS_SUPPORT.reshape(1, 1, -1)).sum(axis=-1)
            retcls_probmax = retcls_prob.max(axis=-1)
            for j, col in enumerate(RETCLS_TARGETS):
                out.loc[e_batch, f"predcls_{col}"] = retcls_cls[:, j]
                out.loc[e_batch, f"predscore_{col}"] = retcls_score[:, j]
                out.loc[e_batch, f"predprobmax_{col}"] = retcls_probmax[:, j]

            fh_logits = outputs["first_hit"].detach().cpu()
            fh_prob = torch.softmax(fh_logits, dim=-1).numpy()
            fh_cls = fh_prob.argmax(axis=-1)
            for j, col in enumerate(FIRST_HIT_TARGETS):
                out.loc[e_batch, f"predcls_{col}"] = fh_cls[:, j]
                out.loc[e_batch, f"predprob_none_{col}"] = fh_prob[:, j, 0]
                out.loc[e_batch, f"predprob_up_{col}"] = fh_prob[:, j, 1]
                out.loc[e_batch, f"predprob_down_{col}"] = fh_prob[:, j, 2]
                # v4-style convenience aliases for future backtest helpers
                out.loc[e_batch, f"predprob_up_{col}"] = fh_prob[:, j, 1]
                out.loc[e_batch, f"predprob_down_{col}"] = fh_prob[:, j, 2]

            tth_logits = outputs["tth"].detach().cpu()
            tth_prob = torch.softmax(tth_logits, dim=-1).numpy()
            tth_cls = tth_prob.argmax(axis=-1) + 1
            tth_exp = (tth_prob * tth_support.reshape(1, 1, -1)).sum(axis=-1)
            for j, col in enumerate(TTH_TARGETS):
                out.loc[e_batch, f"predcls_{col}"] = tth_cls[:, j]
                out.loc[e_batch, f"predexp_{col}"] = tth_exp[:, j]
                out.loc[e_batch, f"predprob_censored_{col}"] = tth_prob[:, j, -1]

            out.loc[e_batch, "pred_ready"] = True

        out = self._apply_entry_derivations(out, policy=(policy or self.default_policy))
        return out


# ---------------------------------------------------------------------------
# Trade-plan helper
# ---------------------------------------------------------------------------


def _empty_trade_plan_diag() -> Dict[str, Any]:
    return {
        "plan": None,
        "policy_gate_passed": False,
        "gate_fail_stage": "",
        "gate_fail_detail": "",
        "gate_fail_value": float("nan"),
        "gate_fail_threshold": float("nan"),
        "side": 0,
        "entry_core": float("nan"),
        "entry_gap": float("nan"),
        "entry_long_core": float("nan"),
        "entry_short_core": float("nan"),
        "hyb_mix": float("nan"),
        "cls_mix": float("nan"),
        "util_long_mix": float("nan"),
        "util_short_mix": float("nan"),
        "utility_10": float("nan"),
        "utility_gap_10": float("nan"),
        "class_score_10": float("nan"),
        "main_confirm_prob": float("nan"),
        "timing_first_hit_prob": float("nan"),
        "timing_expected_bars": float("nan"),
        "timing_censored_prob": float("nan"),
        "tp_n": float("nan"),
        "sl_n": float("nan"),
        "tp_ret": float("nan"),
        "sl_ret": float("nan"),
        "max_hold_bars": 0,
        "scale_ref_t": float("nan"),
        "retcls_alignment_state": "not_checked",
    }


def _compute_entry_from_row(row: Mapping[str, Any], *, policy: PolicyConfig) -> Dict[str, float]:
    long_core = _safe_float(row.get("predentry_long_core"), float("nan"))
    short_core = _safe_float(row.get("predentry_short_core"), float("nan"))
    entry_core = _safe_float(row.get("predentry_core"), float("nan"))
    entry_gap = _safe_float(row.get("predentry_gap"), float("nan"))
    side = _safe_float(row.get("predentry_side"), float("nan"))
    hyb_mix = _safe_float(row.get("predhy_mix"), float("nan"))
    cls_mix = _safe_float(row.get("predcls_mix"), float("nan"))
    util_long_mix = _safe_float(row.get("predutil_long_mix"), float("nan"))
    util_short_mix = _safe_float(row.get("predutil_short_mix"), float("nan"))

    if all(np.isfinite([long_core, short_core, entry_core, entry_gap, side])):
        return {
            "entry_long_core": float(long_core),
            "entry_short_core": float(short_core),
            "entry_core": float(entry_core),
            "entry_gap": float(entry_gap),
            "entry_side": int(np.sign(side) if side != 0 else 1),
            "hyb_mix": float(hyb_mix) if np.isfinite(hyb_mix) else float("nan"),
            "cls_mix": float(cls_mix) if np.isfinite(cls_mix) else float("nan"),
            "util_long_mix": float(util_long_mix) if np.isfinite(util_long_mix) else float("nan"),
            "util_short_mix": float(util_short_mix) if np.isfinite(util_short_mix) else float("nan"),
        }

    weights = normalize_horizon_weights(_policy_to_entry_config(policy))
    hyb = np.asarray([_safe_float(row.get(f"pred_tgt_ret_{h}_n"), float("nan")) for h in (1, 3, 5, 8, 10)], dtype=np.float32)
    cls = np.asarray([_safe_float(row.get(f"predscore_tgt_retcls_{h}"), float("nan")) for h in (1, 3, 5, 8, 10)], dtype=np.float32)
    util_long = np.asarray([_safe_float(row.get(f"pred_tgt_long_utility_{h}"), float("nan")) for h in (1, 3, 5, 8, 10)], dtype=np.float32)
    util_short = np.asarray([_safe_float(row.get(f"pred_tgt_short_utility_{h}"), float("nan")) for h in (1, 3, 5, 8, 10)], dtype=np.float32)

    long_core = short_core = hyb_mix = cls_mix = util_long_mix = util_short_mix = 0.0
    for i, h in enumerate((1, 3, 5, 8, 10)):
        w = float(weights[int(h)])
        hy = float(hyb[i])
        cl = float(cls[i])
        ul = float(util_long[i])
        us = float(util_short[i])

        hyb_mix += w * hy
        cls_mix += w * cl
        util_long_mix += w * ul
        util_short_mix += w * us

        long_core += w * (
            float(policy.hyb_weight) * max(hy, 0.0)
            + float(policy.util_weight) * max(ul, 0.0)
            + float(policy.cls_weight) * max(cl, 0.0)
        )
        short_core += w * (
            float(policy.hyb_weight) * max(-hy, 0.0)
            + float(policy.util_weight) * max(us, 0.0)
            + float(policy.cls_weight) * max(-cl, 0.0)
        )

    side = 1 if long_core >= short_core else -1
    entry_core = long_core if side > 0 else short_core
    other_core = short_core if side > 0 else long_core
    entry_gap = entry_core - other_core
    return {
        "entry_long_core": float(long_core),
        "entry_short_core": float(short_core),
        "entry_core": float(entry_core),
        "entry_gap": float(entry_gap),
        "entry_side": int(side),
        "hyb_mix": float(hyb_mix),
        "cls_mix": float(cls_mix),
        "util_long_mix": float(util_long_mix),
        "util_short_mix": float(util_short_mix),
    }


def diagnose_trade_plan_from_row(row: Mapping[str, Any], *, policy: PolicyConfig) -> Dict[str, Any]:
    diag = _empty_trade_plan_diag()

    def _fail(stage: str, detail: str, value: float = float("nan"), threshold: float = float("nan")) -> Dict[str, Any]:
        diag["policy_gate_passed"] = False
        diag["gate_fail_stage"] = str(stage)
        diag["gate_fail_detail"] = str(detail)
        diag["gate_fail_value"] = float(value) if np.isfinite(value) else float("nan")
        diag["gate_fail_threshold"] = float(threshold) if np.isfinite(threshold) else float("nan")
        return diag

    if not bool(row.get("pred_ready", False)):
        return _fail("pred_not_ready", "row pred_ready == False")

    scale_ref = _safe_float(row.get("pred_scale_ref_t"), 0.0)
    diag["scale_ref_t"] = float(scale_ref) if np.isfinite(scale_ref) else float("nan")
    if not np.isfinite(scale_ref) or scale_ref <= 0.0:
        return _fail("scale_ref_invalid", "pred_scale_ref_t is not finite or <= 0", scale_ref, 0.0)

    entry = _compute_entry_from_row(row, policy=policy)
    side = int(entry["entry_side"])
    entry_core = float(entry["entry_core"])
    entry_gap = float(entry["entry_gap"])
    diag.update({
        "side": int(side),
        "entry_core": float(entry_core),
        "entry_gap": float(entry_gap),
        "entry_long_core": float(entry["entry_long_core"]),
        "entry_short_core": float(entry["entry_short_core"]),
        "hyb_mix": float(entry["hyb_mix"]),
        "cls_mix": float(entry["cls_mix"]),
        "util_long_mix": float(entry["util_long_mix"]),
        "util_short_mix": float(entry["util_short_mix"]),
    })

    if not np.isfinite(entry_core):
        return _fail("entry_core_nan", "entry_core is not finite")
    if entry_core < float(policy.entry_min_score):
        return _fail("entry_core_floor", "entry_core < entry_min_score", entry_core, float(policy.entry_min_score))
    if not np.isfinite(entry_gap):
        return _fail("entry_gap_nan", "entry_gap is not finite")
    if entry_gap < float(policy.entry_min_gap):
        return _fail("entry_gap", "entry_gap < entry_min_gap", entry_gap, float(policy.entry_min_gap))

    long_u10 = _safe_float(row.get("pred_tgt_long_utility_10"), float("nan"))
    short_u10 = _safe_float(row.get("pred_tgt_short_utility_10"), float("nan"))
    if not np.isfinite(long_u10) or not np.isfinite(short_u10):
        return _fail("utility_10_nan", "pred_tgt_long_utility_10 or pred_tgt_short_utility_10 is not finite")

    utility_10 = long_u10 if side > 0 else short_u10
    utility_10_other = short_u10 if side > 0 else long_u10
    utility_gap_10 = utility_10 - utility_10_other
    diag["utility_10"] = float(utility_10)
    diag["utility_gap_10"] = float(utility_gap_10)

    if utility_10 < float(policy.entry_min_utility_10):
        return _fail("utility_10_floor", "utility_10 < entry_min_utility_10", utility_10, float(policy.entry_min_utility_10))
    if utility_gap_10 < float(policy.entry_min_utility_gap_10):
        return _fail("utility_10_gap", "utility_gap_10 < entry_min_utility_gap_10", utility_gap_10, float(policy.entry_min_utility_gap_10))

    cls_score_10 = _safe_float(row.get("predscore_tgt_retcls_10"), float("nan"))
    diag["class_score_10"] = float(cls_score_10) if np.isfinite(cls_score_10) else float("nan")
    if not bool(policy.require_retcls_alignment):
        diag["retcls_alignment_state"] = "not_required"
    elif not np.isfinite(cls_score_10):
        diag["retcls_alignment_state"] = "missing_score"
        return _fail("retcls_alignment_missing", "retcls alignment required but score is missing")
    else:
        aligned = (side > 0 and cls_score_10 >= 0.0) or (side < 0 and cls_score_10 <= 0.0)
        diag["retcls_alignment_state"] = "pass" if aligned else "fail"
        if not aligned:
            return _fail("retcls_alignment", "retcls sign is not aligned with chosen side", cls_score_10, 0.0)

    main_tok = _barrier_token(float(policy.confirm_main_barrier))
    timing_tok = _barrier_token(float(policy.timing_barrier))

    if side > 0:
        main_prob = _safe_float(row.get(f"predprob_tgt_up_hit_{main_tok}_10"), float("nan"))
        timing_prob = _safe_float(row.get(f"predprob_up_tgt_first_hit_{timing_tok}_10"), float("nan"))
        timing_exp = _safe_float(row.get(f"predexp_tgt_tth_up_{timing_tok}_10"), float("nan"))
        timing_censored_prob = _safe_float(row.get(f"predprob_censored_tgt_tth_up_{timing_tok}_10"), float("nan"))
        fav_main_n = _safe_float(row.get("pred_tgt_up_excur_10_n"), float("nan"))
        adv_main_n = _safe_float(row.get("pred_tgt_down_excur_10_n"), float("nan"))
    else:
        main_prob = _safe_float(row.get(f"predprob_tgt_down_hit_{main_tok}_10"), float("nan"))
        timing_prob = _safe_float(row.get(f"predprob_down_tgt_first_hit_{timing_tok}_10"), float("nan"))
        timing_exp = _safe_float(row.get(f"predexp_tgt_tth_down_{timing_tok}_10"), float("nan"))
        timing_censored_prob = _safe_float(row.get(f"predprob_censored_tgt_tth_down_{timing_tok}_10"), float("nan"))
        fav_main_n = _safe_float(row.get("pred_tgt_down_excur_10_n"), float("nan"))
        adv_main_n = _safe_float(row.get("pred_tgt_up_excur_10_n"), float("nan"))

    diag["main_confirm_prob"] = float(main_prob) if np.isfinite(main_prob) else float("nan")
    diag["timing_first_hit_prob"] = float(timing_prob) if np.isfinite(timing_prob) else float("nan")
    diag["timing_expected_bars"] = float(timing_exp) if np.isfinite(timing_exp) else float("nan")
    diag["timing_censored_prob"] = float(timing_censored_prob) if np.isfinite(timing_censored_prob) else float("nan")

    if np.isfinite(main_prob) and main_prob < float(policy.confirm_main_prob):
        return _fail("main_confirm_prob", "main_confirm_prob < confirm_main_prob", main_prob, float(policy.confirm_main_prob))
    if np.isfinite(timing_prob) and timing_prob < float(policy.timing_first_hit_prob):
        return _fail("timing_first_hit_prob", "timing_first_hit_prob < threshold", timing_prob, float(policy.timing_first_hit_prob))
    if np.isfinite(timing_exp) and timing_exp > float(policy.timing_max_expected_bars):
        return _fail("timing_expected_bars", "timing_expected_bars > threshold", timing_exp, float(policy.timing_max_expected_bars))
    if np.isfinite(timing_censored_prob) and timing_censored_prob > float(policy.timing_max_censored_prob):
        return _fail("timing_censored_prob", "timing_censored_prob > threshold", timing_censored_prob, float(policy.timing_max_censored_prob))

    if not np.isfinite(fav_main_n):
        return _fail("fav_path_missing", "selected favorable excursion n is not finite")
    if not np.isfinite(adv_main_n):
        return _fail("adv_path_missing", "selected adverse excursion n is not finite")

    tp_n = float(np.clip(max(float(policy.min_tp_n), fav_main_n), float(policy.min_tp_n), float(policy.max_tp_n)))
    sl_n = float(np.clip(max(float(policy.min_sl_n), adv_main_n), float(policy.min_sl_n), float(policy.max_sl_n)))
    tp_ret = tp_n * scale_ref
    sl_ret = sl_n * scale_ref

    plan = {
        "side": int(side),
        "entry_core": float(entry_core),
        "entry_gap": float(entry_gap),
        "entry_long_core": float(entry["entry_long_core"]),
        "entry_short_core": float(entry["entry_short_core"]),
        "hyb_mix": float(entry["hyb_mix"]),
        "cls_mix": float(entry["cls_mix"]),
        "util_long_mix": float(entry["util_long_mix"]),
        "util_short_mix": float(entry["util_short_mix"]),
        "utility_10": float(utility_10),
        "utility_gap_10": float(utility_gap_10),
        "class_score_10": float(cls_score_10) if np.isfinite(cls_score_10) else float("nan"),
        "main_confirm_prob": float(main_prob) if np.isfinite(main_prob) else float("nan"),
        "timing_first_hit_prob": float(timing_prob) if np.isfinite(timing_prob) else float("nan"),
        "timing_expected_bars": float(timing_exp) if np.isfinite(timing_exp) else float("nan"),
        "timing_censored_prob": float(timing_censored_prob) if np.isfinite(timing_censored_prob) else float("nan"),
        "tp_n": float(tp_n),
        "sl_n": float(sl_n),
        "tp_ret": float(tp_ret),
        "sl_ret": float(sl_ret),
        "max_hold_bars": int(policy.hard_max_hold_bars),
        "scale_ref_t": float(scale_ref),
    }
    diag.update(plan)
    diag["policy_gate_passed"] = True
    diag["plan"] = plan
    diag["gate_fail_stage"] = ""
    diag["gate_fail_detail"] = ""
    diag["gate_fail_value"] = float("nan")
    diag["gate_fail_threshold"] = float("nan")
    return diag


def derive_trade_plan_from_row(row: Mapping[str, Any], *, policy: PolicyConfig) -> Optional[Dict[str, Any]]:
    return diagnose_trade_plan_from_row(row, policy=policy).get("plan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run v5.1 checkpoint inference on a raw/features/dataset frame")
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scaler-json", default="")
    ap.add_argument("--input-kind", default="auto", choices=["auto", "raw", "features", "dataset"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--preserve-input-columns", type=int, default=1)
    ap.add_argument("--policy-json", default="", help="optional policy json for derived entry composite / plan columns")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = read_frame(args.input)
    policy = _resolve_policy_config(json_path=str(args.policy_json)) if str(args.policy_json).strip() else None
    infer = PathDistInferenceV5(args.checkpoint, scaler_json=(args.scaler_json or None), device=str(args.device))
    pred = infer.predict_frame(
        df,
        input_kind=str(args.input_kind),
        batch_size=int(args.batch_size),
        preserve_input_columns=bool(int(args.preserve_input_columns)),
        policy=policy,
    )
    write_frame(pred, args.output)


if __name__ == "__main__":
    main()
