
# -*- coding: utf-8 -*-
"""
trainer_v5.py

V5 trainer aligned with:
- feature_contract_v5.py
- feature_ops_v5.py
- target_contract_v5.py
- label_builder_v5_1.py
- model_v5_1.py

Key policy
----------
- v5 uses the fixed `feat_v5_reactive26_r1` feature contract.
- target semantics follow `target_v5_openref_1_3_5_8_10_dirhyb_path10_noext_r1`.
- entry side/timing is restored via the legacy-inspired `dir + hyb` structure.
- thesis / exit geometry remains `path10 / no-extension`.
- checkpoint selection is consumer-aware:
  * head losses are still optimized,
  * but validation also computes a lightweight entry/thesis proxy score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from feature_contract_v5 import (
    DEFAULT_MODEL_INPUT_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    FEATURE_READY_COL,
    REQUIRED_FEATURE_COLUMNS,
)
from feature_ops_v5 import read_frame
from label_builder_v5_1 import LABEL_AUDIT_COLUMNS, LabelBuildConfig, build_labels, summarize_labeled_frame
from model_v5_1 import (
    BIN_MAIN_TARGETS,
    DIR_MAIN_TARGETS,
    FIRST_HIT_NUM_CLASSES,
    FIRST_HIT_TARGETS,
    INVALID_CLASS_VALUE,
    ModelMetaV5_1,
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
from target_contract_v5 import (
    BARRIERS_ATR_MAIN,
    CLASSIFICATION_TARGETS_MAIN,
    DIR_TARGETS_MAIN,
    FIRST_HIT_BARRIERS_ATR,
    REGRESSION_TARGETS_MAIN,
    RETURN_HORIZONS_MAIN,
    RETURN_TARGETS_MAIN,
    TARGET_CONTRACT_VERSION,
    TTH_BARRIERS_ATR,
    TTH_CENSORED_VALUE_MAIN,
    UTILITY_HORIZONS,
    UTILITY_TARGETS_MAIN,
    build_target_list,
)

EPS = 1.0e-12
RETCLS_SUPPORT = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32) / 3.0

DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "w_ret_main": 0.40,
    "w_dir_main": 0.35,
    "w_entry_hyb": 0.55,
    "w_path_main": 1.00,
    "w_util_main": 0.90,
    "w_retcls": 0.30,
    "w_bin": 0.55,
    "w_first_hit": 0.30,
    "w_tth": 0.20,
}

# ---------------------------------------------------------------------------
# Target-index helpers
# ---------------------------------------------------------------------------


def _barrier_token(barrier: float) -> str:
    return f"{int(round(float(barrier) * 100.0)):03d}"


def _parse_ret_horizon(name: str) -> int:
    # tgt_ret_10_n
    parts = str(name).split("_")
    return int(parts[2])


def _parse_util_side_horizon(name: str) -> Tuple[str, int]:
    # tgt_long_utility_10
    parts = str(name).split("_")
    return str(parts[1]), int(parts[-1])


def _parse_suffix_horizon(name: str) -> int:
    # tgt_retcls_10
    return int(str(name).split("_")[-1])


RET_HORIZON_TO_IDX: Dict[int, int] = {_parse_ret_horizon(name): i for i, name in enumerate(RET_MAIN_TARGETS)}
DIR_HORIZON_TO_IDX: Dict[int, int] = {_parse_suffix_horizon(name): i for i, name in enumerate(DIR_MAIN_TARGETS)}
RETCLS_HORIZON_TO_IDX: Dict[int, int] = {_parse_suffix_horizon(name): i for i, name in enumerate(RETCLS_TARGETS)}

UTILITY_LONG_HORIZON_TO_IDX: Dict[int, int] = {}
UTILITY_SHORT_HORIZON_TO_IDX: Dict[int, int] = {}
for i, name in enumerate(UTILITY_MAIN_TARGETS):
    side, h = _parse_util_side_horizon(name)
    if side == "long":
        UTILITY_LONG_HORIZON_TO_IDX[h] = i
    else:
        UTILITY_SHORT_HORIZON_TO_IDX[h] = i

BIN_TARGET_TO_IDX: Dict[str, int] = {str(name): i for i, name in enumerate(BIN_MAIN_TARGETS)}
FIRST_TARGET_TO_IDX: Dict[str, int] = {str(name): i for i, name in enumerate(FIRST_HIT_TARGETS)}
TTH_TARGET_TO_IDX: Dict[str, int] = {str(name): i for i, name in enumerate(TTH_TARGETS)}

MAIN_CONFIRM_TARGET_UP = f"tgt_up_hit_{_barrier_token(BARRIERS_ATR_MAIN[0])}_10"
MAIN_CONFIRM_TARGET_DOWN = f"tgt_down_hit_{_barrier_token(BARRIERS_ATR_MAIN[0])}_10"
FIRST_HIT_TARGET_100 = f"tgt_first_hit_{_barrier_token(FIRST_HIT_BARRIERS_ATR[0])}_10"
TTH_TARGET_UP_100 = f"tgt_tth_up_{_barrier_token(TTH_BARRIERS_ATR[0])}_10"
TTH_TARGET_DOWN_100 = f"tgt_tth_down_{_barrier_token(TTH_BARRIERS_ATR[0])}_10"


# ---------------------------------------------------------------------------
# Generic utils
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _json_load_path(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return asdict(x)
    return str(x)


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default)


def _resolve_weight_overrides(
    defaults: Mapping[str, float],
    *,
    json_path: str = "",
    inline_json: str = "",
) -> Dict[str, float]:
    out = {str(k): float(v) for k, v in defaults.items()}
    payload: Dict[str, Any] = {}
    if str(json_path).strip():
        payload.update(_json_load_path(json_path))
    if str(inline_json).strip():
        payload.update(json.loads(str(inline_json)))
    for k in list(out.keys()):
        if k in payload:
            out[k] = float(payload[k])
    return out

DEFAULT_TRAIN_JSON_FIELDS: Dict[str, Any] = {
    "rows": 0,
    "seq_len": 160,
    "train_ratio": 0.85,
    "batch_size": 512,
    "epochs": 40,
    "patience": 6,
    "lr": 7.0e-4,
    "weight_decay": 1.0e-5,
    "grad_clip": 1.0,
    "seed": 42,
    "winsor_q": 0.005,
    "d_model": 128,
    "stem_hidden_per_group": 24,
    "stem_dropout": 0.05,
    "tcn_blocks": 5,
    "kernel_size": 3,
    "tcn_dropout": 0.05,
    "xlstm_layers": 2,
    "xlstm_dropout": 0.05,
    "readout_hidden": 160,
    "readout_dropout": 0.05,
    "head_hidden": 128,
    "head_dropout": 0.05,
    "scheduler": "none",
    "warmup_epochs": 0.0,
    "min_lr_ratio": 0.10,
}


def _clip_float(x: Any, lo: float, hi: float) -> float:
    xv = _safe_float(x, lo)
    return float(max(float(lo), min(float(hi), xv)))


def _coerce_cli_value(name: str, value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(default, bool):
        return _safe_bool(value, default)
    if isinstance(default, int) and not isinstance(default, bool):
        return _safe_int(value, default)
    if isinstance(default, float):
        return float(value)
    return str(value)


def _same_as_default(current: Any, default: Any) -> bool:
    if isinstance(default, float):
        try:
            return abs(float(current) - float(default)) <= 1.0e-12
        except Exception:
            return False
    return current == default


def _apply_train_json_overrides(args: argparse.Namespace) -> argparse.Namespace:
    path = str(getattr(args, "train_json", "")).strip()
    if not path:
        return args

    raw = _json_load_path(path)
    alias = {
        "bs": "batch_size",
        "batch-size": "batch_size",
        "wd": "weight_decay",
        "warmup_epoch": "warmup_epochs",
        "warmup": "warmup_epochs",
        "min_lr": "min_lr_ratio",
        "min_lr_ratio": "min_lr_ratio",
    }
    payload: Dict[str, Any] = {}
    for k, v in dict(raw).items():
        payload[alias.get(str(k), str(k))] = v

    # JSON은 "해당 CLI 값이 아직 기본값일 때만" 덮어쓴다.
    # 즉 명시적으로 CLI에서 준 값이 있으면 CLI가 우선한다.
    for key, default in DEFAULT_TRAIN_JSON_FIELDS.items():
        if key not in payload or not hasattr(args, key):
            continue
        current = getattr(args, key)
        if _same_as_default(current, default):
            setattr(args, key, _coerce_cli_value(key, payload[key], default))
    return args


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: float,
    min_lr_ratio: float,
):
    name = str(scheduler_name or "none").strip().lower()
    if name in {"", "none", "off"}:
        return None
    if name not in {"cosine", "cosine_epoch"}:
        raise ValueError(f"unsupported scheduler: {scheduler_name!r}; supported: none, cosine")

    total_epochs = max(int(epochs), 1)
    warm = max(float(warmup_epochs), 0.0)
    min_ratio = _clip_float(float(min_lr_ratio), 1.0e-4, 1.0)

    def _lambda(epoch_idx: int) -> float:
        e = float(epoch_idx + 1)
        if warm > 0.0 and e <= warm:
            return max(1.0e-4, e / max(warm, 1.0e-6))
        if total_epochs <= 1:
            return 1.0
        prog = (e - warm) / max(total_epochs - warm, 1.0)
        prog = _clip_float(prog, 0.0, 1.0)
        return float(min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda)

# ---------------------------------------------------------------------------
# Shared entry scoring helpers (consumed by inference_v5 too)
# ---------------------------------------------------------------------------


@dataclass
class EntryScoringConfig:
    w1: float = 0.28
    w3: float = 0.28
    w5: float = 0.20
    w8: float = 0.14
    w10: float = 0.10

    hyb_weight: float = 0.55
    util_weight: float = 0.30
    cls_weight: float = 0.15

    entry_q: float = 0.85
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

    consumer_min_accept_count: int = 32
    consumer_accept_rate_floor: float = 0.01


def normalize_horizon_weights(cfg: Mapping[str, Any] | EntryScoringConfig) -> Dict[int, float]:
    if isinstance(cfg, EntryScoringConfig):
        raw = {
            1: float(cfg.w1),
            3: float(cfg.w3),
            5: float(cfg.w5),
            8: float(cfg.w8),
            10: float(cfg.w10),
        }
    else:
        raw = {h: max(0.0, _safe_float(cfg.get(f"w{h}", 0.0), 0.0)) for h in RETURN_HORIZONS_MAIN}
    s = float(sum(max(0.0, v) for v in raw.values()))
    if s <= 0.0:
        raw = {1: 0.28, 3: 0.28, 5: 0.20, 8: 0.14, 10: 0.10}
        s = float(sum(raw.values()))
    return {int(h): float(max(0.0, raw[int(h)]) / max(s, EPS)) for h in RETURN_HORIZONS_MAIN}


def _retcls_score_from_prob(retcls_prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(retcls_prob, dtype=np.float32)
    if prob.ndim != 3 or prob.shape[-1] != RETCLS_NUM_CLASSES:
        raise ValueError(f"retcls_prob must have shape [N,H,{RETCLS_NUM_CLASSES}]")
    return (prob * RETCLS_SUPPORT.reshape(1, 1, -1)).sum(axis=-1)


def build_entry_composites_numpy(
    hyb_main: np.ndarray,
    util_main: np.ndarray,
    retcls_score: np.ndarray,
    *,
    config: EntryScoringConfig,
) -> Dict[str, np.ndarray]:
    hyb = np.asarray(hyb_main, dtype=np.float32)
    util = np.asarray(util_main, dtype=np.float32)
    cls = np.asarray(retcls_score, dtype=np.float32)

    if hyb.ndim != 2 or hyb.shape[1] != len(RET_MAIN_TARGETS):
        raise ValueError(f"hyb_main must have shape [N,{len(RET_MAIN_TARGETS)}]")
    if util.ndim != 2 or util.shape[1] != len(UTILITY_MAIN_TARGETS):
        raise ValueError(f"util_main must have shape [N,{len(UTILITY_MAIN_TARGETS)}]")
    if cls.ndim != 2 or cls.shape[1] != len(RETCLS_TARGETS):
        raise ValueError(f"retcls_score must have shape [N,{len(RETCLS_TARGETS)}]")

    weights = normalize_horizon_weights(config)
    n = int(hyb.shape[0])

    long_core = np.zeros(n, dtype=np.float32)
    short_core = np.zeros(n, dtype=np.float32)
    hyb_mix = np.zeros(n, dtype=np.float32)
    cls_mix = np.zeros(n, dtype=np.float32)
    util_long_mix = np.zeros(n, dtype=np.float32)
    util_short_mix = np.zeros(n, dtype=np.float32)

    for h in RETURN_HORIZONS_MAIN:
        w = np.float32(weights[int(h)])
        hy = hyb[:, RET_HORIZON_TO_IDX[int(h)]]
        cls_h = cls[:, RETCLS_HORIZON_TO_IDX[int(h)]]
        util_long = util[:, UTILITY_LONG_HORIZON_TO_IDX[int(h)]]
        util_short = util[:, UTILITY_SHORT_HORIZON_TO_IDX[int(h)]]

        hyb_mix += w * hy
        cls_mix += w * cls_h
        util_long_mix += w * util_long
        util_short_mix += w * util_short

        long_core += w * (
            float(config.hyb_weight) * np.maximum(hy, 0.0)
            + float(config.util_weight) * np.maximum(util_long, 0.0)
            + float(config.cls_weight) * np.maximum(cls_h, 0.0)
        )
        short_core += w * (
            float(config.hyb_weight) * np.maximum(-hy, 0.0)
            + float(config.util_weight) * np.maximum(util_short, 0.0)
            + float(config.cls_weight) * np.maximum(-cls_h, 0.0)
        )

    side = np.where(long_core >= short_core, 1, -1).astype(np.int8)
    entry_core = np.where(side > 0, long_core, short_core).astype(np.float32)
    other_core = np.where(side > 0, short_core, long_core).astype(np.float32)
    entry_gap = (entry_core - other_core).astype(np.float32)

    return {
        "entry_long_core": long_core.astype(np.float32),
        "entry_short_core": short_core.astype(np.float32),
        "entry_side": side,
        "entry_core": entry_core,
        "entry_gap": entry_gap,
        "hyb_mix": hyb_mix.astype(np.float32),
        "cls_mix": cls_mix.astype(np.float32),
        "util_long_mix": util_long_mix.astype(np.float32),
        "util_short_mix": util_short_mix.astype(np.float32),
    }


def consumer_selection_proxy(
    *,
    pred_hyb: np.ndarray,
    pred_util: np.ndarray,
    pred_retcls_prob: np.ndarray,
    pred_bin_prob: np.ndarray,
    pred_first_hit_prob: np.ndarray,
    pred_tth_exp: np.ndarray,
    pred_tth_cens_prob: np.ndarray,
    true_ret_unscaled: np.ndarray,
    true_util: np.ndarray,
    config: EntryScoringConfig,
) -> Dict[str, float]:
    pred_cls_score = _retcls_score_from_prob(pred_retcls_prob)
    entry = build_entry_composites_numpy(pred_hyb, pred_util, pred_cls_score, config=config)

    side = entry["entry_side"]
    entry_core = entry["entry_core"]
    entry_gap = entry["entry_gap"]

    util10_long = pred_util[:, UTILITY_LONG_HORIZON_TO_IDX[10]]
    util10_short = pred_util[:, UTILITY_SHORT_HORIZON_TO_IDX[10]]
    util10_side = np.where(side > 0, util10_long, util10_short)
    util10_other = np.where(side > 0, util10_short, util10_long)
    util10_gap = util10_side - util10_other

    main_up_idx = BIN_TARGET_TO_IDX[MAIN_CONFIRM_TARGET_UP]
    main_dn_idx = BIN_TARGET_TO_IDX[MAIN_CONFIRM_TARGET_DOWN]
    main_prob = np.where(side > 0, pred_bin_prob[:, main_up_idx], pred_bin_prob[:, main_dn_idx])

    fh_idx = FIRST_TARGET_TO_IDX[FIRST_HIT_TARGET_100]
    timing_prob = np.where(side > 0, pred_first_hit_prob[:, fh_idx, 1], pred_first_hit_prob[:, fh_idx, 2])

    tth_up_idx = TTH_TARGET_TO_IDX[TTH_TARGET_UP_100]
    tth_dn_idx = TTH_TARGET_TO_IDX[TTH_TARGET_DOWN_100]
    timing_exp = np.where(side > 0, pred_tth_exp[:, tth_up_idx], pred_tth_exp[:, tth_dn_idx])
    timing_cens = np.where(side > 0, pred_tth_cens_prob[:, tth_up_idx], pred_tth_cens_prob[:, tth_dn_idx])

    cls10 = pred_cls_score[:, RETCLS_HORIZON_TO_IDX[10]]
    aligned = np.where(side > 0, cls10 >= 0.0, cls10 <= 0.0)

    base_valid = (
        np.isfinite(entry_core)
        & np.isfinite(entry_gap)
        & np.isfinite(util10_side)
        & np.isfinite(util10_gap)
        & np.isfinite(main_prob)
        & np.isfinite(timing_prob)
        & np.isfinite(timing_exp)
        & np.isfinite(timing_cens)
    )
    if not np.any(base_valid):
        return {
            "consumer_selection_score": float("nan"),
            "consumer_accept_count": 0.0,
            "consumer_accept_rate": 0.0,
            "consumer_mean_true_util10": float("nan"),
            "consumer_mean_true_hyb_mix": float("nan"),
            "consumer_true_win_rate": float("nan"),
            "consumer_entry_threshold": float("nan"),
        }

    core_valid = entry_core[base_valid]
    try:
        q = float(np.clip(float(config.entry_q), 0.0, 1.0))
        entry_threshold = float(np.quantile(core_valid, q))
    except Exception:
        entry_threshold = float(np.nanmedian(core_valid))
    entry_threshold = max(entry_threshold, float(config.entry_min_score))

    accept = base_valid.copy()
    accept &= entry_core >= float(entry_threshold)
    accept &= entry_gap >= float(config.entry_min_gap)
    accept &= util10_side >= float(config.entry_min_utility_10)
    accept &= util10_gap >= float(config.entry_min_utility_gap_10)
    accept &= main_prob >= float(config.confirm_main_prob)
    accept &= timing_prob >= float(config.timing_first_hit_prob)
    accept &= timing_exp <= float(config.timing_max_expected_bars)
    accept &= timing_cens <= float(config.timing_max_censored_prob)
    if bool(config.require_retcls_alignment):
        accept &= aligned

    accept_count = int(np.sum(accept))
    accept_rate = float(accept_count / max(np.sum(base_valid), 1))

    true_hyb_mix = np.zeros(len(true_ret_unscaled), dtype=np.float32)
    weights = normalize_horizon_weights(config)
    for h in RETURN_HORIZONS_MAIN:
        true_hyb_mix += np.float32(weights[int(h)]) * true_ret_unscaled[:, RET_HORIZON_TO_IDX[int(h)]]

    true_util10_long = true_util[:, UTILITY_LONG_HORIZON_TO_IDX[10]]
    true_util10_short = true_util[:, UTILITY_SHORT_HORIZON_TO_IDX[10]]
    true_util10_side = np.where(side > 0, true_util10_long, true_util10_short)

    min_accept = int(max(1, config.consumer_min_accept_count))
    floor_rate = float(max(0.0, config.consumer_accept_rate_floor))
    no_trade_score = -1.0e9

    if accept_count <= 0:
        return {
            "consumer_selection_score": float(no_trade_score),
            "consumer_selection_score_raw_econ": float("nan"),
            "consumer_selection_penalty_accept_count": 0.0,
            "consumer_selection_penalty_accept_rate": 0.0,
            "consumer_selection_total_penalty": 0.0,
            "consumer_accept_count": 0.0,
            "consumer_accept_rate": float(accept_rate),
            "consumer_mean_true_util10": float("nan"),
            "consumer_mean_true_hyb_mix": float("nan"),
            "consumer_true_win_rate": float("nan"),
            "consumer_entry_threshold": float(entry_threshold),
        }

    mean_true_util10 = float(np.nanmean(true_util10_side[accept]))
    mean_true_hyb_mix = float(np.nanmean(true_hyb_mix[accept]))
    true_win_rate = float(np.nanmean((true_hyb_mix[accept] > 0.0).astype(np.float32)))

    # economics-first raw score
    raw_econ = mean_true_util10 + 0.35 * mean_true_hyb_mix + 0.10 * true_win_rate

    penalty_accept_count = 0.0
    if accept_count < min_accept:
        penalty_accept_count = float(min_accept - accept_count) / float(min_accept)

    penalty_accept_rate = 0.0
    if accept_rate < floor_rate and floor_rate > 0.0:
        penalty_accept_rate = 0.50 * float(floor_rate - accept_rate) / float(floor_rate)

    score = raw_econ - penalty_accept_count - penalty_accept_rate

    return {
        "consumer_selection_score": float(score),  # penalized
        "consumer_selection_score_raw_econ": float(raw_econ),  # pure economics
        "consumer_selection_penalty_accept_count": float(penalty_accept_count),
        "consumer_selection_penalty_accept_rate": float(penalty_accept_rate),
        "consumer_selection_total_penalty": float(penalty_accept_count + penalty_accept_rate),
        "consumer_accept_count": float(accept_count),
        "consumer_accept_rate": float(accept_rate),
        "consumer_mean_true_util10": float(mean_true_util10),
        "consumer_mean_true_hyb_mix": float(mean_true_hyb_mix),
        "consumer_true_win_rate": float(true_win_rate),
        "consumer_entry_threshold": float(entry_threshold),
    }

def _resolve_entry_scoring_config(
    *,
    json_path: str = "",
    inline_json: str = "",
    overrides: Optional[Mapping[str, Any]] = None,
) -> EntryScoringConfig:
    raw = asdict(EntryScoringConfig())
    if str(json_path).strip():
        raw.update(_json_load_path(json_path))
    if str(inline_json).strip():
        raw.update(json.loads(str(inline_json)))
    if overrides:
        for k, v in overrides.items():
            if v is not None and k in raw:
                raw[k] = v
    kwargs: Dict[str, Any] = {}
    field_map = {f.name: f for f in fields(EntryScoringConfig)}
    for k, field_obj in field_map.items():
        v = raw.get(k, getattr(EntryScoringConfig(), k))
        if field_obj.type is bool:
            kwargs[k] = _safe_bool(v, getattr(EntryScoringConfig(), k))
        elif field_obj.type is int:
            kwargs[k] = _safe_int(v, getattr(EntryScoringConfig(), k))
        else:
            kwargs[k] = float(v) if isinstance(getattr(EntryScoringConfig(), k), float) else v
    return EntryScoringConfig(**kwargs)

def resolve_effective_selection_metric(
    valid_stats: Mapping[str, Any],
    selection_cfg: EntryScoringConfig,
) -> Dict[str, float]:
    consumer = float(valid_stats.get("consumer_selection_score", float("nan")))  # penalized
    consumer_raw_econ = float(valid_stats.get("consumer_selection_score_raw_econ", consumer))
    focus = float(valid_stats.get("loss_selection_focus_loss", float("inf")))
    total = float(valid_stats.get("loss_total", float("inf")))
    accept_count = int(round(float(valid_stats.get("consumer_accept_count", 0.0))))
    accept_rate = float(valid_stats.get("consumer_accept_rate", 0.0))
    min_accept = int(max(1, selection_cfg.consumer_min_accept_count))
    floor_rate = float(max(0.0, selection_cfg.consumer_accept_rate_floor))

    checkpoint_eligible = False

    if not np.isfinite(consumer):
        effective = -focus
        mode = "focus_only_nonfinite_consumer"
    elif consumer <= -1.0e8 or accept_count <= 0:
        effective = -focus - 10.0
        mode = "no_trade_focus_penalized"
    elif accept_count < min_accept or (floor_rate > 0.0 and accept_rate < floor_rate):
        anchor = consumer_raw_econ if np.isfinite(consumer_raw_econ) else consumer

        count_blend = min(float(accept_count) / float(min_accept), 1.0)
        rate_blend = 1.0 if floor_rate <= 0.0 else min(float(accept_rate) / float(floor_rate), 1.0)
        blend = min(count_blend, rate_blend)

        effective = blend * anchor + (1.0 - blend) * (-focus)
        mode = "low_accept_blend"
    else:
        effective = consumer_raw_econ if np.isfinite(consumer_raw_econ) else consumer
        mode = "consumer"
        checkpoint_eligible = True

    return {
        "selection_effective_score": float(effective),
        "selection_raw_consumer_score": float(consumer) if np.isfinite(consumer) else float("nan"),
        "selection_raw_consumer_score_raw_econ": float(consumer_raw_econ) if np.isfinite(consumer_raw_econ) else float("nan"),
        "selection_focus_loss": float(focus) if np.isfinite(focus) else float("inf"),
        "selection_total_loss": float(total) if np.isfinite(total) else float("inf"),
        "selection_accept_count": float(accept_count),
        "selection_accept_rate": float(accept_rate),
        "selection_mode": str(mode),
        "selection_checkpoint_eligible": bool(checkpoint_eligible),
    }

# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def _detect_frame_kind(df: pd.DataFrame) -> str:
    has_raw = all(c in df.columns for c in ("timestamp", "open", "high", "low", "close"))
    has_feat = all(c in df.columns for c in REQUIRED_FEATURE_COLUMNS)
    has_tgt = all(c in df.columns for c in build_target_list(include_extension=False))
    if has_raw and has_feat and has_tgt:
        return "dataset"
    if has_raw and has_feat:
        return "features"
    if has_raw:
        return "raw"
    raise ValueError("input frame does not match raw/features/dataset contract")


def maybe_build_labeled_dataset(
    df: pd.DataFrame,
    *,
    input_kind: str = "auto",
    cost_per_side: float = 0.00070,
    slip_per_side: float = 0.00015,
) -> pd.DataFrame:
    resolved = str(input_kind or "auto").strip().lower()
    if resolved == "auto":
        resolved = _detect_frame_kind(df)
    if resolved == "dataset":
        return df.copy()
    if resolved not in {"raw", "features"}:
        raise ValueError(f"unsupported input_kind={input_kind}")
    return build_labels(
        df,
        config=LabelBuildConfig(
            input_kind=resolved,
            cost_per_side=float(cost_per_side),
            slip_per_side=float(slip_per_side),
            cast_float32=True,
            preserve_extra_columns=True,
            feature_mask_not_ready=True,
        ),
    )


# ---------------------------------------------------------------------------
# Robust scaler
# ---------------------------------------------------------------------------


def compute_robust_scaler(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    winsor_q: float = 0.005,
    iqr_floor: float = 1.0e-6,
) -> Dict[str, Dict[str, float]]:
    q = float(max(0.0, min(0.20, winsor_q)))
    scaler: Dict[str, Dict[str, float]] = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").astype("float64")
        arr = s.to_numpy(copy=False)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            scaler[str(c)] = {
                "center": 0.0,
                "scale": 1.0,
                "clip_lo": -1.0,
                "clip_hi": 1.0,
                "kind": "robust_iqr_v1",
            }
            continue
        clip_lo = float(np.quantile(arr, q)) if q > 0.0 else float(np.nanmin(arr))
        clip_hi = float(np.quantile(arr, 1.0 - q)) if q > 0.0 else float(np.nanmax(arr))
        arr_w = np.clip(arr, clip_lo, clip_hi)
        center = float(np.nanmedian(arr_w))
        q1 = float(np.nanquantile(arr_w, 0.25))
        q3 = float(np.nanquantile(arr_w, 0.75))
        scale = max(float(q3 - q1), float(iqr_floor))
        scaler[str(c)] = {
            "center": center,
            "scale": scale,
            "clip_lo": clip_lo,
            "clip_hi": clip_hi,
            "kind": "robust_iqr_v1",
        }
    return scaler


def apply_scaler_to_frame(df: pd.DataFrame, scaler: Mapping[str, Mapping[str, float]], cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce").astype("float64")
        cfg = scaler[str(c)]
        lo = _safe_float(cfg.get("clip_lo"), -np.inf)
        hi = _safe_float(cfg.get("clip_hi"), np.inf)
        center = _safe_float(cfg.get("center"), 0.0)
        scale = max(_safe_float(cfg.get("scale"), 1.0), 1.0e-6)
        s = s.clip(lower=lo, upper=hi)
        out[c] = ((s - center) / scale).astype("float32")
    return out


def transform_numpy_with_scaler(x: np.ndarray, scaler: Mapping[str, Mapping[str, float]], cols: Sequence[str]) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    for j, c in enumerate(cols):
        cfg = scaler[str(c)]
        lo = _safe_float(cfg.get("clip_lo"), -np.inf)
        hi = _safe_float(cfg.get("clip_hi"), np.inf)
        center = _safe_float(cfg.get("center"), 0.0)
        scale = max(_safe_float(cfg.get("scale"), 1.0), 1.0e-6)
        col = out[:, j].astype(np.float64, copy=False)
        finite = np.isfinite(col)
        if np.any(finite):
            clipped = np.clip(col[finite], lo, hi)
            col_out = np.full(col.shape, np.nan, dtype=np.float32)
            col_out[finite] = ((clipped - center) / scale).astype(np.float32)
            out[:, j] = col_out
        else:
            out[:, j] = np.nan
    return out


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------


def build_sequence_starts(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    seq_len: int,
    require_ready_col: str = "dataset_ready_main",
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(len(df))
    seq_len = int(seq_len)
    if seq_len <= 0 or n < seq_len:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    x = df.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    row_valid = np.isfinite(x).all(axis=1)
    invalid = (~row_valid).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(invalid)])

    if "contig_run_id" in df.columns:
        run_src = df["contig_run_id"]
        if isinstance(run_src, pd.DataFrame):
            run_src = run_src.iloc[:, -1]
        run_id = pd.to_numeric(run_src, errors="coerce").fillna(-1).astype("int64").to_numpy(copy=False)
    else:
        run_id = np.zeros(n, dtype=np.int64)

    end_idx = np.arange(seq_len - 1, n, dtype=np.int64)
    start_idx = end_idx - seq_len + 1
    same_run = run_id[start_idx] == run_id[end_idx]
    no_invalid = (csum[end_idx + 1] - csum[start_idx]) == 0

    if require_ready_col in df.columns:
        end_ready = pd.Series(df[require_ready_col]).fillna(False).astype(bool).to_numpy(copy=False)[end_idx]
    else:
        end_ready = row_valid[end_idx]

    mask = same_run & no_invalid & end_ready
    return start_idx[mask], end_idx[mask]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SeqV5Dataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        starts: np.ndarray,
        seq_len: int,
        *,
        reg_ret_main: np.ndarray,
        reg_ret_main_mask: np.ndarray,
        dir_main: np.ndarray,
        dir_main_mask: np.ndarray,
        hyb_main: np.ndarray,
        hyb_main_mask: np.ndarray,
        reg_path_main: np.ndarray,
        reg_path_main_mask: np.ndarray,
        reg_util_main: np.ndarray,
        reg_util_main_mask: np.ndarray,
        retcls: np.ndarray,
        bin_main: np.ndarray,
        first_hit: np.ndarray,
        tth: np.ndarray,
        true_ret_unscaled: np.ndarray,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.reg_ret_main = np.asarray(reg_ret_main, dtype=np.float32)
        self.reg_ret_main_mask = np.asarray(reg_ret_main_mask, dtype=bool)
        self.dir_main = np.asarray(dir_main, dtype=np.float32)
        self.dir_main_mask = np.asarray(dir_main_mask, dtype=bool)
        self.hyb_main = np.asarray(hyb_main, dtype=np.float32)
        self.hyb_main_mask = np.asarray(hyb_main_mask, dtype=bool)
        self.reg_path_main = np.asarray(reg_path_main, dtype=np.float32)
        self.reg_path_main_mask = np.asarray(reg_path_main_mask, dtype=bool)
        self.reg_util_main = np.asarray(reg_util_main, dtype=np.float32)
        self.reg_util_main_mask = np.asarray(reg_util_main_mask, dtype=bool)
        self.retcls = np.asarray(retcls, dtype=np.int64)
        self.bin_main = np.asarray(bin_main, dtype=np.int64)
        self.first_hit = np.asarray(first_hit, dtype=np.int64)
        self.tth = np.asarray(tth, dtype=np.int64)
        self.true_ret_unscaled = np.asarray(true_ret_unscaled, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = int(self.starts[idx])
        x = self.X[s : s + self.seq_len]
        item: Dict[str, torch.Tensor] = {
            "x": torch.from_numpy(x),
            "reg_ret_main": torch.from_numpy(self.reg_ret_main[idx]),
            "reg_ret_main_mask": torch.from_numpy(self.reg_ret_main_mask[idx].astype(np.float32)),
            "dir_main": torch.from_numpy(self.dir_main[idx]),
            "dir_main_mask": torch.from_numpy(self.dir_main_mask[idx].astype(np.float32)),
            "hyb_main": torch.from_numpy(self.hyb_main[idx]),
            "hyb_main_mask": torch.from_numpy(self.hyb_main_mask[idx].astype(np.float32)),
            "reg_path_main": torch.from_numpy(self.reg_path_main[idx]),
            "reg_path_main_mask": torch.from_numpy(self.reg_path_main_mask[idx].astype(np.float32)),
            "reg_util_main": torch.from_numpy(self.reg_util_main[idx]),
            "reg_util_main_mask": torch.from_numpy(self.reg_util_main_mask[idx].astype(np.float32)),
            "retcls": torch.from_numpy(self.retcls[idx]),
            "bin_main": torch.from_numpy(self.bin_main[idx].astype(np.float32)),
            "first_hit": torch.from_numpy(self.first_hit[idx]),
            "tth": torch.from_numpy(self.tth[idx]),
            "true_ret_unscaled": torch.from_numpy(self.true_ret_unscaled[idx]),
        }
        return item


# ---------------------------------------------------------------------------
# Train/valid sample preparation
# ---------------------------------------------------------------------------


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_cols: List[str]
    X_scaled: np.ndarray
    starts_all: np.ndarray
    ends_all: np.ndarray
    train_pos: np.ndarray
    valid_pos: np.ndarray
    scaler: Dict[str, Dict[str, float]]
    split_row: int
    seq_len: int


def prepare_training_data(
    df: pd.DataFrame,
    *,
    seq_len: int,
    train_ratio: float,
    winsor_q: float,
) -> PreparedData:
    feature_cols = list(DEFAULT_MODEL_INPUT_COLUMNS)
    missing_feat = [c for c in feature_cols if c not in df.columns]
    missing_tgt = [c for c in build_target_list(include_extension=False) if c not in df.columns]
    if missing_feat:
        raise ValueError(f"dataset missing required feature columns: {missing_feat}")
    if missing_tgt:
        raise ValueError(f"dataset missing required target columns: {missing_tgt}")
    if "dataset_ready_main" not in df.columns:
        raise ValueError("dataset missing required readiness column: dataset_ready_main")

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    split_row = int(max(seq_len + 1, min(len(df) - 1, math.floor(len(df) * float(train_ratio)))))

    train_for_scaler = df.iloc[:split_row].copy()
    row_valid_train = np.isfinite(train_for_scaler.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    train_for_scaler = train_for_scaler.loc[row_valid_train].copy()
    if train_for_scaler.empty:
        raise ValueError("no finite training rows available to fit scaler")

    scaler = compute_robust_scaler(train_for_scaler, feature_cols, winsor_q=float(winsor_q))
    df_scaled = apply_scaler_to_frame(df, scaler, feature_cols)
    X_scaled = df_scaled.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)

    starts_all, ends_all = build_sequence_starts(df_scaled, feature_cols, seq_len=int(seq_len), require_ready_col="dataset_ready_main")
    if len(starts_all) == 0:
        raise ValueError("no valid contiguous sequences were found")

    train_pos = np.where(ends_all < split_row)[0]
    valid_pos = np.where(ends_all >= split_row)[0]
    if len(train_pos) == 0:
        raise ValueError("no train sequences after split")
    if len(valid_pos) == 0:
        valid_tail = max(1, min(len(starts_all) // 10, 1024))
        train_pos = np.arange(0, max(0, len(starts_all) - valid_tail), dtype=np.int64)
        valid_pos = np.arange(max(0, len(starts_all) - valid_tail), len(starts_all), dtype=np.int64)
        if len(train_pos) == 0 or len(valid_pos) == 0:
            raise ValueError("insufficient sequences for validation fallback split")

    return PreparedData(
        df=df_scaled,
        feature_cols=feature_cols,
        X_scaled=X_scaled,
        starts_all=starts_all,
        ends_all=ends_all,
        train_pos=train_pos.astype(np.int64),
        valid_pos=valid_pos.astype(np.int64),
        scaler=scaler,
        split_row=split_row,
        seq_len=int(seq_len),
    )


def _to_numeric_2d(df: pd.DataFrame, cols: Sequence[str], *, dtype: str = "float32") -> np.ndarray:
    return pd.DataFrame(df.loc[:, list(cols)]).apply(pd.to_numeric, errors="coerce").astype(dtype).to_numpy(copy=False)


def make_dataset(
    prep: PreparedData,
    positions: np.ndarray,
    *,
    reg_target_mode: str,
    y_scale: float,
) -> SeqV5Dataset:
    pos = np.asarray(positions, dtype=np.int64)
    starts = prep.starts_all[pos]
    ends = prep.ends_all[pos]
    df = prep.df

    ret_unscaled = _to_numeric_2d(df, RET_MAIN_TARGETS, dtype="float32")[ends]
    ret_scaled = ret_unscaled * np.float32(float(y_scale))
    reg_ret_main = np.abs(ret_scaled) if str(reg_target_mode).strip().lower() == "magnitude_v2" else ret_scaled.copy()
    reg_ret_main_mask = np.isfinite(reg_ret_main)

    dir_main = _to_numeric_2d(df, DIR_MAIN_TARGETS, dtype="float32")[ends]
    dir_main_mask = np.isfinite(dir_main)
    dir_main = np.nan_to_num(dir_main, nan=0.0)

    hyb_main = ret_scaled.copy()
    hyb_main_mask = np.isfinite(hyb_main)

    reg_path_main = _to_numeric_2d(df, PATH_MAIN_TARGETS, dtype="float32")[ends]
    reg_path_main_mask = np.isfinite(reg_path_main)

    reg_util_main = _to_numeric_2d(df, UTILITY_MAIN_TARGETS, dtype="float32")[ends]
    reg_util_main_mask = np.isfinite(reg_util_main)

    retcls = (
        pd.DataFrame(df.loc[:, list(RETCLS_TARGETS)])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(INVALID_CLASS_VALUE)
        .astype(np.int64)
        .to_numpy(copy=False)[ends]
    )
    bin_main = (
        pd.DataFrame(df.loc[:, list(BIN_MAIN_TARGETS)])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(INVALID_CLASS_VALUE)
        .astype(np.int64)
        .to_numpy(copy=False)[ends]
    )
    first_hit = (
        pd.DataFrame(df.loc[:, list(FIRST_HIT_TARGETS)])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(INVALID_CLASS_VALUE)
        .astype(np.int64)
        .to_numpy(copy=False)[ends]
    )
    tth = (
        pd.DataFrame(df.loc[:, list(TTH_TARGETS)])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(INVALID_CLASS_VALUE)
        .astype(np.int64)
        .to_numpy(copy=False)[ends]
    )

    return SeqV5Dataset(
        prep.X_scaled,
        starts,
        seq_len=int(prep.seq_len),
        reg_ret_main=reg_ret_main,
        reg_ret_main_mask=reg_ret_main_mask,
        dir_main=dir_main,
        dir_main_mask=dir_main_mask,
        hyb_main=hyb_main,
        hyb_main_mask=hyb_main_mask,
        reg_path_main=reg_path_main,
        reg_path_main_mask=reg_path_main_mask,
        reg_util_main=reg_util_main,
        reg_util_main_mask=reg_util_main_mask,
        retcls=retcls,
        bin_main=bin_main,
        first_hit=first_hit,
        tth=tth,
        true_ret_unscaled=ret_unscaled,
    )


# ---------------------------------------------------------------------------
# Weight statistics
# ---------------------------------------------------------------------------


def _normalize_weights_mean_one(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32)
    m = float(np.nanmean(w)) if w.size else float("nan")
    if not np.isfinite(m) or m <= 0.0:
        return np.ones_like(w, dtype=np.float32)
    return (w / np.float32(m)).astype(np.float32)


def compute_binary_pos_weight_stats(
    arr: np.ndarray,
    *,
    valid_kind: str,
    min_weight: float,
    max_weight: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(arr)
    out: List[float] = []
    stats: Dict[str, Any] = {}
    for j in range(x.shape[1]):
        col = x[:, j]
        if valid_kind == "nan_mask":
            valid = np.isfinite(col)
            pos = int(np.sum(valid & (col >= 0.999)))
            neg = int(np.sum(valid & (col <= 0.001)))
            neutral = int(np.sum(~valid))
        else:
            valid = col >= 0
            pos = int(np.sum(valid & (col >= 0.999)))
            neg = int(np.sum(valid & (col <= 0.001)))
            neutral = int(np.sum(~valid))
        if pos > 0 and neg > 0:
            w = float(np.clip(neg / max(pos, 1), min_weight, max_weight))
        else:
            w = 1.0
        out.append(w)
        stats[str(j)] = {
            "pos": float(pos),
            "neg": float(neg),
            "neutral_or_invalid": float(neutral),
            "pos_weight": float(w),
        }
    return np.asarray(out, dtype=np.float32), stats


def compute_class_weight_matrix(
    arr: np.ndarray,
    *,
    num_classes: int,
    invalid_value: int,
    alpha: float,
    min_weight: float,
    max_weight: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(arr, dtype=np.int64)
    out = np.ones((x.shape[1], int(num_classes)), dtype=np.float32)
    stats: Dict[str, Any] = {}
    for j in range(x.shape[1]):
        col = x[:, j]
        valid = col != int(invalid_value)
        vals = col[valid]
        counts = np.bincount(vals, minlength=int(num_classes)).astype(np.float64) if vals.size else np.zeros(int(num_classes), dtype=np.float64)
        total = float(counts.sum())
        if total > 0:
            mean_c = float(total / max(int(num_classes), 1))
            weights = np.ones(int(num_classes), dtype=np.float64)
            nz = counts > 0
            weights[nz] = np.clip((mean_c / counts[nz]) ** float(alpha), float(min_weight), float(max_weight))
            weights = _normalize_weights_mean_one(weights)
            out[j] = weights.astype(np.float32)
        stats[str(j)] = {
            "counts": {str(i): int(counts[i]) for i in range(int(num_classes))},
            "weights": {str(i): float(out[j, i]) for i in range(int(num_classes))},
        }
    return out, stats


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool)
    if not torch.any(valid):
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[valid], target[valid], reduction="mean")


def masked_bce_logits(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool)
    if not torch.any(valid):
        return pred.new_zeros(())
    pred_v = pred[valid]
    target_v = target[valid]
    return F.binary_cross_entropy_with_logits(pred_v, target_v, reduction="mean", pos_weight=pos_weight)


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_value: int = INVALID_CLASS_VALUE,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target = target.long()
    valid = target != int(ignore_value)
    if not torch.any(valid):
        return logits.new_zeros(())
    weight = None
    if class_weight is not None:
        weight = class_weight.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits[valid], target[valid], reduction="mean", weight=weight)


def _loss_mean(items: Sequence[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if not items:
        return ref.new_zeros(())
    return torch.stack(list(items)).mean()


def compute_batch_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    loss_weights: Mapping[str, float],
    retcls_class_weights: Optional[torch.Tensor],
    first_hit_class_weights: Optional[torch.Tensor],
    tth_class_weights: Optional[torch.Tensor],
    dir_pos_weight: Optional[torch.Tensor],
    bin_pos_weight: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    ref = outputs["ret_main"]

    losses: Dict[str, torch.Tensor] = {}
    losses["reg_ret_main"] = masked_huber(outputs["ret_main_decoded"], batch["reg_ret_main"], batch["reg_ret_main_mask"])

    dir_losses: List[torch.Tensor] = []
    for j in range(len(DIR_MAIN_TARGETS)):
        pw = dir_pos_weight[j] if dir_pos_weight is not None else None
        dir_losses.append(
            masked_bce_logits(outputs["dir_main"][:, j], batch["dir_main"][:, j], batch["dir_main_mask"][:, j] > 0.5, pos_weight=pw)
        )
    losses["dir_main"] = _loss_mean(dir_losses, ref)

    losses["entry_hyb"] = masked_huber(outputs["hyb_main"], batch["hyb_main"], batch["hyb_main_mask"])
    losses["reg_path_main"] = masked_huber(outputs["path_main"], batch["reg_path_main"], batch["reg_path_main_mask"])
    losses["reg_util_main"] = masked_huber(outputs["util_main"], batch["reg_util_main"], batch["reg_util_main_mask"])

    retcls_losses: List[torch.Tensor] = []
    for j in range(len(RETCLS_TARGETS)):
        cw = retcls_class_weights[j] if retcls_class_weights is not None else None
        retcls_losses.append(masked_cross_entropy(outputs["retcls"][:, j, :], batch["retcls"][:, j], class_weight=cw))
    losses["retcls"] = _loss_mean(retcls_losses, ref)

    bin_losses: List[torch.Tensor] = []
    for j in range(len(BIN_MAIN_TARGETS)):
        pw = bin_pos_weight[j] if bin_pos_weight is not None else None
        mask = batch["bin_main"][:, j] > (INVALID_CLASS_VALUE + 0.5)
        bin_losses.append(masked_bce_logits(outputs["bin_main"][:, j], batch["bin_main"][:, j], mask, pos_weight=pw))
    losses["bin_main"] = _loss_mean(bin_losses, ref)

    fh_losses: List[torch.Tensor] = []
    for j in range(len(FIRST_HIT_TARGETS)):
        cw = first_hit_class_weights[j] if first_hit_class_weights is not None else None
        fh_losses.append(masked_cross_entropy(outputs["first_hit"][:, j, :], batch["first_hit"][:, j], class_weight=cw))
    losses["first_hit"] = _loss_mean(fh_losses, ref)

    tth_losses: List[torch.Tensor] = []
    for j in range(len(TTH_TARGETS)):
        tgt = batch["tth"][:, j].long()
        tgt = torch.where(tgt > 0, tgt - 1, tgt)
        cw = tth_class_weights[j] if tth_class_weights is not None else None
        tth_losses.append(masked_cross_entropy(outputs["tth"][:, j, :], tgt, class_weight=cw))
    losses["tth"] = _loss_mean(tth_losses, ref)

    # focused subset used as deterministic fallback selection
    util10_cols = [UTILITY_LONG_HORIZON_TO_IDX[10], UTILITY_SHORT_HORIZON_TO_IDX[10]]
    util10_mask = batch["reg_util_main_mask"][:, util10_cols]
    losses["focus_util10"] = masked_huber(outputs["util_main"][:, util10_cols], batch["reg_util_main"][:, util10_cols], util10_mask)

    focus_bin_losses: List[torch.Tensor] = []
    for name in (MAIN_CONFIRM_TARGET_UP, MAIN_CONFIRM_TARGET_DOWN):
        if name in BIN_TARGET_TO_IDX:
            j = BIN_TARGET_TO_IDX[name]
            mask = batch["bin_main"][:, j] > (INVALID_CLASS_VALUE + 0.5)
            pw = bin_pos_weight[j] if bin_pos_weight is not None else None
            focus_bin_losses.append(masked_bce_logits(outputs["bin_main"][:, j], batch["bin_main"][:, j], mask, pos_weight=pw))
    losses["focus_bin100"] = _loss_mean(focus_bin_losses, ref)

    if FIRST_HIT_TARGET_100 in FIRST_TARGET_TO_IDX:
        j = FIRST_TARGET_TO_IDX[FIRST_HIT_TARGET_100]
        cw = first_hit_class_weights[j] if first_hit_class_weights is not None else None
        losses["focus_first_hit100"] = masked_cross_entropy(outputs["first_hit"][:, j, :], batch["first_hit"][:, j], class_weight=cw)
    else:
        losses["focus_first_hit100"] = ref.new_zeros(())

    focus_tth_losses: List[torch.Tensor] = []
    for name in (TTH_TARGET_UP_100, TTH_TARGET_DOWN_100):
        if name in TTH_TARGET_TO_IDX:
            j = TTH_TARGET_TO_IDX[name]
            tgt = batch["tth"][:, j].long()
            tgt = torch.where(tgt > 0, tgt - 1, tgt)
            cw = tth_class_weights[j] if tth_class_weights is not None else None
            focus_tth_losses.append(masked_cross_entropy(outputs["tth"][:, j, :], tgt, class_weight=cw))
    losses["focus_tth100"] = _loss_mean(focus_tth_losses, ref)

    focus_dir_losses: List[torch.Tensor] = []
    if 10 in DIR_HORIZON_TO_IDX:
        j = DIR_HORIZON_TO_IDX[10]
        pw = dir_pos_weight[j] if dir_pos_weight is not None else None
        focus_dir_losses.append(masked_bce_logits(outputs["dir_main"][:, j], batch["dir_main"][:, j], batch["dir_main_mask"][:, j] > 0.5, pos_weight=pw))
    losses["focus_dir10"] = _loss_mean(focus_dir_losses, ref)

    losses["selection_focus_loss"] = (
        0.45 * losses["entry_hyb"]
        + 0.45 * losses["focus_util10"]
        + 0.35 * losses["reg_path_main"]
        + 0.20 * losses["focus_bin100"]
        + 0.12 * losses["focus_first_hit100"]
        + 0.10 * losses["focus_tth100"]
        + 0.12 * losses["focus_dir10"]
    )

    total = ref.new_zeros(())
    total = total + float(loss_weights["w_ret_main"]) * losses["reg_ret_main"]
    total = total + float(loss_weights["w_dir_main"]) * losses["dir_main"]
    total = total + float(loss_weights["w_entry_hyb"]) * losses["entry_hyb"]
    total = total + float(loss_weights["w_path_main"]) * losses["reg_path_main"]
    total = total + float(loss_weights["w_util_main"]) * losses["reg_util_main"]
    total = total + float(loss_weights["w_retcls"]) * losses["retcls"]
    total = total + float(loss_weights["w_bin"]) * losses["bin_main"]
    total = total + float(loss_weights["w_first_hit"]) * losses["first_hit"]
    total = total + float(loss_weights["w_tth"]) * losses["tth"]
    losses["total"] = total
    return losses


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def _acc_from_logits(logits: torch.Tensor, target: torch.Tensor, *, ignore_value: int = INVALID_CLASS_VALUE) -> float:
    tgt = target.long()
    valid = tgt != int(ignore_value)
    if not torch.any(valid):
        return float("nan")
    pred = logits.argmax(dim=-1)
    return float((pred[valid] == tgt[valid]).float().mean().cpu())


@torch.no_grad()
def _binacc_from_logits(logits: torch.Tensor, target: torch.Tensor, *, invalid_value: float = float(INVALID_CLASS_VALUE)) -> float:
    valid = torch.isfinite(target) & (target > invalid_value + 0.5)
    if not torch.any(valid):
        return float("nan")
    pred = (torch.sigmoid(logits) >= 0.5).to(dtype=target.dtype)
    return float((pred[valid] == target[valid]).float().mean().cpu())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    amp_enabled: bool,
    scaler_amp: Optional[torch.amp.GradScaler],
    grad_clip: float,
    loss_weights: Mapping[str, float],
    selection_config: EntryScoringConfig,
    retcls_class_weights: Optional[torch.Tensor],
    first_hit_class_weights: Optional[torch.Tensor],
    tth_class_weights: Optional[torch.Tensor],
    dir_pos_weight: Optional[torch.Tensor],
    bin_pos_weight: Optional[torch.Tensor],
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_keys = [
        "total",
        "reg_ret_main",
        "dir_main",
        "entry_hyb",
        "reg_path_main",
        "reg_util_main",
        "retcls",
        "bin_main",
        "first_hit",
        "tth",
        "selection_focus_loss",
        "focus_util10",
        "focus_bin100",
        "focus_first_hit100",
        "focus_tth100",
        "focus_dir10",
    ]
    total: Dict[str, float] = {k: 0.0 for k in total_keys}
    n_batches = 0
    acc_retcls10: List[float] = []
    acc_bin_up100: List[float] = []
    acc_first_hit100: List[float] = []
    acc_dir10: List[float] = []

    gather_pred_hyb: List[np.ndarray] = []
    gather_pred_util: List[np.ndarray] = []
    gather_pred_retcls_prob: List[np.ndarray] = []
    gather_pred_bin_prob: List[np.ndarray] = []
    gather_pred_first_prob: List[np.ndarray] = []
    gather_pred_tth_exp: List[np.ndarray] = []
    gather_pred_tth_cens: List[np.ndarray] = []
    gather_true_ret: List[np.ndarray] = []
    gather_true_util: List[np.ndarray] = []

    tth_support = np.arange(1, TTH_NUM_CLASSES + 1, dtype=np.float32)

    for batch in loader:
        batch = _move_batch(batch, device)
        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = bool(amp_enabled and device.type in {"cuda", "cpu"})
        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(batch["x"])
            losses = compute_batch_losses(
                outputs,
                batch,
                loss_weights=loss_weights,
                retcls_class_weights=retcls_class_weights,
                first_hit_class_weights=first_hit_class_weights,
                tth_class_weights=tth_class_weights,
                dir_pos_weight=dir_pos_weight,
                bin_pos_weight=bin_pos_weight,
            )
            loss = losses["total"]

        if train_mode:
            if scaler_amp is not None and device.type == "cuda" and amp_enabled:
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                if float(grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss.backward()
                if float(grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()

        n_batches += 1
        for k in total_keys:
            total[k] += float(losses[k].detach().cpu())

        if RETCLS_TARGETS:
            j = RETCLS_HORIZON_TO_IDX[10]
            acc_retcls10.append(_acc_from_logits(outputs["retcls"][:, j, :], batch["retcls"][:, j]))
        if MAIN_CONFIRM_TARGET_UP in BIN_TARGET_TO_IDX:
            j = BIN_TARGET_TO_IDX[MAIN_CONFIRM_TARGET_UP]
            acc_bin_up100.append(_binacc_from_logits(outputs["bin_main"][:, j], batch["bin_main"][:, j]))
        if FIRST_HIT_TARGET_100 in FIRST_TARGET_TO_IDX:
            j = FIRST_TARGET_TO_IDX[FIRST_HIT_TARGET_100]
            acc_first_hit100.append(_acc_from_logits(outputs["first_hit"][:, j, :], batch["first_hit"][:, j]))
        if 10 in DIR_HORIZON_TO_IDX:
            j = DIR_HORIZON_TO_IDX[10]
            valid = batch["dir_main_mask"][:, j] > 0.5
            if torch.any(valid):
                pred = (torch.sigmoid(outputs["dir_main"][:, j]) >= 0.5).to(dtype=batch["dir_main"].dtype)
                acc_dir10.append(float((pred[valid] == batch["dir_main"][:, j][valid]).float().mean().cpu()))

        if not train_mode:
            gather_pred_hyb.append(outputs["hyb_main"].detach().cpu().numpy())
            gather_pred_util.append(outputs["util_main"].detach().cpu().numpy())
            retcls_prob = torch.softmax(outputs["retcls"].detach().cpu(), dim=-1).numpy()
            gather_pred_retcls_prob.append(retcls_prob)
            gather_pred_bin_prob.append(torch.sigmoid(outputs["bin_main"]).detach().cpu().numpy())
            gather_pred_first_prob.append(torch.softmax(outputs["first_hit"].detach().cpu(), dim=-1).numpy())
            tth_prob = torch.softmax(outputs["tth"].detach().cpu(), dim=-1).numpy()
            gather_pred_tth_exp.append((tth_prob * tth_support.reshape(1, 1, -1)).sum(axis=-1))
            gather_pred_tth_cens.append(tth_prob[:, :, -1])
            gather_true_ret.append(batch["true_ret_unscaled"].detach().cpu().numpy())
            gather_true_util.append(batch["reg_util_main"].detach().cpu().numpy())

    out = {f"loss_{k}": (v / max(n_batches, 1)) for k, v in total.items()}
    if acc_retcls10:
        out["acc_retcls_10"] = float(np.nanmean(acc_retcls10))
    if acc_bin_up100:
        out["acc_up_hit_100_10"] = float(np.nanmean(acc_bin_up100))
    if acc_first_hit100:
        out["acc_first_hit_100_10"] = float(np.nanmean(acc_first_hit100))
    if acc_dir10:
        out["acc_dir_10"] = float(np.nanmean(acc_dir10))

    if not train_mode and gather_pred_hyb:
        proxy = consumer_selection_proxy(
            pred_hyb=np.concatenate(gather_pred_hyb, axis=0),
            pred_util=np.concatenate(gather_pred_util, axis=0),
            pred_retcls_prob=np.concatenate(gather_pred_retcls_prob, axis=0),
            pred_bin_prob=np.concatenate(gather_pred_bin_prob, axis=0),
            pred_first_hit_prob=np.concatenate(gather_pred_first_prob, axis=0),
            pred_tth_exp=np.concatenate(gather_pred_tth_exp, axis=0),
            pred_tth_cens_prob=np.concatenate(gather_pred_tth_cens, axis=0),
            true_ret_unscaled=np.concatenate(gather_true_ret, axis=0),
            true_util=np.concatenate(gather_true_util, axis=0),
            config=selection_config,
        )
        out.update(proxy)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train v5.1 multi-head dir+hyb path-distribution model")
    ap.add_argument("--csv", required=True, help="input raw/features/dataset file")
    ap.add_argument("--input-kind", default="auto", choices=["auto", "raw", "features", "dataset"])
    ap.add_argument("--out-pt", required=True, help="output checkpoint .pt path")
    ap.add_argument("--out-scaler-json", default="", help="optional output scaler json path")
    ap.add_argument("--report-json", default="", help="optional report json path")
    ap.add_argument("--rows", type=int, default=0, help="if >0, keep only last N rows before training")

    ap.add_argument("--seq-len", type=int, default=160)
    ap.add_argument("--train-ratio", type=float, default=0.85)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=7.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--winsor-q", type=float, default=0.005)
    ap.add_argument(
        "--train-json",
        default="",
        help="optional recommend_train.json / trainer config json; values are applied only when the matching CLI option is still at default",
    )
    ap.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "off", "cosine", "cosine_epoch"],
        help="epoch-level LR scheduler",
    )
    ap.add_argument(
        "--warmup-epochs",
        type=float,
        default=0.0,
        help="linear warmup length in epochs for the scheduler",
    )
    ap.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.10,
        help="minimum LR ratio for cosine schedule",
    )
    # model args
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--stem-hidden-per-group", type=int, default=24)
    ap.add_argument("--stem-dropout", type=float, default=0.05)
    ap.add_argument("--tcn-blocks", type=int, default=5)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--tcn-dropout", type=float, default=0.05)
    ap.add_argument("--xlstm-layers", type=int, default=2)
    ap.add_argument("--xlstm-dropout", type=float, default=0.05)
    ap.add_argument("--readout-hidden", type=int, default=160)
    ap.add_argument("--readout-dropout", type=float, default=0.05)
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--head-dropout", type=float, default=0.05)
    ap.add_argument("--reg-target-mode", type=str, default="magnitude_v2", choices=["signed_legacy", "magnitude_v2"])
    ap.add_argument("--y-scale", type=float, default=1.0)

    ap.add_argument("--cost-per-side", type=float, default=0.00070)
    ap.add_argument("--slip-per-side", type=float, default=0.00015)

    # training loss weights
    ap.add_argument("--weights-json", default="", help="optional json file with v5 loss weights")
    ap.add_argument("--weights-inline-json", default="", help="optional inline json string with v5 loss weights")
    ap.add_argument("--w-ret-main", type=float, default=DEFAULT_LOSS_WEIGHTS["w_ret_main"])
    ap.add_argument("--w-dir-main", type=float, default=DEFAULT_LOSS_WEIGHTS["w_dir_main"])
    ap.add_argument("--w-entry-hyb", type=float, default=DEFAULT_LOSS_WEIGHTS["w_entry_hyb"])
    ap.add_argument("--w-path-main", type=float, default=DEFAULT_LOSS_WEIGHTS["w_path_main"])
    ap.add_argument("--w-util-main", type=float, default=DEFAULT_LOSS_WEIGHTS["w_util_main"])
    ap.add_argument("--w-retcls", type=float, default=DEFAULT_LOSS_WEIGHTS["w_retcls"])
    ap.add_argument("--w-bin", type=float, default=DEFAULT_LOSS_WEIGHTS["w_bin"])
    ap.add_argument("--w-first-hit", type=float, default=DEFAULT_LOSS_WEIGHTS["w_first_hit"])
    ap.add_argument("--w-tth", type=float, default=DEFAULT_LOSS_WEIGHTS["w_tth"])

    # entry/thesis selection config
    ap.add_argument("--selection-json", default="", help="optional json file with consumer-aware selection config")
    ap.add_argument("--selection-inline-json", default="", help="optional inline json string with consumer-aware selection config")

    # IMPORTANT:
    # default=None means "CLI override not provided".
    # This lets selection_json / selection_inline_json survive intact unless
    # the user explicitly overrides a given field from the CLI.
    ap.add_argument("--sel-w1", type=float, default=None)
    ap.add_argument("--sel-w3", type=float, default=None)
    ap.add_argument("--sel-w5", type=float, default=None)
    ap.add_argument("--sel-w8", type=float, default=None)
    ap.add_argument("--sel-w10", type=float, default=None)
    ap.add_argument("--sel-hyb-weight", type=float, default=None)
    ap.add_argument("--sel-util-weight", type=float, default=None)
    ap.add_argument("--sel-cls-weight", type=float, default=None)
    ap.add_argument("--sel-entry-q", type=float, default=None)
    ap.add_argument("--sel-entry-min-score", type=float, default=None)
    ap.add_argument("--sel-entry-min-gap", type=float, default=None)
    ap.add_argument("--sel-entry-min-utility-10", type=float, default=None)
    ap.add_argument("--sel-entry-min-utility-gap-10", type=float, default=None)
    ap.add_argument("--sel-confirm-main-prob", type=float, default=None)
    ap.add_argument("--sel-timing-first-hit-prob", type=float, default=None)
    ap.add_argument("--sel-timing-max-expected-bars", type=float, default=None)
    ap.add_argument("--sel-timing-max-censored-prob", type=float, default=None)
    ap.add_argument("--sel-require-retcls-alignment", type=int, default=None)
    ap.add_argument("--sel-consumer-min-accept-count", type=int, default=None)
    ap.add_argument("--sel-consumer-accept-rate-floor", type=float, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args = _apply_train_json_overrides(args)
    set_seed(int(args.seed))

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    )

    df = read_frame(args.csv)
    if int(args.rows) > 0 and len(df) > int(args.rows):
        df = df.iloc[-int(args.rows) :].reset_index(drop=True)

    labeled = maybe_build_labeled_dataset(
        df,
        input_kind=str(args.input_kind),
        cost_per_side=float(args.cost_per_side),
        slip_per_side=float(args.slip_per_side),
    )

    prep = prepare_training_data(
        labeled,
        seq_len=int(args.seq_len),
        train_ratio=float(args.train_ratio),
        winsor_q=float(args.winsor_q),
    )

    train_ds = make_dataset(prep, prep.train_pos, reg_target_mode=str(args.reg_target_mode), y_scale=float(args.y_scale))
    valid_ds = make_dataset(prep, prep.valid_pos, reg_target_mode=str(args.reg_target_mode), y_scale=float(args.y_scale))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    loss_weights = {
        "w_ret_main": float(args.w_ret_main),
        "w_dir_main": float(args.w_dir_main),
        "w_entry_hyb": float(args.w_entry_hyb),
        "w_path_main": float(args.w_path_main),
        "w_util_main": float(args.w_util_main),
        "w_retcls": float(args.w_retcls),
        "w_bin": float(args.w_bin),
        "w_first_hit": float(args.w_first_hit),
        "w_tth": float(args.w_tth),
    }
    loss_weights = _resolve_weight_overrides(loss_weights, json_path=str(args.weights_json), inline_json=str(args.weights_inline_json))

    sel_require_retcls_alignment = (
        None
        if args.sel_require_retcls_alignment is None
        else bool(int(args.sel_require_retcls_alignment))
    )

    selection_cfg = _resolve_entry_scoring_config(
        json_path=str(args.selection_json),
        inline_json=str(args.selection_inline_json),
        overrides={
            "w1": args.sel_w1,
            "w3": args.sel_w3,
            "w5": args.sel_w5,
            "w8": args.sel_w8,
            "w10": args.sel_w10,
            "hyb_weight": args.sel_hyb_weight,
            "util_weight": args.sel_util_weight,
            "cls_weight": args.sel_cls_weight,
            "entry_q": args.sel_entry_q,
            "entry_min_score": args.sel_entry_min_score,
            "entry_min_gap": args.sel_entry_min_gap,
            "entry_min_utility_10": args.sel_entry_min_utility_10,
            "entry_min_utility_gap_10": args.sel_entry_min_utility_gap_10,
            "confirm_main_prob": args.sel_confirm_main_prob,
            "timing_first_hit_prob": args.sel_timing_first_hit_prob,
            "timing_max_expected_bars": args.sel_timing_max_expected_bars,
            "timing_max_censored_prob": args.sel_timing_max_censored_prob,
            "require_retcls_alignment": sel_require_retcls_alignment,
            "consumer_min_accept_count": args.sel_consumer_min_accept_count,
            "consumer_accept_rate_floor": args.sel_consumer_accept_rate_floor,
        },
    )

    # train-only statistics for balancing
    ends_train = prep.ends_all[prep.train_pos]
    df_train = prep.df.iloc[ends_train].reset_index(drop=True)

    dir_pos_weight_np, dir_pos_stats = compute_binary_pos_weight_stats(
        _to_numeric_2d(df_train, DIR_MAIN_TARGETS, dtype="float32"),
        valid_kind="nan_mask",
        min_weight=0.5,
        max_weight=4.0,
    )
    bin_pos_weight_np, bin_pos_stats = compute_binary_pos_weight_stats(
        pd.DataFrame(df_train.loc[:, list(BIN_MAIN_TARGETS)]).apply(pd.to_numeric, errors="coerce").fillna(INVALID_CLASS_VALUE).astype(np.int64).to_numpy(copy=False),
        valid_kind="invalid_mask",
        min_weight=1.0,
        max_weight=6.0,
    )
    retcls_w_np, retcls_w_stats = compute_class_weight_matrix(
        pd.DataFrame(df_train.loc[:, list(RETCLS_TARGETS)]).apply(pd.to_numeric, errors="coerce").fillna(INVALID_CLASS_VALUE).astype(np.int64).to_numpy(copy=False),
        num_classes=RETCLS_NUM_CLASSES,
        invalid_value=INVALID_CLASS_VALUE,
        alpha=0.50,
        min_weight=0.25,
        max_weight=4.0,
    )
    first_hit_w_np, first_hit_w_stats = compute_class_weight_matrix(
        pd.DataFrame(df_train.loc[:, list(FIRST_HIT_TARGETS)]).apply(pd.to_numeric, errors="coerce").fillna(INVALID_CLASS_VALUE).astype(np.int64).to_numpy(copy=False),
        num_classes=FIRST_HIT_NUM_CLASSES,
        invalid_value=INVALID_CLASS_VALUE,
        alpha=0.50,
        min_weight=0.35,
        max_weight=4.0,
    )
    tth_train = pd.DataFrame(df_train.loc[:, list(TTH_TARGETS)]).apply(pd.to_numeric, errors="coerce").fillna(INVALID_CLASS_VALUE).astype(np.int64).to_numpy(copy=False)
    tth_train_adj = np.where(tth_train > 0, tth_train - 1, tth_train)
    tth_w_np, tth_w_stats = compute_class_weight_matrix(
        tth_train_adj,
        num_classes=TTH_NUM_CLASSES,
        invalid_value=INVALID_CLASS_VALUE,
        alpha=0.50,
        min_weight=0.35,
        max_weight=4.0,
    )

    dir_pos_weight = torch.tensor(dir_pos_weight_np, dtype=torch.float32, device=device)
    bin_pos_weight = torch.tensor(bin_pos_weight_np, dtype=torch.float32, device=device)
    retcls_class_weights = torch.tensor(retcls_w_np, dtype=torch.float32, device=device)
    first_hit_class_weights = torch.tensor(first_hit_w_np, dtype=torch.float32, device=device)
    tth_class_weights = torch.tensor(tth_w_np, dtype=torch.float32, device=device)

    meta = ModelMetaV5_1(
        seq_len=int(args.seq_len),
        d_model=int(args.d_model),
        stem_hidden_per_group=int(args.stem_hidden_per_group),
        stem_dropout=float(args.stem_dropout),
        tcn_blocks=int(args.tcn_blocks),
        kernel_size=int(args.kernel_size),
        tcn_dropout=float(args.tcn_dropout),
        xlstm_layers=int(args.xlstm_layers),
        xlstm_dropout=float(args.xlstm_dropout),
        readout_hidden=int(args.readout_hidden),
        readout_dropout=float(args.readout_dropout),
        head_hidden=int(args.head_hidden),
        head_dropout=float(args.head_dropout),
        reg_target_mode=str(args.reg_target_mode),
        y_scale=float(args.y_scale),
        reg_activation=("softplus" if str(args.reg_target_mode) == "magnitude_v2" else "identity"),
        features=list(prep.feature_cols),
        cost_per_side=float(args.cost_per_side),
        slip_per_side=float(args.slip_per_side),
    )
    model: PathDistMultiHeadV5_1 = build_model_from_meta(asdict(meta)).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    sched = _build_scheduler(
        opt,
        scheduler_name=str(args.scheduler),
        epochs=int(args.epochs),
        warmup_epochs=float(args.warmup_epochs),
        min_lr_ratio=float(args.min_lr_ratio),
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=bool(int(args.amp))) if device.type == "cuda" else None

    best_consumer = float("-inf")
    best_focus = float("inf")
    best_total = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    patience_left = int(args.patience)
    fallback_focus_nonlow = float("inf")
    fallback_state_nonlow: Optional[Dict[str, Any]] = None
    fallback_focus_any = float("inf")
    fallback_state_any: Optional[Dict[str, Any]] = None

    def _pack_checkpoint_state(
        *,
        cur_effective_score: float,
        cur_raw_consumer_score: float,
        cur_focus_loss: float,
        cur_total_loss: float,
        cur_selection_mode: str,
    ) -> Dict[str, Any]:
        return {
            "state_dict": model.state_dict(),
            "meta": {
                **asdict(meta),
                "loss_weights": dict(loss_weights),
                "selection_config": asdict(selection_cfg),
                "dir_pos_weight": dir_pos_weight_np.tolist(),
                "bin_pos_weight": bin_pos_weight_np.tolist(),
                "retcls_class_weights": retcls_w_np.tolist(),
                "first_hit_class_weights": first_hit_w_np.tolist(),
                "tth_class_weights": tth_w_np.tolist(),
                "scheduler_config": {
                    "scheduler": str(args.scheduler),
                    "warmup_epochs": float(args.warmup_epochs),
                    "min_lr_ratio": float(args.min_lr_ratio),
                },
            },
            "scaler": prep.scaler,
            "best_valid_consumer_selection_score": float(cur_effective_score) if np.isfinite(cur_effective_score) else None,
            "best_valid_raw_consumer_selection_score": float(cur_raw_consumer_score) if np.isfinite(cur_raw_consumer_score) else None,
            "best_valid_selection_focus_loss": float(cur_focus_loss),
            "best_valid_loss_total": float(cur_total_loss),
            "best_valid_selection_mode": str(cur_selection_mode),
            "history": history,
        }
    
    for epoch in range(1, int(args.epochs) + 1):
        lr_epoch_start = float(opt.param_groups[0]["lr"])
        train_stats = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=opt,
            amp_enabled=bool(int(args.amp)),
            scaler_amp=amp_scaler,
            grad_clip=float(args.grad_clip),
            loss_weights=loss_weights,
            selection_config=selection_cfg,
            retcls_class_weights=retcls_class_weights,
            first_hit_class_weights=first_hit_class_weights,
            tth_class_weights=tth_class_weights,
            dir_pos_weight=dir_pos_weight,
            bin_pos_weight=bin_pos_weight,
        )
        with torch.no_grad():
            valid_stats = run_epoch(
                model,
                valid_loader,
                device=device,
                optimizer=None,
                amp_enabled=bool(int(args.amp)),
                scaler_amp=None,
                grad_clip=0.0,
                loss_weights=loss_weights,
                selection_config=selection_cfg,
                retcls_class_weights=retcls_class_weights,
                first_hit_class_weights=first_hit_class_weights,
                tth_class_weights=tth_class_weights,
                dir_pos_weight=dir_pos_weight,
                bin_pos_weight=bin_pos_weight,
            )

        if sched is not None:
            sched.step()
        lr_epoch_end = float(opt.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "lr_epoch_start": float(lr_epoch_start),
            "lr_epoch_end": float(lr_epoch_end),
            "scheduler": str(args.scheduler),
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"valid_{k}": v for k, v in valid_stats.items()},
        }

        sel_meta = resolve_effective_selection_metric(valid_stats, selection_cfg)
        row.update({f"valid_{k}": v for k, v in sel_meta.items()})

        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        cur_raw_consumer = float(
            valid_stats.get(
                "consumer_selection_score_raw_econ",
                valid_stats.get("consumer_selection_score", float("nan")),
            )
        )
        cur_effective = float(sel_meta.get("selection_effective_score", float("nan")))
        cur_focus = float(valid_stats.get("loss_selection_focus_loss", float("inf")))
        cur_total = float(valid_stats.get("loss_total", float("inf")))
        cur_mode = str(sel_meta.get("selection_mode", ""))
        cur_checkpoint_eligible = bool(sel_meta.get("selection_checkpoint_eligible", False))

        if np.isfinite(cur_focus) and cur_focus < fallback_focus_any - 1.0e-8:
            fallback_focus_any = cur_focus
            fallback_state_any = _pack_checkpoint_state(
                cur_effective_score=cur_effective,
                cur_raw_consumer_score=cur_raw_consumer,
                cur_focus_loss=cur_focus,
                cur_total_loss=cur_total,
                cur_selection_mode=cur_mode,
            )

        if cur_mode != "low_accept_blend" and np.isfinite(cur_focus) and cur_focus < fallback_focus_nonlow - 1.0e-8:
            fallback_focus_nonlow = cur_focus
            fallback_state_nonlow = _pack_checkpoint_state(
                cur_effective_score=cur_effective,
                cur_raw_consumer_score=cur_raw_consumer,
                cur_focus_loss=cur_focus,
                cur_total_loss=cur_total,
                cur_selection_mode=cur_mode,
            )

        improved = False
        if cur_checkpoint_eligible and np.isfinite(cur_effective):
            if (not np.isfinite(best_consumer)) or (cur_effective > best_consumer + 1.0e-8):
                improved = True
            elif abs(cur_effective - best_consumer) <= 1.0e-8 and cur_focus < best_focus - 1.0e-8:
                improved = True

        if improved:
            best_consumer = cur_effective
            best_focus = cur_focus
            best_total = cur_total
            patience_left = int(args.patience)
            best_state = _pack_checkpoint_state(
                cur_effective_score=cur_effective,
                cur_raw_consumer_score=cur_raw_consumer,
                cur_focus_loss=cur_focus,
                cur_total_loss=cur_total,
                cur_selection_mode=cur_mode,
            )
        else:
            # low_accept / no_trade epochs do not burn patience before the first
            # consumer-eligible checkpoint appears.
            if best_state is not None or cur_checkpoint_eligible:
                patience_left -= 1
                if patience_left <= 0:
                    print(
                        f"[INFO] early stopping at epoch={epoch} "
                        f"best_consumer={best_consumer:.6f} "
                        f"best_focus={best_focus:.6f} "
                        f"best_total={best_total:.6f}"
                    )
                    break

    if best_state is None:
        if fallback_state_nonlow is not None:
            print("[WARN] no consumer-eligible checkpoint found; using best non-low-accept fallback checkpoint")
            best_state = fallback_state_nonlow
        elif fallback_state_any is not None:
            print("[WARN] no consumer-eligible checkpoint found; using best any-mode fallback checkpoint")
            best_state = fallback_state_any
        else:
            raise RuntimeError("training ended without a checkpoint state")

    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_pt)

    if str(args.out_scaler_json).strip():
        out_scaler = Path(args.out_scaler_json)
        out_scaler.parent.mkdir(parents=True, exist_ok=True)
        out_scaler.write_text(json.dumps(prep.scaler, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "input_path": str(args.csv),
        "output_checkpoint": str(out_pt),
        "device": str(device),
        "rows": int(len(prep.df)),
        "split_row": int(prep.split_row),
        "train_sequences": int(len(prep.train_pos)),
        "valid_sequences": int(len(prep.valid_pos)),
        "best_valid_consumer_selection_score": float(best_consumer) if np.isfinite(best_consumer) else None,
        "best_valid_effective_selection_score": float(best_consumer) if np.isfinite(best_consumer) else None,
        "best_valid_raw_consumer_selection_score": (
            float(best_state.get("best_valid_raw_consumer_selection_score"))
            if (best_state is not None and best_state.get("best_valid_raw_consumer_selection_score") is not None)
            else None
        ),
        "best_valid_selection_mode": (
            str(best_state.get("best_valid_selection_mode"))
            if (best_state is not None and best_state.get("best_valid_selection_mode") is not None)
            else ""
        ),
        "best_valid_selection_focus_loss": float(best_focus),
        "best_valid_loss_total": float(best_total),
        "meta": {
            **asdict(meta),
            "loss_weights": dict(loss_weights),
            "selection_config": asdict(selection_cfg),
            "scheduler_config": {
                "scheduler": str(args.scheduler),
                "warmup_epochs": float(args.warmup_epochs),
                "min_lr_ratio": float(args.min_lr_ratio),
            },
        },
        "balance_stats": {
            "dir_pos_stats": dir_pos_stats,
            "bin_pos_stats": bin_pos_stats,
            "retcls_class_weight_stats": retcls_w_stats,
            "first_hit_class_weight_stats": first_hit_w_stats,
            "tth_class_weight_stats": tth_w_stats,
        },
        "label_summary": summarize_labeled_frame(labeled),
        "history": history,
    }
    if str(args.report_json).strip():
        out_report = Path(args.report_json)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


if __name__ == "__main__":
    main()
