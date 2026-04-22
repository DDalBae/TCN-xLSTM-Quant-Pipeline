# -*- coding: utf-8 -*-
"""
model_v5_1.py

V5.1 model definition aligned with:
- feature_contract_v5.py
- target_contract_v5.py

Design goals
------------
- Keep the v5 reactive26 feature contract fixed.
- Keep the TCN + xLSTM backbone from model_v5.py.
- Extend the target / head semantics so that legacy `dir + hyb` structure is
  formally imported into the v5 stack.
- Keep `path10 / no-extension / open[t+1]` thesis alignment intact.
- Make checkpoint reconstruction possible from checkpoint meta alone.

Important note
--------------
This model intentionally separates:
- regression raw heads for return horizons
- binary direction logits for the same horizons
- derived hybrid signals `hyb_h = decode(reg_h_raw) * tanh(dir_h_logit / 2)`
- path10 exit geometry heads
- multi-horizon utility heads

즉,
- entry 는 `ret + dir + hyb + utility` multi-horizon 조합으로 더 민첩하게 보고,
- thesis / exit 는 계속 path10 bundle 로 구조적으로 다룰 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_contract_v5 import (
    CORE_FEATURE_GROUPS,
    DEFAULT_MODEL_INPUT_COLUMNS,
    FEATURE_CONTRACT_VERSION,
)
from target_contract_v5 import (
    BARRIER_HIT_TARGETS_MAIN,
    DIR_TARGETS_MAIN,
    FIRST_HIT_ID_TO_NAME,
    FIRST_HIT_TARGETS_MAIN,
    PATH_TARGETS_MAIN,
    REGRESSION_TARGETS_MAIN,
    RETCLS_TARGETS_MAIN,
    RET_CLASS_ID_TO_NAME,
    RETURN_TARGETS_MAIN,
    TARGET_CONTRACT_VERSION,
    TTH_CENSORED_VALUE_MAIN,
    TTH_TARGETS_MAIN,
    UTILITY_TARGETS_MAIN,
)

INVALID_CLASS_VALUE = -1

# -----------------------------------------------------------------------------
# Canonical target group helpers
# -----------------------------------------------------------------------------

RET_MAIN_TARGETS: Tuple[str, ...] = tuple(RETURN_TARGETS_MAIN)
PATH_MAIN_TARGETS: Tuple[str, ...] = tuple(PATH_TARGETS_MAIN)
UTILITY_MAIN_TARGETS: Tuple[str, ...] = tuple(UTILITY_TARGETS_MAIN)
DIR_MAIN_TARGETS: Tuple[str, ...] = tuple(DIR_TARGETS_MAIN)
RETCLS_TARGETS: Tuple[str, ...] = tuple(RETCLS_TARGETS_MAIN)
BIN_MAIN_TARGETS: Tuple[str, ...] = tuple(BARRIER_HIT_TARGETS_MAIN)
FIRST_HIT_TARGETS: Tuple[str, ...] = tuple(FIRST_HIT_TARGETS_MAIN)
TTH_TARGETS: Tuple[str, ...] = tuple(TTH_TARGETS_MAIN)

RETCLS_NUM_CLASSES = len(RET_CLASS_ID_TO_NAME)
FIRST_HIT_NUM_CLASSES = len(FIRST_HIT_ID_TO_NAME)
TTH_NUM_CLASSES = int(TTH_CENSORED_VALUE_MAIN)


# -----------------------------------------------------------------------------
# Generic helpers
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


_assert_unique(RET_MAIN_TARGETS, label="RET_MAIN_TARGETS")
_assert_unique(PATH_MAIN_TARGETS, label="PATH_MAIN_TARGETS")
_assert_unique(UTILITY_MAIN_TARGETS, label="UTILITY_MAIN_TARGETS")
_assert_unique(DIR_MAIN_TARGETS, label="DIR_MAIN_TARGETS")
_assert_unique(RETCLS_TARGETS, label="RETCLS_TARGETS")
_assert_unique(BIN_MAIN_TARGETS, label="BIN_MAIN_TARGETS")
_assert_unique(FIRST_HIT_TARGETS, label="FIRST_HIT_TARGETS")
_assert_unique(TTH_TARGETS, label="TTH_TARGETS")


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


# -----------------------------------------------------------------------------
# Legacy dir+hyb decode helpers
# -----------------------------------------------------------------------------


def direction_score_from_logits(logit: torch.Tensor) -> torch.Tensor:
    return torch.tanh(logit / 2.0)



def decode_reg_output(reg_raw: torch.Tensor, *, reg_target_mode: str = "magnitude_v2", y_scale: float = 1.0) -> torch.Tensor:
    mode = str(reg_target_mode or "magnitude_v2").strip().lower()
    if mode == "magnitude_v2":
        reg_eff = F.softplus(reg_raw)
    elif mode == "signed_legacy":
        reg_eff = reg_raw
    else:
        raise ValueError(f"Unsupported reg_target_mode={reg_target_mode}")
    ys = float(y_scale)
    if ys != 1.0:
        reg_eff = reg_eff / ys
    return reg_eff



def hybrid_from_heads(
    reg_raw: torch.Tensor,
    dir_logit: torch.Tensor,
    *,
    reg_target_mode: str = "magnitude_v2",
    y_scale: float = 1.0,
) -> torch.Tensor:
    return decode_reg_output(reg_raw, reg_target_mode=reg_target_mode, y_scale=y_scale) * direction_score_from_logits(dir_logit)


# -----------------------------------------------------------------------------
# Group stem
# -----------------------------------------------------------------------------


def _build_group_indices(feature_cols: Sequence[str]) -> Dict[str, List[int]]:
    feat_to_idx = {str(c): int(i) for i, c in enumerate(feature_cols)}
    out: Dict[str, List[int]] = {}
    for group_name, group_cols in CORE_FEATURE_GROUPS.items():
        idx: List[int] = []
        for c in group_cols:
            if c not in feat_to_idx:
                raise ValueError(f"missing v5 feature in feature_cols: {c}")
            idx.append(feat_to_idx[c])
        out[str(group_name)] = idx
    return out


class GroupStemV5_1(nn.Module):
    def __init__(
        self,
        feature_cols: Sequence[str],
        *,
        stem_hidden_per_group: int = 24,
        d_model: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.group_names = list(CORE_FEATURE_GROUPS.keys())
        group_indices = _build_group_indices(self.feature_cols)

        self.group_projs = nn.ModuleDict()
        for group_name in self.group_names:
            idx = torch.tensor(group_indices[group_name], dtype=torch.long)
            self.register_buffer(f"idx_{group_name}", idx, persistent=False)
            in_dim = len(group_indices[group_name])
            self.group_projs[group_name] = nn.Sequential(
                nn.Linear(in_dim, int(stem_hidden_per_group)),
                nn.GELU(),
                nn.LayerNorm(int(stem_hidden_per_group)),
                nn.Linear(int(stem_hidden_per_group), int(stem_hidden_per_group)),
                nn.GELU(),
                nn.LayerNorm(int(stem_hidden_per_group)),
            )

        self.out_proj = nn.Sequential(
            nn.Linear(int(stem_hidden_per_group) * len(self.group_names), int(d_model)),
            nn.GELU(),
            nn.LayerNorm(int(d_model)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for group_name in self.group_names:
            idx = getattr(self, f"idx_{group_name}")
            g = x.index_select(dim=2, index=idx)
            parts.append(self.group_projs[group_name](g))
        return self.out_proj(torch.cat(parts, dim=-1))


# -----------------------------------------------------------------------------
# Local mixer (causal TCN)
# -----------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.pad = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            int(in_ch),
            int(out_ch),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNResidualBlockV5_1(nn.Module):
    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, ch)
        self.norm2 = nn.GroupNorm(1, ch)
        self.dropout = nn.Dropout(float(dropout))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2).contiguous()
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act(y)
        y = self.dropout(y)
        return x + y.transpose(1, 2).contiguous()


class TCNStackV5_1(nn.Module):
    def __init__(self, ch: int, kernel_size: int, n_blocks: int, dropout: float):
        super().__init__()
        self.net = nn.ModuleList(
            [
                TCNResidualBlockV5_1(ch=ch, kernel_size=kernel_size, dilation=(2 ** i), dropout=dropout)
                for i in range(int(n_blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.net:
            x = block(x)
        return x


# -----------------------------------------------------------------------------
# xLSTM-style backend
# -----------------------------------------------------------------------------


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.05, expansion: int = 4):
        super().__init__()
        hidden = int(max(d_model, d_model * expansion))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, d_model),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class XLSTMStyleBlockV5_1(nn.Module):
    """Practical residual LSTM block used as the xLSTM-style backend."""

    def __init__(self, d_model: int, dropout: float = 0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(int(d_model))
        self.lstm = nn.LSTM(
            input_size=int(d_model),
            hidden_size=int(d_model),
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.lstm_proj = nn.Linear(int(d_model), int(d_model))
        self.drop1 = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.ffn = FeedForwardBlock(int(d_model), dropout=float(dropout), expansion=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(self.norm1(x))
        x = x + self.drop1(self.lstm_proj(y))
        x = x + self.ffn(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# Readout + full multi-head model
# -----------------------------------------------------------------------------


class HorizonAwareReadoutV5_1(nn.Module):
    def __init__(self, d_model: int, *, readout_hidden: int = 160, dropout: float = 0.05):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(int(d_model) * 5, int(readout_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(readout_hidden), int(readout_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.output_dim = int(readout_hidden)

    @staticmethod
    def _mean_tail(x: torch.Tensor, width: int) -> torch.Tensor:
        width = int(max(1, width))
        if x.size(1) <= width:
            return x.mean(dim=1)
        return x[:, -width:, :].mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_last = x[:, -1, :]
        h3 = self._mean_tail(x, 3)
        h5 = self._mean_tail(x, 5)
        h8 = self._mean_tail(x, 8)
        h10 = self._mean_tail(x, 10)
        return self.readout(torch.cat([h_last, h3, h5, h8, h10], dim=-1))


class PathDistMultiHeadV5_1(nn.Module):
    def __init__(
        self,
        feature_cols: Sequence[str],
        *,
        d_model: int = 128,
        stem_hidden_per_group: int = 24,
        stem_dropout: float = 0.05,
        tcn_blocks: int = 5,
        kernel_size: int = 3,
        tcn_dropout: float = 0.05,
        xlstm_layers: int = 2,
        xlstm_dropout: float = 0.05,
        readout_hidden: int = 160,
        readout_dropout: float = 0.05,
        head_hidden: int = 128,
        head_dropout: float = 0.05,
        reg_target_mode: str = "magnitude_v2",
        y_scale: float = 1.0,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.reg_target_mode = str(reg_target_mode)
        self.y_scale = float(y_scale)

        self.stem = GroupStemV5_1(
            feature_cols=self.feature_cols,
            stem_hidden_per_group=int(stem_hidden_per_group),
            d_model=int(d_model),
            dropout=float(stem_dropout),
        )
        self.local_mixer = TCNStackV5_1(
            ch=int(d_model),
            kernel_size=int(kernel_size),
            n_blocks=int(tcn_blocks),
            dropout=float(tcn_dropout),
        )
        self.xlstm_blocks = nn.ModuleList(
            [XLSTMStyleBlockV5_1(int(d_model), dropout=float(xlstm_dropout)) for _ in range(int(xlstm_layers))]
        )
        self.final_norm = nn.LayerNorm(int(d_model))
        self.readout = HorizonAwareReadoutV5_1(int(d_model), readout_hidden=int(readout_hidden), dropout=float(readout_dropout))
        self.head_trunk = nn.Sequential(
            nn.Linear(int(self.readout.output_dim), int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(head_dropout)),
        )
        d = int(head_hidden)

        self.ret_main = nn.Linear(d, len(RET_MAIN_TARGETS))
        self.dir_main = nn.Linear(d, len(DIR_MAIN_TARGETS))
        self.path_main = nn.Linear(d, len(PATH_MAIN_TARGETS))
        self.util_main = nn.Linear(d, len(UTILITY_MAIN_TARGETS))
        self.retcls = nn.Linear(d, len(RETCLS_TARGETS) * RETCLS_NUM_CLASSES)
        self.bin_main = nn.Linear(d, len(BIN_MAIN_TARGETS))
        self.first_hit = nn.Linear(d, len(FIRST_HIT_TARGETS) * FIRST_HIT_NUM_CLASSES)
        self.tth = nn.Linear(d, len(TTH_TARGETS) * TTH_NUM_CLASSES)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.local_mixer(h)
        for block in self.xlstm_blocks:
            h = block(h)
        return self.final_norm(h)

    def decode_reg_main(self, reg_raw: torch.Tensor) -> torch.Tensor:
        return decode_reg_output(reg_raw, reg_target_mode=self.reg_target_mode, y_scale=self.y_scale)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encode(x)
        r = self.head_trunk(self.readout(h))
        ret_raw = self.ret_main(r)
        dir_logits = self.dir_main(r)
        out: Dict[str, torch.Tensor] = {}
        out["ret_main"] = ret_raw
        out["ret_main_decoded"] = self.decode_reg_main(ret_raw)
        out["dir_main"] = dir_logits
        out["dir_score"] = direction_score_from_logits(dir_logits)
        out["hyb_main"] = hybrid_from_heads(
            ret_raw,
            dir_logits,
            reg_target_mode=self.reg_target_mode,
            y_scale=self.y_scale,
        )
        out["path_main"] = F.softplus(self.path_main(r))
        out["util_main"] = self.util_main(r)
        out["retcls"] = self.retcls(r).view(-1, len(RETCLS_TARGETS), RETCLS_NUM_CLASSES)
        out["bin_main"] = self.bin_main(r)
        out["first_hit"] = self.first_hit(r).view(-1, len(FIRST_HIT_TARGETS), FIRST_HIT_NUM_CLASSES)
        out["tth"] = self.tth(r).view(-1, len(TTH_TARGETS), TTH_NUM_CLASSES)
        return out


# -----------------------------------------------------------------------------
# Checkpoint meta
# -----------------------------------------------------------------------------


@dataclass
class ModelMetaV5_1:
    arch: str = "pathdist_v5_1_tcn_xlstm_dirhyb_reactive26"
    seq_len: int = 160
    d_model: int = 128
    stem_hidden_per_group: int = 24
    stem_dropout: float = 0.05
    tcn_blocks: int = 5
    kernel_size: int = 3
    tcn_dropout: float = 0.05
    xlstm_layers: int = 2
    xlstm_dropout: float = 0.05
    readout_hidden: int = 160
    readout_dropout: float = 0.05
    head_hidden: int = 128
    head_dropout: float = 0.05
    reg_target_mode: str = "magnitude_v2"
    y_scale: float = 1.0
    reg_activation: str = "softplus"  # meaningful when reg_target_mode=magnitude_v2
    features: List[str] = field(default_factory=lambda: list(DEFAULT_MODEL_INPUT_COLUMNS))
    feature_contract_version: str = FEATURE_CONTRACT_VERSION
    target_contract_version: str = TARGET_CONTRACT_VERSION
    ret_main_targets: List[str] = field(default_factory=lambda: list(RET_MAIN_TARGETS))
    dir_main_targets: List[str] = field(default_factory=lambda: list(DIR_MAIN_TARGETS))
    hyb_main_targets: List[str] = field(default_factory=lambda: list(RET_MAIN_TARGETS))
    path_main_targets: List[str] = field(default_factory=lambda: list(PATH_MAIN_TARGETS))
    util_main_targets: List[str] = field(default_factory=lambda: list(UTILITY_MAIN_TARGETS))
    retcls_targets: List[str] = field(default_factory=lambda: list(RETCLS_TARGETS))
    bin_main_targets: List[str] = field(default_factory=lambda: list(BIN_MAIN_TARGETS))
    first_hit_targets: List[str] = field(default_factory=lambda: list(FIRST_HIT_TARGETS))
    tth_targets: List[str] = field(default_factory=lambda: list(TTH_TARGETS))
    retcls_num_classes: int = RETCLS_NUM_CLASSES
    first_hit_num_classes: int = FIRST_HIT_NUM_CLASSES
    tth_num_classes: int = TTH_NUM_CLASSES
    invalid_class_value: int = INVALID_CLASS_VALUE
    scaler_kind: str = "robust_iqr_v1"
    cost_per_side: float = 0.00070
    slip_per_side: float = 0.00015
    selection_metric_name: str = "focus_entry_thesis_score_v5_1"
    hybrid_formula: str = "decode(ret_main_raw) * tanh(dir_main / 2.0)"


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def build_model_from_meta(meta: Mapping[str, Any]) -> PathDistMultiHeadV5_1:
    feature_cols = list(meta.get("features") or DEFAULT_MODEL_INPUT_COLUMNS)
    return PathDistMultiHeadV5_1(
        feature_cols=feature_cols,
        d_model=_safe_int(meta.get("d_model", 128), 128),
        stem_hidden_per_group=_safe_int(meta.get("stem_hidden_per_group", 24), 24),
        stem_dropout=float(meta.get("stem_dropout", 0.05)),
        tcn_blocks=_safe_int(meta.get("tcn_blocks", 5), 5),
        kernel_size=_safe_int(meta.get("kernel_size", 3), 3),
        tcn_dropout=float(meta.get("tcn_dropout", 0.05)),
        xlstm_layers=_safe_int(meta.get("xlstm_layers", 2), 2),
        xlstm_dropout=float(meta.get("xlstm_dropout", 0.05)),
        readout_hidden=_safe_int(meta.get("readout_hidden", 160), 160),
        readout_dropout=float(meta.get("readout_dropout", 0.05)),
        head_hidden=_safe_int(meta.get("head_hidden", 128), 128),
        head_dropout=float(meta.get("head_dropout", 0.05)),
        reg_target_mode=str(meta.get("reg_target_mode", "magnitude_v2")),
        y_scale=float(meta.get("y_scale", 1.0)),
    )


__all__ = [
    "INVALID_CLASS_VALUE",
    "RET_MAIN_TARGETS",
    "PATH_MAIN_TARGETS",
    "UTILITY_MAIN_TARGETS",
    "DIR_MAIN_TARGETS",
    "RETCLS_TARGETS",
    "BIN_MAIN_TARGETS",
    "FIRST_HIT_TARGETS",
    "TTH_TARGETS",
    "RETCLS_NUM_CLASSES",
    "FIRST_HIT_NUM_CLASSES",
    "TTH_NUM_CLASSES",
    "direction_score_from_logits",
    "decode_reg_output",
    "hybrid_from_heads",
    "GroupStemV5_1",
    "CausalConv1d",
    "TCNResidualBlockV5_1",
    "TCNStackV5_1",
    "XLSTMStyleBlockV5_1",
    "HorizonAwareReadoutV5_1",
    "PathDistMultiHeadV5_1",
    "ModelMetaV5_1",
    "build_model_from_meta",
]
