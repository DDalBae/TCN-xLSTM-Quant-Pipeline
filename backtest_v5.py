
# -*- coding: utf-8 -*-
"""
backtest_v5.py

V5 full single-tier backtest CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from backtest_contract_v5 import (
    BACKTEST_CONTRACT_VERSION,
    BacktestConfigV5,
    DynamicConfigV5,
    EntryEpisodeConfigV5,
    ExecutionConfigV5,
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
    derive_artifact_paths,
    load_config_dataclass,
)
from backtest_core_v5 import (
    apply_cost_scenarios_v5,
    assemble_objective_v5,
    build_equity_curve_v5,
    ensure_prediction_frame_v5,
    evaluate_prediction_frame_fast_v5,
    load_cost_scenarios_json,
    prepare_fast_eval_cache_v5,
    save_fast_eval_cache_v5,
    simulate_from_prediction_frame_v5,
    summarize_reason_stats_v5,
    summarize_segments_v5,
    summarize_trades_v5,
)
from feature_ops_v5 import read_frame, write_frame
from inference_v5 import PathDistInferenceV5


def _json_default(x: Any) -> Any:
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    return x


def _load_pred_or_build(
    *,
    input_path: str,
    input_kind: str,
    checkpoint: str,
    scaler_json: str,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    kind = str(input_kind or "auto").strip().lower()
    if kind == "pred":
        return ensure_prediction_frame_v5(read_frame(input_path))
    src = read_frame(input_path)
    infer = PathDistInferenceV5(checkpoint, scaler_json=(scaler_json or None), device=str(device))
    pred = infer.predict_frame(
        src,
        input_kind=str(kind),
        batch_size=int(batch_size),
        preserve_input_columns=True,
    )
    return ensure_prediction_frame_v5(pred)


def _crop_eval_frame(
    pred: pd.DataFrame,
    *,
    eval_start: str = "",
    eval_end: str = "",
    tail_rows: int = 0,
    head_rows: int = 0,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    df = pred.copy()
    meta: Dict[str, Any] = {}
    if str(eval_start).strip():
        ts0 = pd.Timestamp(eval_start, tz="UTC") if pd.Timestamp(eval_start).tzinfo is None else pd.Timestamp(eval_start).tz_convert("UTC")
        df = df.loc[df["timestamp"] >= ts0].copy()
        meta["eval_start"] = str(ts0.isoformat())
    if str(eval_end).strip():
        ts1 = pd.Timestamp(eval_end, tz="UTC") if pd.Timestamp(eval_end).tzinfo is None else pd.Timestamp(eval_end).tz_convert("UTC")
        df = df.loc[df["timestamp"] <= ts1].copy()
        meta["eval_end"] = str(ts1.isoformat())
    if int(head_rows) > 0:
        df = df.head(int(head_rows)).copy()
        meta["head_rows"] = int(head_rows)
    if int(tail_rows) > 0:
        df = df.tail(int(tail_rows)).copy()
        meta["tail_rows"] = int(tail_rows)
    df = df.reset_index(drop=True)
    meta["rows_eval"] = int(len(df))
    return df, meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="V5 full single-tier backtest")
    ap.add_argument("--input", required=True)
    ap.add_argument("--input-kind", default="auto", choices=["auto", "raw", "features", "dataset", "pred"])
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--scaler-json", default="")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-out", required=True)

    ap.add_argument("--config-json", default="", help="single json containing nested sections for all configs")
    ap.add_argument("--policy-json", default="")
    ap.add_argument("--policy-inline-json", default="")
    ap.add_argument("--dynamic-json", default="")
    ap.add_argument("--dynamic-inline-json", default="")
    ap.add_argument("--progress-json", default="")
    ap.add_argument("--progress-inline-json", default="")
    ap.add_argument("--tp-window-json", default="")
    ap.add_argument("--tp-window-inline-json", default="")
    ap.add_argument("--entry-episode-json", default="")
    ap.add_argument("--entry-episode-inline-json", default="")
    ap.add_argument("--same-side-json", default="")
    ap.add_argument("--same-side-inline-json", default="")
    ap.add_argument("--regime-detect-json", default="")
    ap.add_argument("--regime-detect-inline-json", default="")
    ap.add_argument("--regime-weight-json", default="")
    ap.add_argument("--regime-weight-inline-json", default="")
    ap.add_argument("--regime-threshold-json", default="")
    ap.add_argument("--regime-threshold-inline-json", default="")
    ap.add_argument("--regime-filter-json", default="")
    ap.add_argument("--regime-filter-inline-json", default="")
    ap.add_argument("--regime-lane-json", default="")
    ap.add_argument("--regime-lane-inline-json", default="")
    ap.add_argument("--execution-json", default="")
    ap.add_argument("--execution-inline-json", default="")
    ap.add_argument("--backtest-json", default="")
    ap.add_argument("--backtest-inline-json", default="")
    ap.add_argument("--objective-json", default="")
    ap.add_argument("--objective-inline-json", default="")
    ap.add_argument("--cost-scenarios-json", default="")

    ap.add_argument("--eval-start", default="")
    ap.add_argument("--eval-end", default="")
    ap.add_argument("--tail-rows", type=int, default=0)
    ap.add_argument("--head-rows", type=int, default=0)

    ap.add_argument("--save-pred", type=int, default=0)
    ap.add_argument("--save-fast-cache", type=int, default=0)
    ap.add_argument("--summary-only", type=int, default=0)
    ap.add_argument("--prefer-fast", type=int, default=-1)
    return ap.parse_args()


def _pick_json(specific: str, fallback: str) -> str:
    return str(specific).strip() or str(fallback).strip()


def main() -> None:
    args = parse_args()
    if str(args.input_kind).strip().lower() != "pred" and (not str(args.checkpoint).strip()):
        raise SystemExit("--checkpoint is required unless --input-kind pred")

    cfg_json = str(args.config_json)

    policy = load_config_dataclass(PolicyConfigV5, json_path=_pick_json(str(args.policy_json), cfg_json), inline_json=str(args.policy_inline_json))
    dynamic = load_config_dataclass(DynamicConfigV5, json_path=_pick_json(str(args.dynamic_json), cfg_json), inline_json=str(args.dynamic_inline_json))
    progress = load_config_dataclass(ProgressProtectConfigV5, json_path=_pick_json(str(args.progress_json), cfg_json), inline_json=str(args.progress_inline_json))
    tp_window = load_config_dataclass(TPWindowConfigV5, json_path=_pick_json(str(args.tp_window_json), cfg_json), inline_json=str(args.tp_window_inline_json))
    entry_episode = load_config_dataclass(EntryEpisodeConfigV5, json_path=_pick_json(str(args.entry_episode_json), cfg_json), inline_json=str(args.entry_episode_inline_json))
    same_side = load_config_dataclass(SameSideHoldConfigV5, json_path=_pick_json(str(args.same_side_json), cfg_json), inline_json=str(args.same_side_inline_json))
    regime_detect = load_config_dataclass(RegimeDetectConfigV5, json_path=_pick_json(str(args.regime_detect_json), cfg_json), inline_json=str(args.regime_detect_inline_json))
    regime_weight = load_config_dataclass(RegimeWeightConfigV5, json_path=_pick_json(str(args.regime_weight_json), cfg_json), inline_json=str(args.regime_weight_inline_json))
    regime_threshold = load_config_dataclass(RegimeThresholdConfigV5, json_path=_pick_json(str(args.regime_threshold_json), cfg_json), inline_json=str(args.regime_threshold_inline_json))
    regime_filter = load_config_dataclass(RegimeFilterConfigV5, json_path=_pick_json(str(args.regime_filter_json), cfg_json), inline_json=str(args.regime_filter_inline_json))
    regime_lane = load_config_dataclass(RegimeLaneConfigV5, json_path=_pick_json(str(args.regime_lane_json), cfg_json), inline_json=str(args.regime_lane_inline_json))
    execution = load_config_dataclass(ExecutionConfigV5, json_path=_pick_json(str(args.execution_json), cfg_json), inline_json=str(args.execution_inline_json))
    backtest = load_config_dataclass(BacktestConfigV5, json_path=_pick_json(str(args.backtest_json), cfg_json), inline_json=str(args.backtest_inline_json))
    objective = load_config_dataclass(ObjectiveConfigV5, json_path=_pick_json(str(args.objective_json), cfg_json), inline_json=str(args.objective_inline_json))

    if int(args.prefer_fast) >= 0:
        backtest = BacktestConfigV5(**{**backtest.to_dict(), "prefer_fast": bool(int(args.prefer_fast))})

    artifacts = derive_artifact_paths(args.out_dir, args.log_out)
    pred = _load_pred_or_build(
        input_path=str(args.input),
        input_kind=str(args.input_kind),
        checkpoint=str(args.checkpoint),
        scaler_json=str(args.scaler_json),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    pred_eval, crop_meta = _crop_eval_frame(
        pred,
        eval_start=str(args.eval_start),
        eval_end=str(args.eval_end),
        tail_rows=int(args.tail_rows),
        head_rows=int(args.head_rows),
    )
    if pred_eval.empty:
        raise SystemExit("evaluation frame is empty after crop")

    if bool(int(args.save_pred)):
        write_frame(pred, artifacts["pred_cache"])

    fast_cache = prepare_fast_eval_cache_v5(pred_eval)
    if bool(int(args.save_fast_cache)):
        save_fast_eval_cache_v5(fast_cache, artifacts["pred_fast"])

    if bool(int(args.summary_only)) or bool(backtest.prefer_fast):
        fast_res = evaluate_prediction_frame_fast_v5(
            fast_cache,
            policy=policy,
            dynamic=dynamic,
            progress_protect=progress,
            tp_window=tp_window,
            entry_episode=entry_episode,
            same_side_hold=same_side,
            regime_detect=regime_detect,
            regime_weight=regime_weight,
            regime_threshold=regime_threshold,
            regime_filter=regime_filter,
            regime_lane=regime_lane,
            execution=execution,
            backtest=backtest,
            objective=objective,
        )
        trades = fast_res["trades"]
        decisions = pd.DataFrame()
        equity = fast_res["equity"]
        segments = fast_res["segments"]
        reason_stats = fast_res["reason_stats"]
        overall = fast_res["overall"]
        objective_payload = fast_res["objective"]
        sim_meta = fast_res["sim_meta"]
    else:
        trades, decisions, sim_meta = simulate_from_prediction_frame_v5(
            pred_eval,
            policy=policy,
            dynamic=dynamic,
            progress_protect=progress,
            tp_window=tp_window,
            entry_episode=entry_episode,
            same_side_hold=same_side,
            regime_detect=regime_detect,
            regime_weight=regime_weight,
            regime_threshold=regime_threshold,
            regime_filter=regime_filter,
            regime_lane=regime_lane,
            execution=execution,
            backtest=backtest,
        )
        equity = build_equity_curve_v5(trades, timestamps=pred_eval["timestamp"], annualization_days=float(backtest.annualization_days))
        segments = summarize_segments_v5(trades, timestamps=pred_eval["timestamp"], segments=int(backtest.segments))
        reason_stats = summarize_reason_stats_v5(trades)
        overall = summarize_trades_v5(trades, equity=equity, segments=segments, sim_meta=sim_meta)
        objective_payload = assemble_objective_v5(overall=overall, segments=segments, sim_meta=sim_meta, objective=objective)

    cost_scenarios = load_cost_scenarios_json(str(args.cost_scenarios_json))
    cost_eval = apply_cost_scenarios_v5(trades, cost_scenarios)

    if not bool(int(args.summary_only)):
        write_frame(decisions, artifacts["decision"])
        write_frame(trades, artifacts["tradelog"])

    summary = {
        "contract_version": BACKTEST_CONTRACT_VERSION,
        "input": str(args.input),
        "input_kind": str(args.input_kind),
        "checkpoint": str(args.checkpoint),
        "scaler_json": str(args.scaler_json),
        "pred_rows_full": int(len(pred)),
        "pred_rows_eval": int(len(pred_eval)),
        "config_snapshot": {
            "policy": policy.to_dict(),
            "dynamic": dynamic.to_dict(),
            "progress_protect": progress.to_dict(),
            "tp_window": tp_window.to_dict(),
            "entry_episode": entry_episode.to_dict(),
            "same_side_hold": same_side.to_dict(),
            "regime_detect": regime_detect.to_dict(),
            "regime_weight": regime_weight.to_dict(),
            "regime_threshold": regime_threshold.to_dict(),
            "regime_filter": regime_filter.to_dict(),
            "regime_lane": regime_lane.to_dict(),
            "execution": execution.to_dict(),
            "backtest": backtest.to_dict(),
            "objective": objective.to_dict(),
        },
        "eval_window": {
            "start_utc": str(pred_eval["timestamp"].iloc[0].isoformat()) if len(pred_eval) else "",
            "end_utc": str(pred_eval["timestamp"].iloc[-1].isoformat()) if len(pred_eval) else "",
            "crop_meta": crop_meta,
        },
        "overall": overall,
        "objective": objective_payload,
        "segments": segments.to_dict(orient="records"),
        "exit_reasons": reason_stats.to_dict(orient="records"),
        "cost_scenarios": cost_eval,
        "maker_taker_stats": {
            "maker_entry_count": int(sim_meta.get("maker_entry_count", 0)),
            "maker_exit_count": int(sim_meta.get("maker_exit_count", 0)),
            "ioc_exit_count": int(sim_meta.get("ioc_exit_count", 0)),
            "market_exit_count": int(sim_meta.get("market_exit_count", 0)),
            "maker_entry_ratio": float(overall.get("maker_entry_ratio", float("nan"))),
            "maker_exit_ratio": float(overall.get("maker_exit_ratio", float("nan"))),
        },
        "fill_stats": {
            "entry_signals": int(sim_meta.get("n_entry_signals", 0)),
            "entry_filled": int(sim_meta.get("n_entry_filled", 0)),
            "entry_unfilled": int(sim_meta.get("n_entry_unfilled", 0)),
            "unfilled_entry_rate": float(overall.get("unfilled_entry_rate", float("nan"))),
            "rearm_entry_signals": int(sim_meta.get("n_rearm_entry_signals", 0)),
            "rearm_entry_filled": int(sim_meta.get("n_rearm_entry_filled", 0)),
        },
        "timing_stats": {
            "hold_p50": float(overall.get("hold_p50", float("nan"))),
            "hold_p90": float(overall.get("hold_p90", float("nan"))),
            "max_hold_rate": float(overall.get("max_hold_rate", 0.0)),
        },
        "regime_stats": {
            "bucket_counts": sim_meta.get("regime_bucket_counts", {}),
            "shock_count": int(sim_meta.get("shock_count", 0)),
            "regime_extreme_frac": float(overall.get("regime_extreme_frac", 0.0)),
        },
        "sim_meta": sim_meta,
        "artifact_paths": {
            "decision": artifacts["decision"],
            "tradelog": artifacts["tradelog"],
            "summary": artifacts["summary"],
            "pred_cache": artifacts["pred_cache"] if bool(int(args.save_pred)) else "",
            "pred_fast": artifacts["pred_fast"] if bool(int(args.save_fast_cache)) else "",
        },
    }

    Path(artifacts["summary"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
