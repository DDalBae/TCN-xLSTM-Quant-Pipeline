# DeepScalper-V5: End-to-End Deep Learning Scalping Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

DeepScalper-V5 is an advanced, end-to-end quantitative trading pipeline designed for high-frequency scalping on Binance USD-M Futures (1-minute intervals). It leverages a robust deep learning backbone (**TCN + xLSTM**) and a highly customized cost-aware utility target system to navigate the intense noise of micro-timeframes.

## Key Features

* **Data-Driven Feature Engineering:** Converts minimal raw schema (OHLCV + taker_buy_base) into 26 reactive core features (Micro-price geometry, Return momentum, Volatility, Flow state).
* **Cost-Aware Target Engineering:** Targets are not simple direction predictions. The label builder explicitly calculates multi-horizon expected utility, penalizing adverse excursions (whipsaws) and incorporating maker/taker fee floors.
* **TCN-xLSTM Architecture:** A hybrid deep learning model combining Temporal Convolutional Networks (for local noise filtering) and xLSTM blocks (for regime and sequence memory) with an 8-head multi-task output (Entry direction, Hybrid signal, Path exit geometry, Utility).
* **Realistic Backtesting Engine:** Simulates realistic execution mechanics including Maker-first with IOC fallback, slippage models, dynamic TP/SL trailing, and regime-based trade filtering.

## Project Structure

The pipeline is strictly divided into functional contracts to prevent data leakage and future-reference bias.

### 1. Data & Features
* `feature_contract_v5.py`: Single source of truth for the reactive 26 feature schema.
* `feature_ops_v5.py`: Builds features iteratively without future references.

### 2. Targets & Labels
* `target_contract_v5.py`: Defines multi-horizon (1/3/5/8/10) entry targets and `path10` exit geometry.
* `label_builder_v5_1.py`: Offline-only module to construct supervised learning targets using future windows.

### 3. Deep Learning Model
* `model_v5_1.py`: Defines the PyTorch architecture (Stem -> TCN Stack -> xLSTM -> Readout -> Multi-Heads).
* `trainer_v5.py`: Consumer-aware model trainer with early stopping and automatic mixed precision (AMP) support.

### 4. Inference & Backtest
* `inference_v5.py`: Batched inference script that decodes model raw outputs into actionable composite signals.
* `backtest_contract_v5.py` / `backtest_core_v5.py`: Core logic for simulated execution, intrabar analysis, and payoff calculation.
* `backtest_v5.py`: CLI tool to run the full single-tier backtest.

## ⚙️ Pipeline Workflow

1.  **Feature Generation:** `raw_data` -> `feature_ops_v5.py` -> `features.parquet`
2.  **Label Generation:** `features.parquet` -> `label_builder_v5_1.py` -> `dataset.parquet`
3.  **Model Training:** `dataset.parquet` -> `trainer_v5.py` -> `model.pt` & `scaler.json`
4.  **Inference:** `features.parquet` + `model.pt` -> `inference_v5.py` -> `predictions.parquet`
5.  **Backtest:** `predictions.parquet` -> `backtest_v5.py` -> `Performance Summary & Trade Logs`

## Disclaimer

This repository is for educational and research purposes only. The algorithms and models provided do not constitute financial advice. High-frequency trading and cryptocurrency markets are extremely volatile and carry a high level of risk. **Do not use this code with real money without extensive forward testing.** The author is not responsible for any financial losses incurred.
