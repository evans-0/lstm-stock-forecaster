# lstm-stock-forecaster

> Multi-layer LSTM for next-day equity log-return forecasting, built with PyTorch.

Trains on a rolling window of technical features — log return, moving-average ratios, and realised volatility — then reconstructs open prices for evaluation. Benchmarked against a naïve baseline on directional accuracy and RMSE.

---

## Contents

- [Features](#features)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Output](#output)
- [Limitations](#limitations)
- [License](#license)

---

## Features

- **GPU-accelerated training** — automatically falls back to CPU when CUDA is unavailable
- **Early stopping** — restores best-epoch weights; configurable patience
- **Gradient clipping** — prevents LSTM exploding gradients
- **Reproducibility** — global seed applied to Python, NumPy, and PyTorch (including CUDA)
- **Leak-free scaling** — MinMaxScaler fitted on the training split only
- **Structured metrics** — return RMSE, price RMSE, and directional accuracy, each compared to a naïve baseline
- **Dataclass config** — all hyperparameters in one place; no scattered magic numbers

---

## Methodology

### Features

| Feature | Description |
|---|---|
| `LogReturn` | `ln(Open_t / Open_{t-1})` |
| `MA10` | 10-day moving average ratio: `MA10 / Open - 1` |
| `MA50` | 50-day moving average ratio: `MA50 / Open - 1` |
| `Volatility` | 10-day rolling standard deviation of log returns |

### Model

```
Input (batch, window=50, features=4)
    └─ LSTM × 4 layers (hidden=96, dropout=0.2)
         └─ Dropout(0.2)
              └─ Linear(96 → 1)
Output: predicted log return at t+1
```

### Train / Test Split

Data is split 80 / 20 chronologically. The scaler is fitted on the training portion only to prevent look-ahead bias.

### Price Reconstruction

Predicted prices are anchored to the *real* prior price at each step:

```
pred_price[t] = real_price[t-1] × exp(pred_return[t])
```

This prevents compounding errors across the test window. See [Limitations](#limitations) for why this flatters price RMSE.

---

## Project Structure

```
lstm-stock-forecaster/
├── lstm_stock_predictor.py   # full pipeline (data → train → evaluate → plot)
├── requirements.txt
└── README.md
```

---

## Quickstart

### Prerequisites

Python 3.10+ and a CUDA-capable GPU (optional but recommended).

### Install

```bash
git clone https://github.com/<your-username>/lstm-stock-forecaster.git
cd lstm-stock-forecaster
pip install -r requirements.txt
```

### Run

```bash
python lstm_stock_predictor.py
```

Training logs stream to stdout. On completion, `lstm_stock_predictions.png` is written to the working directory.

---

## Configuration

All settings live in the `Config` dataclass at the top of `lstm_stock_predictor.py`. Override inline for quick experiments:

```python
from lstm_stock_predictor import Config, main

main(Config(
    tickers        = ["AAPL", "MSFT", "TSLA"],
    epochs         = 100,
    hidden_size    = 128,
    patience       = 15,
    figure_name    = "my_run.png",
))
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `tickers` | `["GOOGL", "NKE"]` | List of Yahoo Finance ticker symbols |
| `window` | `50` | Look-back window (trading days) |
| `hidden_size` | `96` | LSTM hidden units |
| `num_layers` | `4` | Stacked LSTM depth |
| `dropout` | `0.2` | Dropout between layers and before projection |
| `epochs` | `50` | Maximum training epochs |
| `patience` | `10` | Early-stopping patience |
| `learning_rate` | `0.001` | Adam learning rate |
| `clip_grad_norm` | `1.0` | Gradient clipping threshold |
| `train_ratio` | `0.80` | Fraction of data used for training |
| `seed` | `42` | Global random seed |

---

## Output

The script produces a `2 × N` figure (one column per ticker):

| Row | Content |
|---|---|
| Top | Reconstructed open price — real vs predicted |
| Bottom | Directional accuracy bar chart vs naïve baseline |

Console output includes epoch loss, early-stop events, and a final metrics table:

```
17:42:11  INFO      ══════════════════════════════════════════════════════
17:42:11  INFO      GOOGL  Return RMSE  model=0.007821  naive=0.009134  ✓ BEATS naive
17:42:11  INFO      GOOGL  Price  RMSE  model=18.3421   naive=22.1047   ✓ BEATS naive
17:42:11  INFO      GOOGL  Dir accuracy  model=54.2%  naive=50.1%  ✓ BEATS naive
```

---

## Limitations

**Anchored price reconstruction is optimistic.**
Each predicted price resets to the real prior price, so reconstruction errors do not compound. True multi-step price RMSE would be meaningfully higher. Price RMSE figures should not be compared across models unless both use identical reconstruction schemes.

**Single chronological split.**
An 80/20 holdout is sensitive to the boundary date. Walk-forward (expanding or sliding window) cross-validation would give more reliable generalisation estimates.

**No transaction costs or slippage.**
Directional accuracy above 50% does not imply a profitable strategy once trading costs, bid-ask spread, and execution latency are accounted for.

**Feature set is minimal.**
Volume, momentum oscillators (RSI, MACD), and macro variables are absent. These may carry additional signal.

---

## Requirements

```
numpy
pandas
yfinance
matplotlib
scikit-learn
torch
```

Generate `requirements.txt` with pinned versions:

```bash
pip freeze | grep -E "numpy|pandas|yfinance|matplotlib|scikit-learn|torch" > requirements.txt
```

---

## License

MIT
