"""
lstm_stock_predictor.py
=======================
LSTM-based stock return forecasting using PyTorch.

Predicts next-day log returns from a rolling window of technical features,
then reconstructs prices for evaluation. Each predicted price anchors to the
*real* prior price to prevent compounding errors — note this flatters price
RMSE vs. a true multi-step forecast.

Usage
-----
    python lstm_stock_predictor.py

Dependencies
------------
    numpy, pandas, yfinance, matplotlib, scikit-learn, torch
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Central configuration for data, training, and model hyperparameters."""

    # Data
    tickers: list[str] = field(default_factory=lambda: ["GOOGL", "NKE"])
    start_dates: dict[str, str] = field(
        default_factory=lambda: {
            "GOOGL": "2004-08-19",
            "NKE":   "2010-01-04",
        }
    )
    end_date: str  = "2019-12-19"
    train_ratio: float = 0.80
    window: int = 50

    # Model
    hidden_size: int = 96
    num_layers: int = 4
    dropout: float = 0.2

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 10          # early-stopping patience (epochs without improvement)
    clip_grad_norm: float = 1.0 # gradient clipping

    # I/O
    output_dir: Path = Path(".")
    figure_name: str = "lstm_stock_predictions.png"
    seed: int = 42


CFG = Config()


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility across numpy, Python, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return the best available compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s%s", device,
             f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    return device


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM that maps a (batch, seq_len, n_features) input to a
    scalar next-step log-return prediction.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    hidden_size : int
        Number of LSTM hidden units.
    num_layers : int
        Depth of the stacked LSTM.
    dropout : float
        Dropout probability applied between LSTM layers and before the
        final linear projection.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 96,
        num_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)

        Returns
        -------
        Tensor, shape (batch,)
        """
        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden)
        last        = self.dropout(lstm_out[:, -1, :])  # last timestep
        return self.head(last).squeeze(-1)     # (batch,)


# ── Data pipeline ─────────────────────────────────────────────────────────────
@dataclass
class StockDataset:
    """Processed tensors and metadata for a single ticker."""
    X_train:    torch.Tensor
    y_train:    torch.Tensor
    X_test:     torch.Tensor
    y_test:     torch.Tensor
    scaler:     MinMaxScaler
    test_dates: pd.DatetimeIndex
    test_prices: np.ndarray    # raw open prices aligned with test set


FEATURE_COLS = ["LogReturn", "MA10", "MA50", "Volatility"]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features from raw OHLCV data (in-place)."""
    df["LogReturn"]  = np.log(df["Open"] / df["Open"].shift(1))
    df["MA10"]       = df["Open"].rolling(10).mean() / df["Open"] - 1
    df["MA50"]       = df["Open"].rolling(50).mean() / df["Open"] - 1
    df["Volatility"] = df["LogReturn"].rolling(10).std()
    return df.dropna()


def _make_sequences(
    scaled: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build overlapping (X, y) sequences from a 2-D scaled array."""
    X = np.stack([scaled[i - window: i] for i in range(window, len(scaled))], axis=0)
    y = scaled[window:, 0]
    return X.astype(np.float32), y.astype(np.float32)


def load_and_prepare(ticker: str, cfg: Config, device: torch.device) -> StockDataset:
    """
    Download, clean, feature-engineer, scale, and split data for *ticker*.

    Returns a :class:`StockDataset` with all tensors already moved to *device*.
    """
    log.info("Downloading %s  [%s → %s]", ticker, cfg.start_dates[ticker], cfg.end_date)
    df = yf.download(ticker, start=cfg.start_dates[ticker], end=cfg.end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = _engineer_features(df)
    raw_prices   = df["Open"].values.reshape(-1, 1)
    feature_data = df[FEATURE_COLS].values
    dates        = df.index

    # Fit scaler on training portion only (no look-ahead)
    n_raw_train = int(len(feature_data) * cfg.train_ratio)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(feature_data[:n_raw_train])
    scaled = scaler.transform(feature_data)

    X, y = _make_sequences(scaled, cfg.window)
    split = int(len(X) * cfg.train_ratio)

    def _to(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(device)

    return StockDataset(
        X_train     = _to(X[:split]),
        y_train     = _to(y[:split]),
        X_test      = _to(X[split:]),
        y_test      = _to(y[split:]),
        scaler      = scaler,
        test_dates  = dates[cfg.window + split:],
        test_prices = raw_prices[cfg.window + split:],
    )


def inverse_log_returns(scaled_col: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Invert the MinMaxScaler on the first feature column (LogReturn)."""
    dummy = np.zeros((len(scaled_col), scaler.n_features_in_), dtype=np.float32)
    dummy[:, 0] = scaled_col
    return scaler.inverse_transform(dummy)[:, 0]


def reconstruct_prices(
    pred_returns: np.ndarray,
    real_prices:  np.ndarray,
) -> np.ndarray:
    """
    Reconstruct predicted prices by anchoring each step to the *real* prior
    price:  pred_price[t] = real_price[t-1] * exp(pred_return[t]).

    .. note::
        Anchoring prevents error compounding but is **optimistic**: each step
        gets a ground-truth reset, so price RMSE underestimates true multi-
        step forecast error.
    """
    n = len(pred_returns)
    pred_prices    = np.empty(n, dtype=np.float64)
    pred_prices[0] = real_prices[0]
    for t in range(1, n):
        pred_prices[t] = real_prices[t - 1] * np.exp(pred_returns[t])
    return pred_prices


# ── Metrics ───────────────────────────────────────────────────────────────────
@dataclass
class ForecastMetrics:
    return_rmse:       float
    naive_return_rmse: float
    price_rmse:        float
    naive_price_rmse:  float
    dir_accuracy:      float
    naive_dir:         float

    def beats_naive(self, on: str = "direction") -> bool:
        if on == "direction":
            return self.dir_accuracy > self.naive_dir
        if on == "price":
            return self.price_rmse < self.naive_price_rmse
        return self.return_rmse < self.naive_return_rmse

    def log_summary(self, ticker: str) -> None:
        def tag(better: bool) -> str:
            return "✓ BEATS" if better else "✗ LOSES TO"

        log.info("─" * 54)
        log.info("%-6s  Return RMSE  model=%.6f  naive=%.6f  %s naive",
                 ticker, self.return_rmse, self.naive_return_rmse,
                 tag(self.return_rmse < self.naive_return_rmse))
        log.info("%-6s  Price  RMSE  model=%.4f   naive=%.4f   %s naive",
                 ticker, self.price_rmse, self.naive_price_rmse,
                 tag(self.price_rmse < self.naive_price_rmse))
        log.info("%-6s  Dir accuracy  model=%.1f%%  naive=%.1f%%  %s naive",
                 ticker, self.dir_accuracy, self.naive_dir,
                 tag(self.dir_accuracy > self.naive_dir))


def compute_metrics(
    real_returns:  np.ndarray,
    pred_returns:  np.ndarray,
    real_prices:   np.ndarray,
    pred_prices:   np.ndarray,
) -> ForecastMetrics:
    def _dir_acc(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(b))) * 100)

    return ForecastMetrics(
        return_rmse       = float(np.sqrt(np.mean((pred_returns - real_returns) ** 2))),
        naive_return_rmse = float(np.sqrt(np.mean(np.diff(real_returns) ** 2))),
        price_rmse        = float(np.sqrt(np.mean((pred_prices - real_prices) ** 2))),
        naive_price_rmse  = float(np.sqrt(np.mean(np.diff(real_prices) ** 2))),
        dir_accuracy      = _dir_acc(real_prices, pred_prices),
        naive_dir         = _dir_acc(real_prices[:-1], real_prices[1:]),
    )


# ── Training ──────────────────────────────────────────────────────────────────
class EarlyStopper:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int = 10) -> None:
        self.patience   = patience
        self.best_loss  = float("inf")
        self._counter   = 0

    def step(self, loss: float) -> bool:
        """Return True if training should stop."""
        if loss < self.best_loss:
            self.best_loss = loss
            self._counter  = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


def train_model(
    model:   LSTMForecaster,
    dataset: StockDataset,
    cfg:     Config,
) -> LSTMForecaster:
    """
    Train *model* on the training split of *dataset*.

    Uses Adam optimiser with gradient clipping and optional early stopping
    (patience controlled by ``cfg.patience``).
    """
    loader    = DataLoader(
        TensorDataset(dataset.X_train, dataset.y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    stopper   = EarlyStopper(patience=cfg.patience)

    best_weights: Optional[dict] = None

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(dataset.X_train)

        if epoch % 10 == 0 or epoch == 1:
            log.info("  Epoch %3d/%d  loss=%.6f", epoch, cfg.epochs, epoch_loss)

        # Early stopping — track best weights
        if epoch_loss < getattr(stopper, "best_loss", float("inf")):
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        if stopper.step(epoch_loss):
            log.info("  Early stop at epoch %d (patience=%d)", epoch, cfg.patience)
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def evaluate(
    model:   LSTMForecaster,
    dataset: StockDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over the test split and return arrays in original scale.

    Returns
    -------
    real_returns, pred_returns, real_prices, pred_prices — all np.ndarray
    """
    model.eval()
    with torch.no_grad():
        pred_scaled = model(dataset.X_test).cpu().numpy()
        real_scaled = dataset.y_test.cpu().numpy()

    pred_returns  = inverse_log_returns(pred_scaled, dataset.scaler)
    real_returns  = inverse_log_returns(real_scaled, dataset.scaler)
    real_prices   = dataset.test_prices[: len(pred_returns)].flatten()
    pred_prices   = reconstruct_prices(pred_returns, real_prices)

    return real_returns, pred_returns, real_prices, pred_prices


# ── Plotting ──────────────────────────────────────────────────────────────────
_STYLE = {
    "real":    dict(color="#E03030", linewidth=0.9, label="Real price"),
    "pred":    dict(color="#2563EB", linewidth=0.9, label="Predicted price"),
    "beat":    "#2563EB",
    "lose":    "#DC2626",
    "neutral": "#6B7280",
}


def _plot_prices(
    ax:          plt.Axes,
    ticker:      str,
    dates:       pd.DatetimeIndex,
    real_prices: np.ndarray,
    pred_prices: np.ndarray,
    metrics:     ForecastMetrics,
) -> None:
    ax.plot(dates[: len(real_prices)], real_prices, **_STYLE["real"])
    ax.plot(dates[: len(pred_prices)], pred_prices, **_STYLE["pred"])
    ax.set_title(
        f"{ticker}  —  Reconstructed Open Price\n"
        f"Price RMSE {metrics.price_rmse:.2f} (naive {metrics.naive_price_rmse:.2f})  |  "
        f"Dir {metrics.dir_accuracy:.1f}% vs {metrics.naive_dir:.1f}%",
        fontsize=8,
    )
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Opening price (USD)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30, labelsize=7)


def _plot_directional(
    ax:      plt.Axes,
    ticker:  str,
    metrics: ForecastMetrics,
) -> None:
    beat   = metrics.beats_naive("direction")
    colors = [_STYLE["neutral"], _STYLE["beat"] if beat else _STYLE["lose"]]
    bars   = ax.bar(
        ["Naïve baseline", "LSTM"],
        [metrics.naive_dir, metrics.dir_accuracy],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        width=0.45,
    )
    for bar, val in zip(bars, [metrics.naive_dir, metrics.dir_accuracy]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, label="Random (50%)")
    ax.set_ylabel("Directional accuracy (%)", fontsize=8)
    ax.set_title(f"{ticker}  —  Directional Accuracy", fontsize=9)
    ax.set_ylim(40, 70)
    ax.legend(fontsize=7)


def build_figure(
    results: list[tuple[str, StockDataset, ForecastMetrics,
                        np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    cfg:     Config,
) -> Path:
    """Render a 2×N grid (price + bar chart per ticker) and save to disk."""
    n  = len(results)
    fig, axes = plt.subplots(2, n, figsize=(8 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        "LSTM Stock Forecaster (PyTorch)  —  Log-Return + Technical Features",
        fontsize=13,
        fontweight="bold",
    )

    for col, (ticker, dataset, metrics, rr, pr, rp, pp) in enumerate(results):
        _plot_prices(axes[0][col], ticker, dataset.test_dates, rp, pp, metrics)
        _plot_directional(axes[1][col], ticker, metrics)

    plt.tight_layout()
    out = cfg.output_dir / cfg.figure_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved → %s", out.resolve())
    return out


# ── Entry point ───────────────────────────────────────────────────────────────
def main(cfg: Config = CFG) -> None:
    set_seed(cfg.seed)
    device = get_device()

    results = []

    for ticker in cfg.tickers:
        log.info("=" * 54)
        log.info("Processing %s", ticker)

        # Data
        dataset = load_and_prepare(ticker, cfg, device)

        # Model
        n_features = dataset.X_train.shape[2]
        model = LSTMForecaster(
            n_features  = n_features,
            hidden_size = cfg.hidden_size,
            num_layers  = cfg.num_layers,
            dropout     = cfg.dropout,
        ).to(device)
        log.info("Model parameters: %d",
                 sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Train
        model = train_model(model, dataset, cfg)

        # Evaluate
        real_returns, pred_returns, real_prices, pred_prices = evaluate(model, dataset)
        metrics = compute_metrics(real_returns, pred_returns, real_prices, pred_prices)
        metrics.log_summary(ticker)

        results.append((ticker, dataset, metrics,
                        real_returns, pred_returns, real_prices, pred_prices))

    build_figure(results, cfg)


if __name__ == "__main__":
    main()
