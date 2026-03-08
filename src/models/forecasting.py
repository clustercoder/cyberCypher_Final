from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# FallbackForecaster — pure numpy, no torch dependency
# ---------------------------------------------------------------------------

class FallbackForecaster:
    """EWMA-based forecaster — used when LSTM training fails or torch is unavailable."""

    def __init__(self, span: int = 10) -> None:
        self._alpha = 2.0 / (span + 1)

    def predict(self, recent_data: np.ndarray, horizon: int = 10) -> np.ndarray:
        """Extend EWMA trend forward by ``horizon`` steps."""
        if len(recent_data) == 0:
            return np.zeros(horizon)

        alpha = self._alpha
        ewma = float(recent_data[0])
        for v in recent_data[1:]:
            ewma = alpha * float(v) + (1 - alpha) * ewma

        tail = recent_data[-min(10, len(recent_data)):]
        if len(tail) >= 2:
            xs = np.arange(len(tail), dtype=float)
            slope = float(np.polyfit(xs, tail, 1)[0])
        else:
            slope = 0.0

        slope = float(np.clip(slope, -2.0, 2.0))

        return np.array(
            [ewma + slope * (i + 1) for i in range(horizon)], dtype=float
        )

    def predict_congestion(
        self, recent_data: np.ndarray, threshold: float = 85.0, horizon: int = 10
    ) -> dict[str, Any]:
        predictions = self.predict(recent_data, horizon)
        peak = float(predictions.max())
        congested = [i + 1 for i, v in enumerate(predictions) if v >= threshold]
        return {
            "will_congest": len(congested) > 0,
            "minutes_until": congested[0] if congested else None,
            "predicted_peak": round(peak, 4),
            "predictions": [round(float(v), 4) for v in predictions],
        }


# ---------------------------------------------------------------------------
# PyTorch model definition (only defined when torch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):  # type: ignore[misc]
        def __init__(self, hidden_size: int, num_layers: int, forecast_horizon: int, dropout: float) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, forecast_horizon)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.fc(last)


# ---------------------------------------------------------------------------
# LSTMForecaster
# ---------------------------------------------------------------------------

class LSTMForecaster:
    """Single-metric LSTM forecaster.  Predicts the next ``forecast_horizon``
    minutes from a ``seq_length``-minute history window.

    Falls back to FallbackForecaster behaviour if torch is not installed.
    """

    def __init__(
        self,
        seq_length: int = 30,
        forecast_horizon: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
        dropout: float = 0.2,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTMForecaster. "
                "Use FallbackForecaster for a torch-free alternative."
            )
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.dropout = dropout

        self._model: Any = self._build_model()
        self._scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
        self._trained: bool = False
        self._device = torch.device("cpu")

    def _build_model(self) -> Any:
        return _LSTMNet(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            forecast_horizon=self.forecast_horizon,
            dropout=self.dropout,
        )

    def _prepare_sequences(
        self, data: np.ndarray
    ) -> tuple[Any, Any]:
        """Sliding-window split → (X [N, seq, 1], y [N, horizon])."""
        xs, ys = [], []
        total = self.seq_length + self.forecast_horizon
        for i in range(len(data) - total + 1):
            xs.append(data[i : i + self.seq_length])
            ys.append(data[i + self.seq_length : i + total])
        X = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(ys), dtype=torch.float32)
        return X, y

    def train(
        self,
        historical_data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> list[float]:
        """Fit the LSTM.  Returns per-epoch loss history."""
        data = historical_data.reshape(-1, 1)
        scaled = self._scaler.fit_transform(data).flatten()

        X, y = self._prepare_sequences(scaled)
        n = len(X)
        if n < 1:
            raise ValueError(
                f"Not enough data: need at least {self.seq_length + self.forecast_horizon} points, got {len(historical_data)}"
            )

        self._model.to(self._device)
        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        loss_history: list[float] = []
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n)
            X, y = X[perm], y[perm]

            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                xb = X[start : start + batch_size].to(self._device)
                yb = y[start : start + batch_size].to(self._device)

                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            avg_loss = epoch_loss / n
            loss_history.append(avg_loss)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}/{epochs}  loss={avg_loss:.6f}")

        self._model.eval()
        self._trained = True
        return loss_history

    def predict(self, recent_data: np.ndarray) -> np.ndarray:
        """Return ``forecast_horizon`` predictions in original scale."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet.")
        if len(recent_data) < self.seq_length:
            raise ValueError(
                f"Need {self.seq_length} points, got {len(recent_data)}"
            )

        window = recent_data[-self.seq_length :].reshape(-1, 1)
        scaled = self._scaler.transform(window).flatten()

        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        x = x.to(self._device)

        with torch.no_grad():
            pred_scaled = self._model(x).squeeze(0).cpu().numpy()

        pred_scaled = pred_scaled.reshape(-1, 1)
        return self._scaler.inverse_transform(pred_scaled).flatten()

    def predict_congestion(
        self, recent_data: np.ndarray, threshold: float = 85.0
    ) -> dict[str, Any]:
        predictions = self.predict(recent_data)
        peak = float(predictions.max())
        congested_minutes = [
            i + 1 for i, v in enumerate(predictions) if v >= threshold
        ]
        will_congest = len(congested_minutes) > 0
        minutes_until = congested_minutes[0] if will_congest else None
        return {
            "will_congest": will_congest,
            "minutes_until": minutes_until,
            "predicted_peak": round(peak, 4),
            "predictions": [round(float(v), 4) for v in predictions],
        }

    def predict_with_uncertainty(
        self,
        recent_data: np.ndarray,
        n_samples: int = 50,
    ) -> dict[str, Any]:
        """Monte Carlo dropout uncertainty estimation."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet.")
        if len(recent_data) < self.seq_length:
            raise ValueError(f"Need {self.seq_length} points, got {len(recent_data)}")

        window = recent_data[-self.seq_length:].reshape(-1, 1)
        scaled = self._scaler.transform(window).flatten()
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self._device)

        self._model.train()
        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred_scaled = self._model(x).squeeze(0).cpu().numpy().reshape(-1, 1)
                pred = self._scaler.inverse_transform(pred_scaled).flatten()
                samples.append(pred)
        self._model.eval()

        arr = np.array(samples)
        mean_pred = arr.mean(axis=0)
        std_pred = arr.std(axis=0)
        uncertainty_score = float(std_pred.mean())

        return {
            "mean_predictions": [round(float(v), 4) for v in mean_pred],
            "std_predictions": [round(float(v), 4) for v in std_pred],
            "uncertainty_score": round(uncertainty_score, 6),
            "high_uncertainty": uncertainty_score > 0.3,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "scaler_scale": self._scaler.scale_,
                "scaler_min": self._scaler.min_,
                "scaler_data_min": self._scaler.data_min_,
                "scaler_data_max": self._scaler.data_max_,
                "config": {
                    "seq_length": self.seq_length,
                    "forecast_horizon": self.forecast_horizon,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "lr": self.lr,
                },
            },
            path,
        )
        print(f"  Model saved to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device)
        cfg = checkpoint["config"]
        self.seq_length = cfg["seq_length"]
        self.forecast_horizon = cfg["forecast_horizon"]
        self.hidden_size = cfg["hidden_size"]
        self.num_layers = cfg["num_layers"]
        self.dropout = cfg["dropout"]
        self.lr = cfg["lr"]

        self._model = self._build_model()
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()

        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.scale_ = checkpoint["scaler_scale"]
        self._scaler.min_ = checkpoint["scaler_min"]
        self._scaler.data_min_ = checkpoint["scaler_data_min"]
        self._scaler.data_max_ = checkpoint["scaler_data_max"]
        self._scaler.n_features_in_ = 1
        self._trained = True
        print(f"  Model loaded from {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(42)

    print("── Generating synthetic 24h utilisation series ──────────────────")
    minutes = np.arange(24 * 60)
    hours = minutes / 60.0
    diurnal = 0.3 + 0.7 * (0.5 * (1 + np.sin((hours - 16) * math.pi / 12)))
    base_util = 30.0 + 40.0 * diurnal + rng.normal(0, 3, len(minutes))
    base_util = np.clip(base_util, 5.0, 95.0)
    print(f"  Points : {len(base_util):,}  "
          f"min={base_util.min():.1f}  max={base_util.max():.1f}  "
          f"mean={base_util.mean():.1f}")

    train_data = base_util[:1200]
    test_data  = base_util[1200:]

    print("\n── Training LSTM on first 20 hours ─────────────────────────────")
    trained_lstm = False
    if _TORCH_AVAILABLE:
        forecaster = LSTMForecaster(seq_length=30, forecast_horizon=10, lr=0.001)
        try:
            forecaster.train(train_data, epochs=50, batch_size=32)
            print("  Training complete.\n")
            trained_lstm = True
        except Exception as exc:
            print(f"  LSTM training failed: {exc} — falling back to EWMA")
    else:
        print("  torch not available — using FallbackForecaster")

    print("── Predict next 10 min from last 30 min of training data ────────")
    recent = train_data[-30:]
    actual = test_data[:10]

    fb = FallbackForecaster()
    if trained_lstm:
        predictions = forecaster.predict(recent)
    else:
        predictions = fb.predict(recent, horizon=10)

    print(f"  {'Min':>4}  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
    for i, (act, pred) in enumerate(zip(actual, predictions), start=1):
        err = abs(act - pred)
        print(f"  {i:>4}  {act:>8.2f}  {pred:>10.2f}  {err:>8.2f}")
    mae = float(np.mean(np.abs(actual - predictions)))
    print(f"\n  MAE over 10 minutes: {mae:.4f}")

    sys.exit(0)
