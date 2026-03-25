# Models Reference

## Default Model: DLinear + Gaussian Probabilistic Head

### Why DLinear?

DLinear (from "Are Transformers Effective for Time Series Forecasting?", Zeng et al. 2023) is a
deliberately simple model that decomposes time series into trend and seasonal components, then
applies separate linear layers. It's our default because:

1. **Fast**: Trains in seconds on CPU, minutes on GPU — ideal for rapid iteration
2. **Surprisingly competitive**: Beats many Transformer models on forecasting benchmarks
3. **Interpretable**: Clear trend/seasonal decomposition
4. **Easy to extend**: Swap the backbone, change the head, add features

### Architecture

```
Input: [batch, lookback, features]
  │
  ├──▶ Moving Average Kernel ──▶ Trend Component ──▶ Linear_trend ──┐
  │                                                                   │
  └──▶ Input - Trend ──▶ Seasonal Component ──▶ Linear_seasonal ──┤
                                                                      │
                                                              Concatenate
                                                                      │
                                                              ┌───────┴───────┐
                                                              │               │
                                                         Linear_μ       Linear_σ
                                                              │               │
                                                           μ(t)         softplus(·)
                                                              │               │
                                                              └───────┬───────┘
                                                                      │
                                                          N(μ(t), σ(t)²)
                                                                      │
                                                          Sample N=1000 paths
                                                                      │
                                                          Cumulative sum → prices
```

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MovingAvgBlock(nn.Module):
    """Moving average block for trend extraction."""
    
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # Causal padding: pad only on the left
        self.pad = kernel_size - 1
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        # Pad on left only (causal)
        x_padded = F.pad(x.permute(0, 2, 1), (self.pad, 0), mode="replicate")
        x_avg = x_padded.unfold(-1, self.kernel_size, 1).mean(-1)
        return x_avg.permute(0, 2, 1)


class DLinearGaussian(nn.Module):
    """
    DLinear with Gaussian probabilistic head for Synth Subnet mining.
    
    Predicts μ and σ at each future timestep, then samples N paths.
    """
    
    def __init__(
        self,
        lookback: int = 288,        # 288 × 5min = 24h of history
        horizon: int = 288,          # 288 × 5min = 24h forecast
        n_features: int = 1,         # Number of input features (returns_bps)
        kernel_size: int = 25,       # Moving average kernel
        min_sigma: float = 1.0,      # Minimum σ in basis points
        max_sigma: float = 500.0,    # Maximum σ in basis points
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_features = n_features
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
        # Decomposition
        self.moving_avg = MovingAvgBlock(kernel_size)
        
        # Trend path
        self.linear_trend = nn.Linear(lookback, horizon)
        
        # Seasonal path
        self.linear_seasonal = nn.Linear(lookback, horizon)
        
        # Probabilistic head
        self.mu_head = nn.Linear(horizon, horizon)
        self.sigma_head = nn.Linear(horizon, horizon)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch, lookback, n_features] — input returns in basis points
        
        Returns:
            mu: [batch, horizon] — predicted mean returns in bps
            sigma: [batch, horizon] — predicted std in bps
        """
        # Use first feature (returns_bps) for decomposition
        x_main = x[:, :, 0]  # [batch, lookback]
        
        # Decompose
        trend = self.moving_avg(x_main.unsqueeze(-1)).squeeze(-1)  # [batch, lookback]
        seasonal = x_main - trend
        
        # Project to horizon
        trend_out = self.linear_trend(trend)          # [batch, horizon]
        seasonal_out = self.linear_seasonal(seasonal)  # [batch, horizon]
        
        combined = trend_out + seasonal_out  # [batch, horizon]
        
        # Probabilistic head
        mu = self.mu_head(combined)  # [batch, horizon]
        sigma_raw = self.sigma_head(combined)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(sigma_raw)
        
        return mu, sigma
    
    def sample_paths(self, mu, sigma, n_paths=1000, current_price=None):
        """
        Sample N simulated price paths from the predicted distribution.
        
        Args:
            mu: [batch, horizon] — predicted mean returns in bps
            sigma: [batch, horizon] — predicted std in bps  
            n_paths: number of simulation paths
            current_price: starting price for converting bps → prices
        
        Returns:
            paths: [batch, n_paths, horizon+1] — simulated price paths
        """
        batch_size, horizon = mu.shape
        
        # Expand for sampling: [batch, n_paths, horizon]
        mu_exp = mu.unsqueeze(1).expand(-1, n_paths, -1)
        sigma_exp = sigma.unsqueeze(1).expand(-1, n_paths, -1)
        
        # Sample returns in basis points
        eps = torch.randn_like(mu_exp)
        sampled_returns_bps = mu_exp + sigma_exp * eps
        
        # Convert bps returns to multiplicative factors
        # return_bps = (price_new / price_old - 1) * 10000
        # price_new = price_old * (1 + return_bps / 10000)
        factors = 1.0 + sampled_returns_bps / 10000.0
        
        # Cumulative product to get price levels
        cum_factors = torch.cumprod(factors, dim=-1)  # [batch, n_paths, horizon]
        
        # Prepend the starting factor of 1.0
        ones = torch.ones(batch_size, n_paths, 1, device=mu.device)
        cum_factors = torch.cat([ones, cum_factors], dim=-1)  # [batch, n_paths, horizon+1]
        
        if current_price is not None:
            # current_price: [batch] or scalar
            if isinstance(current_price, (int, float)):
                paths = cum_factors * current_price
            else:
                paths = cum_factors * current_price.unsqueeze(1).unsqueeze(2)
        else:
            paths = cum_factors
        
        return paths


class GaussianNLLLoss(nn.Module):
    """Negative log-likelihood loss for Gaussian predictions."""
    
    def forward(self, mu, sigma, target):
        """
        Args:
            mu: [batch, horizon] predicted means
            sigma: [batch, horizon] predicted stds
            target: [batch, horizon] actual returns in bps
        """
        var = sigma ** 2 + 1e-6
        nll = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
        return nll.mean()


class CRPSLoss(nn.Module):
    """
    Approximate CRPS loss for training.
    Uses the closed-form CRPS for Gaussian distributions.
    
    CRPS(N(μ,σ²), x) = σ[x̃(2Φ(x̃)-1) + 2φ(x̃) - 1/√π]
    where x̃ = (x-μ)/σ, Φ is CDF, φ is PDF of standard normal.
    """
    
    def forward(self, mu, sigma, target):
        """
        Args:
            mu: [batch, horizon]
            sigma: [batch, horizon]
            target: [batch, horizon]
        """
        sigma = sigma.clamp(min=1e-6)
        x_norm = (target - mu) / sigma
        
        # Standard normal PDF and CDF
        phi = torch.exp(-0.5 * x_norm ** 2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1 + torch.erf(x_norm / np.sqrt(2)))
        
        crps = sigma * (x_norm * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
        return crps.mean()
```

---

## Custom Model Interface

All models must implement this interface to work with the pipeline:

```python
from abc import ABC, abstractmethod
import torch

class BaseSynthModel(ABC):
    """Abstract base class for Synth Subnet prediction models."""
    
    @abstractmethod
    def __init__(self, config: dict):
        """Initialize model from configuration dictionary."""
        pass
    
    @abstractmethod
    def fit(self, train_data: dict, val_data: dict = None) -> dict:
        """
        Train the model.
        
        Args:
            train_data: {"features": Tensor, "targets": Tensor, "timestamps": array}
            val_data: optional validation data in same format
        
        Returns:
            metrics: {"train_loss": float, "val_loss": float, ...}
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        features: torch.Tensor,
        current_price: float,
        n_paths: int = 1000,
        horizon: int = 288,
    ) -> list[list[float]]:
        """
        Generate N simulated price paths.
        
        Args:
            features: [1, lookback, n_features] input features
            current_price: current asset price
            n_paths: number of paths to generate
            horizon: number of future timesteps
        
        Returns:
            paths: list of n_paths lists, each with horizon+1 floats
                   First value in each path = current_price
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass
    
    @property
    @abstractmethod
    def config(self) -> dict:
        """Model configuration for reproducibility."""
        pass
```

---

## Model Search Space

### Default Search Space for DLinear+Gaussian

```python
DLINEAR_SEARCH_SPACE = {
    "lookback": [144, 288, 576],         # 12h, 24h, 48h of 5-min data
    "kernel_size": [13, 25, 49],          # MA kernel
    "min_sigma": [0.5, 1.0, 2.0],        # Minimum uncertainty
    "max_sigma": [200, 500, 1000],        # Maximum uncertainty
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size": [32, 64, 128],
    "n_epochs": [50, 100, 200],
}

# Total configurations: 3^7 = 2187
# With random search (50 configs): ~1-2 hours on GPU
```

### Adding New Model Types

To add a new model to the search:

1. Create `models/custom/my_model.py` implementing `BaseSynthModel`
2. Add to `config/pipeline.yaml`:
```yaml
models:
  dlinear_gaussian:
    class: models.dlinear_gaussian.DLinearGaussianModel
    search_space: ...
  my_model:
    class: models.custom.my_model.MyModel
    search_space:
      param1: [val1, val2]
      param2: [val3, val4]
```
3. The search loop will automatically include it

### Model Ideas Beyond DLinear

| Model | Complexity | Potential Edge |
|-------|-----------|----------------|
| **Temporal Fusion Transformer** | High | Multi-horizon attention, variable selection |
| **N-BEATS** | Medium | Interpretable basis expansion |
| **PatchTST** | High | Patch-based attention, channel independence |
| **Simple GRU + Gaussian** | Low | Sequence memory, fast |
| **Mixture Density Network** | Medium | Multi-modal distributions (better for regime changes) |
| **Normalizing Flows** | High | Flexible non-Gaussian distributions |
| **Copula Models** | Medium | Cross-asset correlation modeling |
| **GARCH + NN hybrid** | Medium | Econometric vol clustering + neural flexibility |

### Key Insight for Synth

The scoring is CRPS on **basis point changes** at multiple intervals. This means:
- Short intervals (5-min) dominate the score by count (288 of them)
- But longer intervals (3-hour) test distributional accuracy at scale
- The model needs to get BOTH short-term noise AND long-term drift right
- A model that's great at 5-min but terrible at 3-hour will still lose

The Gaussian head assumption is reasonable for short intervals but may be too thin-tailed for
longer ones. Consider:
- Student-t distribution head (heavier tails)
- Mixture of Gaussians (multi-modal)
- Quantile regression (distribution-free)

---

## Training Loop

```python
def train_model(model, train_loader, val_loader, config):
    """Standard training loop with early stopping."""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["n_epochs"]
    )
    
    # Use CRPS loss for training (directly optimizes the scoring metric)
    criterion = CRPSLoss()
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(config["n_epochs"]):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            features, targets = batch
            mu, sigma = model(features)
            loss = criterion(mu, sigma, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                features, targets = batch
                mu, sigma = model(features)
                loss = criterion(mu, sigma, targets)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    model.load_state_dict(torch.load("best_model.pt"))
    return model, {"best_val_loss": best_val_loss}
```
