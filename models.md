# Data Pipeline Reference

## Overview

The data pipeline sources historical price data, engineers features, and creates temporal splits
with strict anti-leakage guarantees. Every decision here affects model validity.

## Data Sources

### Primary: Binance REST API (Historical OHLCV)
```python
# Fetch 1-minute OHLCV candles
# Endpoint: GET /api/v3/klines
# Params: symbol, interval=1m, startTime, endTime, limit=1000

BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XAU": "XAUUSDT",  # Available on Binance as PAXGUSDT or use alternative
    # Tokenized equities — use alternative sources (see below)
}
```

### For Tokenized Equities (SPYX, NVDAX, TSLAX, AAPLX, GOOGLX)
These are Mode Network tokenized versions. Source underlying equity data from:
- Yahoo Finance API (yfinance) for SPY, NVDA, TSLA, AAPL, GOOGL
- Map to tokenized versions (prices track closely)
- Note: equities only trade during market hours — handle gaps appropriately

### Secondary: Pyth Oracle (Live Prices)
```python
# For live scoring and real-time features
# Use Pyth's Hermes API: https://hermes.pyth.network/api/latest_price_feeds
PYTH_PRICE_IDS = {
    "BTC": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    "XAU": "0x765d2ba906dbc32ca17cc11f5310a89e9ee1f6420508c63861f2f8ba4ee34bb2",
}
```

## Data Schema

### Raw OHLCV (1-minute granularity)
```
timestamp       : datetime64[ns, UTC]  — candle open time
open            : float64              — open price
high            : float64              — high price
low             : float64              — low price
close           : float64              — close price
volume          : float64              — base asset volume
quote_volume    : float64              — quote asset volume
num_trades      : int64                — number of trades
```

### Minimum Data Requirements
- **For 24-hour challenge**: Minimum 90 days of 1-min data per asset
- **For 1-hour challenge**: Minimum 30 days of 1-min data per asset
- **Recommended**: 180+ days for robust walk-forward validation

---

## Anti-Leakage Rules

### RULE 1: Temporal Splits Only
```
NEVER do this:
  train, val = train_test_split(data, test_size=0.2, shuffle=True)  # FATAL

ALWAYS do this:
  split_time = data.timestamp.quantile(0.7)
  train = data[data.timestamp < split_time]
  val = data[data.timestamp >= split_time]
```

### RULE 2: Purge Gap Between Train and Val/Test
The forecast horizon creates overlapping information. A 24-hour forecast made at the end of
training could overlap with the start of validation. Insert a gap:

```python
PURGE_GAP = {
    "24h": pd.Timedelta(hours=26),   # 24h horizon + 2h buffer
    "1h":  pd.Timedelta(hours=2),    # 1h horizon + 1h buffer
}

# Example:
train_end = pd.Timestamp("2025-01-30 00:00:00")
val_start = train_end + PURGE_GAP["24h"]  # 2025-01-31 02:00:00
```

### RULE 3: Causal Feature Computation
Every feature must be computable using ONLY past data at the point of prediction.

```python
# WRONG — uses future data in the rolling window
df["vol_20"] = df["returns"].rolling(20).std()  # includes current row!

# RIGHT — shift to exclude current observation
df["vol_20"] = df["returns"].shift(1).rolling(20).std()

# WRONG — centering or bilateral operations
df["smooth"] = df["close"].rolling(20, center=True).mean()

# RIGHT — backward-looking only
df["smooth"] = df["close"].shift(1).rolling(20).mean()
```

### RULE 4: No Embargo Violation
Forecast windows must not overlap between splits.

```
Train forecasts:  |---24h---|---24h---|---24h---| END
                                                  [PURGE GAP]
Val forecasts:                                              |---24h---|---24h---|
```

### RULE 5: Walk-Forward Expanding Window
```python
def create_walk_forward_folds(data, n_folds=5, val_days=10, purge_hours=26):
    """
    Creates expanding-window walk-forward folds.
    
    Fold 0: Train [0, T0)         → Purge → Val [T0+purge, T0+purge+val_days)
    Fold 1: Train [0, T1)         → Purge → Val [T1+purge, T1+purge+val_days)
    ...
    Final:  Train [0, T_final)    → Purge → Test [T_final+purge, end)
    """
    total_days = (data.timestamp.max() - data.timestamp.min()).days
    train_start = data.timestamp.min()
    
    # Reserve last 20 days for final test
    available = total_days - 20
    fold_step = (available - 30) // n_folds  # Start with min 30 days train
    
    folds = []
    for i in range(n_folds):
        train_days = 30 + (i * fold_step)
        train_end = train_start + pd.Timedelta(days=train_days)
        val_start = train_end + pd.Timedelta(hours=purge_hours)
        val_end = val_start + pd.Timedelta(days=val_days)
        
        folds.append({
            "train": (train_start, train_end),
            "val": (val_start, val_end),
            "fold_id": i,
        })
    
    # Final test fold
    test_start = data.timestamp.max() - pd.Timedelta(days=20)
    folds.append({
        "train": (train_start, test_start - pd.Timedelta(hours=purge_hours)),
        "val": (test_start, data.timestamp.max()),
        "fold_id": "test",
    })
    
    return folds
```

---

## Feature Engineering

### Base Features (computed causally)

```python
def compute_features(df, asset):
    """All features use .shift(1) to prevent lookahead."""
    
    # Returns in basis points (what CRPS scores on)
    df["ret_bps"] = (df["close"] / df["close"].shift(1) - 1) * 10000
    
    # Multi-horizon returns
    for horizon in [5, 30, 60, 180, 720, 1440]:  # minutes
        df[f"ret_{horizon}m_bps"] = (
            df["close"] / df["close"].shift(horizon) - 1
        ) * 10000
    
    # Realized volatility (rolling, shifted)
    for window in [30, 60, 288, 1440]:  # minutes
        df[f"rvol_{window}"] = (
            df["ret_bps"].shift(1).rolling(window).std()
        )
    
    # VWAP deviation
    df["vwap"] = (
        (df["close"].shift(1) * df["volume"].shift(1))
        .rolling(60).sum() / df["volume"].shift(1).rolling(60).sum()
    )
    df["vwap_dev_bps"] = (df["close"].shift(1) / df["vwap"] - 1) * 10000
    
    # High-low range (proxy for intraday vol)
    df["hl_range_bps"] = (
        (df["high"].shift(1) - df["low"].shift(1)) / df["close"].shift(1)
    ) * 10000
    
    # Volume features
    df["vol_ratio"] = df["volume"].shift(1) / df["volume"].shift(1).rolling(60).mean()
    
    # Time features (cyclical encoding)
    df["hour_sin"] = np.sin(2 * np.pi * df.timestamp.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.timestamp.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.timestamp.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.timestamp.dt.dayofweek / 7)
    
    return df.dropna()
```

### Features for the Gaussian Head
The model needs to predict both μ (mean) and σ (standard deviation) of price changes.
Key features that inform σ:
- Recent realized volatility at multiple scales
- High-low range
- Volume ratio (unusual volume → higher uncertainty)
- Time of day (Asian session vs US open have different vol profiles)

---

## Data Fetching Implementation

```python
import pandas as pd
import requests
import time
from pathlib import Path

def fetch_binance_klines(symbol, interval="1m", days=90, save_dir="data/raw"):
    """Fetch historical klines from Binance with rate limiting."""
    
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1000,
        }
        
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        if not data:
            break
        
        all_data.extend(data)
        current_start = data[-1][0] + 1  # Next ms after last candle
        time.sleep(0.1)  # Rate limiting
    
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    
    # Keep only needed columns
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "num_trades"]]
    
    save_path = Path(save_dir) / f"{symbol.lower()}_1m.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)
    
    return df

def fetch_all_assets(days=90):
    """Fetch data for all supported assets."""
    SYMBOLS = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        # XAU: use PAXGUSDT as proxy, or alternative gold feed
        "XAU": "PAXGUSDT",
    }
    
    datasets = {}
    for asset, symbol in SYMBOLS.items():
        print(f"Fetching {asset} ({symbol})...")
        datasets[asset] = fetch_binance_klines(symbol, days=days)
        print(f"  → {len(datasets[asset])} candles")
    
    # For equity assets, use yfinance
    # fetch_equity_assets(days=days)  # Separate function
    
    return datasets
```

---

## Resampling for Challenge Intervals

The raw data is 1-minute. For the 24-hour challenge (5-minute intervals), resample:

```python
def resample_to_interval(df_1m, interval_minutes):
    """Resample 1-min data to target interval, preserving OHLCV semantics."""
    df = df_1m.set_index("timestamp")
    resampled = df.resample(f"{interval_minutes}min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "num_trades": "sum",
    }).dropna()
    return resampled.reset_index()
```

## Data Validation Checks

Before training, always run:
```python
def validate_data(df, asset):
    """Sanity checks on processed data."""
    assert df.timestamp.is_monotonic_increasing, "Data not sorted by time"
    assert df.timestamp.dt.tz is not None, "Timestamps must be UTC"
    assert not df.duplicated(subset=["timestamp"]).any(), "Duplicate timestamps"
    assert (df["close"] > 0).all(), "Non-positive prices found"
    assert df.isna().sum().sum() == 0, f"NaN values in {asset} data"
    
    # Check for large gaps
    gaps = df.timestamp.diff()
    large_gaps = gaps[gaps > pd.Timedelta(minutes=5)]
    if len(large_gaps) > 0:
        print(f"WARNING: {len(large_gaps)} gaps > 5min in {asset} data")
    
    print(f"✓ {asset}: {len(df)} rows, {df.timestamp.min()} to {df.timestamp.max()}")
```
