# आवस्था AVASTHA - Market Regime Detection System

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" />
  <img src="https://img.shields.io/badge/python-3.9+-green.svg" />
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" />
</p>

A **hedge-fund grade multi-model market regime detection system** that synthesizes multiple analytical approaches to identify market states with institutional-level precision.

## 🌊 Overview

AVASTHA (आवस्था - Sanskrit for "state" or "condition") is a sophisticated market regime detection engine that combines:

- **Volatility Regime Detection** - GARCH-inspired volatility clustering
- **Momentum Phase Detection** - Multi-factor momentum analysis with divergence detection
- **Trend State Identification** - Multi-timeframe moving average alignment and ADX-style strength
- **Risk Regime Analysis** - VaR, CVaR, drawdown, and tail risk metrics
- **Hidden Markov Model-inspired State Detection** - Gaussian Mixture Model clustering
- **Structural Break Detection** - Change-point and regime transition identification

## 📊 Regime Classifications

| Regime | Description | Typical Exposure |
|--------|-------------|------------------|
| **CRISIS** | Extreme bearish, high volatility, capitulation | 20% |
| **BEAR_ACCELERATION** | Downtrend gaining momentum | 30% |
| **BEAR_DECELERATION** | Downtrend losing steam, potential reversal | 50% |
| **ACCUMULATION** | Sideways, low vol, smart money buying | 70% |
| **EARLY_BULL** | New uptrend emerging | 100% |
| **BULL_TREND** | Established uptrend with momentum | 120% |
| **BULL_EUPHORIA** | Overbought, excessive optimism | 80% |
| **DISTRIBUTION** | Topping pattern, smart money selling | 50% |
| **CHOP** | Rangebound, no clear direction | 60% |
| **TRANSITION** | Regime change in progress | 50% |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

### Analysis Modes

**Individual Scrip Mode** - Deep-dive analysis of any single symbol:
- Enter any NSE symbol (e.g., `RELIANCE.NS`) or Yahoo Finance ticker
- Quick-select buttons for popular indices and stocks
- Full regime breakdown with price chart, oscillator, and probabilities

**Index Universe Mode** - Screen entire universes for regime classification:
- **ETF Universe**: 30 curated ETFs from Pragyam (sectors, indices, commodities)
- **F&O Stocks**: NSE Futures & Options stock universe (~200 stocks)
- **Index Constituents**: NIFTY 50, NIFTY 500, NIFTY BANK, and 13 other NSE indices

### Basic Usage (Python)

```python
from avastha_engine import MarketRegimeEngine, detect_regime
import yfinance as yf

# Fetch data
ticker = yf.Ticker("^NSEI")
df = ticker.history(period="1y")
prices = df['Close']

# Detect regime
signal = detect_regime(prices)

print(f"Regime: {signal.primary_regime.value}")
print(f"Confidence: {signal.primary_confidence*100:.1f}%")
print(f"Oscillator: {signal.composite_oscillator:.1f}")
print(f"Recommended Exposure: {signal.recommended_exposure*100:.0f}%")
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

## 📈 Output: RegimeSignal

The `RegimeSignal` dataclass contains comprehensive regime information:

```python
@dataclass
class RegimeSignal:
    primary_regime: RegimeType          # Main regime classification
    primary_confidence: float           # Confidence level [0-1]
    regime_probabilities: Dict          # Probability distribution across all regimes
    volatility_regime: VolatilityRegime # COMPRESSED/NORMAL/ELEVATED/EXTREME
    volatility_percentile: float        # Where current vol stands [0-100]
    momentum_regime: MomentumRegime     # STRONG_BEARISH to STRONG_BULLISH
    momentum_score: float               # Composite score [-100, 100]
    trend_regime: TrendRegime           # STRONG_DOWNTREND to STRONG_UPTREND
    trend_strength: float               # ADX-style strength [0-100]
    risk_score: float                   # Composite risk [0-100]
    transition_probability: float       # Probability of regime change [0-1]
    recommended_exposure: float         # Suggested portfolio exposure [0-1.5]
    signal_strength: float              # Conviction level [0-1]
    composite_oscillator: float         # Unified oscillator [-100, 100]
    sub_signals: Dict                   # Underlying indicator values
```

## 🔧 Configuration

Customize the engine behavior with `RegimeConfig`:

```python
from avastha_engine import MarketRegimeEngine, RegimeConfig

config = RegimeConfig(
    volatility_lookback=20,      # Volatility calculation window
    momentum_fast=12,            # Fast EMA for MACD
    momentum_slow=26,            # Slow EMA for MACD
    trend_short=20,              # Short-term MA
    trend_medium=50,             # Medium-term MA
    trend_long=200,              # Long-term MA
    hmm_n_states=5,              # Number of GMM states
    vol_percentile_window=252,   # Historical vol percentile window
    risk_free_rate=0.05,         # For Sortino calculation
)

engine = MarketRegimeEngine(config)
```

## 📊 Composite Oscillator

The **Regime Oscillator** is a unified signal ranging from -100 (extremely bearish) to +100 (extremely bullish):

```
[-100] ──────────────── [0] ──────────────── [+100]
 BEARISH              NEUTRAL              BULLISH
```

Components:
- **Momentum Component** (35%): RSI, MACD, ROC
- **Trend Component** (25%): MA alignment × ADX strength
- **Volatility Component** (20%): Inverse of vol percentile
- **Risk Component** (20%): Inverse of risk score

## 🎯 Sub-Regime Analysis

### Volatility Regimes
| State | Description |
|-------|-------------|
| COMPRESSED | Low volatility, often precedes big moves |
| NORMAL | Average volatility |
| ELEVATED | Above average, caution advised |
| EXTREME | Crisis-level volatility |

### Momentum Regimes
| State | Score Range |
|-------|-------------|
| STRONG_BEARISH | < -60, RSI < 30 |
| BEARISH | < -25 |
| NEUTRAL | -25 to +25 |
| BULLISH | > +25 |
| STRONG_BULLISH | > +60, RSI > 70 |

### Trend Regimes
| State | Conditions |
|-------|------------|
| STRONG_DOWNTREND | Alignment ≤ 1, slope < -0.1%, ADX > 25 |
| DOWNTREND | Alignment ≤ 2, slope < 0 |
| SIDEWAYS | No clear direction |
| UPTREND | Alignment ≥ 3, slope > 0 |
| STRONG_UPTREND | Alignment ≥ 4, slope > 0.1%, ADX > 25 |

## 📈 Historical Analysis

Track regime evolution over time:

```python
engine = MarketRegimeEngine()
history = engine.detect_regime_history(prices, lookback_window=100)

# Returns DataFrame with columns:
# date, price, regime, confidence, oscillator, exposure,
# vol_regime, mom_regime, trend_regime, risk_score, signal_strength
```

## 🔌 Integration Examples

### With Trading Systems

```python
signal = engine.detect_regime(prices)

# Position sizing
base_position = 100000
adjusted_position = base_position * signal.recommended_exposure

# Strategy selection
if signal.is_bullish():
    strategy = "Trend Following"
elif signal.is_bearish():
    strategy = "Hedging / Cash"
else:
    strategy = "Range Trading"

# Risk alerts
if signal.is_defensive():
    print("DEFENSIVE POSITIONING ADVISED")
```

### JSON Serialization

```python
signal_dict = signal.to_dict()
import json
json_output = json.dumps(signal_dict, indent=2)
```

## 📁 Project Structure

```
market_regime/
├── app.py              # Streamlit dashboard
├── avastha_engine.py   # Core detection engine (standalone module)
├── examples.py         # Usage examples
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## 🧮 Mathematical Foundation

### Volatility Detection
- **Realized Volatility**: σ = √(252) × std(returns, window)
- **Volatility Percentile**: percentile_rank(current_vol, historical_vol)
- **Vol Clustering**: autocorr(returns², lag=1)

### Momentum Score
```
momentum_score = 0.35 × (RSI - 50) × 2 
               + 0.35 × tanh(MACD_hist / price × 100) × 100
               + 0.30 × clip(ROC_20 × 2, -100, 100)
```

### Trend Strength (ADX-inspired)
```
+DI = 100 × SMA(+DM, 14) / ATR
-DI = 100 × SMA(-DM, 14) / ATR
DX = 100 × |+DI - -DI| / (+DI + -DI)
ADX = SMA(DX, 14)
```

### Risk Score
```
risk_score = 0.25 × drawdown_component
           + 0.25 × var_component
           + 0.20 × tail_risk_component
           + 0.30 × downside_vol_component
```

## 📜 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines first.

---

<p align="center">
  <b>AVASTHA</b> - Understanding Market States with Precision
</p>

## 🏦 ETF Universe (Pragyam Symbols)

The ETF Universe contains 30 curated ETFs from the Pragyam system:

| Symbol | Category | Symbol | Category |
|--------|----------|--------|----------|
| NIFTYIETF.NS | NIFTY 50 | MON100.NS | NIFTY 100 |
| MONIFTY500.NS | NIFTY 500 | MASPTOP50.NS | Top 50 |
| MIDCAPIETF.NS | Midcap | MOSMALL250.NS | Smallcap 250 |
| FINIETF.NS | Financials | PVTBANIETF.NS | Private Banks |
| PSUBNKIETF.NS | PSU Banks | ECAPINSURE.NS | Insurance |
| ITIETF.NS | IT | AUTOIETF.NS | Auto |
| FMCGIETF.NS | FMCG | HEALTHIETF.NS | Healthcare |
| CONSUMIETF.NS | Consumption | INFRAIETF.NS | Infrastructure |
| METALIETF.NS | Metal | OILIETF.NS | Oil & Gas |
| CHEMICAL.NS | Chemicals | MODEFENCE.NS | Defence |
| MOREALTY.NS | Realty | CPSETF.NS | CPSE |
| MAKEINDIA.NS | Make in India | EVINDIA.NS | EV India |
| MNC.NS | MNC | TNIDETF.NS | Tamil Nadu Infra |
| GOLDIETF.NS | Gold | SILVERIETF.NS | Silver |
| COMMOIETF.NS | Commodities | GROWWPOWER.NS | Power & Energy |

## 📈 NSE Index Options

Available indices for constituent screening:

- NIFTY 50, NIFTY NEXT 50, NIFTY 100, NIFTY 200, NIFTY 500
- NIFTY MIDCAP 50, NIFTY MIDCAP 100, NIFTY SMLCAP 100
- NIFTY BANK, NIFTY AUTO, NIFTY FIN SERVICE
- NIFTY FMCG, NIFTY IT, NIFTY MEDIA, NIFTY METAL, NIFTY PHARMA
