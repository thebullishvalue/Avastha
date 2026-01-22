# ◈ AVASTHA - Adaptive Market Regime Detection System

**Version:** 2.0.0  
**Author:** Hemrek Capital

---

## Overview

AVASTHA (अवस्था - Sanskrit for "state/condition") is an **adaptive** market regime detection system that uses advanced quantitative methods to classify current market conditions. Unlike traditional systems that rely on fixed thresholds, AVASTHA uses:

1. **Rolling Percentile Normalization** - All indicators scored relative to their own history
2. **Hidden Markov Models (HMM)** - Learn regime patterns from data
3. **Kalman Filtering** - Adaptive score smoothing
4. **GARCH-inspired Volatility Adjustment** - Sensitivity scales with market volatility
5. **CUSUM Change Point Detection** - Detect structural breaks
6. **Bayesian Confidence Scoring** - Probabilistic regime assessment

---

## Why Adaptive?

**The Problem with Fixed Thresholds:**
```
Traditional: if RSI > 70 then "overbought"
```
This fails because:
- RSI > 70 in a strong bull market is normal
- RSI > 70 in a bear market rally is extreme
- Market microstructure changes over time

**AVASTHA's Solution:**
```
Adaptive: percentile_rank(RSI, rolling_history)
```
Every indicator is scored relative to its own recent history, making the system automatically sensitive to regime changes.

---

## Mathematical Framework

### 1. Percentile Normalization (NO Fixed Thresholds)

Instead of hardcoded thresholds, we use:
```
percentile_rank = (value - rolling_min) / (rolling_max - rolling_min)
adaptive_score = 2 * percentile_rank - 1  # Maps to [-1, +1]
```

### 2. Hidden Markov Model (HMM)

States: Bull (0), Neutral (1), Bear (2)

**Transition Matrix** (learned from data):
```
        Bull    Neutral  Bear
Bull    0.85    0.10     0.05
Neutral 0.10    0.80     0.10  
Bear    0.05    0.10     0.85
```

**Forward Algorithm:**
```
α_t(j) = P(O_1,...,O_t, S_t=j)
P(State | Observations) ∝ P(Observation | State) × P(State | Previous)
```

### 3. Kalman Filter

**State Equation:** x_t = x_{t-1} + w_t (random walk)  
**Observation:** z_t = x_t + v_t

**Kalman Gain:** K = P_{t|t-1} / (P_{t|t-1} + R)

The filter optimally combines prediction and measurement based on their uncertainties.

### 4. GARCH Volatility Regime

```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

Volatility multiplier adjusts score sensitivity:
- **LOW volatility:** multiplier = 1.3 (more sensitive)
- **NORMAL:** multiplier = 1.0
- **HIGH:** multiplier = 0.8 (less sensitive)
- **EXTREME:** multiplier = 0.6

### 5. CUSUM Change Point Detection

```
S⁺_t = max(0, S⁺_{t-1} + (x_t - μ - k))
S⁻_t = max(0, S⁻_{t-1} - (x_t - μ + k))
```

When S⁺_t > h or S⁻_t > h, a structural break is detected.

### 6. Bayesian Confidence

```
P(Regime | Data) ∝ P(Data | Regime) × P(Regime)
```

Confidence combines:
- HMM state probability dominance
- Factor agreement (do all factors point same direction?)
- Data sufficiency
- Average factor confidence

---

## Regime Types

| Regime | Description | Suggested Action |
|--------|-------------|------------------|
| STRONG_BULL 🚀 | Exceptional bullish conditions | Aggressive Bull Mix |
| BULL 🐂 | Clear bullish trend | Bull Market Mix |
| WEAK_BULL 📈 | Mildly bullish | Cautious Bull Mix |
| NEUTRAL 📊 | No clear direction | Balanced Mix |
| WEAK_BEAR 📉 | Mildly bearish | Cautious Bear Mix |
| BEAR 🐻 | Clear bearish trend | Bear Market Mix |
| CRISIS 🔥 | Extreme bearish/panic | Defensive Mix |
| TRANSITION ⚡ | Regime change in progress | Reduce Exposure |

---

## Factors Analyzed

All factors use **rolling percentile normalization**:

| Factor | Weight | What it Measures |
|--------|--------|------------------|
| Momentum | 25% | RSI trends and levels (percentile-based) |
| Trend | 25% | MA alignment, price vs 200 DMA |
| Breadth | 20% | % of universe bullish/bearish |
| Velocity | 15% | Rate of change in momentum |
| Extremes | 10% | Statistical outliers (Z-score based) |
| Volatility | 5% | Bollinger Band Width regime |

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Requirements
- streamlit>=1.28.0
- pandas>=2.0.0
- numpy>=1.24.0
- yfinance>=0.2.31
- plotly>=5.18.0
- requests>=2.31.0

---

## Usage

1. **Select Universe:**
   - ETF Universe (30 curated sectoral ETFs)
   - F&O Stocks (~200+ liquid derivatives)
   - Index Constituents (16 NSE indices)

2. **Select Analysis Type:**
   - Single Day: Point-in-time regime detection
   - Time Series: Track regime evolution over date range

3. **Run Analysis**

4. **Interpret Results:**
   - **HMM Probabilities:** P(Bull), P(Neutral), P(Bear)
   - **Composite Score:** Kalman-filtered adaptive score
   - **Volatility Regime:** Current vol environment
   - **Regime Persistence:** How long current state has lasted
   - **Change Point Alert:** Structural break detection

---

## Key Advantages

### vs Fixed Threshold Systems:
✅ Automatically adapts to changing market conditions  
✅ No manual threshold tuning required  
✅ Captures microstructure changes  
✅ Works across different market regimes  

### vs Simple Moving Average Crossovers:
✅ Probabilistic output (confidence levels)  
✅ Multi-factor analysis  
✅ Regime persistence tracking  
✅ Change point detection  

### vs Traditional Technical Analysis:
✅ Mathematically rigorous (HMM, Kalman, GARCH)  
✅ Data-driven, not rule-based  
✅ Bayesian uncertainty quantification  
✅ Structural break awareness  

---

## Files

```
avastha/
├── app.py                      # Streamlit UI
├── adaptive_regime_detector.py # Core adaptive detection engine
├── regime_detector.py          # Legacy detector (for reference)
├── data_engine.py              # Multi-universe data fetching
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Configuration

### Lookback Period
Default: 60 periods for percentile calculation

### Factor Weights
Modify `FACTOR_WEIGHTS` in `adaptive_regime_detector.py`:
```python
FACTOR_WEIGHTS = {
    'momentum': 0.25,
    'trend': 0.25,
    'breadth': 0.20,
    'velocity': 0.15,
    'extremes': 0.10,
    'volatility': 0.05
}
```

### HMM Parameters
Transition matrix adapts automatically from data, but initial values can be modified in `HMMState` class.

### Volatility Parameters
GARCH parameters in `VolatilityState`:
- omega (ω): 0.0001
- alpha (α): 0.1  
- beta (β): 0.85

---

## License

Proprietary - Hemrek Capital
