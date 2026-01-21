# ◈ AVASTHA - Market Regime Detection System

**A Pragyam Product Family Member**

**Version:** 1.0.0  
**Author:** Hemrek Capital

---

## Overview

AVASTHA (अवस्था - Sanskrit for "state/condition") is an institutional-grade market regime detection system that uses multi-factor analysis to classify current market conditions. It analyzes momentum, trend, breadth, volatility, and statistical extremes across your chosen universe to determine whether the market is in a bullish, bearish, or consolidation state.

## Features

### 🎯 Multi-Factor Analysis
- **7 Analysis Factors** with weighted scoring
- Momentum (30%), Trend (25%), Breadth (15%), Velocity (15%), Extremes (10%), Volatility (5%)

### 📊 7 Regime Classifications
| Regime | Score Range | Suggested Mix |
|--------|-------------|---------------|
| STRONG_BULL | ≥ 1.5 | 🐂 Bull Market Mix |
| BULL | ≥ 1.0 | 🐂 Bull Market Mix |
| WEAK_BULL | ≥ 0.5 | 📊 Chop/Consolidation Mix |
| CHOP | ≥ 0.1 | 📊 Chop/Consolidation Mix |
| WEAK_BEAR | ≥ -0.1 | 📊 Chop/Consolidation Mix |
| BEAR | ≥ -0.5 | 🐻 Bear Market Mix |
| CRISIS | < -0.5 | 🐻 Bear Market Mix |

### 🌐 Multiple Universes
- **ETF Universe**: 28 curated sectoral and thematic ETFs
- **F&O Stocks**: ~200+ Futures & Options eligible stocks (requires nsepython)
- **Index Constituents**: 16 NSE indices including NIFTY 50, 100, 200, 500, sectoral indices

### 📈 Analysis Modes
- **Single Day**: Analyze regime for a specific date
- **Time Series**: Track regime evolution over a date range

## Installation

```bash
# Clone or download the system
cd avastha

# Install dependencies
pip install -r requirements.txt

# Optional: For F&O Stock Universe
pip install nsepython

# Run the application
streamlit run app.py
```

## Usage

1. **Select Universe**: Choose ETF Universe, F&O Stocks, or Index Constituents
2. **Select Analysis Type**: Single Day or Time Series
3. **Set Date(s)**: Choose analysis date or date range
4. **Run Analysis**: Click "RUN ANALYSIS" to execute

## File Structure

```
avastha/
├── app.py              # Main Streamlit application (Pragyam Design System)
├── regime_detector.py  # Core regime detection engine
├── data_engine.py      # Data fetching with universe support
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## Analysis Factors

### 1. Momentum (30%)
- RSI trend and current value analysis
- Liquidity Oscillator trend analysis
- Classifications: STRONG_BULLISH → STRONG_BEARISH

### 2. Trend (25%)
- Price position relative to 200 DMA
- Moving average alignment (90 DMA vs 200 DMA)
- Trend consistency measurement
- Classifications: STRONG_UPTREND → STRONG_DOWNTREND

### 3. Breadth (15%)
- RSI bullish percentage (> 50)
- Oscillator positive percentage
- Divergence detection (narrow vs broad participation)
- Classifications: STRONG_BROAD → CAPITULATION

### 4. Velocity (15%)
- Momentum acceleration/deceleration
- Rate of change analysis
- Classifications: ACCELERATING_UP → ACCELERATING_DOWN

### 5. Extremes (10%)
- Z-score extreme analysis
- Statistical overbought/oversold conditions
- Classifications: DEEPLY_OVERSOLD → DEEPLY_OVERBOUGHT

### 6. Volatility (5%)
- Bollinger Band Width analysis
- Volatility trend direction
- Classifications: SQUEEZE, NORMAL, ELEVATED, PANIC

## API Reference

### MarketRegimeDetector

```python
from regime_detector import MarketRegimeDetector, RegimeType

detector = MarketRegimeDetector(min_periods=10)
result = detector.detect(historical_data)

# Result attributes:
result.regime          # RegimeType enum
result.regime_name     # String name (e.g., "BULL")
result.suggested_mix   # Portfolio mix suggestion
result.confidence      # 0-1 confidence score
result.composite_score # -2 to +2 composite factor score
result.factors         # Dict of FactorAnalysis objects
result.explanation     # Human-readable explanation
result.warnings        # List of warning messages
```

### MarketDataEngine

```python
from data_engine import MarketDataEngine

engine = MarketDataEngine()
engine.set_universe("Index Constituents", "NIFTY 500")
data = engine.get_regime_data(analysis_date, lookback_days=30)
```

## Design System

AVASTHA uses the **Pragyam Design System** for consistent styling across the product family:

- **Primary Color**: #FFC300 (Gold)
- **Background**: #0F0F0F (Near Black)
- **Card Background**: #1A1A1A
- **Success**: #10b981 (Green)
- **Danger**: #ef4444 (Red)
- **Warning**: #f59e0b (Amber)
- **Font**: Inter

## Product Family

AVASTHA is part of the Pragyam Product Family:

- **Pragyam**: Quantitative Portfolio Curation System
- **UMA**: Unified Market Analysis (Signal Intelligence)
- **AVASTHA**: Market Regime Detection System

All products share the same design language and can be used together for comprehensive market analysis.

## Warnings System

AVASTHA generates warnings for:
- **Breadth Divergence**: Narrow market leadership
- **Panic Volatility**: Elevated market turbulence
- **Statistical Extremes**: Overbought/Oversold conditions

## Limitations

- Requires minimum 10 historical periods for analysis
- Data dependent on yfinance availability
- F&O universe requires nsepython package
- Best suited for liquid ETF/stock universe
- Not a trading recommendation system

## License

Proprietary - Hemrek Capital

---

*Part of the Pragyam Quantitative Systems Family*
