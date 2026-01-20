# 🔮 AVASTHA (आवस्था) - Market Regime Detection System

**Version 2.0.0** | Part of the Quantitative Analysis Suite alongside Pragyam and UMA

A hedge-fund grade multi-model market regime detection system that synthesizes 6 independent detectors into a unified market state classification.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Analysis Modes

### 📈 Individual Scrip
Deep-dive analysis of any single symbol with full regime breakdown, price chart, and oscillator visualization.

### 📊 Index Universe
Screen entire universes with comprehensive analysis dashboard:

| Universe | Description |
|----------|-------------|
| **ETF Universe** | 30 curated ETFs across 9 sectors |
| **F&O Stocks** | NSE Futures & Options (~200 stocks) |
| **Index Constituents** | 16 NSE indices (NIFTY 50, 500, BANK, etc.) |

---

## Dashboard Tabs (Index Mode)

| Tab | Features |
|-----|----------|
| **Overview** | 6 key metrics, regime donut, oscillator histogram, actionable insights |
| **Risk** | Risk gauge, breadth bars, oscillator vs exposure scatter |
| **Sectors** | Sector × Regime heatmap, sector statistics (ETF only) |
| **Screener** | Multi-filter (regime, momentum, trend, vol), range sliders, CSV export |
| **Drill-Down** | Select any symbol for detailed individual analysis |

---

## Regime Classifications

| Regime | Description | Typical Exposure |
|--------|-------------|------------------|
| CRISIS | Extreme bearish, high volatility | 20% |
| BEAR_ACCELERATION | Downtrend gaining momentum | 30% |
| BEAR_DECELERATION | Downtrend losing steam | 50% |
| ACCUMULATION | Sideways, smart money buying | 70% |
| EARLY_BULL | New uptrend emerging | 100% |
| BULL_TREND | Established uptrend | 120% |
| BULL_EUPHORIA | Overbought, excessive optimism | 80% |
| DISTRIBUTION | Topping pattern | 50% |
| CHOP | Rangebound, no direction | 60% |
| TRANSITION | Regime change in progress | 50% |

---

## Detection Engine

6 independent detectors synthesized:

1. **Volatility** - GARCH-inspired clustering, percentile ranking
2. **Momentum** - RSI, MACD, ROC, divergence detection
3. **Trend** - Multi-TF MA alignment, ADX-style strength
4. **Risk** - VaR, CVaR, drawdown, Sortino
5. **HMM State** - Gaussian Mixture Model clustering
6. **Structural Break** - Change-point detection

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Composite Oscillator** | Single signal from -100 to +100 |
| **Recommended Exposure** | Position sizing guidance (0-150%) |
| **Weighted Risk Score** | Confidence-weighted regime risk |
| **Breadth** | Bullish vs Bearish ratio |

---

## ETF Universe (30 Symbols)

**Broad Market:** NIFTY 50, 100, 500, Top 50  
**Market Cap:** Midcap, Smallcap 250  
**Banking:** Financials, Private Banks, PSU Banks, Insurance  
**Technology:** IT, MNC  
**Consumer:** FMCG, Consumption, Auto, EV India  
**Healthcare:** Healthcare  
**Industrial:** Infrastructure, CPSE, Make in India, Defence, Realty, TN Infra, Power  
**Materials:** Metal, Oil & Gas, Chemicals  
**Commodities:** Gold, Silver, Commodities  

---

## Programmatic Usage

```python
from avastha_engine import MarketRegimeEngine
import yfinance as yf

# Fetch data
prices = yf.Ticker("RELIANCE.NS").history(period="1y")['Close']

# Detect regime
engine = MarketRegimeEngine()
signal = engine.detect_regime(prices)

print(f"Regime: {signal.primary_regime.value}")
print(f"Confidence: {signal.primary_confidence*100:.0f}%")
print(f"Oscillator: {signal.composite_oscillator:+.1f}")
print(f"Exposure: {signal.recommended_exposure*100:.0f}%")
```

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard |
| `avastha_engine.py` | Core detection engine |
| `examples.py` | Usage examples |
| `requirements.txt` | Dependencies |

---

**License:** MIT
