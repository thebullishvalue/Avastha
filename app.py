"""
AVASTHA (आवस्था) - Market Regime Detection System
==================================================
Hedge-fund grade multi-model regime detection dashboard.

Part of the Quantitative Analysis Suite alongside Pragyam and UMA.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import time
import requests
from io import StringIO

warnings.filterwarnings('ignore')

# Import the engine
from avastha_engine import (
    MarketRegimeEngine, RegimeType, RegimeConfig, RegimeSignal,
    VolatilityRegime, MomentumRegime, TrendRegime,
    detect_regime, get_regime_color, get_regime_description
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "v1.0.0"

# ETF Universe (30 symbols from Pragyam)
ETF_UNIVERSE = [
    "GROWWPOWER.NS", "NIFTYIETF.NS", "MON100.NS", "MAKEINDIA.NS", "SILVERIETF.NS",
    "HEALTHIETF.NS", "CONSUMIETF.NS", "GOLDIETF.NS", "INFRAIETF.NS", "CPSETF.NS",
    "TNIDETF.NS", "COMMOIETF.NS", "MODEFENCE.NS", "MOREALTY.NS", "PSUBNKIETF.NS",
    "MASPTOP50.NS", "FMCGIETF.NS", "CHEMICAL.NS", "ITIETF.NS", "EVINDIA.NS",
    "MNC.NS", "FINIETF.NS", "AUTOIETF.NS", "PVTBANIETF.NS", "MONIFTY500.NS",
    "ECAPINSURE.NS", "MIDCAPIETF.NS", "MOSMALL250.NS", "OILIETF.NS", "METALIETF.NS"
]

# ETF Display Names
ETF_DISPLAY_NAMES = {
    "GROWWPOWER.NS": "Power & Energy", "NIFTYIETF.NS": "NIFTY 50", "MON100.NS": "NIFTY 100",
    "MAKEINDIA.NS": "Make in India", "SILVERIETF.NS": "Silver", "HEALTHIETF.NS": "Healthcare",
    "CONSUMIETF.NS": "Consumption", "GOLDIETF.NS": "Gold", "INFRAIETF.NS": "Infrastructure",
    "CPSETF.NS": "CPSE", "TNIDETF.NS": "Tamil Nadu Infra", "COMMOIETF.NS": "Commodities",
    "MODEFENCE.NS": "Defence", "MOREALTY.NS": "Realty", "PSUBNKIETF.NS": "PSU Banks",
    "MASPTOP50.NS": "Top 50", "FMCGIETF.NS": "FMCG", "CHEMICAL.NS": "Chemicals",
    "ITIETF.NS": "IT", "EVINDIA.NS": "EV India", "MNC.NS": "MNC",
    "FINIETF.NS": "Financials", "AUTOIETF.NS": "Auto", "PVTBANIETF.NS": "Private Banks",
    "MONIFTY500.NS": "NIFTY 500", "ECAPINSURE.NS": "Insurance", "MIDCAPIETF.NS": "Midcap",
    "MOSMALL250.NS": "Smallcap 250", "OILIETF.NS": "Oil & Gas", "METALIETF.NS": "Metal"
}

# NSE Index Configuration
INDEX_LIST = [
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
    "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100", "NIFTY BANK",
    "NIFTY AUTO", "NIFTY FIN SERVICE", "NIFTY FMCG", "NIFTY IT",
    "NIFTY MEDIA", "NIFTY METAL", "NIFTY PHARMA"
]

BASE_URL = "https://archives.nseindia.com/content/indices/"
INDEX_URLS = {
    "NIFTY 50": f"{BASE_URL}ind_nifty50list.csv",
    "NIFTY NEXT 50": f"{BASE_URL}ind_niftynext50list.csv",
    "NIFTY 100": f"{BASE_URL}ind_nifty100list.csv",
    "NIFTY 200": f"{BASE_URL}ind_nifty200list.csv",
    "NIFTY 500": f"{BASE_URL}ind_nifty500list.csv",
    "NIFTY MIDCAP 50": f"{BASE_URL}ind_niftymidcap50list.csv",
    "NIFTY MIDCAP 100": f"{BASE_URL}ind_niftymidcap100list.csv",
    "NIFTY SMLCAP 100": f"{BASE_URL}ind_niftysmallcap100list.csv",
    "NIFTY BANK": f"{BASE_URL}ind_niftybanklist.csv",
    "NIFTY AUTO": f"{BASE_URL}ind_niftyautolist.csv",
    "NIFTY FIN SERVICE": f"{BASE_URL}ind_niftyfinancelist.csv",
    "NIFTY FMCG": f"{BASE_URL}ind_niftyfmcglist.csv",
    "NIFTY IT": f"{BASE_URL}ind_niftyitlist.csv",
    "NIFTY MEDIA": f"{BASE_URL}ind_niftymedialist.csv",
    "NIFTY METAL": f"{BASE_URL}ind_niftymetallist.csv",
    "NIFTY PHARMA": f"{BASE_URL}ind_niftypharmalist.csv"
}

# F&O Stock List URL
FNO_URL = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"

# Universe Options for Index Mode
UNIVERSE_OPTIONS = ["ETF Universe", "F&O Stocks", "Index Constituents"]

# Regime Colors (matching Pragyam aesthetic)
REGIME_COLORS = {
    RegimeType.CRISIS: "#ef4444",           # Danger red
    RegimeType.BEAR_ACCELERATION: "#f97316", # Orange
    RegimeType.BEAR_DECELERATION: "#fb923c", # Light orange
    RegimeType.ACCUMULATION: "#06b6d4",      # Cyan
    RegimeType.EARLY_BULL: "#10b981",        # Success green
    RegimeType.BULL_TREND: "#22c55e",        # Bright green
    RegimeType.BULL_EUPHORIA: "#f59e0b",     # Warning amber
    RegimeType.DISTRIBUTION: "#a855f7",      # Purple
    RegimeType.CHOP: "#888888",              # Neutral gray
    RegimeType.TRANSITION: "#FFC300",        # Primary gold
}

REGIME_EMOJIS = {
    RegimeType.CRISIS: "🔴",
    RegimeType.BEAR_ACCELERATION: "📉",
    RegimeType.BEAR_DECELERATION: "🔻",
    RegimeType.ACCUMULATION: "📦",
    RegimeType.EARLY_BULL: "🌱",
    RegimeType.BULL_TREND: "🐂",
    RegimeType.BULL_EUPHORIA: "🎪",
    RegimeType.DISTRIBUTION: "📤",
    RegimeType.CHOP: "🌀",
    RegimeType.TRANSITION: "🔄",
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG AND STYLING (Pragyam Design System)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AVASTHA | Market Regime Detection System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pragyam-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        
        --success-green: #10b981;
        --danger-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        --neutral: #888888;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main, [data-testid="stSidebar"] {
        background-color: var(--background-color);
        color: var(--text-primary);
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }
    
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
        margin-top: 2.5rem;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 {
        margin: 0;
        font-size: 2.50rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.50px;
        position: relative;
    }
    
    .premium-header .tagline {
        color: var(--text-muted);
        font-size: 1rem;
        margin-top: 0.25rem;
        font-weight: 400;
        position: relative;
    }
    
    .metric-card {
        background-color: var(--bg-card);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        border-color: var(--border-light);
    }
    
    .metric-card h4 {
        color: var(--text-muted);
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h2 {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    
    .metric-card .sub-metric {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    .metric-card.white h2 { color: var(--text-primary); }
    
    .regime-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.25rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: var(--bg-elevated);
        border: 2px solid var(--border-color);
    }
    
    .regime-card {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .regime-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    
    .oscillator-display {
        text-align: center;
        padding: 2rem;
        background: var(--bg-card);
        border-radius: 16px;
        border: 1px solid var(--border-color);
    }
    
    .oscillator-value {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .oscillator-label {
        font-size: 0.85rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-box {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-left: 0px solid var(--primary-color);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
    }
    
    .info-box h4 {
        color: var(--primary-color);
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 700;
    }

    /* Sidebar styling */
    .sidebar-title {
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }
    
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%);
        margin: 1rem 0;
    }

    /* Buttons */
    .stButton>button {
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        font-weight: 700;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6);
        background: var(--primary-color);
        color: #1A1A1A;
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Download Links */
    .download-link {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        text-decoration: none;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .download-link:hover {
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6);
        background: var(--primary-color);
        color: #1A1A1A;
        transform: translateY(-2px);
    }
    
    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        background: var(--bg-card);
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .stMarkdown table th,
    .stMarkdown table td {
        text-align: left !important;
        padding: 12px 10px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .stMarkdown table th {
        background-color: var(--bg-elevated);
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    
    .stMarkdown table tr:last-child td {
        border-bottom: none;
    }
    
    .stMarkdown table tr:hover {
        background-color: var(--bg-elevated);
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted);
        border-bottom: 2px solid transparent;
        transition: color 0.3s, border-bottom 0.3s;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-color);
    }
    .stPlotlyChart, .stDataFrame {
        border-radius: 12px;
        background-color: var(--secondary-background-color);
        padding: 10px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1);
    }
    h2 {
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 10px;
    }
    
    /* Screener table styles */
    .screener-bullish { color: #10b981 !important; }
    .screener-bearish { color: #ef4444 !important; }
    .screener-neutral { color: #888888 !important; }
    
    /* Progress indicator */
    .progress-container {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(symbol: str, period: str = "1y") -> pd.Series:
    """Fetch price data for a symbol using yfinance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        return df['Close']
    except Exception as e:
        st.warning(f"Error fetching {symbol}: {e}")
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fno_stocks() -> list:
    """Fetch F&O stock list from NSE"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(FNO_URL, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if 'SYMBOL' in df.columns:
                symbols = df['SYMBOL'].dropna().unique().tolist()
                # Filter out index symbols and add .NS suffix
                symbols = [f"{s.strip()}.NS" for s in symbols if not s.startswith('NIFTY') and s.strip()]
                return symbols[:200]  # Limit to 200
    except Exception as e:
        st.warning(f"Could not fetch F&O list: {e}")
    
    # Fallback list
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS"
    ]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_constituents(index_name: str) -> list:
    """Fetch index constituents from NSE"""
    if index_name not in INDEX_URLS:
        return []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(INDEX_URLS[index_name], headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].dropna().unique().tolist()
                return [f"{s.strip()}.NS" for s in symbols if s.strip()]
    except Exception as e:
        st.warning(f"Could not fetch {index_name} constituents: {e}")
    
    return []


def get_universe_symbols(universe: str, index_name: str = None) -> list:
    """Get list of symbols based on universe selection"""
    if universe == "ETF Universe":
        return ETF_UNIVERSE
    elif universe == "F&O Stocks":
        return fetch_fno_stocks()
    elif universe == "Index Constituents" and index_name:
        return fetch_index_constituents(index_name)
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_price_chart(prices: pd.Series, signal: RegimeSignal, symbol: str) -> go.Figure:
    """Create price chart with regime overlay"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=[f'{symbol} Price', 'Composite Oscillator']
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode='lines',
            name='Price',
            line=dict(color='#FFC300', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 195, 0, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if len(prices) >= 20:
        ma20 = prices.rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=prices.index, y=ma20, mode='lines', name='MA20',
                      line=dict(color='#06b6d4', width=1, dash='dot')),
            row=1, col=1
        )
    
    if len(prices) >= 50:
        ma50 = prices.rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=prices.index, y=ma50, mode='lines', name='MA50',
                      line=dict(color='#a855f7', width=1, dash='dot')),
            row=1, col=1
        )
    
    # Oscillator subplot
    osc_color = '#10b981' if signal.composite_oscillator > 0 else '#ef4444'
    
    # Create oscillator history (simulated from recent data)
    osc_values = []
    engine = MarketRegimeEngine()
    
    for i in range(min(60, len(prices)), 0, -5):
        try:
            subset = prices.iloc[:-i] if i > 0 else prices
            if len(subset) >= 100:
                sig = engine.detect_regime(subset)
                osc_values.append({'date': subset.index[-1], 'value': sig.composite_oscillator})
        except:
            pass
    
    # Add current value
    osc_values.append({'date': prices.index[-1], 'value': signal.composite_oscillator})
    
    if osc_values:
        osc_df = pd.DataFrame(osc_values)
        
        # Color based on positive/negative
        colors = ['#10b981' if v > 0 else '#ef4444' for v in osc_df['value']]
        
        fig.add_trace(
            go.Bar(
                x=osc_df['date'],
                y=osc_df['value'],
                name='Oscillator',
                marker_color=colors
            ),
            row=2, col=1
        )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#888888", row=2, col=1)
    
    # Layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#1A1A1A',
        font=dict(family="Inter", color='#EAEAEA'),
        margin=dict(l=60, r=40, t=60, b=40),
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(gridcolor='#2A2A2A', showgrid=True)
    fig.update_yaxes(gridcolor='#2A2A2A', showgrid=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Oscillator", range=[-100, 100], row=2, col=1)
    
    return fig


def create_regime_probability_chart(signal: RegimeSignal) -> go.Figure:
    """Create regime probability bar chart"""
    
    # Sort by probability
    sorted_regimes = sorted(signal.regime_probabilities.items(), key=lambda x: -x[1])
    
    regimes = [r[0] for r in sorted_regimes]
    probs = [r[1] * 100 for r in sorted_regimes]
    
    colors = [REGIME_COLORS.get(RegimeType(r), '#888888') for r in regimes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=regimes,
        x=probs,
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1f}%' for p in probs],
        textposition='outside',
        textfont=dict(color='#EAEAEA')
    ))
    
    fig.update_layout(
        height=400,
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#1A1A1A',
        font=dict(family="Inter", color='#EAEAEA'),
        xaxis=dict(title="Probability (%)", gridcolor='#2A2A2A', range=[0, max(probs) * 1.2]),
        yaxis=dict(gridcolor='#2A2A2A', autorange="reversed"),
        margin=dict(l=120, r=60, t=40, b=40),
        showlegend=False
    )
    
    return fig


def create_radar_chart(signal: RegimeSignal) -> go.Figure:
    """Create radar chart of regime components"""
    
    categories = ['Momentum', 'Trend', 'Volatility', 'Risk', 'Signal']
    
    # Normalize values to 0-100 scale
    momentum_score = 50 + signal.composite_oscillator / 2  # -100 to 100 -> 0 to 100
    trend_score = 50 if signal.trend_regime == TrendRegime.SIDEWAYS else (
        75 if signal.trend_regime in [TrendRegime.UPTREND, TrendRegime.STRONG_UPTREND] else 25
    )
    vol_score = 100 - signal.volatility_percentile
    risk_score = 100 - signal.risk_score
    signal_strength = signal.signal_strength * 100
    
    values = [momentum_score, trend_score, vol_score, risk_score, signal_strength]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(255, 195, 0, 0.2)',
        line=dict(color='#FFC300', width=2),
        name='Current State'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#2A2A2A',
                linecolor='#2A2A2A'
            ),
            angularaxis=dict(gridcolor='#2A2A2A', linecolor='#2A2A2A'),
            bgcolor='#1A1A1A'
        ),
        paper_bgcolor='#0F0F0F',
        font=dict(family="Inter", color='#EAEAEA'),
        height=400,
        margin=dict(l=80, r=80, t=40, b=40),
        showlegend=False
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render sidebar with mode selection and parameters"""
    
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: var(--primary-color); margin: 0; font-size: 1.75rem;'>🔮 AVASTHA</h2>
            <p style='color: var(--text-muted); font-size: 0.8rem; margin-top: 0.25rem;'>आवस्था • Market Regime</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Mode Selection
        st.markdown('<div class="sidebar-title">🎯 Analysis Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Select Mode",
            ["📈 Individual Scrip", "📊 Index Universe"],
            label_visibility="collapsed",
            help="Individual: Analyze single symbol | Index: Screen entire universe"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Mode-specific options
        symbol = None
        universe = None
        index_name = None
        
        if "Individual" in mode:
            st.markdown('<div class="sidebar-title">🔍 Symbol</div>', unsafe_allow_html=True)
            symbol = st.text_input(
                "Enter Symbol",
                value="NIFTYIETF.NS",
                placeholder="e.g., RELIANCE.NS, ^NSEI",
                label_visibility="collapsed",
                help="Enter NSE symbol with .NS suffix or Yahoo Finance ticker"
            )
            
            # Quick symbol suggestions
            st.markdown('<div class="sidebar-title">📌 Quick Select</div>', unsafe_allow_html=True)
            quick_symbols = {
                "NIFTY 50 ETF": "NIFTYIETF.NS",
                "NIFTY 50 Index": "^NSEI",
                "SENSEX": "^BSESN",
                "RELIANCE": "RELIANCE.NS",
                "TCS": "TCS.NS",
                "HDFCBANK": "HDFCBANK.NS",
                "GOLD ETF": "GOLDIETF.NS",
                "BANK NIFTY": "^NSEBANK"
            }
            
            cols = st.columns(2)
            for i, (name, sym) in enumerate(quick_symbols.items()):
                with cols[i % 2]:
                    if st.button(name, key=f"quick_{sym}", use_container_width=True):
                        st.session_state['selected_symbol'] = sym
                        st.rerun()
            
            # Check for quick select
            if 'selected_symbol' in st.session_state:
                symbol = st.session_state['selected_symbol']
        
        else:  # Index Universe mode
            st.markdown('<div class="sidebar-title">🎯 Universe Selection</div>', unsafe_allow_html=True)
            universe = st.selectbox(
                "Select Universe",
                UNIVERSE_OPTIONS,
                help="ETF Universe: 30 curated ETFs | F&O: Futures & Options stocks | Index: NSE index constituents"
            )
            
            if universe == "Index Constituents":
                index_name = st.selectbox(
                    "Select Index",
                    INDEX_LIST,
                    index=INDEX_LIST.index("NIFTY 50"),
                    help="Select NSE index for constituent analysis"
                )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Parameters
        st.markdown('<div class="sidebar-title">⚙️ Parameters</div>', unsafe_allow_html=True)
        
        with st.expander("Detection Settings", expanded=False):
            lookback = st.slider("Lookback Period", 100, 500, 252, help="Days of historical data")
            volatility_window = st.slider("Volatility Window", 10, 50, 20)
            momentum_window = st.slider("Momentum Window", 5, 30, 14)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Info box
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Multi-Model Synthesis<br>
                <strong>Regimes:</strong> 10 Classifications
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return mode, symbol, universe, index_name


# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL SCRIP MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_individual_mode(symbol: str):
    """Run analysis for individual symbol"""
    
    # Header
    st.markdown(f"""
    <div class='premium-header'>
        <h1>🔮 AVASTHA</h1>
        <p class='tagline'>Multi-Model Market Regime Detection • {symbol}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner(f"Fetching data for {symbol}..."):
        prices = fetch_price_data(symbol, period="1y")
    
    if prices is None or len(prices) < 100:
        st.error(f"Insufficient data for {symbol}. Please check the symbol and try again.")
        return
    
    # Run regime detection
    engine = MarketRegimeEngine()
    signal = engine.detect_regime(prices)
    
    # Get regime color
    regime_color = REGIME_COLORS.get(signal.primary_regime, '#888888')
    regime_emoji = REGIME_EMOJIS.get(signal.primary_regime, '📊')
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid {regime_color};'>
            <h4>Current Regime</h4>
            <h2 style='color: {regime_color}; font-size: 1.5rem;'>{regime_emoji} {signal.primary_regime.value}</h2>
            <div class='sub-metric'>Confidence: {signal.primary_confidence*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        osc_color = '#10b981' if signal.composite_oscillator > 0 else '#ef4444'
        st.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid {osc_color};'>
            <h4>Composite Oscillator</h4>
            <h2 style='color: {osc_color};'>{signal.composite_oscillator:+.1f}</h2>
            <div class='sub-metric'>Range: -100 to +100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        exp_color = '#10b981' if signal.recommended_exposure >= 0.7 else ('#f59e0b' if signal.recommended_exposure >= 0.4 else '#ef4444')
        st.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid {exp_color};'>
            <h4>Recommended Exposure</h4>
            <h2 style='color: {exp_color};'>{signal.recommended_exposure*100:.0f}%</h2>
            <div class='sub-metric'>Max: 150%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        strength_color = '#FFC300' if signal.signal_strength >= 0.5 else '#888888'
        st.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid {strength_color};'>
            <h4>Signal Strength</h4>
            <h2 style='color: {strength_color};'>{signal.signal_strength*100:.0f}%</h2>
            <div class='sub-metric'>Conviction Level</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sub-regime metrics
    st.markdown("### Sub-Regime Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        vol_color = '#10b981' if signal.volatility_regime == VolatilityRegime.COMPRESSED else (
            '#ef4444' if signal.volatility_regime == VolatilityRegime.EXTREME else '#f59e0b'
        )
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Volatility Regime</h4>
            <h2 style='color: {vol_color}; font-size: 1.25rem;'>{signal.volatility_regime.value}</h2>
            <div class='sub-metric'>Percentile: {signal.volatility_percentile:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        mom_color = '#10b981' if 'BULLISH' in signal.momentum_regime.value else (
            '#ef4444' if 'BEARISH' in signal.momentum_regime.value else '#888888'
        )
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Momentum Regime</h4>
            <h2 style='color: {mom_color}; font-size: 1.25rem;'>{signal.momentum_regime.value}</h2>
            <div class='sub-metric'>Directional Bias</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        trend_color = '#10b981' if 'UP' in signal.trend_regime.value else (
            '#ef4444' if 'DOWN' in signal.trend_regime.value else '#888888'
        )
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Trend Regime</h4>
            <h2 style='color: {trend_color}; font-size: 1.25rem;'>{signal.trend_regime.value}</h2>
            <div class='sub-metric'>Structural State</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        risk_color = '#10b981' if signal.risk_score < 40 else ('#ef4444' if signal.risk_score > 70 else '#f59e0b')
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Risk Score</h4>
            <h2 style='color: {risk_color};'>{signal.risk_score:.0f}</h2>
            <div class='sub-metric'>0 = Low, 100 = High</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Probabilities", "🎯 Radar", "📋 Details"])
    
    with tab1:
        chart = create_price_chart(prices, signal, symbol)
        st.plotly_chart(chart, use_container_width=True)
    
    with tab2:
        prob_chart = create_regime_probability_chart(signal)
        st.plotly_chart(prob_chart, use_container_width=True)
    
    with tab3:
        radar = create_radar_chart(signal)
        st.plotly_chart(radar, use_container_width=True)
    
    with tab4:
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Regime Classification")
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Primary Regime | {signal.primary_regime.value} |
            | Confidence | {signal.primary_confidence*100:.1f}% |
            | Composite Oscillator | {signal.composite_oscillator:+.1f} |
            | Recommended Exposure | {signal.recommended_exposure*100:.0f}% |
            | Signal Strength | {signal.signal_strength*100:.0f}% |
            """)
        
        with col2:
            st.markdown("#### Sub-Regime States")
            st.markdown(f"""
            | Component | State | Score |
            |-----------|-------|-------|
            | Volatility | {signal.volatility_regime.value} | {signal.volatility_percentile:.0f}th pct |
            | Momentum | {signal.momentum_regime.value} | - |
            | Trend | {signal.trend_regime.value} | - |
            | Risk | - | {signal.risk_score:.1f} |
            """)
        
        st.markdown("#### Regime Probabilities")
        prob_df = pd.DataFrame([
            {"Regime": k, "Probability": f"{v*100:.1f}%"}
            for k, v in sorted(signal.regime_probabilities.items(), key=lambda x: -x[1])
        ])
        st.dataframe(prob_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX UNIVERSE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_index_mode(universe: str, index_name: str = None):
    """Run screening for entire universe"""
    
    # Get universe title
    if universe == "ETF Universe":
        universe_title = "ETF Universe (30 Curated ETFs)"
    elif universe == "F&O Stocks":
        universe_title = "F&O Stocks"
    else:
        universe_title = index_name if index_name else "Index"
    
    # Header
    st.markdown(f"""
    <div class='premium-header'>
        <h1>🔮 AVASTHA</h1>
        <p class='tagline'>Universe Regime Screening • {universe_title}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get symbols
    symbols = get_universe_symbols(universe, index_name)
    
    if not symbols:
        st.error("Could not fetch symbols for the selected universe. Please try again.")
        return
    
    st.markdown(f"""
    <div class='info-box'>
        <h4>📊 {universe_title}</h4>
        <p style='color: var(--text-muted); font-size: 0.9rem;'>
            Analyzing {len(symbols)} symbols for regime classification.
            This may take a few minutes depending on data availability.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run button
    if st.button("🚀 Run Regime Screening", use_container_width=True):
        run_universe_screening(symbols, universe)


def run_universe_screening(symbols: list, universe: str):
    """Execute screening for all symbols in universe"""
    
    results = []
    engine = MarketRegimeEngine()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(symbols)
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(symbols):
        status_text.markdown(f"**⏳ Analyzing {symbol}... ({i+1}/{total})**")
        
        try:
            prices = fetch_price_data(symbol, period="1y")
            
            if prices is not None and len(prices) >= 100:
                signal = engine.detect_regime(prices)
                
                # Get display name
                display_name = ETF_DISPLAY_NAMES.get(symbol, symbol.replace('.NS', ''))
                
                results.append({
                    'Symbol': symbol,
                    'Name': display_name,
                    'Regime': signal.primary_regime.value,
                    'Confidence': signal.primary_confidence,
                    'Oscillator': signal.composite_oscillator,
                    'Exposure': signal.recommended_exposure,
                    'Volatility': signal.volatility_regime.value,
                    'Momentum': signal.momentum_regime.value,
                    'Trend': signal.trend_regime.value,
                    'Risk': signal.risk_score,
                    'Signal': signal.signal_strength,
                    'Price': prices.iloc[-1],
                    'Change': ((prices.iloc[-1] / prices.iloc[-2]) - 1) * 100 if len(prices) >= 2 else 0
                })
                successful += 1
            else:
                failed += 1
        
        except Exception as e:
            failed += 1
        
        progress_bar.progress((i + 1) / total)
        time.sleep(0.1)  # Small delay to avoid rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("No data could be retrieved. Please check your internet connection.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    st.markdown("### 📊 Universe Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Count regimes
    bullish_regimes = ['EARLY_BULL', 'BULL_TREND', 'BULL_EUPHORIA', 'ACCUMULATION']
    bearish_regimes = ['CRISIS', 'BEAR_ACCELERATION', 'BEAR_DECELERATION', 'DISTRIBUTION']
    
    bullish_count = len(df[df['Regime'].isin(bullish_regimes)])
    bearish_count = len(df[df['Regime'].isin(bearish_regimes)])
    neutral_count = len(df) - bullish_count - bearish_count
    
    avg_oscillator = df['Oscillator'].mean()
    avg_exposure = df['Exposure'].mean()
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Symbols Analyzed</h4>
            <h2 style='color: var(--primary-color);'>{successful}</h2>
            <div class='sub-metric'>{failed} failed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Bullish Regimes</h4>
            <h2 style='color: #10b981;'>{bullish_count}</h2>
            <div class='sub-metric'>{bullish_count/len(df)*100:.0f}% of universe</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Bearish Regimes</h4>
            <h2 style='color: #ef4444;'>{bearish_count}</h2>
            <div class='sub-metric'>{bearish_count/len(df)*100:.0f}% of universe</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        osc_color = '#10b981' if avg_oscillator > 0 else '#ef4444'
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Avg Oscillator</h4>
            <h2 style='color: {osc_color};'>{avg_oscillator:+.1f}</h2>
            <div class='sub-metric'>Universe bias</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        exp_color = '#10b981' if avg_exposure >= 0.7 else ('#f59e0b' if avg_exposure >= 0.4 else '#ef4444')
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Avg Exposure</h4>
            <h2 style='color: {exp_color};'>{avg_exposure*100:.0f}%</h2>
            <div class='sub-metric'>Recommended</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Full Results", "🐂 Bullish", "🐻 Bearish"])
    
    with tab1:
        # Sortable table
        st.markdown("#### All Symbols")
        
        sort_col = st.selectbox("Sort by", ["Oscillator", "Confidence", "Exposure", "Risk", "Change"], key="sort_all")
        sort_asc = st.checkbox("Ascending", value=False, key="asc_all")
        
        display_df = df.sort_values(sort_col, ascending=sort_asc).copy()
        
        # Format for display
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x*100:.0f}%")
        display_df['Oscillator'] = display_df['Oscillator'].apply(lambda x: f"{x:+.1f}")
        display_df['Exposure'] = display_df['Exposure'].apply(lambda x: f"{x*100:.0f}%")
        display_df['Signal'] = display_df['Signal'].apply(lambda x: f"{x*100:.0f}%")
        display_df['Risk'] = display_df['Risk'].apply(lambda x: f"{x:.0f}")
        display_df['Price'] = display_df['Price'].apply(lambda x: f"₹{x:,.2f}")
        display_df['Change'] = display_df['Change'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(
            display_df[['Symbol', 'Name', 'Regime', 'Confidence', 'Oscillator', 'Exposure', 'Volatility', 'Momentum', 'Trend', 'Price', 'Change']],
            hide_index=True,
            use_container_width=True
        )
    
    with tab2:
        bullish_df = df[df['Regime'].isin(bullish_regimes)].copy()
        if len(bullish_df) > 0:
            bullish_df = bullish_df.sort_values('Oscillator', ascending=False)
            bullish_df['Confidence'] = bullish_df['Confidence'].apply(lambda x: f"{x*100:.0f}%")
            bullish_df['Oscillator'] = bullish_df['Oscillator'].apply(lambda x: f"{x:+.1f}")
            bullish_df['Exposure'] = bullish_df['Exposure'].apply(lambda x: f"{x*100:.0f}%")
            bullish_df['Price'] = bullish_df['Price'].apply(lambda x: f"₹{x:,.2f}")
            bullish_df['Change'] = bullish_df['Change'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(
                bullish_df[['Symbol', 'Name', 'Regime', 'Confidence', 'Oscillator', 'Exposure', 'Price', 'Change']],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No symbols in bullish regimes currently.")
    
    with tab3:
        bearish_df = df[df['Regime'].isin(bearish_regimes)].copy()
        if len(bearish_df) > 0:
            bearish_df = bearish_df.sort_values('Oscillator', ascending=True)
            bearish_df['Confidence'] = bearish_df['Confidence'].apply(lambda x: f"{x*100:.0f}%")
            bearish_df['Oscillator'] = bearish_df['Oscillator'].apply(lambda x: f"{x:+.1f}")
            bearish_df['Exposure'] = bearish_df['Exposure'].apply(lambda x: f"{x*100:.0f}%")
            bearish_df['Price'] = bearish_df['Price'].apply(lambda x: f"₹{x:,.2f}")
            bearish_df['Change'] = bearish_df['Change'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(
                bearish_df[['Symbol', 'Name', 'Regime', 'Confidence', 'Oscillator', 'Exposure', 'Price', 'Change']],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No symbols in bearish regimes currently.")
    
    # Regime distribution chart
    st.markdown("### 📊 Regime Distribution")
    
    regime_counts = df['Regime'].value_counts()
    
    fig = go.Figure()
    
    colors = [REGIME_COLORS.get(RegimeType(r), '#888888') for r in regime_counts.index]
    
    fig.add_trace(go.Bar(
        x=regime_counts.index,
        y=regime_counts.values,
        marker_color=colors,
        text=regime_counts.values,
        textposition='outside',
        textfont=dict(color='#EAEAEA')
    ))
    
    fig.update_layout(
        height=400,
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#1A1A1A',
        font=dict(family="Inter", color='#EAEAEA'),
        xaxis=dict(title="Regime", gridcolor='#2A2A2A'),
        yaxis=dict(title="Count", gridcolor='#2A2A2A'),
        margin=dict(l=60, r=40, t=40, b=80),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Reset formatting for CSV
    csv_df = pd.DataFrame(results)
    csv = csv_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"avastha_{universe.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    
    # Render sidebar and get selections
    mode, symbol, universe, index_name = render_sidebar()
    
    # Route to appropriate mode
    if "Individual" in mode:
        if symbol:
            run_individual_mode(symbol)
        else:
            st.info("Please enter a symbol in the sidebar to begin analysis.")
    else:
        run_index_mode(universe, index_name)


if __name__ == "__main__":
    main()
