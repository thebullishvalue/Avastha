"""
AVASTHA (आवस्था) - Market Regime Detection System
==================================================
Institutional-grade multi-model regime detection dashboard.

Part of the Quantitative Analysis Suite alongside Pragyam and UMA.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import time
import requests
from io import StringIO
import json

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

VERSION = "v2.0.0"

# ETF Universe with Categories (30 symbols from Pragyam)
ETF_UNIVERSE_CATEGORIZED = {
    "Broad Market": {
        "NIFTYIETF.NS": "NIFTY 50",
        "MON100.NS": "NIFTY 100", 
        "MONIFTY500.NS": "NIFTY 500",
        "MASPTOP50.NS": "Top 50",
    },
    "Market Cap": {
        "MIDCAPIETF.NS": "Midcap",
        "MOSMALL250.NS": "Smallcap 250",
    },
    "Banking & Finance": {
        "FINIETF.NS": "Financials",
        "PVTBANIETF.NS": "Private Banks",
        "PSUBNKIETF.NS": "PSU Banks",
        "ECAPINSURE.NS": "Insurance",
    },
    "Technology & Services": {
        "ITIETF.NS": "IT",
        "MNC.NS": "MNC",
    },
    "Consumer": {
        "FMCGIETF.NS": "FMCG",
        "CONSUMIETF.NS": "Consumption",
        "AUTOIETF.NS": "Auto",
        "EVINDIA.NS": "EV India",
    },
    "Healthcare": {
        "HEALTHIETF.NS": "Healthcare",
    },
    "Industrial & Infra": {
        "INFRAIETF.NS": "Infrastructure",
        "CPSETF.NS": "CPSE",
        "MAKEINDIA.NS": "Make in India",
        "MODEFENCE.NS": "Defence",
        "MOREALTY.NS": "Realty",
        "TNIDETF.NS": "Tamil Nadu Infra",
        "GROWWPOWER.NS": "Power & Energy",
    },
    "Materials & Resources": {
        "METALIETF.NS": "Metal",
        "OILIETF.NS": "Oil & Gas",
        "CHEMICAL.NS": "Chemicals",
    },
    "Commodities": {
        "GOLDIETF.NS": "Gold",
        "SILVERIETF.NS": "Silver",
        "COMMOIETF.NS": "Commodities",
    },
}

# Flatten for easy access
ETF_UNIVERSE = []
ETF_DISPLAY_NAMES = {}
ETF_CATEGORIES = {}
for category, symbols in ETF_UNIVERSE_CATEGORIZED.items():
    for symbol, name in symbols.items():
        ETF_UNIVERSE.append(symbol)
        ETF_DISPLAY_NAMES[symbol] = name
        ETF_CATEGORIES[symbol] = category

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

FNO_URL = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
UNIVERSE_OPTIONS = ["ETF Universe", "F&O Stocks", "Index Constituents"]

# Regime Configuration
REGIME_COLORS = {
    RegimeType.CRISIS: "#ef4444",
    RegimeType.BEAR_ACCELERATION: "#f97316",
    RegimeType.BEAR_DECELERATION: "#fb923c",
    RegimeType.ACCUMULATION: "#06b6d4",
    RegimeType.EARLY_BULL: "#10b981",
    RegimeType.BULL_TREND: "#22c55e",
    RegimeType.BULL_EUPHORIA: "#f59e0b",
    RegimeType.DISTRIBUTION: "#a855f7",
    RegimeType.CHOP: "#888888",
    RegimeType.TRANSITION: "#FFC300",
}

REGIME_EMOJIS = {
    RegimeType.CRISIS: "🔴", RegimeType.BEAR_ACCELERATION: "📉",
    RegimeType.BEAR_DECELERATION: "🔻", RegimeType.ACCUMULATION: "📦",
    RegimeType.EARLY_BULL: "🌱", RegimeType.BULL_TREND: "🐂",
    RegimeType.BULL_EUPHORIA: "🎪", RegimeType.DISTRIBUTION: "📤",
    RegimeType.CHOP: "🌀", RegimeType.TRANSITION: "🔄",
}

# Regime risk weights for portfolio analysis
REGIME_RISK_WEIGHTS = {
    "CRISIS": 1.0, "BEAR_ACCELERATION": 0.85, "BEAR_DECELERATION": 0.6,
    "DISTRIBUTION": 0.7, "CHOP": 0.5, "TRANSITION": 0.5,
    "ACCUMULATION": 0.3, "EARLY_BULL": 0.2, "BULL_TREND": 0.15, "BULL_EUPHORIA": 0.4
}

BULLISH_REGIMES = ['EARLY_BULL', 'BULL_TREND', 'BULL_EUPHORIA', 'ACCUMULATION']
BEARISH_REGIMES = ['CRISIS', 'BEAR_ACCELERATION', 'BEAR_DECELERATION', 'DISTRIBUTION']
NEUTRAL_REGIMES = ['CHOP', 'TRANSITION']

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG AND STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AVASTHA | Market Regime Detection System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    .block-container { padding-top: 1rem; max-width: 1600px; }
    
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
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    .premium-header h1 { margin: 0; font-size: 2.50rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 1rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    
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
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.8rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: var(--text-primary); font-size: 2rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    
    .info-box {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
    }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    
    .sidebar-title { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem; }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1rem 0; }
    
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
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    
    .stMarkdown table { width: 100%; border-collapse: collapse; background: var(--bg-card); border-radius: 16px; overflow: hidden; border: 1px solid var(--border-color); }
    .stMarkdown table th, .stMarkdown table td { text-align: left !important; padding: 12px 10px; border-bottom: 1px solid var(--border-color); }
    .stMarkdown table th { background-color: var(--bg-elevated); font-size: 0.9rem; letter-spacing: 0.5px; }
    .stMarkdown table tr:last-child td { border-bottom: none; }
    .stMarkdown table tr:hover { background-color: var(--bg-elevated); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; padding: 0.5rem 1rem; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); }
    
    .stPlotlyChart, .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); }
    
    .insight-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary-color);
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .insight-card h5 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600; }
    .insight-card p { color: var(--text-secondary); margin: 0; font-size: 0.85rem; line-height: 1.5; }
    
    .filter-section {
        background: var(--bg-card);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }
    
    .heatmap-cell {
        padding: 0.5rem;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(symbol: str, period: str = "1y") -> pd.Series:
    """Fetch price data for a symbol"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        return df['Close']
    except Exception as e:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fno_stocks() -> list:
    """Fetch F&O stock list from NSE"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(FNO_URL, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if 'SYMBOL' in df.columns:
                symbols = df['SYMBOL'].dropna().unique().tolist()
                symbols = [f"{s.strip()}.NS" for s in symbols if not s.startswith('NIFTY') and s.strip()]
                return symbols[:200]
    except:
        pass
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_constituents(index_name: str) -> list:
    """Fetch index constituents from NSE"""
    if index_name not in INDEX_URLS:
        return []
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(INDEX_URLS[index_name], headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].dropna().unique().tolist()
                return [f"{s.strip()}.NS" for s in symbols if s.strip()]
    except:
        pass
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
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_universe_metrics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive universe-level metrics"""
    
    bullish_count = len(df[df['Regime'].isin(BULLISH_REGIMES)])
    bearish_count = len(df[df['Regime'].isin(BEARISH_REGIMES)])
    neutral_count = len(df[df['Regime'].isin(NEUTRAL_REGIMES)])
    total = len(df)
    
    # Weighted risk score
    df['RiskWeight'] = df['Regime'].map(REGIME_RISK_WEIGHTS)
    weighted_risk = (df['RiskWeight'] * df['Confidence']).sum() / df['Confidence'].sum() if df['Confidence'].sum() > 0 else 0.5
    
    # Breadth metrics
    bullish_pct = bullish_count / total if total > 0 else 0
    bearish_pct = bearish_count / total if total > 0 else 0
    
    # Oscillator distribution
    osc_mean = df['Oscillator'].mean()
    osc_std = df['Oscillator'].std()
    osc_skew = df['Oscillator'].skew() if len(df) > 2 else 0
    
    # Exposure metrics
    avg_exposure = df['Exposure'].mean()
    exposure_std = df['Exposure'].std()
    
    # Momentum breadth
    strong_bullish = len(df[df['Momentum'].isin(['STRONG_BULLISH', 'BULLISH'])])
    strong_bearish = len(df[df['Momentum'].isin(['STRONG_BEARISH', 'BEARISH'])])
    momentum_breadth = (strong_bullish - strong_bearish) / total if total > 0 else 0
    
    # Trend breadth
    uptrend_count = len(df[df['Trend'].isin(['UPTREND', 'STRONG_UPTREND'])])
    downtrend_count = len(df[df['Trend'].isin(['DOWNTREND', 'STRONG_DOWNTREND'])])
    trend_breadth = (uptrend_count - downtrend_count) / total if total > 0 else 0
    
    # Risk concentration (Herfindahl-like)
    regime_counts = df['Regime'].value_counts(normalize=True)
    concentration = (regime_counts ** 2).sum()
    
    return {
        'total': total,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': neutral_count,
        'bullish_pct': bullish_pct,
        'bearish_pct': bearish_pct,
        'weighted_risk': weighted_risk,
        'osc_mean': osc_mean,
        'osc_std': osc_std,
        'osc_skew': osc_skew,
        'avg_exposure': avg_exposure,
        'exposure_std': exposure_std,
        'momentum_breadth': momentum_breadth,
        'trend_breadth': trend_breadth,
        'concentration': concentration,
        'regime_distribution': regime_counts.to_dict()
    }


def generate_insights(df: pd.DataFrame, metrics: dict) -> list:
    """Generate actionable insights from analysis"""
    
    insights = []
    
    # Market stance insight
    if metrics['bullish_pct'] > 0.6:
        insights.append({
            'type': 'bullish',
            'title': 'Strong Bullish Breadth',
            'text': f"{metrics['bullish_pct']*100:.0f}% of universe in bullish regimes. Consider increasing equity exposure with focus on momentum strategies.",
            'priority': 1
        })
    elif metrics['bearish_pct'] > 0.5:
        insights.append({
            'type': 'bearish',
            'title': 'Elevated Bearish Regime Concentration',
            'text': f"{metrics['bearish_pct']*100:.0f}% of universe in bearish regimes. Defensive positioning and hedging recommended.",
            'priority': 1
        })
    elif metrics['neutral_count'] / metrics['total'] > 0.4:
        insights.append({
            'type': 'neutral',
            'title': 'High Uncertainty Environment',
            'text': f"Over 40% of universe in CHOP or TRANSITION. Reduce position sizing and favor mean-reversion strategies.",
            'priority': 1
        })
    
    # Risk concentration
    if metrics['concentration'] > 0.3:
        dominant_regime = max(metrics['regime_distribution'], key=metrics['regime_distribution'].get)
        insights.append({
            'type': 'warning',
            'title': 'Regime Concentration Risk',
            'text': f"High concentration in {dominant_regime} regime ({metrics['regime_distribution'][dominant_regime]*100:.0f}%). Monitor for regime transition signals.",
            'priority': 2
        })
    
    # Momentum-Trend divergence
    if abs(metrics['momentum_breadth'] - metrics['trend_breadth']) > 0.3:
        if metrics['momentum_breadth'] > metrics['trend_breadth']:
            insights.append({
                'type': 'info',
                'title': 'Momentum Leading Trend',
                'text': 'Momentum breadth exceeds trend breadth - potential trend acceleration ahead. Watch for breakout confirmations.',
                'priority': 2
            })
        else:
            insights.append({
                'type': 'info',
                'title': 'Trend Exhaustion Signal',
                'text': 'Trend breadth exceeds momentum breadth - potential trend deceleration. Consider taking profits on trend-following positions.',
                'priority': 2
            })
    
    # Oscillator extremes
    if metrics['osc_mean'] > 30:
        insights.append({
            'type': 'warning',
            'title': 'Universe Overbought',
            'text': f"Mean oscillator at {metrics['osc_mean']:.1f}. Elevated mean-reversion risk across universe.",
            'priority': 2
        })
    elif metrics['osc_mean'] < -30:
        insights.append({
            'type': 'opportunity',
            'title': 'Universe Oversold',
            'text': f"Mean oscillator at {metrics['osc_mean']:.1f}. Potential contrarian opportunity if fundamentals support.",
            'priority': 2
        })
    
    # Volatility warning
    high_vol_count = len(df[df['Volatility'].isin(['ELEVATED', 'EXTREME'])])
    if high_vol_count / metrics['total'] > 0.4:
        insights.append({
            'type': 'warning',
            'title': 'Elevated Volatility Environment',
            'text': f"{high_vol_count} symbols ({high_vol_count/metrics['total']*100:.0f}%) showing elevated/extreme volatility. Reduce position sizes.",
            'priority': 1
        })
    
    # Sort by priority
    insights.sort(key=lambda x: x['priority'])
    
    return insights


def get_sector_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate analysis by sector/category"""
    
    if 'Category' not in df.columns:
        return None
    
    sector_stats = df.groupby('Category').agg({
        'Oscillator': ['mean', 'std', 'count'],
        'Exposure': 'mean',
        'Confidence': 'mean',
        'Risk': 'mean'
    }).round(2)
    
    sector_stats.columns = ['Osc_Mean', 'Osc_Std', 'Count', 'Avg_Exposure', 'Avg_Confidence', 'Avg_Risk']
    sector_stats = sector_stats.reset_index()
    
    # Calculate bullish percentage per sector
    bullish_pct = df[df['Regime'].isin(BULLISH_REGIMES)].groupby('Category').size() / df.groupby('Category').size()
    sector_stats['Bullish_Pct'] = sector_stats['Category'].map(bullish_pct).fillna(0)
    
    return sector_stats.sort_values('Osc_Mean', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_regime_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create regime distribution donut chart"""
    
    regime_counts = df['Regime'].value_counts()
    colors = [REGIME_COLORS.get(RegimeType(r), '#888888') for r in regime_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=regime_counts.index,
        values=regime_counts.values,
        hole=0.6,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(color='#EAEAEA', size=11)
    )])
    
    fig.update_layout(
        height=350,
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#1A1A1A',
        font=dict(family="Inter", color='#EAEAEA'),
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        annotations=[dict(text=f'{len(df)}<br>Symbols', x=0.5, y=0.5, font_size=16, font_color='#EAEAEA', showarrow=False)]
    )
    
    return fig


def create_oscillator_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create oscillator distribution histogram"""
    
    fig = go.Figure()
    
    # Color by regime type
    bullish_df = df[df['Regime'].isin(BULLISH_REGIMES)]
    bearish_df = df[df['Regime'].isin(BEARISH_REGIMES)]
    neutral_df = df[df['Regime'].isin(NEUTRAL_REGIMES)]
    
    fig.add_trace(go.Histogram(x=bullish_df['Oscillator'], name='Bullish', marker_color='#10b981', opacity=0.7, nbinsx=20))
    fig.add_trace(go.Histogram(x=bearish_df['Oscillator'], name='Bearish', marker_color='#ef4444', opacity=0.7, nbinsx=20))
    fig.add_trace(go.Histogram(x=neutral_df['Oscillator'], name='Neutral', marker_color='#888888', opacity=0.7, nbinsx=20))
    
    # Add mean line
    mean_osc = df['Oscillator'].mean()
    fig.add_vline(x=mean_osc, line_dash="dash", line_color="#FFC300", annotation_text=f"Mean: {mean_osc:.1f}")
    fig.add_vline(x=0, line_dash="dot", line_color="#888888")
    
    fig.update_layout(
        height=300,
        barmode='overlay',
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#1A1A1A',
        font=dict(family="Inter", color='#EAEAEA'),
        xaxis=dict(title="Composite Oscillator", gridcolor='#2A2A2A', range=[-100, 100]),
        yaxis=dict(title="Count", gridcolor='#2A2A2A'),
        margin=dict(l=60, r=40, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_sector_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create sector regime heatmap"""
    
    if 'Category' not in df.columns:
        return None
    
    # Create pivot table
    pivot = df.pivot_table(
        values='Oscillator',
        index='Category',
        columns='Regime',
        aggfunc='count',
        fill_value=0
    )
    
    # Reorder columns
    regime_order = ['CRISIS', 'BEAR_ACCELERATION', 'BEAR_DECELERATION', 'DISTRIBUTION',
                    'CHOP', 'TRANSITION', 'ACCUMULATION', 'EARLY_BULL', 'BULL_TREND', 'BULL_EUPHORIA']
    available_regimes = [r for r in regime_order if r in pivot.columns]
    pivot = pivot[available_regimes]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[
            [0, '#1A1A1A'],
            [0.25, '#2A2A2A'],
            [0.5, '#FFC300'],
            [1, '#FFC300']
        ],
        showscale=True,
        text=pivot.values,
        texttemplate="%{text}",
        textfont=dict(color='#EAEAEA', size=11)
    ))
    
    fig.update_layout(
        height=400,
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#1A1A1A',
        font=dict(family="Inter", color='#EAEAEA'),
        xaxis=dict(title="", tickangle=45),
        yaxis=dict(title=""),
        margin=dict(l=120, r=40, t=30, b=100)
    )
    
    return fig


def create_risk_gauge(weighted_risk: float) -> go.Figure:
    """Create risk gauge visualization"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=weighted_risk * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Universe Risk Score", 'font': {'size': 14, 'color': '#EAEAEA'}},
        number={'suffix': '%', 'font': {'size': 36, 'color': '#EAEAEA'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#EAEAEA'},
            'bar': {'color': '#FFC300'},
            'bgcolor': '#2A2A2A',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': '#10b981'},
                {'range': [30, 60], 'color': '#f59e0b'},
                {'range': [60, 100], 'color': '#ef4444'}
            ],
            'threshold': {
                'line': {'color': '#EAEAEA', 'width': 2},
                'thickness': 0.75,
                'value': weighted_risk * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        paper_bgcolor='#0F0F0F',
        font=dict(family="Inter", color='#EAEAEA'),
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


def create_breadth_chart(metrics: dict) -> go.Figure:
    """Create market breadth bar chart"""
    
    categories = ['Regime<br>Breadth', 'Momentum<br>Breadth', 'Trend<br>Breadth']
    
    regime_breadth = metrics['bullish_pct'] - metrics['bearish_pct']
    values = [regime_breadth, metrics['momentum_breadth'], metrics['trend_breadth']]
    
    colors = ['#10b981' if v > 0 else '#ef4444' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=[v * 100 for v in values],
        marker_color=colors,
        text=[f"{v*100:+.0f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#EAEAEA')
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#888888")
    
    fig.update_layout(
        height=300,
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#1A1A1A',
        font=dict(family="Inter", color='#EAEAEA'),
        xaxis=dict(gridcolor='#2A2A2A'),
        yaxis=dict(title="Breadth %", gridcolor='#2A2A2A', range=[-100, 100]),
        margin=dict(l=60, r=40, t=30, b=50),
        showlegend=False
    )
    
    return fig


def create_scatter_analysis(df: pd.DataFrame) -> go.Figure:
    """Create oscillator vs exposure scatter plot"""
    
    fig = go.Figure()
    
    for regime in df['Regime'].unique():
        regime_df = df[df['Regime'] == regime]
        color = REGIME_COLORS.get(RegimeType(regime), '#888888')
        
        fig.add_trace(go.Scatter(
            x=regime_df['Oscillator'],
            y=regime_df['Exposure'] * 100,
            mode='markers',
            name=regime,
            marker=dict(color=color, size=10, opacity=0.7),
            text=regime_df['Name'],
            hovertemplate='<b>%{text}</b><br>Oscillator: %{x:.1f}<br>Exposure: %{y:.0f}%<extra></extra>'
        ))
    
    fig.add_hline(y=70, line_dash="dash", line_color="#10b981", annotation_text="High Exposure")
    fig.add_hline(y=40, line_dash="dash", line_color="#ef4444", annotation_text="Low Exposure")
    fig.add_vline(x=0, line_dash="dot", line_color="#888888")
    
    fig.update_layout(
        height=450,
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#1A1A1A',
        font=dict(family="Inter", color='#EAEAEA'),
        xaxis=dict(title="Composite Oscillator", gridcolor='#2A2A2A', range=[-100, 100]),
        yaxis=dict(title="Recommended Exposure %", gridcolor='#2A2A2A', range=[0, 150]),
        margin=dict(l=60, r=40, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_price_chart(prices: pd.Series, signal: RegimeSignal, symbol: str) -> go.Figure:
    """Create price chart with regime overlay for individual analysis"""
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines', name='Price',
                             line=dict(color='#FFC300', width=2), fill='tozeroy', fillcolor='rgba(255, 195, 0, 0.1)'), row=1, col=1)
    
    if len(prices) >= 20:
        ma20 = prices.rolling(20).mean()
        fig.add_trace(go.Scatter(x=prices.index, y=ma20, mode='lines', name='MA20', line=dict(color='#06b6d4', width=1, dash='dot')), row=1, col=1)
    if len(prices) >= 50:
        ma50 = prices.rolling(50).mean()
        fig.add_trace(go.Scatter(x=prices.index, y=ma50, mode='lines', name='MA50', line=dict(color='#a855f7', width=1, dash='dot')), row=1, col=1)
    
    # Oscillator subplot
    osc_color = '#10b981' if signal.composite_oscillator > 0 else '#ef4444'
    fig.add_trace(go.Bar(x=[prices.index[-1]], y=[signal.composite_oscillator], name='Oscillator', marker_color=osc_color), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#888888", row=2, col=1)
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#1A1A1A',
        font=dict(family="Inter", color='#EAEAEA'),
        margin=dict(l=60, r=40, t=60, b=40),
        hovermode='x unified'
    )
    fig.update_xaxes(gridcolor='#2A2A2A', showgrid=True)
    fig.update_yaxes(gridcolor='#2A2A2A', showgrid=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Oscillator", range=[-100, 100], row=2, col=1)
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render sidebar with mode selection and parameters"""
    
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: var(--primary-color); margin: 0; font-size: 1.75rem;'>🔮 AVASTHA</h2>
            <p style='color: var(--text-muted); font-size: 0.8rem; margin-top: 0.25rem;'>आवस्था • Market Regime</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">🎯 Analysis Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Select Mode", ["📈 Individual Scrip", "📊 Index Universe"], label_visibility="collapsed")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        symbol = None
        universe = None
        index_name = None
        
        if "Individual" in mode:
            st.markdown('<div class="sidebar-title">🔍 Symbol</div>', unsafe_allow_html=True)
            symbol = st.text_input("Enter Symbol", value="NIFTYIETF.NS", placeholder="e.g., RELIANCE.NS", label_visibility="collapsed")
        
        else:
            st.markdown('<div class="sidebar-title">🎯 Universe Selection</div>', unsafe_allow_html=True)
            universe = st.selectbox("Select Universe", UNIVERSE_OPTIONS)
            
            if universe == "Index Constituents":
                index_name = st.selectbox("Select Index", INDEX_LIST, index=INDEX_LIST.index("NIFTY 50"))
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
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
    
    st.markdown(f"""
    <div class='premium-header'>
        <h1>🔮 AVASTHA</h1>
        <p class='tagline'>Multi-Model Market Regime Detection • {symbol}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner(f"Fetching data for {symbol}..."):
        prices = fetch_price_data(symbol, period="1y")
    
    if prices is None or len(prices) < 100:
        st.error(f"Insufficient data for {symbol}. Please check the symbol and try again.")
        return
    
    engine = MarketRegimeEngine()
    signal = engine.detect_regime(prices)
    regime_color = REGIME_COLORS.get(signal.primary_regime, '#888888')
    regime_emoji = REGIME_EMOJIS.get(signal.primary_regime, '📊')
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card' style='border-top: 3px solid {regime_color};'>
            <h4>Current Regime</h4><h2 style='color: {regime_color}; font-size: 1.5rem;'>{regime_emoji} {signal.primary_regime.value}</h2>
            <div class='sub-metric'>Confidence: {signal.primary_confidence*100:.0f}%</div></div>""", unsafe_allow_html=True)
    with col2:
        osc_color = '#10b981' if signal.composite_oscillator > 0 else '#ef4444'
        st.markdown(f"""<div class='metric-card' style='border-top: 3px solid {osc_color};'>
            <h4>Composite Oscillator</h4><h2 style='color: {osc_color};'>{signal.composite_oscillator:+.1f}</h2>
            <div class='sub-metric'>Range: -100 to +100</div></div>""", unsafe_allow_html=True)
    with col3:
        exp_color = '#10b981' if signal.recommended_exposure >= 0.7 else ('#f59e0b' if signal.recommended_exposure >= 0.4 else '#ef4444')
        st.markdown(f"""<div class='metric-card' style='border-top: 3px solid {exp_color};'>
            <h4>Recommended Exposure</h4><h2 style='color: {exp_color};'>{signal.recommended_exposure*100:.0f}%</h2>
            <div class='sub-metric'>Max: 150%</div></div>""", unsafe_allow_html=True)
    with col4:
        strength_color = '#FFC300' if signal.signal_strength >= 0.5 else '#888888'
        st.markdown(f"""<div class='metric-card' style='border-top: 3px solid {strength_color};'>
            <h4>Signal Strength</h4><h2 style='color: {strength_color};'>{signal.signal_strength*100:.0f}%</h2>
            <div class='sub-metric'>Conviction Level</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sub-regimes
    st.markdown("### Sub-Regime Analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        vol_color = '#10b981' if signal.volatility_regime == VolatilityRegime.COMPRESSED else ('#ef4444' if signal.volatility_regime == VolatilityRegime.EXTREME else '#f59e0b')
        st.markdown(f"""<div class='metric-card'><h4>Volatility</h4><h2 style='color: {vol_color}; font-size: 1.25rem;'>{signal.volatility_regime.value}</h2>
            <div class='sub-metric'>Percentile: {signal.volatility_percentile:.0f}%</div></div>""", unsafe_allow_html=True)
    with col2:
        mom_color = '#10b981' if 'BULLISH' in signal.momentum_regime.value else ('#ef4444' if 'BEARISH' in signal.momentum_regime.value else '#888888')
        st.markdown(f"""<div class='metric-card'><h4>Momentum</h4><h2 style='color: {mom_color}; font-size: 1.25rem;'>{signal.momentum_regime.value}</h2>
            <div class='sub-metric'>Directional Bias</div></div>""", unsafe_allow_html=True)
    with col3:
        trend_color = '#10b981' if 'UP' in signal.trend_regime.value else ('#ef4444' if 'DOWN' in signal.trend_regime.value else '#888888')
        st.markdown(f"""<div class='metric-card'><h4>Trend</h4><h2 style='color: {trend_color}; font-size: 1.25rem;'>{signal.trend_regime.value}</h2>
            <div class='sub-metric'>Structural State</div></div>""", unsafe_allow_html=True)
    with col4:
        risk_color = '#10b981' if signal.risk_score < 40 else ('#ef4444' if signal.risk_score > 70 else '#f59e0b')
        st.markdown(f"""<div class='metric-card'><h4>Risk Score</h4><h2 style='color: {risk_color};'>{signal.risk_score:.0f}</h2>
            <div class='sub-metric'>0 = Low, 100 = High</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chart
    chart = create_price_chart(prices, signal, symbol)
    st.plotly_chart(chart, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX UNIVERSE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_index_mode(universe: str, index_name: str = None):
    """Run screening for entire universe"""
    
    universe_title = "ETF Universe" if universe == "ETF Universe" else ("F&O Stocks" if universe == "F&O Stocks" else index_name)
    
    st.markdown(f"""
    <div class='premium-header'>
        <h1>🔮 AVASTHA</h1>
        <p class='tagline'>Universe Regime Screening • {universe_title}</p>
    </div>
    """, unsafe_allow_html=True)
    
    symbols = get_universe_symbols(universe, index_name)
    
    if not symbols:
        st.error("Could not fetch symbols for the selected universe.")
        return
    
    # Check for cached results
    cache_key = f"avastha_results_{universe}_{index_name}"
    
    if cache_key not in st.session_state:
        st.markdown(f"""
        <div class='info-box'>
            <h4>📊 {universe_title}</h4>
            <p style='color: var(--text-muted); font-size: 0.9rem;'>Ready to analyze {len(symbols)} symbols for regime classification.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run Regime Screening", width='stretch'):
            results = run_screening(symbols, universe)
            if results:
                st.session_state[cache_key] = results
                st.rerun()
    else:
        results = st.session_state[cache_key]
        render_analysis_dashboard(results, universe, universe_title)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Re-run Analysis"):
                del st.session_state[cache_key]
                st.rerun()


def run_screening(symbols: list, universe: str) -> list:
    """Execute screening for all symbols"""
    
    results = []
    engine = MarketRegimeEngine()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        status_text.markdown(f"**⏳ Analyzing {symbol}... ({i+1}/{total})**")
        
        try:
            prices = fetch_price_data(symbol, period="1y")
            
            if prices is not None and len(prices) >= 100:
                signal = engine.detect_regime(prices)
                
                display_name = ETF_DISPLAY_NAMES.get(symbol, symbol.replace('.NS', ''))
                category = ETF_CATEGORIES.get(symbol, 'Other')
                
                results.append({
                    'Symbol': symbol,
                    'Name': display_name,
                    'Category': category,
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
        except:
            pass
        
        progress_bar.progress((i + 1) / total)
        time.sleep(0.05)
    
    progress_bar.empty()
    status_text.empty()
    
    return results


def render_analysis_dashboard(results: list, universe: str, universe_title: str):
    """Render comprehensive analysis dashboard"""
    
    df = pd.DataFrame(results)
    
    if df.empty:
        st.error("No data retrieved. Please check your internet connection.")
        return
    
    # Calculate metrics
    metrics = calculate_universe_metrics(df)
    insights = generate_insights(df, metrics)
    
    # Main Dashboard Tabs
    tab_overview, tab_analysis, tab_sectors, tab_screener, tab_drilldown = st.tabs([
        "📊 Overview", "📈 Risk Analysis", "🏭 Sector View", "🔍 Screener", "🎯 Deep Dive"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_overview:
        
        # Key Metrics Row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown(f"""<div class='metric-card'>
                <h4>Universe Size</h4><h2 style='color: var(--primary-color);'>{metrics['total']}</h2>
                <div class='sub-metric'>Symbols Analyzed</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='metric-card'>
                <h4>Bullish</h4><h2 style='color: #10b981;'>{metrics['bullish_count']}</h2>
                <div class='sub-metric'>{metrics['bullish_pct']*100:.0f}% of universe</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class='metric-card'>
                <h4>Bearish</h4><h2 style='color: #ef4444;'>{metrics['bearish_count']}</h2>
                <div class='sub-metric'>{metrics['bearish_pct']*100:.0f}% of universe</div></div>""", unsafe_allow_html=True)
        with col4:
            osc_color = '#10b981' if metrics['osc_mean'] > 0 else '#ef4444'
            st.markdown(f"""<div class='metric-card'>
                <h4>Mean Oscillator</h4><h2 style='color: {osc_color};'>{metrics['osc_mean']:+.1f}</h2>
                <div class='sub-metric'>σ = {metrics['osc_std']:.1f}</div></div>""", unsafe_allow_html=True)
        with col5:
            exp_color = '#10b981' if metrics['avg_exposure'] >= 0.7 else ('#f59e0b' if metrics['avg_exposure'] >= 0.4 else '#ef4444')
            st.markdown(f"""<div class='metric-card'>
                <h4>Avg Exposure</h4><h2 style='color: {exp_color};'>{metrics['avg_exposure']*100:.0f}%</h2>
                <div class='sub-metric'>Recommended</div></div>""", unsafe_allow_html=True)
        with col6:
            risk_color = '#10b981' if metrics['weighted_risk'] < 0.4 else ('#ef4444' if metrics['weighted_risk'] > 0.6 else '#f59e0b')
            st.markdown(f"""<div class='metric-card'>
                <h4>Risk Score</h4><h2 style='color: {risk_color};'>{metrics['weighted_risk']*100:.0f}</h2>
                <div class='sub-metric'>Weighted Avg</div></div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Regime Distribution")
            fig = create_regime_distribution_chart(df)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### Oscillator Distribution")
            fig = create_oscillator_distribution_chart(df)
            st.plotly_chart(fig, width='stretch')
        
        # Insights Panel
        st.markdown("#### 💡 Actionable Insights")
        
        if insights:
            cols = st.columns(min(len(insights), 3))
            for i, insight in enumerate(insights[:3]):
                with cols[i]:
                    icon = "🟢" if insight['type'] == 'bullish' else ("🔴" if insight['type'] == 'bearish' else ("⚠️" if insight['type'] == 'warning' else "💡"))
                    st.markdown(f"""
                    <div class='insight-card'>
                        <h5>{icon} {insight['title']}</h5>
                        <p>{insight['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: RISK ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_analysis:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Universe Risk Gauge")
            fig = create_risk_gauge(metrics['weighted_risk'])
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### Market Breadth")
            fig = create_breadth_chart(metrics)
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("#### Oscillator vs Exposure Analysis")
        fig = create_scatter_analysis(df)
        st.plotly_chart(fig, width='stretch')
        
        # Risk concentration table
        st.markdown("#### Regime Concentration Analysis")
        regime_dist = pd.DataFrame([
            {'Regime': k, 'Count': int(v * metrics['total']), 'Percentage': f"{v*100:.1f}%", 'Risk Weight': REGIME_RISK_WEIGHTS.get(k, 0.5)}
            for k, v in metrics['regime_distribution'].items()
        ]).sort_values('Count', ascending=False)
        
        st.dataframe(regime_dist, hide_index=True, width='stretch')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: SECTOR VIEW
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_sectors:
        
        if 'Category' in df.columns and universe == "ETF Universe":
            st.markdown("#### Sector Regime Heatmap")
            fig = create_sector_heatmap(df)
            if fig:
                st.plotly_chart(fig, width='stretch')
            
            st.markdown("#### Sector Statistics")
            sector_stats = get_sector_analysis(df)
            if sector_stats is not None:
                sector_stats['Osc_Mean'] = sector_stats['Osc_Mean'].apply(lambda x: f"{x:+.1f}")
                sector_stats['Avg_Exposure'] = sector_stats['Avg_Exposure'].apply(lambda x: f"{x*100:.0f}%")
                sector_stats['Bullish_Pct'] = sector_stats['Bullish_Pct'].apply(lambda x: f"{x*100:.0f}%")
                st.dataframe(sector_stats, hide_index=True, width='stretch')
        else:
            st.info("Sector analysis is available for ETF Universe only.")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: SCREENER
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_screener:
        
        st.markdown("#### Advanced Filters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            regime_filter = st.multiselect("Regime", options=df['Regime'].unique().tolist(), default=None, placeholder="All Regimes")
        with col2:
            momentum_filter = st.multiselect("Momentum", options=df['Momentum'].unique().tolist(), default=None, placeholder="All Momentum")
        with col3:
            trend_filter = st.multiselect("Trend", options=df['Trend'].unique().tolist(), default=None, placeholder="All Trends")
        with col4:
            volatility_filter = st.multiselect("Volatility", options=df['Volatility'].unique().tolist(), default=None, placeholder="All Volatility")
        
        col1, col2 = st.columns(2)
        with col1:
            osc_range = st.slider("Oscillator Range", -100, 100, (-100, 100))
        with col2:
            exp_range = st.slider("Exposure Range %", 0, 150, (0, 150))
        
        # Apply filters
        filtered_df = df.copy()
        if regime_filter:
            filtered_df = filtered_df[filtered_df['Regime'].isin(regime_filter)]
        if momentum_filter:
            filtered_df = filtered_df[filtered_df['Momentum'].isin(momentum_filter)]
        if trend_filter:
            filtered_df = filtered_df[filtered_df['Trend'].isin(trend_filter)]
        if volatility_filter:
            filtered_df = filtered_df[filtered_df['Volatility'].isin(volatility_filter)]
        filtered_df = filtered_df[(filtered_df['Oscillator'] >= osc_range[0]) & (filtered_df['Oscillator'] <= osc_range[1])]
        filtered_df = filtered_df[(filtered_df['Exposure'] * 100 >= exp_range[0]) & (filtered_df['Exposure'] * 100 <= exp_range[1])]
        
        st.markdown(f"#### Results ({len(filtered_df)} symbols)")
        
        # Sort options
        col1, col2 = st.columns([1, 3])
        with col1:
            sort_col = st.selectbox("Sort by", ["Oscillator", "Confidence", "Exposure", "Risk", "Change"])
            sort_asc = st.checkbox("Ascending", value=False)
        
        display_df = filtered_df.sort_values(sort_col, ascending=sort_asc).copy()
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x*100:.0f}%")
        display_df['Oscillator'] = display_df['Oscillator'].apply(lambda x: f"{x:+.1f}")
        display_df['Exposure'] = display_df['Exposure'].apply(lambda x: f"{x*100:.0f}%")
        display_df['Signal'] = display_df['Signal'].apply(lambda x: f"{x*100:.0f}%")
        display_df['Risk'] = display_df['Risk'].apply(lambda x: f"{x:.0f}")
        display_df['Price'] = display_df['Price'].apply(lambda x: f"₹{x:,.2f}")
        display_df['Change'] = display_df['Change'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(display_df[['Symbol', 'Name', 'Regime', 'Confidence', 'Oscillator', 'Exposure', 'Volatility', 'Momentum', 'Trend', 'Risk', 'Price', 'Change']], hide_index=True, width='stretch')
        
        # Download
        csv_df = filtered_df.copy()
        csv = csv_df.to_csv(index=False)
        st.download_button("📥 Download Filtered Results (CSV)", data=csv, file_name=f"avastha_screener_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", width='stretch')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5: DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_drilldown:
        
        st.markdown("#### Symbol Deep Dive")
        
        selected_symbol = st.selectbox("Select Symbol for Detailed Analysis", options=df['Symbol'].tolist(), format_func=lambda x: f"{x} - {ETF_DISPLAY_NAMES.get(x, x.replace('.NS', ''))}")
        
        if selected_symbol:
            with st.spinner(f"Loading detailed analysis for {selected_symbol}..."):
                prices = fetch_price_data(selected_symbol, period="1y")
                
                if prices is not None and len(prices) >= 100:
                    engine = MarketRegimeEngine()
                    signal = engine.detect_regime(prices)
                    
                    # Symbol metrics
                    row = df[df['Symbol'] == selected_symbol].iloc[0]
                    regime_color = REGIME_COLORS.get(signal.primary_regime, '#888888')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""<div class='metric-card' style='border-top: 3px solid {regime_color};'>
                            <h4>Regime</h4><h2 style='color: {regime_color}; font-size: 1.25rem;'>{signal.primary_regime.value}</h2></div>""", unsafe_allow_html=True)
                    with col2:
                        osc_color = '#10b981' if signal.composite_oscillator > 0 else '#ef4444'
                        st.markdown(f"""<div class='metric-card'><h4>Oscillator</h4><h2 style='color: {osc_color};'>{signal.composite_oscillator:+.1f}</h2></div>""", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""<div class='metric-card'><h4>Exposure</h4><h2>{signal.recommended_exposure*100:.0f}%</h2></div>""", unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""<div class='metric-card'><h4>Risk</h4><h2>{signal.risk_score:.0f}</h2></div>""", unsafe_allow_html=True)
                    
                    # Price chart
                    fig = create_price_chart(prices, signal, selected_symbol)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Regime probabilities
                    st.markdown("#### Regime Probabilities")
                    prob_df = pd.DataFrame([
                        {"Regime": k, "Probability": f"{v*100:.1f}%"}
                        for k, v in sorted(signal.regime_probabilities.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(prob_df, hide_index=True, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    mode, symbol, universe, index_name = render_sidebar()
    
    if "Individual" in mode:
        if symbol:
            run_individual_mode(symbol)
        else:
            st.info("Please enter a symbol in the sidebar to begin analysis.")
    else:
        run_index_mode(universe, index_name)


if __name__ == "__main__":
    main()
