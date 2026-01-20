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

warnings.filterwarnings('ignore')

from avastha_engine import (
    MarketRegimeEngine, RegimeType, RegimeConfig, RegimeSignal,
    VolatilityRegime, MomentumRegime, TrendRegime,
    detect_regime, get_regime_color, get_regime_description
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "v2.0.0"

ETF_UNIVERSE_CATEGORIZED = {
    "Broad Market": {"NIFTYIETF.NS": "NIFTY 50", "MON100.NS": "NIFTY 100", "MONIFTY500.NS": "NIFTY 500", "MASPTOP50.NS": "Top 50"},
    "Market Cap": {"MIDCAPIETF.NS": "Midcap", "MOSMALL250.NS": "Smallcap 250"},
    "Banking & Finance": {"FINIETF.NS": "Financials", "PVTBANIETF.NS": "Private Banks", "PSUBNKIETF.NS": "PSU Banks", "ECAPINSURE.NS": "Insurance"},
    "Technology": {"ITIETF.NS": "IT", "MNC.NS": "MNC"},
    "Consumer": {"FMCGIETF.NS": "FMCG", "CONSUMIETF.NS": "Consumption", "AUTOIETF.NS": "Auto", "EVINDIA.NS": "EV India"},
    "Healthcare": {"HEALTHIETF.NS": "Healthcare"},
    "Industrial": {"INFRAIETF.NS": "Infrastructure", "CPSETF.NS": "CPSE", "MAKEINDIA.NS": "Make in India", "MODEFENCE.NS": "Defence", "MOREALTY.NS": "Realty", "TNIDETF.NS": "TN Infra", "GROWWPOWER.NS": "Power"},
    "Materials": {"METALIETF.NS": "Metal", "OILIETF.NS": "Oil & Gas", "CHEMICAL.NS": "Chemicals"},
    "Commodities": {"GOLDIETF.NS": "Gold", "SILVERIETF.NS": "Silver", "COMMOIETF.NS": "Commodities"},
}

ETF_UNIVERSE = []
ETF_DISPLAY_NAMES = {}
ETF_CATEGORIES = {}
for category, symbols in ETF_UNIVERSE_CATEGORIZED.items():
    for symbol, name in symbols.items():
        ETF_UNIVERSE.append(symbol)
        ETF_DISPLAY_NAMES[symbol] = name
        ETF_CATEGORIES[symbol] = category

INDEX_LIST = ["NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500", "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100", "NIFTY BANK", "NIFTY AUTO", "NIFTY FIN SERVICE", "NIFTY FMCG", "NIFTY IT", "NIFTY MEDIA", "NIFTY METAL", "NIFTY PHARMA"]
BASE_URL = "https://archives.nseindia.com/content/indices/"
INDEX_URLS = {idx: f"{BASE_URL}ind_{idx.lower().replace(' ', '')}list.csv" for idx in INDEX_LIST}
INDEX_URLS["NIFTY 50"] = f"{BASE_URL}ind_nifty50list.csv"
INDEX_URLS["NIFTY NEXT 50"] = f"{BASE_URL}ind_niftynext50list.csv"
INDEX_URLS["NIFTY SMLCAP 100"] = f"{BASE_URL}ind_niftysmallcap100list.csv"
INDEX_URLS["NIFTY FIN SERVICE"] = f"{BASE_URL}ind_niftyfinancelist.csv"

FNO_URL = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
UNIVERSE_OPTIONS = ["ETF Universe", "F&O Stocks", "Index Constituents"]

REGIME_COLORS = {
    RegimeType.CRISIS: "#ef4444", RegimeType.BEAR_ACCELERATION: "#f97316", RegimeType.BEAR_DECELERATION: "#fb923c",
    RegimeType.ACCUMULATION: "#06b6d4", RegimeType.EARLY_BULL: "#10b981", RegimeType.BULL_TREND: "#22c55e",
    RegimeType.BULL_EUPHORIA: "#f59e0b", RegimeType.DISTRIBUTION: "#a855f7", RegimeType.CHOP: "#6b7280", RegimeType.TRANSITION: "#FFC300",
}

REGIME_EMOJIS = {
    RegimeType.CRISIS: "🔴", RegimeType.BEAR_ACCELERATION: "📉", RegimeType.BEAR_DECELERATION: "🔻",
    RegimeType.ACCUMULATION: "📦", RegimeType.EARLY_BULL: "🌱", RegimeType.BULL_TREND: "🐂",
    RegimeType.BULL_EUPHORIA: "🎪", RegimeType.DISTRIBUTION: "📤", RegimeType.CHOP: "🌀", RegimeType.TRANSITION: "🔄",
}

REGIME_RISK_WEIGHTS = {"CRISIS": 1.0, "BEAR_ACCELERATION": 0.85, "BEAR_DECELERATION": 0.6, "DISTRIBUTION": 0.7, "CHOP": 0.5, "TRANSITION": 0.5, "ACCUMULATION": 0.3, "EARLY_BULL": 0.2, "BULL_TREND": 0.15, "BULL_EUPHORIA": 0.4}
BULLISH_REGIMES = ['EARLY_BULL', 'BULL_TREND', 'BULL_EUPHORIA', 'ACCUMULATION']
BEARISH_REGIMES = ['CRISIS', 'BEAR_ACCELERATION', 'BEAR_DECELERATION', 'DISTRIBUTION']
NEUTRAL_REGIMES = ['CHOP', 'TRANSITION']

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG AND STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="AVASTHA | Market Regime Detection", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #FFC300;
    --primary-rgb: 255, 195, 0;
    --bg-dark: #0a0a0a;
    --bg-card: #111111;
    --bg-elevated: #1a1a1a;
    --bg-hover: #222222;
    --text-primary: #f5f5f5;
    --text-muted: #737373;
    --border: #262626;
    --border-light: #333333;
    --green: #10b981;
    --red: #ef4444;
    --amber: #f59e0b;
    --cyan: #06b6d4;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.main { background-color: var(--bg-dark); }
[data-testid="stSidebar"] { background-color: var(--bg-card); border-right: 1px solid var(--border); }
.stApp > header { background-color: transparent; }
.block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1600px; }

/* Header */
.app-header {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--primary), transparent);
}
.app-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 0.75rem; }
.app-header p { margin: 0.5rem 0 0 0; font-size: 0.875rem; color: var(--text-muted); }

/* Metric Cards */
.metric-grid { display: grid; gap: 1rem; margin-bottom: 1.5rem; }
.metric-grid-4 { grid-template-columns: repeat(4, 1fr); }
.metric-grid-5 { grid-template-columns: repeat(5, 1fr); }
.metric-grid-6 { grid-template-columns: repeat(6, 1fr); }

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover { border-color: var(--border-light); transform: translateY(-1px); }
.metric-card.accent { border-top: 3px solid var(--primary); padding-top: calc(1.25rem - 3px); }
.metric-card .label { font-size: 0.7rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }
.metric-card .value { font-size: 1.75rem; font-weight: 700; color: var(--text-primary); line-height: 1.2; }
.metric-card .subtext { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; }
.metric-card .value.green { color: var(--green); }
.metric-card .value.red { color: var(--red); }
.metric-card .value.amber { color: var(--amber); }
.metric-card .value.cyan { color: var(--cyan); }
.metric-card .value.primary { color: var(--primary); }

/* Section Title */
.section-title { font-size: 0.875rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }

/* Charts Container */
.chart-container { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
.chart-title { font-size: 0.8rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem; }

/* Insights */
.insight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
.insight-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 8px;
    padding: 1rem 1.25rem;
}
.insight-card .title { font-size: 0.8rem; font-weight: 600; color: var(--primary); margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
.insight-card .text { font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; }

/* Info Box */
.info-box { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin: 1rem 0; }
.info-box h4 { color: var(--primary); margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600; }
.info-box p { color: var(--text-muted); margin: 0; font-size: 0.85rem; line-height: 1.5; }

/* Sidebar */
.sidebar-brand { text-align: center; padding: 1rem 0 0.5rem 0; }
.sidebar-brand h2 { color: var(--primary); margin: 0; font-size: 1.5rem; font-weight: 700; }
.sidebar-brand p { color: var(--text-muted); margin: 0.25rem 0 0 0; font-size: 0.75rem; }
.sidebar-divider { height: 1px; background: var(--border); margin: 1rem 0; }
.sidebar-label { font-size: 0.7rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin: 1rem 0 0.5rem 0; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 2px solid var(--primary) !important;
    color: var(--primary) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.625rem 1.5rem !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { background: var(--primary) !important; color: #000 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0; background: var(--bg-card); border-radius: 8px; padding: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { 
    color: var(--text-muted); 
    background: transparent; 
    border-radius: 6px; 
    padding: 0.5rem 1rem; 
    font-size: 0.8rem; 
    font-weight: 500;
    border: none;
}
.stTabs [aria-selected="true"] { color: var(--primary); background: var(--bg-elevated); }

/* Dataframe & Charts */
.stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
.stPlotlyChart { border-radius: 8px; overflow: hidden; }

/* Tables */
.stMarkdown table { width: 100%; border-collapse: collapse; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; font-size: 0.85rem; }
.stMarkdown table th { background: var(--bg-elevated); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px; padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }
.stMarkdown table td { padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); color: var(--text-primary); }
.stMarkdown table tr:last-child td { border-bottom: none; }
.stMarkdown table tr:hover { background: var(--bg-hover); }

/* Selectbox & inputs */
.stSelectbox > div > div { background: var(--bg-card); border-color: var(--border); }
.stTextInput > div > div > input { background: var(--bg-card); border-color: var(--border); color: var(--text-primary); }
.stMultiSelect > div { background: var(--bg-card); }

/* Slider */
.stSlider > div > div { background: var(--border); }
.stSlider > div > div > div { background: var(--primary); }

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(symbol: str, period: str = "1y") -> pd.Series:
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        return df['Close'] if not df.empty else None
    except:
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fno_stocks() -> list:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(FNO_URL, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if 'SYMBOL' in df.columns:
                return [f"{s.strip()}.NS" for s in df['SYMBOL'].dropna().unique().tolist() if not s.startswith('NIFTY') and s.strip()][:200]
    except:
        pass
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_constituents(index_name: str) -> list:
    if index_name not in INDEX_URLS:
        return []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(INDEX_URLS[index_name], headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if 'Symbol' in df.columns:
                return [f"{s.strip()}.NS" for s in df['Symbol'].dropna().unique().tolist() if s.strip()]
    except:
        pass
    return []

def get_universe_symbols(universe: str, index_name: str = None) -> list:
    if universe == "ETF Universe": return ETF_UNIVERSE
    elif universe == "F&O Stocks": return fetch_fno_stocks()
    elif universe == "Index Constituents" and index_name: return fetch_index_constituents(index_name)
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_universe_metrics(df: pd.DataFrame) -> dict:
    bullish_count = len(df[df['Regime'].isin(BULLISH_REGIMES)])
    bearish_count = len(df[df['Regime'].isin(BEARISH_REGIMES)])
    total = len(df)
    
    df['RiskWeight'] = df['Regime'].map(REGIME_RISK_WEIGHTS)
    weighted_risk = (df['RiskWeight'] * df['Confidence']).sum() / df['Confidence'].sum() if df['Confidence'].sum() > 0 else 0.5
    
    strong_bullish = len(df[df['Momentum'].isin(['STRONG_BULLISH', 'BULLISH'])])
    strong_bearish = len(df[df['Momentum'].isin(['STRONG_BEARISH', 'BEARISH'])])
    uptrend_count = len(df[df['Trend'].isin(['UPTREND', 'STRONG_UPTREND'])])
    downtrend_count = len(df[df['Trend'].isin(['DOWNTREND', 'STRONG_DOWNTREND'])])
    
    regime_counts = df['Regime'].value_counts(normalize=True)
    
    return {
        'total': total, 'bullish_count': bullish_count, 'bearish_count': bearish_count,
        'neutral_count': total - bullish_count - bearish_count,
        'bullish_pct': bullish_count / total if total > 0 else 0,
        'bearish_pct': bearish_count / total if total > 0 else 0,
        'weighted_risk': weighted_risk,
        'osc_mean': df['Oscillator'].mean(), 'osc_std': df['Oscillator'].std(),
        'avg_exposure': df['Exposure'].mean(),
        'momentum_breadth': (strong_bullish - strong_bearish) / total if total > 0 else 0,
        'trend_breadth': (uptrend_count - downtrend_count) / total if total > 0 else 0,
        'concentration': (regime_counts ** 2).sum(),
        'regime_distribution': regime_counts.to_dict()
    }

def generate_insights(df: pd.DataFrame, metrics: dict) -> list:
    insights = []
    
    if metrics['bullish_pct'] > 0.6:
        insights.append({'type': 'bullish', 'title': 'Strong Bullish Breadth', 'text': f"{metrics['bullish_pct']*100:.0f}% of universe in bullish regimes. Consider momentum strategies."})
    elif metrics['bearish_pct'] > 0.5:
        insights.append({'type': 'bearish', 'title': 'Elevated Bearish Concentration', 'text': f"{metrics['bearish_pct']*100:.0f}% in bearish regimes. Defensive positioning recommended."})
    
    if metrics['concentration'] > 0.3:
        dominant = max(metrics['regime_distribution'], key=metrics['regime_distribution'].get)
        insights.append({'type': 'warning', 'title': 'Regime Concentration', 'text': f"High concentration in {dominant} ({metrics['regime_distribution'][dominant]*100:.0f}%). Monitor for transitions."})
    
    if metrics['osc_mean'] > 30:
        insights.append({'type': 'warning', 'title': 'Universe Overbought', 'text': f"Mean oscillator at {metrics['osc_mean']:.1f}. Elevated mean-reversion risk."})
    elif metrics['osc_mean'] < -30:
        insights.append({'type': 'opportunity', 'title': 'Universe Oversold', 'text': f"Mean oscillator at {metrics['osc_mean']:.1f}. Contrarian opportunity if fundamentals support."})
    
    high_vol = len(df[df['Volatility'].isin(['ELEVATED', 'EXTREME'])])
    if high_vol / metrics['total'] > 0.4:
        insights.append({'type': 'warning', 'title': 'High Volatility', 'text': f"{high_vol} symbols ({high_vol/metrics['total']*100:.0f}%) with elevated volatility. Reduce position sizes."})
    
    return insights[:3]

def get_sector_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if 'Category' not in df.columns: return None
    
    stats = df.groupby('Category').agg({
        'Oscillator': ['mean', 'count'],
        'Exposure': 'mean',
        'Risk': 'mean'
    }).round(2)
    stats.columns = ['Osc Mean', 'Count', 'Avg Exposure', 'Avg Risk']
    stats = stats.reset_index()
    
    bullish_pct = df[df['Regime'].isin(BULLISH_REGIMES)].groupby('Category').size() / df.groupby('Category').size()
    stats['Bullish %'] = stats['Category'].map(bullish_pct).fillna(0)
    
    return stats.sort_values('Osc Mean', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

CHART_CONFIG = dict(
    paper_bgcolor='#111111',
    plot_bgcolor='#111111',
    font=dict(family="Inter", color='#f5f5f5', size=11),
    margin=dict(l=50, r=30, t=30, b=50),
)

def create_regime_donut(df: pd.DataFrame) -> go.Figure:
    counts = df['Regime'].value_counts()
    colors = [REGIME_COLORS.get(RegimeType(r), '#6b7280') for r in counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values, hole=0.65,
        marker_colors=colors, textinfo='percent', textposition='outside',
        textfont=dict(size=10, color='#f5f5f5'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(**CHART_CONFIG, height=320, showlegend=False,
        annotations=[dict(text=f'<b>{len(df)}</b><br><span style="font-size:10px">Symbols</span>', 
                         x=0.5, y=0.5, font_size=18, font_color='#f5f5f5', showarrow=False)])
    return fig

def create_oscillator_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    for regime_type, color, name in [(BULLISH_REGIMES, '#10b981', 'Bullish'), (BEARISH_REGIMES, '#ef4444', 'Bearish'), (NEUTRAL_REGIMES, '#6b7280', 'Neutral')]:
        subset = df[df['Regime'].isin(regime_type)]
        if len(subset) > 0:
            fig.add_trace(go.Histogram(x=subset['Oscillator'], name=name, marker_color=color, opacity=0.75, nbinsx=15))
    
    fig.add_vline(x=df['Oscillator'].mean(), line_dash="dash", line_color="#FFC300", annotation_text=f"μ={df['Oscillator'].mean():.1f}", annotation_font_color="#FFC300")
    fig.add_vline(x=0, line_dash="dot", line_color="#6b7280")
    
    fig.update_layout(**CHART_CONFIG, height=320, barmode='overlay',
        xaxis=dict(title="Oscillator", gridcolor='#262626', range=[-100, 100], zeroline=False),
        yaxis=dict(title="Count", gridcolor='#262626', zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10))
    return fig

def create_risk_gauge(risk: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        number={'suffix': '%', 'font': {'size': 32, 'color': '#f5f5f5'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#f5f5f5', 'tickfont': {'size': 10}},
            'bar': {'color': '#FFC300', 'thickness': 0.3},
            'bgcolor': '#262626',
            'borderwidth': 0,
            'steps': [{'range': [0, 35], 'color': '#10b981'}, {'range': [35, 65], 'color': '#f59e0b'}, {'range': [65, 100], 'color': '#ef4444'}],
        }
    ))
    fig.update_layout(**CHART_CONFIG, height=220, margin=dict(l=30, r=30, t=40, b=20))
    return fig

def create_breadth_bars(metrics: dict) -> go.Figure:
    cats = ['Regime', 'Momentum', 'Trend']
    vals = [(metrics['bullish_pct'] - metrics['bearish_pct']) * 100, metrics['momentum_breadth'] * 100, metrics['trend_breadth'] * 100]
    colors = ['#10b981' if v > 0 else '#ef4444' for v in vals]
    
    fig = go.Figure(go.Bar(x=cats, y=vals, marker_color=colors, text=[f"{v:+.0f}%" for v in vals], textposition='outside', textfont=dict(color='#f5f5f5', size=11)))
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(**CHART_CONFIG, height=220, xaxis=dict(gridcolor='#262626'), yaxis=dict(title="Breadth %", gridcolor='#262626', range=[-100, 100], zeroline=False), showlegend=False)
    return fig

def create_scatter_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    for regime in df['Regime'].unique():
        subset = df[df['Regime'] == regime]
        color = REGIME_COLORS.get(RegimeType(regime), '#6b7280')
        fig.add_trace(go.Scatter(
            x=subset['Oscillator'], y=subset['Exposure'] * 100, mode='markers', name=regime,
            marker=dict(color=color, size=9, opacity=0.8, line=dict(width=1, color='#111111')),
            text=subset['Name'], hovertemplate='<b>%{text}</b><br>Osc: %{x:.1f}<br>Exp: %{y:.0f}%<extra></extra>'
        ))
    
    fig.add_hline(y=70, line_dash="dot", line_color="#10b981", annotation_text="High", annotation_font_size=10)
    fig.add_hline(y=40, line_dash="dot", line_color="#ef4444", annotation_text="Low", annotation_font_size=10)
    fig.add_vline(x=0, line_dash="dot", line_color="#6b7280")
    
    fig.update_layout(**CHART_CONFIG, height=400, xaxis=dict(title="Oscillator", gridcolor='#262626', range=[-100, 100], zeroline=False),
        yaxis=dict(title="Exposure %", gridcolor='#262626', range=[0, 150], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font_size=9))
    return fig

def create_sector_heatmap(df: pd.DataFrame) -> go.Figure:
    if 'Category' not in df.columns: return None
    
    pivot = df.pivot_table(values='Oscillator', index='Category', columns='Regime', aggfunc='count', fill_value=0)
    order = ['CRISIS', 'BEAR_ACCELERATION', 'BEAR_DECELERATION', 'DISTRIBUTION', 'CHOP', 'TRANSITION', 'ACCUMULATION', 'EARLY_BULL', 'BULL_TREND', 'BULL_EUPHORIA']
    pivot = pivot[[c for c in order if c in pivot.columns]]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, '#111111'], [0.5, '#FFC300'], [1, '#FFC300']],
        showscale=False, text=pivot.values, texttemplate="%{text}", textfont=dict(color='#f5f5f5', size=11)
    ))
    fig.update_layout(**CHART_CONFIG, height=350, xaxis=dict(tickangle=45, tickfont=dict(size=9)), yaxis=dict(tickfont=dict(size=10)))
    return fig

def create_price_chart(prices: pd.Series, signal: RegimeSignal, symbol: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28])
    
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines', name='Price', line=dict(color='#FFC300', width=2), fill='tozeroy', fillcolor='rgba(255, 195, 0, 0.08)'), row=1, col=1)
    
    if len(prices) >= 20:
        fig.add_trace(go.Scatter(x=prices.index, y=prices.rolling(20).mean(), mode='lines', name='MA20', line=dict(color='#06b6d4', width=1, dash='dot')), row=1, col=1)
    if len(prices) >= 50:
        fig.add_trace(go.Scatter(x=prices.index, y=prices.rolling(50).mean(), mode='lines', name='MA50', line=dict(color='#a855f7', width=1, dash='dot')), row=1, col=1)
    
    osc_color = '#10b981' if signal.composite_oscillator > 0 else '#ef4444'
    fig.add_trace(go.Bar(x=[prices.index[-1]], y=[signal.composite_oscillator], name='Osc', marker_color=osc_color, showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", row=2, col=1)
    
    fig.update_layout(**CHART_CONFIG, height=450, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10), hovermode='x unified')
    fig.update_xaxes(gridcolor='#262626', zeroline=False)
    fig.update_yaxes(gridcolor='#262626', zeroline=False, title_text="Price", row=1, col=1)
    fig.update_yaxes(gridcolor='#262626', zeroline=False, title_text="Osc", range=[-100, 100], row=2, col=1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-brand"><h2>🔮 AVASTHA</h2><p>आवस्था • Market Regime</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-label">Analysis Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Mode", ["📈 Individual Scrip", "📊 Index Universe"], label_visibility="collapsed")
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        symbol, universe, index_name = None, None, None
        
        if "Individual" in mode:
            st.markdown('<div class="sidebar-label">Symbol</div>', unsafe_allow_html=True)
            symbol = st.text_input("Symbol", value="NIFTYIETF.NS", placeholder="e.g., RELIANCE.NS", label_visibility="collapsed")
        else:
            st.markdown('<div class="sidebar-label">Universe</div>', unsafe_allow_html=True)
            universe = st.selectbox("Universe", UNIVERSE_OPTIONS, label_visibility="collapsed")
            if universe == "Index Constituents":
                index_name = st.selectbox("Index", INDEX_LIST, index=0)
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box"><p style="font-size:0.75rem;margin:0;"><b>Version:</b> {VERSION}<br><b>Engine:</b> Multi-Model<br><b>Regimes:</b> 10 States</p></div>', unsafe_allow_html=True)
        
        return mode, symbol, universe, index_name


# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL SCRIP MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_individual_mode(symbol: str):
    st.markdown(f'<div class="app-header"><h1>🔮 AVASTHA</h1><p>Multi-Model Market Regime Detection • {symbol}</p></div>', unsafe_allow_html=True)
    
    with st.spinner(f"Analyzing {symbol}..."):
        prices = fetch_price_data(symbol, period="1y")
    
    if prices is None or len(prices) < 100:
        st.error(f"Insufficient data for {symbol}. Please check the symbol.")
        return
    
    engine = MarketRegimeEngine()
    signal = engine.detect_regime(prices)
    regime_color = REGIME_COLORS.get(signal.primary_regime, '#6b7280')
    emoji = REGIME_EMOJIS.get(signal.primary_regime, '📊')
    
    # Primary Metrics
    osc_class = 'green' if signal.composite_oscillator > 0 else 'red'
    exp_class = 'green' if signal.recommended_exposure >= 0.7 else ('amber' if signal.recommended_exposure >= 0.4 else 'red')
    sig_class = 'primary' if signal.signal_strength >= 0.5 else ''
    
    st.markdown(f"""
    <div class="metric-grid metric-grid-4">
        <div class="metric-card accent" style="border-top-color: {regime_color};">
            <div class="label">Current Regime</div>
            <div class="value" style="color: {regime_color}; font-size: 1.25rem;">{emoji} {signal.primary_regime.value}</div>
            <div class="subtext">Confidence: {signal.primary_confidence*100:.0f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Composite Oscillator</div>
            <div class="value {osc_class}">{signal.composite_oscillator:+.1f}</div>
            <div class="subtext">Range: -100 to +100</div>
        </div>
        <div class="metric-card">
            <div class="label">Recommended Exposure</div>
            <div class="value {exp_class}">{signal.recommended_exposure*100:.0f}%</div>
            <div class="subtext">Based on regime</div>
        </div>
        <div class="metric-card">
            <div class="label">Signal Strength</div>
            <div class="value {sig_class}">{signal.signal_strength*100:.0f}%</div>
            <div class="subtext">Conviction level</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sub-regimes
    vol_class = 'green' if signal.volatility_regime == VolatilityRegime.COMPRESSED else ('red' if signal.volatility_regime == VolatilityRegime.EXTREME else 'amber')
    mom_class = 'green' if 'BULLISH' in signal.momentum_regime.value else ('red' if 'BEARISH' in signal.momentum_regime.value else '')
    trend_class = 'green' if 'UP' in signal.trend_regime.value else ('red' if 'DOWN' in signal.trend_regime.value else '')
    risk_class = 'green' if signal.risk_score < 40 else ('red' if signal.risk_score > 70 else 'amber')
    
    st.markdown('<div class="section-title">Sub-Regime Analysis</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid metric-grid-4">
        <div class="metric-card">
            <div class="label">Volatility</div>
            <div class="value {vol_class}" style="font-size: 1.1rem;">{signal.volatility_regime.value}</div>
            <div class="subtext">Percentile: {signal.volatility_percentile:.0f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Momentum</div>
            <div class="value {mom_class}" style="font-size: 1.1rem;">{signal.momentum_regime.value}</div>
            <div class="subtext">Directional bias</div>
        </div>
        <div class="metric-card">
            <div class="label">Trend</div>
            <div class="value {trend_class}" style="font-size: 1.1rem;">{signal.trend_regime.value}</div>
            <div class="subtext">Structural state</div>
        </div>
        <div class="metric-card">
            <div class="label">Risk Score</div>
            <div class="value {risk_class}">{signal.risk_score:.0f}</div>
            <div class="subtext">0 = Low, 100 = High</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chart
    st.markdown('<div class="section-title">Price & Oscillator</div>', unsafe_allow_html=True)
    fig = create_price_chart(prices, signal, symbol)
    st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX UNIVERSE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_index_mode(universe: str, index_name: str = None):
    title = "ETF Universe" if universe == "ETF Universe" else ("F&O Stocks" if universe == "F&O Stocks" else index_name)
    
    st.markdown(f'<div class="app-header"><h1>🔮 AVASTHA</h1><p>Universe Regime Screening • {title}</p></div>', unsafe_allow_html=True)
    
    symbols = get_universe_symbols(universe, index_name)
    if not symbols:
        st.error("Could not fetch symbols.")
        return
    
    cache_key = f"avastha_{universe}_{index_name}"
    
    if cache_key not in st.session_state:
        st.markdown(f'<div class="info-box"><h4>📊 {title}</h4><p>Ready to analyze {len(symbols)} symbols for regime classification.</p></div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run Screening", use_container_width=True):
            results = run_screening(symbols, universe)
            if results:
                st.session_state[cache_key] = results
                st.rerun()
    else:
        render_dashboard(st.session_state[cache_key], universe, title)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Re-run"):
            del st.session_state[cache_key]
            st.rerun()


def run_screening(symbols: list, universe: str) -> list:
    results = []
    engine = MarketRegimeEngine()
    progress = st.progress(0)
    status = st.empty()
    
    for i, sym in enumerate(symbols):
        status.markdown(f"**Analyzing {sym}... ({i+1}/{len(symbols)})**")
        try:
            prices = fetch_price_data(sym, period="1y")
            if prices is not None and len(prices) >= 100:
                signal = engine.detect_regime(prices)
                results.append({
                    'Symbol': sym, 'Name': ETF_DISPLAY_NAMES.get(sym, sym.replace('.NS', '')),
                    'Category': ETF_CATEGORIES.get(sym, 'Other'),
                    'Regime': signal.primary_regime.value, 'Confidence': signal.primary_confidence,
                    'Oscillator': signal.composite_oscillator, 'Exposure': signal.recommended_exposure,
                    'Volatility': signal.volatility_regime.value, 'Momentum': signal.momentum_regime.value,
                    'Trend': signal.trend_regime.value, 'Risk': signal.risk_score, 'Signal': signal.signal_strength,
                    'Price': prices.iloc[-1], 'Change': ((prices.iloc[-1] / prices.iloc[-2]) - 1) * 100 if len(prices) >= 2 else 0
                })
        except: pass
        progress.progress((i + 1) / len(symbols))
        time.sleep(0.05)
    
    progress.empty()
    status.empty()
    return results


def render_dashboard(results: list, universe: str, title: str):
    df = pd.DataFrame(results)
    if df.empty:
        st.error("No data retrieved.")
        return
    
    metrics = calculate_universe_metrics(df)
    insights = generate_insights(df, metrics)
    
    # Tabs
    tabs = st.tabs(["📊 Overview", "📈 Risk", "🏭 Sectors", "🔍 Screener", "🎯 Drill-Down"])
    
    # === OVERVIEW TAB ===
    with tabs[0]:
        osc_class = 'green' if metrics['osc_mean'] > 0 else 'red'
        exp_class = 'green' if metrics['avg_exposure'] >= 0.7 else ('amber' if metrics['avg_exposure'] >= 0.4 else 'red')
        risk_class = 'green' if metrics['weighted_risk'] < 0.4 else ('red' if metrics['weighted_risk'] > 0.6 else 'amber')
        
        st.markdown(f"""
        <div class="metric-grid metric-grid-6">
            <div class="metric-card"><div class="label">Universe</div><div class="value primary">{metrics['total']}</div><div class="subtext">Symbols</div></div>
            <div class="metric-card"><div class="label">Bullish</div><div class="value green">{metrics['bullish_count']}</div><div class="subtext">{metrics['bullish_pct']*100:.0f}%</div></div>
            <div class="metric-card"><div class="label">Bearish</div><div class="value red">{metrics['bearish_count']}</div><div class="subtext">{metrics['bearish_pct']*100:.0f}%</div></div>
            <div class="metric-card"><div class="label">Mean Osc</div><div class="value {osc_class}">{metrics['osc_mean']:+.1f}</div><div class="subtext">σ={metrics['osc_std']:.1f}</div></div>
            <div class="metric-card"><div class="label">Avg Exposure</div><div class="value {exp_class}">{metrics['avg_exposure']*100:.0f}%</div><div class="subtext">Recommended</div></div>
            <div class="metric-card"><div class="label">Risk Score</div><div class="value {risk_class}">{metrics['weighted_risk']*100:.0f}</div><div class="subtext">Weighted</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">Regime Distribution</div></div>', unsafe_allow_html=True)
            st.plotly_chart(create_regime_donut(df), width='stretch')
        with col2:
            st.markdown('<div class="chart-container"><div class="chart-title">Oscillator Distribution</div></div>', unsafe_allow_html=True)
            st.plotly_chart(create_oscillator_histogram(df), width='stretch')
        
        if insights:
            st.markdown('<div class="section-title">💡 Insights</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-grid">' + ''.join([f'<div class="insight-card"><div class="title">{"🟢" if i["type"]=="bullish" else "🔴" if i["type"]=="bearish" else "⚠️"} {i["title"]}</div><div class="text">{i["text"]}</div></div>' for i in insights]) + '</div>', unsafe_allow_html=True)
    
    # === RISK TAB ===
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">Universe Risk</div></div>', unsafe_allow_html=True)
            st.plotly_chart(create_risk_gauge(metrics['weighted_risk']), width='stretch')
        with col2:
            st.markdown('<div class="chart-container"><div class="chart-title">Market Breadth</div></div>', unsafe_allow_html=True)
            st.plotly_chart(create_breadth_bars(metrics), width='stretch')
        
        st.markdown('<div class="chart-container"><div class="chart-title">Oscillator vs Exposure</div></div>', unsafe_allow_html=True)
        st.plotly_chart(create_scatter_plot(df), width='stretch')
    
    # === SECTORS TAB ===
    with tabs[2]:
        if 'Category' in df.columns and universe == "ETF Universe":
            st.markdown('<div class="chart-container"><div class="chart-title">Sector × Regime Heatmap</div></div>', unsafe_allow_html=True)
            fig = create_sector_heatmap(df)
            if fig: st.plotly_chart(fig, width='stretch')
            
            st.markdown('<div class="section-title">Sector Statistics</div>', unsafe_allow_html=True)
            stats = get_sector_analysis(df)
            if stats is not None:
                stats['Osc Mean'] = stats['Osc Mean'].apply(lambda x: f"{x:+.1f}")
                stats['Avg Exposure'] = stats['Avg Exposure'].apply(lambda x: f"{x*100:.0f}%")
                stats['Bullish %'] = stats['Bullish %'].apply(lambda x: f"{x*100:.0f}%")
                st.dataframe(stats, hide_index=True, width='stretch')
        else:
            st.info("Sector view available for ETF Universe only.")
    
    # === SCREENER TAB ===
    with tabs[3]:
        st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            f_regime = st.multiselect("Regime", df['Regime'].unique().tolist(), placeholder="All")
        with col2:
            f_mom = st.multiselect("Momentum", df['Momentum'].unique().tolist(), placeholder="All")
        with col3:
            f_trend = st.multiselect("Trend", df['Trend'].unique().tolist(), placeholder="All")
        with col4:
            f_vol = st.multiselect("Volatility", df['Volatility'].unique().tolist(), placeholder="All")
        
        col1, col2 = st.columns(2)
        with col1:
            osc_range = st.slider("Oscillator", -100, 100, (-100, 100))
        with col2:
            exp_range = st.slider("Exposure %", 0, 150, (0, 150))
        
        fdf = df.copy()
        if f_regime: fdf = fdf[fdf['Regime'].isin(f_regime)]
        if f_mom: fdf = fdf[fdf['Momentum'].isin(f_mom)]
        if f_trend: fdf = fdf[fdf['Trend'].isin(f_trend)]
        if f_vol: fdf = fdf[fdf['Volatility'].isin(f_vol)]
        fdf = fdf[(fdf['Oscillator'] >= osc_range[0]) & (fdf['Oscillator'] <= osc_range[1])]
        fdf = fdf[(fdf['Exposure'] * 100 >= exp_range[0]) & (fdf['Exposure'] * 100 <= exp_range[1])]
        
        st.markdown(f'<div class="section-title">Results ({len(fdf)})</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            sort_by = st.selectbox("Sort", ["Oscillator", "Confidence", "Exposure", "Risk", "Change"])
            asc = st.checkbox("Asc", value=False)
        
        disp = fdf.sort_values(sort_by, ascending=asc).copy()
        disp['Confidence'] = disp['Confidence'].apply(lambda x: f"{x*100:.0f}%")
        disp['Oscillator'] = disp['Oscillator'].apply(lambda x: f"{x:+.1f}")
        disp['Exposure'] = disp['Exposure'].apply(lambda x: f"{x*100:.0f}%")
        disp['Risk'] = disp['Risk'].apply(lambda x: f"{x:.0f}")
        disp['Price'] = disp['Price'].apply(lambda x: f"₹{x:,.2f}")
        disp['Change'] = disp['Change'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(disp[['Symbol', 'Name', 'Regime', 'Confidence', 'Oscillator', 'Exposure', 'Volatility', 'Momentum', 'Trend', 'Risk', 'Price', 'Change']], hide_index=True, width='stretch')
        
        csv = fdf.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, f"avastha_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
    
    # === DRILL-DOWN TAB ===
    with tabs[4]:
        st.markdown('<div class="section-title">Symbol Analysis</div>', unsafe_allow_html=True)
        
        sel = st.selectbox("Select Symbol", df['Symbol'].tolist(), format_func=lambda x: f"{x} - {ETF_DISPLAY_NAMES.get(x, x.replace('.NS', ''))}")
        
        if sel:
            with st.spinner(f"Loading {sel}..."):
                prices = fetch_price_data(sel, period="1y")
                
                if prices is not None and len(prices) >= 100:
                    engine = MarketRegimeEngine()
                    signal = engine.detect_regime(prices)
                    regime_color = REGIME_COLORS.get(signal.primary_regime, '#6b7280')
                    
                    osc_class = 'green' if signal.composite_oscillator > 0 else 'red'
                    exp_class = 'green' if signal.recommended_exposure >= 0.7 else ('amber' if signal.recommended_exposure >= 0.4 else 'red')
                    
                    st.markdown(f"""
                    <div class="metric-grid metric-grid-4">
                        <div class="metric-card accent" style="border-top-color: {regime_color};">
                            <div class="label">Regime</div>
                            <div class="value" style="color: {regime_color}; font-size: 1.1rem;">{signal.primary_regime.value}</div>
                        </div>
                        <div class="metric-card"><div class="label">Oscillator</div><div class="value {osc_class}">{signal.composite_oscillator:+.1f}</div></div>
                        <div class="metric-card"><div class="label">Exposure</div><div class="value {exp_class}">{signal.recommended_exposure*100:.0f}%</div></div>
                        <div class="metric-card"><div class="label">Risk</div><div class="value">{signal.risk_score:.0f}</div></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig = create_price_chart(prices, signal, sel)
                    st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    mode, symbol, universe, index_name = render_sidebar()
    
    if "Individual" in mode:
        if symbol: run_individual_mode(symbol)
        else: st.info("Enter a symbol in the sidebar.")
    else:
        run_index_mode(universe, index_name)

if __name__ == "__main__":
    main()
