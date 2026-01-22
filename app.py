"""
AVASTHA - Market Regime Detection System

Institutional-grade market regime detection using adaptive methods:
- Hidden Markov Models (HMM) for state discovery
- Kalman Filtering for score smoothing
- Rolling Percentile Normalization (NO fixed thresholds)
- GARCH-inspired volatility regime adjustment
- Bayesian confidence calculation
- CUSUM change point detection

Version: 2.0.0
Author: Hemrek Capital
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import datetime as dt
import logging

# Local imports - Use adaptive detector
from adaptive_regime_detector import (
    AdaptiveRegimeDetector as MarketRegimeDetector,
    AdaptiveRegimeType as RegimeType,
    AdaptiveRegimeResult as RegimeResult
)
from data_engine import (
    MarketDataEngine, 
    get_universe_symbols,
    UNIVERSE_OPTIONS, 
    INDEX_LIST, 
    ETF_UNIVERSE,
    ETF_NAMES,
    get_display_name
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "v2.0.0"
APP_TITLE = "AVASTHA"
APP_SUBTITLE = "Market Regime Detection System"

st.set_page_config(
    page_title=f"{APP_TITLE} | {APP_SUBTITLE}",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM CSS
# ══════════════════════════════════════════════════════════════════════════════

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
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }
    
    /* Sidebar toggle button */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        width: 40px !important;
        height: 40px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important;
        transform: scale(1.05);
    }
    
    [data-testid="collapsedControl"] svg {
        stroke: var(--primary-color) !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] svg {
        stroke: var(--primary-color) !important;
    }
    
    button[kind="header"] { z-index: 999999 !important; }
    
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    
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
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .warning-box { background: rgba(245, 158, 11, 0.1); border: 1px solid var(--warning-amber); border-radius: 10px; padding: 1rem; margin: 1rem 0; }
    
    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
    
    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if 'regime_result' not in st.session_state:
    st.session_state.regime_result = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'time_series_results' not in st.session_state:
    st.session_state.time_series_results = None
if 'latest_df' not in st.session_state:
    st.session_state.latest_df = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_color_class(regime: RegimeType) -> str:
    bull_regimes = [RegimeType.STRONG_BULL, RegimeType.BULL, RegimeType.WEAK_BULL]
    bear_regimes = [RegimeType.BEAR, RegimeType.CRISIS, RegimeType.WEAK_BEAR]
    if regime in bull_regimes:
        return "bull"
    elif regime in bear_regimes:
        return "bear"
    return "chop"


def get_regime_color(regime_name: str) -> str:
    if regime_name in ['STRONG_BULL', 'BULL']:
        return "#10b981"
    elif regime_name in ['BEAR', 'CRISIS']:
        return "#ef4444"
    elif regime_name == 'TRANSITION':
        return "#a855f7"
    return "#f59e0b"


def get_score_color(score: float) -> str:
    if score >= 0.3:
        return "#10b981"
    elif score <= -0.3:
        return "#ef4444"
    return "#f59e0b"


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_regime_gauge(score: float, confidence: float) -> go.Figure:
    """Create gauge chart for regime strength"""
    gauge_value = (score + 2) / 4 * 100
    
    if score >= 0.5:
        color = "#10b981"
        regime_text = "BULLISH"
    elif score <= -0.5:
        color = "#ef4444"
        regime_text = "BEARISH"
    else:
        color = "#f59e0b"
        regime_text = "NEUTRAL"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_value,
        number=dict(font=dict(size=36, color=color, family='Inter'), suffix="%"),
        delta=dict(reference=50, valueformat=".0f", increasing=dict(color="#10b981"), decreasing=dict(color="#ef4444")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor='#3A3A3A', tickvals=[0, 25, 50, 75, 100], tickfont=dict(size=10, color='#888888')),
            bar=dict(color=color, thickness=0.3),
            bgcolor='#1A1A1A',
            borderwidth=2,
            bordercolor='#2A2A2A',
            steps=[
                dict(range=[0, 25], color='rgba(239,68,68,0.15)'),
                dict(range=[25, 50], color='rgba(245,158,11,0.1)'),
                dict(range=[50, 75], color='rgba(245,158,11,0.1)'),
                dict(range=[75, 100], color='rgba(16,185,129,0.15)')
            ],
            threshold=dict(line=dict(color='white', width=2), thickness=0.8, value=gauge_value)
        )
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=250, margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family='Inter', color='#EAEAEA'),
        annotations=[dict(text=f"<b>{regime_text}</b><br>Confidence: {confidence:.0%}", x=0.5, y=-0.1, showarrow=False, font=dict(size=12, color='#888888'))]
    )
    return fig


def create_composite_score_chart(score: float) -> go.Figure:
    """Create visualization for composite regime score position"""
    fig = go.Figure()
    
    # Background bar
    fig.add_trace(go.Bar(x=[4], y=['Score'], orientation='h', marker=dict(color=['rgba(50,50,50,0.5)']), showlegend=False, hoverinfo='none'))
    
    # Score indicator
    normalized_score = score + 2
    color = get_score_color(score)
    
    fig.add_trace(go.Scatter(
        x=[normalized_score], y=['Score'], mode='markers',
        marker=dict(size=25, color=color, symbol='diamond', line=dict(width=2, color='white')),
        showlegend=False, hovertemplate=f"<b>Composite Score:</b> {score:.2f}<extra></extra>"
    ))
    
    zones = [(0, 0.5, '#ef4444'), (0.5, 1.5, '#f87171'), (1.5, 1.9, '#fbbf24'), (1.9, 2.1, '#888888'), (2.1, 2.5, '#86efac'), (2.5, 3.0, '#34d399'), (3.0, 4.0, '#10b981')]
    for start, end, zone_color in zones:
        fig.add_vrect(x0=start, x1=end, fillcolor=zone_color, opacity=0.15, line_width=0)
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#EAEAEA'}, height=100,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(range=[0, 4], tickvals=[0.25, 1, 1.7, 2, 2.3, 2.75, 3.5], ticktext=['Crisis', 'Bear', 'W.Bear', 'Chop', 'W.Bull', 'Bull', 'S.Bull'], tickfont=dict(size=10), showgrid=False),
        yaxis=dict(visible=False)
    )
    return fig


def create_factor_radar_chart(result: RegimeResult) -> go.Figure:
    """Create radar chart for factor analysis - compatible with adaptive factors"""
    factors = result.factors
    if not factors:
        return go.Figure()
    
    categories = []
    values = []
    for name, f in factors.items():
        categories.append(name.upper())
        # Support both old and new factor formats
        if hasattr(f, 'adaptive_score'):
            values.append((f.adaptive_score + 1) / 2 * 100)  # Map -1,1 to 0,100
        elif hasattr(f, 'score'):
            values.append((f.score + 2) / 4 * 100)  # Map -2,2 to 0,100
        else:
            values.append(50)
    
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', fillcolor='rgba(255,195,0,0.2)',
        line=dict(color='#FFC300', width=2), marker=dict(size=8, color='#FFC300'),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticktext=['Bear', '', 'Neutral', '', 'Bull'], gridcolor='rgba(42,42,42,0.5)', linecolor='rgba(42,42,42,0.5)', tickfont=dict(size=9, color='#888888')),
            angularaxis=dict(gridcolor='rgba(42,42,42,0.5)', linecolor='rgba(42,42,42,0.5)', tickfont=dict(size=10, color='#EAEAEA')),
            bgcolor='#1A1A1A'
        ),
        paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=60, r=60, t=30, b=30),
        font=dict(family='Inter', color='#EAEAEA'), showlegend=False
    )
    return fig


def create_factor_breakdown_chart(result: RegimeResult) -> go.Figure:
    """Create horizontal bar chart showing factor contributions - compatible with adaptive factors"""
    factors = result.factors
    if not factors:
        return go.Figure()
    
    names = []
    scores = []
    for name, f in factors.items():
        names.append(name.upper())
        if hasattr(f, 'adaptive_score'):
            scores.append(f.adaptive_score)
        elif hasattr(f, 'score'):
            scores.append(f.score)
        else:
            scores.append(0)
    
    colors = [get_score_color(s) for s in scores]
    
    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation='h', marker_color=colors,
        text=[f"{s:+.2f}" for s in scores], textposition='outside', textfont=dict(color='#EAEAEA'),
        hovertemplate="<b>%{y}</b><br>Score: %{x:+.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', font={'color': '#EAEAEA'}, height=300,
        margin=dict(l=100, r=50, t=30, b=30),
        xaxis=dict(range=[-1.5, 1.5], zeroline=True, zerolinecolor='#888', zerolinewidth=2, gridcolor='rgba(42,42,42,0.5)', title="Adaptive Score"),
        yaxis=dict(gridcolor='rgba(42,42,42,0.5)')
    )
    fig.add_vline(x=-0.5, line_dash="dash", line_color="#ef4444", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#10b981", opacity=0.5)
    return fig


def create_hmm_probability_chart(hmm_probs: dict) -> go.Figure:
    """Create horizontal bar chart for HMM state probabilities"""
    states = list(hmm_probs.keys())
    probs = [hmm_probs[s] * 100 for s in states]
    colors = ['#10b981' if s == 'BULL' else '#ef4444' if s == 'BEAR' else '#888888' for s in states]
    
    fig = go.Figure(go.Bar(
        x=probs, y=states, orientation='h', marker_color=colors,
        text=[f"{p:.1f}%" for p in probs], textposition='outside', textfont=dict(color='#EAEAEA', size=14),
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', font={'color': '#EAEAEA'}, height=200,
        margin=dict(l=80, r=60, t=20, b=20),
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor='rgba(42,42,42,0.5)', title="Probability %"),
        yaxis=dict(gridcolor='rgba(42,42,42,0.5)')
    )
    fig.add_vline(x=50, line_dash="dash", line_color="#FFC300", opacity=0.7)
    return fig


def create_breadth_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Create breadth analysis visualization"""
    rsi_bullish = (df['rsi latest'] > 50).sum()
    rsi_bearish = (df['rsi latest'] < 50).sum()
    osc_positive = (df['osc latest'] > 0).sum()
    osc_negative = (df['osc latest'] < 0).sum()
    
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]], subplot_titles=("RSI Distribution", "Oscillator Distribution"))
    
    fig.add_trace(go.Pie(
        labels=['Bullish (>50)', 'Bearish (<50)'], values=[rsi_bullish, rsi_bearish], hole=0.5,
        marker=dict(colors=['#10b981', '#ef4444']), textinfo='percent+label', textfont=dict(size=10, color='white'),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
    ), row=1, col=1)
    
    fig.add_trace(go.Pie(
        labels=['Positive', 'Negative'], values=[osc_positive, osc_negative], hole=0.5,
        marker=dict(colors=['#10b981', '#ef4444']), textinfo='percent+label', textfont=dict(size=10, color='white'),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
    ), row=1, col=2)
    
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=20, r=20, t=40, b=20), font=dict(family='Inter', color='#EAEAEA'), showlegend=False)
    return fig


def create_symbol_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create heatmap of symbol conditions"""
    symbols = df['symbol'].tolist()
    composite_scores = []
    for _, row in df.iterrows():
        rsi_score = (row['rsi latest'] - 50) / 25
        osc_score = row['osc latest'] / 50
        composite_scores.append((rsi_score + osc_score) / 2)
    
    sorted_indices = np.argsort(composite_scores)
    sorted_symbols = [symbols[i] for i in sorted_indices]
    sorted_scores = [composite_scores[i] for i in sorted_indices]
    
    n_cols = 6
    n_rows = int(np.ceil(len(sorted_symbols) / n_cols))
    while len(sorted_symbols) < n_cols * n_rows:
        sorted_symbols.append("")
        sorted_scores.append(0)
    
    symbols_grid = np.array(sorted_symbols).reshape(n_rows, n_cols)
    scores_grid = np.array(sorted_scores).reshape(n_rows, n_cols)
    normalized_scores = (scores_grid + 2) / 4
    
    colorscale = [[0, '#ef4444'], [0.25, '#f87171'], [0.5, '#888888'], [0.75, '#34d399'], [1, '#10b981']]
    
    fig = go.Figure(data=go.Heatmap(
        z=normalized_scores,
        text=[[f"{s}<br>{v:.2f}" if s else "" for s, v in zip(row_s, row_v)] for row_s, row_v in zip(symbols_grid, scores_grid)],
        texttemplate="%{text}", textfont=dict(size=10, color='white', family='Inter'),
        colorscale=colorscale, showscale=False, hovertemplate="<b>%{text}</b><extra></extra>", xgap=3, ygap=3
    ))
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=max(200, n_rows * 50),
        margin=dict(l=0, r=0, t=10, b=10), xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange='reversed'), font=dict(family='Inter')
    )
    return fig


def create_time_series_chart(ts_results: list) -> go.Figure:
    """Create time series chart for regime evolution"""
    dates = [r['date'] for r in ts_results]
    scores = [r['score'] for r in ts_results]
    regimes = [r['regime'] for r in ts_results]
    colors = [get_regime_color(r) for r in regimes]
    
    fig = go.Figure()
    
    # Fill areas
    fig.add_trace(go.Scatter(x=dates, y=[max(0, s) for s in scores], fill='tozeroy', fillcolor='rgba(16,185,129,0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=[min(0, s) for s in scores], fill='tozeroy', fillcolor='rgba(239,68,68,0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    
    # Main line
    fig.add_trace(go.Scatter(
        x=dates, y=scores, mode='lines+markers', name='Regime Score',
        line=dict(color='#FFC300', width=2), marker=dict(size=8, color=colors, line=dict(width=1, color='white')),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.2f}<extra></extra>"
    ))
    
    fig.add_hline(y=1.0, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'), annotation_text="Bull", annotation_position="right")
    fig.add_hline(y=-0.5, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'), annotation_text="Bear", annotation_position="right")
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1))
    fig.add_hrect(y0=1.0, y1=2.5, fillcolor='rgba(16,185,129,0.08)', line_width=0)
    fig.add_hrect(y0=-2.5, y1=-0.5, fillcolor='rgba(239,68,68,0.08)', line_width=0)
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=400,
        margin=dict(l=10, r=60, t=30, b=50), xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Regime Score', range=[-2, 2]),
        font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified', showlegend=False
    )
    return fig


def create_factor_evolution_chart(ts_results: list) -> go.Figure:
    """Create simplified factor score evolution - stacked area showing bull vs bear factors"""
    if not ts_results or 'factors' not in ts_results[0]:
        return go.Figure()
    
    dates = [r['date'] for r in ts_results]
    
    # Calculate aggregate bullish and bearish factor contributions
    bull_scores = []
    bear_scores = []
    
    for r in ts_results:
        factors = r.get('factors', {})
        bull_sum = sum(max(0, v) for v in factors.values())
        bear_sum = sum(min(0, v) for v in factors.values())
        bull_scores.append(bull_sum)
        bear_scores.append(abs(bear_sum))
    
    fig = go.Figure()
    
    # Bullish factors area
    fig.add_trace(go.Scatter(
        x=dates, y=bull_scores, mode='lines', name='Bullish Factors',
        fill='tozeroy', fillcolor='rgba(16,185,129,0.3)',
        line=dict(color='#10b981', width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Bullish Strength: %{y:.2f}<extra></extra>"
    ))
    
    # Bearish factors area (shown as negative)
    fig.add_trace(go.Scatter(
        x=dates, y=[-b for b in bear_scores], mode='lines', name='Bearish Factors',
        fill='tozeroy', fillcolor='rgba(239,68,68,0.3)',
        line=dict(color='#ef4444', width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Bearish Strength: %{y:.2f}<extra></extra>"
    ))
    
    # Net score line
    net_scores = [b - r for b, r in zip(bull_scores, bear_scores)]
    fig.add_trace(go.Scatter(
        x=dates, y=net_scores, mode='lines', name='Net Factor Score',
        line=dict(color='#FFC300', width=3),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Net Score: %{y:.2f}<extra></extra>"
    ))
    
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.5)', width=1, dash='dash'))
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=350,
        margin=dict(l=10, r=10, t=30, b=50), xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Factor Strength', zeroline=True, zerolinecolor='#888'),
        font=dict(family='Inter', color='#EAEAEA'), hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)')
    )
    return fig


def create_regime_distribution_chart(ts_results: list) -> go.Figure:
    """Create pie chart for regime distribution"""
    regime_counts = {}
    for r in ts_results:
        regime = r['regime']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    labels = list(regime_counts.keys())
    values = list(regime_counts.values())
    color_map = {
        'STRONG_BULL': '#10b981', 'BULL': '#34d399', 'WEAK_BULL': '#86efac', 
        'NEUTRAL': '#888888', 'CHOP': '#888888', 
        'WEAK_BEAR': '#fbbf24', 'BEAR': '#f87171', 'CRISIS': '#ef4444',
        'TRANSITION': '#a855f7'
    }
    colors = [color_map.get(l, '#888888') for l in labels]
    
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5, marker=dict(colors=colors, line=dict(color='#1A1A1A', width=2)),
        textinfo='label+percent', textfont=dict(size=11, color='white'),
        hovertemplate="<b>%{label}</b><br>Days: %{value}<br>%{percent}<extra></extra>"
    ))
    
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=20, r=20, t=30, b=20), font=dict(family='Inter', color='#EAEAEA'), showlegend=False)
    return fig


def create_regime_transition_matrix(ts_results: list) -> go.Figure:
    """Create regime transition probability matrix with readable dark-mode colors"""
    if len(ts_results) < 2:
        return go.Figure()
    
    transitions = {}
    regimes = list(set(r['regime'] for r in ts_results))
    # Sort regimes from bearish to bullish for better readability
    regime_order = ['CRISIS', 'BEAR', 'WEAK_BEAR', 'NEUTRAL', 'CHOP', 'WEAK_BULL', 'BULL', 'STRONG_BULL', 'TRANSITION']
    regimes = [r for r in regime_order if r in regimes]
    
    # Handle case with no valid regimes
    if not regimes:
        return go.Figure()
    
    for from_regime in regimes:
        transitions[from_regime] = {to_regime: 0 for to_regime in regimes}
    
    for i in range(len(ts_results) - 1):
        from_r = ts_results[i]['regime']
        to_r = ts_results[i + 1]['regime']
        if from_r in transitions and to_r in transitions[from_r]:
            transitions[from_r][to_r] += 1
    
    matrix = []
    for from_regime in regimes:
        row = []
        total = sum(transitions[from_regime].values())
        for to_regime in regimes:
            prob = transitions[from_regime][to_regime] / total if total > 0 else 0
            row.append(prob)
        matrix.append(row)
    
    # Custom colorscale: dark purple (low) -> gold (high) for dark mode readability
    colorscale = [
        [0.0, '#1a1a2e'],
        [0.2, '#16213e'],
        [0.4, '#0f3460'],
        [0.6, '#e94560'],
        [0.8, '#f59e0b'],
        [1.0, '#FFC300']
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=regimes, y=regimes, colorscale=colorscale,
        text=[[f"{v:.0%}" for v in row] for row in matrix], texttemplate="%{text}", 
        textfont=dict(size=12, color='white', family='Inter'),
        hovertemplate="<b>From:</b> %{y}<br><b>To:</b> %{x}<br><b>Probability:</b> %{z:.1%}<extra></extra>",
        colorbar=dict(
            title=dict(text="Probability", side="right", font=dict(color='#888888')),
            tickformat=".0%",
            tickfont=dict(color='#888888')
        )
    ))
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=380,
        margin=dict(l=100, r=80, t=40, b=100), 
        xaxis=dict(title="To Regime", tickangle=45, tickfont=dict(size=10), side='bottom'),
        yaxis=dict(title="From Regime", tickfont=dict(size=10), autorange='reversed'),
        font=dict(family='Inter', color='#EAEAEA')
    )
    return fig


def create_momentum_indicator(ts_results: list) -> go.Figure:
    """Create regime momentum/strength indicator"""
    if len(ts_results) < 5:
        return go.Figure()
    
    dates = [r['date'] for r in ts_results]
    scores = [r['score'] for r in ts_results]
    
    momentum = [0] * 3
    for i in range(3, len(scores)):
        momentum.append(scores[i] - scores[i-3])
    
    acceleration = [0] * 3
    for i in range(3, len(momentum)):
        acceleration.append(momentum[i] - momentum[i-3])
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Regime Momentum (3-day ROC)", "Regime Acceleration"))
    
    mom_colors = ['#10b981' if m > 0 else '#ef4444' for m in momentum]
    fig.add_trace(go.Bar(x=dates, y=momentum, marker_color=mom_colors, name='Momentum', hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Momentum: %{y:.3f}<extra></extra>"), row=1, col=1)
    
    acc_colors = ['#10b981' if a > 0 else '#ef4444' for a in acceleration]
    fig.add_trace(go.Bar(x=dates, y=acceleration, marker_color=acc_colors, name='Acceleration', hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Acceleration: %{y:.3f}<extra></extra>"), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=400,
        margin=dict(l=10, r=10, t=40, b=30), font=dict(family='Inter', color='#EAEAEA'), showlegend=False, hovermode='x unified'
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(42,42,42,0.5)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(42,42,42,0.5)')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def display_single_day_result(result: RegimeResult, latest_df: pd.DataFrame = None):
    """Display single day regime detection result with adaptive metrics"""
    regime_class = get_regime_color_class(result.regime)
    detector = MarketRegimeDetector()
    emoji = detector.get_regime_emoji(result.regime)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f"""
        <div class='metric-card {"success" if regime_class == "bull" else "danger" if regime_class == "bear" else "warning"}'>
            <h4>Detected Regime</h4>
            <h2>{emoji} {result.regime_name}</h2>
            <div class='sub-metric'>Analysis: {result.analysis_date}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Suggested Portfolio Mix</h4>
            <h2 style='font-size: 1.4rem;'>{result.suggested_mix}</h2>
            <div class='sub-metric'>Based on adaptive analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence_pct = result.confidence * 100
        conf_class = "success" if confidence_pct >= 75 else "warning" if confidence_pct >= 50 else "neutral"
        st.markdown(f"""
        <div class='metric-card {conf_class}'>
            <h4>Bayesian Confidence</h4>
            <h2>{confidence_pct:.0f}%</h2>
            <div class='sub-metric'>Score: {result.composite_score:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row: HMM and Volatility info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bull_prob = result.hmm_probabilities.get('BULL', 0) * 100
        st.markdown(f"""
        <div class='metric-card {"success" if bull_prob > 50 else "neutral"}'>
            <h4>P(Bull)</h4>
            <h2>{bull_prob:.0f}%</h2>
            <div class='sub-metric'>HMM State</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        bear_prob = result.hmm_probabilities.get('BEAR', 0) * 100
        st.markdown(f"""
        <div class='metric-card {"danger" if bear_prob > 50 else "neutral"}'>
            <h4>P(Bear)</h4>
            <h2>{bear_prob:.0f}%</h2>
            <div class='sub-metric'>HMM State</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        vol_class = "danger" if result.volatility_regime in ["HIGH", "EXTREME"] else "success" if result.volatility_regime == "LOW" else "neutral"
        st.markdown(f"""
        <div class='metric-card {vol_class}'>
            <h4>Volatility Regime</h4>
            <h2>{result.volatility_regime}</h2>
            <div class='sub-metric'>Multiplier: {result.volatility_multiplier:.2f}x</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card primary'>
            <h4>Regime Persistence</h4>
            <h2>{result.regime_persistence:.0f}</h2>
            <div class='sub-metric'>Periods in current state</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Warnings
    if result.warnings:
        for warning in result.warnings:
            st.markdown(f"<div class='warning-box'>⚠️ <strong>Warning:</strong> {warning}</div>", unsafe_allow_html=True)
    
    if result.change_point_detected:
        st.markdown("""
        <div class='warning-box' style='background: rgba(239,68,68,0.2); border-color: #ef4444;'>
            ⚡ <strong>STRUCTURAL BREAK DETECTED</strong> - Market conditions have fundamentally shifted
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🎯 Adaptive Factors", "🔮 HMM Analysis", "📈 Breadth", "🗺️ Heatmap"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("##### Regime Strength Gauge")
            st.plotly_chart(create_regime_gauge(result.composite_score, result.confidence), width="stretch")
        with col2:
            st.markdown("##### Score Position (Adaptive)")
            st.plotly_chart(create_composite_score_chart(result.composite_score), width="stretch")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Analysis Summary")
            # Format explanation nicely
            explanation_html = result.explanation.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
            st.markdown(f"<div class='info-box' style='font-family: monospace; font-size: 0.85rem;'>{explanation_html}</div>", unsafe_allow_html=True)
    
    with tab2:
        if result.factors:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("##### Factor Radar (Percentile-Based)")
                st.plotly_chart(create_factor_radar_chart(result), width="stretch")
            with col2:
                st.markdown("##### Factor Contributions (Adaptive)")
                st.plotly_chart(create_factor_breakdown_chart(result), width="stretch")
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("##### Adaptive Factor Details")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">All scores are percentile-based - NO fixed thresholds</p>', unsafe_allow_html=True)
            
            cols = st.columns(3)
            for i, (name, factor) in enumerate(result.factors.items()):
                with cols[i % 3]:
                    color_class = "success" if factor.adaptive_score >= 0.3 else "danger" if factor.adaptive_score <= -0.3 else "warning"
                    st.markdown(f"""
                    <div class='metric-card {color_class}'>
                        <h4>{name.upper()}</h4>
                        <h2>{factor.adaptive_score:+.2f}</h2>
                        <div class='sub-metric'>
                            Percentile: {factor.percentile_rank:.0%} | Z: {factor.z_score:+.1f}<br>
                            {factor.classification}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Factor data not available.")
    
    with tab3:
        st.markdown("##### Hidden Markov Model State Probabilities")
        st.markdown('<p style="color: #888888; font-size: 0.85rem;">HMM learns regime patterns from data - states discovered adaptively</p>', unsafe_allow_html=True)
        
        # HMM probability bars
        hmm_fig = create_hmm_probability_chart(result.hmm_probabilities)
        st.plotly_chart(hmm_fig, width="stretch")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### How Adaptive Detection Works")
            st.markdown("""
            <div class='info-box'>
                <p style='font-size: 0.85rem; line-height: 1.8;'>
                <strong>1. Percentile Normalization</strong><br>
                All indicators scored relative to rolling history, not fixed thresholds.<br><br>
                <strong>2. Hidden Markov Model</strong><br>
                Learns regime patterns and transition probabilities from data.<br><br>
                <strong>3. Kalman Filter</strong><br>
                Smooths scores adaptively, reduces noise while preserving signals.<br><br>
                <strong>4. GARCH Volatility</strong><br>
                Adjusts sensitivity based on current volatility regime.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("##### Regime Transition Matrix (Learned)")
            st.markdown("""
            <div class='info-box'>
                <p style='font-size: 0.85rem; line-height: 1.8;'>
                The HMM continuously learns transition probabilities:<br><br>
                • <strong>High diagonal</strong> = Regime persistence<br>
                • <strong>Off-diagonal</strong> = Transition patterns<br><br>
                Current regime has persisted for <strong>{}</strong> periods.
                </p>
            </div>
            """.format(int(result.regime_persistence)), unsafe_allow_html=True)
    
    with tab4:
        if latest_df is not None and not latest_df.empty:
            st.markdown("##### Market Breadth Analysis")
            st.plotly_chart(create_breadth_analysis_chart(latest_df), width="stretch")
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            total = len(latest_df)
            rsi_bull = (latest_df['rsi latest'] > 50).sum()
            osc_pos = (latest_df['osc latest'] > 0).sum()
            above_200 = (latest_df['price'] > latest_df['ma200 latest']).sum() if 'ma200 latest' in latest_df.columns else 0
            avg_rsi = latest_df['rsi latest'].mean()
            
            with col1:
                st.markdown(f"<div class='metric-card success'><h4>RSI Bullish</h4><h2>{rsi_bull/total*100:.0f}%</h2><div class='sub-metric'>{rsi_bull} of {total}</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card {'success' if osc_pos > total/2 else 'danger'}'><h4>Oscillator +ve</h4><h2>{osc_pos/total*100:.0f}%</h2><div class='sub-metric'>{osc_pos} of {total}</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card {'success' if above_200/total > 0.5 else 'danger'}'><h4>Above 200 DMA</h4><h2>{above_200/total*100:.0f}%</h2><div class='sub-metric'>{above_200} of {total}</div></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='metric-card {'success' if avg_rsi > 55 else 'danger' if avg_rsi < 45 else 'warning'}'><h4>Avg RSI</h4><h2>{avg_rsi:.1f}</h2><div class='sub-metric'>Universe avg</div></div>", unsafe_allow_html=True)
        else:
            st.info("Breadth data not available.")
    
    with tab5:
        if latest_df is not None and not latest_df.empty:
            st.markdown("##### Symbol Condition Heatmap")
            st.markdown('<p style="color: #888888; font-size: 0.85rem;">Sorted by composite score: Green = Bullish | Red = Bearish</p>', unsafe_allow_html=True)
            st.plotly_chart(create_symbol_heatmap(latest_df), width="stretch")
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🟢 Top Bullish Symbols")
                top_df = latest_df.nlargest(10, 'rsi latest')[['symbol', 'rsi latest', 'osc latest', '% change']].copy()
                top_df.columns = ['Symbol', 'RSI', 'Oscillator', '% Change']
                top_df['RSI'] = top_df['RSI'].round(1)
                top_df['Oscillator'] = top_df['Oscillator'].round(1)
                top_df['% Change'] = top_df['% Change'].round(2)
                st.dataframe(top_df, width="stretch", hide_index=True)
            with col2:
                st.markdown("##### 🔴 Top Bearish Symbols")
                bottom_df = latest_df.nsmallest(10, 'rsi latest')[['symbol', 'rsi latest', 'osc latest', '% change']].copy()
                bottom_df.columns = ['Symbol', 'RSI', 'Oscillator', '% Change']
                bottom_df['RSI'] = bottom_df['RSI'].round(1)
                bottom_df['Oscillator'] = bottom_df['Oscillator'].round(1)
                bottom_df['% Change'] = bottom_df['% Change'].round(2)
                st.dataframe(bottom_df, width="stretch", hide_index=True)
        else:
            st.info("Symbol data not available.")


def display_time_series_results(ts_results: list):
    """Display time series regime analysis results"""
    total_days = len(ts_results)
    bull_days = sum(1 for r in ts_results if r['regime'] in ['STRONG_BULL', 'BULL'])
    bear_days = sum(1 for r in ts_results if r['regime'] in ['BEAR', 'CRISIS'])
    avg_score = np.mean([r['score'] for r in ts_results])
    
    recent_scores = [r['score'] for r in ts_results[-5:]] if len(ts_results) >= 5 else [r['score'] for r in ts_results]
    momentum = recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"<div class='metric-card neutral'><h4>Days Analyzed</h4><h2>{total_days}</h2><div class='sub-metric'>Trading Days</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card success'><h4>Bull Days</h4><h2>{bull_days}</h2><div class='sub-metric'>{bull_days/total_days*100:.1f}%</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card danger'><h4>Bear Days</h4><h2>{bear_days}</h2><div class='sub-metric'>{bear_days/total_days*100:.1f}%</div></div>", unsafe_allow_html=True)
    with col4:
        score_class = "success" if avg_score > 0.5 else "danger" if avg_score < -0.5 else "warning"
        st.markdown(f"<div class='metric-card {score_class}'><h4>Avg Score</h4><h2>{avg_score:+.2f}</h2><div class='sub-metric'>Mean</div></div>", unsafe_allow_html=True)
    with col5:
        mom_class = "success" if momentum > 0.1 else "danger" if momentum < -0.1 else "neutral"
        mom_arrow = "▲" if momentum > 0 else "▼" if momentum < 0 else "→"
        st.markdown(f"<div class='metric-card {mom_class}'><h4>Momentum</h4><h2>{mom_arrow} {abs(momentum):.2f}</h2><div class='sub-metric'>5-day</div></div>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Regime Evolution", "🔄 Factor Evolution", "📊 Statistics", "🔀 Transitions", "📋 Data"])
    
    with tab1:
        st.markdown("##### Regime Score Over Time")
        st.plotly_chart(create_time_series_chart(ts_results), width="stretch")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("##### Regime Momentum & Acceleration")
        st.plotly_chart(create_momentum_indicator(ts_results), width="stretch")
    
    with tab2:
        st.markdown("##### Factor Strength Over Time")
        st.markdown('<p style="color: #888888; font-size: 0.85rem;">Green area = Bullish factor strength | Red area = Bearish factor strength | Gold line = Net balance</p>', unsafe_allow_html=True)
        st.plotly_chart(create_factor_evolution_chart(ts_results), width="stretch")
        
        # Add factor summary metrics
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("##### Factor Contribution Summary")
        
        # Calculate average factor scores
        factor_avgs = {}
        if ts_results and ts_results[0].get('factors'):
            for factor_name in ts_results[0].get('factors', {}).keys():
                values = [r.get('factors', {}).get(factor_name, 0) for r in ts_results]
                factor_avgs[factor_name] = np.mean(values)
        
        if factor_avgs:
            cols = st.columns(min(len(factor_avgs), 6))  # Max 6 columns
            for i, (name, avg) in enumerate(factor_avgs.items()):
                with cols[i % len(cols)]:
                    color_class = "success" if avg >= 0.3 else "danger" if avg <= -0.3 else "warning"
                    arrow = "▲" if avg > 0 else "▼" if avg < 0 else "→"
                    st.markdown(f"""
                    <div class='metric-card {color_class}' style='padding: 0.75rem; text-align: center;'>
                        <h4 style='font-size: 0.65rem;'>{name.upper()}</h4>
                        <h2 style='font-size: 1.25rem;'>{arrow} {avg:+.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Factor data not available for summary.")
    
    with tab3:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("##### Regime Distribution")
            st.plotly_chart(create_regime_distribution_chart(ts_results), width="stretch")
        with col_d2:
            st.markdown("##### Regime Statistics")
            regime_stats = {}
            for r in ts_results:
                regime_stats[r['regime']] = regime_stats.get(r['regime'], 0) + 1
            stats_df = pd.DataFrame([{"Regime": k, "Days": v, "Percentage": f"{v/total_days*100:.1f}%"} for k, v in sorted(regime_stats.items(), key=lambda x: -x[1])])
            st.dataframe(stats_df, width="stretch", hide_index=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Score Statistics")
            scores = [r['score'] for r in ts_results]
            score_stats = pd.DataFrame([
                {"Metric": "Mean", "Value": f"{np.mean(scores):.2f}"},
                {"Metric": "Median", "Value": f"{np.median(scores):.2f}"},
                {"Metric": "Std Dev", "Value": f"{np.std(scores):.2f}"},
                {"Metric": "Min", "Value": f"{np.min(scores):.2f}"},
                {"Metric": "Max", "Value": f"{np.max(scores):.2f}"},
            ])
            st.dataframe(score_stats, width="stretch", hide_index=True)
    
    with tab4:
        st.markdown("##### Regime Transition Probability Matrix")
        st.plotly_chart(create_regime_transition_matrix(ts_results), width="stretch")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("##### Regime Duration Analysis")
        
        streaks = []
        current_regime = ts_results[0]['regime']
        current_streak = 1
        for i in range(1, len(ts_results)):
            if ts_results[i]['regime'] == current_regime:
                current_streak += 1
            else:
                streaks.append({'regime': current_regime, 'duration': current_streak})
                current_regime = ts_results[i]['regime']
                current_streak = 1
        streaks.append({'regime': current_regime, 'duration': current_streak})
        
        duration_stats = {}
        for s in streaks:
            if s['regime'] not in duration_stats:
                duration_stats[s['regime']] = []
            duration_stats[s['regime']].append(s['duration'])
        
        duration_df = pd.DataFrame([{"Regime": regime, "Occurrences": len(durations), "Avg Duration": f"{np.mean(durations):.1f} days", "Max Duration": f"{max(durations)} days"} for regime, durations in duration_stats.items()])
        st.dataframe(duration_df, width="stretch", hide_index=True)
    
    with tab5:
        st.markdown(f"##### Daily Regime Data ({len(ts_results)} trading days)")
        display_df = pd.DataFrame(ts_results)
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        display_df['score'] = display_df['score'].round(2)
        display_df['confidence'] = (display_df['confidence'] * 100).round(0).astype(int).astype(str) + '%'
        display_cols = ['date', 'regime', 'score', 'confidence', 'mix']
        display_df = display_df[[c for c in display_cols if c in display_df.columns]]
        display_df.columns = ['Date', 'Regime', 'Score', 'Confidence', 'Suggested Mix'][:len(display_df.columns)]
        st.dataframe(display_df, width="stretch", hide_index=True, height=400)
        
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Regime Time Series (CSV)", data=csv_data, file_name="avastha_regime_timeseries.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">AVASTHA</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">Market Regime Detection</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">🎯 Universe Selection</div>', unsafe_allow_html=True)
        universe_type = st.selectbox("Analysis Universe", UNIVERSE_OPTIONS, help="Choose the stock/ETF universe")
        
        selected_index = None
        if universe_type == "Index Constituents":
            selected_index = st.selectbox("Select Index", INDEX_LIST, index=INDEX_LIST.index("NIFTY 500"))
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 Analysis Type</div>', unsafe_allow_html=True)
        analysis_mode = st.radio("Select Mode", ["📅 Single Day", "📈 Time Series"], label_visibility="collapsed")
        
        single_date, start_date, end_date = None, None, None
        if "Single" in analysis_mode:
            st.markdown('<div class="sidebar-title">📅 Analysis Date</div>', unsafe_allow_html=True)
            single_date = st.date_input("Select Date", dt.date.today() - timedelta(days=1), max_value=dt.date.today())
        else:
            st.markdown('<div class="sidebar-title">📅 Date Range</div>', unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_date = st.date_input("Start Date", dt.date.today() - timedelta(days=60), max_value=dt.date.today())
            with col_d2:
                end_date = st.date_input("End Date", dt.date.today() - timedelta(days=1), max_value=dt.date.today())
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        run_analysis = st.button("◈ RUN ANALYSIS", type="primary", use_container_width=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"<div class='info-box'><p style='font-size: 0.8rem; margin: 0; color: var(--text-muted);'><strong>Version:</strong> {VERSION}<br><strong>Engine:</strong> Multi-Factor<br><strong>Data:</strong> yfinance</p></div>", unsafe_allow_html=True)
        
        return universe_type, selected_index, analysis_mode, single_date, start_date, end_date, run_analysis


def render_header():
    st.markdown(f"""
    <div class="premium-header">
        <h1>{APP_TITLE} : {APP_SUBTITLE}</h1>
        <div class="tagline">Institutional-grade regime detection using multi-factor analysis</div>
    </div>
    """, unsafe_allow_html=True)


def run_home_page():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px;'>
            <h3 style='color: var(--primary-color); margin-bottom: 1rem;'>🎯 Adaptive Analysis</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>NO fixed thresholds - all scoring is percentile-based relative to rolling history.</p>
            <br><p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Methods:</strong><br>
                • Rolling Percentile Normalization<br>
                • Kalman Filter Smoothing<br>
                • GARCH Volatility Adjustment
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card success' style='min-height: 280px;'>
            <h3 style='color: var(--success-green); margin-bottom: 1rem;'>🔮 Hidden Markov Model</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>HMM learns regime patterns and transition probabilities from data.</p>
            <br><p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Features:</strong><br>
                • State Discovery (Bull/Neutral/Bear)<br>
                • Transition Probabilities<br>
                • Regime Persistence Tracking
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 1rem;'>⚡ Change Detection</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>CUSUM algorithm detects structural breaks in market conditions.</p>
            <br><p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Includes:</strong><br>
                • Structural Break Detection<br>
                • Volatility Regime Classification<br>
                • Bayesian Confidence Scoring
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 Why Adaptive?</h4>
        <p style='line-height: 1.8;'>Traditional regime detection uses fixed thresholds (e.g., "RSI > 70 = overbought") which fail to adapt to changing market conditions. 
        AVASTHA uses <strong>rolling percentile normalization</strong> - every indicator is scored relative to its own recent history, 
        making the system automatically sensitive to microstructure changes without manual threshold tuning.</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    universe_type, selected_index, analysis_mode, single_date, start_date, end_date, run_analysis = render_sidebar()
    render_header()
    
    if run_analysis:
        engine = MarketDataEngine()
        detector = MarketRegimeDetector()
        
        progress = st.progress(0, text="Initializing...")
        universe_msg = engine.set_universe(universe_type, selected_index)
        st.toast(universe_msg, icon="✅" if "✓" in universe_msg else "⚠️")
        
        if "Single" in analysis_mode:
            analysis_dt = datetime.combine(single_date, datetime.min.time())
            
            def update_progress(p, msg):
                progress.progress(p, text=msg)
            
            try:
                historical_data = engine.get_regime_data(analysis_dt, progress_callback=update_progress)
                if not historical_data:
                    st.error("Failed to fetch market data.")
                    progress.empty()
                    return
                
                result = detector.detect(historical_data)
                latest_df = historical_data[-1][1] if historical_data else None
                
                progress.empty()
                st.session_state.regime_result = result
                st.session_state.latest_df = latest_df
                st.session_state.time_series_results = None
                st.success("✅ Regime analysis completed!")
            except Exception as e:
                progress.empty()
                st.error(f"Analysis failed: {str(e)}")
                return
        else:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())
            
            def update_progress(p, msg):
                progress.progress(p * 0.7, text=msg)
            
            try:
                historical_data = engine.get_time_series_regime_data(start_dt, end_dt, progress_callback=update_progress)
                if not historical_data:
                    st.error("Failed to fetch market data.")
                    progress.empty()
                    return
                
                ts_results = []
                total_days = len(historical_data)
                
                for i in range(10, len(historical_data)):
                    window = historical_data[i-10:i+1]
                    result = detector.detect(window)
                    
                    # Extract factor scores - handle both old and new format
                    factor_scores = {}
                    for name, f in result.factors.items():
                        if hasattr(f, 'adaptive_score'):
                            factor_scores[name] = f.adaptive_score
                        elif hasattr(f, 'score'):
                            factor_scores[name] = f.score
                        else:
                            factor_scores[name] = 0
                    
                    ts_results.append({
                        'date': historical_data[i][0], 
                        'regime': result.regime_name, 
                        'score': result.composite_score,
                        'confidence': result.confidence, 
                        'mix': result.suggested_mix, 
                        'factors': factor_scores,
                        'hmm_probs': getattr(result, 'hmm_probabilities', {}),
                        'vol_regime': getattr(result, 'volatility_regime', 'NORMAL'),
                        'change_point': getattr(result, 'change_point_detected', False)
                    })
                    if i % 5 == 0:
                        progress.progress(0.7 + 0.3 * (i / total_days), text=f"Analyzing day {i-9}/{total_days-10}...")
                
                progress.empty()
                st.session_state.time_series_results = ts_results
                st.session_state.regime_result = None
                st.session_state.latest_df = None
                st.success(f"✅ Time series completed! ({len(ts_results)} days)")
            except Exception as e:
                progress.empty()
                st.error(f"Analysis failed: {str(e)}")
                return
    
    if st.session_state.regime_result is not None:
        display_single_day_result(st.session_state.regime_result, st.session_state.latest_df)
    elif st.session_state.time_series_results is not None:
        display_time_series_results(st.session_state.time_series_results)
    else:
        run_home_page()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© {datetime.now().year} AVASTHA | Hemrek Capital | {VERSION}")


if __name__ == "__main__":
    main()
