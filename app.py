"""
AVASTHA - Market Regime Detection System
A Pragyam Product Family Member

Institutional-grade market regime detection using multi-factor analysis
across momentum, trend, breadth, volatility, and statistical extremes.

Version: 1.0.0
Author: Hemrek Capital
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import datetime as dt
import logging
import time

# Local imports
from regime_detector import MarketRegimeDetector, RegimeType, RegimeResult
from data_engine import (
    MarketDataEngine, 
    get_universe_symbols,
    UNIVERSE_OPTIONS, 
    INDEX_LIST, 
    ETF_UNIVERSE,
    get_display_name
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "v1.0.0"
APP_TITLE = "AVASTHA"
APP_SUBTITLE = "Market Regime Detection System"

st.set_page_config(
    page_title=f"{APP_TITLE} | {APP_SUBTITLE}",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# PRAGYAM DESIGN SYSTEM CSS
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
    .premium-header .product-badge { display: inline-block; background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    
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
    
    .regime-card {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .regime-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
    .regime-card.bull::before { background: var(--success-green); }
    .regime-card.bear::before { background: var(--danger-red); }
    .regime-card.chop::before { background: var(--warning-amber); }
    
    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.bull { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.bear { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.chop { background: rgba(245, 158, 11, 0.15); color: var(--warning-amber); border: 1px solid rgba(245, 158, 11, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); border-left: 0px solid var(--primary-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .warning-box { background: rgba(245, 158, 11, 0.1); border: 1px solid var(--warning-amber); border-radius: 10px; padding: 1rem; margin: 1rem 0; }
    
    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    .stButton>button:active { transform: translateY(0); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_color_class(regime: RegimeType) -> str:
    """Get CSS class for regime coloring"""
    bull_regimes = [RegimeType.STRONG_BULL, RegimeType.BULL]
    bear_regimes = [RegimeType.BEAR, RegimeType.CRISIS]
    
    if regime in bull_regimes:
        return "bull"
    elif regime in bear_regimes:
        return "bear"
    else:
        return "chop"


def get_regime_color(regime: RegimeType) -> str:
    """Get hex color for regime"""
    bull_regimes = [RegimeType.STRONG_BULL, RegimeType.BULL]
    bear_regimes = [RegimeType.BEAR, RegimeType.CRISIS]
    
    if regime in bull_regimes:
        return "#10b981"
    elif regime in bear_regimes:
        return "#ef4444"
    else:
        return "#f59e0b"


# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_composite_score_chart(score: float) -> go.Figure:
    """Create visualization for composite regime score"""
    fig = go.Figure()
    
    # Background bar
    fig.add_trace(go.Bar(
        x=[4], y=['Score'], orientation='h',
        marker=dict(color=['rgba(50,50,50,0.5)']),
        showlegend=False, hoverinfo='none'
    ))
    
    # Score indicator
    normalized_score = score + 2  # Shift to 0-4 range
    color = "#10b981" if score >= 0.5 else "#ef4444" if score <= -0.5 else "#f59e0b"
    
    fig.add_trace(go.Scatter(
        x=[normalized_score], y=['Score'],
        mode='markers',
        marker=dict(size=25, color=color, symbol='diamond', line=dict(width=2, color='white')),
        showlegend=False,
        hovertemplate=f"<b>Composite Score:</b> {score:.2f}<extra></extra>"
    ))
    
    # Regime zone backgrounds
    zones = [
        (0, 0.5, '#ef4444'), (0.5, 1.5, '#f87171'), (1.5, 1.9, '#fbbf24'),
        (1.9, 2.1, '#888888'), (2.1, 2.5, '#86efac'), (2.5, 3.0, '#34d399'), (3.0, 4.0, '#10b981')
    ]
    
    for start, end, zone_color in zones:
        fig.add_vrect(x0=start, x1=end, fillcolor=zone_color, opacity=0.15, line_width=0)
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#EAEAEA'}, height=100,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(
            range=[0, 4],
            tickvals=[0.25, 1, 1.7, 2, 2.3, 2.75, 3.5],
            ticktext=['Crisis', 'Bear', 'W.Bear', 'Chop', 'W.Bull', 'Bull', 'S.Bull'],
            tickfont=dict(size=10), showgrid=False
        ),
        yaxis=dict(visible=False)
    )
    
    return fig


def create_factor_breakdown_chart(result: RegimeResult) -> go.Figure:
    """Create horizontal bar chart showing factor contributions"""
    factors = result.factors
    
    names = []
    scores = []
    colors = []
    
    for name, factor in factors.items():
        names.append(name.upper())
        scores.append(factor.score)
        if factor.score >= 0.5:
            colors.append('#10b981')
        elif factor.score <= -0.5:
            colors.append('#ef4444')
        else:
            colors.append('#f59e0b')
    
    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation='h',
        marker_color=colors,
        text=[f"{s:+.2f}" for s in scores],
        textposition='outside',
        textfont=dict(color='#EAEAEA')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A',
        font={'color': '#EAEAEA'}, height=300,
        margin=dict(l=100, r=50, t=30, b=30),
        xaxis=dict(
            range=[-2.5, 2.5], zeroline=True, zerolinecolor='#888', zerolinewidth=2,
            gridcolor='rgba(42,42,42,0.5)', title="Factor Score"
        ),
        yaxis=dict(gridcolor='rgba(42,42,42,0.5)')
    )
    
    fig.add_vline(x=-1, line_dash="dash", line_color="#ef4444", opacity=0.5)
    fig.add_vline(x=1, line_dash="dash", line_color="#10b981", opacity=0.5)
    
    return fig


def create_time_series_chart(ts_results: list) -> go.Figure:
    """Create time series chart for regime evolution"""
    dates = [r['date'] for r in ts_results]
    scores = [r['score'] for r in ts_results]
    regimes = [r['regime'] for r in ts_results]
    
    colors = [get_regime_color(RegimeType[r]) for r in regimes]
    
    fig = go.Figure()
    
    # Fill areas
    fig.add_trace(go.Scatter(
        x=dates, y=[max(0, s) for s in scores],
        fill='tozeroy', fillcolor='rgba(16,185,129,0.15)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=[min(0, s) for s in scores],
        fill='tozeroy', fillcolor='rgba(239,68,68,0.15)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    # Main line with markers
    fig.add_trace(go.Scatter(
        x=dates, y=scores,
        mode='lines+markers',
        line=dict(color='#FFC300', width=2),
        marker=dict(size=8, color=colors, line=dict(width=1, color='white')),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>"
    ))
    
    # Threshold lines
    fig.add_hline(y=1.0, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
    fig.add_hline(y=-0.5, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1))
    
    # Regime zones
    fig.add_hrect(y0=1.0, y1=2.5, fillcolor='rgba(16,185,129,0.08)', line_width=0)
    fig.add_hrect(y0=-2.5, y1=-0.5, fillcolor='rgba(239,68,68,0.08)', line_width=0)
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A',
        height=400, margin=dict(l=10, r=10, t=30, b=50),
        xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', title='Regime Score', range=[-2, 2]),
        font=dict(family='Inter', color='#EAEAEA'),
        hovermode='x unified', showlegend=False
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
        'CHOP': '#888888',
        'WEAK_BEAR': '#fbbf24', 'BEAR': '#f87171', 'CRISIS': '#ef4444'
    }
    colors = [color_map.get(l, '#888888') for l in labels]
    
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='#1A1A1A', width=2)),
        textinfo='label+percent',
        textfont=dict(size=11, color='white')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family='Inter', color='#EAEAEA'),
        showlegend=False
    )
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def display_single_day_result(result: RegimeResult):
    """Display single day regime detection result"""
    regime_class = get_regime_color_class(result.regime)
    detector = MarketRegimeDetector()
    emoji = detector.get_regime_emoji(result.regime)
    
    # Main regime display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f"""
        <div class='metric-card {regime_class}'>
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
            <div class='sub-metric'>Based on multi-factor analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence_pct = result.confidence * 100
        conf_class = "success" if confidence_pct >= 75 else "warning" if confidence_pct >= 60 else "neutral"
        st.markdown(f"""
        <div class='metric-card {conf_class}'>
            <h4>Confidence</h4>
            <h2>{confidence_pct:.0f}%</h2>
            <div class='sub-metric'>Score: {result.composite_score:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Warnings
    if result.warnings:
        for warning in result.warnings:
            st.markdown(f"""
            <div class='warning-box'>
                ⚠️ <strong>Warning:</strong> {warning}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Composite score visualization
    st.markdown("##### Regime Score Position")
    score_chart = create_composite_score_chart(result.composite_score)
    st.plotly_chart(score_chart, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Factor breakdown
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### Factor Contributions")
        breakdown_chart = create_factor_breakdown_chart(result)
        st.plotly_chart(breakdown_chart, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("##### Factor Details")
        for name, factor in result.factors.items():
            score_color = "🟢" if factor.score >= 0.5 else "🔴" if factor.score <= -0.5 else "🟡"
            with st.expander(f"{score_color} **{name.upper()}** — {factor.classification}", expanded=False):
                st.write(f"**Score:** {factor.score:+.2f}")
                if factor.metrics:
                    st.write("**Metrics:**")
                    for k, v in factor.metrics.items():
                        if isinstance(v, float):
                            st.write(f"- {k}: {v:.3f}")
                        else:
                            st.write(f"- {k}: {v}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Full explanation
    st.markdown("##### Analysis Summary")
    st.markdown(f"""
    <div class='info-box'>
        {result.explanation.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)


def display_time_series_results(ts_results: list):
    """Display time series regime analysis results"""
    
    # Summary metrics
    total_days = len(ts_results)
    bull_days = sum(1 for r in ts_results if r['regime'] in ['STRONG_BULL', 'BULL'])
    bear_days = sum(1 for r in ts_results if r['regime'] in ['BEAR', 'CRISIS'])
    chop_days = total_days - bull_days - bear_days
    avg_score = np.mean([r['score'] for r in ts_results])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card neutral'>
            <h4>Days Analyzed</h4>
            <h2>{total_days}</h2>
            <div class='sub-metric'>Trading Days</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card success'>
            <h4>Bull Days</h4>
            <h2>{bull_days}</h2>
            <div class='sub-metric'>{bull_days/total_days*100:.1f}% of period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card danger'>
            <h4>Bear Days</h4>
            <h2>{bear_days}</h2>
            <div class='sub-metric'>{bear_days/total_days*100:.1f}% of period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        score_class = "success" if avg_score > 0.5 else "danger" if avg_score < -0.5 else "warning"
        st.markdown(f"""
        <div class='metric-card {score_class}'>
            <h4>Avg Score</h4>
            <h2>{avg_score:+.2f}</h2>
            <div class='sub-metric'>Mean Regime Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Regime Evolution", "📊 Distribution", "📋 Data Table"])
    
    with tab1:
        st.markdown("##### Regime Score Over Time")
        st.markdown('<p style="color: #888888; font-size: 0.85rem;">Green zone = Bullish | Red zone = Bearish | Yellow line = Threshold</p>', unsafe_allow_html=True)
        
        fig = create_time_series_chart(ts_results)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with tab2:
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("##### Regime Distribution")
            dist_chart = create_regime_distribution_chart(ts_results)
            st.plotly_chart(dist_chart, use_container_width=True, config={'displayModeBar': False})
        
        with col_d2:
            st.markdown("##### Regime Statistics")
            regime_stats = {}
            for r in ts_results:
                regime = r['regime']
                regime_stats[regime] = regime_stats.get(regime, 0) + 1
            
            stats_df = pd.DataFrame([
                {"Regime": k, "Days": v, "Percentage": f"{v/total_days*100:.1f}%"}
                for k, v in sorted(regime_stats.items(), key=lambda x: -x[1])
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
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
            st.dataframe(score_stats, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown(f"##### Daily Regime Data ({len(ts_results)} trading days)")
        
        display_df = pd.DataFrame(ts_results)
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        display_df['score'] = display_df['score'].round(2)
        display_df['confidence'] = (display_df['confidence'] * 100).round(0).astype(int).astype(str) + '%'
        display_df = display_df[['date', 'regime', 'score', 'confidence', 'mix']]
        display_df.columns = ['Date', 'Regime', 'Score', 'Confidence', 'Suggested Mix']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
        
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Regime Time Series (CSV)",
            data=csv_data,
            file_name=f"avastha_regime_timeseries.csv",
            mime="text/csv"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render sidebar with all controls"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">AVASTHA</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">Market Regime Detection</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Universe Selection
        st.markdown('<div class="sidebar-title">🎯 Universe Selection</div>', unsafe_allow_html=True)
        universe_type = st.selectbox(
            "Analysis Universe",
            UNIVERSE_OPTIONS,
            help="Choose the stock/ETF universe for regime analysis"
        )
        
        selected_index = None
        if universe_type == "Index Constituents":
            selected_index = st.selectbox(
                "Select Index",
                INDEX_LIST,
                index=INDEX_LIST.index("NIFTY 500"),
                help="Select the index for constituent analysis"
            )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Analysis Type
        st.markdown('<div class="sidebar-title">📊 Analysis Type</div>', unsafe_allow_html=True)
        analysis_mode = st.radio(
            "Select Mode",
            ["📅 Single Day", "📈 Time Series"],
            label_visibility="collapsed",
            help="Single Day: Analyze one date | Time Series: Track regime over a date range"
        )
        
        # Date Selection
        single_date = None
        start_date = None
        end_date = None
        
        if "Single" in analysis_mode:
            st.markdown('<div class="sidebar-title">📅 Analysis Date</div>', unsafe_allow_html=True)
            single_date = st.date_input(
                "Select Date",
                dt.date.today() - timedelta(days=1),
                max_value=dt.date.today(),
                help="Select the date for regime analysis"
            )
        else:
            st.markdown('<div class="sidebar-title">📅 Date Range</div>', unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_date = st.date_input(
                    "Start Date",
                    dt.date.today() - timedelta(days=60),
                    max_value=dt.date.today(),
                    help="Start of analysis period"
                )
            with col_d2:
                end_date = st.date_input(
                    "End Date",
                    dt.date.today() - timedelta(days=1),
                    max_value=dt.date.today(),
                    help="End of analysis period"
                )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Run Button
        run_analysis = st.button("◈ RUN ANALYSIS", type="primary", use_container_width=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Info
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Multi-Factor Analysis<br>
                <strong>Data:</strong> yfinance Live
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return universe_type, selected_index, analysis_mode, single_date, start_date, end_date, run_analysis


def render_header():
    """Render main header"""
    st.markdown(f"""
    <div class="premium-header">
        <div class="product-badge">A Pragyam Product</div>
        <h1>{APP_TITLE} : {APP_SUBTITLE}</h1>
        <div class="tagline">Institutional-grade regime detection using multi-factor analysis</div>
    </div>
    """, unsafe_allow_html=True)


def run_home_page():
    """Render landing/home page"""
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 260px;'>
            <h3 style='color: var(--primary-color); margin-bottom: 1rem;'>🎯 Multi-Factor Analysis</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                7 analysis factors combine to determine market regime with weighted scoring.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Factors:</strong><br>
                • Momentum (30%)<br>
                • Trend (25%)<br>
                • Breadth (15%)<br>
                • Velocity (15%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card success' style='min-height: 260px;'>
            <h3 style='color: var(--success-green); margin-bottom: 1rem;'>📊 7 Regime Types</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                From CRISIS to STRONG_BULL with suggested portfolio positioning for each.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Classifications:</strong><br>
                • 🐂 Bull Market Mix<br>
                • 📊 Chop/Consolidation Mix<br>
                • 🐻 Bear Market Mix
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card info' style='min-height: 260px;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 1rem;'>📈 Flexible Analysis</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Single Day or Time Series analysis across multiple universes.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Universes:</strong><br>
                • ETF Universe (28 ETFs)<br>
                • F&O Stocks (~200+)<br>
                • Index Constituents
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting started
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 Getting Started</h4>
        <p style='color: var(--text-muted); line-height: 1.7;'>
            Select your analysis <strong>Universe</strong> and <strong>Analysis Type</strong> from the sidebar, 
            then click <strong>RUN ANALYSIS</strong> to detect the current market regime.
            The system will analyze multiple factors across your selected universe to determine
            whether the market is in a bullish, bearish, or choppy state.
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    
    # Render sidebar and get controls
    universe_type, selected_index, analysis_mode, single_date, start_date, end_date, run_analysis = render_sidebar()
    
    # Render header
    render_header()
    
    # Run analysis if button clicked
    if run_analysis:
        # Initialize engine
        engine = MarketDataEngine()
        detector = MarketRegimeDetector()
        
        # Set universe
        progress = st.progress(0, text="Initializing...")
        universe_msg = engine.set_universe(universe_type, selected_index)
        st.toast(universe_msg, icon="✓" if "✓" in universe_msg else "⚠️")
        
        if "Single" in analysis_mode:
            # Single Day Analysis
            analysis_dt = datetime.combine(single_date, datetime.min.time())
            
            def update_progress(p, msg):
                progress.progress(p, text=msg)
            
            try:
                historical_data = engine.get_regime_data(analysis_dt, progress_callback=update_progress)
                
                if not historical_data:
                    st.error("Failed to fetch market data. Please check your date and try again.")
                    progress.empty()
                    return
                
                result = detector.detect(historical_data)
                
                progress.empty()
                st.session_state.regime_result = result
                st.session_state.time_series_results = None
                
                st.success("✅ Regime analysis completed!")
                
            except Exception as e:
                progress.empty()
                st.error(f"Analysis failed: {str(e)}")
                logging.error(f"Analysis error: {e}", exc_info=True)
                return
        
        else:
            # Time Series Analysis
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())
            
            def update_progress(p, msg):
                progress.progress(p * 0.7, text=msg)
            
            try:
                historical_data = engine.get_time_series_regime_data(start_dt, end_dt, progress_callback=update_progress)
                
                if not historical_data:
                    st.error("Failed to fetch market data. Please check your dates and try again.")
                    progress.empty()
                    return
                
                # Run regime detection for each day
                ts_results = []
                total_days = len(historical_data)
                
                for i in range(10, len(historical_data)):
                    window = historical_data[i-10:i+1]
                    result = detector.detect(window)
                    
                    ts_results.append({
                        'date': historical_data[i][0],
                        'regime': result.regime_name,
                        'score': result.composite_score,
                        'confidence': result.confidence,
                        'mix': result.suggested_mix
                    })
                    
                    if i % 5 == 0:
                        progress.progress(0.7 + 0.3 * (i / total_days), text=f"Analyzing day {i-9}/{total_days-10}...")
                
                progress.empty()
                st.session_state.time_series_results = ts_results
                st.session_state.regime_result = None
                
                st.success(f"✅ Time series analysis completed! ({len(ts_results)} days analyzed)")
                
            except Exception as e:
                progress.empty()
                st.error(f"Analysis failed: {str(e)}")
                logging.error(f"Analysis error: {e}", exc_info=True)
                return
    
    # Display results
    if st.session_state.regime_result is not None:
        display_single_day_result(st.session_state.regime_result)
    elif st.session_state.time_series_results is not None:
        display_time_series_results(st.session_state.time_series_results)
    else:
        run_home_page()
    
    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© {datetime.now().year} AVASTHA | Hemrek Capital | {VERSION}")


if __name__ == "__main__":
    main()
