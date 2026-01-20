"""
███╗   ███╗ █████╗ ██████╗ ██╗  ██╗███████╗████████╗    ██████╗ ███████╗ ██████╗ ██╗███╗   ███╗███████╗
████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝╚══██╔══╝    ██╔══██╗██╔════╝██╔════╝ ██║████╗ ████║██╔════╝
██╔████╔██║███████║██████╔╝█████╔╝ █████╗     ██║       ██████╔╝█████╗  ██║  ███╗██║██╔████╔██║█████╗  
██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ██╔══╝     ██║       ██╔══██╗██╔══╝  ██║   ██║██║██║╚██╔╝██║██╔══╝  
██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗███████╗   ██║       ██║  ██║███████╗╚██████╔╝██║██║ ╚═╝ ██║███████╗
╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝       ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚═╝     ╚═╝╚══════╝

AVASTHA (आवस्था) - Hedge Fund Grade Market Regime Detection System
==================================================================

A sophisticated multi-model regime detection engine that synthesizes:
- Hidden Markov Model (HMM) based state detection
- Volatility regime clustering (GARCH-inspired)
- Momentum regime identification
- Correlation regime analysis
- Liquidity regime detection
- Risk regime quantification

Outputs a unified regime signal with probability distributions and confidence metrics.

Author: Quantitative Research
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import io

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeType(Enum):
    """Primary regime classifications"""
    CRISIS = "CRISIS"
    BEAR_ACCELERATION = "BEAR_ACCELERATION"
    BEAR_DECELERATION = "BEAR_DECELERATION"
    ACCUMULATION = "ACCUMULATION"
    EARLY_BULL = "EARLY_BULL"
    BULL_TREND = "BULL_TREND"
    BULL_EUPHORIA = "BULL_EUPHORIA"
    DISTRIBUTION = "DISTRIBUTION"
    CHOP = "CHOP"
    TRANSITION = "TRANSITION"


class VolatilityRegime(Enum):
    """Volatility regime states"""
    COMPRESSED = "COMPRESSED"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    EXTREME = "EXTREME"


class MomentumRegime(Enum):
    """Momentum regime states"""
    STRONG_BEARISH = "STRONG_BEARISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    STRONG_BULLISH = "STRONG_BULLISH"


class TrendRegime(Enum):
    """Trend regime states"""
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"
    UPTREND = "UPTREND"
    STRONG_UPTREND = "STRONG_UPTREND"


@dataclass
class RegimeSignal:
    """Comprehensive regime signal output"""
    primary_regime: RegimeType
    primary_confidence: float
    regime_probabilities: Dict[str, float]
    volatility_regime: VolatilityRegime
    volatility_percentile: float
    momentum_regime: MomentumRegime
    momentum_score: float
    trend_regime: TrendRegime
    trend_strength: float
    risk_score: float
    transition_probability: float
    recommended_exposure: float
    signal_strength: float
    composite_oscillator: float
    sub_signals: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE REGIME DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MarketRegimeEngine:
    """
    Multi-Model Market Regime Detection Engine
    
    Combines multiple analytical approaches to identify market regimes:
    1. Volatility Clustering (GARCH-inspired)
    2. Momentum Phase Detection
    3. Trend State Identification
    4. Risk Regime Analysis
    5. Hidden Markov Model-inspired State Detection
    6. Structural Break Detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        self._regime_history = []
        
    def _default_config(self) -> Dict:
        return {
            'volatility_lookback': 20,
            'momentum_fast': 12,
            'momentum_slow': 26,
            'trend_short': 20,
            'trend_medium': 50,
            'trend_long': 200,
            'hmm_n_states': 5,
            'vol_percentile_window': 252,
            'smoothing_factor': 0.3,
            'transition_sensitivity': 0.7,
            'risk_free_rate': 0.05,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # VOLATILITY REGIME DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Multi-scale volatility analysis with GARCH-inspired clustering
        """
        n = len(returns)
        if n < 30:
            return {'regime': VolatilityRegime.NORMAL, 'percentile': 50.0, 'metrics': {}}
        
        # Realized volatility at multiple scales
        vol_5d = returns.rolling(5).std() * np.sqrt(252)
        vol_20d = returns.rolling(20).std() * np.sqrt(252)
        vol_60d = returns.rolling(60).std() * np.sqrt(252)
        
        current_vol = vol_20d.iloc[-1] if pd.notna(vol_20d.iloc[-1]) else returns.std() * np.sqrt(252)
        
        # Volatility percentile (where we stand historically)
        lookback = min(self.config['vol_percentile_window'], n)
        vol_history = vol_20d.dropna().tail(lookback)
        if len(vol_history) > 0:
            vol_percentile = stats.percentileofscore(vol_history, current_vol)
        else:
            vol_percentile = 50.0
        
        # Volatility clustering detection (GARCH-inspired)
        # High vol tends to cluster - check autocorrelation of squared returns
        squared_returns = returns ** 2
        vol_autocorr = squared_returns.autocorr(lag=1) if n > 10 else 0
        
        # Volatility regime change detection
        vol_zscore = (current_vol - vol_20d.mean()) / (vol_20d.std() + 1e-8)
        
        # Volatility term structure (contango/backwardation proxy)
        vol_term_structure = vol_5d.iloc[-1] / vol_60d.iloc[-1] if pd.notna(vol_60d.iloc[-1]) and vol_60d.iloc[-1] > 0 else 1.0
        
        # Classify volatility regime
        if vol_percentile > 90 or vol_zscore > 2.0:
            regime = VolatilityRegime.EXTREME
        elif vol_percentile > 70 or vol_zscore > 1.0:
            regime = VolatilityRegime.ELEVATED
        elif vol_percentile < 20 or vol_zscore < -1.0:
            regime = VolatilityRegime.COMPRESSED
        else:
            regime = VolatilityRegime.NORMAL
        
        return {
            'regime': regime,
            'percentile': vol_percentile,
            'current_vol': current_vol,
            'vol_zscore': vol_zscore,
            'vol_autocorr': vol_autocorr,
            'vol_term_structure': vol_term_structure,
            'metrics': {
                'vol_5d': vol_5d.iloc[-1] if pd.notna(vol_5d.iloc[-1]) else np.nan,
                'vol_20d': current_vol,
                'vol_60d': vol_60d.iloc[-1] if pd.notna(vol_60d.iloc[-1]) else np.nan,
            }
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # MOMENTUM REGIME DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _calculate_momentum_regime(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """
        Multi-factor momentum regime identification
        """
        n = len(prices)
        if n < 30:
            return {'regime': MomentumRegime.NEUTRAL, 'score': 0.0, 'metrics': {}}
        
        # RSI calculation
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).ewm(span=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # MACD components
        ema_fast = prices.ewm(span=self.config['momentum_fast']).mean()
        ema_slow = prices.ewm(span=self.config['momentum_slow']).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=9).mean()
        macd_histogram = macd - signal_line
        
        # Rate of change at multiple timeframes
        roc_5 = (prices / prices.shift(5) - 1) * 100
        roc_20 = (prices / prices.shift(20) - 1) * 100
        roc_60 = (prices / prices.shift(60) - 1) * 100
        
        # Momentum acceleration (second derivative)
        mom_accel = roc_5.diff(5)
        
        # Composite momentum score [-100, 100]
        rsi_component = (current_rsi - 50) * 2  # [-100, 100]
        macd_component = np.tanh(macd_histogram.iloc[-1] / prices.iloc[-1] * 100) * 100
        roc_component = np.clip(roc_20.iloc[-1] * 2, -100, 100) if pd.notna(roc_20.iloc[-1]) else 0
        
        momentum_score = (
            rsi_component * 0.35 +
            macd_component * 0.35 +
            roc_component * 0.30
        )
        
        # Momentum divergence detection
        price_higher = prices.iloc[-1] > prices.iloc[-20] if n > 20 else False
        rsi_lower = current_rsi < rsi.iloc[-20] if n > 20 else False
        bearish_divergence = price_higher and rsi_lower
        
        price_lower = prices.iloc[-1] < prices.iloc[-20] if n > 20 else False
        rsi_higher = current_rsi > rsi.iloc[-20] if n > 20 else False
        bullish_divergence = price_lower and rsi_higher
        
        # Classify momentum regime
        if momentum_score > 60 and current_rsi > 70:
            regime = MomentumRegime.STRONG_BULLISH
        elif momentum_score > 25:
            regime = MomentumRegime.BULLISH
        elif momentum_score < -60 and current_rsi < 30:
            regime = MomentumRegime.STRONG_BEARISH
        elif momentum_score < -25:
            regime = MomentumRegime.BEARISH
        else:
            regime = MomentumRegime.NEUTRAL
        
        return {
            'regime': regime,
            'score': momentum_score,
            'rsi': current_rsi,
            'macd_histogram': macd_histogram.iloc[-1],
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'metrics': {
                'roc_5': roc_5.iloc[-1] if pd.notna(roc_5.iloc[-1]) else 0,
                'roc_20': roc_20.iloc[-1] if pd.notna(roc_20.iloc[-1]) else 0,
                'roc_60': roc_60.iloc[-1] if pd.notna(roc_60.iloc[-1]) else 0,
                'mom_accel': mom_accel.iloc[-1] if pd.notna(mom_accel.iloc[-1]) else 0,
            }
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # TREND REGIME DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _calculate_trend_regime(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Multi-timeframe trend state identification
        """
        n = len(prices)
        if n < 30:
            return {'regime': TrendRegime.SIDEWAYS, 'strength': 0.0, 'metrics': {}}
        
        # Moving averages
        ma_short = prices.rolling(self.config['trend_short']).mean()
        ma_medium = prices.rolling(self.config['trend_medium']).mean()
        ma_long = prices.rolling(min(self.config['trend_long'], n)).mean()
        
        current_price = prices.iloc[-1]
        
        # Price position relative to MAs
        above_short = current_price > ma_short.iloc[-1] if pd.notna(ma_short.iloc[-1]) else False
        above_medium = current_price > ma_medium.iloc[-1] if pd.notna(ma_medium.iloc[-1]) else False
        above_long = current_price > ma_long.iloc[-1] if pd.notna(ma_long.iloc[-1]) else False
        
        # MA alignment (golden cross / death cross detection)
        ma_short_above_medium = ma_short.iloc[-1] > ma_medium.iloc[-1] if pd.notna(ma_medium.iloc[-1]) else False
        ma_medium_above_long = ma_medium.iloc[-1] > ma_long.iloc[-1] if pd.notna(ma_long.iloc[-1]) else False
        
        # ADX-style trend strength calculation
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        tr = high - low
        atr = tr.rolling(14).mean()
        
        # Directional movement
        plus_dm = prices.diff().where(prices.diff() > 0, 0)
        minus_dm = (-prices.diff()).where(prices.diff() < 0, 0)
        
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr + 1e-8)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr + 1e-8)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(14).mean()
        
        trend_strength = adx.iloc[-1] if pd.notna(adx.iloc[-1]) else 0
        
        # Linear regression slope for trend direction
        if n >= 20:
            x = np.arange(20)
            y = prices.tail(20).values
            slope, _, r_value, _, _ = stats.linregress(x, y)
            r_squared = r_value ** 2
            normalized_slope = slope / prices.iloc[-1] * 100  # Percentage per day
        else:
            normalized_slope = 0
            r_squared = 0
        
        # Composite trend score
        alignment_score = sum([above_short, above_medium, above_long, ma_short_above_medium, ma_medium_above_long])
        
        # Classify trend regime
        if alignment_score >= 4 and normalized_slope > 0.1 and trend_strength > 25:
            regime = TrendRegime.STRONG_UPTREND
        elif alignment_score >= 3 and normalized_slope > 0:
            regime = TrendRegime.UPTREND
        elif alignment_score <= 1 and normalized_slope < -0.1 and trend_strength > 25:
            regime = TrendRegime.STRONG_DOWNTREND
        elif alignment_score <= 2 and normalized_slope < 0:
            regime = TrendRegime.DOWNTREND
        else:
            regime = TrendRegime.SIDEWAYS
        
        return {
            'regime': regime,
            'strength': trend_strength,
            'slope': normalized_slope,
            'r_squared': r_squared,
            'alignment_score': alignment_score,
            'metrics': {
                'ma_short': ma_short.iloc[-1] if pd.notna(ma_short.iloc[-1]) else np.nan,
                'ma_medium': ma_medium.iloc[-1] if pd.notna(ma_medium.iloc[-1]) else np.nan,
                'ma_long': ma_long.iloc[-1] if pd.notna(ma_long.iloc[-1]) else np.nan,
                'adx': trend_strength,
            }
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # RISK REGIME ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _calculate_risk_metrics(self, returns: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive risk regime analysis
        """
        n = len(returns)
        if n < 30:
            return {'risk_score': 50.0, 'metrics': {}}
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        current_drawdown = drawdown.iloc[-1]
        max_drawdown = drawdown.min()
        
        # Value at Risk (Historical VaR)
        var_95 = np.percentile(returns.dropna(), 5) * 100
        var_99 = np.percentile(returns.dropna(), 1) * 100
        
        # Expected Shortfall (CVaR)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100 if len(returns[returns <= np.percentile(returns, 5)]) > 0 else var_95
        
        # Sortino Ratio (downside risk)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        avg_return = returns.mean() * 252
        sortino = (avg_return - self.config['risk_free_rate']) / (downside_std + 1e-8)
        
        # Tail risk (kurtosis)
        kurtosis = stats.kurtosis(returns.dropna())
        skewness = stats.skew(returns.dropna())
        
        # Composite risk score (0-100, higher = more risk)
        drawdown_component = min(abs(current_drawdown) / 20 * 100, 100) * 0.25
        var_component = min(abs(var_95) / 3 * 100, 100) * 0.25
        tail_component = min((kurtosis + 3) / 6 * 100, 100) * 0.20  # Excess kurtosis
        vol_component = min(downside_std / 0.30 * 100, 100) * 0.30
        
        risk_score = drawdown_component + var_component + tail_component + vol_component
        
        return {
            'risk_score': risk_score,
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'sortino': sortino,
            'kurtosis': kurtosis,
            'skewness': skewness,
            'metrics': {
                'drawdown': current_drawdown,
                'var_95': var_95,
                'sortino': sortino,
            }
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # HIDDEN MARKOV MODEL-INSPIRED STATE DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _hmm_state_detection(self, returns: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """
        Gaussian Mixture Model-based state detection (HMM-inspired)
        Identifies latent market states from return distribution
        """
        n = len(returns)
        if n < 100:
            return {'state': 2, 'probabilities': [0.2] * 5, 'transition_prob': 0.1}
        
        # Prepare features for clustering
        features = pd.DataFrame({
            'returns': returns,
            'volatility': returns.rolling(20).std(),
            'momentum': returns.rolling(10).mean(),
        }).dropna()
        
        if len(features) < 50:
            return {'state': 2, 'probabilities': [0.2] * 5, 'transition_prob': 0.1}
        
        # Standardize features
        X = self.scaler.fit_transform(features)
        
        # Fit Gaussian Mixture Model
        n_states = self.config['hmm_n_states']
        gmm = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
        
        try:
            gmm.fit(X)
            current_state = gmm.predict(X[-1:].reshape(1, -1))[0]
            state_probabilities = gmm.predict_proba(X[-1:].reshape(1, -1))[0].tolist()
            
            # Estimate transition probability (simplified)
            states = gmm.predict(X)
            state_changes = np.diff(states) != 0
            transition_prob = state_changes.mean() if len(state_changes) > 0 else 0.1
            
        except Exception as e:
            logging.warning(f"GMM fitting failed: {e}")
            current_state = 2
            state_probabilities = [0.2] * n_states
            transition_prob = 0.1
        
        return {
            'state': current_state,
            'probabilities': state_probabilities,
            'transition_prob': transition_prob,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRUCTURAL BREAK DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _detect_structural_breaks(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Detect structural breaks / change points in price series
        """
        n = len(prices)
        if n < 60:
            return {'recent_break': False, 'break_intensity': 0.0}
        
        # Rolling mean shift detection
        short_ma = prices.rolling(10).mean()
        long_ma = prices.rolling(50).mean()
        
        # Deviation from long-term trend
        deviation = (short_ma - long_ma) / long_ma * 100
        
        # Z-score of deviation
        dev_zscore = (deviation - deviation.rolling(50).mean()) / (deviation.rolling(50).std() + 1e-8)
        
        # Detect recent break (within last 10 bars)
        recent_dev_zscore = dev_zscore.tail(10)
        recent_break = (abs(recent_dev_zscore) > 2.0).any()
        break_intensity = abs(dev_zscore.iloc[-1]) if pd.notna(dev_zscore.iloc[-1]) else 0
        
        # Regime change indicator
        ma_crossover_recent = False
        if n > 20:
            ma_cross_signal = (short_ma > long_ma).astype(int).diff()
            ma_crossover_recent = abs(ma_cross_signal.tail(10)).sum() > 0
        
        return {
            'recent_break': recent_break,
            'break_intensity': break_intensity,
            'ma_crossover_recent': ma_crossover_recent,
            'deviation_zscore': dev_zscore.iloc[-1] if pd.notna(dev_zscore.iloc[-1]) else 0,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMPOSITE REGIME SYNTHESIS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _synthesize_primary_regime(
        self,
        vol_regime: Dict,
        mom_regime: Dict,
        trend_regime: Dict,
        risk_metrics: Dict,
        hmm_state: Dict,
        structural_breaks: Dict
    ) -> Tuple[RegimeType, float, Dict[str, float]]:
        """
        Synthesize all sub-regimes into a primary market regime classification
        """
        
        # Extract key metrics
        vol_state = vol_regime['regime']
        vol_pct = vol_regime['percentile']
        mom_state = mom_regime['regime']
        mom_score = mom_regime['score']
        trend_state = trend_regime['regime']
        trend_strength = trend_regime['strength']
        risk_score = risk_metrics['risk_score']
        current_dd = risk_metrics['current_drawdown']
        transition_prob = hmm_state['transition_prob']
        break_intensity = structural_breaks['break_intensity']
        
        # Initialize regime probabilities
        regime_scores = {r.value: 0.0 for r in RegimeType}
        
        # CRISIS detection
        if (vol_state == VolatilityRegime.EXTREME and
            mom_state in [MomentumRegime.STRONG_BEARISH, MomentumRegime.BEARISH] and
            current_dd < -15):
            regime_scores[RegimeType.CRISIS.value] += 40
        if vol_pct > 90 and mom_score < -50:
            regime_scores[RegimeType.CRISIS.value] += 30
        if risk_score > 80:
            regime_scores[RegimeType.CRISIS.value] += 20
        
        # BEAR_ACCELERATION
        if (trend_state in [TrendRegime.DOWNTREND, TrendRegime.STRONG_DOWNTREND] and
            mom_state == MomentumRegime.STRONG_BEARISH and
            mom_regime.get('mom_accel', mom_regime.get('metrics', {}).get('mom_accel', 0)) < 0):
            regime_scores[RegimeType.BEAR_ACCELERATION.value] += 35
        if trend_strength > 25 and mom_score < -40:
            regime_scores[RegimeType.BEAR_ACCELERATION.value] += 25
        
        # BEAR_DECELERATION
        if (trend_state in [TrendRegime.DOWNTREND, TrendRegime.STRONG_DOWNTREND] and
            mom_regime.get('bullish_divergence', False)):
            regime_scores[RegimeType.BEAR_DECELERATION.value] += 35
        if trend_state == TrendRegime.DOWNTREND and mom_score > -30 and mom_score < 0:
            regime_scores[RegimeType.BEAR_DECELERATION.value] += 25
        
        # ACCUMULATION
        if (vol_state == VolatilityRegime.COMPRESSED and
            trend_state == TrendRegime.SIDEWAYS and
            mom_state == MomentumRegime.NEUTRAL):
            regime_scores[RegimeType.ACCUMULATION.value] += 35
        if vol_pct < 30 and abs(mom_score) < 20 and trend_strength < 20:
            regime_scores[RegimeType.ACCUMULATION.value] += 25
        
        # EARLY_BULL
        if (trend_state == TrendRegime.UPTREND and
            mom_state in [MomentumRegime.BULLISH, MomentumRegime.STRONG_BULLISH] and
            vol_state in [VolatilityRegime.COMPRESSED, VolatilityRegime.NORMAL] and
            structural_breaks.get('ma_crossover_recent', False)):
            regime_scores[RegimeType.EARLY_BULL.value] += 40
        if trend_regime['alignment_score'] >= 3 and mom_score > 20 and mom_score < 50:
            regime_scores[RegimeType.EARLY_BULL.value] += 25
        
        # BULL_TREND
        if (trend_state in [TrendRegime.UPTREND, TrendRegime.STRONG_UPTREND] and
            mom_state in [MomentumRegime.BULLISH, MomentumRegime.STRONG_BULLISH] and
            trend_strength > 25):
            regime_scores[RegimeType.BULL_TREND.value] += 40
        if trend_regime['alignment_score'] >= 4 and mom_score > 30:
            regime_scores[RegimeType.BULL_TREND.value] += 25
        
        # BULL_EUPHORIA
        if (mom_state == MomentumRegime.STRONG_BULLISH and
            mom_regime.get('rsi', 50) > 75 and
            vol_state in [VolatilityRegime.ELEVATED, VolatilityRegime.EXTREME]):
            regime_scores[RegimeType.BULL_EUPHORIA.value] += 40
        if mom_score > 70 and mom_regime.get('bearish_divergence', False):
            regime_scores[RegimeType.BULL_EUPHORIA.value] += 30
        
        # DISTRIBUTION
        if (trend_state in [TrendRegime.UPTREND, TrendRegime.SIDEWAYS] and
            mom_regime.get('bearish_divergence', False) and
            vol_state == VolatilityRegime.ELEVATED):
            regime_scores[RegimeType.DISTRIBUTION.value] += 35
        if trend_regime['alignment_score'] >= 3 and mom_score < 0 and mom_score > -30:
            regime_scores[RegimeType.DISTRIBUTION.value] += 25
        
        # CHOP / SIDEWAYS
        if (trend_state == TrendRegime.SIDEWAYS and
            trend_strength < 20 and
            abs(mom_score) < 25):
            regime_scores[RegimeType.CHOP.value] += 35
        if vol_state == VolatilityRegime.NORMAL and abs(mom_score) < 15:
            regime_scores[RegimeType.CHOP.value] += 20
        
        # TRANSITION
        if transition_prob > 0.3 or break_intensity > 2.0:
            regime_scores[RegimeType.TRANSITION.value] += 30
        if structural_breaks.get('recent_break', False):
            regime_scores[RegimeType.TRANSITION.value] += 20
        
        # Normalize to probabilities
        total_score = sum(regime_scores.values()) + 1e-8
        regime_probabilities = {k: v / total_score for k, v in regime_scores.items()}
        
        # Determine primary regime
        primary_regime = RegimeType(max(regime_scores, key=regime_scores.get))
        primary_confidence = regime_probabilities[primary_regime.value]
        
        # If confidence is too low, default to CHOP
        if primary_confidence < 0.15:
            primary_regime = RegimeType.CHOP
            primary_confidence = regime_probabilities[RegimeType.CHOP.value]
        
        return primary_regime, primary_confidence, regime_probabilities
    
    def _calculate_recommended_exposure(
        self,
        primary_regime: RegimeType,
        confidence: float,
        risk_score: float,
        vol_percentile: float
    ) -> float:
        """
        Calculate recommended portfolio exposure based on regime
        Returns: exposure multiplier [0.0 - 1.5]
        """
        # Base exposure by regime
        regime_exposure = {
            RegimeType.CRISIS: 0.20,
            RegimeType.BEAR_ACCELERATION: 0.30,
            RegimeType.BEAR_DECELERATION: 0.50,
            RegimeType.ACCUMULATION: 0.70,
            RegimeType.EARLY_BULL: 1.00,
            RegimeType.BULL_TREND: 1.20,
            RegimeType.BULL_EUPHORIA: 0.80,
            RegimeType.DISTRIBUTION: 0.50,
            RegimeType.CHOP: 0.60,
            RegimeType.TRANSITION: 0.50,
        }
        
        base_exposure = regime_exposure.get(primary_regime, 0.60)
        
        # Adjust by confidence
        confidence_adj = 0.7 + (confidence * 0.6)  # [0.7 - 1.3]
        
        # Adjust by risk
        risk_adj = 1.0 - (risk_score / 200)  # [0.5 - 1.0]
        
        # Adjust by volatility
        vol_adj = 1.0 if vol_percentile < 70 else (1.0 - (vol_percentile - 70) / 100)
        
        final_exposure = base_exposure * confidence_adj * risk_adj * vol_adj
        return np.clip(final_exposure, 0.0, 1.5)
    
    def _calculate_composite_oscillator(
        self,
        mom_score: float,
        vol_percentile: float,
        trend_strength: float,
        risk_score: float,
        alignment_score: int
    ) -> float:
        """
        Calculate unified regime oscillator [-100, 100]
        Positive = bullish regime, Negative = bearish regime
        """
        # Momentum component (already scaled)
        mom_component = mom_score * 0.35
        
        # Trend component
        trend_direction = (alignment_score - 2.5) / 2.5  # [-1, 1]
        trend_component = trend_direction * trend_strength * 0.25
        
        # Volatility component (compressed vol = bullish, extreme = bearish)
        vol_component = (50 - vol_percentile) * 0.5 * 0.20
        
        # Risk component (low risk = bullish)
        risk_component = (50 - risk_score) * 0.20
        
        oscillator = mom_component + trend_component + vol_component + risk_component
        return np.clip(oscillator, -100, 100)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN DETECTION METHOD
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_regime(self, prices: pd.Series, ohlcv: Optional[pd.DataFrame] = None) -> RegimeSignal:
        """
        Main method: Detect current market regime from price data
        
        Args:
            prices: Close price series
            ohlcv: Optional OHLCV dataframe for additional analysis
            
        Returns:
            RegimeSignal with comprehensive regime information
        """
        if len(prices) < 30:
            return RegimeSignal(
                primary_regime=RegimeType.CHOP,
                primary_confidence=0.5,
                regime_probabilities={r.value: 1/10 for r in RegimeType},
                volatility_regime=VolatilityRegime.NORMAL,
                volatility_percentile=50.0,
                momentum_regime=MomentumRegime.NEUTRAL,
                momentum_score=0.0,
                trend_regime=TrendRegime.SIDEWAYS,
                trend_strength=0.0,
                risk_score=50.0,
                transition_probability=0.1,
                recommended_exposure=0.6,
                signal_strength=0.0,
                composite_oscillator=0.0,
            )
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Run all sub-regime detections
        vol_regime = self._calculate_volatility_metrics(returns)
        mom_regime = self._calculate_momentum_regime(prices, returns)
        trend_regime = self._calculate_trend_regime(prices)
        risk_metrics = self._calculate_risk_metrics(returns, prices)
        hmm_state = self._hmm_state_detection(returns, prices)
        structural_breaks = self._detect_structural_breaks(prices)
        
        # Synthesize primary regime
        primary_regime, confidence, regime_probs = self._synthesize_primary_regime(
            vol_regime, mom_regime, trend_regime, risk_metrics, hmm_state, structural_breaks
        )
        
        # Calculate recommended exposure
        recommended_exposure = self._calculate_recommended_exposure(
            primary_regime, confidence, risk_metrics['risk_score'], vol_regime['percentile']
        )
        
        # Calculate composite oscillator
        composite_osc = self._calculate_composite_oscillator(
            mom_regime['score'],
            vol_regime['percentile'],
            trend_regime['strength'],
            risk_metrics['risk_score'],
            trend_regime['alignment_score']
        )
        
        # Signal strength (conviction level)
        signal_strength = confidence * (1 + abs(composite_osc) / 100) / 2
        
        return RegimeSignal(
            primary_regime=primary_regime,
            primary_confidence=confidence,
            regime_probabilities=regime_probs,
            volatility_regime=vol_regime['regime'],
            volatility_percentile=vol_regime['percentile'],
            momentum_regime=mom_regime['regime'],
            momentum_score=mom_regime['score'],
            trend_regime=trend_regime['regime'],
            trend_strength=trend_regime['strength'],
            risk_score=risk_metrics['risk_score'],
            transition_probability=hmm_state['transition_prob'],
            recommended_exposure=recommended_exposure,
            signal_strength=signal_strength,
            composite_oscillator=composite_osc,
            sub_signals={
                'rsi': mom_regime.get('rsi', 50),
                'macd_hist': mom_regime.get('macd_histogram', 0),
                'vol_zscore': vol_regime.get('vol_zscore', 0),
                'trend_slope': trend_regime.get('slope', 0),
                'drawdown': risk_metrics.get('current_drawdown', 0),
                'var_95': risk_metrics.get('var_95', 0),
                'break_intensity': structural_breaks.get('break_intensity', 0),
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeHistoryAnalyzer:
    """
    Analyze regime history over time for a given asset
    """
    
    def __init__(self, engine: MarketRegimeEngine):
        self.engine = engine
    
    def calculate_regime_history(
        self,
        prices: pd.Series,
        lookback_window: int = 100
    ) -> pd.DataFrame:
        """
        Calculate regime signal for each point in history
        """
        results = []
        n = len(prices)
        
        for i in range(lookback_window, n):
            window_prices = prices.iloc[:i+1]
            signal = self.engine.detect_regime(window_prices)
            
            results.append({
                'date': prices.index[i],
                'price': prices.iloc[i],
                'regime': signal.primary_regime.value,
                'confidence': signal.primary_confidence,
                'oscillator': signal.composite_oscillator,
                'exposure': signal.recommended_exposure,
                'vol_regime': signal.volatility_regime.value,
                'mom_regime': signal.momentum_regime.value,
                'trend_regime': signal.trend_regime.value,
                'risk_score': signal.risk_score,
                'signal_strength': signal.signal_strength,
            })
        
        return pd.DataFrame(results).set_index('date')


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_market_data(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        logging.error(f"Error fetching {symbol}: {e}")
        return None


def load_symbols() -> List[str]:
    """Load F&O symbols"""
    # Default F&O stocks for Indian market
    default_symbols = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "BHARTIARTL.NS", "HINDUNILVR.NS", "KOTAKBANK.NS", "ITC.NS", "SBIN.NS",
        "BAJFINANCE.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "WIPRO.NS", "HCLTECH.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "TITAN.NS",
        "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
        "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "TECHM.NS", "INDUSINDBK.NS"
    ]
    return default_symbols


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="AVASTHA | Market Regime Detection",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS - Institutional Dark Theme
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --accent-primary: #00d4aa;
            --accent-secondary: #7c3aed;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --accent-info: #3b82f6;
            --text-primary: #f0f0f5;
            --text-secondary: #9090a0;
            --text-muted: #606070;
            --border-color: #252530;
            --glow-primary: rgba(0, 212, 170, 0.15);
            --glow-secondary: rgba(124, 58, 237, 0.15);
        }
        
        .main, .stApp {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Space Grotesk', sans-serif;
        }
        
        [data-testid="stSidebar"] {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
        }
        
        .stApp > header { 
            background: transparent; 
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 40px var(--glow-primary);
        }
        
        /* Metric Cards */
        .metric-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        }
        
        .metric-card.bullish::before {
            background: var(--accent-primary);
            box-shadow: 0 0 20px var(--glow-primary);
        }
        
        .metric-card.bearish::before {
            background: var(--accent-danger);
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
        }
        
        .metric-card.neutral::before {
            background: var(--accent-warning);
            box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
        }
        
        /* Regime Badge */
        .regime-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.05em;
        }
        
        .regime-badge.crisis {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
        }
        
        .regime-badge.bear {
            background: rgba(251, 146, 60, 0.2);
            border: 1px solid rgba(251, 146, 60, 0.4);
            color: #fed7aa;
        }
        
        .regime-badge.neutral {
            background: rgba(250, 204, 21, 0.2);
            border: 1px solid rgba(250, 204, 21, 0.4);
            color: #fef08a;
        }
        
        .regime-badge.bull {
            background: rgba(0, 212, 170, 0.2);
            border: 1px solid rgba(0, 212, 170, 0.4);
            color: #6ee7b7;
        }
        
        .regime-badge.euphoria {
            background: rgba(124, 58, 237, 0.2);
            border: 1px solid rgba(124, 58, 237, 0.4);
            color: #c4b5fd;
        }
        
        /* Oscillator Display */
        .oscillator-display {
            background: var(--bg-tertiary);
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            position: relative;
        }
        
        .oscillator-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 4rem;
            font-weight: 700;
            line-height: 1;
        }
        
        .oscillator-value.bullish {
            color: var(--accent-primary);
            text-shadow: 0 0 30px var(--glow-primary);
        }
        
        .oscillator-value.bearish {
            color: var(--accent-danger);
            text-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        }
        
        .oscillator-value.neutral {
            color: var(--accent-warning);
            text-shadow: 0 0 30px rgba(245, 158, 11, 0.3);
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            padding: 12px 24px;
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--bg-tertiary);
            color: var(--accent-primary);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: var(--bg-primary);
            border: none;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px var(--glow-primary);
        }
        
        /* Data Tables */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* Plotly Charts */
        .stPlotlyChart {
            border-radius: 12px;
            background: var(--bg-secondary);
            padding: 16px;
            border: 1px solid var(--border-color);
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        /* Info boxes */
        .info-box {
            background: var(--bg-tertiary);
            border-left: 4px solid var(--accent-info);
            border-radius: 0 8px 8px 0;
            padding: 16px;
            margin: 16px 0;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
        
        /* Sub-regime cards */
        .sub-regime-card {
            background: var(--bg-tertiary);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
        
        .sub-regime-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 8px;
        }
        
        .sub-regime-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)


def get_regime_color(regime: RegimeType) -> str:
    """Get color for regime type"""
    colors = {
        RegimeType.CRISIS: "#ef4444",
        RegimeType.BEAR_ACCELERATION: "#f97316",
        RegimeType.BEAR_DECELERATION: "#fb923c",
        RegimeType.ACCUMULATION: "#eab308",
        RegimeType.EARLY_BULL: "#84cc16",
        RegimeType.BULL_TREND: "#22c55e",
        RegimeType.BULL_EUPHORIA: "#a855f7",
        RegimeType.DISTRIBUTION: "#f59e0b",
        RegimeType.CHOP: "#6b7280",
        RegimeType.TRANSITION: "#3b82f6",
    }
    return colors.get(regime, "#6b7280")


def get_regime_emoji(regime: RegimeType) -> str:
    """Get emoji for regime type"""
    emojis = {
        RegimeType.CRISIS: "🔴",
        RegimeType.BEAR_ACCELERATION: "📉",
        RegimeType.BEAR_DECELERATION: "🐻",
        RegimeType.ACCUMULATION: "📦",
        RegimeType.EARLY_BULL: "🌱",
        RegimeType.BULL_TREND: "🐂",
        RegimeType.BULL_EUPHORIA: "🚀",
        RegimeType.DISTRIBUTION: "⚠️",
        RegimeType.CHOP: "〰️",
        RegimeType.TRANSITION: "🔄",
    }
    return emojis.get(regime, "❓")


def render_regime_card(signal: RegimeSignal):
    """Render the main regime display card"""
    regime = signal.primary_regime
    color = get_regime_color(regime)
    emoji = get_regime_emoji(regime)
    
    # Determine card class
    if regime in [RegimeType.CRISIS, RegimeType.BEAR_ACCELERATION]:
        card_class = "bearish"
    elif regime in [RegimeType.BULL_TREND, RegimeType.BULL_EUPHORIA, RegimeType.EARLY_BULL]:
        card_class = "bullish"
    else:
        card_class = "neutral"
    
    st.markdown(f"""
    <div class="metric-card {card_class}" style="text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 16px;">{emoji}</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: {color}; margin-bottom: 8px;">
            {regime.value.replace('_', ' ')}
        </div>
        <div style="font-size: 0.9rem; color: var(--text-secondary);">
            Confidence: <span style="color: {color}; font-weight: 600;">{signal.primary_confidence*100:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_oscillator_gauge(value: float):
    """Render the composite oscillator gauge"""
    if value > 20:
        color_class = "bullish"
        color = "#00d4aa"
    elif value < -20:
        color_class = "bearish"
        color = "#ef4444"
    else:
        color_class = "neutral"
        color = "#f59e0b"
    
    st.markdown(f"""
    <div class="oscillator-display">
        <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 12px;">
            REGIME OSCILLATOR
        </div>
        <div class="oscillator-value {color_class}">
            {value:+.1f}
        </div>
        <div style="margin-top: 16px; display: flex; justify-content: center; gap: 40px; font-size: 0.8rem; color: var(--text-secondary);">
            <span>BEARISH -100</span>
            <span style="color: {color};">|</span>
            <span>BULLISH +100</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_regime_chart(df: pd.DataFrame, regime_history: Optional[pd.DataFrame] = None) -> go.Figure:
    """Create price chart with regime overlay"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price Action', 'Regime Oscillator', 'Regime Timeline')
    )
    
    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00d4aa',
            decreasing_line_color='#ef4444',
        ),
        row=1, col=1
    )
    
    # Add moving averages
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    ma_200 = df['close'].rolling(200).mean()
    
    fig.add_trace(go.Scatter(x=df.index, y=ma_20, name='MA20', line=dict(color='#3b82f6', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma_50, name='MA50', line=dict(color='#f59e0b', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma_200, name='MA200', line=dict(color='#a855f7', width=1)), row=1, col=1)
    
    if regime_history is not None and len(regime_history) > 0:
        # Oscillator subplot
        fig.add_trace(
            go.Scatter(
                x=regime_history.index,
                y=regime_history['oscillator'],
                name='Oscillator',
                line=dict(color='#00d4aa', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 170, 0.1)'
            ),
            row=2, col=1
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(0, 212, 170, 0.3)", row=2, col=1)
        fig.add_hline(y=-50, line_dash="dot", line_color="rgba(239, 68, 68, 0.3)", row=2, col=1)
        
        # Regime timeline (as colored bars)
        regime_colors = {
            'CRISIS': '#ef4444',
            'BEAR_ACCELERATION': '#f97316',
            'BEAR_DECELERATION': '#fb923c',
            'ACCUMULATION': '#eab308',
            'EARLY_BULL': '#84cc16',
            'BULL_TREND': '#22c55e',
            'BULL_EUPHORIA': '#a855f7',
            'DISTRIBUTION': '#f59e0b',
            'CHOP': '#6b7280',
            'TRANSITION': '#3b82f6',
        }
        
        # Create regime numeric mapping for visualization
        regime_map = {r: i for i, r in enumerate(regime_colors.keys())}
        regime_numeric = regime_history['regime'].map(regime_map)
        
        colors = [regime_colors.get(r, '#6b7280') for r in regime_history['regime']]
        
        fig.add_trace(
            go.Bar(
                x=regime_history.index,
                y=[1] * len(regime_history),
                name='Regime',
                marker_color=colors,
                showlegend=False,
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#12121a',
        font=dict(family='JetBrains Mono', color='#f0f0f5'),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(18, 18, 26, 0.8)'
        ),
        margin=dict(l=60, r=20, t=80, b=40),
        height=800,
        xaxis_rangeslider_visible=False,
    )
    
    fig.update_xaxes(gridcolor='#252530', showgrid=True)
    fig.update_yaxes(gridcolor='#252530', showgrid=True)
    
    return fig


def create_regime_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Create regime probability distribution chart"""
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    
    colors = [get_regime_color(RegimeType(r)) for r in sorted_probs.keys()]
    
    fig = go.Figure(go.Bar(
        x=list(sorted_probs.values()),
        y=[r.replace('_', ' ') for r in sorted_probs.keys()],
        orientation='h',
        marker_color=colors,
        text=[f'{v*100:.1f}%' for v in sorted_probs.values()],
        textposition='outside',
        textfont=dict(family='JetBrains Mono', size=11),
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#12121a',
        font=dict(family='JetBrains Mono', color='#f0f0f5'),
        title=dict(text='Regime Probability Distribution', font=dict(size=14)),
        margin=dict(l=140, r=60, t=60, b=40),
        height=400,
        xaxis=dict(
            title='Probability',
            gridcolor='#252530',
            range=[0, max(sorted_probs.values()) * 1.3]
        ),
        yaxis=dict(gridcolor='#252530'),
    )
    
    return fig


def create_sub_regime_chart(signal: RegimeSignal) -> go.Figure:
    """Create sub-regime radar chart"""
    categories = ['Momentum', 'Trend', 'Volatility', 'Risk', 'Signal']
    
    # Normalize values to 0-100 scale
    values = [
        (signal.momentum_score + 100) / 2,  # [-100, 100] -> [0, 100]
        signal.trend_strength,
        100 - signal.volatility_percentile,  # Invert (low vol = good)
        100 - signal.risk_score,  # Invert (low risk = good)
        signal.signal_strength * 100,
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.2)',
        line=dict(color='#00d4aa', width=2),
        name='Current State'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#12121a',
        font=dict(family='JetBrains Mono', color='#f0f0f5'),
        polar=dict(
            bgcolor='#12121a',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#252530',
            ),
            angularaxis=dict(gridcolor='#252530'),
        ),
        showlegend=False,
        margin=dict(l=80, r=80, t=40, b=40),
        height=350,
    )
    
    return fig


def main():
    """Main application entry point"""
    setup_page()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 40px 0;">
        <h1 style="margin-bottom: 8px;">आवस्था AVASTHA</h1>
        <p style="color: var(--text-secondary); font-size: 1.1rem; font-family: 'JetBrains Mono', monospace;">
            Hedge Fund Grade Market Regime Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize engine
    engine = MarketRegimeEngine()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Symbol selection
        symbols = load_symbols()
        
        # Add index options
        index_symbols = ["^NSEI", "^NSEBANK", "^BSESN"]
        all_symbols = index_symbols + symbols
        
        selected_symbol = st.selectbox(
            "Select Symbol",
            all_symbols,
            index=0,
            help="Select a stock or index to analyze"
        )
        
        # Custom symbol input
        custom_symbol = st.text_input("Or enter custom symbol", placeholder="e.g., AAPL")
        if custom_symbol:
            selected_symbol = custom_symbol.upper()
            if not selected_symbol.endswith('.NS') and not selected_symbol.startswith('^'):
                selected_symbol += '.NS'
        
        period = st.selectbox(
            "Analysis Period",
            ["6mo", "1y", "2y", "5y"],
            index=1
        )
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Mode")
        
        analysis_mode = st.radio(
            "Mode",
            ["Current Regime", "Historical Analysis"],
            help="Current: Latest regime signal\nHistorical: Regime evolution over time"
        )
        
        if analysis_mode == "Historical Analysis":
            history_window = st.slider(
                "History Window (days)",
                min_value=50,
                max_value=252,
                value=100,
                help="Number of bars to analyze historically"
            )
    
    # Main content
    if st.sidebar.button("🔍 Analyze Regime", type="primary", use_container_width=True):
        with st.spinner(f"Fetching data for {selected_symbol}..."):
            df = fetch_market_data(selected_symbol, period)
        
        if df is None or len(df) < 100:
            st.error(f"Insufficient data for {selected_symbol}. Please try a different symbol or period.")
            return
        
        prices = df['close']
        
        # Current regime detection
        with st.spinner("Detecting market regime..."):
            signal = engine.detect_regime(prices, df)
        
        # Display Results
        st.markdown("---")
        
        # Top row: Primary regime and oscillator
        col1, col2, col3 = st.columns([1, 1.5, 1])
        
        with col1:
            render_regime_card(signal)
        
        with col2:
            render_oscillator_gauge(signal.composite_oscillator)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px;">
                    RECOMMENDED EXPOSURE
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; font-weight: 700; color: var(--accent-primary);">
                    {signal.recommended_exposure*100:.0f}%
                </div>
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 12px;">
                    Transition Risk: <span style="color: {'#ef4444' if signal.transition_probability > 0.3 else '#f59e0b' if signal.transition_probability > 0.15 else '#00d4aa'};">
                        {signal.transition_probability*100:.1f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sub-regime indicators
        st.markdown("### 📈 Sub-Regime Analysis")
        
        cols = st.columns(5)
        
        with cols[0]:
            vol_color = "#00d4aa" if signal.volatility_regime == VolatilityRegime.COMPRESSED else "#f59e0b" if signal.volatility_regime == VolatilityRegime.NORMAL else "#ef4444"
            st.markdown(f"""
            <div class="sub-regime-card">
                <div class="sub-regime-label">Volatility</div>
                <div class="sub-regime-value" style="color: {vol_color};">{signal.volatility_regime.value}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">{signal.volatility_percentile:.0f}th percentile</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            mom_color = "#00d4aa" if 'BULLISH' in signal.momentum_regime.value else "#ef4444" if 'BEARISH' in signal.momentum_regime.value else "#f59e0b"
            st.markdown(f"""
            <div class="sub-regime-card">
                <div class="sub-regime-label">Momentum</div>
                <div class="sub-regime-value" style="color: {mom_color};">{signal.momentum_regime.value.replace('_', ' ')}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Score: {signal.momentum_score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            trend_color = "#00d4aa" if 'UP' in signal.trend_regime.value else "#ef4444" if 'DOWN' in signal.trend_regime.value else "#f59e0b"
            st.markdown(f"""
            <div class="sub-regime-card">
                <div class="sub-regime-label">Trend</div>
                <div class="sub-regime-value" style="color: {trend_color};">{signal.trend_regime.value.replace('_', ' ')}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">ADX: {signal.trend_strength:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[3]:
            risk_color = "#00d4aa" if signal.risk_score < 40 else "#f59e0b" if signal.risk_score < 70 else "#ef4444"
            st.markdown(f"""
            <div class="sub-regime-card">
                <div class="sub-regime-label">Risk</div>
                <div class="sub-regime-value" style="color: {risk_color};">{signal.risk_score:.0f}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">VaR95: {signal.sub_signals.get('var_95', 0):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[4]:
            strength_color = "#00d4aa" if signal.signal_strength > 0.6 else "#f59e0b" if signal.signal_strength > 0.4 else "#6b7280"
            st.markdown(f"""
            <div class="sub-regime-card">
                <div class="sub-regime-label">Signal Strength</div>
                <div class="sub-regime-value" style="color: {strength_color};">{signal.signal_strength*100:.0f}%</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Conviction Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts
        if analysis_mode == "Historical Analysis":
            with st.spinner("Calculating historical regimes..."):
                analyzer = RegimeHistoryAnalyzer(engine)
                regime_history = analyzer.calculate_regime_history(prices, history_window)
        else:
            regime_history = None
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart", "📈 Probabilities", "🎯 Radar", "📋 Details"])
        
        with tab1:
            chart = create_regime_chart(df, regime_history)
            st.plotly_chart(chart, use_container_width=True)
        
        with tab2:
            prob_chart = create_regime_probability_chart(signal.regime_probabilities)
            st.plotly_chart(prob_chart, use_container_width=True)
        
        with tab3:
            radar_chart = create_sub_regime_chart(signal)
            st.plotly_chart(radar_chart, use_container_width=True)
        
        with tab4:
            st.markdown("#### Detailed Signal Components")
            
            detail_cols = st.columns(3)
            
            with detail_cols[0]:
                st.markdown("**Momentum Indicators**")
                st.write(f"- RSI: {signal.sub_signals.get('rsi', 0):.1f}")
                st.write(f"- MACD Histogram: {signal.sub_signals.get('macd_hist', 0):.4f}")
                st.write(f"- Momentum Score: {signal.momentum_score:.1f}")
            
            with detail_cols[1]:
                st.markdown("**Trend Indicators**")
                st.write(f"- Trend Strength (ADX): {signal.trend_strength:.1f}")
                st.write(f"- Trend Slope: {signal.sub_signals.get('trend_slope', 0):.4f}%/day")
            
            with detail_cols[2]:
                st.markdown("**Risk Metrics**")
                st.write(f"- Current Drawdown: {signal.sub_signals.get('drawdown', 0):.2f}%")
                st.write(f"- VaR 95%: {signal.sub_signals.get('var_95', 0):.2f}%")
                st.write(f"- Volatility Z-Score: {signal.sub_signals.get('vol_zscore', 0):.2f}")
            
            st.markdown("---")
            
            st.markdown("#### Regime Interpretation")
            
            regime_descriptions = {
                RegimeType.CRISIS: "**CRISIS**: Extreme bearish conditions with high volatility. Maximum capital protection advised. Consider defensive positions only.",
                RegimeType.BEAR_ACCELERATION: "**BEAR ACCELERATION**: Downtrend gaining momentum. Avoid longs, consider hedging existing positions.",
                RegimeType.BEAR_DECELERATION: "**BEAR DECELERATION**: Downtrend losing steam. Watch for potential reversal signals. Begin building watchlists.",
                RegimeType.ACCUMULATION: "**ACCUMULATION**: Sideways consolidation with low volatility. Smart money accumulating. Prepare for breakout.",
                RegimeType.EARLY_BULL: "**EARLY BULL**: New uptrend emerging. High-conviction long entries. Trend-following strategies optimal.",
                RegimeType.BULL_TREND: "**BULL TREND**: Established uptrend with strong momentum. Maintain long exposure. Trail stops higher.",
                RegimeType.BULL_EUPHORIA: "**BULL EUPHORIA**: Overbought conditions with excessive optimism. Tighten stops, take partial profits.",
                RegimeType.DISTRIBUTION: "**DISTRIBUTION**: Smart money distributing to retail. Reduce long exposure, raise cash.",
                RegimeType.CHOP: "**CHOP**: No clear trend, rangebound action. Reduce position sizes, avoid trend strategies.",
                RegimeType.TRANSITION: "**TRANSITION**: Regime change in progress. High uncertainty. Wait for confirmation before acting.",
            }
            
            st.markdown(regime_descriptions.get(signal.primary_regime, "Unknown regime"))
        
        # Historical regime analysis
        if regime_history is not None and len(regime_history) > 0:
            st.markdown("---")
            st.markdown("### 📜 Regime History Statistics")
            
            regime_counts = regime_history['regime'].value_counts()
            regime_pcts = (regime_counts / len(regime_history) * 100).round(1)
            
            hist_cols = st.columns(5)
            for i, (regime, pct) in enumerate(regime_pcts.items()):
                if i < 5:
                    color = get_regime_color(RegimeType(regime))
                    with hist_cols[i]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 10px; background: var(--bg-tertiary); border-radius: 8px;">
                            <div style="font-size: 0.75rem; color: var(--text-muted);">{regime.replace('_', ' ')}</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{pct}%</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    else:
        # Show placeholder
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; background: var(--bg-secondary); border-radius: 16px; margin-top: 40px;">
            <div style="font-size: 4rem; margin-bottom: 20px;">🌊</div>
            <h3 style="color: var(--text-secondary); font-weight: 400;">Select a symbol and click Analyze to detect market regime</h3>
            <p style="color: var(--text-muted); margin-top: 16px;">
                AVASTHA uses multi-model synthesis including volatility clustering, momentum phase detection,<br>
                trend state identification, and HMM-inspired state detection.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
