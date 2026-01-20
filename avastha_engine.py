"""
AVASTHA (आवस्था) - Core Market Regime Detection Engine
======================================================

A hedge-fund grade multi-model market regime detection system.

This module can be imported and used independently of the Streamlit UI.

Example Usage:
-------------
    from avastha_engine import MarketRegimeEngine, RegimeType
    import yfinance as yf
    
    # Fetch data
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="1y")
    prices = df['Close']
    
    # Initialize engine
    engine = MarketRegimeEngine()
    
    # Detect regime
    signal = engine.detect_regime(prices)
    
    print(f"Current Regime: {signal.primary_regime.value}")
    print(f"Confidence: {signal.primary_confidence*100:.1f}%")
    print(f"Oscillator: {signal.composite_oscillator:.1f}")
    print(f"Recommended Exposure: {signal.recommended_exposure*100:.0f}%")

Author: Quantitative Research
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeType(Enum):
    """Primary regime classifications with market state semantics"""
    CRISIS = "CRISIS"                       # Extreme bearish, high vol, capitulation
    BEAR_ACCELERATION = "BEAR_ACCELERATION" # Downtrend gaining momentum
    BEAR_DECELERATION = "BEAR_DECELERATION" # Downtrend losing steam, potential reversal
    ACCUMULATION = "ACCUMULATION"           # Sideways, low vol, smart money buying
    EARLY_BULL = "EARLY_BULL"               # New uptrend emerging
    BULL_TREND = "BULL_TREND"               # Established uptrend with momentum
    BULL_EUPHORIA = "BULL_EUPHORIA"         # Overbought, excessive optimism
    DISTRIBUTION = "DISTRIBUTION"           # Topping pattern, smart money selling
    CHOP = "CHOP"                           # Rangebound, no clear direction
    TRANSITION = "TRANSITION"               # Regime change in progress


class VolatilityRegime(Enum):
    """Volatility state classification"""
    COMPRESSED = "COMPRESSED"  # Low vol, often precedes big moves
    NORMAL = "NORMAL"          # Average volatility
    ELEVATED = "ELEVATED"      # Above average, caution advised
    EXTREME = "EXTREME"        # Crisis-level volatility


class MomentumRegime(Enum):
    """Momentum state classification"""
    STRONG_BEARISH = "STRONG_BEARISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    STRONG_BULLISH = "STRONG_BULLISH"


class TrendRegime(Enum):
    """Trend state classification"""
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"
    UPTREND = "UPTREND"
    STRONG_UPTREND = "STRONG_UPTREND"


@dataclass
class RegimeSignal:
    """
    Comprehensive regime signal output containing all detection results.
    
    Attributes:
        primary_regime: Main regime classification
        primary_confidence: Confidence level [0-1] for primary regime
        regime_probabilities: Probability distribution across all regimes
        volatility_regime: Current volatility state
        volatility_percentile: Where current vol stands historically [0-100]
        momentum_regime: Current momentum state
        momentum_score: Composite momentum score [-100, 100]
        trend_regime: Current trend state
        trend_strength: ADX-style trend strength [0-100]
        risk_score: Composite risk score [0-100]
        transition_probability: Probability of regime change [0-1]
        recommended_exposure: Suggested portfolio exposure [0-1.5]
        signal_strength: Overall conviction level [0-1]
        composite_oscillator: Unified regime oscillator [-100, 100]
        sub_signals: Dictionary of underlying indicator values
    """
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization"""
        return {
            'primary_regime': self.primary_regime.value,
            'primary_confidence': self.primary_confidence,
            'regime_probabilities': self.regime_probabilities,
            'volatility_regime': self.volatility_regime.value,
            'volatility_percentile': self.volatility_percentile,
            'momentum_regime': self.momentum_regime.value,
            'momentum_score': self.momentum_score,
            'trend_regime': self.trend_regime.value,
            'trend_strength': self.trend_strength,
            'risk_score': self.risk_score,
            'transition_probability': self.transition_probability,
            'recommended_exposure': self.recommended_exposure,
            'signal_strength': self.signal_strength,
            'composite_oscillator': self.composite_oscillator,
            'sub_signals': self.sub_signals,
        }
    
    def is_bullish(self) -> bool:
        """Check if regime is bullish"""
        return self.primary_regime in [
            RegimeType.EARLY_BULL, 
            RegimeType.BULL_TREND, 
            RegimeType.ACCUMULATION
        ]
    
    def is_bearish(self) -> bool:
        """Check if regime is bearish"""
        return self.primary_regime in [
            RegimeType.CRISIS, 
            RegimeType.BEAR_ACCELERATION, 
            RegimeType.DISTRIBUTION
        ]
    
    def is_defensive(self) -> bool:
        """Check if defensive positioning is advised"""
        return (
            self.primary_regime in [RegimeType.CRISIS, RegimeType.DISTRIBUTION] or
            self.risk_score > 70 or
            self.volatility_regime == VolatilityRegime.EXTREME
        )


@dataclass 
class RegimeConfig:
    """Configuration for regime detection engine"""
    volatility_lookback: int = 20
    momentum_fast: int = 12
    momentum_slow: int = 26
    trend_short: int = 20
    trend_medium: int = 50
    trend_long: int = 200
    hmm_n_states: int = 5
    vol_percentile_window: int = 252
    smoothing_factor: float = 0.3
    transition_sensitivity: float = 0.7
    risk_free_rate: float = 0.05
    
    def to_dict(self) -> Dict:
        return {
            'volatility_lookback': self.volatility_lookback,
            'momentum_fast': self.momentum_fast,
            'momentum_slow': self.momentum_slow,
            'trend_short': self.trend_short,
            'trend_medium': self.trend_medium,
            'trend_long': self.trend_long,
            'hmm_n_states': self.hmm_n_states,
            'vol_percentile_window': self.vol_percentile_window,
            'smoothing_factor': self.smoothing_factor,
            'transition_sensitivity': self.transition_sensitivity,
            'risk_free_rate': self.risk_free_rate,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION COMPONENTS (Abstract Base)
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeDetector(ABC):
    """Abstract base class for regime detection components"""
    
    @abstractmethod
    def detect(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Detect regime state"""
        pass


class VolatilityRegimeDetector(RegimeDetector):
    """
    Volatility regime detection using GARCH-inspired clustering.
    
    Analyzes:
    - Multi-scale realized volatility (5d, 20d, 60d)
    - Volatility percentile ranking
    - Volatility clustering (autocorrelation of squared returns)
    - Volatility term structure
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def detect(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        n = len(returns)
        if n < 30:
            return {
                'regime': VolatilityRegime.NORMAL,
                'percentile': 50.0,
                'current_vol': 0.0,
                'vol_zscore': 0.0,
                'vol_autocorr': 0.0,
                'vol_term_structure': 1.0,
                'metrics': {}
            }
        
        # Realized volatility at multiple scales
        vol_5d = returns.rolling(5).std() * np.sqrt(252)
        vol_20d = returns.rolling(20).std() * np.sqrt(252)
        vol_60d = returns.rolling(60).std() * np.sqrt(252)
        
        current_vol = vol_20d.iloc[-1] if pd.notna(vol_20d.iloc[-1]) else returns.std() * np.sqrt(252)
        
        # Volatility percentile
        lookback = min(self.config.vol_percentile_window, n)
        vol_history = vol_20d.dropna().tail(lookback)
        vol_percentile = stats.percentileofscore(vol_history, current_vol) if len(vol_history) > 0 else 50.0
        
        # Volatility clustering (GARCH-style autocorrelation)
        squared_returns = returns ** 2
        vol_autocorr = squared_returns.autocorr(lag=1) if n > 10 else 0
        
        # Volatility z-score
        vol_zscore = (current_vol - vol_20d.mean()) / (vol_20d.std() + 1e-8)
        
        # Volatility term structure
        vol_term_structure = vol_5d.iloc[-1] / vol_60d.iloc[-1] if pd.notna(vol_60d.iloc[-1]) and vol_60d.iloc[-1] > 0 else 1.0
        
        # Classify regime
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


class MomentumRegimeDetector(RegimeDetector):
    """
    Momentum regime detection using multi-factor analysis.
    
    Analyzes:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Rate of Change at multiple timeframes
    - Momentum divergences
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def detect(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        n = len(prices)
        if n < 30:
            return {
                'regime': MomentumRegime.NEUTRAL,
                'score': 0.0,
                'rsi': 50.0,
                'macd_histogram': 0.0,
                'bullish_divergence': False,
                'bearish_divergence': False,
                'metrics': {}
            }
        
        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).ewm(span=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # MACD
        ema_fast = prices.ewm(span=self.config.momentum_fast).mean()
        ema_slow = prices.ewm(span=self.config.momentum_slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=9).mean()
        macd_histogram = macd - signal_line
        
        # Rate of change
        roc_5 = (prices / prices.shift(5) - 1) * 100
        roc_20 = (prices / prices.shift(20) - 1) * 100
        roc_60 = (prices / prices.shift(60) - 1) * 100
        
        # Momentum acceleration
        mom_accel = roc_5.diff(5)
        
        # Composite momentum score [-100, 100]
        rsi_component = (current_rsi - 50) * 2
        macd_component = np.tanh(macd_histogram.iloc[-1] / prices.iloc[-1] * 100) * 100
        roc_component = np.clip(roc_20.iloc[-1] * 2, -100, 100) if pd.notna(roc_20.iloc[-1]) else 0
        
        momentum_score = (
            rsi_component * 0.35 +
            macd_component * 0.35 +
            roc_component * 0.30
        )
        
        # Divergence detection
        price_higher = prices.iloc[-1] > prices.iloc[-20] if n > 20 else False
        rsi_lower = current_rsi < rsi.iloc[-20] if n > 20 else False
        bearish_divergence = price_higher and rsi_lower
        
        price_lower = prices.iloc[-1] < prices.iloc[-20] if n > 20 else False
        rsi_higher = current_rsi > rsi.iloc[-20] if n > 20 else False
        bullish_divergence = price_lower and rsi_higher
        
        # Classify regime
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


class TrendRegimeDetector(RegimeDetector):
    """
    Trend regime detection using multi-timeframe analysis.
    
    Analyzes:
    - Moving average alignment (short/medium/long)
    - Price position relative to MAs
    - ADX-style trend strength
    - Linear regression slope
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def detect(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        n = len(prices)
        if n < 30:
            return {
                'regime': TrendRegime.SIDEWAYS,
                'strength': 0.0,
                'slope': 0.0,
                'r_squared': 0.0,
                'alignment_score': 2.5,
                'metrics': {}
            }
        
        # Moving averages
        ma_short = prices.rolling(self.config.trend_short).mean()
        ma_medium = prices.rolling(self.config.trend_medium).mean()
        ma_long = prices.rolling(min(self.config.trend_long, n)).mean()
        
        current_price = prices.iloc[-1]
        
        # Price position relative to MAs
        above_short = current_price > ma_short.iloc[-1] if pd.notna(ma_short.iloc[-1]) else False
        above_medium = current_price > ma_medium.iloc[-1] if pd.notna(ma_medium.iloc[-1]) else False
        above_long = current_price > ma_long.iloc[-1] if pd.notna(ma_long.iloc[-1]) else False
        
        # MA alignment
        ma_short_above_medium = ma_short.iloc[-1] > ma_medium.iloc[-1] if pd.notna(ma_medium.iloc[-1]) else False
        ma_medium_above_long = ma_medium.iloc[-1] > ma_long.iloc[-1] if pd.notna(ma_long.iloc[-1]) else False
        
        # ADX-style trend strength
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        tr = high - low
        atr = tr.rolling(14).mean()
        
        plus_dm = prices.diff().where(prices.diff() > 0, 0)
        minus_dm = (-prices.diff()).where(prices.diff() < 0, 0)
        
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr + 1e-8)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr + 1e-8)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(14).mean()
        
        trend_strength = adx.iloc[-1] if pd.notna(adx.iloc[-1]) else 0
        
        # Linear regression slope
        if n >= 20:
            x = np.arange(20)
            y = prices.tail(20).values
            slope, _, r_value, _, _ = stats.linregress(x, y)
            r_squared = r_value ** 2
            normalized_slope = slope / prices.iloc[-1] * 100
        else:
            normalized_slope = 0
            r_squared = 0
        
        # Alignment score
        alignment_score = sum([above_short, above_medium, above_long, ma_short_above_medium, ma_medium_above_long])
        
        # Classify regime
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


class RiskRegimeDetector(RegimeDetector):
    """
    Risk regime detection using comprehensive risk metrics.
    
    Analyzes:
    - Drawdown analysis
    - Value at Risk (VaR)
    - Expected Shortfall (CVaR)
    - Sortino Ratio
    - Tail risk (kurtosis, skewness)
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def detect(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        n = len(returns)
        if n < 30:
            return {
                'risk_score': 50.0,
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'sortino': 0.0,
                'kurtosis': 0.0,
                'skewness': 0.0,
                'metrics': {}
            }
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        current_drawdown = drawdown.iloc[-1]
        max_drawdown = drawdown.min()
        
        # Value at Risk
        var_95 = np.percentile(returns.dropna(), 5) * 100
        var_99 = np.percentile(returns.dropna(), 1) * 100
        
        # Expected Shortfall (CVaR)
        var_threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_threshold]
        cvar_95 = tail_returns.mean() * 100 if len(tail_returns) > 0 else var_95
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        avg_return = returns.mean() * 252
        sortino = (avg_return - self.config.risk_free_rate) / (downside_std + 1e-8)
        
        # Tail risk
        kurtosis = stats.kurtosis(returns.dropna())
        skewness = stats.skew(returns.dropna())
        
        # Composite risk score (0-100)
        drawdown_component = min(abs(current_drawdown) / 20 * 100, 100) * 0.25
        var_component = min(abs(var_95) / 3 * 100, 100) * 0.25
        tail_component = min((kurtosis + 3) / 6 * 100, 100) * 0.20
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


class HMMStateDetector(RegimeDetector):
    """
    Hidden Markov Model-inspired state detection using Gaussian Mixture Models.
    
    Identifies latent market states from return distribution characteristics.
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.scaler = StandardScaler()
    
    def detect(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        n = len(returns)
        if n < 100:
            return {
                'state': 2,
                'probabilities': [0.2] * 5,
                'transition_prob': 0.1
            }
        
        # Prepare features
        features = pd.DataFrame({
            'returns': returns,
            'volatility': returns.rolling(20).std(),
            'momentum': returns.rolling(10).mean(),
        }).dropna()
        
        if len(features) < 50:
            return {
                'state': 2,
                'probabilities': [0.2] * 5,
                'transition_prob': 0.1
            }
        
        # Standardize
        X = self.scaler.fit_transform(features)
        
        # Fit GMM
        n_states = self.config.hmm_n_states
        gmm = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
        
        try:
            gmm.fit(X)
            current_state = gmm.predict(X[-1:].reshape(1, -1))[0]
            state_probabilities = gmm.predict_proba(X[-1:].reshape(1, -1))[0].tolist()
            
            # Transition probability
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


class StructuralBreakDetector(RegimeDetector):
    """
    Structural break / change point detection in price series.
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def detect(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        n = len(prices)
        if n < 60:
            return {
                'recent_break': False,
                'break_intensity': 0.0,
                'ma_crossover_recent': False,
                'deviation_zscore': 0.0
            }
        
        # Rolling mean shift detection
        short_ma = prices.rolling(10).mean()
        long_ma = prices.rolling(50).mean()
        
        # Deviation from long-term trend
        deviation = (short_ma - long_ma) / long_ma * 100
        
        # Z-score of deviation
        dev_zscore = (deviation - deviation.rolling(50).mean()) / (deviation.rolling(50).std() + 1e-8)
        
        # Recent break detection
        recent_dev_zscore = dev_zscore.tail(10)
        recent_break = (abs(recent_dev_zscore) > 2.0).any()
        break_intensity = abs(dev_zscore.iloc[-1]) if pd.notna(dev_zscore.iloc[-1]) else 0
        
        # MA crossover
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN REGIME ENGINE
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
    
    Usage:
        engine = MarketRegimeEngine()
        signal = engine.detect_regime(prices)
        print(signal.primary_regime.value)
    """
    
    def __init__(self, config: Optional[Union[RegimeConfig, Dict]] = None):
        if config is None:
            self.config = RegimeConfig()
        elif isinstance(config, dict):
            self.config = RegimeConfig(**config)
        else:
            self.config = config
        
        # Initialize detectors
        self.volatility_detector = VolatilityRegimeDetector(self.config)
        self.momentum_detector = MomentumRegimeDetector(self.config)
        self.trend_detector = TrendRegimeDetector(self.config)
        self.risk_detector = RiskRegimeDetector(self.config)
        self.hmm_detector = HMMStateDetector(self.config)
        self.break_detector = StructuralBreakDetector(self.config)
        
        self._regime_history = []
    
    def _synthesize_primary_regime(
        self,
        vol_regime: Dict,
        mom_regime: Dict,
        trend_regime: Dict,
        risk_metrics: Dict,
        hmm_state: Dict,
        structural_breaks: Dict
    ) -> Tuple[RegimeType, float, Dict[str, float]]:
        """Synthesize all sub-regimes into primary classification"""
        
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
        
        # Initialize scores
        regime_scores = {r.value: 0.0 for r in RegimeType}
        
        # CRISIS
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
            mom_state == MomentumRegime.STRONG_BEARISH):
            regime_scores[RegimeType.BEAR_ACCELERATION.value] += 35
        if trend_strength > 25 and mom_score < -40:
            regime_scores[RegimeType.BEAR_ACCELERATION.value] += 25
        
        # BEAR_DECELERATION
        if (trend_state in [TrendRegime.DOWNTREND, TrendRegime.STRONG_DOWNTREND] and
            mom_regime.get('bullish_divergence', False)):
            regime_scores[RegimeType.BEAR_DECELERATION.value] += 35
        if trend_state == TrendRegime.DOWNTREND and -30 < mom_score < 0:
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
        if trend_regime['alignment_score'] >= 3 and 20 < mom_score < 50:
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
        if trend_regime['alignment_score'] >= 3 and -30 < mom_score < 0:
            regime_scores[RegimeType.DISTRIBUTION.value] += 25
        
        # CHOP
        if (trend_state == TrendRegime.SIDEWAYS and trend_strength < 20 and abs(mom_score) < 25):
            regime_scores[RegimeType.CHOP.value] += 35
        if vol_state == VolatilityRegime.NORMAL and abs(mom_score) < 15:
            regime_scores[RegimeType.CHOP.value] += 20
        
        # TRANSITION
        if transition_prob > 0.3 or break_intensity > 2.0:
            regime_scores[RegimeType.TRANSITION.value] += 30
        if structural_breaks.get('recent_break', False):
            regime_scores[RegimeType.TRANSITION.value] += 20
        
        # Normalize
        total_score = sum(regime_scores.values()) + 1e-8
        regime_probabilities = {k: v / total_score for k, v in regime_scores.items()}
        
        # Primary regime
        primary_regime = RegimeType(max(regime_scores, key=regime_scores.get))
        primary_confidence = regime_probabilities[primary_regime.value]
        
        # Default to CHOP if confidence too low
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
        """Calculate recommended portfolio exposure [0.0 - 1.5]"""
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
        
        base = regime_exposure.get(primary_regime, 0.60)
        confidence_adj = 0.7 + (confidence * 0.6)
        risk_adj = 1.0 - (risk_score / 200)
        vol_adj = 1.0 if vol_percentile < 70 else (1.0 - (vol_percentile - 70) / 100)
        
        return np.clip(base * confidence_adj * risk_adj * vol_adj, 0.0, 1.5)
    
    def _calculate_composite_oscillator(
        self,
        mom_score: float,
        vol_percentile: float,
        trend_strength: float,
        risk_score: float,
        alignment_score: int
    ) -> float:
        """Calculate unified regime oscillator [-100, 100]"""
        mom_component = mom_score * 0.35
        trend_direction = (alignment_score - 2.5) / 2.5
        trend_component = trend_direction * trend_strength * 0.25
        vol_component = (50 - vol_percentile) * 0.5 * 0.20
        risk_component = (50 - risk_score) * 0.20
        
        return np.clip(mom_component + trend_component + vol_component + risk_component, -100, 100)
    
    def detect_regime(self, prices: pd.Series, ohlcv: Optional[pd.DataFrame] = None) -> RegimeSignal:
        """
        Main method: Detect current market regime from price data.
        
        Args:
            prices: Close price series (pd.Series with DatetimeIndex)
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
        
        # Run all detectors
        vol_regime = self.volatility_detector.detect(prices, returns)
        mom_regime = self.momentum_detector.detect(prices, returns)
        trend_regime = self.trend_detector.detect(prices, returns)
        risk_metrics = self.risk_detector.detect(prices, returns)
        hmm_state = self.hmm_detector.detect(prices, returns)
        structural_breaks = self.break_detector.detect(prices, returns)
        
        # Synthesize primary regime
        primary_regime, confidence, regime_probs = self._synthesize_primary_regime(
            vol_regime, mom_regime, trend_regime, risk_metrics, hmm_state, structural_breaks
        )
        
        # Calculate outputs
        recommended_exposure = self._calculate_recommended_exposure(
            primary_regime, confidence, risk_metrics['risk_score'], vol_regime['percentile']
        )
        
        composite_osc = self._calculate_composite_oscillator(
            mom_regime['score'],
            vol_regime['percentile'],
            trend_regime['strength'],
            risk_metrics['risk_score'],
            trend_regime['alignment_score']
        )
        
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
    
    def detect_regime_history(
        self,
        prices: pd.Series,
        lookback_window: int = 100
    ) -> pd.DataFrame:
        """
        Calculate regime signal for each point in history.
        
        Args:
            prices: Close price series
            lookback_window: Minimum history required for detection
            
        Returns:
            DataFrame with regime history
        """
        results = []
        n = len(prices)
        
        for i in range(lookback_window, n):
            window_prices = prices.iloc[:i+1]
            signal = self.detect_regime(window_prices)
            
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
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regime(prices: pd.Series, config: Optional[RegimeConfig] = None) -> RegimeSignal:
    """
    Convenience function to detect regime from price series.
    
    Args:
        prices: Close price series
        config: Optional configuration
        
    Returns:
        RegimeSignal
    """
    engine = MarketRegimeEngine(config)
    return engine.detect_regime(prices)


def get_regime_color(regime: RegimeType) -> str:
    """Get hex color for regime visualization"""
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


def get_regime_description(regime: RegimeType) -> str:
    """Get description for regime type"""
    descriptions = {
        RegimeType.CRISIS: "Extreme bearish conditions with high volatility. Maximum capital protection advised.",
        RegimeType.BEAR_ACCELERATION: "Downtrend gaining momentum. Avoid longs, consider hedging.",
        RegimeType.BEAR_DECELERATION: "Downtrend losing steam. Watch for reversal signals.",
        RegimeType.ACCUMULATION: "Sideways consolidation with low volatility. Smart money accumulating.",
        RegimeType.EARLY_BULL: "New uptrend emerging. High-conviction long entries.",
        RegimeType.BULL_TREND: "Established uptrend with strong momentum. Maintain long exposure.",
        RegimeType.BULL_EUPHORIA: "Overbought conditions with excessive optimism. Tighten stops.",
        RegimeType.DISTRIBUTION: "Smart money distributing. Reduce long exposure.",
        RegimeType.CHOP: "No clear trend, rangebound action. Reduce position sizes.",
        RegimeType.TRANSITION: "Regime change in progress. Wait for confirmation.",
    }
    return descriptions.get(regime, "Unknown regime")


# Export public API
__all__ = [
    'MarketRegimeEngine',
    'RegimeSignal',
    'RegimeConfig',
    'RegimeType',
    'VolatilityRegime',
    'MomentumRegime',
    'TrendRegime',
    'detect_regime',
    'get_regime_color',
    'get_regime_description',
]
