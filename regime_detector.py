"""
AVASTHA - Market Regime Detection Engine

Core regime detection logic using multi-factor analysis across
momentum, trend, breadth, volatility, and statistical extremes.

Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class RegimeType(Enum):
    """Enumeration of market regime types"""
    CRISIS = "CRISIS"
    BEAR = "BEAR"
    WEAK_BEAR = "WEAK_BEAR"
    CHOP = "CHOP"
    WEAK_BULL = "WEAK_BULL"
    BULL = "BULL"
    STRONG_BULL = "STRONG_BULL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class RegimeThreshold:
    """Configuration for regime classification thresholds"""
    score: float
    confidence: float


@dataclass
class FactorAnalysis:
    """Result of a single factor analysis"""
    name: str
    classification: str
    score: float
    metrics: Dict


@dataclass
class RegimeResult:
    """Complete regime detection result"""
    regime: RegimeType
    regime_name: str
    suggested_mix: str
    confidence: float
    composite_score: float
    factors: Dict[str, FactorAnalysis]
    explanation: str
    analysis_date: str
    warnings: List[str]


class MarketRegimeDetector:
    """
    Institutional-grade market regime detection system.
    
    Uses multi-factor analysis to classify current market conditions:
    - Momentum analysis (RSI, Oscillator trends)
    - Trend quality (MA alignment, price position)
    - Market breadth (participation metrics)
    - Volatility regime (Bollinger Width analysis)
    - Statistical extremes (Z-score analysis)
    - Correlation/dispersion analysis
    - Momentum velocity and acceleration
    """
    
    # Factor weights for composite score calculation
    FACTOR_WEIGHTS = {
        'momentum': 0.30,
        'trend': 0.25,
        'breadth': 0.15,
        'volatility': 0.05,
        'extremes': 0.10,
        'correlation': 0.00,
        'velocity': 0.15
    }
    
    # Regime thresholds (score -> regime mapping)
    REGIME_THRESHOLDS = {
        RegimeType.CRISIS: RegimeThreshold(score=-1.0, confidence=0.85),
        RegimeType.BEAR: RegimeThreshold(score=-0.5, confidence=0.75),
        RegimeType.WEAK_BEAR: RegimeThreshold(score=-0.1, confidence=0.65),
        RegimeType.CHOP: RegimeThreshold(score=0.1, confidence=0.60),
        RegimeType.WEAK_BULL: RegimeThreshold(score=0.5, confidence=0.65),
        RegimeType.BULL: RegimeThreshold(score=1.0, confidence=0.75),
        RegimeType.STRONG_BULL: RegimeThreshold(score=1.5, confidence=0.85),
    }
    
    # Regime to portfolio mix mapping
    REGIME_MIX_MAPPING = {
        RegimeType.STRONG_BULL: "Bull Market Mix",
        RegimeType.BULL: "Bull Market Mix",
        RegimeType.WEAK_BULL: "Chop/Consolidation Mix",
        RegimeType.CHOP: "Chop/Consolidation Mix",
        RegimeType.WEAK_BEAR: "Chop/Consolidation Mix",
        RegimeType.BEAR: "Bear Market Mix",
        RegimeType.CRISIS: "Bear Market Mix",
        RegimeType.INSUFFICIENT_DATA: "Bull Market Mix (Default)"
    }
    
    # Regime emoji mapping
    REGIME_EMOJI = {
        RegimeType.STRONG_BULL: "🚀",
        RegimeType.BULL: "🐂",
        RegimeType.WEAK_BULL: "📈",
        RegimeType.CHOP: "📊",
        RegimeType.WEAK_BEAR: "📉",
        RegimeType.BEAR: "🐻",
        RegimeType.CRISIS: "⚠️",
        RegimeType.INSUFFICIENT_DATA: "❓"
    }
    
    def __init__(self, min_periods: int = 10):
        """
        Initialize the regime detector.
        
        Args:
            min_periods: Minimum number of historical periods required for analysis
        """
        self.min_periods = min_periods
    
    def detect(self, historical_data: List[Tuple]) -> RegimeResult:
        """
        Detect current market regime from historical data.
        
        Args:
            historical_data: List of (date, DataFrame) tuples with market data
            
        Returns:
            RegimeResult with complete analysis
        """
        warnings = []
        
        # Validate input data
        if len(historical_data) < self.min_periods:
            return RegimeResult(
                regime=RegimeType.INSUFFICIENT_DATA,
                regime_name="INSUFFICIENT_DATA",
                suggested_mix=self.REGIME_MIX_MAPPING[RegimeType.INSUFFICIENT_DATA],
                confidence=0.30,
                composite_score=0.0,
                factors={},
                explanation=f"⚠️ Insufficient data: {len(historical_data)} periods provided, minimum {self.min_periods} required.",
                analysis_date="N/A",
                warnings=[f"Only {len(historical_data)} periods available"]
            )
        
        # Use last 10 periods for analysis
        analysis_window = historical_data[-self.min_periods:]
        latest_date, latest_df = analysis_window[-1]
        
        # Run all factor analyses
        factors = {
            'momentum': self._analyze_momentum(analysis_window),
            'trend': self._analyze_trend(analysis_window),
            'breadth': self._analyze_breadth(latest_df),
            'volatility': self._analyze_volatility(analysis_window),
            'extremes': self._analyze_extremes(latest_df),
            'correlation': self._analyze_correlation(latest_df),
            'velocity': self._analyze_velocity(analysis_window)
        }
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(factors)
        
        # Classify regime
        regime, confidence = self._classify_regime(composite_score, factors)
        
        # Check for warnings
        if factors['breadth'].classification == 'DIVERGENT':
            warnings.append("Breadth divergence detected - narrow leadership may not be sustainable")
        
        if factors['volatility'].classification == 'PANIC':
            warnings.append("Elevated volatility regime - expect increased market turbulence")
        
        if factors['extremes'].classification in ['DEEPLY_OVERSOLD', 'DEEPLY_OVERBOUGHT']:
            warnings.append(f"Statistical extreme detected: {factors['extremes'].classification}")
        
        # Generate explanation
        explanation = self._generate_explanation(regime, confidence, factors, composite_score)
        
        return RegimeResult(
            regime=regime,
            regime_name=regime.value,
            suggested_mix=self.REGIME_MIX_MAPPING[regime],
            confidence=confidence,
            composite_score=composite_score,
            factors={name: f for name, f in factors.items()},
            explanation=explanation,
            analysis_date=latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date),
            warnings=warnings
        )
    
    def _analyze_momentum(self, window: List[Tuple]) -> FactorAnalysis:
        """Analyze momentum regime using RSI and oscillator trends"""
        try:
            rsi_values = [df['rsi latest'].mean() for _, df in window]
            osc_values = [df['osc latest'].mean() for _, df in window]
            
            current_rsi = rsi_values[-1]
            rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
            current_osc = osc_values[-1]
            osc_trend = np.polyfit(range(len(osc_values)), osc_values, 1)[0]
            
            # Classification logic
            if current_rsi > 65 and rsi_trend > 0.5:
                classification, score = 'STRONG_BULLISH', 2.0
            elif current_rsi > 55 and rsi_trend >= 0:
                classification, score = 'BULLISH', 1.0
            elif current_rsi < 35 and rsi_trend < -0.5:
                classification, score = 'STRONG_BEARISH', -2.0
            elif current_rsi < 45 and rsi_trend <= 0:
                classification, score = 'BEARISH', -1.0
            else:
                classification, score = 'NEUTRAL', 0.0
            
            return FactorAnalysis(
                name='momentum',
                classification=classification,
                score=score,
                metrics={
                    'current_rsi': current_rsi,
                    'rsi_trend': rsi_trend,
                    'current_osc': current_osc,
                    'osc_trend': osc_trend
                }
            )
        except Exception:
            return FactorAnalysis('momentum', 'NEUTRAL', 0.0, {})
    
    def _analyze_trend(self, window: List[Tuple]) -> FactorAnalysis:
        """Analyze trend quality using moving average alignment"""
        try:
            above_ma200_pct = [(df['price'] > df['ma200 latest']).mean() for _, df in window]
            ma_alignment = [(df['ma90 latest'] > df['ma200 latest']).mean() for _, df in window]
            
            current_above_200 = above_ma200_pct[-1]
            current_alignment = ma_alignment[-1]
            trend_consistency = np.polyfit(range(len(above_ma200_pct)), above_ma200_pct, 1)[0]
            
            if current_above_200 > 0.75 and current_alignment > 0.70 and trend_consistency >= 0:
                classification, score = 'STRONG_UPTREND', 2.0
            elif current_above_200 > 0.60 and current_alignment > 0.55:
                classification, score = 'UPTREND', 1.0
            elif current_above_200 < 0.30 and current_alignment < 0.30 and trend_consistency < 0:
                classification, score = 'STRONG_DOWNTREND', -2.0
            elif current_above_200 < 0.45 and current_alignment < 0.45:
                classification, score = 'DOWNTREND', -1.0
            else:
                classification, score = 'TRENDLESS', 0.0
            
            return FactorAnalysis(
                name='trend',
                classification=classification,
                score=score,
                metrics={
                    'above_200dma_pct': current_above_200,
                    'ma_alignment_pct': current_alignment,
                    'trend_consistency': trend_consistency
                }
            )
        except Exception:
            return FactorAnalysis('trend', 'TRENDLESS', 0.0, {})
    
    def _analyze_breadth(self, df: pd.DataFrame) -> FactorAnalysis:
        """Analyze market breadth and participation"""
        try:
            rsi_bullish = (df['rsi latest'] > 50).mean()
            osc_positive = (df['osc latest'] > 0).mean()
            rsi_weak = (df['rsi latest'] < 40).mean()
            osc_oversold = (df['osc latest'] < -50).mean()
            divergence = abs(rsi_bullish - osc_positive)
            
            if rsi_bullish > 0.70 and osc_positive > 0.60 and divergence < 0.15:
                classification, score = 'STRONG_BROAD', 2.0
            elif rsi_bullish > 0.55 and osc_positive > 0.45:
                classification, score = 'HEALTHY', 1.0
            elif rsi_weak > 0.60 and osc_oversold > 0.50:
                classification, score = 'CAPITULATION', -2.0
            elif rsi_weak > 0.45 and osc_oversold > 0.35:
                classification, score = 'WEAK', -1.0
            elif divergence > 0.25:
                classification, score = 'DIVERGENT', -0.5
            else:
                classification, score = 'MIXED', 0.0
            
            return FactorAnalysis(
                name='breadth',
                classification=classification,
                score=score,
                metrics={
                    'rsi_bullish_pct': rsi_bullish,
                    'osc_positive_pct': osc_positive,
                    'rsi_weak_pct': rsi_weak,
                    'osc_oversold_pct': osc_oversold,
                    'divergence': divergence
                }
            )
        except Exception:
            return FactorAnalysis('breadth', 'MIXED', 0.0, {})
    
    def _analyze_volatility(self, window: List[Tuple]) -> FactorAnalysis:
        """Analyze volatility regime using Bollinger Band Width"""
        try:
            bb_widths = [
                ((4 * df['dev20 latest']) / (df['ma20 latest'] + 1e-6)).mean() 
                for _, df in window
            ]
            current_bbw = bb_widths[-1]
            vol_trend = np.polyfit(range(len(bb_widths)), bb_widths, 1)[0]
            
            if current_bbw < 0.08 and vol_trend < 0:
                classification, score = 'SQUEEZE', 0.5
            elif current_bbw > 0.15 and vol_trend > 0:
                classification, score = 'PANIC', -1.0
            elif current_bbw > 0.12:
                classification, score = 'ELEVATED', -0.5
            else:
                classification, score = 'NORMAL', 0.0
            
            return FactorAnalysis(
                name='volatility',
                classification=classification,
                score=score,
                metrics={
                    'current_bbw': current_bbw,
                    'vol_trend': vol_trend
                }
            )
        except Exception:
            return FactorAnalysis('volatility', 'NORMAL', 0.0, {})
    
    def _analyze_extremes(self, df: pd.DataFrame) -> FactorAnalysis:
        """Analyze statistical extremes using Z-scores"""
        try:
            extreme_oversold = (df['zscore latest'] < -2.0).mean()
            extreme_overbought = (df['zscore latest'] > 2.0).mean()
            
            if extreme_oversold > 0.40:
                classification, score = 'DEEPLY_OVERSOLD', 1.5
            elif extreme_overbought > 0.40:
                classification, score = 'DEEPLY_OVERBOUGHT', -1.5
            elif extreme_oversold > 0.20:
                classification, score = 'OVERSOLD', 0.75
            elif extreme_overbought > 0.20:
                classification, score = 'OVERBOUGHT', -0.75
            else:
                classification, score = 'NORMAL', 0.0
            
            return FactorAnalysis(
                name='extremes',
                classification=classification,
                score=score,
                metrics={
                    'oversold_pct': extreme_oversold,
                    'overbought_pct': extreme_overbought
                }
            )
        except Exception:
            return FactorAnalysis('extremes', 'NORMAL', 0.0, {})
    
    def _analyze_correlation(self, df: pd.DataFrame) -> FactorAnalysis:
        """Analyze correlation/dispersion regime"""
        try:
            avg_std = (
                df['rsi latest'].std() / 100 + 
                df['osc latest'].std() / 100 + 
                df['zscore latest'].std() / 5
            ) / 3
            
            if avg_std < 0.15:
                classification, score = 'HIGH_CORRELATION', -0.5
            elif avg_std > 0.30:
                classification, score = 'LOW_CORRELATION', 0.5
            else:
                classification, score = 'NORMAL', 0.0
            
            return FactorAnalysis(
                name='correlation',
                classification=classification,
                score=score,
                metrics={'dispersion': avg_std}
            )
        except Exception:
            return FactorAnalysis('correlation', 'NORMAL', 0.0, {})
    
    def _analyze_velocity(self, window: List[Tuple]) -> FactorAnalysis:
        """Analyze momentum velocity and acceleration"""
        try:
            if len(window) < 5:
                return FactorAnalysis('velocity', 'UNKNOWN', 0.0, {})
            
            recent_rsis = [w[1]['rsi latest'].mean() for w in window[-5:]]
            rsi_changes = np.diff(recent_rsis)
            avg_velocity = np.mean(rsi_changes)
            acceleration = rsi_changes[-1] - rsi_changes[0]
            
            if avg_velocity > 2 and acceleration > 0:
                classification, score = 'ACCELERATING_UP', 1.0
            elif avg_velocity > 1:
                classification, score = 'RISING', 0.5
            elif avg_velocity < -2 and acceleration < 0:
                classification, score = 'ACCELERATING_DOWN', -1.0
            elif avg_velocity < -1:
                classification, score = 'FALLING', -0.5
            else:
                classification, score = 'STABLE', 0.0
            
            return FactorAnalysis(
                name='velocity',
                classification=classification,
                score=score,
                metrics={
                    'avg_velocity': avg_velocity,
                    'acceleration': acceleration
                }
            )
        except Exception:
            return FactorAnalysis('velocity', 'STABLE', 0.0, {})
    
    def _calculate_composite_score(self, factors: Dict[str, FactorAnalysis]) -> float:
        """Calculate weighted composite score from all factors"""
        return sum(
            factors[name].score * weight 
            for name, weight in self.FACTOR_WEIGHTS.items()
            if name in factors
        )
    
    def _classify_regime(
        self, 
        score: float, 
        factors: Dict[str, FactorAnalysis]
    ) -> Tuple[RegimeType, float]:
        """Classify regime based on composite score and factor analysis"""
        
        # Special case: Crisis detection
        if (factors['volatility'].classification == 'PANIC' and 
            score < -0.5 and 
            factors['breadth'].classification == 'CAPITULATION'):
            return RegimeType.CRISIS, 0.90
        
        # Standard classification based on score thresholds
        sorted_thresholds = sorted(
            self.REGIME_THRESHOLDS.items(), 
            key=lambda x: x[1].score
        )
        
        for regime, threshold in reversed(sorted_thresholds):
            if score >= threshold.score:
                confidence = threshold.confidence
                # Reduce confidence if breadth is divergent
                if factors['breadth'].classification == 'DIVERGENT':
                    confidence *= 0.75
                return regime, confidence
        
        return RegimeType.CRISIS, 0.85
    
    def _generate_explanation(
        self, 
        regime: RegimeType, 
        confidence: float, 
        factors: Dict[str, FactorAnalysis],
        score: float
    ) -> str:
        """Generate human-readable explanation of regime detection"""
        
        rationales = {
            RegimeType.STRONG_BULL: "Strong upward momentum with broad participation. Favor momentum strategies.",
            RegimeType.BULL: "Positive trend with healthy breadth. Conditions support growth strategies.",
            RegimeType.WEAK_BULL: "Uptrend showing signs of fatigue or divergence. Rotate to defensive positions.",
            RegimeType.CHOP: "No clear directional bias. Favors mean reversion and relative value strategies.",
            RegimeType.WEAK_BEAR: "Downtrend developing. Begin defensive positioning.",
            RegimeType.BEAR: "Established downtrend with weak breadth. Favor defensive strategies.",
            RegimeType.CRISIS: "Severe market stress. Focus on capital preservation and oversold opportunities.",
            RegimeType.INSUFFICIENT_DATA: "Insufficient data for regime classification."
        }
        
        emoji = self.REGIME_EMOJI.get(regime, "")
        
        lines = [
            f"**Detected Regime:** {emoji} {regime.value} (Score: {score:.2f}, Confidence: {confidence:.0%})",
            "",
            f"**Rationale:** {rationales.get(regime, 'Market conditions unclear.')}"
        ]
        
        # Add warnings
        if factors['breadth'].classification == 'DIVERGENT':
            lines.append("")
            lines.append("⚠️ **Warning:** Breadth divergence detected - narrow leadership may not be sustainable.")
        
        # Add key factor details
        lines.append("")
        lines.append("**Key Factors:**")
        
        momentum = factors['momentum']
        lines.append(f"• **Momentum:** {momentum.classification} (RSI: {momentum.metrics.get('current_rsi', 0):.1f})")
        
        trend = factors['trend']
        lines.append(f"• **Trend:** {trend.classification} ({trend.metrics.get('above_200dma_pct', 0):.0%} > 200DMA)")
        
        breadth = factors['breadth']
        lines.append(f"• **Breadth:** {breadth.classification} ({breadth.metrics.get('rsi_bullish_pct', 0):.0%} bullish)")
        
        volatility = factors['volatility']
        lines.append(f"• **Volatility:** {volatility.classification} (BBW: {volatility.metrics.get('current_bbw', 0):.3f})")
        
        extremes = factors['extremes']
        if extremes.classification != 'NORMAL':
            lines.append(f"• **Extremes:** {extremes.classification} detected")
        
        velocity = factors['velocity']
        if velocity.classification not in ['STABLE', 'UNKNOWN']:
            lines.append(f"• **Velocity:** {velocity.classification}")
        
        return "\n".join(lines)
    
    def get_regime_emoji(self, regime: RegimeType) -> str:
        """Get emoji for regime type"""
        return self.REGIME_EMOJI.get(regime, "")
    
    def get_regime_color(self, regime: RegimeType) -> str:
        """Get color class for regime type"""
        bull_regimes = [RegimeType.STRONG_BULL, RegimeType.BULL]
        bear_regimes = [RegimeType.BEAR, RegimeType.CRISIS]
        
        if regime in bull_regimes:
            return "success"
        elif regime in bear_regimes:
            return "danger"
        else:
            return "warning"
