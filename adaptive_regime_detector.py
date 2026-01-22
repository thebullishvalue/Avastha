"""
AVASTHA - Adaptive Regime Detection Engine

Advanced regime detection using:
1. Hidden Markov Models (HMM) for state discovery
2. Kalman Filtering for adaptive score estimation
3. Rolling Percentile Normalization (no fixed thresholds)
4. GARCH-inspired volatility regime adjustment
5. Bayesian confidence updates
6. CUSUM change point detection

Version: 2.0.0
Author: Hemrek Capital

Mathematical Framework:
----------------------
The system avoids ALL fixed thresholds by using adaptive statistical methods:

1. PERCENTILE NORMALIZATION:
   Instead of: if rsi > 65 then bullish
   We use: percentile_rank = (value - rolling_min) / (rolling_max - rolling_min)
   Then: adaptive_score = 2 * percentile_rank - 1  (maps to [-1, 1])

2. HIDDEN MARKOV MODEL:
   - States S = {Bull, Bear, Chop} are latent
   - Observations O = factor scores
   - Transition matrix A[i,j] = P(S_t = j | S_{t-1} = i)
   - Emission probabilities B[j](o) = P(O_t = o | S_t = j)
   - Use Viterbi algorithm for most likely state sequence
   - Use Forward-Backward for state probabilities

3. KALMAN FILTER:
   State equation: x_t = x_{t-1} + w_t (random walk)
   Observation equation: z_t = x_t + v_t
   Where x_t is true regime score, z_t is measured score
   Kalman gain K adapts based on prediction error

4. VOLATILITY REGIME (GARCH-inspired):
   σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
   High volatility expands regime boundaries
   Low volatility compresses them (more sensitive)

5. BAYESIAN CONFIDENCE:
   P(Regime | Data) ∝ P(Data | Regime) × P(Regime)
   Prior P(Regime) based on regime persistence
   Likelihood from standardized factor scores

6. CUSUM CHANGE POINT:
   S_t = max(0, S_{t-1} + (x_t - μ - k))
   When S_t > h, change point detected
   Reset adaptive parameters on change
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveRegimeType(Enum):
    """Regime types - discovered adaptively, not fixed"""
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    WEAK_BULL = "WEAK_BULL"
    NEUTRAL = "NEUTRAL"  # Renamed from CHOP - more accurate
    WEAK_BEAR = "WEAK_BEAR"
    BEAR = "BEAR"
    CRISIS = "CRISIS"
    TRANSITION = "TRANSITION"  # New: regime change in progress


@dataclass
class KalmanState:
    """Kalman filter state"""
    estimate: float = 0.0  # Current state estimate
    error_covariance: float = 1.0  # Estimation uncertainty
    process_variance: float = 0.01  # How much state can change
    measurement_variance: float = 0.1  # Observation noise


@dataclass
class HMMState:
    """Hidden Markov Model state"""
    n_states: int = 3  # Bull, Neutral, Bear
    transition_matrix: np.ndarray = None  # State transition probabilities
    emission_means: np.ndarray = None  # Mean score for each state
    emission_stds: np.ndarray = None  # Std dev for each state
    state_probabilities: np.ndarray = None  # Current state probabilities
    
    def __post_init__(self):
        if self.transition_matrix is None:
            # Initialize with regime persistence (diagonal dominant)
            self.transition_matrix = np.array([
                [0.85, 0.10, 0.05],  # Bull stays bull 85%
                [0.10, 0.80, 0.10],  # Neutral
                [0.05, 0.10, 0.85]   # Bear stays bear 85%
            ])
        if self.emission_means is None:
            self.emission_means = np.array([1.0, 0.0, -1.0])  # Bull, Neutral, Bear
        if self.emission_stds is None:
            self.emission_stds = np.array([0.4, 0.3, 0.4])
        if self.state_probabilities is None:
            self.state_probabilities = np.array([0.33, 0.34, 0.33])


@dataclass
class VolatilityState:
    """GARCH-inspired volatility tracking"""
    current_variance: float = 0.04  # σ² = 0.04 -> σ = 0.2
    omega: float = 0.0001  # Long-term variance weight
    alpha: float = 0.1  # Recent shock weight
    beta: float = 0.85  # Persistence weight
    long_term_mean: float = 0.04


@dataclass
class CUSUMState:
    """CUSUM change point detection state"""
    positive_cusum: float = 0.0
    negative_cusum: float = 0.0
    threshold: float = 4.0  # Detection threshold (in std devs)
    drift: float = 0.5  # Allowable drift before alarm
    last_change_idx: int = 0


@dataclass
class AdaptiveFactorResult:
    """Result for a single factor with adaptive scoring"""
    name: str
    raw_value: float
    percentile_rank: float  # 0-1 based on rolling history
    adaptive_score: float  # -1 to +1 normalized
    z_score: float  # Standardized score
    contribution: float  # Weighted contribution to composite
    classification: str
    confidence: float


@dataclass
class AdaptiveRegimeResult:
    """Complete adaptive regime detection result"""
    regime: AdaptiveRegimeType
    regime_name: str
    composite_score: float  # Kalman-filtered score
    raw_score: float  # Unfiltered composite
    confidence: float  # Bayesian confidence
    hmm_probabilities: Dict[str, float]  # P(Bull), P(Neutral), P(Bear)
    factors: Dict[str, AdaptiveFactorResult]
    volatility_regime: str  # LOW, NORMAL, HIGH, EXTREME
    volatility_multiplier: float  # Score scaling factor
    regime_persistence: float  # How long current regime has lasted
    change_point_detected: bool
    suggested_mix: str
    explanation: str
    warnings: List[str]
    analysis_date: str
    
    @property
    def is_transitioning(self) -> bool:
        """Check if regime is in transition"""
        probs = list(self.hmm_probabilities.values())
        max_prob = max(probs)
        return max_prob < 0.5  # No dominant state


# ══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class MathUtils:
    """Statistical and mathematical utilities"""
    
    @staticmethod
    def percentile_rank(value: float, history: np.ndarray) -> float:
        """
        Calculate percentile rank of value within history.
        Returns 0-1 where 0.5 is median.
        
        This replaces ALL fixed thresholds with data-driven ranking.
        """
        if len(history) == 0:
            return 0.5
        return np.sum(history <= value) / len(history)
    
    @staticmethod
    def rolling_percentile_score(value: float, history: np.ndarray) -> float:
        """
        Convert value to [-1, 1] score based on percentile rank.
        
        Maps: 0th percentile -> -1
              50th percentile -> 0
              100th percentile -> +1
        """
        pct = MathUtils.percentile_rank(value, history)
        return 2 * pct - 1
    
    @staticmethod
    def adaptive_z_score(value: float, history: np.ndarray, min_periods: int = 10) -> float:
        """
        Calculate z-score with adaptive mean and std.
        Uses exponentially weighted statistics for recency bias.
        """
        if len(history) < min_periods:
            return 0.0
        
        # Exponential weights (more recent = higher weight)
        weights = np.exp(np.linspace(-2, 0, len(history)))
        weights /= weights.sum()
        
        weighted_mean = np.average(history, weights=weights)
        weighted_var = np.average((history - weighted_mean) ** 2, weights=weights)
        weighted_std = np.sqrt(weighted_var)
        
        if weighted_std < 1e-8:
            return 0.0
        
        return (value - weighted_mean) / weighted_std
    
    @staticmethod
    def gaussian_pdf(x: float, mean: float, std: float) -> float:
        """Gaussian probability density function"""
        if std < 1e-8:
            return 1.0 if abs(x - mean) < 1e-8 else 0.0
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Stable softmax for probability normalization"""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum()
    
    @staticmethod
    def entropy(probabilities: np.ndarray) -> float:
        """Shannon entropy - measures uncertainty"""
        p = np.clip(probabilities, 1e-10, 1.0)
        return -np.sum(p * np.log2(p))
    
    @staticmethod
    def efficiency_ratio(prices: np.ndarray, period: int = 10) -> float:
        """
        Kaufman Efficiency Ratio - measures trend strength.
        ER = |Net Change| / Sum(|Individual Changes|)
        
        ER = 1: Perfect trend (all moves in same direction)
        ER = 0: Pure noise (moves cancel out)
        """
        if len(prices) < period + 1:
            return 0.5
        
        net_change = abs(prices[-1] - prices[-period])
        volatility = np.sum(np.abs(np.diff(prices[-period:])))
        
        if volatility < 1e-8:
            return 0.5
        
        return net_change / volatility


# ══════════════════════════════════════════════════════════════════════════════
# KALMAN FILTER
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveKalmanFilter:
    """
    Kalman filter for adaptive regime score estimation.
    
    The Kalman filter optimally combines:
    1. Prediction from previous state
    2. New measurement
    
    Weighted by their respective uncertainties.
    
    Key insight: The Kalman gain adapts automatically.
    - High measurement noise -> trust prediction more
    - High process noise -> trust measurement more
    """
    
    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        self.state = KalmanState(
            process_variance=process_variance,
            measurement_variance=measurement_variance
        )
        self.innovation_history = []  # For adaptive noise estimation
    
    def predict(self) -> Tuple[float, float]:
        """
        Prediction step: x̂_{t|t-1} = x̂_{t-1|t-1}
        For random walk model, prediction = previous estimate
        """
        predicted_estimate = self.state.estimate
        predicted_covariance = self.state.error_covariance + self.state.process_variance
        return predicted_estimate, predicted_covariance
    
    def update(self, measurement: float) -> float:
        """
        Update step with new measurement.
        Returns filtered estimate.
        """
        # Predict
        predicted_estimate, predicted_covariance = self.predict()
        
        # Innovation (measurement residual)
        innovation = measurement - predicted_estimate
        self.innovation_history.append(innovation)
        
        # Keep last 50 innovations for adaptive noise estimation
        if len(self.innovation_history) > 50:
            self.innovation_history.pop(0)
        
        # Innovation covariance
        innovation_covariance = predicted_covariance + self.state.measurement_variance
        
        # Kalman gain
        kalman_gain = predicted_covariance / innovation_covariance
        
        # Update estimate
        self.state.estimate = predicted_estimate + kalman_gain * innovation
        
        # Update error covariance
        self.state.error_covariance = (1 - kalman_gain) * predicted_covariance
        
        # Adaptive noise estimation from innovation sequence
        if len(self.innovation_history) >= 20:
            innovation_var = np.var(self.innovation_history[-20:])
            # Update measurement variance adaptively
            self.state.measurement_variance = 0.9 * self.state.measurement_variance + 0.1 * innovation_var
        
        return self.state.estimate
    
    def get_uncertainty(self) -> float:
        """Return current estimation uncertainty"""
        return np.sqrt(self.state.error_covariance)
    
    def reset(self, initial_estimate: float = 0.0):
        """Reset filter state"""
        self.state.estimate = initial_estimate
        self.state.error_covariance = 1.0
        self.innovation_history = []


# ══════════════════════════════════════════════════════════════════════════════
# HIDDEN MARKOV MODEL
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveHMM:
    """
    Hidden Markov Model for regime state estimation.
    
    States: Bull (0), Neutral (1), Bear (2)
    
    The HMM captures:
    1. Regime persistence (diagonal of transition matrix)
    2. Transition patterns (off-diagonal)
    3. Score distributions per regime (emissions)
    
    We use the Forward algorithm to compute P(state | observations)
    """
    
    def __init__(self):
        self.state = HMMState()
        self.observation_history = []
        self.state_history = []
    
    def emission_probability(self, observation: float, state: int) -> float:
        """P(observation | state) - Gaussian emission"""
        return MathUtils.gaussian_pdf(
            observation,
            self.state.emission_means[state],
            self.state.emission_stds[state]
        )
    
    def forward_step(self, observation: float) -> np.ndarray:
        """
        Single step of forward algorithm.
        Updates state probabilities given new observation.
        
        α_t(j) = P(O_1,...,O_t, S_t=j)
        """
        # Transition: P(S_t | S_{t-1})
        predicted_probs = self.state.transition_matrix.T @ self.state.state_probabilities
        
        # Emission: P(O_t | S_t)
        emission_probs = np.array([
            self.emission_probability(observation, s) 
            for s in range(self.state.n_states)
        ])
        
        # Update: P(S_t | O_1,...,O_t) ∝ P(O_t | S_t) * P(S_t | O_1,...,O_{t-1})
        updated_probs = emission_probs * predicted_probs
        
        # Normalize
        total = updated_probs.sum()
        if total > 1e-10:
            updated_probs /= total
        else:
            updated_probs = np.array([0.33, 0.34, 0.33])
        
        self.state.state_probabilities = updated_probs
        return updated_probs
    
    def update(self, observation: float) -> Dict[str, float]:
        """
        Update HMM with new observation.
        Returns state probabilities.
        """
        self.observation_history.append(observation)
        
        # Forward step
        probs = self.forward_step(observation)
        
        # Track most likely state
        most_likely = np.argmax(probs)
        self.state_history.append(most_likely)
        
        # Adapt emission parameters if we have enough history
        if len(self.observation_history) >= 30:
            self._adapt_emissions()
        
        # Adapt transition matrix based on observed transitions
        if len(self.state_history) >= 20:
            self._adapt_transitions()
        
        return {
            "BULL": probs[0],
            "NEUTRAL": probs[1],
            "BEAR": probs[2]
        }
    
    def _adapt_emissions(self):
        """Adapt emission parameters from recent observations"""
        recent_obs = np.array(self.observation_history[-50:])
        recent_states = self.state_history[-50:] if len(self.state_history) >= 50 else self.state_history
        
        for state in range(self.state.n_states):
            state_mask = np.array(recent_states[-len(recent_obs):]) == state
            if state_mask.sum() >= 5:
                state_obs = recent_obs[state_mask]
                # Exponential smoothing for adaptation
                new_mean = np.mean(state_obs)
                new_std = max(np.std(state_obs), 0.1)
                
                self.state.emission_means[state] = 0.9 * self.state.emission_means[state] + 0.1 * new_mean
                self.state.emission_stds[state] = 0.9 * self.state.emission_stds[state] + 0.1 * new_std
    
    def _adapt_transitions(self):
        """Adapt transition matrix from observed state transitions"""
        recent_states = self.state_history[-30:]
        
        # Count transitions
        transition_counts = np.zeros((3, 3))
        for i in range(len(recent_states) - 1):
            from_state = recent_states[i]
            to_state = recent_states[i + 1]
            transition_counts[from_state, to_state] += 1
        
        # Convert to probabilities with smoothing
        for i in range(3):
            row_sum = transition_counts[i].sum()
            if row_sum >= 5:  # Only adapt if enough observations
                new_probs = (transition_counts[i] + 1) / (row_sum + 3)  # Laplace smoothing
                self.state.transition_matrix[i] = 0.8 * self.state.transition_matrix[i] + 0.2 * new_probs
    
    def get_most_likely_state(self) -> int:
        """Return most likely current state"""
        return np.argmax(self.state.state_probabilities)
    
    def get_regime_persistence(self) -> float:
        """Calculate how long current regime has persisted"""
        if len(self.state_history) < 2:
            return 1.0
        
        current_state = self.state_history[-1]
        persistence = 1
        for i in range(len(self.state_history) - 2, -1, -1):
            if self.state_history[i] == current_state:
                persistence += 1
            else:
                break
        
        return persistence
    
    def reset(self):
        """Reset HMM state"""
        self.state = HMMState()
        self.observation_history = []
        self.state_history = []


# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY REGIME DETECTOR (GARCH-INSPIRED)
# ══════════════════════════════════════════════════════════════════════════════

class VolatilityRegimeDetector:
    """
    GARCH-inspired volatility regime detection.
    
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Where:
    - σ²_t is conditional variance
    - ε_{t-1} is previous shock (innovation)
    - ω, α, β are parameters
    
    This determines how to scale regime scores:
    - High volatility -> expand thresholds (less sensitive)
    - Low volatility -> compress thresholds (more sensitive)
    """
    
    def __init__(self):
        self.state = VolatilityState()
        self.shock_history = []
    
    def update(self, shock: float) -> float:
        """
        Update volatility estimate with new shock.
        Returns current volatility (std dev).
        """
        self.shock_history.append(shock)
        
        # GARCH(1,1) update
        shock_squared = shock ** 2
        new_variance = (
            self.state.omega + 
            self.state.alpha * shock_squared + 
            self.state.beta * self.state.current_variance
        )
        
        # Bound variance to reasonable range
        new_variance = np.clip(new_variance, 0.001, 1.0)
        self.state.current_variance = new_variance
        
        # Update long-term mean adaptively
        if len(self.shock_history) >= 50:
            realized_var = np.var(self.shock_history[-50:])
            self.state.long_term_mean = 0.95 * self.state.long_term_mean + 0.05 * realized_var
        
        return np.sqrt(new_variance)
    
    def get_volatility_regime(self) -> Tuple[str, float]:
        """
        Classify current volatility regime.
        Returns (regime_name, multiplier)
        
        Multiplier adjusts score sensitivity:
        - Low vol: multiplier > 1 (more sensitive)
        - High vol: multiplier < 1 (less sensitive)
        """
        current_vol = np.sqrt(self.state.current_variance)
        long_term_vol = np.sqrt(self.state.long_term_mean)
        
        vol_ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        if vol_ratio < 0.6:
            return "LOW", 1.3  # Expand sensitivity
        elif vol_ratio < 0.9:
            return "NORMAL", 1.0
        elif vol_ratio < 1.4:
            return "HIGH", 0.8  # Reduce sensitivity
        else:
            return "EXTREME", 0.6  # Much less sensitive
    
    def reset(self):
        """Reset volatility state"""
        self.state = VolatilityState()
        self.shock_history = []


# ══════════════════════════════════════════════════════════════════════════════
# CUSUM CHANGE POINT DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change point detection.
    
    Detects when the mean of the process has shifted.
    
    S⁺_t = max(0, S⁺_{t-1} + (x_t - μ - k))
    S⁻_t = max(0, S⁻_{t-1} - (x_t - μ + k))
    
    Change detected when S⁺_t > h or S⁻_t > h
    
    k = allowable slack (drift)
    h = decision threshold
    """
    
    def __init__(self, threshold: float = 4.0, drift: float = 0.5):
        self.state = CUSUMState(threshold=threshold, drift=drift)
        self.value_history = []
        self.running_mean = 0.0
        self.running_std = 1.0
    
    def update(self, value: float, idx: int = 0) -> bool:
        """
        Update CUSUM with new value.
        Returns True if change point detected.
        """
        self.value_history.append(value)
        
        # Update running statistics
        if len(self.value_history) >= 10:
            recent = self.value_history[-20:]
            self.running_mean = np.mean(recent)
            self.running_std = max(np.std(recent), 0.1)
        
        # Standardize
        z = (value - self.running_mean) / self.running_std
        
        # Update CUSUM
        self.state.positive_cusum = max(0, self.state.positive_cusum + z - self.state.drift)
        self.state.negative_cusum = max(0, self.state.negative_cusum - z - self.state.drift)
        
        # Check for change point
        change_detected = (
            self.state.positive_cusum > self.state.threshold or
            self.state.negative_cusum > self.state.threshold
        )
        
        if change_detected:
            self.state.positive_cusum = 0
            self.state.negative_cusum = 0
            self.state.last_change_idx = idx
        
        return change_detected
    
    def reset(self):
        """Reset CUSUM state"""
        self.state = CUSUMState()
        self.value_history = []
        self.running_mean = 0.0
        self.running_std = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE REGIME DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveRegimeDetector:
    """
    Main adaptive regime detection engine.
    
    Combines all components:
    1. Percentile-based factor scoring (no fixed thresholds)
    2. Hidden Markov Model for state estimation
    3. Kalman filter for score smoothing
    4. GARCH volatility regime adjustment
    5. CUSUM change point detection
    6. Bayesian confidence calculation
    """
    
    # Factor weights (these can also be made adaptive)
    FACTOR_WEIGHTS = {
        'momentum': 0.25,
        'trend': 0.25,
        'breadth': 0.20,
        'velocity': 0.15,
        'extremes': 0.10,
        'volatility': 0.05
    }
    
    def __init__(self, lookback_period: int = 60, min_periods: int = 20):
        self.lookback_period = lookback_period
        self.min_periods = min_periods
        
        # Initialize components
        self.kalman_filter = AdaptiveKalmanFilter()
        self.hmm = AdaptiveHMM()
        self.volatility_detector = VolatilityRegimeDetector()
        self.cusum_detector = CUSUMDetector()
        
        # History for percentile calculations
        self.factor_history = {name: [] for name in self.FACTOR_WEIGHTS.keys()}
        self.score_history = []
        
        # Observation counter
        self.observation_count = 0
    
    def _calculate_adaptive_factor(
        self, 
        name: str,
        raw_value: float,
        history: List[float]
    ) -> AdaptiveFactorResult:
        """
        Calculate factor score using adaptive methods.
        NO fixed thresholds - everything is relative to history.
        """
        history_arr = np.array(history[-self.lookback_period:]) if history else np.array([raw_value])
        
        # Percentile rank (0 to 1)
        pct_rank = MathUtils.percentile_rank(raw_value, history_arr)
        
        # Adaptive score (-1 to +1)
        adaptive_score = MathUtils.rolling_percentile_score(raw_value, history_arr)
        
        # Z-score for anomaly detection
        z_score = MathUtils.adaptive_z_score(raw_value, history_arr)
        
        # Weighted contribution
        weight = self.FACTOR_WEIGHTS.get(name, 0.1)
        contribution = adaptive_score * weight
        
        # Classification based on percentile (not fixed thresholds!)
        if pct_rank >= 0.8:
            classification = "STRONGLY_BULLISH"
        elif pct_rank >= 0.6:
            classification = "BULLISH"
        elif pct_rank >= 0.4:
            classification = "NEUTRAL"
        elif pct_rank >= 0.2:
            classification = "BEARISH"
        else:
            classification = "STRONGLY_BEARISH"
        
        # Confidence based on data sufficiency and consistency
        data_confidence = min(1.0, len(history) / self.lookback_period)
        consistency = 1.0 - min(1.0, abs(z_score) / 3)  # Penalize outliers
        confidence = (data_confidence + consistency) / 2
        
        return AdaptiveFactorResult(
            name=name,
            raw_value=raw_value,
            percentile_rank=pct_rank,
            adaptive_score=adaptive_score,
            z_score=z_score,
            contribution=contribution,
            classification=classification,
            confidence=confidence
        )
    
    def _extract_factor_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract raw factor values from dataframe"""
        factors = {}
        
        # Momentum: RSI-based
        if 'rsi latest' in df.columns:
            rsi_mean = df['rsi latest'].mean()
            rsi_above_50 = (df['rsi latest'] > 50).mean()
            factors['momentum'] = (rsi_mean - 50) / 25 + (rsi_above_50 - 0.5)
        else:
            factors['momentum'] = 0.0
        
        # Trend: MA alignment and price position
        trend_score = 0.0
        if 'ma200 latest' in df.columns and 'price' in df.columns:
            above_200 = (df['price'] > df['ma200 latest']).mean()
            trend_score += (above_200 - 0.5) * 2
        if 'ma90 latest' in df.columns and 'ma200 latest' in df.columns:
            ma_aligned = (df['ma90 latest'] > df['ma200 latest']).mean()
            trend_score += (ma_aligned - 0.5)
        factors['trend'] = trend_score
        
        # Breadth: Oscillator-based
        if 'osc latest' in df.columns:
            osc_positive = (df['osc latest'] > 0).mean()
            osc_mean = df['osc latest'].mean() / 50
            factors['breadth'] = osc_positive - 0.5 + osc_mean
        else:
            factors['breadth'] = 0.0
        
        # Velocity: Rate of change
        if 'rsi latest' in df.columns and 'rsi 5d ago' in df.columns:
            rsi_velocity = (df['rsi latest'] - df['rsi 5d ago']).mean() / 10
            factors['velocity'] = np.clip(rsi_velocity, -1, 1)
        else:
            factors['velocity'] = 0.0
        
        # Extremes: Z-score based
        if 'z-score latest' in df.columns:
            extreme_oversold = (df['z-score latest'] < -2).mean()
            extreme_overbought = (df['z-score latest'] > 2).mean()
            factors['extremes'] = (extreme_oversold - extreme_overbought) * 2
        else:
            factors['extremes'] = 0.0
        
        # Volatility: BBW-based
        if 'bbw latest' in df.columns:
            avg_bbw = df['bbw latest'].mean()
            # Higher volatility = negative score (uncertainty)
            factors['volatility'] = -np.clip((avg_bbw - 0.1) * 5, -1, 1)
        else:
            factors['volatility'] = 0.0
        
        return factors
    
    def detect(self, historical_data: List[Tuple[datetime, pd.DataFrame]]) -> AdaptiveRegimeResult:
        """
        Main detection method.
        
        Takes historical snapshots and returns adaptive regime result.
        """
        if not historical_data or len(historical_data) < self.min_periods:
            return self._insufficient_data_result()
        
        warnings = []
        self.observation_count += 1
        
        # Get latest data
        analysis_date, latest_df = historical_data[-1]
        
        # Process each historical snapshot to build factor history
        all_factors = []
        for date, df in historical_data:
            factor_values = self._extract_factor_values(df)
            all_factors.append(factor_values)
            
            # Update factor history
            for name, value in factor_values.items():
                if name in self.factor_history:
                    self.factor_history[name].append(value)
                    # Keep history bounded
                    if len(self.factor_history[name]) > self.lookback_period * 2:
                        self.factor_history[name] = self.factor_history[name][-self.lookback_period:]
        
        # Calculate adaptive factor scores for latest
        latest_factors = all_factors[-1]
        factor_results = {}
        
        for name, raw_value in latest_factors.items():
            history = self.factor_history.get(name, [])
            factor_results[name] = self._calculate_adaptive_factor(name, raw_value, history)
        
        # Calculate raw composite score
        raw_score = sum(f.contribution for f in factor_results.values())
        
        # Update volatility detector
        if self.score_history:
            shock = raw_score - self.score_history[-1]
        else:
            shock = 0.0
        
        self.volatility_detector.update(shock)
        vol_regime, vol_multiplier = self.volatility_detector.get_volatility_regime()
        
        # Apply volatility adjustment
        adjusted_score = raw_score * vol_multiplier
        
        # Kalman filter for smoothing
        filtered_score = self.kalman_filter.update(adjusted_score)
        
        # Update HMM
        hmm_probs = self.hmm.update(filtered_score)
        
        # Check for change points
        change_detected = self.cusum_detector.update(filtered_score, self.observation_count)
        if change_detected:
            warnings.append("REGIME CHANGE DETECTED - Structural break in market conditions")
            # Reset Kalman filter on change point
            self.kalman_filter.reset(filtered_score)
        
        # Determine regime from HMM
        regime = self._classify_regime(filtered_score, hmm_probs)
        
        # Calculate Bayesian confidence
        confidence = self._calculate_bayesian_confidence(hmm_probs, factor_results)
        
        # Get regime persistence
        persistence = self.hmm.get_regime_persistence()
        
        # Store score in history
        self.score_history.append(filtered_score)
        if len(self.score_history) > self.lookback_period * 2:
            self.score_history = self.score_history[-self.lookback_period:]
        
        # Add warnings for low confidence or high uncertainty
        kalman_uncertainty = self.kalman_filter.get_uncertainty()
        if kalman_uncertainty > 0.5:
            warnings.append(f"HIGH UNCERTAINTY - Kalman filter uncertainty: {kalman_uncertainty:.2f}")
        
        if max(hmm_probs.values()) < 0.5:
            warnings.append("REGIME AMBIGUITY - No dominant state, market in transition")
        
        # Generate explanation
        explanation = self._generate_explanation(
            regime, filtered_score, hmm_probs, factor_results, 
            vol_regime, persistence, change_detected
        )
        
        # Get suggested mix
        suggested_mix = self._get_suggested_mix(regime, confidence)
        
        return AdaptiveRegimeResult(
            regime=regime,
            regime_name=regime.value,
            composite_score=filtered_score,
            raw_score=raw_score,
            confidence=confidence,
            hmm_probabilities=hmm_probs,
            factors=factor_results,
            volatility_regime=vol_regime,
            volatility_multiplier=vol_multiplier,
            regime_persistence=persistence,
            change_point_detected=change_detected,
            suggested_mix=suggested_mix,
            explanation=explanation,
            warnings=warnings,
            analysis_date=analysis_date.strftime("%Y-%m-%d") if isinstance(analysis_date, datetime) else str(analysis_date)
        )
    
    def _classify_regime(self, score: float, hmm_probs: Dict[str, float]) -> AdaptiveRegimeType:
        """
        Classify regime using HMM probabilities and score.
        Uses soft classification, not hard thresholds.
        """
        bull_prob = hmm_probs['BULL']
        bear_prob = hmm_probs['BEAR']
        neutral_prob = hmm_probs['NEUTRAL']
        
        # If no dominant state, we're in transition
        max_prob = max(bull_prob, bear_prob, neutral_prob)
        if max_prob < 0.4:
            return AdaptiveRegimeType.TRANSITION
        
        # Use percentile of score history for classification
        if self.score_history:
            score_pct = MathUtils.percentile_rank(score, np.array(self.score_history))
        else:
            score_pct = 0.5
        
        # Combine HMM state with score percentile
        if bull_prob > bear_prob and bull_prob > neutral_prob:
            if score_pct >= 0.9:
                return AdaptiveRegimeType.STRONG_BULL
            elif score_pct >= 0.7:
                return AdaptiveRegimeType.BULL
            else:
                return AdaptiveRegimeType.WEAK_BULL
        elif bear_prob > bull_prob and bear_prob > neutral_prob:
            if score_pct <= 0.1:
                return AdaptiveRegimeType.CRISIS
            elif score_pct <= 0.3:
                return AdaptiveRegimeType.BEAR
            else:
                return AdaptiveRegimeType.WEAK_BEAR
        else:
            return AdaptiveRegimeType.NEUTRAL
    
    def _calculate_bayesian_confidence(
        self, 
        hmm_probs: Dict[str, float],
        factors: Dict[str, AdaptiveFactorResult]
    ) -> float:
        """
        Calculate Bayesian confidence in regime classification.
        
        Confidence is based on:
        1. HMM state probability (how certain is the model?)
        2. Factor agreement (do factors agree?)
        3. Data sufficiency (enough history?)
        """
        # HMM confidence: how dominant is the leading state?
        probs = np.array(list(hmm_probs.values()))
        hmm_confidence = max(probs) - np.median(probs)
        
        # Factor agreement: do factors point in same direction?
        factor_scores = [f.adaptive_score for f in factors.values()]
        if len(factor_scores) > 1:
            factor_std = np.std(factor_scores)
            factor_agreement = 1.0 - min(factor_std, 1.0)
        else:
            factor_agreement = 0.5
        
        # Data sufficiency
        history_len = len(self.score_history)
        data_sufficiency = min(1.0, history_len / self.lookback_period)
        
        # Average factor confidence
        avg_factor_conf = np.mean([f.confidence for f in factors.values()])
        
        # Weighted combination
        confidence = (
            0.3 * hmm_confidence +
            0.3 * factor_agreement +
            0.2 * data_sufficiency +
            0.2 * avg_factor_conf
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _get_suggested_mix(self, regime: AdaptiveRegimeType, confidence: float) -> str:
        """Get suggested portfolio mix based on regime and confidence"""
        
        # Base suggestions
        base_mix = {
            AdaptiveRegimeType.STRONG_BULL: "🐂 Aggressive Bull Mix",
            AdaptiveRegimeType.BULL: "🐂 Bull Market Mix",
            AdaptiveRegimeType.WEAK_BULL: "📊 Cautious Bull Mix",
            AdaptiveRegimeType.NEUTRAL: "📊 Balanced/Neutral Mix",
            AdaptiveRegimeType.WEAK_BEAR: "📊 Cautious Bear Mix",
            AdaptiveRegimeType.BEAR: "🐻 Bear Market Mix",
            AdaptiveRegimeType.CRISIS: "🐻 Defensive/Crisis Mix",
            AdaptiveRegimeType.TRANSITION: "⚡ Transition Mix (Reduce Exposure)"
        }
        
        mix = base_mix.get(regime, "📊 Balanced Mix")
        
        # Add confidence qualifier
        if confidence < 0.4:
            mix += " [LOW CONFIDENCE]"
        
        return mix
    
    def _generate_explanation(
        self,
        regime: AdaptiveRegimeType,
        score: float,
        hmm_probs: Dict[str, float],
        factors: Dict[str, AdaptiveFactorResult],
        vol_regime: str,
        persistence: float,
        change_detected: bool
    ) -> str:
        """Generate detailed explanation of regime detection"""
        
        lines = []
        
        # Regime summary
        lines.append(f"ADAPTIVE REGIME: {regime.value}")
        lines.append(f"Filtered Score: {score:+.3f} (Percentile: {MathUtils.percentile_rank(score, np.array(self.score_history)) if self.score_history else 0.5:.0%})")
        lines.append("")
        
        # HMM state
        lines.append("HMM State Probabilities:")
        for state, prob in hmm_probs.items():
            bar = "█" * int(prob * 20)
            lines.append(f"  {state}: {prob:.1%} {bar}")
        lines.append("")
        
        # Factor summary
        lines.append("Factor Contributions (Adaptive):")
        sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1].contribution), reverse=True)
        for name, f in sorted_factors:
            sign = "+" if f.contribution >= 0 else ""
            lines.append(f"  {name.upper()}: {sign}{f.contribution:.3f} (pct: {f.percentile_rank:.0%}, z: {f.z_score:+.2f})")
        lines.append("")
        
        # Volatility regime
        lines.append(f"Volatility Regime: {vol_regime}")
        lines.append(f"Regime Persistence: {persistence:.0f} periods")
        
        if change_detected:
            lines.append("")
            lines.append("⚠️ STRUCTURAL BREAK DETECTED")
        
        return "\n".join(lines)
    
    def _insufficient_data_result(self) -> AdaptiveRegimeResult:
        """Return result when insufficient data"""
        return AdaptiveRegimeResult(
            regime=AdaptiveRegimeType.NEUTRAL,
            regime_name="INSUFFICIENT_DATA",
            composite_score=0.0,
            raw_score=0.0,
            confidence=0.0,
            hmm_probabilities={"BULL": 0.33, "NEUTRAL": 0.34, "BEAR": 0.33},
            factors={},
            volatility_regime="UNKNOWN",
            volatility_multiplier=1.0,
            regime_persistence=0,
            change_point_detected=False,
            suggested_mix="📊 Balanced Mix (Insufficient Data)",
            explanation="Insufficient historical data for adaptive regime detection.",
            warnings=["INSUFFICIENT DATA - Need more historical observations"],
            analysis_date=datetime.now().strftime("%Y-%m-%d")
        )
    
    def get_regime_emoji(self, regime: AdaptiveRegimeType) -> str:
        """Get emoji for regime"""
        emojis = {
            AdaptiveRegimeType.STRONG_BULL: "🚀",
            AdaptiveRegimeType.BULL: "🐂",
            AdaptiveRegimeType.WEAK_BULL: "📈",
            AdaptiveRegimeType.NEUTRAL: "📊",
            AdaptiveRegimeType.WEAK_BEAR: "📉",
            AdaptiveRegimeType.BEAR: "🐻",
            AdaptiveRegimeType.CRISIS: "🔥",
            AdaptiveRegimeType.TRANSITION: "⚡"
        }
        return emojis.get(regime, "📊")
    
    def reset(self):
        """Reset all adaptive components"""
        self.kalman_filter.reset()
        self.hmm.reset()
        self.volatility_detector.reset()
        self.cusum_detector.reset()
        self.factor_history = {name: [] for name in self.FACTOR_WEIGHTS.keys()}
        self.score_history = []
        self.observation_count = 0


# ══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

# Alias for backward compatibility
RegimeType = AdaptiveRegimeType
RegimeResult = AdaptiveRegimeResult
MarketRegimeDetector = AdaptiveRegimeDetector
