"""
AVASTHA Example Usage
=====================

Demonstrates how to use the Market Regime Detection Engine programmatically.
"""

import yfinance as yf
import pandas as pd
from avastha_engine import (
    MarketRegimeEngine,
    RegimeConfig,
    RegimeType,
    detect_regime,
    get_regime_color,
    get_regime_description
)


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Fetch data
    ticker = yf.Ticker("^NSEI")  # Nifty 50
    df = ticker.history(period="1y")
    prices = df['Close']
    
    # Initialize engine
    engine = MarketRegimeEngine()
    
    # Detect regime
    signal = engine.detect_regime(prices)
    
    # Print results
    print(f"\nSymbol: NIFTY 50")
    print(f"Latest Price: {prices.iloc[-1]:.2f}")
    print(f"\n{'─' * 40}")
    print(f"PRIMARY REGIME: {signal.primary_regime.value}")
    print(f"Confidence: {signal.primary_confidence * 100:.1f}%")
    print(f"{'─' * 40}")
    print(f"\nComposite Oscillator: {signal.composite_oscillator:+.1f}")
    print(f"Recommended Exposure: {signal.recommended_exposure * 100:.0f}%")
    print(f"Signal Strength: {signal.signal_strength * 100:.1f}%")
    
    print(f"\n{'─' * 40}")
    print("SUB-REGIMES:")
    print(f"  Volatility: {signal.volatility_regime.value} ({signal.volatility_percentile:.0f}th percentile)")
    print(f"  Momentum: {signal.momentum_regime.value} (score: {signal.momentum_score:.1f})")
    print(f"  Trend: {signal.trend_regime.value} (strength: {signal.trend_strength:.1f})")
    print(f"  Risk Score: {signal.risk_score:.1f}/100")
    
    print(f"\n{'─' * 40}")
    print("INTERPRETATION:")
    print(f"  {get_regime_description(signal.primary_regime)}")
    
    return signal


def example_custom_config():
    """Example with custom configuration"""
    print("\n" + "=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)
    
    # Custom configuration for more responsive detection
    config = RegimeConfig(
        volatility_lookback=15,      # Shorter volatility window
        momentum_fast=8,              # Faster momentum
        momentum_slow=21,
        trend_short=15,               # Shorter trend windows
        trend_medium=40,
        trend_long=150,
        hmm_n_states=4,               # Fewer states for cleaner classification
    )
    
    # Fetch data
    ticker = yf.Ticker("RELIANCE.NS")
    df = ticker.history(period="1y")
    prices = df['Close']
    
    # Initialize with custom config
    engine = MarketRegimeEngine(config)
    signal = engine.detect_regime(prices)
    
    print(f"\nSymbol: RELIANCE")
    print(f"Regime: {signal.primary_regime.value}")
    print(f"Oscillator: {signal.composite_oscillator:+.1f}")
    
    return signal


def example_regime_history():
    """Example calculating historical regimes"""
    print("\n" + "=" * 60)
    print("HISTORICAL REGIME ANALYSIS")
    print("=" * 60)
    
    # Fetch data
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="2y")
    prices = df['Close']
    
    # Initialize engine
    engine = MarketRegimeEngine()
    
    # Calculate historical regimes
    print("\nCalculating historical regimes (this may take a moment)...")
    history = engine.detect_regime_history(prices, lookback_window=100)
    
    # Print regime distribution
    print(f"\nRegime Distribution (last {len(history)} days):")
    regime_counts = history['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(history) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")
    
    # Find regime transitions
    transitions = history[history['regime'] != history['regime'].shift(1)].dropna()
    print(f"\nRegime Transitions: {len(transitions)}")
    print("\nRecent Transitions:")
    for idx, row in transitions.tail(5).iterrows():
        prev_regime = history.loc[:idx, 'regime'].iloc[-2] if len(history.loc[:idx]) > 1 else "N/A"
        print(f"  {idx.strftime('%Y-%m-%d')}: {prev_regime} → {row['regime']}")
    
    return history


def example_multiple_symbols():
    """Example analyzing multiple symbols"""
    print("\n" + "=" * 60)
    print("MULTI-SYMBOL ANALYSIS")
    print("=" * 60)
    
    symbols = ["^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    engine = MarketRegimeEngine()
    results = []
    
    print("\nAnalyzing symbols...\n")
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1y")
            
            if df.empty or len(df) < 100:
                print(f"  {symbol}: Insufficient data")
                continue
            
            prices = df['Close']
            signal = engine.detect_regime(prices)
            
            results.append({
                'Symbol': symbol.replace('.NS', '').replace('^', ''),
                'Regime': signal.primary_regime.value,
                'Confidence': f"{signal.primary_confidence * 100:.0f}%",
                'Oscillator': f"{signal.composite_oscillator:+.1f}",
                'Exposure': f"{signal.recommended_exposure * 100:.0f}%",
                'Risk': f"{signal.risk_score:.0f}",
            })
            
            print(f"  {symbol}: {signal.primary_regime.value} ({signal.primary_confidence * 100:.0f}%)")
            
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
    
    if results:
        print("\n" + "─" * 80)
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
    
    return results


def example_programmatic_integration():
    """Example of programmatic integration with trading logic"""
    print("\n" + "=" * 60)
    print("PROGRAMMATIC INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Fetch data
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="1y")
    prices = df['Close']
    
    # Detect regime
    signal = detect_regime(prices)  # Using convenience function
    
    # Example trading logic based on regime
    print("\nTrading Logic Example:")
    print("─" * 40)
    
    # Position sizing based on regime
    base_position = 100000  # Base position size
    adjusted_position = base_position * signal.recommended_exposure
    print(f"Base Position: ₹{base_position:,.0f}")
    print(f"Adjusted Position: ₹{adjusted_position:,.0f}")
    
    # Strategy selection based on regime
    if signal.is_bullish():
        strategy = "Trend Following / Momentum"
        action = "LONG"
    elif signal.is_bearish():
        strategy = "Hedging / Cash"
        action = "DEFENSIVE"
    else:
        strategy = "Range Trading / Neutral"
        action = "NEUTRAL"
    
    print(f"\nRecommended Strategy: {strategy}")
    print(f"Action: {action}")
    
    # Risk management
    if signal.is_defensive():
        print("\n⚠️  DEFENSIVE POSITIONING ADVISED")
        print("   - Reduce equity exposure")
        print("   - Consider hedges (puts/shorts)")
        print("   - Raise cash levels")
    
    # Alert conditions
    print("\nAlert Conditions:")
    if signal.volatility_regime.value == "EXTREME":
        print("  🚨 EXTREME VOLATILITY - Reduce position sizes")
    if signal.risk_score > 70:
        print("  🚨 HIGH RISK SCORE - Maximum caution advised")
    if signal.transition_probability > 0.3:
        print("  ⚡ HIGH TRANSITION PROBABILITY - Regime change possible")
    
    return signal


def example_convert_to_dict():
    """Example converting signal to dictionary"""
    print("\n" + "=" * 60)
    print("SIGNAL SERIALIZATION EXAMPLE")
    print("=" * 60)
    
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="1y")
    signal = detect_regime(df['Close'])
    
    # Convert to dict (useful for JSON serialization, database storage, etc.)
    signal_dict = signal.to_dict()
    
    print("\nSignal as Dictionary:")
    for key, value in signal_dict.items():
        if key != 'regime_probabilities':
            print(f"  {key}: {value}")
    
    print("\nRegime Probabilities:")
    for regime, prob in sorted(signal_dict['regime_probabilities'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {regime}: {prob * 100:.1f}%")
    
    return signal_dict


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  AVASTHA - Market Regime Detection Engine Examples")
    print("█" * 60 + "\n")
    
    # Run examples
    example_basic_usage()
    example_custom_config()
    example_regime_history()
    example_multiple_symbols()
    example_programmatic_integration()
    example_convert_to_dict()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
