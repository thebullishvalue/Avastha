"""
AVASTHA - Market Data Engine

Handles data fetching, indicator calculation, and historical snapshot generation
with support for multiple universes (ETF, F&O, Index Constituents).

Version: 1.0.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
import logging
import warnings
import requests
import io

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ETF Universe - Curated sectoral and thematic ETFs
ETF_UNIVERSE = [
    "NIFTYIETF.NS", "MON100.NS", "MAKEINDIA.NS", "SILVERIETF.NS",
    "HEALTHIETF.NS", "CONSUMIETF.NS", "GOLDIETF.NS", "INFRAIETF.NS",
    "CPSEETF.NS", "TNIDETF.NS", "COMMOIETF.NS", "MODEFENCE.NS",
    "MOREALTY.NS", "PSUBNKIETF.NS", "MASPTOP50.NS", "FMCGIETF.NS",
    "ITIETF.NS", "EVINDIA.NS", "MNC.NS", "FINIETF.NS",
    "AUTOIETF.NS", "PVTBANIETF.NS", "MONIFTY500.NS", "ECAPINSURE.NS",
    "MIDCAPIETF.NS", "MOSMALL250.NS", "OILIETF.NS", "METALIETF.NS",
    "CHEMICAL.NS", "GROWWPOWER.NS"
]

ETF_NAMES = {
    "NIFTYIETF.NS": "NIFTY 50", "MON100.NS": "NIFTY 100",
    "MAKEINDIA.NS": "Make India", "SILVERIETF.NS": "Silver",
    "HEALTHIETF.NS": "Healthcare", "CONSUMIETF.NS": "Consumer",
    "GOLDIETF.NS": "Gold", "INFRAIETF.NS": "Infra",
    "CPSEETF.NS": "CPSE", "TNIDETF.NS": "TN Index",
    "COMMOIETF.NS": "Commodities", "MODEFENCE.NS": "Defence",
    "MOREALTY.NS": "Realty", "PSUBNKIETF.NS": "PSU Bank",
    "MASPTOP50.NS": "Top 50", "FMCGIETF.NS": "FMCG",
    "ITIETF.NS": "IT", "EVINDIA.NS": "EV India",
    "MNC.NS": "MNC", "FINIETF.NS": "Financial",
    "AUTOIETF.NS": "Auto", "PVTBANIETF.NS": "Pvt Bank",
    "MONIFTY500.NS": "NIFTY 500", "ECAPINSURE.NS": "Insurance",
    "MIDCAPIETF.NS": "Midcap", "MOSMALL250.NS": "Smallcap",
    "OILIETF.NS": "Oil & Gas", "METALIETF.NS": "Metal",
    "CHEMICAL.NS": "Chemical", "GROWWPOWER.NS": "Power"
}

# Index URL mapping for constituent fetching
INDEX_LIST = [
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
    "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100", "NIFTY BANK",
    "NIFTY AUTO", "NIFTY FIN SERVICE", "NIFTY FMCG", "NIFTY IT",
    "NIFTY MEDIA", "NIFTY METAL", "NIFTY PHARMA"
]

BASE_URL = "https://www.niftyindices.com/IndexConstituent/"
INDEX_URL_MAP = {
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

UNIVERSE_OPTIONS = ["ETF Universe", "F&O Stocks", "Index Constituents"]


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE FETCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_fno_stock_list() -> Tuple[Optional[List[str]], str]:
    """Fetch F&O stock list from NSE"""
    try:
        from nsepython import nse_get_advances_declines
        stock_data = nse_get_advances_declines()
        if not isinstance(stock_data, pd.DataFrame):
            return None, f"API returned unexpected type: {type(stock_data)}"
        
        if 'symbol' in stock_data.columns:
            symbols = stock_data['symbol'].tolist()
        elif 'Symbol' in stock_data.columns:
            symbols = stock_data['Symbol'].tolist()
        else:
            return None, "No symbol column found in F&O data"
        
        symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
        return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O stocks"
    except ImportError:
        return None, "nsepython not installed. Run: pip install nsepython"
    except Exception as e:
        return None, f"Error fetching F&O list: {e}"


def get_index_constituents(index_name: str) -> Tuple[Optional[List[str]], str]:
    """Fetch index constituents from NSE"""
    if index_name not in INDEX_URL_MAP:
        return None, f"Unknown index: {index_name}"
    
    url = INDEX_URL_MAP[index_name]
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/csv,application/csv,text/plain,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        response.raise_for_status()
        
        csv_file = io.StringIO(response.text)
        stock_df = pd.read_csv(csv_file)
        
        if 'Symbol' in stock_df.columns:
            symbols = stock_df['Symbol'].tolist()
            symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
            return symbols_ns, f"✓ Fetched {len(symbols_ns)} {index_name} constituents"
        else:
            return None, f"No Symbol column found in {index_name} data"
            
    except Exception as e:
        return None, f"Error fetching {index_name}: {e}"


def get_universe_symbols(universe_type: str, index_name: Optional[str] = None) -> Tuple[List[str], str]:
    """
    Get symbols for the selected universe type.
    
    Args:
        universe_type: One of "ETF Universe", "F&O Stocks", "Index Constituents"
        index_name: Required if universe_type is "Index Constituents"
        
    Returns:
        Tuple of (symbol_list, status_message)
    """
    if universe_type == "ETF Universe":
        return ETF_UNIVERSE, f"✓ Using {len(ETF_UNIVERSE)} curated ETFs"
    
    elif universe_type == "F&O Stocks":
        symbols, msg = get_fno_stock_list()
        if symbols:
            return symbols, msg
        else:
            logging.warning(f"F&O fetch failed: {msg}. Falling back to ETF Universe.")
            return ETF_UNIVERSE, f"⚠️ {msg}. Using ETF Universe instead."
    
    elif universe_type == "Index Constituents":
        if not index_name:
            return ETF_UNIVERSE, "⚠️ No index selected. Using ETF Universe."
        symbols, msg = get_index_constituents(index_name)
        if symbols:
            return symbols, msg
        else:
            logging.warning(f"Index fetch failed: {msg}. Falling back to ETF Universe.")
            return ETF_UNIVERSE, f"⚠️ {msg}. Using ETF Universe instead."
    
    return ETF_UNIVERSE, "⚠️ Unknown universe type. Using ETF Universe."


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

class LiquidityOscillator:
    """
    Calculates the Liquidity Oscillator indicator.
    """
    
    def __init__(self, length: int = 20, impact_window: int = 3):
        if length <= 0 or impact_window <= 0:
            raise ValueError("Length and impact_window must be positive integers.")
        self.length = length
        self.impact_window = impact_window

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(data.columns):
            return pd.Series(dtype=float)

        df = data.copy()
        df['spread'] = (df['high'] + df['low']) / 2 - df['open']
        df['vol_ma'] = df['volume'].rolling(window=self.length).mean()
        safe_vol_ma = df['vol_ma'].replace(0, pd.NA)
        
        df['vwap_spread'] = (
            df['spread'] * df['volume'] / safe_vol_ma
        ).rolling(window=self.length).mean()
        
        close_shifted = df['close'].shift(self.impact_window)
        df['price_impact'] = (
            (df['close'] - close_shifted) * df['volume'] / safe_vol_ma
        ).rolling(window=self.length).mean()
        
        df['liquidity_score'] = df['vwap_spread'] - df['price_impact']
        df['source_value'] = df['close'] + df['liquidity_score']
        df['lowest_value'] = df['source_value'].rolling(window=self.length).min()
        df['highest_value'] = df['source_value'].rolling(window=self.length).max()
        
        range_value = df['highest_value'] - df['lowest_value']
        safe_range_value = range_value.replace(0, pd.NA)
        
        oscillator = 200 * (df['source_value'] - df['lowest_value']) / safe_range_value - 100
        return oscillator.rename('liquidity_oscillator')


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV data to weekly timeframe"""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    
    logic = {
        'open': 'first', 
        'high': 'max', 
        'low': 'min', 
        'close': 'last', 
        'volume': 'sum'
    }
    return df.resample('W-FRI').apply(logic).dropna()


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    if data.empty or 'close' not in data.columns or len(data) < period:
        return pd.Series(index=data.index, dtype=float)
    
    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    
    return rsi


# ══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class MarketDataEngine:
    """
    Engine for fetching and processing market data.
    """
    
    INDICATOR_PERIODS = [20, 90, 200]
    MAX_INDICATOR_PERIOD = max(INDICATOR_PERIODS)
    
    COLUMN_ORDER = [
        'date', 'symbol', 'price', 'rsi latest', 'rsi weekly',
        '% change', 'osc latest', 'osc weekly',
        '9ema osc latest', '9ema osc weekly',
        '21ema osc latest', '21ema osc weekly',
        'zscore latest', 'zscore weekly',
        'ma20 latest', 'ma90 latest', 'ma200 latest',
        'ma20 weekly', 'ma90 weekly', 'ma200 weekly',
        'dev20 latest', 'dev20 weekly'
    ]
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """Initialize the data engine with symbols"""
        self.symbols = symbols or ETF_UNIVERSE
        self.oscillator = LiquidityOscillator(length=20, impact_window=3)
    
    def set_universe(self, universe_type: str, index_name: Optional[str] = None) -> str:
        """
        Set the analysis universe.
        
        Returns:
            Status message
        """
        self.symbols, msg = get_universe_symbols(universe_type, index_name)
        return msg
    
    def _calculate_all_indicators(self, symbol_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate all technical indicators for a symbol's entire history"""
        daily_data = symbol_data.copy()
        if daily_data.empty:
            return None
        
        weekly_data = resample_to_weekly(daily_data)
        
        all_results_df = pd.DataFrame(index=daily_data.index)
        all_results_df['price'] = daily_data['close']
        all_results_df['% change'] = daily_data['close'].pct_change()
        
        timeframes = {'latest': daily_data, 'weekly': weekly_data}
        
        for tf_name, df in timeframes.items():
            if len(df) < 2:
                continue
            
            # Calculate oscillator
            osc = self.oscillator.calculate(df)
            if not osc.dropna().empty:
                all_results_df[f'osc {tf_name}'] = osc
                all_results_df[f'9ema osc {tf_name}'] = osc.ewm(span=9).mean()
                all_results_df[f'21ema osc {tf_name}'] = osc.ewm(span=21).mean()
                
                if len(osc.dropna()) >= 20:
                    osc_sma20 = osc.rolling(window=20).mean()
                    osc_std20 = osc.rolling(window=20).std()
                    safe_std20 = osc_std20.replace(0, pd.NA)
                    all_results_df[f'zscore {tf_name}'] = (osc - osc_sma20) / safe_std20
            
            # Calculate RSI
            rsi_series = calculate_rsi(df)
            if rsi_series is not None and not rsi_series.dropna().empty:
                all_results_df[f'rsi {tf_name}'] = rsi_series
            
            # Calculate moving averages
            for period in self.INDICATOR_PERIODS:
                if len(df) >= period:
                    all_results_df[f'ma{period} {tf_name}'] = df['close'].rolling(window=period).mean()
                    if period == 20:
                        all_results_df[f'dev{period} {tf_name}'] = df['close'].rolling(window=period).std()
        
        # Reindex and forward-fill weekly data
        all_results_df = all_results_df.reindex(daily_data.index)
        weekly_cols = [col for col in all_results_df.columns if 'weekly' in col]
        all_results_df[weekly_cols] = all_results_df[weekly_cols].ffill()
        
        return all_results_df
    
    def generate_historical_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        progress_callback=None
    ) -> List[Tuple[datetime, pd.DataFrame]]:
        """
        Generate historical indicator snapshots.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            progress_callback: Optional callback function(progress, message)
            
        Returns:
            List of (date, DataFrame) tuples with daily snapshots
        """
        symbols_to_process = self.symbols
        
        if not symbols_to_process:
            logging.error("No symbols provided for data generation")
            return []
        
        if progress_callback:
            progress_callback(0.05, f"Downloading data for {len(symbols_to_process)} symbols...")
        
        logging.info(f"Downloading data for {len(symbols_to_process)} symbols...")
        
        try:
            all_data = yf.download(
                symbols_to_process,
                start=start_date,
                end=end_date + timedelta(days=1),
                progress=False
            )
        except Exception as e:
            logging.error(f"yfinance download failed: {e}")
            return []
        
        if all_data.empty or all_data['Close'].dropna(how='all').empty:
            logging.error("Downloaded data is empty")
            return []
        
        # Clean up failed tickers
        if len(symbols_to_process) > 1:
            valid_tickers = all_data['Close'].dropna(how='all', axis=1).columns
            invalid_tickers = [s for s in symbols_to_process if s not in valid_tickers]
            
            if invalid_tickers:
                logging.warning(f"Failed tickers: {len(invalid_tickers)}")
                all_data = all_data.loc[:, (slice(None), valid_tickers)]
                symbols_to_process = list(valid_tickers)
                
                if not symbols_to_process:
                    logging.error("No valid tickers remaining")
                    return []
        
        if progress_callback:
            progress_callback(0.30, f"Processing {len(symbols_to_process)} symbols...")
        
        logging.info(f"Download successful. Processing {len(symbols_to_process)} symbols...")
        all_data.columns.names = ['Indicator', 'Symbol']
        
        # Pre-calculate indicators for all symbols
        ticker_cache = {}
        total_tickers = len(symbols_to_process)
        
        for idx, ticker in enumerate(symbols_to_process):
            try:
                if len(symbols_to_process) > 1:
                    symbol_df = all_data.xs(ticker, level='Symbol', axis=1).copy()
                else:
                    symbol_df = all_data.copy()
                
                symbol_df.columns = [col.lower() for col in symbol_df.columns]
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in symbol_df.columns:
                        symbol_df[col] = pd.to_numeric(symbol_df[col], errors='coerce')
                
                symbol_df = symbol_df.dropna(subset=['close', 'volume'])
                
                if not symbol_df.empty:
                    indicators_df = self._calculate_all_indicators(symbol_df)
                    ticker_cache[ticker] = indicators_df
                
                if progress_callback and idx % 10 == 0:
                    progress_callback(0.30 + 0.50 * (idx / total_tickers), f"Processing {ticker}...")
                    
            except Exception as e:
                logging.warning(f"Skipping {ticker}: {e}")
                continue
        
        if progress_callback:
            progress_callback(0.85, "Generating daily snapshots...")
        
        # Generate daily snapshots
        results = []
        date_range = all_data.index.normalize().unique()
        
        for snapshot_date in date_range:
            if snapshot_date < (start_date + timedelta(days=self.MAX_INDICATOR_PERIOD)):
                continue
            if snapshot_date > end_date:
                continue
            
            daily_results = []
            for ticker in symbols_to_process:
                if ticker not in ticker_cache:
                    continue
                
                full_df = ticker_cache[ticker]
                if full_df is None or snapshot_date not in full_df.index:
                    continue
                
                try:
                    row = full_df.loc[snapshot_date]
                    if row.isnull().all() or pd.isna(row.get('price')):
                        continue
                    
                    indicators = row.to_dict()
                    indicators['symbol'] = ticker.replace('.NS', '')
                    indicators['date'] = snapshot_date.strftime('%dth %b')
                    indicators['% change'] = indicators['% change'] * 100
                    
                    daily_results.append(indicators)
                except KeyError:
                    continue
            
            if daily_results:
                final_df = pd.DataFrame(daily_results)
                for col in self.COLUMN_ORDER:
                    if col not in final_df.columns:
                        final_df[col] = pd.NA
                
                final_df = final_df[self.COLUMN_ORDER]
                results.append((snapshot_date, final_df))
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        logging.info(f"Generated {len(results)} daily snapshots")
        return results
    
    def get_regime_data(
        self, 
        analysis_date: datetime,
        lookback_days: int = 30,
        progress_callback=None
    ) -> List[Tuple[datetime, pd.DataFrame]]:
        """
        Get data specifically for regime analysis.
        
        Args:
            analysis_date: Date to analyze
            lookback_days: Number of days to look back
            progress_callback: Optional progress callback
            
        Returns:
            Historical data suitable for regime detection
        """
        total_days = int((lookback_days + self.MAX_INDICATOR_PERIOD) * 1.5) + 30
        fetch_start = analysis_date - timedelta(days=total_days)
        
        return self.generate_historical_data(fetch_start, analysis_date, progress_callback)
    
    def get_time_series_regime_data(
        self,
        start_date: datetime,
        end_date: datetime,
        progress_callback=None
    ) -> List[Tuple[datetime, pd.DataFrame]]:
        """
        Get data for time series regime analysis.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            progress_callback: Optional progress callback
            
        Returns:
            Historical data for the full time period
        """
        # Need extra data for indicator warmup
        total_days = int(self.MAX_INDICATOR_PERIOD * 1.5) + 30
        fetch_start = start_date - timedelta(days=total_days)
        
        return self.generate_historical_data(fetch_start, end_date, progress_callback)


def get_display_name(symbol: str) -> str:
    """Get display name for a symbol"""
    return ETF_NAMES.get(symbol, symbol.replace(".NS", ""))
