import os
import time
import pandas as pd
import numpy as np
import random
import logging
import yfinance as yf
import json
from datetime import datetime, timedelta, time as dt_time
import requests
import pytz
import schedule
import threading
from threading import Thread
from bs4 import BeautifulSoup
import re
import holidays

# Configure logging
os.makedirs("./data_cache", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./recommendations", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_CACHE_DIR = "./data_cache"
RECOMMENDATIONS_DIR = "./recommendations"
CACHE_EXPIRY_HOURS = 2
INTRADAY_CACHE_EXPIRY_MINUTES = 15  # Short expiry for intraday data
MAX_RETRY_ATTEMPTS = 3
MIN_REQUEST_DELAY = 2
MAX_REQUEST_DELAY = 5

# Get Telegram credentials from GitHub Secrets/Environment Variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Verify telegram credentials are available
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.error("Telegram credentials not found in environment variables!")
    logger.error("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as GitHub secrets or environment variables")
    # Keep running but won't send messages
    TELEGRAM_ENABLED = False
else:
    TELEGRAM_ENABLED = True

# Indian market holidays
INDIAN_HOLIDAYS = holidays.India()

# Improved stock symbols list with categories
INDICES = [
    "^NSEI",     # Nifty 50 Index
    "^NSEBANK"   # Bank Nifty Index
]

# Top traded stocks for regular monitoring
TOP_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "INFY.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "AXISBANK.NS", "HINDALCO.NS", "ITC.NS", "BHARTIARTL.NS",
    "ADANIENT.NS", "ONGC.NS", "NTPC.NS", "JINDALSTEL.NS",
    "BAJFINANCE.NS", "KOTAKBANK.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "LT.NS", "WIPRO.NS", "COALINDIA.NS", "JSWSTEEL.NS",
    "TATAPOWER.NS", "HCLTECH.NS", "BPCL.NS", "HINDUNILVR.NS",
    "M&M.NS", "ADANIPORTS.NS", "POWERGRID.NS", "VEDL.NS",
    "BANKBARODA.NS", "SAIL.NS", "ASIANPAINT.NS", "APOLLOHOSP.NS",
    "PFC.NS", "RECLTD.NS", "GAIL.NS", "PNB.NS",
    "ASHOKLEY.NS", "IOC.NS", "AMBUJACEM.NS", "DRREDDY.NS",
    "BAJAJ-AUTO.NS", "INDUSINDBK.NS", "HEROMOTOCO.NS", "HINDPETRO.NS",
    "IDEA.NS", "BIOCON.NS", "FEDERALBNK.NS", "CANBK.NS",
    "TECHM.NS", "CIPLA.NS", "IDFC.NS", "NMDC.NS",
    "BHEL.NS", "ADANIPOWER.NS", "IDBI.NS", "DLF.NS",
    "NATIONALUM.NS", "RBLBANK.NS", "TITAN.NS",
    "UNIONBANK.NS", "MOTHERSON.NS", "DIVISLAB.NS", "INDHOTEL.NS",
    "IRCTC.NS", "ZOMATO.NS", "TATACONSUM.NS",
    "PIIND.NS", "RVNL.NS", "CUMMINSIND.NS", "SUZLON.NS",
    "GRASIM.NS", "BEL.NS", "LUPIN.NS", "PAYTM.NS",
    "IRFC.NS", "BRITANNIA.NS", "HAVELLS.NS", "GODREJCP.NS",
    "UPL.NS", "CONCOR.NS", "HAL.NS", "NESTLEIND.NS",
    "INDIGO.NS", "SUNTV.NS", "DABUR.NS", "MAHINDCIE.NS",
    "SIEMENS.NS", "AUROPHARMA.NS", "JUBLFOOD.NS",
    "BHARATFORG.NS", "ABFRL.NS", "BANDHANBNK.NS", "BOSCHLTD.NS",
    "CHOLAFIN.NS", "EICHERMOT.NS", "EXIDEIND.NS", "HDFC.NS"
]

# Options for high volatility intraday trading
INTRADAY_STOCKS = INDICES + TOP_STOCKS[:12]  # Only use indices and top 12 stocks for intraday

# All stocks for daily analysis
ALL_STOCKS = INDICES + TOP_STOCKS + [
    "HCLTECH.NS", "BPCL.NS", "HINDUNILVR.NS", "M&M.NS", "ADANIPORTS.NS",
    "POWERGRID.NS", "VEDL.NS", "BANKBARODA.NS", "SAIL.NS", "ASIANPAINT.NS",
    "APOLLOHOSP.NS", "PFC.NS", "RECLTD.NS", "GAIL.NS", "PNB.NS",
    "ASHOKLEY.NS", "IOC.NS", "AMBUJACEM.NS", "DRREDDY.NS", "BAJAJ-AUTO.NS",
    "INDUSINDBK.NS", "HEROMOTOCO.NS", "HINDPETRO.NS", "IDEA.NS", "BIOCON.NS",
    "FEDERALBNK.NS", "CANBK.NS", "TECHM.NS", "CIPLA.NS", "IDFC.NS",
    "NMDC.NS", "BHEL.NS", "ADANIPOWER.NS", "IDBI.NS", "DLF.NS"
]

# News sources
NEWS_SOURCES = [
    {"name": "MoneyControl", "url": "https://www.moneycontrol.com/news/business/markets/"},
    {"name": "Economic Times", "url": "https://economictimes.indiatimes.com/markets/stocks/news"},
    {"name": "LiveMint", "url": "https://www.livemint.com/market/stock-market-news"}
]

# Check cache validity
def is_cache_valid(symbol: str, interval: str) -> bool:
    cache_file = os.path.join(DATA_CACHE_DIR, f"{symbol}_{interval}.csv")
    if not os.path.exists(cache_file):
        return False
    
    mod_time = os.path.getmtime(cache_file)
    mod_datetime = datetime.fromtimestamp(mod_time)
    now = datetime.now()
    
    # Different expiry times for different intervals
    if interval in ['5m', '15m', '30m', '60m', '1h']:
        expiry_seconds = INTRADAY_CACHE_EXPIRY_MINUTES * 60
    else:
        expiry_seconds = CACHE_EXPIRY_HOURS * 3600
        
    if (now - mod_datetime).total_seconds() > expiry_seconds:
        return False
    
    try:
        df = pd.read_csv(
            cache_file,
            index_col=0,
            parse_dates=True,
            date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        )
        
        # Check if required columns exist and have valid data
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                return False
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().all():
                return False
        
        return not df.empty and len(df) >= 5
    except Exception as e:
        logger.warning(f"Corrupted cache for {symbol}: {e}")
        return False

# Fetch stock data with retries
def fetch_stock_data(symbol: str, interval: str = '1d', period: str = '90d') -> pd.DataFrame:
    cache_file = os.path.join(DATA_CACHE_DIR, f"{symbol}_{interval}.csv")
    
    # Check if cached data is valid
    if is_cache_valid(symbol, interval):
        try:
            df = pd.read_csv(
                cache_file,
                index_col=0,
                parse_dates=True,
                date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            )
            
            # Ensure numeric columns are properly converted
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Using cached data for {symbol} ({interval})")
            return df
        except Exception as e:
            logger.warning(f"Error reading cache for {symbol}: {e}")
    
    # Fetch new data with exponential backoff
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            # Add random delay to avoid rate limiting
            delay = random.uniform(MIN_REQUEST_DELAY, MAX_REQUEST_DELAY) * (1.5 ** attempt)
            logger.info(f"Fetching data for {symbol} ({interval}, attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS}), waiting {delay:.2f}s")
            time.sleep(delay)
            
            # Use a more explicit approach to download
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period)
            
            # Verify data validity
            if df is None or df.empty or len(df) <= 5:
                logger.warning(f"Empty or insufficient data returned for {symbol}")
                continue
                
            # Ensure all required columns are present and valid
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for {symbol}")
                continue
                
            # Check for NaN values in critical columns
            if df[required_cols].isna().all().any():
                logger.warning(f"NaN values in critical columns for {symbol}")
                continue
                
            # Data is valid, save to cache
            df.to_csv(cache_file)
            logger.info(f"Data fetched and cached for {symbol} ({interval}) - {len(df)} records")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol} on attempt {attempt + 1}: {e}")
    
    logger.error(f"Failed to fetch data for {symbol} after {MAX_RETRY_ATTEMPTS} attempts")
    return None

# Send message to Telegram with extended retries
def send_telegram_message(message: str, max_retries=3):
    if not TELEGRAM_ENABLED:
        logger.warning("Telegram messaging disabled. Message not sent: " + message[:50] + "...")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending Telegram message (attempt {attempt+1}/{max_retries}): {message[:50]}...")
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info("Message sent to Telegram successfully.")
                return True
            else:
                logger.error(f"Failed to send Telegram message: {response.status_code} - {response.text}")
                # If message is too long, truncate it
                if response.status_code == 400 and "message is too long" in response.text.lower():
                    if len(message) > 1000:
                        message = message[:1000] + "...\n\n(Message truncated due to length limits)"
                        payload["text"] = message
                        logger.info("Truncated message for retry")
                    else:
                        # Try without markdown formatting
                        payload["parse_mode"] = ""
                        logger.info("Removing markdown formatting for retry")
            
            time.sleep(2 * (attempt + 1))  # Exponential backoff
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            time.sleep(2 * (attempt + 1))
    
    logger.error("All attempts to send Telegram message failed")
    return False

# Calculate technical indicators - IMPROVED with more reliable indicators
def calculate_indicators(df):
    if df is None or len(df) < 14:
        return None
    
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Calculate 14-day RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Moving Averages
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()
    
    # Calculate MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal']
    
    # Calculate Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    data['BB_Percent'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Calculate Average Directional Index (ADX)
    high_diff = data['High'].diff()
    low_diff = -data['Low'].diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = data['High'] - data['Low']
    tr2 = abs(data['High'] - data['Close'].shift(1))
    tr3 = abs(data['Low'] - data['Close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    atr = tr.rolling(14).mean()
    data['ATR'] = atr
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    data['ADX'] = dx.rolling(14).mean()
    data['Plus_DI'] = plus_di
    data['Minus_DI'] = minus_di
    
    # Calculate stochastic oscillator
    data['14-high'] = data['High'].rolling(14).max()
    data['14-low'] = data['Low'].rolling(14).min()
    data['%K'] = (data['Close'] - data['14-low']) * 100 / (data['14-high'] - data['14-low'])
    data['%D'] = data['%K'].rolling(3).mean()
    
    # Volume indicators
    data['Volume_SMA20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA20']
    
    # Price momentum
    data['Price_Change_1D'] = data['Close'].pct_change(1) * 100  # 1-day percentage change
    data['Price_Change_5D'] = data['Close'].pct_change(5) * 100  # 5-day percentage change
    data['Price_Change_20D'] = data['Close'].pct_change(20) * 100  # 20-day percentage change
    
    # NEW: Calculate OBV (On-Balance Volume)
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()
    
    # NEW: Ichimoku Cloud
    high_9 = data['High'].rolling(window=9).max()
    low_9 = data['Low'].rolling(window=9).min()
    data['Tenkan_Sen'] = (high_9 + low_9) / 2  # Conversion Line
    
    high_26 = data['High'].rolling(window=26).max()
    low_26 = data['Low'].rolling(window=26).min()
    data['Kijun_Sen'] = (high_26 + low_26) / 2  # Base Line
    
    data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(26)  # Leading Span A
    data['Senkou_Span_B'] = ((data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(26)  # Leading Span B
    
    # NEW: Calculate rate of change
    data['ROC'] = data['Close'].pct_change(10) * 100
    
    # Intraday specific indicators
    if len(data) >= 7:  # Need at least 7 periods for these indicators
        # Supertrend calculation (simplified)
        atr_period = 7
        multiplier = 3.0
        
        # ATR already calculated above
        data['Upper_Band'] = ((data['High'] + data['Low']) / 2) + (multiplier * data['ATR'])
        data['Lower_Band'] = ((data['High'] + data['Low']) / 2) - (multiplier * data['ATR'])
        
        # Initialize Supertrend arrays
        st_values = np.full(len(data), np.nan)
        st_direction = np.full(len(data), np.nan)
        
        # Calculate Supertrend - using arrays instead of direct DataFrame assignment
        for i in range(atr_period, len(data)):
            if data['Close'].iloc[i] > data['Upper_Band'].iloc[i-1]:
                st_values[i] = data['Lower_Band'].iloc[i]
                st_direction[i] = 1  # Uptrend
            elif data['Close'].iloc[i] < data['Lower_Band'].iloc[i-1]:
                st_values[i] = data['Upper_Band'].iloc[i]
                st_direction[i] = -1  # Downtrend
            else:
                st_values[i] = st_values[i-1]
                st_direction[i] = st_direction[i-1]
        
        # Assign the arrays to DataFrame columns
        data['Supertrend'] = st_values
        data['ST_Direction'] = st_direction
        
        # VWAP (Volume Weighted Average Price) - for intraday
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    return data

# Analyze stock and generate score - IMPROVED with better signal weighting
def analyze_stock(symbol: str, data: pd.DataFrame, intraday=False):
    try:
        min_data_points = 14 if intraday else 30
        if data is None or len(data) < min_data_points:
            logger.warning(f"Not enough data to analyze {symbol}")
            return None
        
        # Calculate indicators
        data_with_indicators = calculate_indicators(data)
        if data_with_indicators is None:
            return None
        
        # Get latest values
        latest = data_with_indicators.iloc[-1]
        prev = data_with_indicators.iloc[-2]
        latest_close = latest['Close']
        
        # Initialize signals and score
        buy_signals = []
        sell_signals = []
        score = 0  # Positive = bullish, Negative = bearish
        
        # RSI signals (IMPROVED weighting)
        if not np.isnan(latest['RSI']):
            if latest['RSI'] < 30:
                buy_signals.append(f"RSI oversold ({latest['RSI']:.1f})")
                score += 3  # Increased weight
            elif latest['RSI'] < 40:
                buy_signals.append(f"RSI approaching oversold ({latest['RSI']:.1f})")
                score += 1
            elif latest['RSI'] > 70:
                sell_signals.append(f"RSI overbought ({latest['RSI']:.1f})")
                score -= 3  # Increased weight
            elif latest['RSI'] > 60:
                sell_signals.append(f"RSI approaching overbought ({latest['RSI']:.1f})")
                score -= 1
                
        # Moving Average signals
        if not np.isnan(latest['SMA20']) and not np.isnan(latest['SMA50']):
            # Price above multiple moving averages (stronger signal)
            if latest_close > latest['SMA20'] and latest_close > latest['SMA50'] and latest_close > latest['EMA21']:
                buy_signals.append(f"Price above multiple MAs (strong uptrend)")
                score += 2
            elif latest_close > latest['SMA20'] and latest_close > latest['SMA50']:
                buy_signals.append(f"Price above both SMA20 and SMA50")
                score += 1
                
            # Price below multiple moving averages (stronger signal)
            if latest_close < latest['SMA20'] and latest_close < latest['SMA50'] and latest_close < latest['EMA21']:
                sell_signals.append(f"Price below multiple MAs (strong downtrend)")
                score -= 2
            elif latest_close < latest['SMA20'] and latest_close < latest['SMA50']:
                sell_signals.append(f"Price below both SMA20 and SMA50")
                score -= 1
                
            # Golden Cross (short term MA crosses above long term MA)
            if prev['SMA20'] <= prev['SMA50'] and latest['SMA20'] > latest['SMA50']:
                buy_signals.append("Golden Cross detected (SMA20 crossed above SMA50)")
                score += 3
            
            # Death Cross (short term MA crosses below long term MA)
            elif prev['SMA20'] >= prev['SMA50'] and latest['SMA20'] < latest['SMA50']:
                sell_signals.append("Death Cross detected (SMA20 crossed below SMA50)")
                score -= 3
                
            # EMA crosses
            if prev['EMA9'] <= prev['EMA21'] and latest['EMA9'] > latest['EMA21']:
                buy_signals.append("EMA9 crossed above EMA21 (bullish)")
                score += 2  # More reliable than EMA9 crossing SMA20
            elif prev['EMA9'] >= prev['EMA21'] and latest['EMA9'] < latest['EMA21']:
                sell_signals.append("EMA9 crossed below EMA21 (bearish)")
                score -= 2
        
        # MACD signals
        if not np.isnan(latest['MACD']) and not np.isnan(latest['Signal']):
            # MACD crosses above signal line
            if prev['MACD'] <= prev['Signal'] and latest['MACD'] > latest['Signal']:
                buy_signals.append("MACD crossed above signal line (bullish)")
                score += 2
            
            # MACD crosses below signal line
            elif prev['MACD'] >= prev['Signal'] and latest['MACD'] < latest['Signal']:
                sell_signals.append("MACD crossed below signal line (bearish)")
                score -= 2
                
            # MACD histogram increasing or decreasing - more granular analysis
            if latest['MACD_Hist'] > 0 and prev['MACD_Hist'] > 0 and latest['MACD_Hist'] > prev['MACD_Hist']:
                buy_signals.append("MACD histogram expanding positively (strong bullish momentum)")
                score += 2
            elif latest['MACD_Hist'] < 0 and prev['MACD_Hist'] < 0 and latest['MACD_Hist'] < prev['MACD_Hist']:
                sell_signals.append("MACD histogram expanding negatively (strong bearish momentum)")
                score -= 2
            elif latest['MACD_Hist'] > 0 and prev['MACD_Hist'] < 0:
                buy_signals.append("MACD histogram turned positive")
                score += 1
            elif latest['MACD_Hist'] < 0 and prev['MACD_Hist'] > 0:
                sell_signals.append("MACD histogram turned negative")
                score -= 1
        
        # Bollinger Bands signals
        if not np.isnan(latest['BB_Upper']) and not np.isnan(latest['BB_Lower']):
            if latest_close < latest['BB_Lower']:
                buy_signals.append("Price below lower Bollinger Band (potential reversal)")
                score += 2
            elif latest_close > latest['BB_Upper']:
                sell_signals.append("Price above upper Bollinger Band (potential reversal)")
                score -= 2
                
            # BB percentage for more granular analysis
            if latest['BB_Percent'] < 0.2:
                buy_signals.append(f"Price near lower BB ({latest['BB_Percent']:.1%})")
                score += 1
            elif latest['BB_Percent'] > 0.8:
                sell_signals.append(f"Price near upper BB ({latest['BB_Percent']:.1%})")
                score -= 1
                
            # Bollinger Band width expanding (increasing volatility)
            if latest['BB_Width'] > prev['BB_Width'] * 1.1:
                if latest_close > latest['BB_Middle']:
                    buy_signals.append("Expanding volatility with upward momentum")
                    score += 1
                else:
                    sell_signals.append("Expanding volatility with downward momentum")
                    score -= 1
        
        # ADX signals (trend strength) - More emphasis on ADX for stronger trends
        if not np.isnan(latest['ADX']):
            if latest['ADX'] > 30:  # Strong trend
                if latest['Plus_DI'] > latest['Minus_DI']:
                    buy_signals.append(f"Strong uptrend (ADX: {latest['ADX']:.1f})")
                    score += 3  # Increased weight for strong trends
                else:
                    sell_signals.append(f"Strong downtrend (ADX: {latest['ADX']:.1f})")
                    score -= 3  # Increased weight for strong trends
            elif latest['ADX'] > 20:  # Moderate trend
                if latest['Plus_DI'] > latest['Minus_DI']:
                    buy_signals.append(f"Moderate uptrend (ADX: {latest['ADX']:.1f})")
                    score += 2
                else:
                    sell_signals.append(f"Moderate downtrend (ADX: {latest['ADX']:.1f})")
                    score -= 2
                    
            # DI Crossovers - strong signals
            if prev['Plus_DI'] <= prev['Minus_DI'] and latest['Plus_DI'] > latest['Minus_DI']:
                buy_signals.append("Bullish DI crossover")
                score += 2
            elif prev['Plus_DI'] >= prev['Minus_DI'] and latest['Plus_DI'] < latest['Minus_DI']:
                sell_signals.append("Bearish DI crossover")
                score -= 2
        
        # Stochastic signals
        if not np.isnan(latest['%K']) and not np.isnan(latest['%D']):
            if latest['%K'] < 20 and latest['%D'] < 20:
                buy_signals.append("Stochastic oversold")
                score += 1
            elif latest['%K'] > 80 and latest['%D'] > 80:
                sell_signals.append("Stochastic overbought")
                score -= 1
                
            # Stochastic crossover
            if prev['%K'] <= prev['%D'] and latest['%K'] > latest['%D']:
                buy_signals.append("Stochastic %K crossed above %D (bullish)")
                score += 1
            elif prev['%K'] >= prev['%D'] and latest['%K'] < latest['%D']:
                sell_signals.append("Stochastic %K crossed below %D (bearish)")
                score -= 1
        
        # Volume indicators - Volume confirmation is important!
        if 'Volume' in data_with_indicators.columns and not np.isnan(latest['Volume_Ratio']):
            if latest['Volume_Ratio'] > 2 and latest_close > prev['Close']:
                buy_signals.append(f"High volume upward move (Vol ratio: {latest['Volume_Ratio']:.1f})")
                score += 3  # Increased weight for volume confirmation
            elif latest['Volume_Ratio'] > 2 and latest_close < prev['Close']:
                sell_signals.append(f"High volume downward move (Vol ratio: {latest['Volume_Ratio']:.1f})")
                score -= 3  # Increased weight for volume confirmation
            
            # OBV signals
            if 'OBV' in data_with_indicators.columns and 'OBV_EMA' in data_with_indicators.columns:
                if latest['OBV'] > latest['OBV_EMA'] and prev['OBV'] <= prev['OBV_EMA']:
                    buy_signals.append("OBV crossed above its EMA (bullish volume)")
                    score += 2
                elif latest['OBV'] < latest['OBV_EMA'] and prev['OBV'] >= prev['OBV_EMA']:
                    sell_signals.append("OBV crossed below its EMA (bearish volume)")
                    score -= 2
        
        # Price momentum - Look for strong moves
        if not np.isnan(latest['Price_Change_5D']):
            if latest['Price_Change_5D'] > 10:
                buy_signals.append(f"Strong momentum: {latest['Price_Change_5D']:.1f}% in 5 days")
                score += 1
            elif latest['Price_Change_5D'] < -10:
                sell_signals.append(f"Weak momentum: {latest['Price_Change_5D']:.1f}% in 5 days")
                score -= 1
        
        # NEW: Ichimoku signals
        if 'Tenkan_Sen' in data_with_indicators.columns and not np.isnan(latest['Tenkan_Sen']):
            # Tenkan-Sen crosses above Kijun-Sen (bullish)
            if prev['Tenkan_Sen'] <= prev['Kijun_Sen'] and latest['Tenkan_Sen'] > latest['Kijun_Sen']:
                buy_signals.append("Tenkan-Sen crossed above Kijun-Sen (bullish)")
                score += 2
           # Tenkan-Sen crosses below Kijun-Sen (bearish)
            elif prev['Tenkan_Sen'] >= prev['Kijun_Sen'] and latest['Tenkan_Sen'] < latest['Kijun_Sen']:
                sell_signals.append("Tenkan-Sen crossed below Kijun-Sen (bearish)")
                score -= 2
                
            # Price above/below the cloud
            if (latest['Close'] > latest['Senkou_Span_A'] and latest['Close'] > latest['Senkou_Span_B']):
                buy_signals.append("Price above Ichimoku Cloud (bullish)")
                score += 2
            elif (latest['Close'] < latest['Senkou_Span_A'] and latest['Close'] < latest['Senkou_Span_B']):
                sell_signals.append("Price below Ichimoku Cloud (bearish)")
                score -= 2
        
        # NEW: ROC signals
        if 'ROC' in data_with_indicators.columns and not np.isnan(latest['ROC']):
            if latest['ROC'] > 5:
                buy_signals.append(f"Strong positive rate of change: {latest['ROC']:.1f}%")
                score += 1
            elif latest['ROC'] < -5:
                sell_signals.append(f"Strong negative rate of change: {latest['ROC']:.1f}%")
                score -= 1
        
        # Intraday specific signals
        if intraday and 'Supertrend' in data_with_indicators.columns:
            # Supertrend signals
            if latest['ST_Direction'] == 1 and prev['ST_Direction'] != 1:
                buy_signals.append("Supertrend turned bullish")
                score += 3  # Strong intraday signal
            elif latest['ST_Direction'] == -1 and prev['ST_Direction'] != -1:
                sell_signals.append("Supertrend turned bearish")
                score -= 3  # Strong intraday signal
                
            # VWAP signals for intraday
            if 'VWAP' in data_with_indicators.columns:
                if latest_close > latest['VWAP'] and prev['Close'] <= prev['VWAP']:
                    buy_signals.append("Price crossed above VWAP")
                    score += 2
                elif latest_close < latest['VWAP'] and prev['Close'] >= prev['VWAP']:
                    sell_signals.append("Price crossed below VWAP")
                    score -= 2
        
        # Final result
        result = {
            'symbol': symbol,
            'close': latest_close,
            'change': latest['Close'] / prev['Close'] - 1 if not np.isnan(prev['Close']) else 0,
            'score': score,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'timestamp': latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

# Determine market hours and state
def check_market_state():
    """Check if the market is open, pre-market, post-market, or closed."""
    
    # Get current time in Indian timezone
    india_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india_tz)

    # Check if today is a weekend
    if now.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        return "closed", "Weekend"
    
    # Check if today is a holiday
    today_date = now.date()
    if today_date in INDIAN_HOLIDAYS:
        return "closed", f"Holiday: {INDIAN_HOLIDAYS[today_date]}"
    
    # Define market hours in IST
    market_open = datetime.combine(today_date, dt_time(9, 15, 0)).replace(tzinfo=india_tz)
    market_close = datetime.combine(today_date, dt_time(15, 30, 0)).replace(tzinfo=india_tz)
    pre_market_start = datetime.combine(today_date, dt_time(8, 45, 0)).replace(tzinfo=india_tz)
    post_market_end = datetime.combine(today_date, dt_time(16, 0, 0)).replace(tzinfo=india_tz)
    
    if now < pre_market_start:
        minutes_to_pre_market = int((pre_market_start - now).total_seconds() / 60)
        return "closed", f"Pre-market starts in {minutes_to_pre_market} minutes"
    elif pre_market_start <= now < market_open:
        return "pre-market", f"Market opens in {int((market_open - now).total_seconds() / 60)} minutes"
    elif market_open <= now < market_close:
        minutes_elapsed = int((now - market_open).total_seconds() / 60)
        minutes_remaining = int((market_close - now).total_seconds() / 60)
        return "open", f"Market open for {minutes_elapsed} minutes, closes in {minutes_remaining} minutes"
    elif market_close <= now < post_market_end:
        return "post-market", f"Post-market, closes in {int((post_market_end - now).total_seconds() / 60)} minutes"
    else:
        tomorrow = today_date + timedelta(days=1)
        while tomorrow.weekday() >= 5 or tomorrow in INDIAN_HOLIDAYS:
            tomorrow += timedelta(days=1)
        
        next_market_open = datetime.combine(tomorrow, dt_time(9, 15, 0)).replace(tzinfo=india_tz)
        hours_to_open = int((next_market_open - now).total_seconds() / 3600)
        
        return "closed", f"Market closed, opens in {hours_to_open} hours"

# Get news from financial websites
def get_market_news(max_articles=5):
    """Fetch latest market news from financial websites."""
    all_news = []
    
    for source in NEWS_SOURCES:
        try:
            response = requests.get(source["url"], headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Different parsing logic for different sources
            if source["name"] == "MoneyControl":
                articles = soup.select('.article_box, .common_article')
                
                for article in articles[:max_articles]:
                    title_elem = article.select_one('h3 a, .article_title a')
                    if title_elem:
                        title = title_elem.text.strip()
                        link = title_elem.get('href', '')
                        if title and link:
                            all_news.append({
                                "source": source["name"],
                                "title": title,
                                "link": link if link.startswith('http') else f"https://www.moneycontrol.com{link}"
                            })
                
            elif source["name"] == "Economic Times":
                articles = soup.select('.eachStory, .story-card')
                
                for article in articles[:max_articles]:
                    title_elem = article.select_one('h3 a, .title a')
                    if title_elem:
                        title = title_elem.text.strip()
                        link = title_elem.get('href', '')
                        if title and link:
                            all_news.append({
                                "source": source["name"],
                                "title": title,
                                "link": link if link.startswith('http') else f"https://economictimes.indiatimes.com{link}"
                            })
                
            elif source["name"] == "LiveMint":
                articles = soup.select('.headline, .card')
                
                for article in articles[:max_articles]:
                    title_elem = article.select_one('h2 a, .headline a')
                    if title_elem:
                        title = title_elem.text.strip()
                        link = title_elem.get('href', '')
                        if title and link:
                            all_news.append({
                                "source": source["name"],
                                "title": title,
                                "link": link if link.startswith('http') else f"https://www.livemint.com{link}"
                            })
            
            logger.info(f"Fetched {len(articles[:max_articles])} news items from {source['name']}")
            
        except Exception as e:
            logger.error(f"Error fetching news from {source['name']}: {e}")
    
    return all_news[:max_articles]

# Generate technical signals for all stocks
def generate_technical_signals(stocks_list, interval='1d', intraday=False):
    """Generate technical signals for a list of stocks."""
    signals = []
    
    for symbol in stocks_list:
        try:
            # Use appropriate interval based on whether it's intraday or not
            if intraday:
                df = fetch_stock_data(symbol, interval=interval, period='5d')
            else:
                df = fetch_stock_data(symbol, interval='1d', period='90d')
                
            if df is not None and not df.empty:
                result = analyze_stock(symbol, df, intraday)
                if result:
                    signals.append(result)
                    logger.info(f"Generated signals for {symbol}: Score={result['score']}")
            else:
                logger.warning(f"No data available for {symbol}")
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
    
    # Sort signals by absolute score (strongest signals first)
    signals.sort(key=lambda x: abs(x['score']), reverse=True)
    return signals

# Generate daily insights report
def generate_daily_insights():
    """Generate a comprehensive daily market insights report."""
    try:
        # Check market state
        market_state, market_message = check_market_state()
        logger.info(f"Market state: {market_state} - {market_message}")
        
        # Get market news
        logger.info("Fetching market news...")
        news = get_market_news(max_articles=5)
        
        # Get index data
        logger.info("Analyzing market indices...")
        nifty_data = fetch_stock_data("^NSEI", interval='1d', period='90d')
        banknifty_data = fetch_stock_data("^NSEBANK", interval='1d', period='90d')
        
        # Analyze indices
        nifty_analysis = analyze_stock("^NSEI", nifty_data) if nifty_data is not None else None
        banknifty_analysis = analyze_stock("^NSEBANK", banknifty_data) if banknifty_data is not None else None
        
        # Generate signals for top stocks
        logger.info("Generating signals for top stocks...")
        top_signals = generate_technical_signals(TOP_STOCKS)
        
        # Prepare report
        report = []
        
        # Add report header
        today_date = datetime.now().strftime("%d %b %Y")
        report.append(f"📊 *Daily Market Insights - {today_date}*")
        report.append(f"📈 *Market Status:* {market_state.upper()} - {market_message}")
        report.append("")
        
        # Add index analysis
        report.append("🔍 *INDEX ANALYSIS*")
        
        if nifty_analysis:
            nifty_change_pct = nifty_analysis['change'] * 100
            direction = "🟢" if nifty_change_pct >= 0 else "🔴"
            report.append(f"*NIFTY 50:* {direction} {nifty_analysis['close']:.2f} ({nifty_change_pct:.2f}%)")
            
            if nifty_analysis['score'] > 3:
                report.append("  *Bullish signals:* " + ", ".join(nifty_analysis['buy_signals'][:3]))
            elif nifty_analysis['score'] < -3:
                report.append("  *Bearish signals:* " + ", ".join(nifty_analysis['sell_signals'][:3]))
            else:
                report.append("  *Neutral trend*")
        
        if banknifty_analysis:
            bn_change_pct = banknifty_analysis['change'] * 100
            direction = "🟢" if bn_change_pct >= 0 else "🔴"
            report.append(f"*BANK NIFTY:* {direction} {banknifty_analysis['close']:.2f} ({bn_change_pct:.2f}%)")
            
            if banknifty_analysis['score'] > 3:
                report.append("  *Bullish signals:* " + ", ".join(banknifty_analysis['buy_signals'][:3]))
            elif banknifty_analysis['score'] < -3:
                report.append("  *Bearish signals:* " + ", ".join(banknifty_analysis['sell_signals'][:3]))
            else:
                report.append("  *Neutral trend*")
        
        report.append("")
        
        # Add top bullish stocks
        bullish_stocks = [s for s in top_signals if s['score'] > 3]
        if bullish_stocks:
            report.append("🟢 *TOP BULLISH STOCKS*")
            for stock in bullish_stocks[:5]:
                symbol_display = stock['symbol'].replace('.NS', '')
                change_pct = stock['change'] * 100
                report.append(f"*{symbol_display}:* {stock['close']:.2f} ({change_pct:.2f}%) - Score: {stock['score']}")
                report.append("  *Signals:* " + ", ".join(stock['buy_signals'][:3]))
            report.append("")
        
        # Add top bearish stocks
        bearish_stocks = [s for s in top_signals if s['score'] < -3]
        if bearish_stocks:
            report.append("🔴 *TOP BEARISH STOCKS*")
            for stock in bearish_stocks[:5]:
                symbol_display = stock['symbol'].replace('.NS', '')
                change_pct = stock['change'] * 100
                report.append(f"*{symbol_display}:* {stock['close']:.2f} ({change_pct:.2f}%) - Score: {stock['score']}")
                report.append("  *Signals:* " + ", ".join(stock['sell_signals'][:3]))
            report.append("")
        
        # Add market news
        if news:
            report.append("📰 *LATEST MARKET NEWS*")
            for article in news:
                report.append(f"*{article['source']}:* [{article['title']}]({article['link']})")
            report.append("")
        
        # Add disclaimer
        report.append("⚠️ *Disclaimer:* This is an automated analysis and should not be considered as financial advice. Always do your own research before making investment decisions.")
        
        # Join report and save to file
        report_text = "\n".join(report)
        
        # Save report to file
        report_file = os.path.join(RECOMMENDATIONS_DIR, f"daily_report_{datetime.now().strftime('%Y%m%d')}.md")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Daily report generated and saved to {report_file}")
        
        # Send report via Telegram
        if TELEGRAM_ENABLED:
            send_telegram_message(report_text)
        
        return report_text
    
    except Exception as e:
        error_msg = f"Error generating daily insights: {e}"
        logger.error(error_msg)
        if TELEGRAM_ENABLED:
            send_telegram_message(f"⚠️ *Error:* {error_msg}")
        return None

# Intraday alert scanning function
def scan_intraday_signals():
    """Scan for intraday trading signals and send alerts."""
    try:
        # Check if market is open
        market_state, _ = check_market_state()
        if market_state != "open":
            logger.info(f"Market is not open, skipping intraday scan")
            return
        
        logger.info("Starting intraday signal scan...")
        
        # Generate intraday signals
        signals = generate_technical_signals(INTRADAY_STOCKS, interval='15m', intraday=True)
        
        # Filter strong signals (absolute score > 5)
        strong_signals = [s for s in signals if abs(s['score']) > 5]
        
        if not strong_signals:
            logger.info("No strong intraday signals detected")
            return
        
        # Prepare alert message
        alert_message = []
        alert_message.append("🚨 *INTRADAY TRADING ALERTS* 🚨")
        alert_message.append(f"Generated at {datetime.now().strftime('%H:%M:%S')}")
        alert_message.append("")
        
        # Add bullish signals
        bullish = [s for s in strong_signals if s['score'] > 5]
        if bullish:
            alert_message.append("🟢 *BULLISH SIGNALS*")
            for signal in bullish[:3]:
                symbol_display = signal['symbol'].replace('.NS', '')
                change_pct = signal['change'] * 100
                alert_message.append(f"*{symbol_display}:* {signal['close']:.2f} ({change_pct:.2f}%) - Score: {signal['score']}")
                alert_message.append("  *Key signals:* " + ", ".join(signal['buy_signals'][:3]))
            alert_message.append("")
        
        # Add bearish signals
        bearish = [s for s in strong_signals if s['score'] < -5]
        if bearish:
            alert_message.append("🔴 *BEARISH SIGNALS*")
            for signal in bearish[:3]:
                symbol_display = signal['symbol'].replace('.NS', '')
                change_pct = signal['change'] * 100
                alert_message.append(f"*{symbol_display}:* {signal['close']:.2f} ({change_pct:.2f}%) - Score: {signal['score']}")
                alert_message.append("  *Key signals:* " + ", ".join(signal['sell_signals'][:3]))
            alert_message.append("")
        
        # Add disclaimer
        alert_message.append("⚠️ *Disclaimer:* This is an automated alert and should not be considered as financial advice.")
        
        # Send alert via Telegram
        if TELEGRAM_ENABLED:
            send_telegram_message("\n".join(alert_message))
        
        logger.info(f"Sent intraday alerts for {len(strong_signals)} signals")
        
    except Exception as e:
        error_msg = f"Error scanning intraday signals: {e}"
        logger.error(error_msg)
        if TELEGRAM_ENABLED:
            send_telegram_message(f"⚠️ *Error:* {error_msg}")

# Weekly market outlook function
def generate_weekly_outlook():
    """Generate a weekly market outlook report."""
    try:
        logger.info("Generating weekly market outlook...")
        
        # Fetch extended data for indices
        nifty_data = fetch_stock_data("^NSEI", interval='1d', period='180d')
        banknifty_data = fetch_stock_data("^NSEBANK", interval='1d', period='180d')
        
        # Get sector analysis
        sector_indices = [
            "NIFTYIT.NS",    # IT Sector
            "NIFTYFMCG.NS",  # FMCG Sector
            "NIFTYAUTO.NS",  # Auto Sector
            "NIFTYMETAL.NS", # Metal Sector
            "NIFTYPHARM.NS", # Pharma Sector
            "NIFTYENERGY.NS" # Energy Sector
        ]
        
        sector_analysis = []
        for sector in sector_indices:
            try:
                data = fetch_stock_data(sector, interval='1d', period='90d')
                if data is not None:
                    analysis = analyze_stock(sector, data)
                    if analysis:
                        sector_analysis.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing sector {sector}: {e}")
        
        # Generate signals for all stocks for weekly perspective
        all_signals = generate_technical_signals(ALL_STOCKS)
        
        # Calculate market breadth
        bullish_count = len([s for s in all_signals if s['score'] > 3])
        bearish_count = len([s for s in all_signals if s['score'] < -3])
        neutral_count = len(all_signals) - bullish_count - bearish_count
        
        # Market breadth indicator (percentage of stocks in positive trend)
        if len(all_signals) > 0:
            market_breadth = bullish_count / len(all_signals) * 100
        else:
            market_breadth = 0
        
        # Prepare weekly outlook report
        report = []
        
        # Add report header
        week_start = datetime.now().strftime("%d %b %Y")
        week_end = (datetime.now() + timedelta(days=7)).strftime("%d %b %Y")
        report.append(f"📅 *WEEKLY MARKET OUTLOOK* 📅")
        report.append(f"*Period:* {week_start} to {week_end}")
        report.append("")
        
        # Add market breadth analysis
        report.append("🔍 *MARKET BREADTH*")
        
        if market_breadth > 60:
            outlook = "Bullish (Strong)"
        elif market_breadth > 50:
            outlook = "Bullish (Moderate)"
        elif market_breadth > 40:
            outlook = "Neutral with Bullish Bias"
        elif market_breadth > 30:
            outlook = "Neutral with Bearish Bias"
        elif market_breadth > 20:
            outlook = "Bearish (Moderate)"
        else:
            outlook = "Bearish (Strong)"
            
        report.append(f"*Weekly Outlook:* {outlook}")
        report.append(f"*Bullish Stocks:* {bullish_count} ({bullish_count/len(all_signals)*100:.1f}%)")
        report.append(f"*Neutral Stocks:* {neutral_count} ({neutral_count/len(all_signals)*100:.1f}%)")
        report.append(f"*Bearish Stocks:* {bearish_count} ({bearish_count/len(all_signals)*100:.1f}%)")
        report.append("")
        
        # Add index analysis
        report.append("📈 *MAJOR INDICES OUTLOOK*")
        
        if nifty_data is not None:
            nifty_analysis = analyze_stock("^NSEI", nifty_data)
            if nifty_analysis:
                nifty_change_pct = nifty_analysis['change'] * 100
                direction = "🟢" if nifty_change_pct >= 0 else "🔴"
                report.append(f"*NIFTY 50:* {direction} {nifty_analysis['close']:.2f} ({nifty_change_pct:.2f}%)")
                
                if nifty_analysis['score'] > 3:
                    report.append("  *Weekly Outlook:* Bullish")
                    report.append("  *Key Signals:* " + ", ".join(nifty_analysis['buy_signals'][:3]))
                elif nifty_analysis['score'] < -3:
                    report.append("  *Weekly Outlook:* Bearish")
                    report.append("  *Key Signals:* " + ", ".join(nifty_analysis['sell_signals'][:3]))
                else:
                    report.append("  *Weekly Outlook:* Neutral")
        
        if banknifty_data is not None:
            banknifty_analysis = analyze_stock("^NSEBANK", banknifty_data)
            if banknifty_analysis:
                bn_change_pct = banknifty_analysis['change'] * 100
                direction = "🟢" if bn_change_pct >= 0 else "🔴"
                report.append(f"*BANK NIFTY:* {direction} {banknifty_analysis['close']:.2f} ({bn_change_pct:.2f}%)")
                
                if banknifty_analysis['score'] > 3:
                    report.append("  *Weekly Outlook:* Bullish")
                    report.append("  *Key Signals:* " + ", ".join(banknifty_analysis['buy_signals'][:3]))
                elif banknifty_analysis['score'] < -3:
                    report.append("  *Weekly Outlook:* Bearish")
                    report.append("  *Key Signals:* " + ", ".join(banknifty_analysis['sell_signals'][:3]))
                else:
                    report.append("  *Weekly Outlook:* Neutral")
        
        report.append("")
        
        # Add sector analysis
        report.append("🏭 *SECTOR PERFORMANCE*")
        
        # Sort sectors by score
        sector_analysis.sort(key=lambda x: x['score'], reverse=True)
        
        for sector in sector_analysis:
            sector_name = sector['symbol'].replace('.NS', '')
            change_pct = sector['change'] * 100
            direction = "🟢" if change_pct >= 0 else "🔴"
            
            report.append(f"*{sector_name}:* {direction} {sector['close']:.2f} ({change_pct:.2f}%)")
            
            if sector['score'] > 3:
                report.append("  *Outlook:* Bullish")
            elif sector['score'] < -3:
                report.append("  *Outlook:* Bearish")
            else:
                report.append("  *Outlook:* Neutral")
        
        report.append("")
        
        # Add top stocks to watch
        report.append("👀 *TOP STOCKS TO WATCH*")
        
        # Sort by absolute score first, then by actual score for display
        strongest_signals = sorted(all_signals, key=lambda x: abs(x['score']), reverse=True)[:10]
        strongest_signals.sort(key=lambda x: x['score'], reverse=True)
        
        for signal in strongest_signals:
            symbol_display = signal['symbol'].replace('.NS', '')
            change_pct = signal['change'] * 100
            direction = "🟢" if signal['score'] > 0 else "🔴" if signal['score'] < 0 else "⚪"
            
            report.append(f"*{symbol_display}:* {direction} {signal['close']:.2f} ({change_pct:.2f}%)")
            
            if signal['score'] > 3:
                report.append("  *Outlook:* Bullish - " + ", ".join(signal['buy_signals'][:2]))
            elif signal['score'] < -3:
                report.append("  *Outlook:* Bearish - " + ", ".join(signal['sell_signals'][:2]))
            else:
                report.append("  *Outlook:* Neutral")
        
        report.append("")
        
        # Add levels to watch
        if nifty_analysis:
            report.append("🎯 *KEY LEVELS TO WATCH*")
            
            # Calculate support and resistance for Nifty
            if nifty_data is not None and len(nifty_data) > 20:
                nifty_close = nifty_analysis['close']
                
                # Simple support/resistance calculation
                recent_highs = nifty_data['High'][-20:].nlargest(3)
                recent_lows = nifty_data['Low'][-20:].nsmallest(3)
                
                # Get closest resistance levels above current price
                resistance_levels = [price for price in recent_highs if price > nifty_close]
                resistance_levels.sort()
                
                # Get closest support levels below current price
                support_levels = [price for price in recent_lows if price < nifty_close]
                support_levels.sort(reverse=True)
                
                # Report levels
                report.append(f"*NIFTY 50 Support Levels:* " + ", ".join([f"{level:.1f}" for level in support_levels[:2]]))
                report.append(f"*NIFTY 50 Resistance Levels:* " + ", ".join([f"{level:.1f}" for level in resistance_levels[:2]]))
        
        report.append("")
        
        # Add disclaimer
        report.append("⚠️ *Disclaimer:* This weekly outlook is generated using automated technical analysis and should not be considered as financial advice. Always do your own research before making investment decisions.")
        
        # Join report and save to file
        report_text = "\n".join(report)
        
        # Save report to file
        report_file = os.path.join(RECOMMENDATIONS_DIR, f"weekly_outlook_{datetime.now().strftime('%Y%m%d')}.md")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Weekly outlook generated and saved to {report_file}")
        
        # Send report via Telegram
        if TELEGRAM_ENABLED:
            send_telegram_message(report_text)
        
        return report_text
    
    except Exception as e:
        error_msg = f"Error generating weekly outlook: {e}"
        logger.error(error_msg)
        if TELEGRAM_ENABLED:
            send_telegram_message(f"⚠️ *Error:* {error_msg}")
        return None

# Schedule and run tasks
def schedule_tasks():
    """Schedule all tasks."""
    india_tz = pytz.timezone('Asia/Kolkata')
    
    # Check market state before scheduling
    market_state, _ = check_market_state()
    logger.info(f"Scheduling tasks - Current market state: {market_state}")
    
    # Daily tasks
    # Generate daily insights at 8:30 AM IST (before market open)
    schedule.every().day.at("08:30").do(generate_daily_insights)
    
    # Intraday scanning during market hours
    schedule.every().day.at("09:45").do(scan_intraday_signals)  # 30 min after open
    schedule.every().day.at("11:00").do(scan_intraday_signals)  # Mid-morning
    schedule.every().day.at("12:30").do(scan_intraday_signals)  # Lunch time
    schedule.every().day.at("14:00").do(scan_intraday_signals)  # Mid-afternoon
    schedule.every().day.at("15:15").do(scan_intraday_signals)  # Before close
    
    # Weekly tasks
    # Generate weekly outlook on Sunday at 18:00 IST
    schedule.every().sunday.at("18:00").do(generate_weekly_outlook)
    
    logger.info("All tasks scheduled successfully")
    
    # Send startup notification
    startup_message = f"🤖 *Trading Bot Started* 🤖\nCurrent Market State: {market_state.upper()}\nScheduled tasks have been set up successfully."
    if TELEGRAM_ENABLED:
        send_telegram_message(startup_message)

# Run the scheduler continuously
# Run the scheduler continuously
def run_scheduler():
    """Run the task scheduler in an infinite loop."""
    try:
        logger.info("Starting scheduler loop")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except Exception as e:
        logger.error(f"Error in scheduler loop: {e}")
        # Only log to terminal, not Telegram
        print(f"Error in scheduler loop: {e}")

# Function to display trading recommendations
# Function to display trading recommendations
def display_trading_recommendations(signals):
    """Display actual trading recommendations based on signal scores."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n===== STOCK RECOMMENDATIONS =====")
    print(f"Generated on: {timestamp}\n")
    
    # Check if signals is a string instead of a dictionary
    if isinstance(signals, str):
        print("No tradable signals available or error in generation:")
        print(signals)
        print("================================")
        return []
    
    # Continue with dictionary processing as before
    # Sort stocks by signal score (highest first)
    sorted_stocks = sorted(signals.items(), key=lambda x: x[1].get('score', 0), reverse=True)
    
    # Rest of the function remains the same...
    
    # Prepare recommendations for display and Telegram
    telegram_recommendations = []
    
    for stock, data in sorted_stocks:
        score = data.get('score', 0)
        action = "STRONG BUY" if score >= 8 else "BUY" if score >= 5 else "HOLD" if score >= 3 else "SELL"
        price = data.get('current_price', 'N/A')
        
        if isinstance(price, (int, float)):
            price_display = f"₹{price:.2f}"
        else:
            price_display = str(price)
        
        # Calculate target price based on technical indicators (example calculation)
        target_price = None
        if isinstance(price, (int, float)):
            # Simple target calculation based on score and current price
            if score >= 8:
                target_price = price * 1.10  # 10% growth potential for strong buys
            elif score >= 5:
                target_price = price * 1.05  # 5% growth potential for buys
            elif score <= 2:
                target_price = price * 0.95  # 5% drop expected for sells
        
        target_display = f"₹{target_price:.2f}" if target_price else "N/A"
        
        # Extract reasons based on indicators
        reasons = []
        if 'indicators' in data:
            indicators = data['indicators']
            if 'rsi_oversold' in indicators:
                reasons.append("Oversold (RSI)")
            if 'macd_bullish' in indicators:
                reasons.append("Bullish MACD crossover")
            if 'golden_cross' in indicators:
                reasons.append("Golden Cross (MA crossover)")
            if 'breakout' in indicators:
                reasons.append("Price breakout")
            if 'support' in indicators:
                reasons.append("At support level")
            if 'resistance' in indicators:
                reasons.append("Near resistance")
            if 'volume_spike' in indicators:
                reasons.append("Unusual volume")
            
        # If no specific reasons were extracted, use generic ones based on score
        if not reasons:
            if score >= 8:
                reasons.append("Strong technical indicators")
            elif score >= 5:
                reasons.append("Positive technical setup")
            elif score <= 2:
                reasons.append("Bearish technical indicators")
                
        reason_text = ", ".join(reasons) if reasons else "Based on technical analysis"
        
        # Display on terminal
        print(f"{stock}: {action} (Score: {score})")
        print(f"  Current Price: {price_display}")
        print(f"  Target Price: {target_display}")
        print(f"  Reason: {reason_text}")
        if 'indicators' in data:
            print(f"  Technical indicators: {', '.join(data['indicators'])}")
        print("")  # Empty line between stocks
        
        # Only add BUY and STRONG BUY recommendations to Telegram message
        if score >= 5:  # Only BUY and STRONG BUY
            telegram_recommendations.append({
                'symbol': stock,
                'action': action,
                'price': price_display,
                'target': target_display,
                'reason': reason_text
            })
    
    print("================================")
    
    # Send only recommendations to Telegram, not status messages
    if TELEGRAM_ENABLED and telegram_recommendations:
        telegram_message = f"📈 *Stock Recommendations ({timestamp})*\n\n"
        for rec in telegram_recommendations[:5]:  # Limit to top 5 to avoid message size issues
            telegram_message += f"*{rec['symbol']}*: {rec['action']}\n"
            telegram_message += f"Price: {rec['price']} → Target: {rec['target']}\n"
            telegram_message += f"Reason: {rec['reason']}\n\n"
        
        send_telegram_message(telegram_message)
    
    return sorted_stocks

# Main function
def main():
    """Main function to start the trading bot."""
    try:
        print("=" * 50)
        print("TRADING BOT STARTING")
        print("=" * 50)
        logger.info("Trading bot starting")
        
        # Check for required directories
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        os.makedirs(RECOMMENDATIONS_DIR, exist_ok=True)
        
        # Initialize the scheduler
        schedule_tasks()
        
        # Generate initial insights and display them
        print("Generating initial insights on startup")
        daily_insights = generate_daily_insights()
        
        # Display actual trading recommendations rather than just logging
        if daily_insights:
            recommended_stocks = display_trading_recommendations(daily_insights)
            print(f"Generated recommendations for {len(recommended_stocks)} stocks")
        
        # Run the scheduler in a separate thread
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        print("Scheduler running in background. Press Ctrl+C to exit.")
        
        # Keep the main thread alive with a more responsive interrupt handler
        try:
            while True:
                time.sleep(1)  # More responsive to keyboard interrupt
        except KeyboardInterrupt:
            # Graceful exit
            print("\nShutting down trading bot. Please wait...")
            # Give tasks time to clean up
            time.sleep(2)
            print("Trading bot stopped.")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("Trading bot stopped by user")
        logger.info("Trading bot stopped by user")
    except Exception as e:
        error_msg = f"Critical error in main: {e}"
        print(error_msg)
        logger.error(error_msg)
        raise

# API endpoints with improved result handling
def start_api_server(port=5000):
    """Start a simple API server for external integrations."""
    from flask import Flask, jsonify, request, render_template_string
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        market_state, market_message = check_market_state()
        return jsonify({
            'status': 'ok',
            'market_state': market_state,
            'market_message': market_message,
            'time': datetime.now().isoformat()
        })
    
    @app.route('/signals', methods=['GET'])
    def get_signals():
        """Get signals for specific stocks."""
        symbols = request.args.get('symbols', '').split(',')
        interval = request.args.get('interval', '1d')
        intraday = request.args.get('intraday', 'false').lower() == 'true'
        
        if not symbols or symbols[0] == '':
            # Default to indices if no symbols provided
            symbols = INDICES
        
        # Generate signals
        signals = generate_technical_signals(symbols, interval, intraday)
        
        # Add target prices and reasons to the signals
        for symbol, data in signals.items():
            if 'current_price' in data and isinstance(data['current_price'], (int, float)):
                price = data['current_price']
                score = data.get('score', 0)
                
                # Calculate target price
                if score >= 8:
                    data['target_price'] = price * 1.10
                elif score >= 5:
                    data['target_price'] = price * 1.05
                elif score <= 2:
                    data['target_price'] = price * 0.95
                else:
                    data['target_price'] = price
                
                # Add reasoning
                reasons = []
                if 'indicators' in data:
                    indicators = data['indicators']
                    if 'rsi_oversold' in indicators:
                        reasons.append("Oversold (RSI)")
                    if 'macd_bullish' in indicators:
                        reasons.append("Bullish MACD crossover")
                    if 'golden_cross' in indicators:
                        reasons.append("Golden Cross (MA crossover)")
                    if 'breakout' in indicators:
                        reasons.append("Price breakout")
                
                data['reasons'] = reasons if reasons else ["Based on technical analysis"]
        
        # Serve HTML if requested
        if request.args.get('format', '').lower() == 'html':
            # Enhanced HTML template with target prices and reasons
            template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Stock Signals</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .buy { color: green; font-weight: bold; }
                    .sell { color: red; font-weight: bold; }
                    .hold { color: orange; }
                </style>
            </head>
            <body>
                <h1>Stock Signals ({{ timestamp }})</h1>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Score</th>
                        <th>Action</th>
                        <th>Current Price</th>
                        <th>Target Price</th>
                        <th>Reasons</th>
                    </tr>
                    {% for symbol, data in signals.items() %}
                    <tr>
                        <td>{{ symbol }}</td>
                        <td>{{ data.get('score', 'N/A') }}</td>
                        <td class="{% if data.get('score', 0) >= 5 %}buy{% elif data.get('score', 0) <= 2 %}sell{% else %}hold{% endif %}">
                            {% if data.get('score', 0) >= 8 %}STRONG BUY
                            {% elif data.get('score', 0) >= 5 %}BUY
                            {% elif data.get('score', 0) >= 3 %}HOLD
                            {% else %}SELL{% endif %}
                        </td>
                        <td>{% if data.get('current_price') %}₹{{ "%.2f"|format(data.get('current_price')) }}{% else %}N/A{% endif %}</td>
                        <td>{% if data.get('target_price') %}₹{{ "%.2f"|format(data.get('target_price')) }}{% else %}N/A{% endif %}</td>
                        <td>{{ ', '.join(data.get('reasons', [])) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </body>
            </html>
            """
            return render_template_string(template, signals=signals, timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Otherwise return JSON
        return jsonify({
            'signals': signals,
            'count': len(signals),
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/reports/daily', methods=['GET'])
    def get_daily_report():
        """Generate and return daily report."""
        report = generate_daily_insights()
        
        # Return as HTML if requested
        if request.args.get('format', '').lower() == 'html':
            with open(os.path.join(RECOMMENDATIONS_DIR, f"daily_report_{datetime.now().strftime('%Y%m%d')}.md"), 'r') as f:
                content = f.read()
                return render_template_string("<!DOCTYPE html><html><head><title>Daily Report</title></head><body><pre>{{ content }}</pre></body></html>", content=content)
        
        return jsonify({
            'report': report,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/reports/weekly', methods=['GET'])
    def get_weekly_report():
        """Generate and return weekly report."""
        report = generate_weekly_outlook()
        return jsonify({
            'report': report,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/', methods=['GET'])
    def home():
        """Home page with links to available endpoints."""
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Bot API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; max-width: 800px; }
                .endpoint { background: #f7f7f7; padding: 10px; margin: 10px 0; border-radius: 5px; }
                h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Trading Bot API</h1>
            <p>Server time: {{ time }}</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <h3>/health</h3>
                <p>Check system health and market status</p>
                <a href="/health">View Health Status</a>
            </div>
            
            <div class="endpoint">
                <h3>/signals</h3>
                <p>Get signals for stocks</p>
                <p>Parameters: symbols (comma-separated), interval (1d, 5m, etc), intraday (true/false), format (json/html)</p>
                <a href="/signals?format=html">View Default Signals</a>
            </div>
            
            <div class="endpoint">
                <h3>/reports/daily</h3>
                <p>Generate daily market report</p>
                <a href="/reports/daily?format=html">View Daily Report</a>
            </div>
            
            <div class="endpoint">
                <h3>/reports/weekly</h3>
                <p>Generate weekly market outlook</p>
                <a href="/reports/weekly">View Weekly Report</a>
            </div>
        </body>
        </html>
        """, time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Start the server in a separate thread
    api_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port, debug=False))
    api_thread.daemon = True
    api_thread.start()
    
    print(f"\n✅ API server running at http://localhost:{port}")
    logger.info(f"API server started on port {port}")

# Command-line interface function
def parse_args():
    """Parse command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Indian Stock Market Trading Bot')
    parser.add_argument('--no-telegram', action='store_true', help='Disable Telegram notifications')
    parser.add_argument('--api', action='store_true', help='Start API server')
    parser.add_argument('--api-port', type=int, default=5000, help='API server port (default: 5000)')
    parser.add_argument('--daily-report', action='store_true', help='Generate daily report and exit')
    parser.add_argument('--weekly-report', action='store_true', help='Generate weekly report and exit')
    parser.add_argument('--analyze', type=str, help='Analyze specific stock symbol and exit')
    parser.add_argument('--interval', type=str, default='1d', help='Interval for analysis (default: 1d)')
    parser.add_argument('--show-results', action='store_true', help='Display results on console instead of just logging')
    
    return parser.parse_args()

# Modify send_telegram_message to only send trade recommendations
def send_telegram_message(message):
    """Send message to Telegram channel, but only for recommendations."""
    if not TELEGRAM_ENABLED:
        return
    
    # Only send messages that contain stock recommendations
    # Check if this is a recommendation message (contains 'Stock Recommendations')
    if "Stock Recommendations" not in message and "BUY" not in message and "SELL" not in message:
        # Just log other messages locally instead of sending
        logger.info(f"[Not sent to Telegram] {message}")
        return
    
    # Continue with sending recommendation messages
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        print(f"Failed to send Telegram message: {e}")

# Entry point
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Check for one-time actions
    if args.no_telegram:
        TELEGRAM_ENABLED = False
        print("Telegram notifications disabled by command-line argument")
    
    if args.daily_report:
        print("Generating daily report")
        daily_insights = generate_daily_insights()
        if daily_insights and args.show_results:
            display_trading_recommendations(daily_insights)
        sys.exit(0)
    
    if args.weekly_report:
        print("Generating weekly report")
        weekly_outlook = generate_weekly_outlook()
        print(f"Weekly report saved to {RECOMMENDATIONS_DIR}/weekly_outlook_{datetime.now().strftime('%Y%m%d')}.md")
        sys.exit(0)
    
    if args.analyze:
        print(f"Analyzing stock {args.analyze}")
        symbol = args.analyze
        if not symbol.endswith('.NS') and symbol not in INDICES:
            symbol = f"{symbol}.NS"
        
        try:
            df = fetch_stock_data(symbol, interval=args.interval)
            if df is not None:
                analysis = analyze_stock(symbol, df, intraday=args.interval in ['5m', '15m', '30m', '60m', '1h'])
                if analysis:
                    # Calculate target price
                    score = analysis.get('score', 0)
                    current_price = analysis.get('current_price')
                    
                    if isinstance(current_price, (int, float)):
                        if score >= 8:
                            target_price = current_price * 1.10
                        elif score >= 5:
                            target_price = current_price * 1.05
                        elif score <= 2:
                            target_price = current_price * 0.95
                        else:
                            target_price = current_price
                        
                        analysis['target_price'] = target_price
                    
                    # Add reasoning
                    reasons = []
                    if 'indicators' in analysis:
                        indicators = analysis['indicators']
                        if 'rsi_oversold' in indicators:
                            reasons.append("Oversold (RSI)")
                        if 'macd_bullish' in indicators:
                            reasons.append("Bullish MACD crossover")
                        if 'golden_cross' in indicators:
                            reasons.append("Golden Cross (MA crossover)")
                        if 'breakout' in indicators:
                            reasons.append("Price breakout")
                    
                    analysis['reasons'] = reasons if reasons else ["Based on technical analysis"]
                    
                    print(json.dumps(analysis, indent=2))
                else:
                    print(f"Could not analyze {symbol}")
            else:
                print(f"No data available for {symbol}")
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
        
        sys.exit(0)
    
    # Start API server if requested
    if args.api:
        print(f"Starting API server on port {args.api_port}")
        start_api_server(port=args.api_port)
    
    # Start the main function
    main()
