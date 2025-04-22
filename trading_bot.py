import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
import logging
import json
import sqlite3
import threading
import schedule
import requests
import pytz
from datetime import datetime, time as dt_time
from queue import Queue
from bs4 import BeautifulSoup
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

# Configuration
CONFIG = {
    'cache': {
        'data_dir': './data_cache',
        'recommendations_dir': './recommendations',
        'logs_dir': './logs',
        'expiry_hours': 2,
        'intraday_expiry_minutes': 15
    },
    'api': {
        'max_retry_attempts': 3,
        'min_request_delay': 2,
        'max_request_delay': 5,
        'calls_per_minute': 30  # Reduced to avoid rate limiting
    },
    'stocks': {
        'indices': ['^NSEI', '^NSEBANK'],
        'analysis': [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
            "INFY.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "AXISBANK.NS", "HINDALCO.NS",
            "ITC.NS", "BHARTIARTL.NS", "ADANIENT.NS", "ONGC.NS", "NTPC.NS",
            "JINDALSTEL.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "MARUTI.NS", "SUNPHARMA.NS",
            "LT.NS", "WIPRO.NS", "COALINDIA.NS", "JSWSTEEL.NS", "TATAPOWER.NS",
            "HCLTECH.NS", "BPCL.NS", "HINDUNILVR.NS", "M&M.NS", "ADANIPORTS.NS",
            "POWERGRID.NS", "VEDL.NS", "BANKBARODA.NS", "SAIL.NS", "ASIANPAINT.NS",
            "APOLLOHOSP.NS", "PFC.NS", "RECLTD.NS", "GAIL.NS", "PNB.NS",
            "ASHOKLEY.NS", "IOC.NS", "AMBUJACEM.NS", "DRREDDY.NS", "BAJAJ-AUTO.NS",
            "INDUSINDBK.NS", "HEROMOTOCO.NS", "HINDPETRO.NS", "IDEA.NS", "BIOCON.NS",
            "FEDERALBNK.NS", "CANBK.NS", "TECHM.NS", "CIPLA.NS", "IDFC.NS",
            "NMDC.NS", "BHEL.NS", "ADANIPOWER.NS", "IDBI.NS", "DLF.NS",
            "NATIONALUM.NS", "GMRINFRA.NS", "RBLBANK.NS", "TITAN.NS", "UNIONBANK.NS",
            "MOTHERSON.NS", "DIVISLAB.NS", "INDHOTEL.NS", "IRCTC.NS", "IDFCFIRSTB.NS",
            "ZOMATO.NS", "TATACONSUM.NS", "PIIND.NS", "RVNL.NS", "CUMMINSIND.NS",
            "SUZLON.NS", "GRASIM.NS", "BEL.NS", "LUPIN.NS", "PAYTM.NS",
            "IRFC.NS", "BRITANNIA.NS", "HAVELLS.NS", "GODREJCP.NS", "UPL.NS",
            "CONCOR.NS", "HAL.NS", "NESTLEIND.NS", "INDIGO.NS", "SUNTV.NS",
            "DABUR.NS", "MAHINDCIE.NS", "SIEMENS.NS", "AUROPHARMA.NS", "MCDOWELL-N.NS",
            "JUBLFOOD.NS", "BHARATFORG.NS", "ABFRL.NS", "BANDHANBNK.NS", "BOSCHLTD.NS",
            "CHOLAFIN.NS", "EICHERMOT.NS", "EXIDEIND.NS"
        ]
    },
    'news': [
        {'name': 'MoneyControl', 'url': 'https://www.moneycontrol.com/news/business/markets/'},
        {'name': 'Economic Times', 'url': 'https://economictimes.indiatimes.com/markets/stocks/news'},
        {'name': 'LiveMint', 'url': 'https://www.livemint.com/market/stock-market-news'}
    ],
    'workflow': {
        'daily_run_time': '20:00',  # 8 PM IST for full summary
        'btst_time': '14:00'  # 2 PM IST for BTST recommendations
    }
}

# Required packages list
REQUIRED_PACKAGES = [
    'pandas==2.0.3', 'numpy==1.24.4', 'matplotlib==3.7.5', 'scikit-learn==1.3.2',
    'yfinance==0.2.44', 'alpha_vantage==2.3.1', 'schedule==1.2.2', 'ccxt==4.4.40',
    'python-dotenv==1.0.1', 'requests==2.32.3', 'tensorflow==2.15.0', 'torch==2.2.2',
    'pytz==2024.2', 'plotly==5.24.1', 'typing_extensions==4.12.2', 'bs4==0.0.2',
    'dash==2.18.1', 'TA-Lib==0.4.24'
]

# Setup directories
def setup_directories():
    for directory in [
        CONFIG['cache']['data_dir'],
        CONFIG['cache']['recommendations_dir'],
        CONFIG['cache']['logs_dir']
    ]:
        os.makedirs(directory, exist_ok=True)

# Logging setup
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(CONFIG['cache']['logs_dir'], 'trading_bot.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Install required packages with better error handling
def install_packages():
    logger.info("Checking and installing required packages...")
    success_count = 0
    failure_count = 0
    
    for package in REQUIRED_PACKAGES:
        package_name = package.split('==')[0]
        module_name = package_name.replace('-', '_')
        
        try:
            # Try to import first to avoid unnecessary installations
            __import__(module_name)
            logger.info(f"{package_name} already installed.")
            success_count += 1
        except ImportError:
            logger.info(f"Installing {package}...")
            try:
                # Use --no-cache-dir to avoid cache issues
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 
                    '--no-cache-dir', package
                ])
                logger.info(f"Successfully installed {package}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                failure_count += 1
                
                # Special handling for TA-Lib
                if package_name == 'TA-Lib':
                    logger.info("Attempting to install TA-Lib prebuilt wheel...")
                    try:
                        # Try to install the pre-built wheel for the correct Python version
                        python_version = f"{sys.version_info.major}{sys.version_info.minor}"
                        wheel_url = f"https://github.com/TA-Lib/ta-lib-python/releases/download/v0.4.24/ta_lib-0.4.24-cp{python_version}-cp{python_version}-win_amd64.whl"
                        subprocess.check_call([
                            sys.executable, '-m', 'pip', 'install', 
                            '--no-cache-dir', wheel_url
                        ])
                        logger.info(f"Successfully installed TA-Lib from wheel")
                        success_count += 1
                        failure_count -= 1  # Correct the count since we recovered
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to install TA-Lib wheel: {e}")
    
    logger.info(f"Package installation completed: {success_count} succeeded, {failure_count} failed")
    if failure_count > 0:
        logger.warning("Some packages failed to install. This may impact functionality.")

# Telegram setup with thread-safe implementation
class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.bot_token and self.chat_id)
        self.message_queue = Queue()
        self.max_retries = 3
        self.lock = threading.Lock()
        
        if self.enabled:
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()
            logger.info("Telegram notification system initialized")
        else:
            logger.warning("Telegram notifications disabled (missing credentials)")
    
    def send_message(self, message):
        if not self.enabled:
            logger.info(f"Telegram disabled. Message: {message[:50]}...")
            return False
        
        self.message_queue.put(message)
        return True
    
    def _process_queue(self):
        while True:
            try:
                message = self.message_queue.get()
                self._send_message_with_retry(message)
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Error in Telegram queue processor: {e}")
            time.sleep(1)
    
    def _send_message_with_retry(self, message, attempt=0):
        if attempt >= self.max_retries:
            logger.error(f"Failed to send Telegram message after {self.max_retries} attempts")
            return False
        
        try:
            with self.lock:
                response = requests.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                    json={'chat_id': self.chat_id, 'text': message, 'parse_mode': 'Markdown'},
                    timeout=30
                )
                response.raise_for_status()
                logger.info("Telegram message sent successfully")
                return True
        except Exception as e:
            logger.error(f"Telegram send attempt {attempt+1}/{self.max_retries} failed: {e}")
            time.sleep(2 * (attempt + 1))
            return self._send_message_with_retry(message, attempt + 1)

# Initialize Telegram notifier
telegram = TelegramNotifier()

# Improved SQLite database connection handling
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Stock data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS stock_data (
                        symbol TEXT,
                        interval TEXT,
                        period TEXT,
                        timestamp REAL,
                        data TEXT,
                        PRIMARY KEY (symbol, interval, period)
                    )
                ''')
                # News cache table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS news_cache (
                        source TEXT,
                        timestamp REAL,
                        data TEXT,
                        PRIMARY KEY (source, timestamp)
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def _get_connection(self):
        with self.lock:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return self.conn
    
    def close(self):
        with self.lock:
            if self.conn:
                self.conn.close()
                self.conn = None

    def execute(self, query, params=(), fetch_one=False, fetch_all=False):
        with self.lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    conn.commit()
                    
                    if fetch_one:
                        return cursor.fetchone()
                    elif fetch_all:
                        return cursor.fetchall()
                    return True
            except sqlite3.Error as e:
                logger.error(f"Database error: {e} in query {query}")
                return None

# Initialize database manager
db_manager = DatabaseManager(os.path.join(CONFIG['cache']['data_dir'], 'cache.db'))

# Improved rate limiting decorator
def rate_limit(calls_per_minute):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    lock = threading.Lock()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                elapsed = time.time() - last_called[0]
                wait_time = min_interval - elapsed
                
                if wait_time > 0:
                    time.sleep(wait_time)
                
                last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Enhanced yfinance data fetching
@rate_limit(CONFIG['api']['calls_per_minute'])
def fetch_stock_data(symbol, interval='1d', period='1mo'):
    """Fetch stock data for a single symbol with caching"""
    try:
        # Check if valid cache exists
        query = "SELECT timestamp, data FROM stock_data WHERE symbol=? AND interval=? AND period=?"
        result = db_manager.execute(query, (symbol, interval, period), fetch_one=True)
        
        now = time.time()
        expiry_seconds = (CONFIG['cache']['intraday_expiry_minutes'] * 60 
                         if interval in ['5m', '15m', '30m', '60m', '1h']
                         else CONFIG['cache']['expiry_hours'] * 3600)
        
        # Use cached data if valid
        if result and (now - result[0]) <= expiry_seconds:
            try:
                df = pd.read_json(result[1])
                if not df.empty and len(df) >= 5:
                    logger.debug(f"Using cached data for {symbol}")
                    return df
            except Exception as e:
                logger.warning(f"Corrupted cache for {symbol}: {e}")
        
        # Import yfinance here to avoid startup issues if not installed
        import yfinance as yf
        
        # Fetch fresh data
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty or len(df) < 5:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        # Cache the fresh data
        query = "INSERT OR REPLACE INTO stock_data VALUES (?, ?, ?, ?, ?)"
        db_manager.execute(query, (symbol, interval, period, now, df.to_json()))
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

# Batch stock data fetching with parallel processing
def fetch_batch_stock_data(symbols, interval='1d', period='1mo', max_workers=5):
    """Fetch stock data for multiple symbols in parallel"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all symbols for processing
        future_to_symbol = {
            executor.submit(fetch_stock_data, symbol, interval, period): symbol
            for symbol in symbols
        }
        
        # Process completed futures
        for future in future_to_symbol:
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                results[symbol] = data
            except Exception as e:
                logger.error(f"Error in parallel fetch for {symbol}: {e}")
                results[symbol] = None
    
    return results

# Calculate technical indicators
def calculate_indicators(df):
    if df is None or len(df) < 14:
        return None
    
    try:
        data = df.copy()
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Moving averages
        for window in [20, 50, 200]:
            data[f'SMA{window}'] = close.rolling(window=window).mean()
        
        # Exponential moving averages
        for span, name in [(9, 'EMA9'), (12, 'EMA12'), (21, 'EMA21'), (26, 'EMA26')]:
            data[name] = close.ewm(span=span, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = close.rolling(window=20).mean()
        data['BB_Std'] = close.rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        data['ATR'] = tr.rolling(14).mean()
        
        return data
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

# Analyze stock with improved error handling
def analyze_stock(symbol, data, is_btst=False):
    try:
        min_data_points = 30
        if data is None or len(data) < min_data_points:
            logger.warning(f"Insufficient data points for {symbol}: {len(data) if data is not None else 0} < {min_data_points}")
            return None
        
        data_with_indicators = calculate_indicators(data)
        if data_with_indicators is None:
            logger.warning(f"Failed to calculate indicators for {symbol}")
            return None
        
        latest = data_with_indicators.iloc[-1]
        prev = data_with_indicators.iloc[-2] if len(data_with_indicators) > 1 else None
        latest_close = latest['Close']
        
        buy_signals = []
        sell_signals = []
        score = 0
        
        # RSI analysis
        if 'RSI' in latest and not pd.isna(latest['RSI']):
            if latest['RSI'] < 30:
                buy_signals.append(f"RSI oversold ({latest['RSI']:.1f})")
                score += 3
            elif latest['RSI'] > 70:
                sell_signals.append(f"RSI overbought ({latest['RSI']:.1f})")
                score -= 3
        
        # Moving average analysis
        if 'SMA20' in latest and 'SMA50' in latest and not pd.isna(latest['SMA20']) and not pd.isna(latest['SMA50']):
            if latest_close > latest['SMA20'] and latest_close > latest['SMA50']:
                buy_signals.append("Price above SMA20 and SMA50")
                score += 1
            elif latest_close < latest['SMA20'] and latest_close < latest['SMA50']:
                sell_signals.append("Price below SMA20 and SMA50")
                score -= 1
        
        # MACD analysis
        if 'MACD' in latest and 'Signal' in latest and not pd.isna(latest['MACD']) and not pd.isna(latest['Signal']):
            if latest['MACD'] > latest['Signal']:
                buy_signals.append("MACD crossover (bullish)")
                score += 2
            elif latest['MACD'] < latest['Signal']:
                sell_signals.append("MACD crossover (bearish)")
                score -= 2
        
        # Bollinger Bands analysis
        if 'BB_Lower' in latest and 'BB_Upper' in latest and not pd.isna(latest['BB_Lower']) and not pd.isna(latest['BB_Upper']):
            if latest_close < latest['BB_Lower']:
                buy_signals.append("Price below lower Bollinger Band")
                score += 2
            elif latest_close > latest['BB_Upper']:
                sell_signals.append("Price above upper Bollinger Band")
                score -= 2
        
        # BTST-specific logic
        if is_btst and score >= 5:
            buy_signals.append("BTST candidate")
            score += 2
        
        # Calculate price change
        price_change = 0
        if prev is not None and not pd.isna(prev['Close']):
            price_change = latest_close / prev['Close'] - 1
            
            # Add momentum to score
            if price_change > 0.02:  # 2% up
                buy_signals.append(f"Strong upward momentum ({price_change:.1%})")
                score += 2
            elif price_change < -0.02:  # 2% down
                sell_signals.append(f"Strong downward momentum ({price_change:.1%})")
                score -= 2
        
        result = {
            'symbol': symbol,
            'close': latest_close,
            'change': price_change,
            'score': score,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'timestamp': latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
        }
        return result
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

# Market state check
def check_market_state():
    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(india_tz)
        
        if now.weekday() >= 5:
            return 'closed', 'Weekend'
        
        market_open = datetime.combine(now.date(), dt_time(9, 15)).replace(tzinfo=india_tz)
        market_close = datetime.combine(now.date(), dt_time(15, 30)).replace(tzinfo=india_tz)
        
        if market_open <= now < market_close:
            return 'open', 'Market open'
        
        return 'closed', 'Outside trading hours'
    except Exception as e:
        logger.error(f"Error checking market state: {e}")
        return 'unknown', 'Unable to determine market state'

# Improved news fetching with better error handling
def get_market_news(max_articles=5):
    try:
        all_news = []
        now = time.time()
        cache_expiry = 3600  # 1 hour cache
        
        # Try to get news from cache first
        query = "SELECT timestamp, data FROM news_cache WHERE timestamp > ?"
        result = db_manager.execute(query, (now - cache_expiry,), fetch_one=True)
        
        if result:
            try:
                cached_news = json.loads(result[1])
                logger.info(f"Using cached news ({len(cached_news)} articles)")
                return cached_news[:max_articles]
            except json.JSONDecodeError:
                logger.warning("Corrupted news cache, fetching fresh data")
        
        # Fetch fresh news
        for source in CONFIG['news']:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive'
                }
                
                response = requests.get(source['url'], headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = []
                
                if source['name'] == 'MoneyControl':
                    articles = soup.select('.article_box')[:max_articles]
                    for article in articles:
                        title_elem = article.select_one('h3 a')
                        if title_elem:
                            title = title_elem.text.strip()
                            link = title_elem.get('href', '')
                            if link and not link.startswith('http'):
                                link = f"https://www.moneycontrol.com{link}"
                            
                            all_news.append({
                                'source': source['name'],
                                'title': title,
                                'link': link
                            })
                elif source['name'] == 'Economic Times':
                    articles = soup.select('.eachStory')[:max_articles]
                    for article in articles:
                        title_elem = article.select_one('h3 a')
                        if title_elem:
                            title = title_elem.text.strip()
                            link = title_elem.get('href', '')
                            if link and not link.startswith('http'):
                                link = f"https://economictimes.indiatimes.com{link}"
                            
                            all_news.append({
                                'source': source['name'],
                                'title': title,
                                'link': link
                            })
                elif source['name'] == 'LiveMint':
                    articles = soup.select('.headline')[:max_articles]
                    for article in articles:
                        title_elem = article.select_one('a')
                        if title_elem:
                            title = title_elem.text.strip()
                            link = title_elem.get('href', '')
                            if link and not link.startswith('http'):
                                link = f"https://www.livemint.com{link}"
                            
                            all_news.append({
                                'source': source['name'],
                                'title': title,
                                'link': link
                            })
                
                logger.info(f"Fetched {len(articles)} articles from {source['name']}")
                
            except Exception as e:
                logger.error(f"Error fetching news from {source['name']}: {e}")
        
        # Cache the news results
        if all_news:
            query = "INSERT OR REPLACE INTO news_cache VALUES (?, ?, ?)"
            db_manager.execute(query, ('combined', now, json.dumps(all_news)))
        
        return all_news[:max_articles]
    except Exception as e:
        logger.error(f"Error in get_market_news: {e}")
        return []

# Generate technical signals with improved processing
def generate_technical_signals(stocks_list, interval='1d', is_btst=False):
    signals = []
    batch_size = 10
    total_stocks = len(stocks_list)
    processed = 0
    
    logger.info(f"Starting technical analysis for {total_stocks} stocks")
    
    try:
        for i in range(0, total_stocks, batch_size):
            batch = stocks_list[i:min(i + batch_size, total_stocks)]
            data = fetch_batch_stock_data(batch, interval=interval, period='90d')
            
            for symbol in batch:
                try:
                    df = data.get(symbol)
                    if df is not None and not df.empty:
                        result = analyze_stock(symbol, df, is_btst)
                        if result:
                            signals.append(result)
                            logger.info(f"Generated signals for {symbol}: Score={result['score']}")
                except Exception as e:
                    logger.error(f"Error generating signals for {symbol}: {e}")
            
            processed += len(batch)
            logger.info(f"Processed {processed}/{total_stocks} stocks")
            
            # Add a short delay between batches to avoid overwhelming the API
            time.sleep(1)
    
    except Exception as e:
        logger.error(f"Error in generate_technical_signals: {e}")
    
    # Sort signals by absolute score (strongest signals first)
    signals.sort(key=lambda x: abs(x['score']), reverse=True)
    logger.info(f"Generated {len(signals)} signals in total")
    
    return signals

# Generate recommendations based on time
def generate_recommendations(is_btst=False):
    try:
        logger.info(f"Generating {'BTST' if is_btst else 'daily'} recommendations")
        
        market_state, market_message = check_market_state()
        news = get_market_news(max_articles=5)
        signals = generate_technical_signals(CONFIG['stocks']['analysis'], is_btst=is_btst)
        
        report = []
        today_date = datetime.now().strftime('%d %b %Y')
        time_now = datetime.now().strftime('%H:%M')
        
        # Greeting based on time and recommendation type
        greeting = "🌞 Good Morning!" if is_btst or market_state == 'open' else "🌙 Good Evening!"
        report.append(f"{greeting} 📊 *Daily Market Insights - {today_date} {time_now}*")
        report.append(f"📈 *Market Status:* {market_state.upper()} - {market_message}")
        report.append("")
        
        # Categorize recommendations
        recommendations = {'BUY': [], 'SELL': [], 'Changed': []}
        
        for signal in signals:
            score = signal['score']
            price = signal['close']
            change_pct = signal['change'] * 100
            
            if isinstance(price, (int, float)):
                target_price = price * 1.10 if score >= 8 else price * 1.05 if score >= 5 else price * 1.03
            stop_loss = price * 0.95 if score >= 8 else price * 0.97 if score >= 5 else price * 0.98
            
            if score >= 5:
                signal_type = "BUY"
                recommendations['BUY'].append({
                    'symbol': signal['symbol'],
                    'price': price,
                    'target': target_price,
                    'stop_loss': stop_loss,
                    'score': score,
                    'signals': signal['buy_signals']
                })
            elif score <= -5:
                signal_type = "SELL"
                recommendations['SELL'].append({
                    'symbol': signal['symbol'],
                    'price': price,
                    'target': price * 0.90,
                    'stop_loss': price * 1.03,
                    'score': score,
                    'signals': signal['sell_signals']
                })
            elif abs(change_pct) >= 2.0:
                recommendations['Changed'].append({
                    'symbol': signal['symbol'],
                    'price': price,
                    'change': change_pct,
                    'score': score
                })
        
        # Add news to report
        if news:
            report.append("📰 *Latest Market News:*")
            for i, article in enumerate(news, 1):
                report.append(f"{i}. [{article['title']}]({article['link']}) - {article['source']}")
            report.append("")
        
        # Index performances
        try:
            indices_data = fetch_batch_stock_data(CONFIG['stocks']['indices'], interval='1d', period='2d')
            report.append("📊 *Index Performance:*")
            
            for index in CONFIG['stocks']['indices']:
                df = indices_data.get(index)
                if df is not None and len(df) >= 2:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2]
                    change_pct = (latest['Close'] / prev['Close'] - 1) * 100
                    
                    index_name = "Nifty 50" if index == "^NSEI" else "Nifty Bank" if index == "^NSEBANK" else index
                    report.append(f"- {index_name}: {latest['Close']:.2f} ({change_pct:+.2f}%)")
            
            report.append("")
        except Exception as e:
            logger.error(f"Error adding index performance: {e}")
        
        # Add BUY recommendations
        if recommendations['BUY']:
            report.append("🟢 *BUY Recommendations:*")
            for rec in recommendations['BUY'][:5]:  # Top 5 BUY recommendations
                report.append(f"- {rec['symbol'].replace('.NS', '')}: ₹{rec['price']:.2f} | Target: ₹{rec['target']:.2f} | SL: ₹{rec['stop_loss']:.2f}")
                report.append(f"  *Signals:* {', '.join(rec['signals'])}")
            report.append("")
        
        # Add SELL recommendations
        if recommendations['SELL']:
            report.append("🔴 *SELL Recommendations:*")
            for rec in recommendations['SELL'][:5]:  # Top 5 SELL recommendations
                report.append(f"- {rec['symbol'].replace('.NS', '')}: ₹{rec['price']:.2f} | Target: ₹{rec['target']:.2f} | SL: ₹{rec['stop_loss']:.2f}")
                report.append(f"  *Signals:* {', '.join(rec['signals'])}")
            report.append("")
        
        # Add significant movers
        if recommendations['Changed']:
            report.append("⚡ *Significant Movers Today:*")
            for rec in recommendations['Changed'][:5]:  # Top 5 movers
                change_symbol = "🔺" if rec['change'] > 0 else "🔻"
                report.append(f"- {rec['symbol'].replace('.NS', '')}: ₹{rec['price']:.2f} {change_symbol} {abs(rec['change']):.2f}%")
            report.append("")
        
        # BTST specific content
        if is_btst:
            report.append("🌙 *BTST (Buy Today Sell Tomorrow) Picks:*")
            btst_picks = [rec for rec in recommendations['BUY'] if rec['score'] >= 6]
            
            if btst_picks:
                for pick in btst_picks[:3]:  # Top 3 BTST picks
                    report.append(f"- {pick['symbol'].replace('.NS', '')}: ₹{pick['price']:.2f} | Target: ₹{pick['target']:.2f} | SL: ₹{pick['stop_loss']:.2f}")
            else:
                report.append("No strong BTST candidates identified today.")
            report.append("")
        
        # Add disclaimer
        report.append("⚠️ *Disclaimer:* These recommendations are based on technical analysis only. Always do your own research before investing. Past performance is not indicative of future results.")
        
        # Save recommendations to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"{'btst' if is_btst else 'daily'}_recommendations_{timestamp}.json"
        filepath = os.path.join(CONFIG['cache']['recommendations_dir'], filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'market_state': market_state,
                'recommendations': recommendations,
                'news': news
            }, f, indent=4)
            
        logger.info(f"Saved recommendations to {filepath}")
        
        # Generate telegram message
        telegram_message = "\n".join(report)
        if telegram.send_message(telegram_message):
            logger.info("Recommendations sent via Telegram")
        else:
            logger.warning("Failed to send recommendations via Telegram")
            
        return "\n".join(report)
    
    except Exception as e:
        error_msg = f"Error generating recommendations: {e}"
        logger.error(error_msg)
        telegram.send_message(f"❌ *Error in Market Analysis*\n\n{error_msg}")
        return error_msg

# Schedule regular tasks
def setup_scheduler():
    india_tz = pytz.timezone('Asia/Kolkata')
    
    # Daily evening summary at 8 PM IST
    schedule.every().day.at(CONFIG['workflow']['daily_run_time']).do(generate_recommendations, is_btst=False)
    
    # BTST recommendations at 2 PM IST
    schedule.every().day.at(CONFIG['workflow']['btst_time']).do(generate_recommendations, is_btst=True)
    
    # Database cleanup at midnight
    schedule.every().day.at("00:00").do(lambda: db_manager.execute("DELETE FROM stock_data WHERE timestamp < ?", (time.time() - 86400,)))
    
    logger.info("Scheduler set up with daily tasks")

# Command line interface
def run_cli():
    parser = argparse.ArgumentParser(description='Stock Market Analysis Bot')
    parser.add_argument('--daily', action='store_true', help='Generate daily recommendations')
    parser.add_argument('--btst', action='store_true', help='Generate BTST recommendations')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon with scheduled tasks')
    parser.add_argument('--check-packages', action='store_true', help='Check and install required packages')
    
    args = parser.parse_args()
    
    if args.check_packages:
        install_packages()
        return
    
    if args.daily:
        print(generate_recommendations(is_btst=False))
    elif args.btst:
        print(generate_recommendations(is_btst=True))
    elif args.daemon:
        setup_scheduler()
        logger.info("Starting daemon mode with scheduled tasks")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        # Interactive mode
        print("📊 Stock Market Analysis Bot")
        print("1. Generate daily recommendations")
        print("2. Generate BTST recommendations")
        print("3. Run as daemon with scheduled tasks")
        print("4. Check and install required packages")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            print(generate_recommendations(is_btst=False))
        elif choice == '2':
            print(generate_recommendations(is_btst=True))
        elif choice == '3':
            setup_scheduler()
            print("Running as daemon with scheduled tasks. Press Ctrl+C to exit.")
            try:
                while True:
                    schedule.run_pending()
                    time.sleep(60)
            except KeyboardInterrupt:
                print("Exiting daemon mode")
        elif choice == '4':
            install_packages()
        elif choice == '5':
            print("Exiting")
        else:
            print("Invalid choice")

# Main function
def main():
    try:
        # Ensure directories exist
        setup_directories()
        
        # Check if this is a first-time run
        first_run_marker = os.path.join(CONFIG['cache']['data_dir'], '.first_run')
        if not os.path.exists(first_run_marker):
            # Install required packages on first run
            install_packages()
            # Create first run marker
            with open(first_run_marker, 'w') as f:
                f.write(str(datetime.now()))
        
        # Run the CLI
        run_cli()
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        telegram.send_message(f"🚨 *Critical Error*\n\nThe trading bot has encountered a critical error: {e}")
    finally:
        # Clean up resources
        db_manager.close()

if __name__ == "__main__":
    import argparse
    main()
