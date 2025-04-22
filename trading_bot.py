import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
import logging
import yfinance as yf
import json
from datetime import datetime, time as dt_time
import requests
import pytz
import schedule
import threading
import sqlite3
from bs4 import BeautifulSoup
from queue import Queue
from ratelimit import limits, sleep_and_retry

# List of required packages with pinned versions
REQUIRED_PACKAGES = [
    'pandas==2.0.3', 'numpy==1.24.4', 'matplotlib==3.7.5', 'scikit-learn==1.3.2',
    'yfinance==0.2.44', 'alpha_vantage==2.3.1', 'schedule==1.2.2', 'ccxt==4.3.0',
    'python-dotenv==1.0.1', 'requests==2.32.3', 'tensorflow==2.15.0', 'torch==2.2.2',
    'pytz==2024.2', 'plotly==5.24.1', 'ratelimit==2.2.1', 'typing_extensions==4.12.2',
    'dash==2.18.1', 'TA-Lib==0.4.24'
]

# Install required packages
def install_packages():
    logger.info("Checking and installing required packages...")
    for package in REQUIRED_PACKAGES:
        package_name = package.split('==')[0]
        module_name = package_name.replace('-', '_')
        try:
            __import__(module_name)
            logger.info(f"{package_name} already installed.")
        except ImportError:
            logger.info(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                if package_name == 'TA-Lib':
                    logger.info("Attempting to install TA-Lib prebuilt wheel...")
                    try:
                        wheel_url = "https://github.com/TA-Lib/ta-lib-python/releases/download/v0.4.24/ta_lib-0.4.24-cp38-cp38-win_amd64.whl"
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', wheel_url])
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to install TA-Lib wheel: {e}")
                        sys.exit(1)
                else:
                    sys.exit(1)
    logger.info("All packages installed successfully.")

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
        'calls_per_minute': 60
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
        'daily_run_time': '20:00',  # Changed to 8 PM IST for full summary
        'btst_time': '14:00'  # 2 PM IST for BTST recommendations
    }
}

# Setup directories
os.makedirs(CONFIG['cache']['data_dir'], exist_ok=True)
os.makedirs(CONFIG['cache']['recommendations_dir'], exist_ok=True)
os.makedirs(CONFIG['cache']['logs_dir'], exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG['cache']['logs_dir'], 'trading_bot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Telegram setup
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
telegram_queue = Queue()

# Derived stock lists
ALL_STOCKS = list(set(CONFIG['stocks']['indices'] + CONFIG['stocks']['analysis']))

# SQLite cache initialization
def init_cache_db():
    conn = sqlite3.connect(os.path.join(CONFIG['cache']['data_dir'], 'cache.db'))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stock_data
                 (symbol TEXT, interval TEXT, period TEXT, timestamp REAL, data TEXT,
                  PRIMARY KEY (symbol, interval, period))''')
    c.execute('''CREATE TABLE IF NOT EXISTS news_cache
                 (timestamp REAL, data TEXT, PRIMARY KEY (timestamp))''')
    conn.commit()
    return conn

# Cache validation
def is_cache_valid(symbol, interval):
    conn = init_cache_db()
    c = conn.cursor()
    c.execute('SELECT timestamp, data FROM stock_data WHERE symbol=? AND interval=?',
              (symbol, interval))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return False
    
    timestamp, data = result
    now = time.time()
    expiry_seconds = (CONFIG['cache']['intraday_expiry_minutes'] * 60 if interval in ['5m', '15m', '30m', '60m', '1h']
                     else CONFIG['cache']['expiry_hours'] * 3600)
    
    if (now - timestamp) > expiry_seconds:
        return False
    
    try:
        df = pd.read_json(data)
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns or df[col].isna().all():
                return False
        return not df.empty and len(df) >= 5
    except Exception as e:
        logger.warning(f"Corrupted cache for {symbol}: {e}")
        return False

# Batch stock data fetching
@sleep_and_retry
@limits(calls=CONFIG['api']['calls_per_minute'], period=60)
def fetch_batch_stock_data(symbols, interval='1d', period='1mo'):
    try:
        df = yf.download(symbols, period=period, interval=interval, group_by='ticker', progress=False)
        data = {}
        for symbol in symbols:
            if symbol in df.columns.levels[0]:
                symbol_data = df[symbol].dropna()
                if not symbol_data.empty and len(symbol_data) >= 5:
                    conn = init_cache_db()
                    c = conn.cursor()
                    c.execute('INSERT OR REPLACE INTO stock_data VALUES (?, ?, ?, ?, ?)',
                              (symbol, interval, period, time.time(), symbol_data.to_json()))
                    conn.commit()
                    conn.close()
                    data[symbol] = symbol_data
                else:
                    data[symbol] = None
            else:
                data[symbol] = None
        logger.info(f"Fetched batch data for {len(symbols)} symbols")
        return data
    except Exception as e:
        logger.error(f"Error fetching batch data: {e}")
        return {symbol: None for symbol in symbols}

# Calculate technical indicators
def calculate_indicators(df):
    if df is None or len(df) < 14:
        return None
    
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
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

# Analyze stock
def analyze_stock(symbol, data, is_btst=False):
    try:
        min_data_points = 30
        if data is None or len(data) < min_data_points:
            return None
        
        data_with_indicators = calculate_indicators(data)
        if data_with_indicators is None:
            return None
        
        latest = data_with_indicators.iloc[-1]
        prev = data_with_indicators.iloc[-2]
        latest_close = latest['Close']
        
        buy_signals = []
        sell_signals = []
        score = 0
        
        if not np.isnan(latest['RSI']):
            if latest['RSI'] < 30:
                buy_signals.append(f"RSI oversold ({latest['RSI']:.1f})")
                score += 3
            elif latest['RSI'] > 70:
                sell_signals.append(f"RSI overbought ({latest['RSI']:.1f})")
                score -= 3
        
        if not np.isnan(latest['SMA20']) and not np.isnan(latest['SMA50']):
            if latest_close > latest['SMA20'] and latest_close > latest['SMA50']:
                buy_signals.append("Price above SMA20 and SMA50")
                score += 1
            elif latest_close < latest['SMA20'] and latest_close < latest['SMA50']:
                sell_signals.append("Price below SMA20 and SMA50")
                score -= 1
        
        # BTST-specific logic
        if is_btst and score >= 5:
            buy_signals.append("BTST candidate")
            score += 2
        
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

# Market state check
def check_market_state():
    india_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india_tz)
    
    if now.weekday() >= 5:
        return 'closed', 'Weekend'
    
    market_open = datetime.combine(now.date(), dt_time(9, 15)).replace(tzinfo=india_tz)
    market_close = datetime.combine(now.date(), dt_time(15, 30)).replace(tzinfo=india_tz)
    
    if market_open <= now < market_close:
        return 'open', 'Market open'
    return 'closed', 'Outside trading hours'

# News fetching
def get_market_news(max_articles=5):
    cache_file = os.path.join(CONFIG['cache']['data_dir'], 'news_cache.json')
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 3600:
        with open(cache_file, 'r') as f:
            return json.load(f)[:max_articles]
    
    all_news = []
    for source in CONFIG['news']:
        try:
            response = requests.get(source['url'], headers={
                'User-Agent': 'Mozilla/5.0'
            }, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if source['name'] == 'MoneyControl':
                articles = soup.select('.article_box')[:max_articles]
                for article in articles:
                    title_elem = article.select_one('h3 a')
                    if title_elem:
                        title = title_elem.text.strip()
                        link = title_elem.get('href', '')
                        all_news.append({
                            'source': source['name'],
                            'title': title,
                            'link': link if link.startswith('http') else f"https://www.moneycontrol.com{link}"
                        })
        except Exception as e:
            logger.error(f"Error fetching news from {source['name']}: {e}")
    
    with open(cache_file, 'w') as f:
        json.dump(all_news, f)
    return all_news[:max_articles]

# Generate technical signals
def generate_technical_signals(stocks_list, interval='1d', is_btst=False):
    signals = []
    batch_size = 10
    for i in range(0, len(stocks_list), batch_size):
        batch = stocks_list[i:i + batch_size]
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
    
    signals.sort(key=lambda x: abs(x['score']), reverse=True)
    return signals

# Generate recommendations based on time
def generate_recommendations(is_btst=False):
    try:
        market_state, market_message = check_market_state()
        news = get_market_news(max_articles=5)
        signals = generate_technical_signals(CONFIG['stocks']['analysis'], is_btst=is_btst)
        
        report = []
        today_date = datetime.now().strftime('%d %b %Y')
        greeting = "🌞 Good Morning!" if is_btst or market_state == 'open' else "🌙 Good Evening!"
        report.append(f"{greeting} 📊 *Daily Market Insights - {today_date}*")
        report.append(f"📈 *Market Status:* {market_state.upper()} - {market_message}")
        report.append("")
        
        recommendations = {'BUY': [], 'SELL': [], 'Changed': []}
        for signal in signals:
            score = signal['score']
            price = signal['close']
            target_price = price * 1.10 if score >= 8 else price * 1.05 if score >= 5 else price * 0.95
            price_display = f"₹{price:.2f}" if isinstance(price, (int, float)) else str(price)
            target_display = f"₹{target_price:.2f}" if isinstance(target_price, (int, float)) else 'N/A'
            reasons = signal['buy_signals'] if score > 0 else signal['sell_signals']
            reason_text = ', '.join(reasons) if reasons else 'Based on technical analysis'
            
            rec = {
                'symbol': signal['symbol'].replace('.NS', ''),
                'price': price_display,
                'target': target_display,
                'reason': reason_text,
                'change': signal['change'] * 100
            }
            
            if score >= 5:
                recommendations['BUY'].append(rec)
            elif score <= 2:
                recommendations['SELL'].append(rec)
            if abs(signal['change']) > 0.01:
                recommendations['Changed'].append(rec)
        
        if is_btst:
            if recommendations['BUY']:
                report.append("🟢 *BTST Recommendations*")
                for rec in recommendations['BUY'][:3]:
                    report.append(f"*{rec['symbol']}*: Buy at {rec['price']}, Target: {rec['target']}")
                    report.append(f"Reason: {rec['reason']} 🚀")
        elif market_state == 'open':
            if recommendations['BUY']:
                report.append("🟢 *BUY Recommendations*")
                for rec in recommendations['BUY'][:5]:
                    report.append(f"*{rec['symbol']}*: Buy at {rec['price']}, Target: {rec['target']}")
                    report.append(f"Reason: {rec['reason']} 🚀")
        else:
            report.append("📝 *Market Summary*")
            for category in ['BUY', 'SELL', 'Changed']:
                if recommendations[category]:
                    report.append(f"*{category.upper()}*")
                    for rec in recommendations[category][:3]:
                        report.append(f"*{rec['symbol']}*: {rec['price']} ({rec['change']:.2f}%)")
                        report.append(f"Target: {rec['target']}, Reason: {rec['reason']} {'🚀' if category == 'BUY' else '🔻'}")
        
        if news:
            report.append("📰 *Latest Market News*")
            for article in news:
                report.append(f"*{article['source']}:* [{article['title']}]({article['link']})")
        
        report.append("⚠️ *Disclaimer:* Not financial advice. 🙏 Trade wisely!")
        report_text = "\n".join(report)
        
        report_file = os.path.join(CONFIG['cache']['recommendations_dir'],
                                  f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.md")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        if TELEGRAM_ENABLED:
            send_telegram_message(report_text)
        
        return {'report': report_text, 'signals': signals}
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return None

# Telegram message sending
def send_telegram_message(message, max_retries=3):
    if not TELEGRAM_ENABLED:
        logger.warning(f"Telegram disabled. Message: {message[:50]}...")
        return False
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'},
                timeout=30
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(2 * (attempt + 1))
    
    logger.error("All attempts failed, queuing message")
    telegram_queue.put(message)
    return False

# Schedule tasks
def schedule_tasks():
    schedule.every().day.at(CONFIG['workflow']['btst_time']).do(lambda: generate_recommendations(is_btst=True))
    schedule.every().day.at(CONFIG['workflow']['daily_run_time']).do(generate_recommendations)
    logger.info(f"Scheduled BTST at {CONFIG['workflow']['btst_time']} and daily run at {CONFIG['workflow']['daily_run_time']} IST")

# Run scheduler
def run_scheduler():
    try:
        logger.info("Starting scheduler loop")
        while True:
            schedule.run_pending()
            time.sleep(60)
    except Exception as e:
        logger.error(f"Scheduler loop crashed: {e}")
        raise

# Workflow runner
def workflow_runner():
    logger.info("Starting workflow runner")
    install_packages()  # Install packages at runtime
    schedule_tasks()
    
    # Run immediately in GitHub Actions or manual trigger
    if os.environ.get('GITHUB_ACTIONS') or datetime.now().strftime('%H:%M') == CONFIG['workflow']['daily_run_time']:
        result = generate_recommendations()
        if result and isinstance(result, dict) and 'signals' in result:
            logger.info("Generated initial recommendations")
        else:
            logger.error("No valid signals for initial recommendations")
    
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    logger.info("Workflow runner started. Scheduler running in background.")
    return scheduler_thread

# Main function
def main():
    print("=" * 50)
    print("TRADING BOT STARTING")
    print("=" * 50)
    logger.info("Trading bot starting")
    
    # Start workflow runner
    scheduler_thread = workflow_runner()
    
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down trading bot...")
        time.sleep(2)
        print("Trading bot stopped.")
        sys.exit(0)

if __name__ == '__main__':
    main()
