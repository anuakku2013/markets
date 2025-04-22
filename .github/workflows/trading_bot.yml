```yaml
name: Trading Bot Workflow

on:
  schedule:
    - cron: '0 3 * * *' # Run daily at 03:00 UTC (08:30 IST)
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch: # Allow manual trigger

jobs:
  run-trading-bot:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'

      - name: Install TA-Lib C library
        run: |
          wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
          tar -xvzf ta-lib-0.4.0-src.tar.gz
          cd ta-lib
          ./configure --prefix=/usr
          make
          sudo make install
          cd ..

      - name: Install Python dependencies
        run: |
          pip install --upgrade pip
          pip install typing_extensions==4.12.2
          pip install pandas==2.0.3 numpy==1.24.4 matplotlib==3.7.5 scikit-learn==1.3.2 yfinance==0.2.44 alpha_vantage==2.3.1 TA-Lib==0.4.24 schedule==1.2.2 ccxt==4.3.0 python-dotenv==1.0.1 requests==2.32.3 tensorflow==2.15.0 torch==2.2.2 pytz==2024.2 plotly==5.24.1 ratelimit==2.2.1 dash==2.18.1

      - name: Run trading bot
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          python trading_bot.py

      - name: Upload logs
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: trading-bot-logs
          path: ./logs/trading_bot.log

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: trading-bot-reports
          path: ./recommendations/*.md
```
