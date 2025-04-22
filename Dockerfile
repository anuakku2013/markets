FROM python:3.10-slim

# Install cron and other dependencies
RUN apt-get update && apt-get install -y cron && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trading bot script
COPY . .

# Make sure scripts are executable
RUN chmod +x /app/tradingbot.py

# Create empty log file
RUN touch /app/logs/cron.log

CMD ["bash"]
