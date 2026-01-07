FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY examples/prediction_market/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY examples/prediction_market/ ./

# Create non-root user
RUN useradd -m -u 1000 trader
USER trader

# Environment variables (override at runtime)
ENV KALSHI_ENVIRONMENT=demo
ENV LOG_LEVEL=INFO
ENV TRADING_INTERVAL_MINUTES=15
ENV INITIAL_CAPITAL=1000

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s \
    CMD python -c "import os; assert os.path.exists('kalshi_bot.log')" || exit 1

# Run the bot
CMD ["python", "run_live.py"]
