#!/usr/bin/env python3
"""
Production runner for the Kalshi prediction market bot.

Usage:
    # Set environment variables
    export KALSHI_API_KEY="your-api-key"
    export KALSHI_PRIVATE_KEY_PATH="/path/to/private_key.pem"
    export KALSHI_ENVIRONMENT="demo"  # or "prod"
    
    # Run the bot
    python run_live.py
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Optional

import schedule

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kalshi_execution import (
    KalshiExecutor,
    EdgeBasedStrategy,
    KellyPositionSizer,
    KalshiAccount,
)
from kalshi_execution.exchange import LiveKalshiExchange, SimulatedKalshiExchange
from kalshi_execution.strategy import ExitStrategy

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kalshi_bot.log"),
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Production trading bot with lifecycle management."""
    
    def __init__(
        self,
        api_key: str,
        private_key_path: str,
        environment: str = "demo",
        initial_capital: float = 1000.0,
    ):
        self.api_key = api_key
        self.private_key_path = private_key_path
        self.environment = environment
        self.initial_capital = initial_capital
        
        # State
        self.running = False
        self.account: Optional[KalshiAccount] = None
        self.exchange: Optional[LiveKalshiExchange] = None
        self.executor: Optional[KalshiExecutor] = None
        self.strategy: Optional[EdgeBasedStrategy] = None
        
        # Metrics
        self.cycles_run = 0
        self.errors_count = 0
        self.last_run_time: Optional[datetime] = None
        
    def initialize(self):
        """Initialize all components."""
        logger.info(f"Initializing bot for {self.environment} environment")
        
        # Account
        self.account = KalshiAccount(initial_cash=self.initial_capital)
        
        # Exchange
        self.exchange = LiveKalshiExchange(
            api_key=self.api_key,
            private_key_path=self.private_key_path,
            environment=self.environment,
            rate_limit=5.0,  # Conservative rate limit
        )
        
        # Executor
        self.executor = KalshiExecutor(
            exchange=self.exchange,
            account=self.account,
            max_orders_per_step=5,
            retry_failed_orders=True,
            max_retries=3,
            verbose=True,
        )
        
        # Strategy - Conservative settings for live trading
        self.strategy = EdgeBasedStrategy(
            edge_threshold=0.10,  # Require 10% edge
            position_sizer=KellyPositionSizer(
                kelly_fraction=0.10,  # Only 10% of Kelly
                max_position_pct=0.03,  # Max 3% per position
                max_quantity=100,  # Max 100 contracts
            ),
            max_positions=10,
            risk_degree=0.3,  # Only use 30% of capital
        )
        
        # Exit strategy
        self.exit_strategy = ExitStrategy(
            profit_target=0.20,
            stop_loss=0.30,
            time_decay_exit_hours=2.0,
        )
        
        logger.info("Bot initialized successfully")
    
    def run_trading_cycle(self):
        """Execute one trading cycle."""
        try:
            cycle_start = datetime.now()
            logger.info(f"Starting trading cycle {self.cycles_run + 1}")
            
            # 1. Fetch current market data
            # In production, you'd call Kalshi API to get active markets
            signals = self._fetch_signals()
            
            if signals.empty:
                logger.info("No trading signals available")
                return
            
            # 2. Get current positions
            positions = self._get_positions_dict()
            
            # 3. Check for exits
            exit_orders = self._check_exit_signals(signals)
            if exit_orders:
                logger.info(f"Executing {len(exit_orders)} exit orders")
                # Execute exits...
            
            # 4. Generate new trades
            decision = self.strategy.generate_trade_decision(
                signals=signals,
                current_positions=positions,
                available_capital=self.account.available_cash,
                current_time=cycle_start,
            )
            
            # 5. Execute trades
            if len(decision) > 0:
                logger.info(f"Executing {len(decision)} new orders")
                results = self.executor.execute(decision, cycle_start)
                self._log_results(results)
            else:
                logger.info("No trades to execute")
            
            # 6. Log portfolio status
            self._log_portfolio_status()
            
            self.cycles_run += 1
            self.last_run_time = cycle_start
            
        except Exception as e:
            self.errors_count += 1
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            
            if self.errors_count >= 5:
                logger.critical("Too many errors, stopping bot")
                self.stop()
    
    def _fetch_signals(self):
        """Fetch signals from model/API."""
        # TODO: Implement actual signal fetching
        # This would:
        # 1. Call Kalshi API to get active markets
        # 2. Run your ML model to get probability estimates
        # 3. Return DataFrame with market_id, model_prob, market_price
        import pandas as pd
        return pd.DataFrame()  # Placeholder
    
    def _get_positions_dict(self):
        """Convert positions to dict format."""
        positions = {}
        for market_id, market_positions in self.account.get_all_positions().items():
            positions[market_id] = {}
            for contract_type, pos in market_positions.items():
                if pos.quantity > 0:
                    positions[market_id][contract_type.lower()] = pos.quantity
        return positions
    
    def _check_exit_signals(self, signals):
        """Check for positions that should be exited."""
        # TODO: Implement exit checking
        return []
    
    def _log_results(self, results):
        """Log execution results."""
        for result in results:
            if result["success"]:
                order = result["order"]
                logger.info(
                    f"FILLED: {order.direction.name} {result['filled_quantity']}x "
                    f"{order.market_id} @ ${result['fill_price']:.2f}"
                )
            else:
                logger.warning(f"FAILED: {result.get('error', 'Unknown error')}")
    
    def _log_portfolio_status(self):
        """Log current portfolio status."""
        summary = self.account.get_performance_summary()
        logger.info(
            f"Portfolio: Cash=${summary['current_cash']:.2f}, "
            f"Positions=${summary['position_value']:.2f}, "
            f"Total=${summary['total_value']:.2f}, "
            f"P&L=${summary['realized_pnl']:.2f}"
        )
    
    def start(self, interval_minutes: int = 15):
        """Start the bot with scheduled execution."""
        self.running = True
        
        logger.info(f"Starting bot with {interval_minutes} minute intervals")
        
        # Run immediately
        self.run_trading_cycle()
        
        # Schedule recurring runs
        schedule.every(interval_minutes).minutes.do(self.run_trading_cycle)
        
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self.running = False
        
        # Cancel any pending orders
        if self.executor:
            cancelled = self.executor.cancel_all_pending()
            logger.info(f"Cancelled {cancelled} pending orders")
        
        # Log final status
        if self.account:
            self._log_portfolio_status()
        
        logger.info("Bot stopped")


def main():
    # Load configuration from environment
    api_key = os.getenv("KALSHI_API_KEY")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    environment = os.getenv("KALSHI_ENVIRONMENT", "demo")
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "1000"))
    interval = int(os.getenv("TRADING_INTERVAL_MINUTES", "15"))
    
    if not api_key or not private_key_path:
        logger.error("Missing KALSHI_API_KEY or KALSHI_PRIVATE_KEY_PATH")
        sys.exit(1)
    
    # Create bot
    bot = TradingBot(
        api_key=api_key,
        private_key_path=private_key_path,
        environment=environment,
        initial_capital=initial_capital,
    )
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and start
    try:
        bot.initialize()
        bot.start(interval_minutes=interval)
    except NotImplementedError as e:
        logger.error(f"Live trading not yet implemented: {e}")
        logger.info("Use --mode backtest for testing, or implement LiveKalshiExchange")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
