# Copyright (c) Prediction Market Bot Example
# Full Trading Bot Example - Demonstrates the complete execution pipeline

"""
Prediction Market Trading Bot Example

This example demonstrates how to build a prediction market trading bot
on top of Qlib's infrastructure. The architecture follows Qlib's patterns:

    ┌─────────────────────────────────────────────────────────────────┐
    │                     EXECUTION PIPELINE                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  Model (Qlib)          Strategy              Executor           │
    │  ┌──────────┐        ┌───────────┐        ┌──────────────┐     │
    │  │ Predict  │──────▶│ Generate  │──────▶│   Execute    │      │
    │  │ Probs    │        │ Decisions │        │   Orders     │      │
    │  └──────────┘        └───────────┘        └──────────────┘     │
    │       │                    │                     │              │
    │       ▼                    ▼                     ▼              │
    │  ┌──────────┐        ┌───────────┐        ┌──────────────┐     │
    │  │ Signal   │        │ Position  │        │   Account    │      │
    │  │ DataFrame│        │ Sizing    │        │   Update     │      │
    │  └──────────┘        └───────────┘        └──────────────┘     │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    python example_bot.py --mode backtest
    python example_bot.py --mode paper
    python example_bot.py --mode live --api_key YOUR_KEY
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Import our execution layer components
from kalshi_execution import (
    KalshiExchange,
    KalshiExecutor,
    ContractOrder,
    ContractOrderDir,
    PredictionMarketDecision,
    EdgeBasedStrategy,
    KellyPositionSizer,
    KalshiAccount,
)
from kalshi_execution.exchange import SimulatedKalshiExchange, LiveKalshiExchange
from kalshi_execution.strategy import ExitStrategy, FixedFractionSizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SAMPLE DATA GENERATOR (for demonstration)
# =============================================================================

def generate_sample_market_data(
    num_markets: int = 20,
    num_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic prediction market data for backtesting.
    
    In production, this would be replaced with real Kalshi data.
    """
    np.random.seed(seed)
    
    records = []
    base_time = pd.Timestamp("2024-01-01")
    
    # Create markets with different characteristics
    markets = []
    for i in range(num_markets):
        # Each market has a "true" probability that prices evolve around
        true_prob = np.random.uniform(0.2, 0.8)
        volatility = np.random.uniform(0.02, 0.10)
        expiration = base_time + pd.Timedelta(days=np.random.randint(5, 60))
        
        markets.append({
            "market_id": f"MARKET-{i:03d}",
            "true_prob": true_prob,
            "volatility": volatility,
            "expiration": expiration,
        })
    
    # Generate time series for each market
    for market in markets:
        current_price = market["true_prob"] + np.random.normal(0, 0.05)
        current_price = np.clip(current_price, 0.05, 0.95)
        
        for day in range(num_days):
            timestamp = base_time + pd.Timedelta(days=day)
            
            if timestamp > market["expiration"]:
                # Market has expired - record final result
                result = "yes" if np.random.random() < market["true_prob"] else "no"
                records.append({
                    "timestamp": timestamp,
                    "market_id": market["market_id"],
                    "yes_bid": 1.0 if result == "yes" else 0.0,
                    "yes_ask": 1.0 if result == "yes" else 0.0,
                    "volume": 0,
                    "expiration": market["expiration"],
                    "result": result,
                })
            else:
                # Random walk with mean reversion to true probability
                drift = (market["true_prob"] - current_price) * 0.1
                noise = np.random.normal(0, market["volatility"])
                current_price = current_price + drift + noise
                current_price = np.clip(current_price, 0.05, 0.95)
                
                # Add spread
                spread = np.random.uniform(0.01, 0.03)
                yes_bid = current_price - spread/2
                yes_ask = current_price + spread/2
                
                records.append({
                    "timestamp": timestamp,
                    "market_id": market["market_id"],
                    "yes_bid": np.clip(yes_bid, 0.01, 0.99),
                    "yes_ask": np.clip(yes_ask, 0.01, 0.99),
                    "volume": np.random.randint(100, 10000),
                    "expiration": market["expiration"],
                    "result": None,
                })
    
    return pd.DataFrame(records)


def generate_model_signals(
    market_data: pd.DataFrame,
    timestamp: pd.Timestamp,
    noise_std: float = 0.05,
) -> pd.DataFrame:
    """
    Generate model predictions for current markets.
    
    In production, this would come from your trained Qlib model.
    The model would predict event probabilities based on features.
    """
    # Get current market state
    current = market_data[
        (market_data["timestamp"] == timestamp) &
        (market_data["result"].isna())
    ].copy()
    
    if current.empty:
        return pd.DataFrame()
    
    # Simulate model predictions (in reality, from trained model)
    # Add some noise to simulate model uncertainty
    signals = []
    for _, row in current.iterrows():
        market_price = (row["yes_bid"] + row["yes_ask"]) / 2
        
        # Simulated model prediction (would be real ML model in production)
        # Adding random noise to market price to simulate model
        model_prob = market_price + np.random.normal(0, noise_std)
        model_prob = np.clip(model_prob, 0.01, 0.99)
        
        signals.append({
            "market_id": row["market_id"],
            "model_prob": model_prob,
            "market_price": market_price,
            "expiration": row["expiration"],
        })
    
    return pd.DataFrame(signals)


# =============================================================================
# BACKTEST RUNNER
# =============================================================================

class PredictionMarketBacktest:
    """
    Backtest engine for prediction market strategies.
    
    Similar to Qlib's backtest module but adapted for binary contracts.
    """
    
    def __init__(
        self,
        strategy: EdgeBasedStrategy,
        initial_capital: float = 10000.0,
        verbose: bool = True,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.verbose = verbose
        
        # Results tracking
        self.equity_curve: List[Dict] = []
        self.trade_log: List[Dict] = []
    
    def run(
        self,
        market_data: pd.DataFrame,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Parameters
        ----------
        market_data : pd.DataFrame
            Historical market data
        start_date : pd.Timestamp
            Backtest start
        end_date : pd.Timestamp
            Backtest end
            
        Returns
        -------
        Dict
            Backtest results and metrics
        """
        # Initialize components
        account = KalshiAccount(initial_cash=self.initial_capital)
        exchange = SimulatedKalshiExchange(
            historical_data=market_data,
            slippage=0.01,
            fill_probability=0.9,
        )
        executor = KalshiExecutor(
            exchange=exchange,
            account=account,
            verbose=self.verbose,
        )
        
        # Set up exit strategy
        exit_strategy = ExitStrategy(
            profit_target=0.25,
            stop_loss=0.40,
            time_decay_exit_hours=4.0,
        )
        
        # Get trading dates
        dates = market_data["timestamp"].sort_values().unique()
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        
        logger.info(f"Running backtest from {dates[0]} to {dates[-1]}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Main backtest loop
        for current_date in dates:
            exchange.set_time(current_date)
            
            # 1. Process settlements
            self._process_settlements(account, exchange, current_date)
            
            # 2. Generate model signals
            signals = generate_model_signals(market_data, current_date)
            
            if signals.empty:
                continue
            
            # 3. Get current positions
            positions = self._get_position_dict(account)
            
            # 4. Check for exit signals
            exit_orders = self._check_exits(
                account, exchange, exit_strategy, signals, current_date
            )
            if exit_orders:
                exit_decision = PredictionMarketDecision(orders=exit_orders)
                executor.execute(exit_decision, current_date)
            
            # 5. Generate new trade decision
            decision = self.strategy.generate_trade_decision(
                signals=signals,
                current_positions=positions,
                available_capital=account.available_cash,
                current_time=current_date,
            )
            
            # 6. Execute decision
            if len(decision) > 0:
                results = executor.execute(decision, current_date)
                self.trade_log.extend(results)
            
            # 7. Record equity
            self.equity_curve.append({
                "date": current_date,
                "cash": account.cash,
                "position_value": account.total_position_value,
                "total_value": account.total_value,
                "realized_pnl": account.realized_pnl,
                "num_positions": len([
                    p for mkt in account.get_all_positions().values()
                    for p in mkt.values() if p.quantity > 0
                ]),
            })
        
        # Final settlements
        self._settle_all_remaining(account, exchange, dates[-1])
        
        # Calculate metrics
        return self._calculate_metrics(account)
    
    def _process_settlements(
        self,
        account: KalshiAccount,
        exchange: SimulatedKalshiExchange,
        current_date: pd.Timestamp,
    ):
        """Process any settled markets"""
        positions = account.get_all_positions()
        
        for market_id in list(positions.keys()):
            if market_id in exchange._settled_markets:
                result = exchange._settled_markets[market_id]
                proceeds = account.settle_position(market_id, result, current_date)
                if self.verbose:
                    logger.info(f"Settled {market_id}: {result}, proceeds: ${proceeds:.2f}")
    
    def _get_position_dict(self, account: KalshiAccount) -> Dict:
        """Convert positions to simple dict format"""
        positions = {}
        for market_id, market_positions in account.get_all_positions().items():
            positions[market_id] = {}
            for contract_type, pos in market_positions.items():
                if pos.quantity > 0:
                    positions[market_id][contract_type.lower()] = pos.quantity
        return positions
    
    def _check_exits(
        self,
        account: KalshiAccount,
        exchange: SimulatedKalshiExchange,
        exit_strategy: ExitStrategy,
        signals: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> List[ContractOrder]:
        """Check for positions that should be exited"""
        exit_orders = []
        positions = account.get_all_positions()
        
        # Build price and probability dicts
        market_quotes = {}
        model_probs = {}
        
        for _, row in signals.iterrows():
            market_id = row["market_id"]
            try:
                quote = exchange.get_quote(market_id)
                market_quotes[market_id] = quote
                model_probs[market_id] = row["model_prob"]
            except:
                continue
        
        # Check each position
        for market_id, market_positions in positions.items():
            for contract_type, position in market_positions.items():
                if position.quantity == 0:
                    continue
                if market_id not in market_quotes:
                    continue
                
                quote = market_quotes[market_id]
                model_prob = model_probs.get(market_id, quote.yes_mid)
                
                current_price = quote.yes_mid if contract_type == "YES" else quote.no_mid
                time_to_exp = (quote.expiration - current_date).total_seconds() / 3600
                
                pos_dict = {
                    "contract_type": contract_type,
                    "avg_price": position.avg_price,
                }
                
                should_exit, reason = exit_strategy.should_exit(
                    position=pos_dict,
                    current_price=current_price,
                    model_prob=model_prob,
                    time_to_expiration_hours=time_to_exp,
                )
                
                if should_exit:
                    direction = (ContractOrderDir.SELL_YES 
                               if contract_type == "YES" 
                               else ContractOrderDir.SELL_NO)
                    
                    exit_orders.append(ContractOrder(
                        market_id=market_id,
                        contract_type=contract_type,
                        direction=direction,
                        quantity=position.quantity,
                        limit_price=current_price,
                        start_time=current_date,
                        end_time=quote.expiration,
                    ))
                    
                    if self.verbose:
                        logger.info(f"Exit signal for {market_id} ({contract_type}): {reason}")
        
        return exit_orders
    
    def _settle_all_remaining(
        self,
        account: KalshiAccount,
        exchange: SimulatedKalshiExchange,
        final_date: pd.Timestamp,
    ):
        """Settle all remaining positions at backtest end"""
        positions = account.get_all_positions()
        
        for market_id in list(positions.keys()):
            if market_id in exchange._settled_markets:
                result = exchange._settled_markets[market_id]
            else:
                # Assume random outcome for unsettled markets
                result = "yes" if np.random.random() < 0.5 else "no"
            
            account.settle_position(market_id, result, final_date)
    
    def _calculate_metrics(self, account: KalshiAccount) -> Dict:
        """Calculate backtest performance metrics"""
        equity_df = pd.DataFrame(self.equity_curve)
        
        if equity_df.empty:
            return {"error": "No trades executed"}
        
        # Basic metrics
        final_value = account.total_value
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Daily returns
        equity_df["daily_return"] = equity_df["total_value"].pct_change()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (
            equity_df["daily_return"].mean() / equity_df["daily_return"].std()
            * np.sqrt(252)  # Annualized
            if equity_df["daily_return"].std() > 0 else 0
        )
        
        # Max drawdown
        equity_df["peak"] = equity_df["total_value"].cummax()
        equity_df["drawdown"] = (equity_df["total_value"] - equity_df["peak"]) / equity_df["peak"]
        max_drawdown = equity_df["drawdown"].min()
        
        # Win rate from settlements
        settlements = account.get_settlement_history()
        if not settlements.empty:
            win_rate = (settlements["realized_pnl"] > 0).mean()
            avg_win = settlements[settlements["realized_pnl"] > 0]["realized_pnl"].mean()
            avg_loss = settlements[settlements["realized_pnl"] < 0]["realized_pnl"].mean()
        else:
            win_rate = avg_win = avg_loss = 0
        
        results = {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": f"{total_return * 100:.2f}%",
            "realized_pnl": account.realized_pnl,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": f"{max_drawdown * 100:.2f}%",
            "num_trades": len(account.get_trade_history()),
            "num_settlements": len(settlements),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "equity_curve": equity_df,
        }
        
        return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_backtest_example():
    """Run a sample backtest"""
    logger.info("=" * 60)
    logger.info("Prediction Market Trading Bot - Backtest Example")
    logger.info("=" * 60)
    
    # Generate sample data
    logger.info("Generating sample market data...")
    market_data = generate_sample_market_data(
        num_markets=30,
        num_days=60,
        seed=42,
    )
    
    # Create strategy
    strategy = EdgeBasedStrategy(
        edge_threshold=0.08,  # 8% edge required
        position_sizer=KellyPositionSizer(
            kelly_fraction=0.15,  # 15% of full Kelly
            max_position_pct=0.05,  # Max 5% per position
        ),
        max_positions=15,
        risk_degree=0.6,  # Use 60% of capital
    )
    
    # Run backtest
    backtest = PredictionMarketBacktest(
        strategy=strategy,
        initial_capital=10000.0,
        verbose=True,
    )
    
    results = backtest.run(market_data)
    
    # Print results
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Initial Capital:  ${results['initial_capital']:,.2f}")
    logger.info(f"Final Value:      ${results['final_value']:,.2f}")
    logger.info(f"Total Return:     {results['total_return_pct']}")
    logger.info(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown:     {results['max_drawdown_pct']}")
    logger.info(f"Win Rate:         {results['win_rate']*100:.1f}%")
    logger.info(f"Number of Trades: {results['num_trades']}")
    logger.info("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Prediction Market Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Trading mode",
    )
    parser.add_argument("--api_key", help="Kalshi API key (for live trading)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    
    args = parser.parse_args()
    
    if args.mode == "backtest":
        run_backtest_example()
    elif args.mode == "paper":
        logger.info("Paper trading mode - requires Kalshi demo API")
        logger.info("Set up credentials and use LiveKalshiExchange with environment='demo'")
    elif args.mode == "live":
        logger.info("Live trading mode - USE WITH CAUTION")
        logger.info("Ensure you have proper risk management in place")
        if not args.api_key:
            logger.error("API key required for live trading")
    

if __name__ == "__main__":
    main()
