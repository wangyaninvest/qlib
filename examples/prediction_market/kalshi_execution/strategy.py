# Copyright (c) Prediction Market Bot Example
# Strategy Layer - Extends Qlib's BaseStrategy for prediction markets

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from .decision import ContractOrder, ContractOrderDir, PredictionMarketDecision
from .exchange import BaseKalshiExchange, MarketQuote


class BaseMarketStrategy:
    """
    Base strategy for prediction market trading.
    
    Modeled after Qlib's BaseStrategy but adapted for binary contracts.
    """
    
    def __init__(
        self,
        exchange: BaseKalshiExchange = None,
        risk_degree: float = 0.5,  # Max fraction of capital to deploy
    ):
        """
        Parameters
        ----------
        exchange : BaseKalshiExchange
            Exchange for market data
        risk_degree : float
            Maximum fraction of capital to use [0, 1]
        """
        self.exchange = exchange
        self.risk_degree = risk_degree
    
    @abstractmethod
    def generate_trade_decision(
        self,
        signals: pd.DataFrame,
        current_positions: Dict,
        available_capital: float,
        current_time: pd.Timestamp,
    ) -> PredictionMarketDecision:
        """
        Generate trade decision based on signals.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Model predictions with columns: market_id, model_prob, market_price
        current_positions : Dict
            Current portfolio positions
        available_capital : float
            Cash available for trading
        current_time : pd.Timestamp
            Current time
            
        Returns
        -------
        PredictionMarketDecision
        """
        raise NotImplementedError


class EdgeBasedStrategy(BaseMarketStrategy):
    """
    Strategy that trades when model probability differs from market price.
    
    The "edge" is the difference between model probability and market price.
    Trades are generated when edge exceeds a threshold.
    """
    
    def __init__(
        self,
        edge_threshold: float = 0.05,
        position_sizer: "BasePositionSizer" = None,
        max_positions: int = 20,
        min_edge_decay: float = 0.01,  # Edge decays as expiration approaches
        **kwargs,
    ):
        """
        Parameters
        ----------
        edge_threshold : float
            Minimum edge to generate a trade signal
        position_sizer : BasePositionSizer
            Position sizing strategy (default: Kelly)
        max_positions : int
            Maximum number of positions to hold
        min_edge_decay : float
            Reduce edge threshold as expiration approaches
        """
        super().__init__(**kwargs)
        self.edge_threshold = edge_threshold
        self.position_sizer = position_sizer or KellyPositionSizer()
        self.max_positions = max_positions
        self.min_edge_decay = min_edge_decay
    
    def generate_trade_decision(
        self,
        signals: pd.DataFrame,
        current_positions: Dict,
        available_capital: float,
        current_time: pd.Timestamp,
    ) -> PredictionMarketDecision:
        """Generate decision based on edge"""
        
        decision = PredictionMarketDecision(
            strategy=self,
            created_time=current_time,
            edge_threshold_used=self.edge_threshold,
        )
        
        # Track capital allocation
        remaining_capital = available_capital * self.risk_degree
        num_positions = len(current_positions)
        
        # Score opportunities by edge
        opportunities = []
        
        for _, row in signals.iterrows():
            market_id = row["market_id"]
            model_prob = row["model_prob"]
            market_price = row["market_price"]
            expiration = row.get("expiration")
            
            # Calculate edges
            edge_yes = model_prob - market_price
            edge_no = (1 - model_prob) - (1 - market_price)
            
            # Adjust threshold based on time to expiration
            threshold = self._adjusted_threshold(current_time, expiration)
            
            if abs(edge_yes) > threshold:
                opportunities.append({
                    "market_id": market_id,
                    "contract_type": "YES" if edge_yes > 0 else "NO",
                    "edge": abs(edge_yes) if edge_yes > 0 else abs(edge_no),
                    "model_prob": model_prob if edge_yes > 0 else 1 - model_prob,
                    "market_price": market_price if edge_yes > 0 else 1 - market_price,
                    "direction": ContractOrderDir.BUY_YES if edge_yes > 0 else ContractOrderDir.BUY_NO,
                    "expiration": expiration,
                })
        
        # Sort by edge (best opportunities first)
        opportunities.sort(key=lambda x: x["edge"], reverse=True)
        
        # Generate orders
        for opp in opportunities:
            if num_positions >= self.max_positions:
                break
            if remaining_capital < 1.0:  # Minimum $1 order
                break
            
            # Skip if already have position in this market
            if opp["market_id"] in current_positions:
                continue
            
            # Size the position
            quantity, order_value = self.position_sizer.size_position(
                edge=opp["edge"],
                price=opp["market_price"],
                available_capital=remaining_capital,
                model_prob=opp["model_prob"],
            )
            
            if quantity > 0:
                order = ContractOrder(
                    market_id=opp["market_id"],
                    contract_type=opp["contract_type"],
                    direction=opp["direction"],
                    quantity=quantity,
                    limit_price=opp["market_price"],
                    start_time=current_time,
                    end_time=opp["expiration"] or current_time + pd.Timedelta(hours=24),
                )
                decision.add_order(order)
                remaining_capital -= order_value
                num_positions += 1
        
        return decision
    
    def _adjusted_threshold(
        self,
        current_time: pd.Timestamp,
        expiration: Optional[pd.Timestamp],
    ) -> float:
        """Adjust edge threshold based on time to expiration"""
        if expiration is None:
            return self.edge_threshold
        
        time_to_exp = (expiration - current_time).total_seconds() / 3600  # hours
        
        if time_to_exp < 1:
            # Very close to expiration - require higher edge (more uncertainty)
            return self.edge_threshold * 2
        elif time_to_exp < 24:
            # Within a day - slightly higher threshold
            return self.edge_threshold * 1.5
        else:
            return self.edge_threshold


class BasePositionSizer:
    """Base class for position sizing strategies"""
    
    @abstractmethod
    def size_position(
        self,
        edge: float,
        price: float,
        available_capital: float,
        model_prob: float = None,
    ) -> Tuple[int, float]:
        """
        Determine position size.
        
        Returns
        -------
        Tuple[int, float]
            (quantity, total_value)
        """
        raise NotImplementedError


class KellyPositionSizer(BasePositionSizer):
    """
    Position sizing using Kelly Criterion.
    
    Kelly formula for binary bets:
    f* = (p * b - q) / b
    where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = odds received on the bet
        
    For binary contracts:
        b = (1 - price) / price  (payout if win / cost)
        p = model_prob (our estimated probability)
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,  # Fractional Kelly for risk reduction
        max_position_pct: float = 0.1,  # Max 10% of capital per position
        min_quantity: int = 1,
        max_quantity: int = 1000,
    ):
        """
        Parameters
        ----------
        kelly_fraction : float
            Fraction of full Kelly to use (0.25 = quarter Kelly)
        max_position_pct : float
            Maximum position as fraction of capital
        min_quantity : int
            Minimum contracts per order
        max_quantity : int
            Maximum contracts per order
        """
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
    
    def size_position(
        self,
        edge: float,
        price: float,
        available_capital: float,
        model_prob: float = None,
    ) -> Tuple[int, float]:
        """Calculate position size using Kelly criterion"""
        
        if price <= 0 or price >= 1:
            return 0, 0.0
        
        # Calculate Kelly fraction
        if model_prob is None:
            model_prob = price + edge
        
        odds = (1 - price) / price
        q = 1 - model_prob
        
        # Kelly formula
        kelly_full = (model_prob * odds - q) / odds if odds > 0 else 0
        kelly_full = max(0, kelly_full)  # Can't be negative
        
        # Apply fractional Kelly
        kelly_adj = kelly_full * self.kelly_fraction
        
        # Apply max position cap
        position_pct = min(kelly_adj, self.max_position_pct)
        
        # Calculate dollar amount
        position_value = available_capital * position_pct
        
        # Convert to contracts
        quantity = int(position_value / price)
        quantity = max(self.min_quantity, min(quantity, self.max_quantity))
        
        # Recalculate actual value
        actual_value = quantity * price
        
        if actual_value > available_capital:
            quantity = int(available_capital / price)
            actual_value = quantity * price
        
        return quantity, actual_value


class FixedFractionSizer(BasePositionSizer):
    """Simple fixed fraction position sizing"""
    
    def __init__(
        self,
        fraction: float = 0.02,  # 2% of capital per position
        min_quantity: int = 1,
        max_quantity: int = 500,
    ):
        self.fraction = fraction
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
    
    def size_position(
        self,
        edge: float,
        price: float,
        available_capital: float,
        model_prob: float = None,
    ) -> Tuple[int, float]:
        """Fixed fraction sizing"""
        
        if price <= 0 or price >= 1:
            return 0, 0.0
        
        position_value = available_capital * self.fraction
        quantity = int(position_value / price)
        quantity = max(self.min_quantity, min(quantity, self.max_quantity))
        actual_value = quantity * price
        
        return quantity, actual_value


class ExitStrategy:
    """
    Strategy for exiting positions.
    
    Manages when to sell positions before expiration.
    """
    
    def __init__(
        self,
        profit_target: float = 0.30,  # Exit at 30% profit
        stop_loss: float = 0.50,      # Exit at 50% loss
        edge_reversal_exit: bool = True,  # Exit if edge reverses
        time_decay_exit_hours: float = 2.0,  # Exit N hours before expiration
    ):
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.edge_reversal_exit = edge_reversal_exit
        self.time_decay_exit_hours = time_decay_exit_hours
    
    def should_exit(
        self,
        position: Dict,
        current_price: float,
        model_prob: float,
        time_to_expiration_hours: float,
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be exited.
        
        Returns
        -------
        Tuple[bool, str]
            (should_exit, reason)
        """
        entry_price = position["avg_price"]
        contract_type = position["contract_type"]
        
        # Calculate P&L
        if contract_type == "YES":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = ((1 - current_price) - entry_price) / entry_price
        
        # Check profit target
        if pnl_pct >= self.profit_target:
            return True, "profit_target"
        
        # Check stop loss
        if pnl_pct <= -self.stop_loss:
            return True, "stop_loss"
        
        # Check time decay
        if time_to_expiration_hours < self.time_decay_exit_hours:
            return True, "time_decay"
        
        # Check edge reversal
        if self.edge_reversal_exit:
            if contract_type == "YES" and model_prob < current_price:
                return True, "edge_reversal"
            elif contract_type == "NO" and model_prob > current_price:
                return True, "edge_reversal"
        
        return False, ""
    
    def generate_exit_orders(
        self,
        positions: Dict,
        market_quotes: Dict[str, MarketQuote],
        model_probs: Dict[str, float],
        current_time: pd.Timestamp,
    ) -> List[ContractOrder]:
        """Generate exit orders for positions that should be closed"""
        
        exit_orders = []
        
        for market_id, position in positions.items():
            quote = market_quotes.get(market_id)
            model_prob = model_probs.get(market_id)
            
            if quote is None or model_prob is None:
                continue
            
            current_price = quote.yes_mid
            time_to_exp = (quote.expiration - current_time).total_seconds() / 3600
            
            should_exit, reason = self.should_exit(
                position=position,
                current_price=current_price,
                model_prob=model_prob,
                time_to_expiration_hours=time_to_exp,
            )
            
            if should_exit:
                # Create sell order
                contract_type = position["contract_type"]
                direction = (ContractOrderDir.SELL_YES 
                           if contract_type == "YES" 
                           else ContractOrderDir.SELL_NO)
                
                # Use bid price for selling
                limit_price = quote.yes_bid if contract_type == "YES" else quote.no_bid
                
                order = ContractOrder(
                    market_id=market_id,
                    contract_type=contract_type,
                    direction=direction,
                    quantity=position["quantity"],
                    limit_price=limit_price,
                    start_time=current_time,
                    end_time=quote.expiration,
                )
                exit_orders.append(order)
        
        return exit_orders
