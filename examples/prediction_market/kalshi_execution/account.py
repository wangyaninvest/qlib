# Copyright (c) Prediction Market Bot Example
# Account and Position Management - Extends Qlib's account/position for prediction markets

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict

import pandas as pd
import numpy as np


@dataclass
class ContractPosition:
    """
    Position in a prediction market contract.
    
    Tracks holdings in YES/NO contracts for a specific market.
    """
    market_id: str
    contract_type: str  # "YES" or "NO"
    quantity: int
    avg_price: float
    
    # Position metadata
    entry_time: pd.Timestamp = None
    last_update: pd.Timestamp = None
    
    @property
    def cost_basis(self) -> float:
        """Total cost of position"""
        return self.quantity * self.avg_price
    
    def market_value(self, current_price: float) -> float:
        """Current market value"""
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized profit/loss"""
        return self.market_value(current_price) - self.cost_basis
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Unrealized P&L as percentage"""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.cost_basis
    
    def add(self, quantity: int, price: float, time: pd.Timestamp = None):
        """Add to position with average cost tracking"""
        total_cost = self.cost_basis + (quantity * price)
        self.quantity += quantity
        self.avg_price = total_cost / self.quantity if self.quantity > 0 else 0
        self.last_update = time or pd.Timestamp.now()
    
    def reduce(self, quantity: int, time: pd.Timestamp = None) -> float:
        """
        Reduce position.
        
        Returns
        -------
        float
            Cost basis of reduced shares (for P&L calculation)
        """
        if quantity > self.quantity:
            raise ValueError(f"Cannot reduce by {quantity}, only have {self.quantity}")
        
        cost_basis_sold = quantity * self.avg_price
        self.quantity -= quantity
        self.last_update = time or pd.Timestamp.now()
        
        if self.quantity == 0:
            self.avg_price = 0
        
        return cost_basis_sold
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "market_id": self.market_id,
            "contract_type": self.contract_type,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "cost_basis": self.cost_basis,
            "entry_time": self.entry_time,
        }


class KalshiAccount:
    """
    Trading account for prediction markets.
    
    Manages cash, positions, and P&L tracking.
    Modeled after Qlib's Account class.
    """
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        margin_requirement: float = 1.0,  # 100% margin for selling
    ):
        """
        Parameters
        ----------
        initial_cash : float
            Starting cash balance
        margin_requirement : float
            Margin required for short positions (1.0 = 100%)
        """
        self.initial_cash = initial_cash
        self.margin_requirement = margin_requirement
        
        # Account state
        self._cash = initial_cash
        self._positions: Dict[str, Dict[str, ContractPosition]] = defaultdict(dict)
        # Structure: {market_id: {"YES": ContractPosition, "NO": ContractPosition}}
        
        # P&L tracking
        self._realized_pnl = 0.0
        self._trade_history: List[Dict] = []
        
        # Settlement tracking
        self._settled_positions: List[Dict] = []
    
    @property
    def cash(self) -> float:
        """Current cash balance"""
        return self._cash
    
    @property
    def available_cash(self) -> float:
        """Cash available for new positions (accounting for margin)"""
        # In prediction markets, buying YES/NO requires upfront payment
        # Margin is only needed for selling (shorting)
        return self._cash
    
    @property
    def total_position_value(self) -> float:
        """Sum of all position cost bases"""
        total = 0.0
        for market_positions in self._positions.values():
            for position in market_positions.values():
                total += position.cost_basis
        return total
    
    @property
    def total_value(self) -> float:
        """Total account value (cash + positions at cost)"""
        return self._cash + self.total_position_value
    
    @property
    def realized_pnl(self) -> float:
        """Total realized P&L"""
        return self._realized_pnl
    
    def get_position(
        self,
        market_id: str,
        contract_type: str = None,
    ) -> Optional[ContractPosition]:
        """Get position for a market"""
        if market_id not in self._positions:
            return None
        
        if contract_type:
            return self._positions[market_id].get(contract_type)
        
        # Return first non-empty position
        for pos in self._positions[market_id].values():
            if pos.quantity > 0:
                return pos
        return None
    
    def get_all_positions(self) -> Dict[str, Dict[str, ContractPosition]]:
        """Get all positions"""
        return dict(self._positions)
    
    def get_positions_dataframe(self) -> pd.DataFrame:
        """Get positions as DataFrame"""
        records = []
        for market_id, positions in self._positions.items():
            for contract_type, position in positions.items():
                if position.quantity > 0:
                    records.append(position.to_dict())
        return pd.DataFrame(records)
    
    def add_position(
        self,
        market_id: str,
        contract_type: str,
        quantity: int,
        avg_price: float,
        time: pd.Timestamp = None,
    ):
        """Add or increase a position"""
        time = time or pd.Timestamp.now()
        
        if contract_type not in self._positions[market_id]:
            self._positions[market_id][contract_type] = ContractPosition(
                market_id=market_id,
                contract_type=contract_type,
                quantity=0,
                avg_price=0,
                entry_time=time,
            )
        
        position = self._positions[market_id][contract_type]
        position.add(quantity, avg_price, time)
        
        # Record trade
        self._trade_history.append({
            "time": time,
            "market_id": market_id,
            "contract_type": contract_type,
            "action": "BUY",
            "quantity": quantity,
            "price": avg_price,
            "value": quantity * avg_price,
        })
    
    def reduce_position(
        self,
        market_id: str,
        contract_type: str,
        quantity: int,
        sell_price: float,
        time: pd.Timestamp = None,
    ) -> float:
        """
        Reduce a position and realize P&L.
        
        Returns
        -------
        float
            Realized P&L from this sale
        """
        time = time or pd.Timestamp.now()
        
        if market_id not in self._positions:
            raise ValueError(f"No position in market {market_id}")
        if contract_type not in self._positions[market_id]:
            raise ValueError(f"No {contract_type} position in market {market_id}")
        
        position = self._positions[market_id][contract_type]
        cost_basis = position.reduce(quantity, time)
        
        # Calculate realized P&L
        proceeds = quantity * sell_price
        realized = proceeds - cost_basis
        self._realized_pnl += realized
        
        # Add proceeds to cash
        self._cash += proceeds
        
        # Record trade
        self._trade_history.append({
            "time": time,
            "market_id": market_id,
            "contract_type": contract_type,
            "action": "SELL",
            "quantity": quantity,
            "price": sell_price,
            "value": proceeds,
            "realized_pnl": realized,
        })
        
        return realized
    
    def deduct_cash(self, amount: float):
        """Deduct cash for purchases"""
        if amount > self._cash:
            raise ValueError(f"Insufficient cash: need {amount}, have {self._cash}")
        self._cash -= amount
    
    def add_cash(self, amount: float):
        """Add cash (deposits, settlements)"""
        self._cash += amount
    
    def settle_position(
        self,
        market_id: str,
        result: str,  # "yes" or "no"
        time: pd.Timestamp = None,
    ) -> float:
        """
        Settle a position when market resolves.
        
        Parameters
        ----------
        market_id : str
            Market that has settled
        result : str
            Settlement result ("yes" or "no")
        time : pd.Timestamp
            Settlement time
            
        Returns
        -------
        float
            Settlement proceeds
        """
        time = time or pd.Timestamp.now()
        
        if market_id not in self._positions:
            return 0.0
        
        total_settlement = 0.0
        
        for contract_type, position in self._positions[market_id].items():
            if position.quantity == 0:
                continue
            
            # Determine settlement value
            # YES contracts pay $1 if result is "yes", $0 otherwise
            # NO contracts pay $1 if result is "no", $0 otherwise
            if (contract_type == "YES" and result == "yes") or \
               (contract_type == "NO" and result == "no"):
                settlement_price = 1.0
            else:
                settlement_price = 0.0
            
            proceeds = position.quantity * settlement_price
            cost_basis = position.cost_basis
            realized = proceeds - cost_basis
            
            total_settlement += proceeds
            self._realized_pnl += realized
            self._cash += proceeds
            
            # Record settlement
            self._settled_positions.append({
                "time": time,
                "market_id": market_id,
                "contract_type": contract_type,
                "result": result,
                "quantity": position.quantity,
                "cost_basis": cost_basis,
                "settlement_proceeds": proceeds,
                "realized_pnl": realized,
            })
            
            # Clear position
            position.quantity = 0
            position.avg_price = 0
        
        # Remove market from positions
        del self._positions[market_id]
        
        return total_settlement
    
    def get_performance_summary(self) -> Dict:
        """Get account performance summary"""
        return {
            "initial_cash": self.initial_cash,
            "current_cash": self._cash,
            "position_value": self.total_position_value,
            "total_value": self.total_value,
            "realized_pnl": self._realized_pnl,
            "total_return": (self.total_value - self.initial_cash) / self.initial_cash,
            "num_positions": sum(
                1 for mkt in self._positions.values()
                for pos in mkt.values() if pos.quantity > 0
            ),
            "num_trades": len(self._trade_history),
            "num_settlements": len(self._settled_positions),
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        return pd.DataFrame(self._trade_history)
    
    def get_settlement_history(self) -> pd.DataFrame:
        """Get settlement history as DataFrame"""
        return pd.DataFrame(self._settled_positions)
    
    def calculate_unrealized_pnl(
        self,
        market_prices: Dict[str, float],
    ) -> float:
        """
        Calculate total unrealized P&L given current prices.
        
        Parameters
        ----------
        market_prices : Dict[str, float]
            Current YES prices by market_id
        """
        total_unrealized = 0.0
        
        for market_id, positions in self._positions.items():
            if market_id not in market_prices:
                continue
            
            yes_price = market_prices[market_id]
            
            for contract_type, position in positions.items():
                if position.quantity == 0:
                    continue
                
                current_price = yes_price if contract_type == "YES" else (1 - yes_price)
                total_unrealized += position.unrealized_pnl(current_price)
        
        return total_unrealized
    
    def reset(self):
        """Reset account to initial state"""
        self._cash = self.initial_cash
        self._positions = defaultdict(dict)
        self._realized_pnl = 0.0
        self._trade_history = []
        self._settled_positions = []
