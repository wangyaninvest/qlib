# Copyright (c) Prediction Market Bot Example
# Executor Layer - Extends Qlib's BaseExecutor for prediction markets

from __future__ import annotations

import copy
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from .decision import ContractOrder, PredictionMarketDecision, DecisionGenerator
from .exchange import BaseKalshiExchange, MarketQuote
from .position import ContractPosition
from .account import KalshiAccount

import logging

logger = logging.getLogger(__name__)


class KalshiExecutor:
    """
    Executor for prediction market trading.
    
    Modeled after Qlib's BaseExecutor but adapted for binary contracts.
    Handles order execution, position updates, and P&L tracking.
    """
    
    def __init__(
        self,
        exchange: BaseKalshiExchange,
        account: KalshiAccount,
        time_per_step: str = "1H",  # Execution frequency
        max_orders_per_step: int = 10,
        retry_failed_orders: bool = True,
        max_retries: int = 3,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        exchange : BaseKalshiExchange
            The exchange to execute orders on
        account : KalshiAccount
            Account for position and capital management
        time_per_step : str
            Trading frequency (e.g., "1H", "30min", "1D")
        max_orders_per_step : int
            Maximum orders to submit per execution step
        retry_failed_orders : bool
            Whether to retry failed orders
        max_retries : int
            Maximum retry attempts
        verbose : bool
            Print detailed execution info
        """
        self.exchange = exchange
        self.account = account
        self.time_per_step = time_per_step
        self.max_orders_per_step = max_orders_per_step
        self.retry_failed_orders = retry_failed_orders
        self.max_retries = max_retries
        self.verbose = verbose
        
        # Execution tracking
        self._current_step = 0
        self._execution_history: List[Dict] = []
        self._pending_orders: Dict[str, ContractOrder] = {}
    
    def reset(
        self,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
    ):
        """Reset executor state"""
        self._current_step = 0
        self._execution_history = []
        self._pending_orders = {}
        
        if hasattr(self.exchange, 'set_time') and start_time:
            self.exchange.set_time(start_time)
    
    def execute(
        self,
        trade_decision: PredictionMarketDecision,
        current_time: pd.Timestamp = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a trade decision.
        
        Parameters
        ----------
        trade_decision : PredictionMarketDecision
            The decision containing orders to execute
        current_time : pd.Timestamp
            Current simulation/execution time
            
        Returns
        -------
        List[Dict]
            Execution results for each order
        """
        if current_time and hasattr(self.exchange, 'set_time'):
            self.exchange.set_time(current_time)
        
        results = []
        orders = trade_decision.get_order_list()
        
        # Limit orders per step
        orders = orders[:self.max_orders_per_step]
        
        for order in orders:
            result = self._execute_single_order(order)
            results.append(result)
            
            if self.verbose:
                self._log_execution(order, result)
        
        # Update account positions
        self._update_positions(results)
        
        # Record execution
        self._execution_history.append({
            "step": self._current_step,
            "time": current_time,
            "num_orders": len(orders),
            "results": results,
        })
        
        self._current_step += 1
        
        return results
    
    def _execute_single_order(
        self,
        order: ContractOrder,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Execute a single order with retry logic"""
        
        try:
            # Pre-execution checks
            is_valid, error = self.exchange.check_order_validity(order)
            if not is_valid:
                return {
                    "order": order,
                    "success": False,
                    "error": error,
                    "filled_quantity": 0,
                    "fill_price": 0,
                }
            
            # Check account has sufficient capital
            if order.is_buy:
                required = order.max_cost
                available = self.account.available_cash
                if required > available:
                    return {
                        "order": order,
                        "success": False,
                        "error": f"Insufficient funds: need {required:.2f}, have {available:.2f}",
                        "filled_quantity": 0,
                        "fill_price": 0,
                    }
            
            # Submit order
            order_id = self.exchange.submit_order(order)
            order.order_id = order_id
            
            # Check execution result
            if order.status == "filled":
                return {
                    "order": order,
                    "success": True,
                    "order_id": order_id,
                    "filled_quantity": order.filled_quantity,
                    "fill_price": order.avg_fill_price,
                    "total_cost": order.filled_quantity * order.avg_fill_price,
                }
            elif order.status == "pending":
                self._pending_orders[order_id] = order
                return {
                    "order": order,
                    "success": True,
                    "order_id": order_id,
                    "filled_quantity": 0,
                    "fill_price": 0,
                    "status": "pending",
                }
            else:
                return {
                    "order": order,
                    "success": False,
                    "error": f"Order failed with status: {order.status}",
                    "filled_quantity": 0,
                    "fill_price": 0,
                }
                
        except Exception as e:
            if self.retry_failed_orders and retry_count < self.max_retries:
                logger.warning(f"Order failed, retrying ({retry_count + 1}/{self.max_retries}): {e}")
                return self._execute_single_order(order, retry_count + 1)
            
            return {
                "order": order,
                "success": False,
                "error": str(e),
                "filled_quantity": 0,
                "fill_price": 0,
            }
    
    def _update_positions(self, results: List[Dict]):
        """Update account positions based on execution results"""
        
        for result in results:
            if not result["success"] or result["filled_quantity"] == 0:
                continue
            
            order = result["order"]
            
            # Update position
            if order.is_buy:
                self.account.add_position(
                    market_id=order.market_id,
                    contract_type=order.contract_type,
                    quantity=result["filled_quantity"],
                    avg_price=result["fill_price"],
                )
                # Deduct cash
                self.account.deduct_cash(result["total_cost"])
            else:
                self.account.reduce_position(
                    market_id=order.market_id,
                    contract_type=order.contract_type,
                    quantity=result["filled_quantity"],
                    sell_price=result["fill_price"],
                )
    
    def _log_execution(self, order: ContractOrder, result: Dict):
        """Log execution details"""
        status = "✓" if result["success"] else "✗"
        print(f"{status} {order.direction.name} {order.quantity}x {order.market_id} "
              f"@ {order.limit_price:.2f} -> "
              f"filled {result['filled_quantity']}x @ {result.get('fill_price', 0):.2f}")
    
    def check_pending_orders(self) -> List[Dict]:
        """Check and update status of pending orders"""
        results = []
        
        filled_orders = []
        for order_id, order in self._pending_orders.items():
            status = self.exchange.get_order_status(order_id)
            
            if status.get("status") == "filled":
                order.status = "filled"
                order.filled_quantity = status["filled_quantity"]
                order.avg_fill_price = status["avg_fill_price"]
                filled_orders.append(order_id)
                
                results.append({
                    "order": order,
                    "success": True,
                    "filled_quantity": order.filled_quantity,
                    "fill_price": order.avg_fill_price,
                })
        
        # Remove filled orders from pending
        for order_id in filled_orders:
            del self._pending_orders[order_id]
        
        # Update positions for newly filled orders
        self._update_positions(results)
        
        return results
    
    def cancel_all_pending(self) -> int:
        """Cancel all pending orders"""
        cancelled = 0
        for order_id in list(self._pending_orders.keys()):
            if self.exchange.cancel_order(order_id):
                cancelled += 1
                del self._pending_orders[order_id]
        return cancelled
    
    def get_execution_summary(self) -> pd.DataFrame:
        """Get summary of all executions"""
        if not self._execution_history:
            return pd.DataFrame()
        
        records = []
        for entry in self._execution_history:
            for result in entry["results"]:
                order = result["order"]
                records.append({
                    "step": entry["step"],
                    "time": entry["time"],
                    "market_id": order.market_id,
                    "contract_type": order.contract_type,
                    "direction": order.direction.name,
                    "quantity": order.quantity,
                    "limit_price": order.limit_price,
                    "filled_quantity": result["filled_quantity"],
                    "fill_price": result.get("fill_price", 0),
                    "success": result["success"],
                    "error": result.get("error", ""),
                })
        
        return pd.DataFrame(records)


class NestedKalshiExecutor(KalshiExecutor):
    """
    Nested executor for multi-level execution strategies.
    
    Similar to Qlib's NestedExecutor - allows inner executors
    for order splitting, TWAP execution, etc.
    """
    
    def __init__(
        self,
        exchange: BaseKalshiExchange,
        account: KalshiAccount,
        inner_executor: KalshiExecutor,
        split_strategy: str = "equal",  # "equal", "time_weighted", "volume_weighted"
        num_splits: int = 5,
        **kwargs,
    ):
        """
        Parameters
        ----------
        inner_executor : KalshiExecutor
            Executor for individual split orders
        split_strategy : str
            How to split orders: equal, time_weighted, volume_weighted
        num_splits : int
            Number of splits for large orders
        """
        super().__init__(exchange, account, **kwargs)
        self.inner_executor = inner_executor
        self.split_strategy = split_strategy
        self.num_splits = num_splits
    
    def execute(
        self,
        trade_decision: PredictionMarketDecision,
        current_time: pd.Timestamp = None,
    ) -> List[Dict[str, Any]]:
        """Execute with order splitting"""
        
        all_results = []
        
        for order in trade_decision.get_order_list():
            # Split large orders
            split_orders = self._split_order(order)
            
            for split_order in split_orders:
                split_decision = PredictionMarketDecision(
                    orders=[split_order],
                    strategy=trade_decision.strategy,
                )
                results = self.inner_executor.execute(split_decision, current_time)
                all_results.extend(results)
        
        return all_results
    
    def _split_order(self, order: ContractOrder) -> List[ContractOrder]:
        """Split a large order into smaller pieces"""
        
        if order.quantity <= self.num_splits:
            return [order]
        
        split_qty = order.quantity // self.num_splits
        remainder = order.quantity % self.num_splits
        
        split_orders = []
        for i in range(self.num_splits):
            qty = split_qty + (1 if i < remainder else 0)
            if qty > 0:
                split_order = ContractOrder(
                    market_id=order.market_id,
                    contract_type=order.contract_type,
                    direction=order.direction,
                    quantity=qty,
                    limit_price=order.limit_price,
                    start_time=order.start_time,
                    end_time=order.end_time,
                )
                split_orders.append(split_order)
        
        return split_orders
