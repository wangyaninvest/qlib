# Copyright (c) Prediction Market Bot Example
# Exchange Layer - Extends Qlib's Exchange for Kalshi API interaction

from __future__ import annotations

import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np

from .decision import ContractOrder, ContractOrderDir, PredictionMarketDecision

logger = logging.getLogger(__name__)


@dataclass
class MarketQuote:
    """Real-time quote for a prediction market"""
    market_id: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    yes_bid_size: int
    yes_ask_size: int
    volume_24h: int
    open_interest: int
    status: str  # "open", "closed", "settled"
    expiration: pd.Timestamp
    result: Optional[str] = None  # "yes", "no", None if not settled
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    
    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2
    
    @property
    def no_mid(self) -> float:
        return (self.no_bid + self.no_ask) / 2
    
    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid


class BaseKalshiExchange:
    """
    Abstract base class for Kalshi exchange interactions.
    
    Mirrors Qlib's Exchange class but adapted for prediction markets.
    Can be implemented as:
    - LiveKalshiExchange: Real API calls
    - SimulatedKalshiExchange: Backtesting with historical data
    """
    
    def __init__(
        self,
        trading_fee: float = 0.0,  # Kalshi currently has no trading fees
        min_order_value: float = 1.0,
        max_position_per_market: int = 25000,
    ):
        """
        Parameters
        ----------
        trading_fee : float
            Fee per contract (currently 0 on Kalshi)
        min_order_value : float
            Minimum order value in dollars
        max_position_per_market : int
            Kalshi's position limit per market
        """
        self.trading_fee = trading_fee
        self.min_order_value = min_order_value
        self.max_position_per_market = max_position_per_market
        
        # Cache for market quotes
        self._quote_cache: Dict[str, MarketQuote] = {}
        self._cache_ttl = 1.0  # seconds
        self._cache_timestamps: Dict[str, float] = {}
    
    @abstractmethod
    def get_quote(self, market_id: str) -> MarketQuote:
        """Fetch current quote for a market"""
        raise NotImplementedError
    
    @abstractmethod
    def submit_order(self, order: ContractOrder) -> str:
        """Submit order and return order_id"""
        raise NotImplementedError
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        raise NotImplementedError
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get current status of an order"""
        raise NotImplementedError
    
    def is_tradable(
        self,
        market_id: str,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
    ) -> bool:
        """Check if a market is currently tradable"""
        try:
            quote = self.get_quote(market_id)
            return quote.status == "open"
        except Exception as e:
            logger.warning(f"Failed to check tradability for {market_id}: {e}")
            return False
    
    def check_order_validity(self, order: ContractOrder) -> Tuple[bool, str]:
        """
        Validate an order before submission.
        
        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
        """
        # Check market is tradable
        if not self.is_tradable(order.market_id):
            return False, f"Market {order.market_id} is not tradable"
        
        # Check price bounds
        if order.limit_price < 0.01 or order.limit_price > 0.99:
            return False, f"Price {order.limit_price} out of bounds [0.01, 0.99]"
        
        # Check minimum order value
        if order.max_cost < self.min_order_value:
            return False, f"Order value {order.max_cost} below minimum {self.min_order_value}"
        
        # Check position limit
        quote = self.get_quote(order.market_id)
        if order.quantity > self.max_position_per_market:
            return False, f"Order quantity {order.quantity} exceeds position limit"
        
        return True, ""
    
    def calculate_transaction_cost(self, order: ContractOrder) -> float:
        """Calculate total transaction cost including fees"""
        return order.quantity * self.trading_fee
    
    def get_best_price(
        self,
        market_id: str,
        contract_type: str,
        side: str,
    ) -> Optional[float]:
        """
        Get best available price for a trade.
        
        Parameters
        ----------
        market_id : str
            Market identifier
        contract_type : str
            "YES" or "NO"
        side : str
            "buy" or "sell"
        """
        quote = self.get_quote(market_id)
        
        if contract_type == "YES":
            return quote.yes_ask if side == "buy" else quote.yes_bid
        else:
            return quote.no_ask if side == "buy" else quote.no_bid


class SimulatedKalshiExchange(BaseKalshiExchange):
    """
    Simulated exchange for backtesting.
    
    Uses historical data to simulate order execution.
    Mirrors Qlib's backtesting exchange but for prediction markets.
    """
    
    def __init__(
        self,
        historical_data: pd.DataFrame,
        slippage: float = 0.01,
        fill_probability: float = 0.8,
        **kwargs,
    ):
        """
        Parameters
        ----------
        historical_data : pd.DataFrame
            Historical market data with columns:
            - timestamp, market_id, yes_bid, yes_ask, volume, result
        slippage : float
            Simulated slippage (price impact)
        fill_probability : float
            Probability of limit order getting filled
        """
        super().__init__(**kwargs)
        self.historical_data = historical_data
        self.slippage = slippage
        self.fill_probability = fill_probability
        
        # Simulation state
        self._current_time: pd.Timestamp = historical_data["timestamp"].min()
        self._pending_orders: Dict[str, ContractOrder] = {}
        self._order_counter = 0
        self._settled_markets: Dict[str, str] = {}  # market_id -> result
    
    def set_time(self, timestamp: pd.Timestamp):
        """Advance simulation time"""
        self._current_time = timestamp
        self._process_settlements()
    
    def _process_settlements(self):
        """Check for settled markets at current time"""
        mask = (
            (self.historical_data["timestamp"] <= self._current_time) &
            (self.historical_data["result"].notna())
        )
        settled = self.historical_data[mask].drop_duplicates("market_id", keep="last")
        
        for _, row in settled.iterrows():
            self._settled_markets[row["market_id"]] = row["result"]
    
    def get_quote(self, market_id: str) -> MarketQuote:
        """Get simulated quote from historical data"""
        
        # Find most recent data point
        mask = (
            (self.historical_data["market_id"] == market_id) &
            (self.historical_data["timestamp"] <= self._current_time)
        )
        data = self.historical_data[mask]
        
        if data.empty:
            raise ValueError(f"No data for market {market_id} at {self._current_time}")
        
        row = data.iloc[-1]
        
        # Check if settled
        result = self._settled_markets.get(market_id)
        status = "settled" if result else "open"
        
        return MarketQuote(
            market_id=market_id,
            yes_bid=row["yes_bid"],
            yes_ask=row["yes_ask"],
            no_bid=1 - row["yes_ask"],
            no_ask=1 - row["yes_bid"],
            yes_bid_size=row.get("yes_bid_size", 100),
            yes_ask_size=row.get("yes_ask_size", 100),
            volume_24h=row.get("volume", 0),
            open_interest=row.get("open_interest", 0),
            status=status,
            expiration=row.get("expiration", self._current_time + pd.Timedelta(days=1)),
            result=result,
            timestamp=self._current_time,
        )
    
    def submit_order(self, order: ContractOrder) -> str:
        """Submit order to simulated exchange"""
        
        # Validate order
        is_valid, error = self.check_order_validity(order)
        if not is_valid:
            raise ValueError(f"Invalid order: {error}")
        
        # Generate order ID
        self._order_counter += 1
        order_id = f"SIM-{self._order_counter:08d}"
        order.order_id = order_id
        
        # Simulate immediate execution with probability
        if np.random.random() < self.fill_probability:
            self._execute_order(order)
        else:
            order.status = "pending"
            self._pending_orders[order_id] = order
        
        return order_id
    
    def _execute_order(self, order: ContractOrder):
        """Simulate order execution with slippage"""
        quote = self.get_quote(order.market_id)
        
        # Apply slippage
        if order.is_buy:
            base_price = quote.yes_ask if order.contract_type == "YES" else quote.no_ask
            fill_price = min(base_price + self.slippage, 0.99)
        else:
            base_price = quote.yes_bid if order.contract_type == "YES" else quote.no_bid
            fill_price = max(base_price - self.slippage, 0.01)
        
        # Check if limit price is met
        if order.is_buy and fill_price > order.limit_price:
            order.status = "pending"
            return
        if not order.is_buy and fill_price < order.limit_price:
            order.status = "pending"
            return
        
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.status = "filled"
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self._pending_orders:
            self._pending_orders[order_id].status = "cancelled"
            del self._pending_orders[order_id]
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if order_id in self._pending_orders:
            order = self._pending_orders[order_id]
        else:
            # Would need to look up in order history
            return {"status": "unknown"}
        
        return {
            "order_id": order.order_id,
            "status": order.status,
            "filled_quantity": order.filled_quantity,
            "avg_fill_price": order.avg_fill_price,
        }


class LiveKalshiExchange(BaseKalshiExchange):
    """
    Live exchange for real trading on Kalshi.
    
    Requires Kalshi API credentials.
    """
    
    def __init__(
        self,
        api_key: str,
        private_key_path: str,
        environment: str = "demo",  # "demo" or "prod"
        rate_limit: float = 10.0,  # requests per second
        **kwargs,
    ):
        """
        Parameters
        ----------
        api_key : str
            Kalshi API key ID
        private_key_path : str
            Path to RSA private key for signing
        environment : str
            "demo" for paper trading, "prod" for live
        rate_limit : float
            Maximum API requests per second
        """
        super().__init__(**kwargs)
        
        self.api_key = api_key
        self.private_key_path = private_key_path
        self.environment = environment
        self.rate_limit = rate_limit
        
        # API base URL
        if environment == "demo":
            self.base_url = "https://demo-api.kalshi.co"
        else:
            self.base_url = "https://api.kalshi.com"
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_interval = 1.0 / rate_limit
        
        # Initialize API client
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Kalshi API client"""
        # Note: This would use the actual Kalshi SDK
        # For now, we show the structure
        try:
            # from kalshi_python import ApiInstance, Configuration
            # config = Configuration()
            # config.host = self.base_url
            # self._client = ApiInstance(config)
            # self._client.login(self.api_key, self.private_key_path)
            logger.info(f"Initialized Kalshi client for {self.environment}")
        except Exception as e:
            logger.error(f"Failed to initialize Kalshi client: {e}")
            raise
    
    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request_time = time.time()
    
    def get_quote(self, market_id: str) -> MarketQuote:
        """Fetch live quote from Kalshi API"""
        self._rate_limit_wait()
        
        # Check cache first
        cache_age = time.time() - self._cache_timestamps.get(market_id, 0)
        if market_id in self._quote_cache and cache_age < self._cache_ttl:
            return self._quote_cache[market_id]
        
        # API call would go here
        # response = self._client.get_market(market_id)
        # For demonstration:
        raise NotImplementedError(
            "Live API integration requires Kalshi SDK. "
            "Install with: pip install kalshi-python"
        )
    
    def submit_order(self, order: ContractOrder) -> str:
        """Submit order to live Kalshi API"""
        self._rate_limit_wait()
        
        is_valid, error = self.check_order_validity(order)
        if not is_valid:
            raise ValueError(f"Invalid order: {error}")
        
        # API call would go here
        # response = self._client.create_order(order.to_dict())
        # return response.order_id
        raise NotImplementedError("Live API integration requires Kalshi SDK")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order via Kalshi API"""
        self._rate_limit_wait()
        
        # response = self._client.cancel_order(order_id)
        # return response.success
        raise NotImplementedError("Live API integration requires Kalshi SDK")
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status from Kalshi API"""
        self._rate_limit_wait()
        
        # response = self._client.get_order(order_id)
        raise NotImplementedError("Live API integration requires Kalshi SDK")
