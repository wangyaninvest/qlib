# Prediction Market Trading Bot

This example demonstrates how to build a prediction market trading bot (e.g., for Kalshi) using Qlib's infrastructure patterns.

## Architecture

The execution layer follows Qlib's design patterns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXECUTION LAYER ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Signal    │    │  Strategy   │    │  Executor   │    │   Account   │  │
│  │  (Model)    │───▶│ (Decision)  │───▶│  (Orders)   │───▶│ (Position)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │           │
│        ▼                  ▼                  ▼                  ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Probability │    │ Edge-based  │    │ Order       │    │ Position    │  │
│  │ Estimates   │    │ + Kelly     │    │ Execution   │    │ & P&L       │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         EXCHANGE LAYER                                 │  │
│  │  ┌─────────────────┐              ┌─────────────────┐                 │  │
│  │  │  Simulated      │              │  Live Kalshi    │                 │  │
│  │  │  (Backtest)     │              │  (Production)   │                 │  │
│  │  └─────────────────┘              └─────────────────┘                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Decision Layer (`decision.py`)
- `ContractOrder`: Order representation for binary contracts
- `PredictionMarketDecision`: Collection of orders (like Qlib's `TradeDecisionWO`)
- `DecisionGenerator`: Creates orders from model signals

### 2. Exchange Layer (`exchange.py`)
- `BaseKalshiExchange`: Abstract interface for market operations
- `SimulatedKalshiExchange`: Backtesting with historical data
- `LiveKalshiExchange`: Real trading via Kalshi API

### 3. Executor Layer (`executor.py`)
- `KalshiExecutor`: Executes decisions, manages order lifecycle
- `NestedKalshiExecutor`: Order splitting (TWAP-like execution)

### 4. Strategy Layer (`strategy.py`)
- `EdgeBasedStrategy`: Trade when model prob differs from market
- `KellyPositionSizer`: Kelly criterion position sizing
- `ExitStrategy`: Profit targets, stop losses, time decay exits

### 5. Account Layer (`account.py`)
- `ContractPosition`: Individual position tracking
- `KalshiAccount`: Cash, positions, P&L management

## Usage

### Backtest Mode
```bash
cd examples/prediction_market
python example_bot.py --mode backtest --capital 10000
```

### Paper Trading (Demo API)
```bash
python example_bot.py --mode paper --api_key YOUR_DEMO_KEY
```

### Live Trading
```bash
python example_bot.py --mode live --api_key YOUR_API_KEY --capital 1000
```

## Key Differences from Stock Trading

| Aspect | Qlib (Stocks) | Prediction Markets |
|--------|---------------|-------------------|
| Price Range | Unbounded | $0.01 - $0.99 |
| Settlement | Continuous | Binary ($0 or $1) |
| Objective | Price direction | Event probability |
| Position Sizing | Portfolio weights | Kelly criterion |
| Risk Metric | Sharpe ratio | Brier score, log loss |

## Extending the Bot

### Custom Strategy
```python
from kalshi_execution import BaseMarketStrategy, PredictionMarketDecision

class MyStrategy(BaseMarketStrategy):
    def generate_trade_decision(self, signals, current_positions, 
                                 available_capital, current_time):
        decision = PredictionMarketDecision()
        # Your logic here
        return decision
```

### Custom Position Sizer
```python
from kalshi_execution.strategy import BasePositionSizer

class MyPositionSizer(BasePositionSizer):
    def size_position(self, edge, price, available_capital, model_prob):
        # Your sizing logic
        quantity = ...
        return quantity, quantity * price
```

## Integration with Qlib Models

You can use Qlib's models for probability estimation:

```python
import qlib
from qlib.contrib.model.gbdt import LGBModel

# Train model on historical event data
model = LGBModel()
model.fit(dataset)

# Get probability predictions
predictions = model.predict(dataset)

# Convert to signals for our strategy
signals = pd.DataFrame({
    "market_id": market_ids,
    "model_prob": predictions,
    "market_price": current_prices,
})
```

## Risk Management

The bot includes several risk controls:
- **Position limits**: Max contracts per market
- **Capital allocation**: Max % of capital deployed
- **Kelly fraction**: Reduced Kelly for safety
- **Stop losses**: Automatic exit on large losses
- **Profit targets**: Lock in gains
- **Time decay exits**: Close before expiration
