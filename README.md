# Backtesting Engine

A comprehensive Python backtesting engine for trading strategies with performance metrics calculation and visualization capabilities.

## Features

✅ **Strategy Framework**: Abstract base class for easy strategy development  
✅ **Performance Metrics**: Sharpe ratio, drawdown, turnover, net PnL, inference time  
✅ **Comprehensive Visualization**: PnL curves, drawdown charts, trade analysis  
✅ **Flexible Position Sizing**: Fixed or percentage-based position sizing  
✅ **Transaction Costs**: Configurable transaction costs and slippage  
✅ **Day-by-Day Execution**: Real-world backtesting simulation

## (MoMs Sep 28) 

This backtesting engine is designed to meet the Infinium competition requirements:

### Strategy Structure
- **Class with 2 functions**: `__init__()` and `predict()`
- **Init takes training data**: Loads OHLCV training data from CSV
- **Predict function**: Takes previous day data + today's data and outputs today's predicted signal
- **Time constraint**: Given fixed amount of time (like 10s), the model makes as many predictions as possible

### Required Metrics
1. **Sharpe Ratio**: Must be >= 2.0 for competitive performance
2. **Returns (PnL)**: Net profit and loss calculation
3. **Drawdown**: Maximum peak-to-trough decline measurement
4. **Turnover**: Portfolio turnover rate calculation
5. **Inference Time**: Model prediction speed measurement

### Security Requirements
- **No file access**: Strategies receive dataframes/lists as inputs, not file paths
- **No internet access**: Test scripts run in isolated Docker environment
- **Data isolation**: No external data sources during testing

### Implementation Notes
- The engine supports both the competition format and local development
- For competition submission, strategies should work with DataFrame inputs
- Local testing uses CSV file paths for convenience
- All required metrics are automatically calculated and reported  

## Quick Start

### 1. Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project and install dependencies
uv sync
```

### 2. Create Your Strategy

```python
from strategy import BaseStrategy
import numpy as np

class MyStrategy(BaseStrategy):
    def __init__(self, train_data_path: str):
        super().__init__(train_data_path)
        # Initialize your model here
    
    def predict(self, test_data):
        # Your prediction logic here
        # Return array of positions (-1, 0, 1 for short, hold, long)
        return np.random.choice([-1, 0, 1], size=len(test_data))
```

### 3. Run Backtest

```python
from backtesting_engine import BacktestingEngine

# Initialize engine
engine = BacktestingEngine(
    initial_capital=100000,
    transaction_cost=0.001,
    slippage=0.0001
)

# Create strategy
strategy = MyStrategy('train_data.csv')

# Run backtest
results = engine.run_backtest(
    strategy=strategy,
    test_data_path='test_data.csv',
    position_sizing='percentage',
    position_size=0.1
)

# Print results
print(engine.get_summary_stats())
```

### 4. Visualize Results

```python
from visualization import BacktestVisualizer

visualizer = BacktestVisualizer(results)
visualizer.create_full_dashboard()
visualizer.plot_pnl_curve()
```

## Demo

Run the comprehensive demo to see all features:

```bash
uv run demo.py
```

This will:
- Generate sample OHLCV data
- Test multiple strategies (Moving Average, Mean Reversion, Momentum)
- Compare performance metrics
- Create detailed visualizations

## Core Components

### Strategy Class (`strategy.py`)
- **BaseStrategy**: Abstract base class for all trading strategies
- **SimpleMovingAverageStrategy**: Example implementation using moving averages
- Automatic CSV loading and OHLCV validation
- Built-in data preprocessing and error handling

### Backtesting Engine (`backtesting_engine.py`)
- **BacktestingEngine**: Main execution engine
- Day-by-day strategy execution
- Position tracking and PnL calculation
- Transaction cost and slippage simulation
- Performance metrics calculation

### Visualization (`visualization.py`)
- **BacktestVisualizer**: Comprehensive plotting capabilities
- Dashboard with multiple charts
- PnL curves with drawdown highlighting
- Trade analysis and distribution plots
- Customizable chart styling

## Performance Metrics

The engine calculates the following metrics:

1. **Sharpe Ratio**: Risk-adjusted returns measure
2. **Maximum Drawdown**: Largest peak-to-trough decline
3. **Turnover**: Portfolio turnover rate (annualized)
4. **Net PnL**: Total profit and loss
5. **Inference Time**: Strategy execution timing

## Data Format

### Training Data (OHLCV CSV)
```csv
date,open,high,low,close,volume
2023-01-01,100.0,102.0,99.5,101.5,1000000
2023-01-02,101.5,103.0,101.0,102.5,1200000
...
```

### Test Data
Same format as training data. The strategy's `predict()` method processes this data day by day.

## Advanced Features

### Custom Position Sizing
```python
# Fixed position size
results = engine.run_backtest(
    strategy=strategy,
    test_data_path='test.csv',
    position_sizing='fixed',
    position_size=100  # 100 shares
)

# Percentage of capital
results = engine.run_backtest(
    strategy=strategy,
    test_data_path='test.csv',
    position_sizing='percentage', 
    position_size=0.2  # 20% of capital
)
```

### Transaction Costs Configuration
```python
engine = BacktestingEngine(
    initial_capital=100000,
    transaction_cost=0.001,  # 0.1% per trade
    slippage=0.0001         # 0.01% price impact
)
```

### Multiple Strategy Comparison
The demo script shows how to compare multiple strategies side by side with comprehensive metrics and visualizations.

## Example Strategies Included

1. **SimpleMovingAverageStrategy**: Basic moving average crossover
2. **MeanReversionStrategy**: Bollinger Bands mean reversion
3. **MomentumStrategy**: Price momentum with volume confirmation

## Project Structure

```
Backtesting-Engine/
├── README.md
├── pyproject.toml        # uv project configuration
├── .python-version       # Python version for uv
├── .gitignore           # Git ignore patterns
├── Makefile             # Development commands
├── __init__.py
├── strategy.py           # Base strategy classes
├── backtesting_engine.py # Core backtesting engine
├── visualization.py      # Plotting and visualization
└── demo.py              # Demonstration script
```

## Requirements

- [uv](https://astral.sh/uv/) - Fast Python package installer and resolver
- Python 3.8+ (managed by uv)
- Dependencies are automatically managed via `pyproject.toml`

## Development with uv

### Setting up the development environment

```bash
# Clone the repository
git clone https://github.com/UberMayinch/Backtesting-Engine.git
cd Backtesting-Engine

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv sync

# Install development dependencies
uv sync --extra dev
```

### Running tests and linting

Using Make (recommended):
```bash
# Run tests
make test

# Format code
make format

# Run linting
make lint

# Run demo
make demo

# See all available commands
make help
```

Direct uv commands:
```bash
# Run tests
uv run pytest

# Format code with black
uv run black .

# Lint code with ruff
uv run ruff check .

# Type checking with mypy
uv run mypy .
```

### Adding dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add a dependency with version constraint
uv add "package-name>=1.0.0"
```

### Why uv?

This project uses uv for modern Python dependency management:
- ⚡ **Much faster** dependency resolution and installation
- 🔒 **Automatic lock file management** for reproducible builds
- 🛡️ **Built-in security** with dependency verification
- 🚀 **Modern Python tooling** with integrated project management
- 🎯 **Single command setup** - no activation needed

## Contributing

Feel free to contribute by:
- Adding new strategy examples
- Improving performance metrics
- Enhancing visualizations
- Adding new features

## License

This project is open source and available under the MIT License.
