"""
Example Strategies and Demonstration Script

This module contains example strategy implementations and a demonstration
of how to use the backtesting engine.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from strategy import BaseStrategy, SimpleMovingAverageStrategy
from backtesting_engine import BacktestingEngine
from visualization import BacktestVisualizer, create_sample_visualizations


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands.
    
    This strategy:
    - Goes long when price is below lower Bollinger Band
    - Goes short when price is above upper Bollinger Band  
    - Exits when price returns to middle band (moving average)
    """
    
    def __init__(self, train_data_path: str, window: int = 20, num_std: float = 2.0):
        """
        Initialize mean reversion strategy.
        
        Args:
            train_data_path (str): Path to training data
            window (int): Moving average window for Bollinger Bands
            num_std (float): Number of standard deviations for bands
        """
        super().__init__(train_data_path)
        self.window = window
        self.num_std = num_std
        
        print(f"Mean Reversion Strategy initialized:")
        print(f"  - Bollinger Band window: {window}")
        print(f"  - Standard deviations: {num_std}")
    
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Generate mean reversion signals using Bollinger Bands."""
        
        if len(test_data) < self.window:
            return np.zeros(len(test_data))
        
        # Calculate Bollinger Bands
        close_prices = test_data['close']
        rolling_mean = close_prices.rolling(window=self.window).mean()
        rolling_std = close_prices.rolling(window=self.window).std()
        
        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)
        
        # Generate signals
        signals = np.zeros(len(test_data))
        
        for i in range(self.window, len(test_data)):
            current_price = close_prices.iloc[i]
            
            # Long signal: price below lower band
            if current_price < lower_band.iloc[i]:
                signals[i] = 1.0
            # Short signal: price above upper band
            elif current_price > upper_band.iloc[i]:
                signals[i] = -1.0
            # Exit signal: price near middle band
            elif abs(current_price - rolling_mean.iloc[i]) < rolling_std.iloc[i] * 0.5:
                signals[i] = 0.0
            else:
                # Hold previous position if no clear signal
                signals[i] = signals[i-1] if i > 0 else 0.0
        
        return signals


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using price momentum and volume confirmation.
    
    This strategy:
    - Calculates price momentum over multiple timeframes
    - Uses volume to confirm momentum signals
    - Implements dynamic position sizing based on momentum strength
    """
    
    def __init__(self, train_data_path: str, 
                 short_window: int = 5, 
                 long_window: int = 20,
                 volume_window: int = 10):
        """
        Initialize momentum strategy.
        
        Args:
            train_data_path (str): Path to training data
            short_window (int): Short-term momentum window
            long_window (int): Long-term momentum window  
            volume_window (int): Volume average window
        """
        super().__init__(train_data_path)
        self.short_window = short_window
        self.long_window = long_window
        self.volume_window = volume_window
        
        print(f"Momentum Strategy initialized:")
        print(f"  - Short momentum window: {short_window}")
        print(f"  - Long momentum window: {long_window}")
        print(f"  - Volume confirmation window: {volume_window}")
    
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Generate momentum signals with volume confirmation."""
        
        if len(test_data) < self.long_window:
            return np.zeros(len(test_data))
        
        close_prices = test_data['close']
        volumes = test_data['volume']
        
        # Calculate momentum indicators
        short_momentum = close_prices.pct_change(self.short_window)
        long_momentum = close_prices.pct_change(self.long_window)
        
        # Volume confirmation
        avg_volume = volumes.rolling(window=self.volume_window).mean()
        volume_ratio = volumes / avg_volume
        
        # Price strength
        rsi = self._calculate_rsi(close_prices, 14)
        
        signals = np.zeros(len(test_data))
        
        for i in range(self.long_window, len(test_data)):
            short_mom = short_momentum.iloc[i]
            long_mom = long_momentum.iloc[i]
            vol_confirm = volume_ratio.iloc[i]
            current_rsi = rsi.iloc[i]
            
            # Strong bullish momentum
            if (short_mom > 0.02 and long_mom > 0.01 and 
                vol_confirm > 1.2 and current_rsi < 70):
                signals[i] = 1.0
            
            # Strong bearish momentum  
            elif (short_mom < -0.02 and long_mom < -0.01 and
                  vol_confirm > 1.2 and current_rsi > 30):
                signals[i] = -1.0
            
            # Weak signals - partial positions
            elif (short_mom > 0.01 and long_mom > 0.005 and current_rsi < 80):
                signals[i] = 0.5
            elif (short_mom < -0.01 and long_mom < -0.005 and current_rsi > 20):
                signals[i] = -0.5
            
            # No clear signal - hold previous or neutral
            else:
                signals[i] = signals[i-1] * 0.8 if i > 0 else 0.0  # Decay positions
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def generate_sample_data(filename: str, num_days: int = 252, 
                        initial_price: float = 100.0, 
                        volatility: float = 0.02) -> None:
    """
    Generate sample OHLCV data for testing.
    
    Args:
        filename (str): Output CSV filename
        num_days (int): Number of trading days to generate
        initial_price (float): Starting price
        volatility (float): Daily volatility
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate price series with trend and noise
    returns = np.random.normal(0.0005, volatility, num_days)  # Slight upward drift
    prices = [initial_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove initial price
    
    # Generate OHLCV data
    data = []
    for i, close in enumerate(prices):
        # Generate realistic OHLC relationships
        daily_range = close * np.random.uniform(0.005, 0.03)  # 0.5% to 3% daily range
        
        high = close + np.random.uniform(0, daily_range * 0.7)
        low = close - np.random.uniform(0, daily_range * 0.7)
        
        # Ensure high >= close >= low
        high = max(high, close)
        low = min(low, close)
        
        # Open price based on previous close with gap
        if i == 0:
            open_price = close * np.random.uniform(0.98, 1.02)
        else:
            gap = np.random.normal(0, 0.005)  # Small overnight gaps
            open_price = prices[i-1] * (1 + gap)
        
        # Volume with some correlation to price movement
        base_volume = 1000000
        volatility_mult = abs(returns[i]) * 10 + 1
        volume = int(base_volume * volatility_mult * np.random.uniform(0.5, 2.0))
        
        # Date
        date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated sample data: {filename} ({num_days} days)")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")


def demo_backtesting_engine():
    """
    Comprehensive demonstration of the backtesting engine.
    
    This function:
    1. Generates sample data
    2. Tests multiple strategies
    3. Compares performance
    4. Creates visualizations
    """
    
    print("=" * 60)
    print("BACKTESTING ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n1. Generating sample data...")
    generate_sample_data('train_data.csv', num_days=500, initial_price=100.0)
    generate_sample_data('test_data.csv', num_days=252, initial_price=120.0)
    
    # 2. Initialize strategies
    print("\n2. Initializing strategies...")
    
    strategies = {
        'Moving Average': SimpleMovingAverageStrategy('train_data.csv', 
                                                     short_window=10, 
                                                     long_window=30),
        'Mean Reversion': MeanReversionStrategy('train_data.csv',
                                               window=20,
                                               num_std=2.0),
        'Momentum': MomentumStrategy('train_data.csv',
                                   short_window=5,
                                   long_window=20,
                                   volume_window=10)
    }
    
    # 3. Run backtests
    print("\n3. Running backtests...")
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n   Testing {name} Strategy:")
        print(f"   {'-' * (len(name) + 18)}")
        
        # Initialize backtest engine
        engine = BacktestingEngine(
            initial_capital=100000.0,
            transaction_cost=0.001,  # 0.1% transaction cost
            slippage=0.0001          # 0.01% slippage
        )
        
        # Run backtest
        result = engine.run_backtest(
            strategy=strategy,
            test_data_path='test_data.csv',
            position_sizing='percentage',
            position_size=0.1  # Use 10% of capital per position
        )
        
        results[name] = result
        
        # Print summary
        print(engine.get_summary_stats())
    
    # 4. Compare strategies
    print("\n4. Strategy Comparison:")
    print("=" * 80)
    
    comparison_data = []
    for name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Strategy': name,
            'Net PnL': metrics['net_pnl'],
            'Total Return': f"{metrics['total_return']:.1%}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
            'Max Drawdown': f"{metrics['max_drawdown']:.1%}",
            'Trades': metrics['trade_statistics']['total_trades'],
            'Turnover': f"{metrics['turnover']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 5. Create visualizations
    print("\n5. Creating visualizations...")
    
    for name, result in results.items():
        print(f"\n   Creating visualizations for {name} Strategy...")
        
        visualizer = BacktestVisualizer(result)
        
        # Create individual strategy dashboard
        visualizer.create_full_dashboard(
            figsize=(16, 12),
            save_path=f'{name.lower().replace(" ", "_")}_dashboard.png',
            show_plot=False
        )
        
        # Create PnL curve
        visualizer.plot_pnl_curve(
            save_path=f'{name.lower().replace(" ", "_")}_pnl.png',
            show_plot=False
        )
        
        print(f"   ✓ Visualizations saved for {name}")
    
    # 6. Performance summary
    print("\n6. Final Summary:")
    print("=" * 40)
    
    best_strategy = max(results.keys(), 
                       key=lambda x: results[x]['metrics']['net_pnl'])
    best_pnl = results[best_strategy]['metrics']['net_pnl']
    
    print(f"Best performing strategy: {best_strategy}")
    print(f"Best Net PnL: ${best_pnl:,.2f}")
    
    best_sharpe_strategy = max(results.keys(),
                              key=lambda x: results[x]['metrics']['sharpe_ratio'])
    best_sharpe = results[best_sharpe_strategy]['metrics']['sharpe_ratio']
    
    print(f"Best Sharpe ratio: {best_sharpe_strategy} ({best_sharpe:.3f})")
    
    print("\nFiles created:")
    print("- train_data.csv (training data)")
    print("- test_data.csv (test data)")
    for name in strategies.keys():
        print(f"- {name.lower().replace(' ', '_')}_dashboard.png")
        print(f"- {name.lower().replace(' ', '_')}_pnl.png")
    
    print("\nDemo completed successfully! 🎉")
    
    return results


if __name__ == "__main__":
    # Run the demonstration
    demo_results = demo_backtesting_engine()
    
    # Additional analysis can be performed here with demo_results
    print("\nTo use this backtesting engine in your own projects:")
    print("1. Create a strategy class inheriting from BaseStrategy")
    print("2. Implement the predict() method")
    print("3. Initialize BacktestingEngine with your parameters")
    print("4. Run backtest with your strategy and test data")
    print("5. Use BacktestVisualizer to analyze results")
