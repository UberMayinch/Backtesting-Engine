"""
Backtesting Engine Core Module

This module contains the main BacktestingEngine class that executes
trading strategies and calculates performance metrics.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings

from strategy import BaseStrategy


class BacktestingEngine:
    """
    Main backtesting engine that executes strategies and calculates metrics.
    
    This class handles:
    - Strategy execution on test data
    - Position tracking and PnL calculation
    - Performance metrics computation
    - Results aggregation and reporting
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0001):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital (float): Starting capital for backtesting
            transaction_cost (float): Transaction cost as fraction of trade value
            slippage (float): Slippage as fraction of price
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Results storage
        self.results = {}
        self.trade_log = []
        self.daily_returns = []
        self.daily_pnl = []
        self.positions = []
        self.portfolio_value = []
        
        # Performance tracking
        self.inference_times = []
        self.total_inference_time = 0.0
        
        # State variables
        self.current_position = 0.0
        self.current_capital = initial_capital
        self.trades_executed = 0
        
    def run_backtest(self, 
                     strategy: BaseStrategy, 
                     test_data_path: str,
                     position_sizing: str = 'fixed',
                     position_size: float = 1.0) -> Dict:
        """
        Execute a complete backtest for the given strategy.
        
        Args:
            strategy (BaseStrategy): Strategy instance to test
            test_data_path (str): Path to test data CSV
            position_sizing (str): Position sizing method ('fixed', 'percentage')
            position_size (float): Size of each position
            
        Returns:
            Dict: Complete backtest results including metrics and logs
        """
        print(f"Starting backtest with initial capital: ${self.initial_capital:,.2f}")
        
        # Load and validate test data
        test_data = self._load_test_data(test_data_path)
        
        # Initialize tracking variables
        self._initialize_backtest(len(test_data))
        
        # Run day-by-day backtesting
        start_time = time.time()
        
        for i in range(len(test_data)):
            self._execute_trading_day(strategy, test_data, i, position_sizing, position_size)
            
        self.total_inference_time = time.time() - start_time
        
        # Calculate final metrics
        results = self._calculate_final_metrics(test_data)
        
        print(f"Backtest completed. Total inference time: {self.total_inference_time:.2f} seconds")
        print(f"Final portfolio value: ${self.current_capital:.2f}")
        
        return results
    
    def _load_test_data(self, test_data_path: str) -> pd.DataFrame:
        """Load and validate test data."""
        try:
            test_data = pd.read_csv(test_data_path)
            
            # Validate OHLCV columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns 
                             if col.lower() not in [c.lower() for c in test_data.columns]]
            
            if missing_columns:
                raise ValueError(f"Missing required OHLCV columns: {missing_columns}")
            
            # Standardize column names
            column_mapping = {}
            for col in test_data.columns:
                for req_col in required_columns:
                    if col.lower() == req_col.lower():
                        column_mapping[col] = req_col
                        break
            
            test_data = test_data.rename(columns=column_mapping)
            
            # Handle datetime index
            date_columns = [col for col in test_data.columns 
                           if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_columns:
                test_data[date_columns[0]] = pd.to_datetime(test_data[date_columns[0]])
                test_data.set_index(date_columns[0], inplace=True)
            
            print(f"Loaded test data: {len(test_data)} rows")
            return test_data
            
        except Exception as e:
            raise ValueError(f"Error loading test data from {test_data_path}: {str(e)}")
    
    def _initialize_backtest(self, num_days: int):
        """Initialize tracking arrays for the backtest."""
        self.daily_returns = np.zeros(num_days)
        self.daily_pnl = np.zeros(num_days)
        self.positions = np.zeros(num_days)
        self.portfolio_value = np.zeros(num_days)
        self.trade_log = []
        self.inference_times = []
        
        self.current_position = 0.0
        self.current_capital = self.initial_capital
        self.trades_executed = 0
    
    def _execute_trading_day(self, 
                           strategy: BaseStrategy, 
                           test_data: pd.DataFrame, 
                           day_idx: int,
                           position_sizing: str,
                           position_size: float):
        """Execute trading logic for a single day."""
        
        # Get current day data (up to current day for prediction)
        current_data = test_data.iloc[:day_idx + 1]
        current_row = test_data.iloc[day_idx]
        
        # Skip if not enough data
        if len(current_data) < 2:
            self._record_day_results(day_idx, current_row, 0.0, 0.0)
            return
        
        # Get prediction from strategy (time this)
        pred_start_time = time.time()
        
        try:
            # Get prediction for current day
            predictions = strategy.predict(current_data)
            target_position = predictions[-1] if len(predictions) > 0 else 0.0
            
            inference_time = time.time() - pred_start_time
            self.inference_times.append(inference_time)
            
        except Exception as e:
            warnings.warn(f"Strategy prediction failed on day {day_idx}: {str(e)}")
            target_position = 0.0
            inference_time = time.time() - pred_start_time
            self.inference_times.append(inference_time)
        
        # Calculate position size
        if position_sizing == 'fixed':
            target_shares = target_position * position_size
        elif position_sizing == 'percentage':
            target_value = self.current_capital * position_size * target_position
            target_shares = target_value / current_row['close'] if current_row['close'] > 0 else 0.0
        else:
            target_shares = target_position
        
        # Execute trades if position changes
        position_change = target_shares - self.current_position
        trade_pnl = 0.0
        
        if abs(position_change) > 1e-6:  # Avoid tiny trades
            trade_pnl = self._execute_trade(current_row, position_change, day_idx)
            self.current_position = target_shares
        
        # Calculate daily PnL from position changes
        if day_idx > 0:
            prev_close = test_data.iloc[day_idx - 1]['close']
            price_change = (current_row['close'] - prev_close) / prev_close
            position_pnl = self.current_position * prev_close * price_change
        else:
            position_pnl = 0.0
        
        daily_pnl = position_pnl + trade_pnl
        daily_return = daily_pnl / self.current_capital if self.current_capital > 0 else 0.0
        
        # Update capital
        self.current_capital += daily_pnl
        
        # Record results
        self._record_day_results(day_idx, current_row, daily_return, daily_pnl)
    
    def _execute_trade(self, current_row: pd.Series, position_change: float, day_idx: int) -> float:
        """Execute a trade and calculate transaction costs."""
        
        # Calculate trade value
        trade_price = current_row['close'] * (1 + self.slippage * np.sign(position_change))
        trade_value = abs(position_change * trade_price)
        
        # Calculate transaction costs
        transaction_cost = trade_value * self.transaction_cost
        
        # Log the trade
        trade_log_entry = {
            'day': day_idx,
            'date': current_row.name if hasattr(current_row, 'name') else day_idx,
            'action': 'BUY' if position_change > 0 else 'SELL',
            'shares': abs(position_change),
            'price': trade_price,
            'value': trade_value,
            'cost': transaction_cost,
            'position_before': self.current_position,
            'position_after': self.current_position + position_change
        }
        
        self.trade_log.append(trade_log_entry)
        self.trades_executed += 1
        
        return -transaction_cost  # Transaction costs are negative PnL
    
    def _record_day_results(self, day_idx: int, current_row: pd.Series, 
                          daily_return: float, daily_pnl: float):
        """Record daily results."""
        self.daily_returns[day_idx] = daily_return
        self.daily_pnl[day_idx] = daily_pnl
        self.positions[day_idx] = self.current_position
        self.portfolio_value[day_idx] = self.current_capital
    
    def _calculate_final_metrics(self, test_data: pd.DataFrame) -> Dict:
        """Calculate all performance metrics and compile results."""
        
        # Remove any invalid values
        valid_returns = self.daily_returns[~np.isnan(self.daily_returns)]
        valid_pnl = self.daily_pnl[~np.isnan(self.daily_pnl)]
        
        metrics = {
            'net_pnl': self.current_capital - self.initial_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': self._calculate_sharpe_ratio(valid_returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'turnover': self._calculate_turnover(test_data),
            'inference_time': {
                'total_seconds': self.total_inference_time,
                'per_day_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0,
                'total_predictions': len(self.inference_times)
            },
            'trade_statistics': {
                'total_trades': self.trades_executed,
                'total_transaction_costs': sum([trade['cost'] for trade in self.trade_log])
            }
        }
        
        # Compile complete results
        self.results = {
            'metrics': metrics,
            'daily_data': {
                'returns': self.daily_returns,
                'pnl': self.daily_pnl,
                'positions': self.positions,
                'portfolio_value': self.portfolio_value
            },
            'trade_log': self.trade_log,
            'config': {
                'initial_capital': self.initial_capital,
                'transaction_cost': self.transaction_cost,
                'slippage': self.slippage
            }
        }
        
        return self.results
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Convert annual risk-free rate to daily
        daily_risk_free = risk_free_rate / 252
        
        excess_returns = returns - daily_risk_free
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.portfolio_value) == 0:
            return 0.0
        
        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(self.portfolio_value)
        
        # Calculate drawdown at each point
        drawdowns = (self.portfolio_value - running_max) / running_max
        
        return abs(np.min(drawdowns))
    
    def _calculate_turnover(self, test_data: pd.DataFrame) -> float:
        """Calculate portfolio turnover."""
        if len(self.trade_log) == 0:
            return 0.0
        
        total_trade_value = sum([trade['value'] for trade in self.trade_log])
        avg_portfolio_value = np.mean(self.portfolio_value[self.portfolio_value > 0])
        
        if avg_portfolio_value == 0:
            return 0.0
        
        # Annualized turnover
        days = len(test_data)
        return (total_trade_value / avg_portfolio_value) * (252 / days)
    
    def get_summary_stats(self) -> str:
        """Get formatted summary of backtest results."""
        if not self.results:
            return "No backtest results available. Run backtest first."
        
        metrics = self.results['metrics']
        
        summary = f"""
BACKTEST SUMMARY
================
Net PnL: ${metrics['net_pnl']:,.2f}
Total Return: {metrics['total_return']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
Max Drawdown: {metrics['max_drawdown']:.2%}
Turnover (Annualized): {metrics['turnover']:.2f}

EXECUTION STATS
===============
Total Trades: {metrics['trade_statistics']['total_trades']}
Transaction Costs: ${metrics['trade_statistics']['total_transaction_costs']:.2f}
Inference Time: {metrics['inference_time']['total_seconds']:.2f}s total
Avg per prediction: {metrics['inference_time']['per_day_ms']:.2f}ms

CONFIGURATION
=============
Initial Capital: ${self.results['config']['initial_capital']:,.2f}
Transaction Cost: {self.results['config']['transaction_cost']:.1%}
Slippage: {self.results['config']['slippage']:.2%}
        """
        
        return summary.strip()
