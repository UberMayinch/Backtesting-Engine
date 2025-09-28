"""
Base Strategy Class for Backtesting Engine

This module defines the abstract base class for trading strategies.
All custom strategies should inherit from this class and implement
the required methods.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union, Optional


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement __init__ and predict methods.
    The __init__ method should load training data from CSV,
    and the predict method should generate predictions for test data.
    """
    
    def __init__(self, train_data_path: str):
        """
        Initialize the strategy with training data.
        
        Args:
            train_data_path (str): Path to CSV file containing OHLCV training data
        """
        self.train_data_path = train_data_path
        self.train_data = None
        self.model = None
        self.is_fitted = False
        
        # Load and validate training data
        self._load_training_data()
        
    def _load_training_data(self):
        """Load training data from CSV and validate OHLCV format."""
        try:
            self.train_data = pd.read_csv(self.train_data_path)
            
            # Validate OHLCV columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns 
                             if col.lower() not in [c.lower() for c in self.train_data.columns]]
            
            if missing_columns:
                raise ValueError(f"Missing required OHLCV columns: {missing_columns}")
                
            # Standardize column names to lowercase
            column_mapping = {}
            for col in self.train_data.columns:
                for req_col in required_columns:
                    if col.lower() == req_col.lower():
                        column_mapping[col] = req_col
                        break
                        
            self.train_data = self.train_data.rename(columns=column_mapping)
            
            # Ensure datetime index if 'date' or 'datetime' column exists
            date_columns = [col for col in self.train_data.columns 
                           if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_columns:
                self.train_data[date_columns[0]] = pd.to_datetime(self.train_data[date_columns[0]])
                self.train_data.set_index(date_columns[0], inplace=True)
            
            print(f"Loaded training data: {len(self.train_data)} rows")
            print(f"Date range: {self.train_data.index.min()} to {self.train_data.index.max()}")
            
        except Exception as e:
            raise ValueError(f"Error loading training data from {self.train_data_path}: {str(e)}")
    
    @abstractmethod
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for test data.
        
        This method must be implemented by each strategy.
        It should return predictions for each row in the test data.
        
        Args:
            test_data (pd.DataFrame): Test data with same format as training data
            
        Returns:
            np.ndarray: Array of predictions (typically -1, 0, 1 for short, hold, long)
        """
        pass
    
    def get_train_data_summary(self) -> dict:
        """Get summary statistics of the training data."""
        if self.train_data is None:
            return {}
            
        return {
            'rows': len(self.train_data),
            'date_range': {
                'start': str(self.train_data.index.min()),
                'end': str(self.train_data.index.max())
            },
            'price_stats': {
                'mean_close': self.train_data['close'].mean(),
                'std_close': self.train_data['close'].std(),
                'min_close': self.train_data['close'].min(),
                'max_close': self.train_data['close'].max()
            },
            'volume_stats': {
                'mean_volume': self.train_data['volume'].mean(),
                'total_volume': self.train_data['volume'].sum()
            }
        }


class SimpleMovingAverageStrategy(BaseStrategy):
    """
    Example strategy implementation using simple moving averages.
    
    This demonstrates how to inherit from BaseStrategy and implement
    the required methods.
    """
    
    def __init__(self, train_data_path: str, short_window: int = 10, long_window: int = 30):
        """
        Initialize the moving average strategy.
        
        Args:
            train_data_path (str): Path to training data CSV
            short_window (int): Period for short moving average
            long_window (int): Period for long moving average
        """
        super().__init__(train_data_path)
        self.short_window = short_window
        self.long_window = long_window
        
        # Calculate moving averages on training data to validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate that window parameters are appropriate for the data."""
        if self.long_window >= len(self.train_data):
            raise ValueError(f"Long window ({self.long_window}) must be less than data length ({len(self.train_data)})")
            
        if self.short_window >= self.long_window:
            raise ValueError(f"Short window ({self.short_window}) must be less than long window ({self.long_window})")
    
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals based on moving average crossover.
        
        Returns:
            1 for long position (short MA > long MA)
            0 for hold/no position
            -1 for short position (short MA < long MA)
        """
        # Calculate moving averages
        short_ma = test_data['close'].rolling(window=self.short_window).mean()
        long_ma = test_data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals = np.zeros(len(test_data))
        
        # Long signal when short MA crosses above long MA
        signals[short_ma > long_ma] = 1
        
        # Short signal when short MA crosses below long MA  
        signals[short_ma < long_ma] = -1
        
        # Handle NaN values (not enough data for MA calculation)
        signals = np.nan_to_num(signals, nan=0.0)
        
        return signals
