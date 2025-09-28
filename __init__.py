"""
Backtesting Engine Package

A comprehensive Python backtesting engine for trading strategies with
performance metrics calculation and visualization capabilities.
"""

from .strategy import BaseStrategy, SimpleMovingAverageStrategy
from .backtesting_engine import BacktestingEngine
from .visualization import BacktestVisualizer, create_sample_visualizations

__version__ = "1.0.0"
__author__ = "Backtesting Engine Team"

__all__ = [
    'BaseStrategy',
    'SimpleMovingAverageStrategy', 
    'BacktestingEngine',
    'BacktestVisualizer',
    'create_sample_visualizations'
]