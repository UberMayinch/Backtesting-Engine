"""
Visualization Module for Backtesting Engine

This module provides comprehensive visualization capabilities for
backtesting results including PnL charts, drawdown plots, and
performance analytics.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class BacktestVisualizer:
    """
    Comprehensive visualization class for backtesting results.
    
    Provides various charts and plots to analyze strategy performance
    including PnL curves, drawdowns, trade analysis, and metrics dashboard.
    """
    
    def __init__(self, results: Dict):
        """
        Initialize visualizer with backtest results.
        
        Args:
            results (Dict): Results dictionary from BacktestingEngine
        """
        self.results = results
        self.metrics = results['metrics']
        self.daily_data = results['daily_data']
        self.trade_log = results['trade_log']
        self.config = results['config']
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_full_dashboard(self, 
                            figsize: Tuple[int, int] = (16, 12),
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> plt.Figure:
        """
        Create a comprehensive dashboard with all key visualizations.
        
        Args:
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=figsize)
        
        # Create subplot layout (2x3 grid)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio Value Over Time (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_portfolio_value(ax1)
        
        # 2. Drawdown Chart (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_drawdown(ax2)
        
        # 3. Daily Returns Distribution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_returns_distribution(ax3)
        
        # 4. Position Over Time (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_position_over_time(ax4)
        
        # 5. Cumulative PnL (bottom spanning both columns)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_cumulative_pnl(ax5)
        
        # Add main title
        fig.suptitle('Backtesting Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        # Add metrics text box
        self._add_metrics_textbox(fig)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        if show_plot:
            plt.show()
            
        return fig
    
    def plot_pnl_curve(self, 
                      figsize: Tuple[int, int] = (12, 6),
                      save_path: Optional[str] = None,
                      show_plot: bool = True) -> plt.Figure:
        """
        Plot the main PnL curve with annotations.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save figure
            show_plot: Whether to display plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative PnL
        cumulative_pnl = np.cumsum(self.daily_data['pnl'])
        days = range(len(cumulative_pnl))
        
        # Plot PnL curve
        ax.plot(days, cumulative_pnl, linewidth=2, color='#2E86AB', label='Cumulative PnL')
        ax.fill_between(days, cumulative_pnl, 0, alpha=0.3, color='#2E86AB')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Highlight major drawdowns
        portfolio_values = self.daily_data['portfolio_value']
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        
        # Find major drawdown periods (>5%)
        major_dd_mask = drawdowns < -0.05
        if np.any(major_dd_mask):
            ax.fill_between(days, cumulative_pnl, 0, where=major_dd_mask, 
                           color='red', alpha=0.2, label='Major Drawdown (>5%)')
        
        # Formatting
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.set_title('Portfolio Profit & Loss Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key statistics
        final_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0
        max_pnl = np.max(cumulative_pnl) if len(cumulative_pnl) > 0 else 0
        min_pnl = np.min(cumulative_pnl) if len(cumulative_pnl) > 0 else 0
        
        textstr = f'Final PnL: ${final_pnl:,.0f}\\nMax PnL: ${max_pnl:,.0f}\\nMin PnL: ${min_pnl:,.0f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig
    
    def _plot_portfolio_value(self, ax):
        """Plot portfolio value over time."""
        portfolio_values = self.daily_data['portfolio_value']
        days = range(len(portfolio_values))
        
        ax.plot(days, portfolio_values, linewidth=2, color='#A23B72')
        ax.axhline(y=self.config['initial_capital'], color='gray', 
                  linestyle='--', alpha=0.7, label='Initial Capital')
        
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Value Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _plot_drawdown(self, ax):
        """Plot drawdown chart."""
        portfolio_values = self.daily_data['portfolio_value']
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max * 100
        
        days = range(len(drawdowns))
        
        ax.fill_between(days, drawdowns, 0, color='red', alpha=0.6)
        ax.plot(days, drawdowns, color='darkred', linewidth=1)
        
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Portfolio Drawdown')
        ax.grid(True, alpha=0.3)
        
        # Highlight max drawdown
        max_dd = np.min(drawdowns)
        max_dd_day = np.argmin(drawdowns)
        ax.plot(max_dd_day, max_dd, 'ro', markersize=8, 
                label=f'Max DD: {max_dd:.1f}%')
        ax.legend()
    
    def _plot_returns_distribution(self, ax):
        """Plot daily returns distribution."""
        returns = self.daily_data['returns'] * 100  # Convert to percentage
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        if len(returns) == 0:
            ax.text(0.5, 0.5, 'No valid returns data', ha='center', va='center')
            ax.set_title('Daily Returns Distribution')
            return
        
        # Histogram
        n_bins = min(50, len(returns) // 5) if len(returns) > 10 else 10
        ax.hist(returns, bins=n_bins, alpha=0.7, color='#F18F01', edgecolor='black')
        
        # Add normal distribution overlay
        mu, sigma = np.mean(returns), np.std(returns)
        x = np.linspace(returns.min(), returns.max(), 100)
        y = ((1 / (sigma * np.sqrt(2 * np.pi))) * 
             np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        
        # Scale normal curve to match histogram
        y = y * len(returns) * (returns.max() - returns.min()) / n_bins
        ax.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
        
        ax.axvline(mu, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mu:.2f}%')
        
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Daily Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_position_over_time(self, ax):
        """Plot position sizes over time."""
        positions = self.daily_data['positions']
        days = range(len(positions))
        
        # Create step plot for positions
        ax.step(days, positions, where='post', linewidth=2, color='#C73E1D')
        ax.fill_between(days, positions, 0, step='post', alpha=0.3, color='#C73E1D')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Position Size')
        ax.set_title('Position Size Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add annotations for position statistics
        max_pos = np.max(np.abs(positions))
        avg_abs_pos = np.mean(np.abs(positions))
        textstr = f'Max Position: {max_pos:.2f}\\nAvg Abs Position: {avg_abs_pos:.2f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    def _plot_cumulative_pnl(self, ax):
        """Plot cumulative PnL with additional details."""
        cumulative_pnl = np.cumsum(self.daily_data['pnl'])
        days = range(len(cumulative_pnl))
        
        # Main PnL line
        ax.plot(days, cumulative_pnl, linewidth=3, color='#2E86AB', label='Cumulative PnL')
        
        # Add trade markers if we have trade log
        if self.trade_log:
            trade_days = [trade['day'] for trade in self.trade_log]
            trade_pnls = [cumulative_pnl[day] if day < len(cumulative_pnl) else 0 
                         for day in trade_days]
            
            buy_days = [day for trade, day in zip(self.trade_log, trade_days) 
                       if trade['action'] == 'BUY']
            sell_days = [day for trade, day in zip(self.trade_log, trade_days) 
                        if trade['action'] == 'SELL']
            
            buy_pnls = [cumulative_pnl[day] if day < len(cumulative_pnl) else 0 
                       for day in buy_days]
            sell_pnls = [cumulative_pnl[day] if day < len(cumulative_pnl) else 0 
                        for day in sell_days]
            
            if buy_days:
                ax.scatter(buy_days, buy_pnls, color='green', marker='^', 
                          s=50, alpha=0.7, label='Buy Trades')
            if sell_days:
                ax.scatter(sell_days, sell_pnls, color='red', marker='v', 
                          s=50, alpha=0.7, label='Sell Trades')
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.set_title('Cumulative Profit & Loss with Trade Markers', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _add_metrics_textbox(self, fig):
        """Add a text box with key metrics to the figure."""
        metrics_text = f"""KEY METRICS
Net PnL: ${self.metrics['net_pnl']:,.0f}
Total Return: {self.metrics['total_return']:.1%}
Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}
Max Drawdown: {self.metrics['max_drawdown']:.1%}
Turnover: {self.metrics['turnover']:.2f}
Total Trades: {self.metrics['trade_statistics']['total_trades']}
Inference Time: {self.metrics['inference_time']['total_seconds']:.1f}s"""
        
        # Add text box in figure coordinates
        fig.text(0.02, 0.02, metrics_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                verticalalignment='bottom')
    
    def plot_trade_analysis(self, 
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
        """
        Create detailed trade analysis plots.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional save path
            show_plot: Whether to show plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.trade_log:
            print("No trades to analyze.")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Convert trade log to DataFrame for easier analysis
        trades_df = pd.DataFrame(self.trade_log)
        
        # 1. Trade sizes distribution
        ax1.hist(trades_df['shares'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Trade Size (Shares)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Trade Sizes')
        ax1.grid(True, alpha=0.3)
        
        # 2. Trade values over time
        ax2.plot(trades_df['day'], trades_df['value'], 'o-', alpha=0.7)
        ax2.set_xlabel('Trading Day')
        ax2.set_ylabel('Trade Value ($)')
        ax2.set_title('Trade Values Over Time')
        ax2.grid(True, alpha=0.3)
        
        # 3. Transaction costs over time
        ax3.bar(range(len(trades_df)), trades_df['cost'], alpha=0.7, color='orange')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Transaction Cost ($)')
        ax3.set_title('Transaction Costs per Trade')
        ax3.grid(True, alpha=0.3)
        
        # 4. Buy vs Sell analysis
        action_counts = trades_df['action'].value_counts()
        if len(action_counts) > 1:
            ax4.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%',
                   colors=['lightgreen', 'lightcoral'])
            ax4.set_title('Buy vs Sell Distribution')
        else:
            ax4.text(0.5, 0.5, f'Only {action_counts.index[0]} trades', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Trade Action Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig


def create_sample_visualizations(results: Dict, output_dir: str = './'):
    """
    Convenience function to create all standard visualizations.
    
    Args:
        results: Backtest results dictionary
        output_dir: Directory to save plots
    """
    visualizer = BacktestVisualizer(results)
    
    print("Creating visualizations...")
    
    # Main dashboard
    visualizer.create_full_dashboard(
        save_path=f"{output_dir}/backtest_dashboard.png",
        show_plot=False
    )
    print("✓ Dashboard created")
    
    # PnL curve
    visualizer.plot_pnl_curve(
        save_path=f"{output_dir}/pnl_curve.png",
        show_plot=False
    )
    print("✓ PnL curve created")
    
    # Trade analysis (if trades exist)
    if results.get('trade_log'):
        visualizer.plot_trade_analysis(
            save_path=f"{output_dir}/trade_analysis.png",
            show_plot=False
        )
        print("✓ Trade analysis created")
    
    print(f"All visualizations saved to {output_dir}")
