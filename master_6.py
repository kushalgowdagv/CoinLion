import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import datetime
warnings.filterwarnings('ignore')
import concurrent.futures
import itertools
from tqdm import tqdm


def get_output_directory(symbol, timeframe, strategy_type):
    """
    Create a timestamped output directory structure.
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        strategy_type (str): Strategy type
        
    Returns:
        tuple: (base_output_dir, strategy_dir)
    """
    # Create timestamp string
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base output directory
    base_output_dir = os.path.join('output', f"{symbol}_{timeframe}_{timestamp}")
    
    # Create strategy directory
    strategy_dir = os.path.join(base_output_dir, f"{strategy_type}_strategy")
    
    # Create directories if they don't exist
    os.makedirs(strategy_dir, exist_ok=True)
    
    return base_output_dir, strategy_dir

class DataHandler:
    def __init__(self, base_dir=None):
        """
        Initialize the DataHandler with base directory.
        If no base_dir is provided, use current working directory.
        """
        self.base_dir = base_dir if base_dir else os.getcwd()
        # Initialize data integrity tracking
        self.data_integrity_issues = {
            'missing_values': {},
            'duplicate_timestamps': [],
            'non_uniform_intervals': []
        }
        # Output directory structure will be set when data is loaded
        self.output_base_dir = None
        self.output_strategy_dir = None
        
    def load_data(self, symbol, timeframe='10m', strategy_type='buy_sell'):
        """
        Load data for a specific symbol and timeframe.
        
        Args:
            symbol (str): Symbol name (e.g., 'btcusd', 'ethusd')
            timeframe (str): Timeframe (e.g., '10m')
            strategy_type (str): Strategy type for directory naming
            
        Returns:
            pd.DataFrame: Processed dataframe with OHLCV data
        """
        # Set output directories
        self.output_base_dir, self.output_strategy_dir = get_output_directory(symbol, timeframe, strategy_type)
        self.symbol = symbol
        self.timeframe = timeframe
        
        filepath = os.path.join(self.base_dir, f'data/{symbol}_{timeframe}.csv')
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        # Load data
        df = pd.read_csv(filepath)
        
        # Convert time columns to datetime
        df['time_utc'] = pd.to_datetime(df['time_utc'])
        df['time_est'] = pd.to_datetime(df['time_est'])
        
        # Set time_utc as index
        df.set_index('time_utc', inplace=True)
        
        # Perform data integrity checks
        self._check_data_integrity(df)
        
        # Save data integrity issues to CSV
        self._save_data_integrity_issues()
        
        return df
    
    def _check_data_integrity(self, df):
        """
        Check data integrity:
        - No missing values in OHLCV
        - No duplicate timestamps
        - Uniform time intervals
        
        Args:
            df (pd.DataFrame): DataFrame to check
        """
        # Check for missing values in OHLCV
        ohlcv_cols = ['o', 'h', 'l', 'c', 'v']
        for col in ohlcv_cols:
            missing = df[df[col].isnull()]
            if not missing.empty:
                self.data_integrity_issues['missing_values'][col] = missing.index.tolist()
                print(f"Warning: Found {len(missing)} missing values in '{col}' column")
        
        # Check for duplicate timestamps
        duplicates = df.index[df.index.duplicated()].tolist()
        if duplicates:
            self.data_integrity_issues['duplicate_timestamps'] = duplicates
            print(f"Warning: Found {len(duplicates)} duplicate timestamps")
        
        # Check for uniform time intervals
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=10)
        non_uniform = time_diffs[time_diffs != expected_diff]
        if not non_uniform.empty:
            self.data_integrity_issues['non_uniform_intervals'] = [
                {'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'), 
                 'interval': diff.total_seconds() / 60} 
                for idx, diff in non_uniform.items()
            ]
            print(f"Warning: Found {len(non_uniform)} non-uniform time intervals")
    
    def _save_data_integrity_issues(self):
        """
        Save data integrity issues to CSV in the output directory structure.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save missing values
        missing_values_data = []
        for col, timestamps in self.data_integrity_issues['missing_values'].items():
            for ts in timestamps:
                missing_values_data.append({
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'column': col
                })
                
        if missing_values_data:
            missing_df = pd.DataFrame(missing_values_data)
            missing_file = os.path.join(self.output_strategy_dir, 
                                        f"{self.symbol}_{self.timeframe}_missing_values_{timestamp}.csv")
            missing_df.to_csv(missing_file, index=False)
            
        # Save duplicate timestamps
        if self.data_integrity_issues['duplicate_timestamps']:
            duplicates_df = pd.DataFrame({
                'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') 
                             for ts in self.data_integrity_issues['duplicate_timestamps']]
            })
            duplicates_file = os.path.join(self.output_strategy_dir, 
                                          f"{self.symbol}_{self.timeframe}_duplicate_timestamps_{timestamp}.csv")
            duplicates_df.to_csv(duplicates_file, index=False)
            
        # Save non-uniform intervals
        if self.data_integrity_issues['non_uniform_intervals']:
            non_uniform_df = pd.DataFrame(self.data_integrity_issues['non_uniform_intervals'])
            non_uniform_file = os.path.join(self.output_strategy_dir, 
                                           f"{self.symbol}_{self.timeframe}_non_uniform_intervals_{timestamp}.csv")
            non_uniform_df.to_csv(non_uniform_file, index=False)
    
    def resample_data(self, df, timeframe='10T'):
        """
        Resample data to ensure uniform time intervals.
        
        Args:
            df (pd.DataFrame): DataFrame to resample
            timeframe (str): Pandas time frequency string
            
        Returns:
            pd.DataFrame: Resampled dataframe
        """
        # Save original data for comparison
        original_df = df.copy()
        
        # Resample to uniform intervals
        resampled = df.resample(timeframe).agg({
            'o': 'first',
            'h': 'max',
            'l': 'min',
            'c': 'last',
            'v': 'sum',
            'symbol': 'first',
            'time_est': 'first'
        })
        
        # Forward fill symbol and time_est
        resampled['symbol'] = resampled['symbol'].ffill()
        resampled['time_est'] = resampled['time_est'].ffill()
        
        # Track resampled data changes
        resampled_changes = []
        
        # Check for missing values after resampling
        missing_after = resampled[['o', 'h', 'l', 'c', 'v']].isnull().sum().sum()
        if missing_after > 0:
            print(f"Warning: After resampling, found {missing_after} missing values")
            
            # Save information about which values were interpolated
            missing_mask = resampled[['o', 'h', 'l', 'c', 'v']].isnull()
            for idx in resampled[missing_mask.any(axis=1)].index:
                missing_cols = [col for col in ['o', 'h', 'l', 'c', 'v'] if missing_mask.loc[idx, col]]
                resampled_changes.append({
                    'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'interpolated',
                    'columns': ','.join(missing_cols)
                })
            
            # Fill missing values using interpolation for OHLC
            resampled[['o', 'h', 'l', 'c']] = resampled[['o', 'h', 'l', 'c']].interpolate(method='linear')
            # Fill missing volume with 0
            resampled['v'] = resampled['v'].fillna(0)
        
        # Find newly added timestamps (timestamps in resampled but not in original)
        original_timestamps = set(original_df.index)
        resampled_timestamps = set(resampled.index)
        new_timestamps = resampled_timestamps - original_timestamps
        
        for ts in new_timestamps:
            resampled_changes.append({
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'added',
                'columns': 'o,h,l,c,v'
            })
        
        # Save resampled changes to CSV
        if resampled_changes:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            changes_df = pd.DataFrame(resampled_changes)
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown'
            
            changes_file = os.path.join(self.output_strategy_dir, 
                                      f"{symbol}_{timeframe}_resampled_changes_{timestamp}.csv")
            changes_df.to_csv(changes_file, index=False)
            
        return resampled
    

class MACDStrategy:
    def __init__(self, short_window=12, long_window=26, signal_window=9):
        """
        Initialize MACD strategy with customizable parameters.
        
        Args:
            short_window (int): Short EMA window
            long_window (int): Long EMA window
            signal_window (int): Signal EMA window
        """
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        
    def calculate_indicators(self, df):
        """
        Calculate MACD and signal line.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with MACD indicators added
        """
        # Create a copy to avoid modifying original data
        result = df.copy()
        
        # Calculate EMAs
        result['ema_short'] = result['c'].ewm(span=self.short_window, adjust=False).mean()
        result['ema_long'] = result['c'].ewm(span=self.long_window, adjust=False).mean()
        
        # Calculate MACD and signal line
        result['macd'] = result['ema_short'] - result['ema_long']
        result['macd_signal'] = result['macd'].ewm(span=self.signal_window, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        return result
    
    def generate_signals(self, df, strategy_type='buy_sell'):
        """
        Generate buy/sell signals based on MACD crossovers.
        
        Args:
            df (pd.DataFrame): DataFrame with MACD indicators
            strategy_type (str): Type of strategy ('buy_hold', 'buy_sell', 'reversal')
            
        Returns:
            pd.DataFrame: DataFrame with signals added
        """
        # Make sure indicators are calculated
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Create a copy to avoid modifying original data
        result = df.copy()
        
        # Initialize signal column
        result['signal'] = 0
        
        # Calculate crossover points
        result['crossover'] = np.where(
            (result['macd'] > result['macd_signal']) & 
            (result['macd'].shift(1) <= result['macd_signal'].shift(1)),
            1,  # Bullish crossover
            np.where(
                (result['macd'] < result['macd_signal']) & 
                (result['macd'].shift(1) >= result['macd_signal'].shift(1)),
                -1,  # Bearish crossover
                0  # No crossover
            )
        )
        
        # Apply strategy logic
        if strategy_type == 'buy_hold':
            # Only buy signals, never sell
            result['signal'] = result['crossover'].apply(lambda x: 1 if x == 1 else 0)
            
        elif strategy_type == 'buy_sell':
            # Buy on bullish crossover, sell on bearish crossover
            result['signal'] = result['crossover']
            
        elif strategy_type == 'reversal':
            # Buy on bullish crossover, sell and short on bearish crossover
            # Cover short on bullish crossover
            result['signal'] = result['crossover']
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return result
    
class RiskManager:
    def __init__(self, take_profit=None, stop_loss=None, trailing_stop=None):
        """
        Initialize risk management system.
        
        Args:
            take_profit (float, optional): Take profit percentage
            stop_loss (float, optional): Stop loss percentage
            trailing_stop (float, optional): Trailing stop percentage
        """
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        
        # Validate that either fixed TP/SL or trailing stop is used
        if (take_profit is not None or stop_loss is not None) and trailing_stop is not None:
            print("Warning: Both fixed TP/SL and trailing stop are set. Using trailing stop.")
            self.take_profit = None
            self.stop_loss = None
        
    def check_exit_conditions(self, position, current_candle):
        """
        Check if exit conditions are met for a position.
        
        Args:
            position (dict): Current position information
            current_candle (pd.Series): Current price candle
            
        Returns:
            tuple: (exit_triggered, exit_reason, exit_price)
        """
        if not position or position['size'] == 0:
            return False, None, None
        
        entry_price = position['entry_price']
        current_price = current_candle['c']
        position_type = position['type']  # 'long' or 'short'
        
        # Initialize variables
        exit_triggered = False
        exit_reason = None
        exit_price = None
        
        # Check for fixed take profit and stop loss
        if self.take_profit is not None or self.stop_loss is not None:
            # For long positions
            if position_type == 'long':
                # Check take profit
                if self.take_profit and current_price >= entry_price * (1 + self.take_profit):
                    exit_triggered = True
                    exit_reason = 'take_profit'
                    exit_price = entry_price * (1 + self.take_profit)
                
                # Check stop loss
                elif self.stop_loss and current_price <= entry_price * (1 - self.stop_loss):
                    exit_triggered = True
                    exit_reason = 'stop_loss'
                    exit_price = entry_price * (1 - self.stop_loss)
            
            # For short positions
            elif position_type == 'short':
                # Check take profit
                if self.take_profit and current_price <= entry_price * (1 - self.take_profit):
                    exit_triggered = True
                    exit_reason = 'take_profit'
                    exit_price = entry_price * (1 - self.take_profit)
                
                # Check stop loss
                elif self.stop_loss and current_price >= entry_price * (1 + self.stop_loss):
                    exit_triggered = True
                    exit_reason = 'stop_loss'
                    exit_price = entry_price * (1 + self.stop_loss)
        
        # Check for trailing stop
        elif self.trailing_stop is not None:
            highest_price = position.get('highest_price', entry_price)
            lowest_price = position.get('lowest_price', entry_price)
            
            # For long positions
            if position_type == 'long':
                # Update highest price if current price is higher
                if current_price > highest_price:
                    position['highest_price'] = current_price
                    highest_price = current_price
                
                # Check trailing stop
                trailing_stop_price = highest_price * (1 - self.trailing_stop)
                if current_price <= trailing_stop_price:
                    exit_triggered = True
                    exit_reason = 'trailing_stop'
                    exit_price = trailing_stop_price
            
            # For short positions
            elif position_type == 'short':
                # Update lowest price if current price is lower
                if current_price < lowest_price:
                    position['lowest_price'] = current_price
                    lowest_price = current_price
                
                # Check trailing stop
                trailing_stop_price = lowest_price * (1 + self.trailing_stop)
                if current_price >= trailing_stop_price:
                    exit_triggered = True
                    exit_reason = 'trailing_stop'
                    exit_price = trailing_stop_price
        
        return exit_triggered, exit_reason, exit_price


class BacktestEngine:
    def __init__(self, data, strategy, risk_manager, initial_capital=10000, trade_size=1.0, output_dir=None):
        """
        Initialize the backtesting engine.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and signals
            strategy: Strategy object
            risk_manager: Risk manager object
            initial_capital (float): Initial capital
            trade_size (float): Trade size as a fraction of capital (0-1)
            output_dir (str): Output directory for saving files
        """
        self.data = data
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.trade_size = trade_size
        self.output_dir = output_dir
        self.strategy_type = getattr(strategy, 'strategy_type', 'buy_sell') # Default to 'buy_sell' if not set
        
        # Initialize results storage
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
        # Initialize detailed trade tracking
        self.detailed_tracking = []
        
        # Initialize benchmark data
        self.benchmark_data = None
        
    def run_backtest(self):
        """
        Run the backtest.
        
        Returns:
            dict: Backtest results
        """
        # Make sure we have signals
        if 'signal' not in self.data.columns:
            self.data = self.strategy.generate_signals(self.data)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = None
        equity_history = []
        
        # Calculate benchmark (buy and hold from start)
        benchmark_start_price = self.data.iloc[0]['c']
        benchmark_shares = self.initial_capital / benchmark_start_price
        benchmark_history = []
        
        # Process each candle
        for i in range(1, len(self.data)):
            prev_candle = self.data.iloc[i-1]
            current_candle = self.data.iloc[i]
            
            # Calculate benchmark value
            benchmark_value = benchmark_shares * current_candle['c']
            benchmark_history.append({
                'timestamp': current_candle.name,
                'value': benchmark_value
            })
            
            # Track equity at this point
            if position and position['size'] > 0:
                # Calculate unrealized PnL
                if position['type'] == 'long':
                    unrealized_pnl = position['size'] * (current_candle['o'] - position['entry_price'])
                else:  # short
                    unrealized_pnl = position['size'] * (position['entry_price'] - current_candle['o'])
                
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
                unrealized_pnl = 0
            
            equity_history.append({
                'timestamp': current_candle.name,
                'equity': current_equity
            })
            
            # Initialize trade tracking for this candle
            candle_tracking = {
                'timestamp': current_candle.name,
                'open': current_candle['o'],
                'high': current_candle['h'],
                'low': current_candle['l'],
                'close': current_candle['c'],
                'volume': current_candle['v'],
                'ema_short': current_candle['ema_short'],
                'ema_long': current_candle['ema_long'],
                'macd': current_candle['macd'],
                'macd_signal': current_candle['macd_signal'],
                'macd_hist': current_candle['macd_hist'],
                'signal': prev_candle['signal'],  # Previous candle signal affects current action
                'cash': capital,
                'unrealized_pnl': unrealized_pnl,
                'position_type': position['type'] if position else 'none',
                'position_size': position['size'] if position else 0,
                'position_entry': position['entry_price'] if position else 0,
                'position_entry_time': position['entry_time'] if position else None,
                'highest_price': position.get('highest_price', 0) if position else 0,
                'lowest_price': position.get('lowest_price', 0) if position else 0,
                'exit_triggered': False,
                'exit_reason': None,
                'exit_price': None,
                'trailing_stop_level': 0,
                'stop_loss_level': 0,
                'take_profit_level': 0,
                'benchmark_value': benchmark_value
            }
            
            # Calculate risk levels
            if position and position['size'] > 0:
                entry_price = position['entry_price']
                
                # For long positions
                if position['type'] == 'long':
                    # Stop loss level
                    if self.risk_manager.stop_loss:
                        candle_tracking['stop_loss_level'] = entry_price * (1 - self.risk_manager.stop_loss)
                    
                    # Take profit level
                    if self.risk_manager.take_profit:
                        candle_tracking['take_profit_level'] = entry_price * (1 + self.risk_manager.take_profit)
                    
                    # Trailing stop level
                    if self.risk_manager.trailing_stop:
                        highest = position.get('highest_price', entry_price)
                        candle_tracking['trailing_stop_level'] = highest * (1 - self.risk_manager.trailing_stop)
                
                # For short positions
                elif position['type'] == 'short':
                    # Stop loss level
                    if self.risk_manager.stop_loss:
                        candle_tracking['stop_loss_level'] = entry_price * (1 + self.risk_manager.stop_loss)
                    
                    # Take profit level
                    if self.risk_manager.take_profit:
                        candle_tracking['take_profit_level'] = entry_price * (1 - self.risk_manager.take_profit)
                    
                    # Trailing stop level
                    if self.risk_manager.trailing_stop:
                        lowest = position.get('lowest_price', entry_price)
                        candle_tracking['trailing_stop_level'] = lowest * (1 + self.risk_manager.trailing_stop)
            
            # Check if we need to exit existing position based on risk management
            if position and position['size'] > 0:
                exit_triggered, exit_reason, exit_price = self.risk_manager.check_exit_conditions(
                    position, prev_candle
                )
                
                if exit_triggered:
                    # Update tracking
                    candle_tracking['exit_triggered'] = True
                    candle_tracking['exit_reason'] = exit_reason
                    candle_tracking['exit_price'] = exit_price
                    
                    # Execute exit at current candle open
                    exit_price = current_candle['o']  # Use open price for execution
                    
                    # Calculate profit/loss
                    if position['type'] == 'long':
                        pnl = position['size'] * (exit_price - position['entry_price'])
                    else:  # short
                        pnl = position['size'] * (position['entry_price'] - exit_price)
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trade = {
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_time': current_candle.name,
                        'exit_price': exit_price,
                        'size': position['size'],
                        'type': position['type'],
                        'pnl': pnl,
                        'pnl_pct': pnl / (position['size'] * position['entry_price']),
                        'exit_reason': exit_reason
                    }
                    self.trades.append(trade)
                    
                    # Clear position
                    position = None
            
            # Check for exit signal
            if position and position['size'] > 0 and (
                (position['type'] == 'long' and prev_candle['signal'] == -1) or
                (position['type'] == 'short' and prev_candle['signal'] == 1)
            ):
                # Update tracking
                candle_tracking['exit_triggered'] = True
                candle_tracking['exit_reason'] = 'signal'
                candle_tracking['exit_price'] = current_candle['o']
                
                # Execute exit at current candle open
                exit_price = current_candle['o']
                
                # Calculate profit/loss
                if position['type'] == 'long':
                    pnl = position['size'] * (exit_price - position['entry_price'])
                else:  # short
                    pnl = position['size'] * (position['entry_price'] - exit_price)
                
                # Update capital
                capital += pnl
                
                # Record trade
                trade = {
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': current_candle.name,
                    'exit_price': exit_price,
                    'size': position['size'],
                    'type': position['type'],
                    'pnl': pnl,
                    'pnl_pct': pnl / (position['size'] * position['entry_price']),
                    'exit_reason': 'signal'
                }
                self.trades.append(trade)
                
                # Clear position
                position = None
            

            if not position and prev_candle['signal'] != 0:
                # For buy_sell strategy, only enter long positions on buy signals
                if self.strategy_type == 'buy_sell' and prev_candle['signal'] == -1:
                    # Skip entering short positions for buy_sell strategy
                    pass
                else:
                    # Calculate position size
                    position_capital = capital * self.trade_size
                    position_size = position_capital / current_candle['o']
                    
                    # Update tracking
                    position_type = 'long' if prev_candle['signal'] == 1 else 'short'
                    
                    # Record position
                    position = {
                        'entry_time': current_candle.name,
                        'entry_price': current_candle['o'],
                        'size': position_size,
                        'type': position_type,
                        'highest_price': current_candle['o'],
                        'lowest_price': current_candle['o']
                    }
                    self.positions.append(position)
                    # Update tracking with new position
                    candle_tracking['position_type'] = position_type
                    candle_tracking['position_size'] = position_size
                    candle_tracking['position_entry'] = current_candle['o']
                    candle_tracking['position_entry_time'] = current_candle.name
                    candle_tracking['highest_price'] = current_candle['o']
                    candle_tracking['lowest_price'] = current_candle['o']
            
            # Save detailed tracking for this candle
            self.detailed_tracking.append(candle_tracking)



        # Close any open position at the end
        if position and position['size'] > 0:
            last_candle = self.data.iloc[-1]
            
            # Execute exit at last candle close
            exit_price = last_candle['c']
            
            # Calculate profit/loss
            if position['type'] == 'long':
                pnl = position['size'] * (exit_price - position['entry_price'])
            else:  # short
                pnl = position['size'] * (position['entry_price'] - exit_price)
            
            # Update capital
            capital += pnl
            
            # Record trade
            trade = {
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': last_candle.name,
                'exit_price': exit_price,
                'size': position['size'],
                'type': position['type'],
                'pnl': pnl,
                'pnl_pct': pnl / (position['size'] * position['entry_price']),
                'exit_reason': 'end_of_data'
            }
            self.trades.append(trade)
            
            # Update last tracking record
            self.detailed_tracking[-1]['exit_triggered'] = True
            self.detailed_tracking[-1]['exit_reason'] = 'end_of_data'
            self.detailed_tracking[-1]['exit_price'] = exit_price
        
        # Store equity curve
        self.equity_curve = pd.DataFrame(equity_history)
        
        # Store benchmark data
        self.benchmark_data = pd.DataFrame(benchmark_history)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(capital)
        
        # Save detailed tracking to CSV
        self._save_detailed_tracking()
        
        # Save performance metrics to CSV
        self._save_performance_metrics(metrics)
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'benchmark_data': self.benchmark_data,
            'metrics': metrics
        }
    
    def _save_detailed_tracking(self):
        """
        Save detailed trade tracking to CSV.
        """
        if not self.detailed_tracking or not self.output_dir:
            return
            
        # Convert to DataFrame
        tracking_df = pd.DataFrame(self.detailed_tracking)
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get symbol from data
        symbol = self.data['symbol'].iloc[0] if 'symbol' in self.data.columns else 'unknown'
        
        # Save tracking data
        detailed_tracking_file = os.path.join(self.output_dir, 
                                             f"{symbol}_detailed_tracking_{timestamp}.csv")
        tracking_df.to_csv(detailed_tracking_file, index=False)
        print(f"Detailed trade tracking saved to {detailed_tracking_file}")
    
    def _save_performance_metrics(self, metrics):
        """
        Save performance metrics to CSV.
        
        Args:
            metrics (dict): Performance metrics dictionary
        """
        if not self.output_dir:
            return
            
        # Convert metrics to DataFrame format
        metrics_data = []
        for key, value in metrics.items():
            # Handle nested dictionaries
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    metrics_data.append({
                        'metric': f"{key}_{sub_key}",
                        'value': sub_value
                    })
            else:
                metrics_data.append({
                    'metric': key,
                    'value': value
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get symbol from data
        symbol = self.data['symbol'].iloc[0] if 'symbol' in self.data.columns else 'unknown'
        
        # Save metrics
        metrics_file = os.path.join(self.output_dir, 
                                   f"{symbol}_performance_metrics_{timestamp}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Performance metrics saved to {metrics_file}")
    
    def calculate_performance_metrics(self, final_capital):
        """
        Calculate comprehensive performance metrics.
        
        Args:
            final_capital (float): Final capital
            
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Get start and end dates
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        period_delta = end_date - start_date
        
        # Calculate benchmark return (buy and hold)
        start_price = self.data.iloc[0]['c']
        end_price = self.data.iloc[-1]['c']
        benchmark_return = (end_price / start_price) - 1
        
        # If no trades, return basic metrics
        if not self.trades:
            return {
                'start': start_date,
                'end': end_date,
                'period': period_delta,
                'start_value': self.initial_capital,
                'end_value': final_capital,
                'total_return': total_return,
                'benchmark_return': benchmark_return,
                'max_gross_exposure': 0,
                'total_fees_paid': 0,
                'max_drawdown': 0,
                'max_drawdown_duration': pd.Timedelta(0),
                'total_trades': 0,
                'total_closed_trades': 0,
                'total_open_trades': 0,
                'open_trade_pnl': 0,
                'win_rate': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_winning_trade': 0,
                'avg_losing_trade': 0,
                'avg_winning_trade_duration': pd.Timedelta(0),
                'avg_losing_trade_duration': pd.Timedelta(0),
                'profit_factor': 0,
                'expectancy': 0,
                'sharpe_ratio': 0,
                'calmar_ratio': 0,
                'omega_ratio': 0,
                'sortino_ratio': 0
            }
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate trade metrics
        num_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        # Calculate best and worst trades
        best_trade_pct = winning_trades['pnl_pct'].max() * 100 if not winning_trades.empty else 0
        worst_trade_pct = losing_trades['pnl_pct'].min() * 100 if not losing_trades.empty else 0
        
        # Calculate average winning and losing trade percentages
        avg_winning_trade_pct = winning_trades['pnl_pct'].mean() * 100 if not winning_trades.empty else 0
        avg_losing_trade_pct = losing_trades['pnl_pct'].mean() * 100 if not losing_trades.empty else 0
        

        # Calculate trade durations - with error handling
        try:
            # Make sure entry_time and exit_time are datetime objects
            if not pd.api.types.is_datetime64_dtype(trades_df['entry_time']):
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            
            if not pd.api.types.is_datetime64_dtype(trades_df['exit_time']):
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Calculate duration and add as a new column
            trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
            
            # Now create winning and losing trade subsets WITH EXPLICIT COPIES
            winning_trades = trades_df[trades_df['pnl'] > 0].copy()
            losing_trades = trades_df[trades_df['pnl'] <= 0].copy()
            
            # Calculate average durations for winning and losing trades
            avg_winning_duration = winning_trades['duration'].mean() if not winning_trades.empty else pd.Timedelta(0)
            avg_losing_duration = losing_trades['duration'].mean() if not losing_trades.empty else pd.Timedelta(0)
        except Exception as e:
            import traceback
            print(f"Warning: Could not calculate trade durations: {e}")
            print(traceback.format_exc())  # Add this to get full error details
            # Set default values if duration calculation fails
            avg_winning_duration = pd.Timedelta(0)
            avg_losing_duration = pd.Timedelta(0)
        
        # Profit factor (sum of wins / sum of losses)
        total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate expectancy
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Calculate drawdown
        if not self.equity_curve.empty:
            equity = self.equity_curve['equity']
            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
            
            # Calculate drawdown duration
            is_drawdown = (equity < running_max)
            drawdown_periods = []
            current_period = None
            
            for i, in_drawdown in enumerate(is_drawdown):
                if in_drawdown:
                    if current_period is None:
                        current_period = {'start': i}
                else:
                    if current_period is not None:
                        current_period['end'] = i - 1
                        drawdown_periods.append(current_period)
                        current_period = None
            
            # Handle case where still in drawdown at the end
            if current_period is not None:
                current_period['end'] = len(is_drawdown) - 1
                drawdown_periods.append(current_period)
            
            # Calculate duration of each drawdown period
            max_duration = pd.Timedelta(0)
            for period in drawdown_periods:
                start_time = self.equity_curve.iloc[period['start']]['timestamp']
                end_time = self.equity_curve.iloc[period['end']]['timestamp']
                duration = end_time - start_time
                max_duration = max(max_duration, duration)
        else:
            max_drawdown = 0
            max_duration = pd.Timedelta(0)
        
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            equity = self.equity_curve['equity']
            returns = equity.pct_change().dropna()
            
            # Annualize based on number of periods in a year
            periods_per_day = 24 * 6  # 10-minute candles per day
            annualization_factor = np.sqrt(365 * periods_per_day)
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            sharpe_ratio = (mean_return / std_return) * annualization_factor if std_return > 0 else 0
            
            # Calculate Sortino ratio (using downside deviation)
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std()
            sortino_ratio = (mean_return / downside_deviation) * annualization_factor if downside_deviation > 0 else 0
            
            # Calculate Calmar ratio
            years = period_delta.days / 365 if hasattr(period_delta, 'days') else 1
            annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Calculate Omega ratio
            threshold = 0  # Use 0 as threshold
            omega_numerator = returns[returns > threshold].sum()
            omega_denominator = abs(returns[returns < threshold].sum())
            omega_ratio = omega_numerator / omega_denominator if omega_denominator > 0 else float('inf')
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
            omega_ratio = 0
        
        # Calculate annualized return
        years = period_delta.days / 365 if hasattr(period_delta, 'days') else 1
        annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
        
        # Calculate max gross exposure
        max_exposure = trades_df['size'].max() * trades_df['entry_price'].max() / self.initial_capital if not trades_df.empty else 0
        
        # Count open trades
        open_positions_count = len(self.positions) - num_trades
        open_positions_pnl = 0
        
        return {
            'start': start_date,
            'end': end_date,
            'period': period,
            'start_value': self.initial_capital,
            'end_value': final_capital,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'max_gross_exposure': max_exposure,
            'total_fees_paid': 0,  # No fees in this backtest
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_duration,
            'total_trades': num_trades,
            'total_closed_trades': num_trades,
            'total_open_trades': open_positions_count,
            'open_trade_pnl': open_positions_pnl,
            'win_rate': win_rate,
            'best_trade': best_trade_pct,
            'worst_trade': worst_trade_pct,
            'avg_winning_trade': avg_winning_trade_pct,
            'avg_losing_trade': avg_losing_trade_pct,
            'avg_winning_trade_duration': avg_winning_duration,
            'avg_losing_trade_duration': avg_losing_duration,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'sortino_ratio': sortino_ratio,
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict()
        }
    

class BacktestVisualizer:
    def __init__(self, data, backtest_results, output_dir=None):
        """
        Initialize the visualizer.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and indicators
            backtest_results (dict): Results from the backtest
            output_dir (str): Output directory for saving files
        """
        self.data = data
        self.results = backtest_results
        self.output_dir = output_dir
        
    def plot_price_and_signals(self):
        """
        Plot price chart with buy/sell signals - optimized version.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Sample the data to reduce points if too large
        if len(self.data) > 10000:
            # Downsample data for better performance
            data_sample = self.data.iloc[::int(len(self.data)/5000)]
        else:
            data_sample = self.data
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=('Price Chart', 'MACD'))
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data_sample.index,
                open=data_sample['o'],
                high=data_sample['h'],
                low=data_sample['l'],
                close=data_sample['c'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Add EMAs
        fig.add_trace(
            go.Scatter(
                x=data_sample.index,
                y=data_sample['ema_short'],
                name=f"Short EMA ({data_sample['ema_short'].name})",
                line=dict(color='rgba(33, 150, 243, 0.7)', width=1.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sample.index,
                y=data_sample['ema_long'],
                name=f"Long EMA ({data_sample['ema_long'].name})",
                line=dict(color='rgba(255, 152, 0, 0.7)', width=1.5)
            ),
            row=1, col=1
        )
        
        # Add MACD
        fig.add_trace(
            go.Scatter(
                x=data_sample.index,
                y=data_sample['macd'],
                name='MACD Line',
                line=dict(color='blue', width=1.5)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sample.index,
                y=data_sample['macd_signal'],
                name='Signal Line',
                line=dict(color='red', width=1.5)
            ),
            row=2, col=1
        )
        
        # Create custom histograms for MACD
        colors = np.where(data_sample['macd_hist'] >= 0, 'green', 'red')
        
        fig.add_trace(
            go.Bar(
                x=data_sample.index,
                y=data_sample['macd_hist'],
                name='MACD Histogram',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Find trade points efficiently
        trades_df = pd.DataFrame(self.results['trades'])
        if not trades_df.empty:
            # Convert string dates to datetime if necessary
            if isinstance(trades_df['entry_time'].iloc[0], str):
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Plot entry points
            for i, trade in trades_df.iterrows():
                color = 'green' if trade['type'] == 'long' else 'red'
                
                # Add entry markers
                fig.add_trace(
                    go.Scatter(
                        x=[trade['entry_time']],
                        y=[trade['entry_price']],
                        mode='markers',
                        marker=dict(size=10, color=color, symbol='circle'),
                        name=f"Entry ({trade['type']})",
                        showlegend=i==0  # Only show the first one in the legend
                    ),
                    row=1, col=1
                )
                
                # Add exit markers
                exit_color = {
                    'signal': 'blue',
                    'take_profit': 'green',
                    'stop_loss': 'red',
                    'trailing_stop': 'purple',
                    'end_of_data': 'black'
                }.get(trade['exit_reason'], 'gray')
                
                fig.add_trace(
                    go.Scatter(
                        x=[trade['exit_time']],
                        y=[trade['exit_price']],
                        mode='markers',
                        marker=dict(size=10, color=exit_color, symbol='x'),
                        name=f"Exit ({trade['exit_reason']})",
                        showlegend=i==0  # Only show the first one in the legend
                    ),
                    row=1, col=1
                )
        
        # Update chart layout
        fig.update_layout(
            title='OHLC Chart with MACD Strategy',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=800,
            width=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def plot_order_signals(self):
        """
        Plot order signals with OHLCV price vs date.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['c'],
                name='Close Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add buy signals
        buy_signals = self.data[self.data['signal'] == 1]
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['c'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(width=1, color='black')
                ),
                name='Buy Signal'
            )
        )
        
        # Add sell signals
        sell_signals = self.data[self.data['signal'] == -1]
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['c'],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=1, color='black')
                ),
                name='Sell Signal'
            )
        )
        
        # Add trade entries and exits
        trades_df = pd.DataFrame(self.results['trades'])
        if not trades_df.empty:
            # Plot entries
            entries = pd.DataFrame({
                'time': pd.to_datetime(trades_df['entry_time']),
                'price': trades_df['entry_price'],
                'type': trades_df['type']
            })
            
            long_entries = entries[entries['type'] == 'long']
            short_entries = entries[entries['type'] == 'short']
            
            fig.add_trace(
                go.Scatter(
                    x=long_entries['time'],
                    y=long_entries['price'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='green',
                        line=dict(width=1, color='black')
                    ),
                    name='Long Entry'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=short_entries['time'],
                    y=short_entries['price'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='red',
                        line=dict(width=1, color='black')
                    ),
                    name='Short Entry'
                )
            )
            
            # Plot exits
            exits = pd.DataFrame({
                'time': pd.to_datetime(trades_df['exit_time']),
                'price': trades_df['exit_price'],
                'reason': trades_df['exit_reason']
            })
            
            for reason in exits['reason'].unique():
                reason_exits = exits[exits['reason'] == reason]
                
                fig.add_trace(
                    go.Scatter(
                        x=reason_exits['time'],
                        y=reason_exits['price'],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=10,
                            color={
                                'signal': 'blue',
                                'take_profit': 'green',
                                'stop_loss': 'red',
                                'trailing_stop': 'purple',
                                'end_of_data': 'black'
                            }.get(reason, 'gray'),
                            line=dict(width=1, color='black')
                        ),
                        name=f'Exit ({reason})'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title='Orders',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600,
            width=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def plot_trade_returns(self):
        """
        Plot trade returns vs date.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Create DataFrame for trades
        trades_df = pd.DataFrame(self.results['trades'])
        if trades_df.empty:
            # Create empty figure if no trades
            fig = go.Figure()
            fig.update_layout(
                title='Trade Returns',
                xaxis_title='Date',
                yaxis_title='Trade Returns',
                height=600,
                width=1200
            )
            return fig
        
        # Convert timestamps to datetime if they are strings
        if isinstance(trades_df['exit_time'].iloc[0], str):
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Calculate return percentage for display
        trades_df['pnl_pct_display'] = trades_df['pnl_pct'] * 100
        
        # Create figure
        fig = go.Figure()
        
        # Add winning trades
        winning_trades = trades_df[trades_df['pnl'] > 0]
        if not winning_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=winning_trades['exit_time'],
                    y=winning_trades['pnl_pct_display'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='green',
                        line=dict(width=1, color='black')
                    ),
                    name='Profitable Trade',
                    text=[f"Entry: {row['entry_time']}<br>Exit: {row['exit_time']}<br>Type: {row['type']}<br>Return: {row['pnl_pct_display']:.2f}%<br>Reason: {row['exit_reason']}" 
                          for _, row in winning_trades.iterrows()],
                    hoverinfo='text'
                )
            )
        
        # Add losing trades
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        if not losing_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=losing_trades['exit_time'],
                    y=losing_trades['pnl_pct_display'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='red',
                        line=dict(width=1, color='black')
                    ),
                    name='Losing Trade',
                    text=[f"Entry: {row['entry_time']}<br>Exit: {row['exit_time']}<br>Type: {row['type']}<br>Return: {row['pnl_pct_display']:.2f}%<br>Reason: {row['exit_reason']}" 
                          for _, row in losing_trades.iterrows()],
                    hoverinfo='text'
                )
            )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[trades_df['exit_time'].min(), trades_df['exit_time'].max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                name='Break Even'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Trade Returns',
            xaxis_title='Date',
            yaxis_title='Trade Returns (%)',
            height=600,
            width=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def plot_cumulative_returns(self):
        """
        Plot cumulative returns and benchmark return vs date.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Create figure
        fig = go.Figure()
        
        # Add strategy equity curve
        fig.add_trace(
            go.Scatter(
                x=self.results['equity_curve']['timestamp'],
                y=self.results['equity_curve']['equity'],
                name='Strategy',
                line=dict(color='rgb(75, 192, 192)', width=2),
                fill='tozeroy',
                fillcolor='rgba(75, 192, 192, 0.2)'
            )
        )
        
        # Add benchmark return
        if 'benchmark_data' in self.results and not self.results['benchmark_data'].empty:
            fig.add_trace(
                go.Scatter(
                    x=self.results['benchmark_data']['timestamp'],
                    y=self.results['benchmark_data']['value'],
                    name='Benchmark (Buy & Hold)',
                    line=dict(color='rgb(153, 102, 255)', width=2)
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative returns',
            height=600,
            width=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def plot_equity_curve(self):
        """
        Plot equity curve and drawdown.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Create subplots
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            subplot_titles=('Equity Curve', 'Drawdown'),
                            row_heights=[0.7, 0.3])
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=self.results['equity_curve']['timestamp'],
                y=self.results['equity_curve']['equity'],
                name='Equity',
                line=dict(color='rgb(75, 192, 192)', width=2),
                fill='tozeroy',
                fillcolor='rgba(75, 192, 192, 0.2)'
            ),
            row=1, col=1
        )
        
        # Calculate drawdown
        equity = self.results['equity_curve']['equity']
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100  # Convert to percentage
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                x=self.results['equity_curve']['timestamp'],
                y=drawdown,
                name='Drawdown',
                line=dict(color='rgba(255, 99, 132, 1)', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 99, 132, 0.2)'
            ),
            row=2, col=1
        )
        
        # Add horizontal line at zero for drawdown
        fig.add_trace(
            go.Scatter(
                x=[self.results['equity_curve']['timestamp'].iloc[0], 
                   self.results['equity_curve']['timestamp'].iloc[-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Equity Curve and Drawdown',
            xaxis2_title='Date',
            yaxis_title='Equity',
            yaxis2_title='Drawdown (%)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600,
            width=1200,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update y-axis for drawdown
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def create_summary_table(self):
        """
        Create a summary table of backtest metrics.
        
        Returns:
            pd.DataFrame: Summary table
        """
        metrics = self.results['metrics']
        
        # Format metrics for display
        summary = {
            'Metric': [
                'Start',
                'End',
                'Period',
                'Start Value',
                'End Value',
                'Total Return [%]',
                'Benchmark Return [%]',
                'Max Gross Exposure [%]',
                'Total Fees Paid',
                'Max Drawdown [%]',
                'Max Drawdown Duration',
                'Total Trades',
                'Total Closed Trades',
                'Total Open Trades',
                'Open Trade PnL',
                'Win Rate [%]',
                'Best Trade [%]',
                'Worst Trade [%]',
                'Avg Winning Trade [%]',
                'Avg Losing Trade [%]',
                'Avg Winning Trade Duration',
                'Avg Losing Trade Duration',
                'Profit Factor',
                'Expectancy',
                'Sharpe Ratio',
                'Calmar Ratio',
                'Omega Ratio',
                'Sortino Ratio'
            ],
            'Value': [
                metrics['start'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(metrics['start'], pd.Timestamp) else str(metrics['start']),
                metrics['end'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(metrics['end'], pd.Timestamp) else str(metrics['end']),
                str(metrics['period']),
                f"{metrics['start_value']:.6f}",
                f"{metrics['end_value']:.6f}",
                f"{metrics['total_return']*100:.6f}",
                f"{metrics['benchmark_return']*100:.6f}",
                f"{metrics['max_gross_exposure']*100:.6f}",
                f"{metrics['total_fees_paid']:.6f}",
                f"{metrics['max_drawdown']*100:.6f}",
                str(metrics['max_drawdown_duration']),
                f"{metrics['total_trades']}",
                f"{metrics['total_closed_trades']}",
                f"{metrics['total_open_trades']}",
                f"{metrics['open_trade_pnl']:.6f}",
                f"{metrics['win_rate']*100:.6f}",
                f"{metrics['best_trade']:.6f}",
                f"{metrics['worst_trade']:.6f}",
                f"{metrics['avg_winning_trade']:.6f}",
                f"{metrics['avg_losing_trade']:.6f}",
                str(metrics['avg_winning_trade_duration']),
                str(metrics['avg_losing_trade_duration']),
                f"{metrics['profit_factor']:.6f}",
                f"{metrics['expectancy']:.6f}",
                f"{metrics['sharpe_ratio']:.6f}",
                f"{metrics['calmar_ratio']:.6f}",
                f"{metrics['omega_ratio']:.6f}",
                f"{metrics['sortino_ratio']:.6f}"
            ]
        }
        
        return pd.DataFrame(summary)
    
    def save_trade_log(self, symbol, timeframe):
        """
        Save trade log to CSV.
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe
        """
        if not self.output_dir:
            return
            
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.results['trades'])
        
        # Save to CSV
        trades_file = os.path.join(self.output_dir, f"{symbol}_{timeframe}_macd_trades_{timestamp}.csv")
        trades_df.to_csv(trades_file, index=False)
        
        print(f"Trade log saved to {trades_file}")
    
    def save_performance_summary(self, symbol, timeframe):
        """
        Save performance summary to CSV.
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe
        """
        if not self.output_dir:
            return
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get summary table
        summary_df = self.create_summary_table()
        
        # Save to CSV
        summary_file = os.path.join(self.output_dir, f"{symbol}_{timeframe}_performance_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Performance summary saved to {summary_file}")


def run_macd_backtest(
    symbol='btcusd', 
    timeframe='10m',
    short_window=12,
    long_window=26,
    signal_window=9,
    strategy_type='buy_sell',
    take_profit=None,
    stop_loss=None,
    trailing_stop=0.02,
    initial_capital=10000,
    trade_size=0.1
):
    """
    Run a complete MACD backtest.
    
    Args:
        symbol (str): Symbol to test (e.g., 'btcusd', 'ethusd')
        timeframe (str): Timeframe to test
        short_window (int): Short EMA window
        long_window (int): Long EMA window
        signal_window (int): Signal EMA window
        strategy_type (str): Strategy type ('buy_hold', 'buy_sell', 'reversal')
        take_profit (float, optional): Take profit percentage
        stop_loss (float, optional): Stop loss percentage
        trailing_stop (float, optional): Trailing stop percentage
        initial_capital (float): Initial capital
        trade_size (float): Trade size as fraction of capital
        
    Returns:
        dict: Complete backtest results
    """
    print(f"Running MACD backtest for {symbol.upper()} on {timeframe} timeframe")
    print(f"MACD Parameters: Short={short_window}, Long={long_window}, Signal={signal_window}")
    
    # 1. Load and prepare data
    data_handler = DataHandler()
    data = data_handler.load_data(symbol, timeframe, strategy_type)
    
    # Get output directories
    output_base_dir, output_strategy_dir = data_handler.output_base_dir, data_handler.output_strategy_dir
    
    # Check for uniform time intervals and resample if needed
    if timeframe == '10m':
        resampling_timeframe = '10T'  # 10 minutes
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    data = data_handler.resample_data(data, resampling_timeframe)
    
    # 2. Initialize strategy
    strategy = MACDStrategy(short_window, long_window, signal_window)
    strategy.strategy_type = strategy_type
    data = strategy.calculate_indicators(data)
    data = strategy.generate_signals(data, strategy_type)
    
    # 3. Initialize risk management
    risk_manager = RiskManager(take_profit, stop_loss, trailing_stop)
    
    # 4. Run backtest
    backtest_engine = BacktestEngine(data, strategy, risk_manager, initial_capital, trade_size, output_strategy_dir)
    results = backtest_engine.run_backtest()
    
    # 5. Visualize results
    visualizer = BacktestVisualizer(data, results, output_strategy_dir)
    
    price_chart = visualizer.plot_price_and_signals()
    equity_chart = visualizer.plot_equity_curve()
    order_chart = visualizer.plot_order_signals()
    trade_returns_chart = visualizer.plot_trade_returns()
    cumulative_returns_chart = visualizer.plot_cumulative_returns()
    summary_table = visualizer.create_summary_table()
    
    # Save trade log and performance summary
    visualizer.save_trade_log(symbol, timeframe)
    visualizer.save_performance_summary(symbol, timeframe)
    
    # Print summary
    print("\nBacktest Summary:")
    print(summary_table.to_string(index=False))
    
    return {
        'data': data,
        'results': results,
        'price_chart': price_chart,
        'equity_chart': equity_chart,
        'order_chart': order_chart,
        'trade_returns_chart': trade_returns_chart,
        'cumulative_returns_chart': cumulative_returns_chart,
        'summary_table': summary_table
    }

# # Example usage
# if __name__ == "__main__":
#     # Run backtest for BTC with default parameters
#     btc_results = run_macd_backtest(
#         symbol='btcusd',
#         timeframe='10m',
#         short_window=12,
#         long_window=26,
#         signal_window=9,
#         strategy_type='reversal',
#         trailing_stop=0.02,
#         initial_capital=10000,
#         trade_size=1
#     )
    
#     # Display charts
#     btc_results['price_chart'].show()
#     btc_results['equity_chart'].show()
#     btc_results['order_chart'].show()
#     btc_results['trade_returns_chart'].show()
#     btc_results['cumulative_returns_chart'].show()

#     btc_results = run_macd_backtest(
#         symbol='btcusd',
#         timeframe='10m',
#         short_window=12,
#         long_window=26,
#         signal_window=9,
#         strategy_type='buy_sell',
#         trailing_stop=0.02,
#         initial_capital=10000,
#         trade_size=1
#     )
    
#     # Display charts
#     btc_results['price_chart'].show()
#     btc_results['equity_chart'].show()
#     btc_results['order_chart'].show()
#     btc_results['trade_returns_chart'].show()
#     btc_results['cumulative_returns_chart'].show()

#     btc_results = run_macd_backtest(
#         symbol='btcusd',
#         timeframe='10m',
#         short_window=12,
#         long_window=26,
#         signal_window=9,
#         strategy_type='buy_hold',
#         # take_profit=0.05,
#         # stop_loss=0.03,
#         trailing_stop=0.02,
#         initial_capital=10000,
#         trade_size=1
#     )
    
#     # Display charts
#     btc_results['price_chart'].show()
#     btc_results['equity_chart'].show()
#     btc_results['order_chart'].show()
#     btc_results['trade_returns_chart'].show()
#     btc_results['cumulative_returns_chart'].show()
    
#     # # Run backtest for ETH with custom parameters
#     # eth_results = run_macd_backtest(
#     #     symbol='ethusd',
#     #     timeframe='10m',
#     #     short_window=8,
#     #     long_window=21,
#     #     signal_window=5,
#     #     strategy_type='reversal',
#     #     take_profit=0.05,
#     #     stop_loss=0.03,
#     #     initial_capital=10000,
#     #     trade_size=1
#     # )
    
#     # # Display charts
#     # eth_results['price_chart'].show()
#     # eth_results['equity_chart'].show()
#     # eth_results['order_chart'].show()
#     # eth_results['trade_returns_chart'].show()
#     # eth_results['cumulative_returns_chart'].show()
    
#     # # Compare results
#     # print("\nComparison of BTC vs ETH:")
#     # btc_metrics = btc_results['results']['metrics']
#     # eth_metrics = eth_results['results']['metrics']
    
#     # comparison = pd.DataFrame({
#     #     'Metric': [
#     #         'Total Return',
#     #         'Annualized Return',
#     #         'Total Trades',
#     #         'Win Rate',
#     #         'Profit Factor',
#     #         'Max Drawdown',
#     #         'Sharpe Ratio',
#     #         'Sortino Ratio',
#     #         'Calmar Ratio',
#     #         'Omega Ratio'
#     #     ],
#     #     'BTC': [
#     #         f"{btc_metrics['total_return']*100:.2f}%",
#     #         f"{btc_metrics.get('annualized_return', 0)*100:.2f}%",
#     #         f"{btc_metrics['total_trades']}",
#     #         f"{btc_metrics['win_rate']*100:.2f}%",
#     #         f"{btc_metrics['profit_factor']:.2f}",
#     #         f"{btc_metrics['max_drawdown']*100:.2f}%",
#     #         f"{btc_metrics['sharpe_ratio']:.2f}",
#     #         f"{btc_metrics['sortino_ratio']:.2f}",
#     #         f"{btc_metrics['calmar_ratio']:.2f}",
#     #         f"{btc_metrics['omega_ratio']:.2f}"
#     #     ],
#     #     'ETH': [
#     #         f"{eth_metrics['total_return']*100:.2f}%",
#     #         f"{eth_metrics.get('annualized_return', 0)*100:.2f}%",
#     #         f"{eth_metrics['total_trades']}",
#     #         f"{eth_metrics['win_rate']*100:.2f}%",
#     #         f"{eth_metrics['profit_factor']:.2f}",
#     #         f"{eth_metrics['max_drawdown']*100:.2f}%",
#     #         f"{eth_metrics['sharpe_ratio']:.2f}",
#     #         f"{eth_metrics['sortino_ratio']:.2f}",
#     #         f"{eth_metrics['calmar_ratio']:.2f}",
#     #         f"{eth_metrics['omega_ratio']:.2f}"
#     #     ]
#     # })
    
#     # print(comparison.to_string(index=False))





class WalkForwardTester:
    """
    Class for performing walk-forward testing of trading strategies.
    This avoids overfitting by training on in-sample data and testing on out-of-sample data.
    """
    
    def __init__(self, data_handler, full_data=None, symbol='btcusd', timeframe='10m', 
                 base_output_dir=None):
        """
        Initialize the walk-forward tester.
        
        Args:
            data_handler (DataHandler): Data handler object
            full_data (pd.DataFrame, optional): Full dataset if already loaded
            symbol (str): Symbol to test
            timeframe (str): Timeframe to test
            base_output_dir (str, optional): Base output directory
        """
        self.data_handler = data_handler
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Load data if not provided
        if full_data is None:
            print(f"Loading data for {symbol} on {timeframe} timeframe...")
            self.full_data = data_handler.load_data(symbol, timeframe, 'walk_forward')
            
            # Resample to ensure uniform intervals
            if timeframe == '10m':
                resampling_timeframe = '10T'
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
                
            self.full_data = data_handler.resample_data(self.full_data, resampling_timeframe)
        else:
            self.full_data = full_data
        
        # Set up output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_output_dir is None:
            self.base_output_dir = os.path.join('output', f"{symbol}_{timeframe}_walkforward_{timestamp}")
        else:
            self.base_output_dir = os.path.join(base_output_dir, f"{symbol}_{timeframe}_walkforward_{timestamp}")
        
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # Store results
        self.results = {
            'period_results': [],
            'combined_equity_curve': None,
            'optimized_parameters': {},
            'final_metrics': {}
        }
        
    def split_data_by_year(self):
        """
        Split the full dataset into yearly periods.
        
        Returns:
            list: List of tuples (year, start_date, end_date, data)
        """
        # Extract year from index
        self.full_data['year'] = self.full_data.index.year
        
        # Get unique years
        years = sorted(self.full_data['year'].unique())
        
        # Split data by year
        yearly_data = []
        for year in years:
            year_data = self.full_data[self.full_data['year'] == year].copy()
            start_date = year_data.index.min()
            end_date = year_data.index.max()
            
            yearly_data.append((year, start_date, end_date, year_data))
        
        # Remove the year column
        self.full_data.drop('year', axis=1, inplace=True)
        
        return yearly_data
    
    def optimize_parameters(self, train_data, parameter_grid):
        """
        Optimize strategy parameters using grid search on the training data.
        
        Args:
            train_data (pd.DataFrame): Training data
            parameter_grid (dict): Grid of parameters to search
            
        Returns:
            tuple: (best_params, best_metrics)
        """
        print(f"Optimizing parameters on training data ({len(train_data)} rows)...")
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            parameter_grid['short_window'],
            parameter_grid['long_window'],
            parameter_grid['signal_window'],
            parameter_grid['strategy_type'],
            parameter_grid['trailing_stop']
        ))
        
        # Filter invalid combinations (short_window must be < long_window)
        valid_combinations = [
            (short, long, signal, strategy, trailing_stop)
            for short, long, signal, strategy, trailing_stop in param_combinations
            if short < long
        ]
        
        # Create a directory for optimization results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        opt_dir = os.path.join(self.base_output_dir, f"optimization_{timestamp}")
        os.makedirs(opt_dir, exist_ok=True)
        
        # Run backtests with multithreading
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_params = {
                executor.submit(
                    self._run_backtest_for_optimization,
                    train_data.copy(),
                    short_window,
                    long_window,
                    signal_window,
                    strategy_type,
                    trailing_stop,
                    opt_dir
                ): (short_window, long_window, signal_window, strategy_type, trailing_stop)
                for short_window, long_window, signal_window, strategy_type, trailing_stop in valid_combinations
            }
            
            # Use tqdm for progress tracking
            for future in tqdm(concurrent.futures.as_completed(future_to_params), 
                              total=len(future_to_params),
                              desc="Testing parameter combinations"):
                params = future_to_params[future]
                try:
                    metrics = future.result()
                    results.append((params, metrics))
                except Exception as e:
                    print(f"Error with parameters {params}: {str(e)}")
        
        # Find best parameters based on Sharpe ratio
        if not results:
            raise ValueError("No valid parameter combinations found")
        
        # Sort by Sharpe ratio (or other chosen metric)
        results.sort(key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        # Save all optimization results to CSV
        opt_results_df = pd.DataFrame([
            {
                'short_window': params[0],
                'long_window': params[1],
                'signal_window': params[2],
                'strategy_type': params[3],
                'trailing_stop': params[4],
                'total_return': metrics['total_return'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'max_drawdown': metrics['max_drawdown'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio'],
                'calmar_ratio': metrics['calmar_ratio'],
                'omega_ratio': metrics['omega_ratio'],
                'total_trades': metrics['total_trades']
            }
            for params, metrics in results
        ])
        
        opt_results_file = os.path.join(opt_dir, f"optimization_results.csv")
        opt_results_df.to_csv(opt_results_file, index=False)
        
        # Return best parameters and metrics
        best_params, best_metrics = results[0]
        
        print(f"Best parameters found: {best_params} with Sharpe ratio: {best_metrics['sharpe_ratio']:.4f}")
        
        return best_params, best_metrics
    
    def _run_backtest_for_optimization(self, data, short_window, long_window, signal_window, 
                                      strategy_type, trailing_stop, output_dir):
        """
        Run a single backtest during the optimization process.
        
        Args:
            data (pd.DataFrame): Data for backtest
            short_window (int): Short EMA window
            long_window (int): Long EMA window
            signal_window (int): Signal EMA window
            strategy_type (str): Strategy type
            trailing_stop (float): Trailing stop percentage
            output_dir (str): Output directory
            
        Returns:
            dict: Performance metrics
        """
        # Create strategy
        strategy = MACDStrategy(short_window, long_window, signal_window)
        strategy.strategy_type = strategy_type
        
        # Calculate indicators and signals
        data = strategy.calculate_indicators(data)
        data = strategy.generate_signals(data, strategy_type)
        
        # Create risk manager
        risk_manager = RiskManager(take_profit=None, stop_loss=None, trailing_stop=trailing_stop)
        
        # Run backtest
        backtest_engine = BacktestEngine(
            data, 
            strategy, 
            risk_manager, 
            initial_capital=10000, 
            trade_size=1.0, 
            output_dir=None  # Don't save files during optimization
        )
        
        results = backtest_engine.run_backtest()
        
        # Return just the metrics
        return results['metrics']
    
    def run_walk_forward_test(self, parameter_grid, training_years=1, testing_months=3, 
                            step_months=3, return_type='sharpe'):
        """
        Run walk-forward testing using the specified parameters.
        
        Args:
            parameter_grid (dict): Grid of parameters to search
            training_years (int): Number of years to use for training
            testing_months (int): Number of months to use for testing
            step_months (int): Number of months to step forward
            return_type (str): Metric to use for parameter selection
            
        Returns:
            dict: Walk-forward test results
        """
        print(f"Starting walk-forward testing for {self.symbol} on {self.timeframe} timeframe...")
        
        # Create timestamp for output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.base_output_dir, f"walkforward_results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Get full date range
        start_date = self.full_data.index.min()
        end_date = self.full_data.index.max()
        
        # Calculate time windows
        training_period = pd.DateOffset(years=training_years)
        testing_period = pd.DateOffset(months=testing_months)
        step_period = pd.DateOffset(months=step_months)
        
        # Initialize first window
        train_start = start_date
        train_end = train_start + training_period
        test_start = train_end
        test_end = test_start + testing_period
        
        # Store all equity curves and performance metrics
        all_equity_curves = []
        all_period_results = []
        
        # Store test period data for plotting
        test_periods = []
        
        # Counter for naming
        period_count = 1
        
        # Walk forward through the data
        while test_end <= end_date:
            print(f"\nPeriod {period_count}:")
            print(f"Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"Testing: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            # Get training and testing data
            train_data = self.full_data[
                (self.full_data.index >= train_start) & 
                (self.full_data.index < train_end)
            ].copy()
            
            test_data = self.full_data[
                (self.full_data.index >= test_start) & 
                (self.full_data.index < test_end)
            ].copy()
            
            if len(train_data) < 100 or len(test_data) < 20:
                print(f"Insufficient data for period {period_count}. Skipping.")
                
                # Move window forward
                train_start = train_start + step_period
                train_end = train_start + training_period
                test_start = train_end
                test_end = test_start + testing_period
                period_count += 1
                continue
            
            # Create directory for this period
            period_dir = os.path.join(results_dir, f"period_{period_count}")
            os.makedirs(period_dir, exist_ok=True)
            
            try:
                # Optimize parameters on training data
                best_params, best_metrics = self.optimize_parameters(train_data, parameter_grid)
                
                # Extract parameters
                short_window, long_window, signal_window, strategy_type, trailing_stop = best_params
                
                # Run backtest on testing data with best parameters
                test_results = self.run_test_with_parameters(
                    test_data,
                    short_window,
                    long_window,
                    signal_window,
                    strategy_type,
                    trailing_stop,
                    period_dir
                )
                
                # Store results
                period_result = {
                    'period': period_count,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'best_params': best_params,
                    'train_metrics': best_metrics,
                    'test_metrics': test_results['results']['metrics'],
                    'test_equity_curve': test_results['results']['equity_curve'],
                    'test_trades': test_results['results']['trades']
                }
                
                all_period_results.append(period_result)
                
                # Add equity curve to collection
                test_equity = test_results['results']['equity_curve'].copy()
                test_equity['period'] = period_count
                all_equity_curves.append(test_equity)
                
                # Store test period
                test_periods.append((test_start, test_end, period_count))
                
                # Save period results
                period_summary = pd.DataFrame([{
                    'period': period_count,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'short_window': short_window,
                    'long_window': long_window,
                    'signal_window': signal_window,
                    'strategy_type': strategy_type,
                    'trailing_stop': trailing_stop,
                    'train_return': best_metrics['total_return'],
                    'train_sharpe': best_metrics['sharpe_ratio'],
                    'test_return': test_results['results']['metrics']['total_return'],
                    'test_sharpe': test_results['results']['metrics']['sharpe_ratio'],
                    'test_win_rate': test_results['results']['metrics']['win_rate'],
                    'test_max_drawdown': test_results['results']['metrics']['max_drawdown'],
                    'test_trades': test_results['results']['metrics']['total_trades']
                }])
                
                period_summary_file = os.path.join(period_dir, f"period_{period_count}_summary.csv")
                period_summary.to_csv(period_summary_file, index=False)
                
            except Exception as e:
                print(f"Error in period {period_count}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Move window forward
            train_start = train_start + step_period
            train_end = train_start + training_period
            test_start = train_end
            test_end = test_start + testing_period
            period_count += 1
        
        # Combine all equity curves
        if all_equity_curves:
            combined_equity = pd.concat(all_equity_curves)
            
            # Calculate combined metrics
            combined_metrics = self.calculate_combined_metrics(combined_equity, all_period_results)
            
            # Create combined equity curve visualization
            combined_chart = self.plot_combined_equity_curve(combined_equity, test_periods)
            
            # Save combined equity curve
            combined_equity_file = os.path.join(results_dir, "combined_equity_curve.csv")
            combined_equity.to_csv(combined_equity_file)
            
            # Save combined metrics
            combined_metrics_df = pd.DataFrame([combined_metrics])
            combined_metrics_file = os.path.join(results_dir, "combined_metrics.csv")
            combined_metrics_df.to_csv(combined_metrics_file, index=False)
            
            # Save period summaries
            period_summaries = []
            for period_result in all_period_results:
                period_summaries.append({
                    'period': period_result['period'],
                    'train_start': period_result['train_start'],
                    'train_end': period_result['train_end'],
                    'test_start': period_result['test_start'],
                    'test_end': period_result['test_end'],
                    'best_params': str(period_result['best_params']),
                    'train_return': period_result['train_metrics']['total_return'],
                    'test_return': period_result['test_metrics']['total_return'],
                    'train_sharpe': period_result['train_metrics']['sharpe_ratio'],
                    'test_sharpe': period_result['test_metrics']['sharpe_ratio'],
                    'test_trades': period_result['test_metrics']['total_trades'],
                    'test_win_rate': period_result['test_metrics']['win_rate'],
                    'test_max_drawdown': period_result['test_metrics']['max_drawdown']
                })
            
            periods_df = pd.DataFrame(period_summaries)
            periods_file = os.path.join(results_dir, "all_periods_summary.csv")
            periods_df.to_csv(periods_file, index=False)
            
            # Store results
            self.results['period_results'] = all_period_results
            self.results['combined_equity_curve'] = combined_equity
            self.results['final_metrics'] = combined_metrics
            
            print(f"\nWalk-forward testing completed. Results saved to {results_dir}")
            print(f"Combined metrics: Return: {combined_metrics['total_return']*100:.2f}%, "
                 f"Sharpe: {combined_metrics['sharpe_ratio']:.2f}, "
                 f"Win Rate: {combined_metrics['win_rate']*100:.2f}%")
            
            return self.results
        else:
            print("No valid periods found in walk-forward test.")
            return None
    
    def run_test_with_parameters(self, data, short_window, long_window, signal_window,
                               strategy_type, trailing_stop, output_dir):
        """
        Run a backtest with specific parameters.
        
        Args:
            data (pd.DataFrame): Data for backtest
            short_window (int): Short EMA window
            long_window (int): Long EMA window
            signal_window (int): Signal EMA window
            strategy_type (str): Strategy type
            trailing_stop (float): Trailing stop percentage
            output_dir (str): Output directory
            
        Returns:
            dict: Backtest results
        """
        # Create strategy
        strategy = MACDStrategy(short_window, long_window, signal_window)
        strategy.strategy_type = strategy_type
        
        # Calculate indicators and signals
        data = strategy.calculate_indicators(data)
        data = strategy.generate_signals(data, strategy_type)
        
        # Create risk manager
        risk_manager = RiskManager(take_profit=None, stop_loss=None, trailing_stop=trailing_stop)
        
        # Run backtest
        backtest_engine = BacktestEngine(
            data, 
            strategy, 
            risk_manager, 
            initial_capital=10000, 
            trade_size=1.0, 
            output_dir=output_dir
        )
        
        results = backtest_engine.run_backtest()
        
        # Visualize the results
        visualizer = BacktestVisualizer(data, results, output_dir)
        
        equity_chart = visualizer.plot_equity_curve()
        price_chart = visualizer.plot_price_and_signals()
        
        # Save charts
        equity_chart_path = os.path.join(output_dir, "equity_curve.html")
        price_chart_path = os.path.join(output_dir, "price_chart.html")
        
        equity_chart.write_html(equity_chart_path)
        price_chart.write_html(price_chart_path)
        
        return {
            'data': data,
            'results': results,
            'equity_chart': equity_chart,
            'price_chart': price_chart
        }
    
    def calculate_combined_metrics(self, combined_equity, period_results):
        """
        Calculate performance metrics for the combined equity curve.
        
        Args:
            combined_equity (pd.DataFrame): Combined equity curve
            period_results (list): Results from each period
            
        Returns:
            dict: Combined performance metrics
        """
        # Calculate total return
        initial_equity = combined_equity['equity'].iloc[0]
        final_equity = combined_equity['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate drawdown
        running_max = combined_equity['equity'].cummax()
        drawdown = (combined_equity['equity'] - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        
        # Calculate Sharpe ratio
        returns = combined_equity['equity'].pct_change().dropna()
        
        # Annualize based on number of periods in a year
        periods_per_day = 24 * 6  # 10-minute candles per day
        annualization_factor = np.sqrt(365 * periods_per_day)
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        sharpe_ratio = (mean_return / std_return) * annualization_factor if std_return > 0 else 0
        
        # Collect trade statistics
        all_trades = []
        for period in period_results:
            all_trades.extend(period['test_trades'])
        
        # Calculate trade metrics
        num_trades = len(all_trades)
        winning_trades = [trade for trade in all_trades if trade['pnl'] > 0]
        losing_trades = [trade for trade in all_trades if trade['pnl'] <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        # Calculate profit factor
        total_profit = sum(trade['pnl'] for trade in winning_trades)
        total_loss = abs(sum(trade['pnl'] for trade in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate expectancy
        avg_win = sum(trade['pnl'] for trade in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(trade['pnl'] for trade in losing_trades) / len(losing_trades) if losing_trades else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    def plot_combined_equity_curve(self, combined_equity, test_periods):
        """
        Plot combined equity curve with test periods highlighted.
        
        Args:
            combined_equity (pd.DataFrame): Combined equity curve
            test_periods (list): List of test periods (start, end, period_num)
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Create figure with subplots for equity and drawdown
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           subplot_titles=('Walk-Forward Equity Curve', 'Drawdown'),
                           row_heights=[0.7, 0.3])
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=combined_equity['timestamp'],
                y=combined_equity['equity'],
                name='Equity',
                line=dict(color='rgb(75, 192, 192)', width=2),
                fill='tozeroy',
                fillcolor='rgba(75, 192, 192, 0.2)'
            ),
            row=1, col=1
        )
        
        # Calculate drawdown
        equity = combined_equity['equity']
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100  # Convert to percentage
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                x=combined_equity['timestamp'],
                y=drawdown,
                name='Drawdown',
                line=dict(color='rgba(255, 99, 132, 1)', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 99, 132, 0.2)'
            ),
            row=2, col=1
        )
        
        # Add horizontal line at zero for drawdown
        fig.add_trace(
            go.Scatter(
                x=[combined_equity['timestamp'].iloc[0], 
                   combined_equity['timestamp'].iloc[-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add background colors for different test periods
        colors = ['rgba(173, 216, 230, 0.3)', 'rgba(144, 238, 144, 0.3)', 
                 'rgba(255, 192, 203, 0.3)', 'rgba(255, 255, 153, 0.3)',
                 'rgba(221, 160, 221, 0.3)', 'rgba(255, 228, 196, 0.3)']
        
        for i, (start, end, period_num) in enumerate(test_periods):
            color = colors[i % len(colors)]
            
            # Add shaded area for test period in equity plot
            fig.add_trace(
                go.Scatter(
                    x=[start, start, end, end],
                    y=[combined_equity['equity'].min(), combined_equity['equity'].max(), 
                       combined_equity['equity'].max(), combined_equity['equity'].min()],
                    fill="toself",
                    fillcolor=color,
                    line=dict(width=0),
                    showlegend=True,
                    name=f"Period {period_num}"
                ),
                row=1, col=1
            )
            
            # Add shaded area for test period in drawdown plot
            fig.add_trace(
                go.Scatter(
                    x=[start, start, end, end],
                    y=[drawdown.min(), 0, 0, drawdown.min()],
                    fill="toself",
                    fillcolor=color,
                    line=dict(width=0),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Walk-Forward Equity Curve and Drawdown',
            xaxis2_title='Date',
            yaxis_title='Equity',
            yaxis2_title='Drawdown (%)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600,
            width=1200,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def walk_forward_parameter_analysis(self):
        """
        Analyze how parameters change across different periods.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with parameter evolution
        """
        if not self.results['period_results']:
            print("No results available for parameter analysis.")
            return None
        
        # Extract parameters from each period
        periods = []
        short_windows = []
        long_windows = []
        signal_windows = []
        strategy_types = []
        trailing_stops = []
        
        for period_result in self.results['period_results']:
            periods.append(period_result['period'])
            short_window, long_window, signal_window, strategy_type, trailing_stop = period_result['best_params']
            
            short_windows.append(short_window)
            long_windows.append(long_window)
            signal_windows.append(signal_window)
            strategy_types.append(strategy_type)
            trailing_stops.append(trailing_stop)
        
        # Create figure with subplots
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=('MACD Windows', 'Strategy Type', 'Trailing Stop'))
        
        # Plot MACD windows
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=short_windows,
                mode='lines+markers',
                name='Short Window',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=long_windows,
                mode='lines+markers',
                name='Long Window',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=signal_windows,
                mode='lines+markers',
                name='Signal Window',
                line=dict(color='green')
            ),
            row=1, col=1
        )
        
        # Plot strategy types
        strategy_mapping = {
            'buy_hold': 1,
            'buy_sell': 2,
            'reversal': 3
        }
        
        strategy_numeric = [strategy_mapping.get(s, 0) for s in strategy_types]
        
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=strategy_numeric,
                mode='lines+markers',
                name='Strategy Type',
                text=strategy_types,
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Add y-axis labels for strategy types
        fig.update_yaxes(
            tickvals=[1, 2, 3],
            ticktext=['buy_hold', 'buy_sell', 'reversal'],
            row=2, col=1
        )
        
        # Plot trailing stop
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=trailing_stops,
                mode='lines+markers',
                name='Trailing Stop',
                line=dict(color='orange')
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Parameter Evolution Across Walk-Forward Periods',
            xaxis3_title='Period',
            yaxis_title='Window Size',
            yaxis2_title='Strategy Type',
            yaxis3_title='Trailing Stop',
            height=800,
            width=1200,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_performance_comparison(self):
        """
        Create a comparison of in-sample vs out-of-sample performance.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with performance comparison
        """
        if not self.results['period_results']:
            print("No results available for performance comparison.")
            return None
        
        # Extract performance metrics
        periods = []
        train_returns = []
        test_returns = []
        train_sharpes = []
        test_sharpes = []
        
        for period_result in self.results['period_results']:
            periods.append(period_result['period'])
            train_returns.append(period_result['train_metrics']['total_return'] * 100)  # Convert to percentage
            test_returns.append(period_result['test_metrics']['total_return'] * 100)  # Convert to percentage
            train_sharpes.append(period_result['train_metrics']['sharpe_ratio'])
            test_sharpes.append(period_result['test_metrics']['sharpe_ratio'])
        
        # Create figure with subplots
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=('Returns (%)', 'Sharpe Ratio'))
        
        # Plot returns
        fig.add_trace(
            go.Bar(
                x=periods,
                y=train_returns,
                name='In-Sample Return',
                marker_color='blue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=periods,
                y=test_returns,
                name='Out-of-Sample Return',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Plot Sharpe ratios
        fig.add_trace(
            go.Bar(
                x=periods,
                y=train_sharpes,
                name='In-Sample Sharpe',
                marker_color='blue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=periods,
                y=test_sharpes,
                name='Out-of-Sample Sharpe',
                marker_color='green',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='In-Sample vs Out-of-Sample Performance Comparison',
            xaxis2_title='Period',
            yaxis_title='Return (%)',
            yaxis2_title='Sharpe Ratio',
            barmode='group',
            height=600,
            width=1200,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig


def run_walk_forward_analysis(symbol='btcusd', timeframe='10m', 
                              training_years=1, testing_months=3, step_months=3):
    """
    Run a complete walk-forward analysis for a given symbol and timeframe.
    
    Args:
        symbol (str): Symbol to test (e.g., 'btcusd', 'ethusd')
        timeframe (str): Timeframe to test
        training_years (int): Number of years to use for training
        testing_months (int): Number of months to use for testing
        step_months (int): Number of months to step forward
        
    Returns:
        dict: Walk-forward analysis results
    """
    print(f"Running walk-forward analysis for {symbol.upper()} on {timeframe} timeframe")
    
    # 1. Initialize data handler
    data_handler = DataHandler()
    
    # 2. Initialize walk-forward tester
    wf_tester = WalkForwardTester(data_handler, symbol=symbol, timeframe=timeframe)
    
    # 3. Define parameter grid
    parameter_grid = {
        'short_window': [8, 12, 16],
        'long_window': [21, 26, 34],
        'signal_window': [5, 9, 13],
        'strategy_type': ['buy_sell', 'reversal'],
        'trailing_stop': [0.01, 0.02, 0.03]
    }
    
    # 4. Run walk-forward test
    results = wf_tester.run_walk_forward_test(
        parameter_grid=parameter_grid,
        training_years=training_years,
        testing_months=testing_months,
        step_months=step_months
    )
    
    if results:
        # 5. Generate parameter analysis
        param_analysis = wf_tester.walk_forward_parameter_analysis()
        
        # 6. Generate performance comparison
        perf_comparison = wf_tester.create_performance_comparison()
        
        # 7. Print overall results
        print("\nWalk-Forward Analysis Results:")
        print(f"Total Return: {results['final_metrics']['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {results['final_metrics']['sharpe_ratio']:.4f}")
        print(f"Win Rate: {results['final_metrics']['win_rate']*100:.2f}%")
        print(f"Max Drawdown: {results['final_metrics']['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {results['final_metrics']['total_trades']}")
        
        return {
            'results': results,
            'parameter_analysis': param_analysis,
            'performance_comparison': perf_comparison
        }
    
    return None


# Example usage
if __name__ == "__main__":
    # Run walk-forward analysis for BTC
    btc_wf_results = run_walk_forward_analysis(
        symbol='btcusd',
        timeframe='10m',
        training_years=1,
        testing_months=6,
        step_months=6
    )
    
    # Display charts
    if btc_wf_results:
        btc_wf_results['parameter_analysis'].show()
        btc_wf_results['performance_comparison'].show()