import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

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
        
    def load_data(self, symbol, timeframe='10m'):
        """
        Load data for a specific symbol and timeframe.
        
        Args:
            symbol (str): Symbol name (e.g., 'btcusd', 'ethusd')
            timeframe (str): Timeframe (e.g., '10m')
            
        Returns:
            pd.DataFrame: Processed dataframe with OHLCV data
        """
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
        self._save_data_integrity_issues(symbol, timeframe)
        
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
    
    def _save_data_integrity_issues(self, symbol, timeframe):
        """
        Save data integrity issues to CSV.
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe
        """
        # Create directory if it doesn't exist
        if not os.path.exists('data_integrity'):
            os.makedirs('data_integrity')
            
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
            missing_df.to_csv(f'data_integrity/{symbol}_{timeframe}_missing_values.csv', index=False)
            
        # Save duplicate timestamps
        if self.data_integrity_issues['duplicate_timestamps']:
            duplicates_df = pd.DataFrame({
                'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') 
                             for ts in self.data_integrity_issues['duplicate_timestamps']]
            })
            duplicates_df.to_csv(f'data_integrity/{symbol}_{timeframe}_duplicate_timestamps.csv', index=False)
            
        # Save non-uniform intervals
        if self.data_integrity_issues['non_uniform_intervals']:
            non_uniform_df = pd.DataFrame(self.data_integrity_issues['non_uniform_intervals'])
            non_uniform_df.to_csv(f'data_integrity/{symbol}_{timeframe}_non_uniform_intervals.csv', index=False)
    
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
            if not os.path.exists('data_integrity'):
                os.makedirs('data_integrity')
                
            changes_df = pd.DataFrame(resampled_changes)
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown'
            changes_df.to_csv(f'data_integrity/{symbol}_{timeframe}_resampled_changes.csv', index=False)
            
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
    def __init__(self, data, strategy, risk_manager, initial_capital=10000, trade_size=1.0):
        """
        Initialize the backtesting engine.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and signals
            strategy: Strategy object
            risk_manager: Risk manager object
            initial_capital (float): Initial capital
            trade_size (float): Trade size as a fraction of capital (0-1)
        """
        self.data = data
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.trade_size = trade_size
        
        # Initialize results storage
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
        # Initialize detailed trade tracking
        self.detailed_tracking = []
        
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
        
        # Process each candle
        for i in range(1, len(self.data)):
            prev_candle = self.data.iloc[i-1]
            current_candle = self.data.iloc[i]
            
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
                'take_profit_level': 0
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
            elif position and position['size'] > 0 and (
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
            
            # Check for entry signal
            if not position and prev_candle['signal'] != 0:
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
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(capital)
        
        # Save detailed tracking to CSV
        self._save_detailed_tracking()
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': metrics
        }
    
    def _save_detailed_tracking(self):
        """
        Save detailed trade tracking to CSV.
        """
        if not self.detailed_tracking:
            return
            
        # Convert to DataFrame
        tracking_df = pd.DataFrame(self.detailed_tracking)
        
        # Save to CSV
        if not os.path.exists('trade_tracking'):
            os.makedirs('trade_tracking')
            
        # Get symbol from data
        symbol = self.data['symbol'].iloc[0] if 'symbol' in self.data.columns else 'unknown'
        
        # Save tracking data
        tracking_df.to_csv(f'trade_tracking/{symbol}_detailed_tracking.csv', index=False)
        print(f"Detailed trade tracking saved to trade_tracking/{symbol}_detailed_tracking.csv")
    
    def calculate_performance_metrics(self, final_capital):
        """
        Calculate performance metrics.
        
        Args:
            final_capital (float): Final capital
            
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # If no trades, return basic metrics
        if not self.trades:
            return {
                'total_return': total_return,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate trade metrics
        num_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        # Profit factor (sum of wins / sum of losses)
        total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate drawdown
        if not self.equity_curve.empty:
            equity = self.equity_curve['equity']
            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        else:
            max_drawdown = 0
        
        # Calculate Sharpe ratio (using daily returns)
        if len(self.equity_curve) > 1:
            equity = self.equity_curve['equity']
            returns = equity.pct_change().dropna()
            
            # Annualize based on number of periods in a year
            periods_per_day = 24 * 6  # 10-minute candles per day
            annualization_factor = np.sqrt(365 * periods_per_day)
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            sharpe_ratio = (mean_return / std_return) * annualization_factor if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate average trade duration
        trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
        avg_duration = trades_df['duration'].mean().total_seconds() / 60  # in minutes
        
        # Expectancy
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Calculate annualized return
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        years = (end_date - start_date).days / 365
        annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration_min': avg_duration,
            'expectancy': expectancy,
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict()
        }
    

class BacktestVisualizer:
    def __init__(self, data, backtest_results):
        """
        Initialize the visualizer.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and indicators
            backtest_results (dict): Results from the backtest
        """
        self.data = data
        self.results = backtest_results
        
    def plot_price_and_signals(self):
        """
        Plot price chart with buy/sell signals.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Create subplots: price chart and MACD
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.1, 
                            subplot_titles=('Price Chart', 'MACD'),
                            row_heights=[0.7, 0.3])
        
        # Add price chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['o'],
                high=self.data['h'],
                low=self.data['l'],
                close=self.data['c'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add EMAs
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['ema_short'],
                name=f'EMA ({self.data["ema_short"].name})',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['ema_long'],
                name=f'EMA ({self.data["ema_long"].name})',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        # Add MACD
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['macd'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['macd_signal'],
                name='Signal',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Add MACD histogram
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['macd_hist'],
                name='Histogram',
                marker=dict(
                    color=np.where(self.data['macd_hist'] < 0, 'red', 'green'),
                    line=dict(color='black', width=1)
                )
            ),
            row=2, col=1
        )
        
        # Add buy signals
        buy_signals = self.data[self.data['signal'] == 1]
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['l'] * 0.99,  # Slightly below low price
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green',
                    line=dict(color='black', width=1)
                ),
                name='Buy Signal'
            ),
            row=1, col=1
        )
        
        # Add sell signals
        sell_signals = self.data[self.data['signal'] == -1]
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['h'] * 1.01,  # Slightly above high price
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red',
                    line=dict(color='black', width=1)
                ),
                name='Sell Signal'
            ),
            row=1, col=1
        )
        
        # Add trades
        for trade in self.results['trades']:
            # Add entry marker
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time']],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color='green' if trade['type'] == 'long' else 'red',
                        line=dict(color='black', width=1)
                    ),
                    name='Entry'
                ),
                row=1, col=1
            )
            
            # Add exit marker
            marker_color = {
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
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color=marker_color,
                        line=dict(color='black', width=1)
                    ),
                    name=f"Exit ({trade['exit_reason']})"
                ),
                row=1, col=1
            )
            
            # Add trade line
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time'], trade['exit_time']],
                    y=[trade['entry_price'], trade['exit_price']],
                    mode='lines',
                    line=dict(
                        color='green' if trade['pnl'] > 0 else 'red',
                        width=1,
                        dash='dot'
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Price Chart with MACD Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=800,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update candlestick colors
        fig.update_layout(
            xaxis_rangeslider_visible=False
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
                line=dict(color='blue', width=2)
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
                line=dict(color='red', width=1)
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
                'Total Return',
                'Annualized Return',
                'Total Trades',
                'Win Rate',
                'Profit Factor',
                'Max Drawdown',
                'Sharpe Ratio',
                'Avg Trade Duration (min)',
                'Expectancy'
            ],
            'Value': [
                f"{metrics['total_return']:.2%}",
                f"{metrics['annualized_return']:.2%}",
                f"{metrics['total_trades']}",
                f"{metrics['win_rate']:.2%}",
                f"{metrics['profit_factor']:.2f}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['avg_trade_duration_min']:.1f}",
                f"${metrics['expectancy']:.2f}"
            ]
        }
        
        return pd.DataFrame(summary)
    
    def save_trade_log(self, filepath):
        """
        Save trade log to CSV.
        
        Args:
            filepath (str): Path to save the file
        """
        trades_df = pd.DataFrame(self.results['trades'])
        trades_df.to_csv(filepath, index=False)
        
        print(f"Trade log saved to {filepath}")

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
    data = data_handler.load_data(symbol, timeframe)
    
    # Check for uniform time intervals and resample if needed
    if timeframe == '10m':
        resampling_timeframe = '10T'  # 10 minutes
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    data = data_handler.resample_data(data, resampling_timeframe)
    
    # 2. Initialize strategy
    strategy = MACDStrategy(short_window, long_window, signal_window)
    data = strategy.calculate_indicators(data)
    data = strategy.generate_signals(data, strategy_type)
    
    # 3. Initialize risk management
    risk_manager = RiskManager(take_profit, stop_loss, trailing_stop)
    
    # 4. Run backtest
    backtest_engine = BacktestEngine(data, strategy, risk_manager, initial_capital, trade_size)
    results = backtest_engine.run_backtest()
    
    # 5. Visualize results
    visualizer = BacktestVisualizer(data, results)
    
    price_chart = visualizer.plot_price_and_signals()
    equity_chart = visualizer.plot_equity_curve()
    summary_table = visualizer.create_summary_table()
    
    # Save trade log
    visualizer.save_trade_log(f"{symbol}_{timeframe}_macd_trades.csv")
    
    # Print summary
    print("\nBacktest Summary:")
    print(summary_table.to_string(index=False))
    
    return {
        'data': data,
        'results': results,
        'price_chart': price_chart,
        'equity_chart': equity_chart,
        'summary_table': summary_table
    }

# Example usage
if __name__ == "__main__":
    # Run backtest for BTC with default parameters
    # btc_results = run_macd_backtest(
    #     symbol='btcusd',
    #     timeframe='10m',
    #     short_window=5,
    #     long_window=50,
    #     signal_window=9,
    #     strategy_type='buy_sell',
    #     trailing_stop=0.02,
    #     initial_capital=10000,
    #     trade_size=0.1
    # )
    
    # btc_results = run_macd_backtest(
    #     symbol='btcusd',
    #     timeframe='10m',
    #     short_window=5,
    #     long_window=50,
    #     signal_window=9,
    #     strategy_type='buy_sell',
    #     trailing_stop=0.02,
    #     initial_capital=10000,
    #     trade_size=0.1
    # )

    btc_results = run_macd_backtest(
        symbol='btcusd',
        timeframe='10m',
        short_window=5,
        long_window=50,
        signal_window=9,
        strategy_type='buy_hold',
        trailing_stop=0.02,
        initial_capital=10000,
        trade_size=0.1
    )

    # Display charts
    btc_results['price_chart'].show()
    btc_results['equity_chart'].show()
    
    # Run backtest for ETH with custom parameters
    # eth_results = run_macd_backtest(
    #     symbol='ethusd',
    #     timeframe='10m',
    #     short_window=5,
    #     long_window=50,
    #     signal_window=5,
    #     strategy_type='buy_sell',
    #     take_profit=0.05,
    #     stop_loss=0.03,
    #     initial_capital=10000,
    #     trade_size=0.1
    # )

    eth_results = run_macd_backtest(
        symbol='ethusd',
        timeframe='10m',
        short_window=5,
        long_window=50,
        signal_window=5,
        strategy_type='reversal',
        take_profit=0.05,
        stop_loss=0.03,
        initial_capital=10000,
        trade_size=0.1
    )
    
    # Display charts
    eth_results['price_chart'].show()
    eth_results['equity_chart'].show()
    
    # Compare results
    print("\nComparison of BTC vs ETH:")
    btc_metrics = btc_results['results']['metrics']
    eth_metrics = eth_results['results']['metrics']
    
    comparison = pd.DataFrame({
        'Metric': [
            'Total Return',
            'Annualized Return',
            'Total Trades',
            'Win Rate',
            'Profit Factor',
            'Max Drawdown',
            'Sharpe Ratio'
        ],
        'BTC': [
            f"{btc_metrics['total_return']:.2%}",
            f"{btc_metrics['annualized_return']:.2%}",
            f"{btc_metrics['total_trades']}",
            f"{btc_metrics['win_rate']:.2%}",
            f"{btc_metrics['profit_factor']:.2f}",
            f"{btc_metrics['max_drawdown']:.2%}",
            f"{btc_metrics['sharpe_ratio']:.2f}"
        ],
        'ETH': [
            f"{eth_metrics['total_return']:.2%}",
            f"{eth_metrics['annualized_return']:.2%}",
            f"{eth_metrics['total_trades']}",
            f"{eth_metrics['win_rate']:.2%}",
            f"{eth_metrics['profit_factor']:.2f}",
            f"{eth_metrics['max_drawdown']:.2%}",
            f"{eth_metrics['sharpe_ratio']:.2f}"
        ]
    })
    
    print(comparison.to_string(index=False))