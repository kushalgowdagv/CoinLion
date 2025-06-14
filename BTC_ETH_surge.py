import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
import concurrent.futures
import itertools
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom Exceptions
class BacktestError(Exception):
    """Base exception for backtest errors"""
    pass

class DataError(BacktestError):
    """Data-related errors"""
    pass

class StrategyError(BacktestError):
    """Strategy-related errors"""
    pass

class RiskManagementError(BacktestError):
    """Risk management-related errors"""
    pass

# Configuration Classes
@dataclass
class StrategyConfig:
    """Configuration for BTC Volume → ETH strategy parameters"""
    volume_window: int = 20  # Rolling window for volume average
    volume_threshold: float = 2.0  # Volume surge threshold (2x average)
    momentum_periods: int = 3  # Periods for momentum calculation
    momentum_threshold: float = 0.01  # 1% momentum threshold
    signal_delay: int = 2  # Delay in periods for ETH signal generation
    strategy_type: str = 'btc_volume_eth'  # Strategy identifier
    
    def __post_init__(self) -> None:
        self._validate()
    
    def _validate(self) -> None:
        if self.volume_window <= 0:
            raise ValueError("Volume window must be positive")
        if self.volume_threshold <= 1.0:
            raise ValueError("Volume threshold must be greater than 1.0")
        if self.momentum_periods <= 0:
            raise ValueError("Momentum periods must be positive")
        if self.momentum_threshold <= 0:
            raise ValueError("Momentum threshold must be positive")
        if self.signal_delay < 0:
            raise ValueError("Signal delay must be non-negative")

@dataclass
class RiskConfig:
    """Configuration for risk management parameters"""
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    trailing_stop: Optional[float] = 0.02
    position_size: float = 1.0
    
    def __post_init__(self) -> None:
        self._validate()
    
    def _validate(self) -> None:
        if self.take_profit is not None and (self.take_profit <= 0 or self.take_profit > 1):
            raise ValueError("Take profit must be between 0 and 1")
        if self.stop_loss is not None and (self.stop_loss <= 0 or self.stop_loss > 1):
            raise ValueError("Stop loss must be between 0 and 1")
        if self.trailing_stop is not None and (self.trailing_stop <= 0 or self.trailing_stop > 1):
            raise ValueError("Trailing stop must be between 0 and 1")
        if self.position_size <= 0 or self.position_size > 1:
            raise ValueError("Position size must be between 0 and 1")

@dataclass
class BacktestConfig:
    """Configuration for backtest parameters"""
    initial_capital: float = 10000.0
    trade_size: float = 1.0
    periods_per_day: int = 144  # 10-minute candles per day
    annualization_factor: float = field(init=False)
    
    def __post_init__(self) -> None:
        self.annualization_factor = np.sqrt(365 * self.periods_per_day)
        self._validate()
    
    def _validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.trade_size <= 0 or self.trade_size > 1:
            raise ValueError("Trade size must be between 0 and 1")

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward testing"""
    training_years: int = 1
    testing_months: int = 3
    step_months: int = 3
    optimization_metric: str = 'sharpe_ratio'
    max_workers: Optional[int] = None
    
    def __post_init__(self) -> None:
        if self.max_workers is None:
            self.max_workers = os.cpu_count()
        self._validate()
    
    def _validate(self) -> None:
        if self.training_years <= 0:
            raise ValueError("Training years must be positive")
        if self.testing_months <= 0:
            raise ValueError("Testing months must be positive")
        if self.step_months <= 0:
            raise ValueError("Step months must be positive")
        
        # Updated list of allowed optimization metrics
        allowed_metrics = [
            'sharpe_ratio', 'total_return', 'profit_factor', 'win_rate', 
            'net_profit', 'avg_winner', 'avg_loser', 'expectancy', 
            'sortino_ratio', 'calmar_ratio', 'max_drawdown_inverse'
        ]
        
        if self.optimization_metric not in allowed_metrics:
            raise ValueError(f"Optimization metric must be one of: {', '.join(allowed_metrics)}")

# Utility Functions
def get_output_directory(symbol: str, timeframe: str, strategy_type: str) -> Tuple[str, str]:
    """
    Create a timestamped output directory structure.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        strategy_type: Strategy type
        
    Returns:
        Tuple of (base_output_dir, strategy_dir)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join('output', f"{symbol}_{timeframe}_{timestamp}")
    strategy_dir = os.path.join(base_output_dir, f"{strategy_type}_strategy")
    os.makedirs(strategy_dir, exist_ok=True)
    return base_output_dir, strategy_dir

# Abstract Strategy Base Class
class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    @abstractmethod
    def calculate_indicators(self, btc_data: pd.DataFrame, eth_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy"""
        pass
    
    @abstractmethod
    def generate_signals(self, btc_data: pd.DataFrame, eth_data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on indicators"""
        pass

# Data Handler Class
class DataHandler:
    """Handles data loading, validation, and preprocessing for BTC and ETH"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir if base_dir else os.getcwd()
        self.data_integrity_issues: Dict[str, Any] = {
            'missing_values': {},
            'duplicate_timestamps': [],
            'non_uniform_intervals': []
        }
        self.output_base_dir: Optional[str] = None
        self.output_strategy_dir: Optional[str] = None
        logger.info(f"Initialized DataHandler with base directory: {self.base_dir}")
        
    def load_paired_data(self, btc_symbol: str = 'btcusd', eth_symbol: str = 'ethusd', 
                        timeframe: str = '10m', strategy_type: str = 'btc_volume_eth') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data for both BTC and ETH symbols.
        
        Args:
            btc_symbol: BTC symbol name
            eth_symbol: ETH symbol name
            timeframe: Timeframe
            strategy_type: Strategy type for directory naming
            
        Returns:
            Tuple of (btc_dataframe, eth_dataframe)
            
        Raises:
            DataError: If data files are not found or invalid
        """
        try:
            self.output_base_dir, self.output_strategy_dir = get_output_directory(
                f"{btc_symbol}_{eth_symbol}", timeframe, strategy_type
            )
            self.btc_symbol = btc_symbol
            self.eth_symbol = eth_symbol
            self.timeframe = timeframe
            
            # Load BTC data
            btc_filepath = os.path.join(self.base_dir, f'data/{btc_symbol}_{timeframe}.csv')
            if not os.path.exists(btc_filepath):
                raise DataError(f"BTC data file not found: {btc_filepath}")
            
            # Load ETH data
            eth_filepath = os.path.join(self.base_dir, f'data/{eth_symbol}_{timeframe}.csv')
            if not os.path.exists(eth_filepath):
                raise DataError(f"ETH data file not found: {eth_filepath}")
                
            btc_df = pd.read_csv(btc_filepath)
            eth_df = pd.read_csv(eth_filepath)
            
            btc_df = self._process_data(btc_df)
            eth_df = self._process_data(eth_df)
            
            # Align timestamps
            btc_df, eth_df = self._align_timestamps(btc_df, eth_df)
            
            self._check_data_integrity(btc_df, "BTC")
            self._check_data_integrity(eth_df, "ETH")
            self._save_data_integrity_issues()
            
            logger.info(f"Successfully loaded {len(btc_df)} rows of BTC data and {len(eth_df)} rows of ETH data")
            return btc_df, eth_df
            
        except Exception as e:
            logger.error(f"Error loading paired data: {str(e)}")
            raise DataError(f"Failed to load paired data: {str(e)}") from e
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into required format"""
        df['time_utc'] = pd.to_datetime(df['time_utc'])
        df['time_est'] = pd.to_datetime(df['time_est'])
        df.set_index('time_utc', inplace=True)
        return df
    
    def _align_timestamps(self, btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align timestamps between BTC and ETH data"""
        # Find common timestamps
        common_timestamps = btc_df.index.intersection(eth_df.index)
        
        if len(common_timestamps) == 0:
            raise DataError("No common timestamps found between BTC and ETH data")
        
        btc_aligned = btc_df.loc[common_timestamps].copy()
        eth_aligned = eth_df.loc[common_timestamps].copy()
        
        logger.info(f"Aligned data to {len(common_timestamps)} common timestamps")
        return btc_aligned, eth_aligned
    
    def _check_data_integrity(self, df: pd.DataFrame, symbol: str) -> None:
        """Check data integrity and log issues"""
        self._check_missing_values(df, symbol)
        self._check_duplicate_timestamps(df, symbol)
        self._check_uniform_intervals(df, symbol)
    
    def _check_missing_values(self, df: pd.DataFrame, symbol: str) -> None:
        """Check for missing values in OHLCV columns"""
        ohlcv_cols = ['o', 'h', 'l', 'c', 'v']
        for col in ohlcv_cols:
            missing = df[df[col].isnull()]
            if not missing.empty:
                key = f"{symbol}_{col}"
                self.data_integrity_issues['missing_values'][key] = missing.index.tolist()
                logger.warning(f"Found {len(missing)} missing values in '{symbol}' column '{col}'")
    
    def _check_duplicate_timestamps(self, df: pd.DataFrame, symbol: str) -> None:
        """Check for duplicate timestamps"""
        duplicates = df.index[df.index.duplicated()].tolist()
        if duplicates:
            self.data_integrity_issues['duplicate_timestamps'].extend([
                {'symbol': symbol, 'timestamp': ts} for ts in duplicates
            ])
            logger.warning(f"Found {len(duplicates)} duplicate timestamps in {symbol}")
    
    def _check_uniform_intervals(self, df: pd.DataFrame, symbol: str) -> None:
        """Check for uniform time intervals"""
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=10)
        non_uniform = time_diffs[time_diffs != expected_diff]
        if not non_uniform.empty:
            non_uniform_list = [
                {
                    'symbol': symbol,
                    'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'), 
                    'interval': diff.total_seconds() / 60
                } 
                for idx, diff in non_uniform.items()
            ]
            self.data_integrity_issues['non_uniform_intervals'].extend(non_uniform_list)
            logger.warning(f"Found {len(non_uniform)} non-uniform time intervals in {symbol}")
    
    def _save_data_integrity_issues(self) -> None:
        """Save data integrity issues to CSV files"""
        if not self.output_strategy_dir:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save missing values
        self._save_missing_values_report(timestamp)
        self._save_duplicate_timestamps_report(timestamp)
        self._save_non_uniform_intervals_report(timestamp)
    
    def _save_missing_values_report(self, timestamp: str) -> None:
        """Save missing values report"""
        missing_values_data = []
        for key, timestamps in self.data_integrity_issues['missing_values'].items():
            symbol, column = key.split('_', 1)
            for ts in timestamps:
                missing_values_data.append({
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'column': column
                })
                
        if missing_values_data:
            missing_df = pd.DataFrame(missing_values_data)
            missing_file = os.path.join(
                self.output_strategy_dir, 
                f"{self.btc_symbol}_{self.eth_symbol}_{self.timeframe}_missing_values_{timestamp}.csv"
            )
            missing_df.to_csv(missing_file, index=False)
            logger.info(f"Missing values report saved to {missing_file}")
    
    def _save_duplicate_timestamps_report(self, timestamp: str) -> None:
        """Save duplicate timestamps report"""
        if self.data_integrity_issues['duplicate_timestamps']:
            duplicates_df = pd.DataFrame(self.data_integrity_issues['duplicate_timestamps'])
            duplicates_file = os.path.join(
                self.output_strategy_dir, 
                f"{self.btc_symbol}_{self.eth_symbol}_{self.timeframe}_duplicate_timestamps_{timestamp}.csv"
            )
            duplicates_df.to_csv(duplicates_file, index=False)
            logger.info(f"Duplicate timestamps report saved to {duplicates_file}")
    
    def _save_non_uniform_intervals_report(self, timestamp: str) -> None:
        """Save non-uniform intervals report"""
        if self.data_integrity_issues['non_uniform_intervals']:
            non_uniform_df = pd.DataFrame(self.data_integrity_issues['non_uniform_intervals'])
            non_uniform_file = os.path.join(
                self.output_strategy_dir, 
                f"{self.btc_symbol}_{self.eth_symbol}_{self.timeframe}_non_uniform_intervals_{timestamp}.csv"
            )
            non_uniform_df.to_csv(non_uniform_file, index=False)
            logger.info(f"Non-uniform intervals report saved to {non_uniform_file}")
    
    def resample_data(self, df: pd.DataFrame, timeframe: str = '10T') -> pd.DataFrame:
        """
        Resample data to ensure uniform time intervals.
        
        Args:
            df: DataFrame to resample
            timeframe: Pandas time frequency string
            
        Returns:
            Resampled dataframe
        """
        try:
            original_df = df.copy()
            
            resampled = df.resample(timeframe).agg({
                'o': 'first',
                'h': 'max',
                'l': 'min',
                'c': 'last',
                'v': 'sum',
                'symbol': 'first',
                'time_est': 'first'
            })
            
            resampled['symbol'] = resampled['symbol'].ffill()
            resampled['time_est'] = resampled['time_est'].ffill()
            
            self._handle_missing_after_resampling(resampled)
            self._save_resampling_changes(original_df, resampled, timeframe)
            
            logger.info(f"Resampled data from {len(df)} to {len(resampled)} rows")
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {str(e)}")
            raise DataError(f"Failed to resample data: {str(e)}") from e
    
    def _handle_missing_after_resampling(self, resampled: pd.DataFrame) -> None:
        """Handle missing values after resampling"""
        missing_after = resampled[['o', 'h', 'l', 'c', 'v']].isnull().sum().sum()
        if missing_after > 0:
            logger.warning(f"After resampling, found {missing_after} missing values")
            resampled[['o', 'h', 'l', 'c']] = resampled[['o', 'h', 'l', 'c']].interpolate(method='linear')
            resampled['v'] = resampled['v'].fillna(0)
    
    def _save_resampling_changes(self, original_df: pd.DataFrame, 
                                resampled: pd.DataFrame, timeframe: str) -> None:
        """Save resampling changes to CSV"""
        if not self.output_strategy_dir:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol = original_df['symbol'].iloc[0] if 'symbol' in original_df.columns else 'unknown'
        
        changes_file = os.path.join(
            self.output_strategy_dir, 
            f"{symbol}_{timeframe}_resampled_changes_{timestamp}.csv"
        )
        
        # Create changes report
        original_timestamps = set(original_df.index)
        resampled_timestamps = set(resampled.index)
        new_timestamps = resampled_timestamps - original_timestamps
        
        if new_timestamps:
            changes_data = [
                {
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'added',
                    'columns': 'o,h,l,c,v'
                }
                for ts in new_timestamps
            ]
            
            changes_df = pd.DataFrame(changes_data)
            changes_df.to_csv(changes_file, index=False)
            logger.info(f"Resampling changes saved to {changes_file}")

# BTC Volume → ETH Strategy Implementation
class BTCVolumeETHStrategy(TradingStrategy):
    """BTC Volume Surge → ETH Price Strategy implementation"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
    def calculate_indicators(self, btc_data: pd.DataFrame, eth_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate BTC volume and momentum indicators for ETH trading"""
        try:
            # Create combined dataset with ETH prices and BTC indicators
            result = eth_data.copy()
            
            # Add BTC data columns with prefix
            result['btc_close'] = btc_data['c']
            result['btc_volume'] = btc_data['v']
            
            # Calculate BTC volume indicators
            result['btc_volume_avg'] = result['btc_volume'].rolling(
                window=self.config.volume_window, min_periods=1
            ).mean()
            
            result['btc_volume_ratio'] = result['btc_volume'] / result['btc_volume_avg']
            result['btc_volume_surge'] = (result['btc_volume_ratio'] >= self.config.volume_threshold).astype(int)
            
            # Calculate BTC momentum indicators
            result['btc_momentum'] = (
                result['btc_close'] / result['btc_close'].shift(self.config.momentum_periods) - 1
            )
            
            result['btc_bullish_momentum'] = (result['btc_momentum'] > self.config.momentum_threshold).astype(int)
            result['btc_bearish_momentum'] = (result['btc_momentum'] < -self.config.momentum_threshold).astype(int)
            
            # Calculate volume surge quality (sustained for 1-2 periods)
            result['btc_volume_sustained'] = (
                result['btc_volume_surge'].rolling(window=2, min_periods=1).sum() >= 1
            ).astype(int)
            
            logger.debug("BTC Volume → ETH indicators calculated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating BTC Volume → ETH indicators: {str(e)}")
            raise StrategyError(f"Failed to calculate indicators: {str(e)}") from e
    
    def generate_signals(self, btc_data: pd.DataFrame, eth_data: pd.DataFrame) -> pd.DataFrame:
        """Generate ETH buy/sell signals based on BTC volume surges and momentum"""
        try:
            # Calculate indicators if not already present
            if 'btc_volume_surge' not in eth_data.columns:
                eth_data = self.calculate_indicators(btc_data, eth_data)
            
            result = eth_data.copy()
            result['signal'] = 0
            
            # Generate primary signals based on BTC conditions
            result['btc_signal_conditions'] = self._calculate_btc_signal_conditions(result)
            
            # Apply signal delay (institutional rotation delay)
            result['delayed_btc_conditions'] = result['btc_signal_conditions'].shift(self.config.signal_delay)
            
            # Generate final ETH signals
            result['signal'] = self._apply_signal_logic(result['delayed_btc_conditions'])
            
            # Add signal strength for position sizing
            result['signal_strength'] = self._calculate_signal_strength(result)
            
            logger.debug(f"Generated ETH signals using BTC Volume strategy")
            return result
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise StrategyError(f"Failed to generate signals: {str(e)}") from e
    
    def _calculate_btc_signal_conditions(self, data: pd.DataFrame) -> pd.Series:
        """Calculate BTC signal conditions (volume surge + momentum)"""
        # Long signal: Volume surge + bullish momentum
        long_condition = (
            (data['btc_volume_sustained'] == 1) & 
            (data['btc_bullish_momentum'] == 1)
        )
        
        # Short signal: Volume surge + bearish momentum
        short_condition = (
            (data['btc_volume_sustained'] == 1) & 
            (data['btc_bearish_momentum'] == 1)
        )
        
        return np.where(long_condition, 1, np.where(short_condition, -1, 0))
    
    def _apply_signal_logic(self, delayed_conditions: pd.Series) -> pd.Series:
        """Apply strategy-specific logic to delayed BTC conditions"""
        # For this strategy, we trade ETH based on BTC signals
        # Long ETH when BTC shows bullish volume + momentum
        # Short ETH when BTC shows bearish volume + momentum
        return delayed_conditions.fillna(0)
    
    def _calculate_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signal strength for position sizing"""
        strength = pd.Series(0.0, index=data.index)
        
        # Strong signals: High volume ratio (>3x) + strong momentum (>2%)
        strong_volume = data['btc_volume_ratio'] > 3.0
        strong_momentum = np.abs(data['btc_momentum']) > 0.02
        strong_signals = strong_volume & strong_momentum & (data['signal'] != 0)
        
        # Moderate signals: Standard conditions met
        moderate_signals = (data['signal'] != 0) & ~strong_signals
        
        strength.loc[strong_signals] = 1.0
        strength.loc[moderate_signals] = 0.7
        
        return strength

# Risk Manager Class (unchanged)
class RiskManager:
    """Handles risk management logic"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self._validate_configuration()
        logger.info(f"Initialized RiskManager with config: {config}")
        
    def _validate_configuration(self) -> None:
        """Validate risk management configuration"""
        if (self.config.take_profit is not None or self.config.stop_loss is not None) and \
           self.config.trailing_stop is not None:
            logger.warning("Both fixed TP/SL and trailing stop are set. Using trailing stop.")
            self.config.take_profit = None
            self.config.stop_loss = None
        
    def check_exit_conditions(self, position: Dict[str, Any], 
                            current_candle: pd.Series) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if exit conditions are met for a position.
        
        Args:
            position: Current position information
            current_candle: Current price candle
            
        Returns:
            Tuple of (exit_triggered, exit_reason, exit_price)
        """
        try:
            if not position or position['size'] == 0:
                return False, None, None
            
            entry_price = position['entry_price']
            current_price = current_candle['c']
            position_type = position['type']
            
            # Check fixed TP/SL conditions
            if self.config.take_profit is not None or self.config.stop_loss is not None:
                return self._check_fixed_conditions(position_type, entry_price, current_price)
            
            # Check trailing stop conditions
            elif self.config.trailing_stop is not None:
                return self._check_trailing_stop_conditions(position, current_price)
            
            return False, None, None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
            raise RiskManagementError(f"Failed to check exit conditions: {str(e)}") from e
    
    def _check_fixed_conditions(self, position_type: str, entry_price: float, 
                               current_price: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check fixed take profit and stop loss conditions"""
        if position_type == 'long':
            return self._check_long_fixed_conditions(entry_price, current_price)
        elif position_type == 'short':
            return self._check_short_fixed_conditions(entry_price, current_price)
        return False, None, None
    
    def _check_long_fixed_conditions(self, entry_price: float, 
                                   current_price: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check fixed conditions for long positions"""
        if self.config.take_profit and current_price >= entry_price * (1 + self.config.take_profit):
            return True, 'take_profit', entry_price * (1 + self.config.take_profit)
        
        if self.config.stop_loss and current_price <= entry_price * (1 - self.config.stop_loss):
            return True, 'stop_loss', entry_price * (1 - self.config.stop_loss)
        
        return False, None, None
    
    def _check_short_fixed_conditions(self, entry_price: float, 
                                    current_price: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check fixed conditions for short positions"""
        if self.config.take_profit and current_price <= entry_price * (1 - self.config.take_profit):
            return True, 'take_profit', entry_price * (1 - self.config.take_profit)
        
        if self.config.stop_loss and current_price >= entry_price * (1 + self.config.stop_loss):
            return True, 'stop_loss', entry_price * (1 + self.config.stop_loss)
        
        return False, None, None
    
    def _check_trailing_stop_conditions(self, position: Dict[str, Any], 
                                      current_price: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check trailing stop conditions"""
        position_type = position['type']
        entry_price = position['entry_price']
        
        if position_type == 'long':
            return self._check_long_trailing_stop(position, current_price)
        elif position_type == 'short':
            return self._check_short_trailing_stop(position, current_price)
        
        return False, None, None
    
    def _check_long_trailing_stop(self, position: Dict[str, Any], 
                                current_price: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check trailing stop for long positions"""
        highest_price = position.get('highest_price', position['entry_price'])
        
        if current_price > highest_price:
            position['highest_price'] = current_price
            highest_price = current_price
        
        trailing_stop_price = highest_price * (1 - self.config.trailing_stop)
        if current_price <= trailing_stop_price:
            return True, 'trailing_stop', trailing_stop_price
        
        return False, None, None
    
    def _check_short_trailing_stop(self, position: Dict[str, Any], 
                                 current_price: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check trailing stop for short positions"""
        lowest_price = position.get('lowest_price', position['entry_price'])
        
        if current_price < lowest_price:
            position['lowest_price'] = current_price
            lowest_price = current_price
        
        trailing_stop_price = lowest_price * (1 + self.config.trailing_stop)
        if current_price >= trailing_stop_price:
            return True, 'trailing_stop', trailing_stop_price
        
        return False, None, None

# Backtest Engine Class
class BacktestEngine:
    """Core engine for running backtests"""
    
    def __init__(self, eth_data: pd.DataFrame, strategy: TradingStrategy, 
                 risk_manager: RiskManager, config: BacktestConfig, 
                 output_dir: Optional[str] = None):
        self.eth_data = eth_data  # Main trading data (ETH)
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config
        self.output_dir = output_dir
        
        # Initialize results storage
        self.positions: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: pd.DataFrame = pd.DataFrame()
        self.detailed_tracking: List[Dict[str, Any]] = []
        self.benchmark_data: Optional[pd.DataFrame] = None
        
        logger.info("Initialized BacktestEngine for BTC Volume → ETH Strategy")
        
    def run_backtest(self) -> Dict[str, Any]:
        """Run the complete backtest"""
        try:
            logger.info("Starting BTC Volume → ETH backtest execution")
            
            # Initialize tracking variables
            capital = self.config.initial_capital
            position: Optional[Dict[str, Any]] = None
            
            # Calculate benchmark (ETH buy and hold)
            benchmark_history = self._calculate_benchmark()
            
            # Process each candle
            equity_history, final_capital = self._process_candles(capital, position, benchmark_history)
            
            # Store results
            self.equity_curve = pd.DataFrame(equity_history)
            self.benchmark_data = pd.DataFrame(benchmark_history)
            
            # Calculate metrics
            metrics = self._calculate_performance_metrics(final_capital)
            
            # Save results
            self._save_results(metrics)
            
            logger.info("BTC Volume → ETH backtest completed successfully")
            logger.info(f"Initial Capital: {self.config.initial_capital:.2f}")
            logger.info(f"Final Capital: {final_capital:.2f}")
            logger.info(f"Total Return: {((final_capital - self.config.initial_capital) / self.config.initial_capital) * 100:.2f}%")
            
            return {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'benchmark_data': self.benchmark_data,
                'metrics': metrics,
                'final_capital': final_capital
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise BacktestError(f"Backtest execution failed: {str(e)}") from e
    
    def _process_candles(self, initial_capital: float, initial_position: Optional[Dict[str, Any]],
                        benchmark_history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        """Process each candle in the backtest"""
        capital = initial_capital
        position = initial_position
        equity_history = []
        
        for i in range(1, len(self.eth_data)):
            prev_candle = self.eth_data.iloc[i-1]
            current_candle = self.eth_data.iloc[i]
            
            # Track equity
            current_equity = self._calculate_current_equity(capital, position, current_candle)
            equity_history.append({
                'timestamp': current_candle.name,
                'equity': current_equity
            })
            
            # Track detailed information
            candle_tracking = self._create_candle_tracking(
                current_candle, prev_candle, capital, position, benchmark_history[i]['value']
            )
            
            # Check risk management exit
            capital, position = self._check_risk_exit(capital, position, prev_candle, current_candle, candle_tracking)
            
            # Check signal exit
            capital, position = self._check_signal_exit(capital, position, prev_candle, current_candle, candle_tracking)
            
            # Check new position entry
            position = self._check_position_entry(capital, position, prev_candle, current_candle, candle_tracking)
            
            self.detailed_tracking.append(candle_tracking)
        
        # Close final position if open
        if position and position['size'] > 0:
            capital = self._close_final_position(capital, position)
        
        return equity_history, capital

    def _calculate_benchmark(self) -> List[Dict[str, Any]]:
        """Calculate benchmark (ETH buy and hold) performance"""
        benchmark_start_price = self.eth_data.iloc[0]['c']
        benchmark_shares = self.config.initial_capital / benchmark_start_price
        benchmark_values = benchmark_shares * self.eth_data['c']
        return [
            {'timestamp': ts, 'value': val}
            for ts, val in zip(self.eth_data.index, benchmark_values)
        ]
        
    def _calculate_current_equity(self, capital: float, position: Optional[Dict[str, Any]], 
                                current_candle: pd.Series) -> float:
        """Calculate current equity including unrealized PnL"""
        if position and position['size'] > 0:
            if position['type'] == 'long':
                unrealized_pnl = position['size'] * (current_candle['o'] - position['entry_price'])
            else:  # short
                unrealized_pnl = position['size'] * (position['entry_price'] - current_candle['o'])
            return capital + unrealized_pnl
        return capital
    
    def _create_candle_tracking(self, current_candle: pd.Series, prev_candle: pd.Series,
                              capital: float, position: Optional[Dict[str, Any]], 
                              benchmark_value: float) -> Dict[str, Any]:
        """Create detailed tracking information for current candle"""
        unrealized_pnl = 0
        if position and position['size'] > 0:
            if position['type'] == 'long':
                unrealized_pnl = position['size'] * (current_candle['o'] - position['entry_price'])
            else:
                unrealized_pnl = position['size'] * (position['entry_price'] - current_candle['o'])
        
        tracking = {
            'timestamp': current_candle.name,
            'eth_open': current_candle['o'],
            'eth_high': current_candle['h'],
            'eth_low': current_candle['l'],
            'eth_close': current_candle['c'],
            'eth_volume': current_candle['v'],
            'btc_close': current_candle.get('btc_close', 0),
            'btc_volume': current_candle.get('btc_volume', 0),
            'btc_volume_avg': current_candle.get('btc_volume_avg', 0),
            'btc_volume_ratio': current_candle.get('btc_volume_ratio', 0),
            'btc_volume_surge': current_candle.get('btc_volume_surge', 0),
            'btc_momentum': current_candle.get('btc_momentum', 0),
            'btc_bullish_momentum': current_candle.get('btc_bullish_momentum', 0),
            'btc_bearish_momentum': current_candle.get('btc_bearish_momentum', 0),
            'signal': prev_candle.get('signal', 0),
            'signal_strength': prev_candle.get('signal_strength', 0),
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
            self._calculate_risk_levels(tracking, position)
        
        return tracking
    
    def _calculate_risk_levels(self, tracking: Dict[str, Any], position: Dict[str, Any]) -> None:
        """Calculate risk management levels for tracking"""
        entry_price = position['entry_price']
        position_type = position['type']
        
        if position_type == 'long':
            if self.risk_manager.config.stop_loss:
                tracking['stop_loss_level'] = entry_price * (1 - self.risk_manager.config.stop_loss)
            if self.risk_manager.config.take_profit:
                tracking['take_profit_level'] = entry_price * (1 + self.risk_manager.config.take_profit)
            if self.risk_manager.config.trailing_stop:
                highest = position.get('highest_price', entry_price)
                tracking['trailing_stop_level'] = highest * (1 - self.risk_manager.config.trailing_stop)
        
        elif position_type == 'short':
            if self.risk_manager.config.stop_loss:
                tracking['stop_loss_level'] = entry_price * (1 + self.risk_manager.config.stop_loss)
            if self.risk_manager.config.take_profit:
                tracking['take_profit_level'] = entry_price * (1 - self.risk_manager.config.take_profit)
            if self.risk_manager.config.trailing_stop:
                lowest = position.get('lowest_price', entry_price)
                tracking['trailing_stop_level'] = lowest * (1 + self.risk_manager.config.trailing_stop)
    
    def _check_risk_exit(self, capital: float, position: Optional[Dict[str, Any]], 
                        prev_candle: pd.Series, current_candle: pd.Series,
                        candle_tracking: Dict[str, Any]) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Check and execute risk management exits"""
        if not position or position['size'] == 0:
            return capital, position
        
        exit_triggered, exit_reason, exit_price = self.risk_manager.check_exit_conditions(
            position, prev_candle
        )
        
        if exit_triggered:
            candle_tracking.update({
                'exit_triggered': True,
                'exit_reason': exit_reason,
                'exit_price': exit_price
            })
            
            capital = self._execute_exit(capital, position, current_candle['o'], exit_reason)
            position = None
        
        return capital, position
    
    def _check_signal_exit(self, capital: float, position: Optional[Dict[str, Any]], 
                          prev_candle: pd.Series, current_candle: pd.Series,
                          candle_tracking: Dict[str, Any]) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Check and execute signal-based exits"""
        if not position or position['size'] == 0:
            return capital, position
        
        # Exit on opposite signal
        should_exit = (
            (position['type'] == 'long' and prev_candle.get('signal', 0) == -1) or
            (position['type'] == 'short' and prev_candle.get('signal', 0) == 1)
        )
        
        if should_exit:
            candle_tracking.update({
                'exit_triggered': True,
                'exit_reason': 'signal',
                'exit_price': current_candle['o']
            })
            
            capital = self._execute_exit(capital, position, current_candle['o'], 'signal')
            position = None
        
        return capital, position
    
    def _check_position_entry(self, capital: float, position: Optional[Dict[str, Any]], 
                            prev_candle: pd.Series, current_candle: pd.Series,
                            candle_tracking: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check and execute new position entries"""
        if position or prev_candle.get('signal', 0) == 0:
            return position
        
        # Calculate position size based on signal strength
        signal_strength = prev_candle.get('signal_strength', 1.0)
        position_capital = capital * self.config.trade_size * signal_strength
        position_size = position_capital / current_candle['o']
        position_type = 'long' if prev_candle['signal'] == 1 else 'short'
        
        # Create new position
        new_position = {
            'entry_time': current_candle.name,
            'entry_price': current_candle['o'],
            'size': position_size,
            'type': position_type,
            'highest_price': current_candle['o'],
            'lowest_price': current_candle['o']
        }
        
        self.positions.append(new_position)
        
        # Update tracking
        candle_tracking.update({
            'position_type': position_type,
            'position_size': position_size,
            'position_entry': current_candle['o'],
            'position_entry_time': current_candle.name,
            'highest_price': current_candle['o'],
            'lowest_price': current_candle['o']
        })
        
        return new_position
    
    def _execute_exit(self, capital: float, position: Dict[str, Any], 
                     exit_price: float, exit_reason: str) -> float:
        """Execute position exit and record trade"""
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
            'exit_time': self.eth_data.index[len(self.detailed_tracking)],  # Current timestamp
            'exit_price': exit_price,
            'size': position['size'],
            'type': position['type'],
            'pnl': pnl,
            'pnl_pct': pnl / (position['size'] * position['entry_price']),
            'exit_reason': exit_reason
        }
        self.trades.append(trade)
        
        return capital
    
    def _close_final_position(self, capital: float, position: Dict[str, Any]) -> float:
        """Close any remaining open position at the end of backtest"""
        last_candle = self.eth_data.iloc[-1]
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
        if self.detailed_tracking:
            self.detailed_tracking[-1].update({
                'exit_triggered': True,
                'exit_reason': 'end_of_data',
                'exit_price': exit_price
            })
        
        return capital
    
    def _calculate_performance_metrics(self, final_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            metrics_calculator = PerformanceMetricsCalculator(
                self.trades, self.equity_curve, self.eth_data, 
                self.config.initial_capital, final_capital, self.config
            )
            return metrics_calculator.calculate_all_metrics()
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise BacktestError(f"Failed to calculate metrics: {str(e)}") from e
    
    def _save_results(self, metrics: Dict[str, Any]) -> None:
        """Save backtest results to files"""
        if not self.output_dir:
            return
            
        try:
            self._save_detailed_tracking()
            self._save_performance_metrics(metrics)
            logger.info("Backtest results saved successfully")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def _save_detailed_tracking(self) -> None:
        """Save detailed trade tracking to CSV"""
        if not self.detailed_tracking:
            return
            
        tracking_df = pd.DataFrame(self.detailed_tracking)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        detailed_tracking_file = os.path.join(
            self.output_dir, 
            f"btc_eth_detailed_tracking_{timestamp}.csv"
        )
        tracking_df.to_csv(detailed_tracking_file, index=False)
        logger.info(f"Detailed tracking saved to {detailed_tracking_file}")
    
    def _save_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save performance metrics to CSV"""
        metrics_data = []
        for key, value in metrics.items():
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metrics_file = os.path.join(
            self.output_dir, 
            f"btc_eth_performance_metrics_{timestamp}.csv"
        )
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Performance metrics saved to {metrics_file}")

# Performance Metrics Calculator (mostly unchanged)
class PerformanceMetricsCalculator:
    """Calculate performance metrics for backtests"""
    
    def __init__(self, trades: List[Dict[str, Any]], equity_curve: pd.DataFrame,
                 data: pd.DataFrame, initial_capital: float, final_capital: float,
                 config: BacktestConfig):
        self.trades = trades
        self.equity_curve = equity_curve
        self.data = data
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.config = config
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all performance metrics"""
        # Basic metrics
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        period_delta = end_date - start_date
        
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        
        # Benchmark return (ETH buy and hold)
        start_price = self.data.iloc[0]['c']
        end_price = self.data.iloc[-1]['c']
        benchmark_return = (end_price / start_price) - 1
        
        logger.info(f"Performance calculation: Initial={self.initial_capital:.2f}, Final={self.final_capital:.2f}, Return={total_return*100:.2f}%")
        
        # If no trades, return basic metrics
        if not self.trades:
            return self._get_basic_metrics(start_date, end_date, period_delta, 
                                         total_return, benchmark_return)
        
        # Calculate trade metrics
        trade_metrics = self._calculate_trade_metrics()
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(total_return, period_delta)
        
        # Calculate additional optimization metrics
        optimization_metrics = self._calculate_optimization_metrics()
        
        # Combine all metrics
        metrics = {
            'start': start_date,
            'end': end_date,
            'period': period_delta,
            'start_value': self.initial_capital,
            'end_value': self.final_capital,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            **trade_metrics,
            **risk_metrics,
            **optimization_metrics
        }
        
        return metrics
    
    def _calculate_trade_metrics(self) -> Dict[str, Any]:
        """Calculate trade-related metrics"""
        trades_df = pd.DataFrame(self.trades)
        
        num_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        # Best and worst trades
        best_trade_pct = winning_trades['pnl_pct'].max() * 100 if not winning_trades.empty else 0
        worst_trade_pct = losing_trades['pnl_pct'].min() * 100 if not losing_trades.empty else 0
        
        # Average trade performance (in absolute terms for optimization)
        avg_winning_trade = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_losing_trade = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        # Percentage versions for reporting
        avg_winning_trade_pct = winning_trades['pnl_pct'].mean() * 100 if not winning_trades.empty else 0
        avg_losing_trade_pct = losing_trades['pnl_pct'].mean() * 100 if not losing_trades.empty else 0
        
        # Trade durations
        avg_winning_duration, avg_losing_duration = self._calculate_trade_durations(
            trades_df, winning_trades, losing_trades
        )
        
        # Profit factor
        total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Expectancy
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Net profit (total PnL)
        net_profit = trades_df['pnl'].sum()
        
        # Max exposure
        max_exposure = trades_df['size'].max() * trades_df['entry_price'].max() / self.initial_capital if not trades_df.empty else 0
        
        return {
            'max_gross_exposure': max_exposure,
            'total_fees_paid': 0,  # No fees in this implementation
            'total_trades': num_trades,
            'total_closed_trades': num_trades,
            'total_open_trades': 0,
            'open_trade_pnl': 0,
            'win_rate': win_rate,
            'best_trade': best_trade_pct,
            'worst_trade': worst_trade_pct,
            'avg_winning_trade': avg_winning_trade_pct,  # Percentage for display
            'avg_losing_trade': avg_losing_trade_pct,    # Percentage for display
            'avg_winner': avg_winning_trade,             # Absolute for optimization
            'avg_loser': avg_losing_trade,               # Absolute for optimization
            'avg_winning_trade_duration': avg_winning_duration,
            'avg_losing_trade_duration': avg_losing_duration,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'net_profit': net_profit,
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict()
        }
    
    def _calculate_optimization_metrics(self) -> Dict[str, Any]:
        """Calculate additional metrics specifically for optimization"""
        optimization_metrics = {}
        
        # Max drawdown inverse (for optimization - higher is better)
        max_drawdown, _ = self._calculate_drawdown_metrics()
        optimization_metrics['max_drawdown_inverse'] = 1 - max_drawdown if max_drawdown > 0 else 1
        
        # Additional risk-adjusted returns
        if len(self.equity_curve) > 1:
            equity = self.equity_curve['equity']
            returns = equity.pct_change().dropna()
            
            if not returns.empty and returns.std() > 0:
                # Risk-adjusted expectancy
                if self.trades:
                    trades_df = pd.DataFrame(self.trades)
                    avg_trade_return = trades_df['pnl'].mean()
                    trade_return_std = trades_df['pnl'].std()
                    risk_adjusted_expectancy = avg_trade_return / trade_return_std if trade_return_std > 0 else 0
                    optimization_metrics['risk_adjusted_expectancy'] = risk_adjusted_expectancy
                
                # Consistency score (percentage of positive months)
                monthly_returns = equity.resample('M').last().pct_change().dropna()
                if not monthly_returns.empty:
                    consistency_score = (monthly_returns > 0).sum() / len(monthly_returns)
                    optimization_metrics['consistency_score'] = consistency_score
        
        return optimization_metrics
    
    def _get_basic_metrics(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
                          period_delta: pd.Timedelta, total_return: float,
                          benchmark_return: float) -> Dict[str, Any]:
        """Return basic metrics when no trades exist"""
        return {
            'start': start_date,
            'end': end_date,
            'period': period_delta,
            'start_value': self.initial_capital,
            'end_value': self.final_capital,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'max_gross_exposure': 0,
            'total_fees_paid': 0,
            'max_drawdown': 0,
            'max_drawdown_duration': pd.Timedelta(0),
            'max_drawdown_inverse': 1,
            'total_trades': 0,
            'total_closed_trades': 0,
            'total_open_trades': 0,
            'open_trade_pnl': 0,
            'win_rate': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'avg_winner': 0,
            'avg_loser': 0,
            'avg_winning_trade_duration': pd.Timedelta(0),
            'avg_losing_trade_duration': pd.Timedelta(0),
            'profit_factor': 0,
            'expectancy': 0,
            'net_profit': 0,
            'sharpe_ratio': 0,
            'calmar_ratio': 0,
            'omega_ratio': 0,
            'sortino_ratio': 0,
            'risk_adjusted_expectancy': 0,
            'consistency_score': 0
        }
    
    def _calculate_trade_durations(self, trades_df: pd.DataFrame, 
                                  winning_trades: pd.DataFrame, 
                                  losing_trades: pd.DataFrame) -> Tuple[pd.Timedelta, pd.Timedelta]:
        """Calculate average trade durations"""
        try:
            # Ensure datetime types
            if not pd.api.types.is_datetime64_dtype(trades_df['entry_time']):
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            if not pd.api.types.is_datetime64_dtype(trades_df['exit_time']):
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
            
            # Recalculate winning/losing trades with duration
            winning_trades = trades_df[trades_df['pnl'] > 0].copy()
            losing_trades = trades_df[trades_df['pnl'] <= 0].copy()
            
            avg_winning_duration = winning_trades['duration'].mean() if not winning_trades.empty else pd.Timedelta(0)
            avg_losing_duration = losing_trades['duration'].mean() if not losing_trades.empty else pd.Timedelta(0)
            
            return avg_winning_duration, avg_losing_duration
            
        except Exception as e:
            logger.warning(f"Could not calculate trade durations: {e}")
            return pd.Timedelta(0), pd.Timedelta(0)
    
    def _calculate_risk_metrics(self, total_return: float, period_delta: pd.Timedelta) -> Dict[str, Any]:
        """Calculate risk-related metrics"""
        # Drawdown metrics
        max_drawdown, max_drawdown_duration = self._calculate_drawdown_metrics()
        
        # Return-based metrics
        if len(self.equity_curve) > 1:
            sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio = self._calculate_return_metrics(
                total_return, period_delta, max_drawdown
            )
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = omega_ratio = 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio
        }
    
    def _calculate_drawdown_metrics(self) -> Tuple[float, pd.Timedelta]:
        """Calculate drawdown metrics"""
        if self.equity_curve.empty:
            return 0, pd.Timedelta(0)
        
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
        
        # Calculate maximum duration
        max_duration = pd.Timedelta(0)
        for period in drawdown_periods:
            start_time = self.equity_curve.iloc[period['start']]['timestamp']
            end_time = self.equity_curve.iloc[period['end']]['timestamp']
            duration = end_time - start_time
            max_duration = max(max_duration, duration)
        
        return max_drawdown, max_duration
    
    def _calculate_return_metrics(self, total_return: float, period_delta: pd.Timedelta,
                                 max_drawdown: float) -> Tuple[float, float, float, float]:
        """Calculate return-based risk metrics"""
        equity = self.equity_curve['equity']
        returns = equity.pct_change().dropna()
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Sharpe ratio
        sharpe_ratio = (mean_return / std_return) * self.config.annualization_factor if std_return > 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std()
        sortino_ratio = (mean_return / downside_deviation) * self.config.annualization_factor if downside_deviation > 0 else 0
        
        # Calmar ratio
        years = period_delta.days / 365 if hasattr(period_delta, 'days') else 1
        annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Omega ratio
        threshold = 0
        omega_numerator = returns[returns > threshold].sum()
        omega_denominator = abs(returns[returns < threshold].sum())
        omega_ratio = omega_numerator / omega_denominator if omega_denominator > 0 else float('inf')
        
        return sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio

# Backtest Visualizer Class
class BacktestVisualizer:
    """Handles visualization of BTC Volume → ETH strategy results"""
    
    def __init__(self, eth_data: pd.DataFrame, backtest_results: Dict[str, Any], 
                 output_dir: Optional[str] = None):
        self.eth_data = eth_data
        self.results = backtest_results
        self.output_dir = output_dir
        logger.info("Initialized BacktestVisualizer for BTC Volume → ETH Strategy")
        
    def plot_price_and_signals(self) -> go.Figure:
        """Plot ETH price chart with BTC indicators and signals"""
        try:
            # Sample data for performance if too large
            data_sample = self._sample_data_if_needed()
            
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=('ETH Price Chart', 'BTC Volume Analysis', 'BTC Momentum')
            )
            
            self._add_eth_candlestick_chart(fig, data_sample)
            self._add_btc_volume_indicators(fig, data_sample)
            self._add_btc_momentum_indicators(fig, data_sample)
            self._add_trade_markers(fig)
            self._update_price_chart_layout(fig)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price chart: {str(e)}")
            raise BacktestError(f"Failed to create price chart: {str(e)}") from e
    
    def _sample_data_if_needed(self) -> pd.DataFrame:
        """Sample data if too large for performance"""
        if len(self.eth_data) > 10000:
            return self.eth_data.iloc[::int(len(self.eth_data)/5000)]
        return self.eth_data
    
    def _add_eth_candlestick_chart(self, fig: go.Figure, data: pd.DataFrame) -> None:
        """Add ETH candlestick chart to figure"""
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['o'],
                high=data['h'],
                low=data['l'],
                close=data['c'],
                name='ETH OHLC'
            ),
            row=1, col=1
        )
    
    def _add_btc_volume_indicators(self, fig: go.Figure, data: pd.DataFrame) -> None:
        """Add BTC volume indicators to subplot"""
        # BTC Volume
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.get('btc_volume', []),
                name="BTC Volume",
                line=dict(color='blue', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.3)'
            ),
            row=2, col=1
        )
        
        # BTC Volume Average
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.get('btc_volume_avg', []),
                name="BTC Volume Average",
                line=dict(color='red', width=1.5, dash='dash')
            ),
            row=2, col=1
        )
        
        # Volume surge markers
        volume_surge_data = data[data.get('btc_volume_surge', 0) == 1]
        if not volume_surge_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=volume_surge_data.index,
                    y=volume_surge_data.get('btc_volume', []),
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='triangle-up'),
                    name="Volume Surge",
                    showlegend=True
                ),
                row=2, col=1
            )
    
    def _add_btc_momentum_indicators(self, fig: go.Figure, data: pd.DataFrame) -> None:
        """Add BTC momentum indicators to subplot"""
        # BTC Momentum line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.get('btc_momentum', []),
                name='BTC Momentum',
                line=dict(color='purple', width=1.5)
            ),
            row=3, col=1
        )
        
        # Momentum threshold lines
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[0.01, 0.01],  # 1% threshold
                mode='lines',
                line=dict(color='green', width=1, dash='dash'),
                name='Bullish Threshold',
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[-0.01, -0.01],  # -1% threshold
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name='Bearish Threshold',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Zero line
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dot'),
                showlegend=False
            ),
            row=3, col=1
        )
    
    def _add_trade_markers(self, fig: go.Figure) -> None:
        """Add trade entry and exit markers"""
        trades_df = pd.DataFrame(self.results['trades'])
        if trades_df.empty:
            return
        
        # Convert to datetime if needed
        if isinstance(trades_df['entry_time'].iloc[0], str):
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Add entry and exit markers
        for i, trade in trades_df.iterrows():
            entry_color = 'green' if trade['type'] == 'long' else 'red'
            exit_color = {
                'signal': 'blue',
                'take_profit': 'green',
                'stop_loss': 'red',
                'trailing_stop': 'purple',
                'end_of_data': 'black'
            }.get(trade['exit_reason'], 'gray')
            
            # Entry marker
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time']],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(size=10, color=entry_color, symbol='circle'),
                    name=f"Entry ({trade['type']})",
                    showlegend=i==0
                ),
                row=1, col=1
            )
            
            # Exit marker
            fig.add_trace(
                go.Scatter(
                    x=[trade['exit_time']],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(size=10, color=exit_color, symbol='x'),
                    name=f"Exit ({trade['exit_reason']})",
                    showlegend=i==0
                ),
                row=1, col=1
            )
    
    def _update_price_chart_layout(self, fig: go.Figure) -> None:
        """Update layout for price chart"""
        fig.update_layout(
            title='BTC Volume → ETH Strategy Analysis',
            xaxis3_title='Date',
            yaxis_title='ETH Price',
            yaxis2_title='BTC Volume',
            yaxis3_title='BTC Momentum',
            xaxis_rangeslider_visible=False,
            height=900,
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
    
    def plot_equity_curve(self) -> go.Figure:
        """Plot equity curve and drawdown"""
        try:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                subplot_titles=('Equity Curve', 'Drawdown'),
                row_heights=[0.7, 0.3]
            )
            
            # Add equity curve
            fig.add_trace(
                go.Scatter(
                    x=self.results['equity_curve']['timestamp'],
                    y=self.results['equity_curve']['equity'],
                    name='Strategy Equity',
                    line=dict(color='rgb(75, 192, 192)', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(75, 192, 192, 0.2)'
                ),
                row=1, col=1
            )
            
            # Add benchmark (ETH buy and hold)
            if 'benchmark_data' in self.results and self.results['benchmark_data'] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=self.results['benchmark_data']['timestamp'],
                        y=self.results['benchmark_data']['value'],
                        name='ETH Buy & Hold',
                        line=dict(color='orange', width=2, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Calculate and add drawdown
            equity = self.results['equity_curve']['equity']
            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max * 100
            
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
            
            # Add zero line for drawdown
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
            
            fig.update_layout(
                title='BTC Volume → ETH Strategy: Equity Curve and Drawdown',
                xaxis2_title='Date',
                yaxis_title='Equity',
                yaxis2_title='Drawdown (%)',
                height=600,
                width=1200,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating equity chart: {str(e)}")
            raise BacktestError(f"Failed to create equity chart: {str(e)}") from e
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of backtest metrics"""
        try:
            metrics = self.results['metrics']
            
            summary_data = {
                'Metric': [
                    'Start', 'End', 'Period', 'Start Value', 'End Value',
                    'Total Return [%]', 'Benchmark Return [%]', 'Max Gross Exposure [%]',
                    'Total Fees Paid', 'Max Drawdown [%]', 'Max Drawdown Duration',
                    'Total Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
                    'Profit Factor', 'Expectancy', 'Sharpe Ratio', 'Sortino Ratio'
                ],
                'Value': [
                    metrics['start'].strftime('%Y-%m-%d %H:%M:%S'),
                    metrics['end'].strftime('%Y-%m-%d %H:%M:%S'),
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
                    f"{metrics['win_rate']*100:.6f}",
                    f"{metrics['best_trade']:.6f}",
                    f"{metrics['worst_trade']:.6f}",
                    f"{metrics['profit_factor']:.6f}",
                    f"{metrics['expectancy']:.6f}",
                    f"{metrics['sharpe_ratio']:.6f}",
                    f"{metrics['sortino_ratio']:.6f}"
                ]
            }
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logger.error(f"Error creating summary table: {str(e)}")
            raise BacktestError(f"Failed to create summary table: {str(e)}") from e
    
    def save_trade_log(self, btc_symbol: str, eth_symbol: str, timeframe: str) -> None:
        """Save trade log to CSV"""
        if not self.output_dir:
            return
            
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            trades_df = pd.DataFrame(self.results['trades'])
            
            trades_file = os.path.join(
                self.output_dir, 
                f"{btc_symbol}_{eth_symbol}_{timeframe}_btc_volume_eth_trades_{timestamp}.csv"
            )
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Trade log saved to {trades_file}")
            
        except Exception as e:
            logger.error(f"Error saving trade log: {str(e)}")

# Main execution functions
def run_btc_volume_eth_backtest(
    btc_symbol: str = 'btcusd', 
    eth_symbol: str = 'ethusd',
    timeframe: str = '10m',
    strategy_config: Optional[StrategyConfig] = None,
    risk_config: Optional[RiskConfig] = None,
    backtest_config: Optional[BacktestConfig] = None
) -> Dict[str, Any]:
    """
    Run a complete BTC Volume → ETH backtest.
    
    Args:
        btc_symbol: BTC symbol to test
        eth_symbol: ETH symbol to test
        timeframe: Timeframe to test
        strategy_config: Strategy configuration
        risk_config: Risk management configuration
        backtest_config: Backtest configuration
        
    Returns:
        Complete backtest results
    """
    try:
        # Use default configurations if not provided
        if strategy_config is None:
            strategy_config = StrategyConfig()
        if risk_config is None:
            risk_config = RiskConfig()
        if backtest_config is None:
            backtest_config = BacktestConfig()
        
        logger.info(f"Running BTC Volume → ETH backtest for {btc_symbol.upper()} → {eth_symbol.upper()} on {timeframe} timeframe")
        logger.info(f"Strategy config: {strategy_config}")
        logger.info(f"Risk config: {risk_config}")
        
        # Load and prepare data
        data_handler = DataHandler()
        btc_data, eth_data = data_handler.load_paired_data(btc_symbol, eth_symbol, timeframe, strategy_config.strategy_type)
        
        # Resample data if needed
        if timeframe == '10m':
            btc_data = data_handler.resample_data(btc_data, '10T')
            eth_data = data_handler.resample_data(eth_data, '10T')
            
            # Re-align after resampling
            btc_data, eth_data = data_handler._align_timestamps(btc_data, eth_data)
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Initialize strategy and risk manager
        strategy = BTCVolumeETHStrategy(strategy_config)
        risk_manager = RiskManager(risk_config)
        
        # Calculate indicators and signals
        eth_data = strategy.calculate_indicators(btc_data, eth_data)
        eth_data = strategy.generate_signals(btc_data, eth_data)
        
        # Run backtest
        backtest_engine = BacktestEngine(
            eth_data, strategy, risk_manager, backtest_config, 
            data_handler.output_strategy_dir
        )
        results = backtest_engine.run_backtest()
        
        logger.info("BTC Volume → ETH backtest completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error running BTC Volume → ETH backtest: {str(e)}")
        raise BacktestError(f"Backtest failed: {str(e)}") from e

def create_comprehensive_btc_eth_report(
    btc_symbol: str = 'btcusd',
    eth_symbol: str = 'ethusd',
    timeframe: str = '10m',
    strategy_config: Optional[StrategyConfig] = None,
    risk_config: Optional[RiskConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
    save_charts: bool = True
) -> Dict[str, Any]:
    """
    Create a comprehensive BTC Volume → ETH backtest report with all visualizations.
    
    Args:
        btc_symbol: BTC symbol to test
        eth_symbol: ETH symbol to test
        timeframe: Timeframe to test
        strategy_config: Strategy configuration
        risk_config: Risk management configuration
        backtest_config: Backtest configuration
        save_charts: Whether to save charts to files
        
    Returns:
        Complete backtest report with charts and metrics
    """
    try:
        logger.info(f"Creating comprehensive BTC Volume → ETH backtest report")
        
        # Run backtest
        results = run_btc_volume_eth_backtest(
            btc_symbol=btc_symbol,
            eth_symbol=eth_symbol,
            timeframe=timeframe,
            strategy_config=strategy_config,
            risk_config=risk_config,
            backtest_config=backtest_config
        )
        
        # Load data for visualization
        data_handler = DataHandler()
        btc_data, eth_data = data_handler.load_paired_data(btc_symbol, eth_symbol, timeframe, 
                                                          strategy_config.strategy_type if strategy_config else 'btc_volume_eth')
        
        if timeframe == '10m':
            btc_data = data_handler.resample_data(btc_data, '10T')
            eth_data = data_handler.resample_data(eth_data, '10T')
            btc_data, eth_data = data_handler._align_timestamps(btc_data, eth_data)
        
        # Create strategy and calculate indicators
        if strategy_config is None:
            strategy_config = StrategyConfig()
        
        strategy = BTCVolumeETHStrategy(strategy_config)
        eth_data = strategy.calculate_indicators(btc_data, eth_data)
        eth_data = strategy.generate_signals(btc_data, eth_data)
        
        # Create visualizer
        visualizer = BacktestVisualizer(eth_data, results, data_handler.output_strategy_dir)
        
        # Generate all charts
        price_chart = visualizer.plot_price_and_signals()
        equity_chart = visualizer.plot_equity_curve()
        summary_table = visualizer.create_summary_table()
        
        # Save charts if requested
        if save_charts and data_handler.output_strategy_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            price_chart_file = os.path.join(
                data_handler.output_strategy_dir, 
                f"{btc_symbol}_{eth_symbol}_price_chart_{timestamp}.html"
            )
            equity_chart_file = os.path.join(
                data_handler.output_strategy_dir, 
                f"{btc_symbol}_{eth_symbol}_equity_chart_{timestamp}.html"
            )
            
            price_chart.write_html(price_chart_file)
            equity_chart.write_html(equity_chart_file)
            
            # Save trade log and summary
            visualizer.save_trade_log(btc_symbol, eth_symbol, timeframe)
            
            logger.info(f"Charts saved to {data_handler.output_strategy_dir}")
        
        # Print summary
        logger.info("\nBTC Volume → ETH Strategy Backtest Summary:")
        logger.info("\n" + summary_table.to_string(index=False))
        
        return {
            'backtest_results': results,
            'eth_data': eth_data,
            'price_chart': price_chart,
            'equity_chart': equity_chart,
            'summary_table': summary_table,
            'output_directory': data_handler.output_strategy_dir
        }
        
    except Exception as e:
        logger.error(f"Error creating comprehensive BTC Volume → ETH report: {str(e)}")
        raise BacktestError(f"Failed to create report: {str(e)}") from e

# Example usage and main execution
def main():
    """Main function demonstrating BTC Volume → ETH strategy"""
    try:
        logger.info("Starting BTC Volume → ETH Strategy Backtest")
        
        # Example: Basic backtest with custom configurations
        logger.info("\n" + "="*50)
        logger.info("BTC Volume → ETH Strategy Backtest")
        logger.info("="*50)
        
        strategy_config = StrategyConfig(
            volume_window=20,
            volume_threshold=2.0,
            momentum_periods=3,
            momentum_threshold=0.01,
            signal_delay=2,
            strategy_type='btc_volume_eth'
        )
        
        risk_config = RiskConfig(
            trailing_stop=0.02,
            position_size=1.0
        )
        
        backtest_config = BacktestConfig(
            initial_capital=10000,
            trade_size=1.0
        )
        
        # Run comprehensive report
        comprehensive_report = create_comprehensive_btc_eth_report(
            btc_symbol='btcusd',
            eth_symbol='ethusd',
            timeframe='10m',
            strategy_config=strategy_config,
            risk_config=risk_config,
            backtest_config=backtest_config,
            save_charts=True
        )
        
        logger.info("\n" + "="*50)
        logger.info("BTC Volume → ETH Strategy Analysis completed successfully!")
        logger.info("="*50)
        
        return comprehensive_report
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Run main analysis
    try:
        results = main()
        logger.info("BTC Volume → ETH Strategy analysis completed successfully!")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise