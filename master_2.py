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
    """Configuration for trading strategy parameters"""
    short_window: int = 12
    long_window: int = 26
    signal_window: int = 9
    strategy_type: str = 'buy_sell'
    
    def __post_init__(self) -> None:
        self._validate()
    
    def _validate(self) -> None:
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")
        if self.short_window <= 0 or self.long_window <= 0 or self.signal_window <= 0:
            raise ValueError("All window values must be positive")
        if self.strategy_type not in ['buy_hold', 'buy_sell', 'reversal']:
            raise ValueError("Strategy type must be 'buy_hold', 'buy_sell', or 'reversal'")

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

# @dataclass
# class WalkForwardConfig:
#     """Configuration for walk-forward testing"""
#     training_years: int = 1
#     testing_months: int = 3
#     step_months: int = 3
#     optimization_metric: str = 'sharpe_ratio'
#     max_workers: Optional[int] = None
    
#     def __post_init__(self) -> None:
#         if self.max_workers is None:
#             self.max_workers = os.cpu_count()
#         self._validate()
    
#     def _validate(self) -> None:
#         if self.training_years <= 0:
#             raise ValueError("Training years must be positive")
#         if self.testing_months <= 0:
#             raise ValueError("Testing months must be positive")
#         if self.step_months <= 0:
#             raise ValueError("Step months must be positive")
#         if self.optimization_metric not in ['sharpe_ratio', 'total_return', 'profit_factor']:
#             raise ValueError("Optimization metric must be 'sharpe_ratio', 'total_return', or 'profit_factor'")

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
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on indicators"""
        pass

# Data Handler Class
class DataHandler:
    """Handles data loading, validation, and preprocessing"""
    
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
        
    def load_data(self, symbol: str, timeframe: str = '10m', 
                  strategy_type: str = 'buy_sell') -> pd.DataFrame:
        """
        Load data for a specific symbol and timeframe.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            strategy_type: Strategy type for directory naming
            
        Returns:
            Processed dataframe with OHLCV data
            
        Raises:
            DataError: If data file is not found or invalid
        """
        try:
            self.output_base_dir, self.output_strategy_dir = get_output_directory(
                symbol, timeframe, strategy_type
            )
            self.symbol = symbol
            self.timeframe = timeframe
            
            filepath = os.path.join(self.base_dir, f'data/{symbol}_{timeframe}.csv')
            
            if not os.path.exists(filepath):
                raise DataError(f"Data file not found: {filepath}")
                
            df = pd.read_csv(filepath)
            df = self._process_data(df)
            self._check_data_integrity(df)
            self._save_data_integrity_issues()
            
            logger.info(f"Successfully loaded {len(df)} rows of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            raise DataError(f"Failed to load data: {str(e)}") from e
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into required format"""
        df['time_utc'] = pd.to_datetime(df['time_utc'])
        df['time_est'] = pd.to_datetime(df['time_est'])
        df.set_index('time_utc', inplace=True)
        return df
    
    def _check_data_integrity(self, df: pd.DataFrame) -> None:
        """Check data integrity and log issues"""
        self._check_missing_values(df)
        self._check_duplicate_timestamps(df)
        self._check_uniform_intervals(df)
    
    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """Check for missing values in OHLCV columns"""
        ohlcv_cols = ['o', 'h', 'l', 'c', 'v']
        for col in ohlcv_cols:
            missing = df[df[col].isnull()]
            if not missing.empty:
                self.data_integrity_issues['missing_values'][col] = missing.index.tolist()
                logger.warning(f"Found {len(missing)} missing values in '{col}' column")
    
    def _check_duplicate_timestamps(self, df: pd.DataFrame) -> None:
        """Check for duplicate timestamps"""
        duplicates = df.index[df.index.duplicated()].tolist()
        if duplicates:
            self.data_integrity_issues['duplicate_timestamps'] = duplicates
            logger.warning(f"Found {len(duplicates)} duplicate timestamps")
    
    def _check_uniform_intervals(self, df: pd.DataFrame) -> None:
        """Check for uniform time intervals"""
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=10)
        non_uniform = time_diffs[time_diffs != expected_diff]
        if not non_uniform.empty:
            self.data_integrity_issues['non_uniform_intervals'] = [
                {'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'), 
                 'interval': diff.total_seconds() / 60} 
                for idx, diff in non_uniform.items()
            ]
            logger.warning(f"Found {len(non_uniform)} non-uniform time intervals")
    
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
        for col, timestamps in self.data_integrity_issues['missing_values'].items():
            for ts in timestamps:
                missing_values_data.append({
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'column': col
                })
                
        if missing_values_data:
            missing_df = pd.DataFrame(missing_values_data)
            missing_file = os.path.join(
                self.output_strategy_dir, 
                f"{self.symbol}_{self.timeframe}_missing_values_{timestamp}.csv"
            )
            missing_df.to_csv(missing_file, index=False)
            logger.info(f"Missing values report saved to {missing_file}")
    
    def _save_duplicate_timestamps_report(self, timestamp: str) -> None:
        """Save duplicate timestamps report"""
        if self.data_integrity_issues['duplicate_timestamps']:
            duplicates_df = pd.DataFrame({
                'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') 
                             for ts in self.data_integrity_issues['duplicate_timestamps']]
            })
            duplicates_file = os.path.join(
                self.output_strategy_dir, 
                f"{self.symbol}_{self.timeframe}_duplicate_timestamps_{timestamp}.csv"
            )
            duplicates_df.to_csv(duplicates_file, index=False)
            logger.info(f"Duplicate timestamps report saved to {duplicates_file}")
    
    def _save_non_uniform_intervals_report(self, timestamp: str) -> None:
        """Save non-uniform intervals report"""
        if self.data_integrity_issues['non_uniform_intervals']:
            non_uniform_df = pd.DataFrame(self.data_integrity_issues['non_uniform_intervals'])
            non_uniform_file = os.path.join(
                self.output_strategy_dir, 
                f"{self.symbol}_{self.timeframe}_non_uniform_intervals_{timestamp}.csv"
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

# MACD Strategy Implementation
class MACDStrategy(TradingStrategy):
    """MACD strategy implementation"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD and signal line"""
        try:
            result = data.copy()
            
            # Calculate EMAs
            result['ema_short'] = result['c'].ewm(
                span=self.config.short_window, adjust=False
            ).mean()
            result['ema_long'] = result['c'].ewm(
                span=self.config.long_window, adjust=False
            ).mean()
            
            # Calculate MACD and signal line
            result['macd'] = result['ema_short'] - result['ema_long']
            result['macd_signal'] = result['macd'].ewm(
                span=self.config.signal_window, adjust=False
            ).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
            
            logger.debug("MACD indicators calculated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating MACD indicators: {str(e)}")
            raise StrategyError(f"Failed to calculate indicators: {str(e)}") from e
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on MACD crossovers"""
        try:
            if 'macd' not in data.columns or 'macd_signal' not in data.columns:
                data = self.calculate_indicators(data)
            
            result = data.copy()
            result['signal'] = 0
            
            # Calculate crossover points
            result['crossover'] = self._calculate_crossovers(result)
            
            # Apply strategy logic
            result['signal'] = self._apply_strategy_logic(result['crossover'])
            
            logger.debug(f"Generated signals using {self.config.strategy_type} strategy")
            return result
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise StrategyError(f"Failed to generate signals: {str(e)}") from e
    
    def _calculate_crossovers(self, data: pd.DataFrame) -> pd.Series:
        """Calculate MACD crossover points"""
        return np.where(
            (data['macd'] > data['macd_signal']) & 
            (data['macd'].shift(1) <= data['macd_signal'].shift(1)),
            1,  # Bullish crossover
            np.where(
                (data['macd'] < data['macd_signal']) & 
                (data['macd'].shift(1) >= data['macd_signal'].shift(1)),
                -1,  # Bearish crossover
                0  # No crossover
            )
        )
    
    def _apply_strategy_logic(self, crossover: pd.Series) -> pd.Series:
        """Apply strategy-specific logic to crossover signals"""
        if self.config.strategy_type == 'buy_hold':
            return crossover.apply(lambda x: 1 if x == 1 else 0)
        elif self.config.strategy_type == 'buy_sell':
            return crossover
        elif self.config.strategy_type == 'reversal':
            return crossover
        else:
            raise StrategyError(f"Unknown strategy type: {self.config.strategy_type}")

# Risk Manager Class
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
# class BacktestEngine:
#     """Core engine for running backtests"""
    
#     def __init__(self, data: pd.DataFrame, strategy: TradingStrategy, 
#                  risk_manager: RiskManager, config: BacktestConfig, 
#                  output_dir: Optional[str] = None):
#         self.data = data
#         self.strategy = strategy
#         self.risk_manager = risk_manager
#         self.config = config
#         self.output_dir = output_dir
        
#         # Initialize results storage
#         self.positions: List[Dict[str, Any]] = []
#         self.trades: List[Dict[str, Any]] = []
#         self.equity_curve: pd.DataFrame = pd.DataFrame()
#         self.detailed_tracking: List[Dict[str, Any]] = []
#         self.benchmark_data: Optional[pd.DataFrame] = None
        
#         logger.info("Initialized BacktestEngine")
        
#     def run_backtest(self) -> Dict[str, Any]:
#         """Run the complete backtest"""
#         try:
#             logger.info("Starting backtest execution")
            
#             # Ensure signals are generated
#             if 'signal' not in self.data.columns:
#                 self.data = self.strategy.generate_signals(self.data)
            
#             # Initialize tracking variables
#             capital = self.config.initial_capital
#             position: Optional[Dict[str, Any]] = None
            
#             # Calculate benchmark
#             benchmark_history = self._calculate_benchmark()
            
#             # Process each candle
#             equity_history = self._process_candles(capital, position, benchmark_history)
            
#             # Store results
#             self.equity_curve = pd.DataFrame(equity_history)
#             self.benchmark_data = pd.DataFrame(benchmark_history)
            
#             # Calculate metrics
#             metrics = self._calculate_performance_metrics(capital)
            
#             # Save results
#             self._save_results(metrics)
            
#             logger.info("Backtest completed successfully")
            
#             return {
#                 'trades': self.trades,
#                 'equity_curve': self.equity_curve,
#                 'benchmark_data': self.benchmark_data,
#                 'metrics': metrics
#             }
            
#         except Exception as e:
#             logger.error(f"Error running backtest: {str(e)}")
#             raise BacktestError(f"Backtest execution failed: {str(e)}") from e
    
    # def _calculate_benchmark(self) -> List[Dict[str, Any]]:
    #     """Calculate benchmark (buy and hold) performance"""
    #     benchmark_start_price = self.data.iloc[0]['c']
    #     benchmark_shares = self.config.initial_capital / benchmark_start_price
    #     return [
    #         {
    #             'timestamp': candle.name,
    #             'value': benchmark_shares * candle['c']
    #         }
    #         for candle in self.data.itertuples()
    #     ]
class BacktestEngine:
    """Core engine for running backtests"""
    
    def __init__(self, data: pd.DataFrame, strategy: TradingStrategy, 
                 risk_manager: RiskManager, config: BacktestConfig, 
                 output_dir: Optional[str] = None):
        self.data = data
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
        
        logger.info("Initialized BacktestEngine")
        
    def run_backtest(self) -> Dict[str, Any]:
        """Run the complete backtest"""
        try:
            logger.info("Starting backtest execution")
            
            # Ensure signals are generated
            if 'signal' not in self.data.columns:
                self.data = self.strategy.generate_signals(self.data)
            
            # Initialize tracking variables
            capital = self.config.initial_capital
            position: Optional[Dict[str, Any]] = None
            
            # Calculate benchmark
            benchmark_history = self._calculate_benchmark()
            
            # Process each candle - FIX: Now returns both equity_history and final_capital
            equity_history, final_capital = self._process_candles(capital, position, benchmark_history)
            
            # Store results
            self.equity_curve = pd.DataFrame(equity_history)
            self.benchmark_data = pd.DataFrame(benchmark_history)
            
            # Calculate metrics - FIX: Pass the actual final_capital
            metrics = self._calculate_performance_metrics(final_capital)
            
            # Save results
            self._save_results(metrics)
            
            logger.info("Backtest completed successfully")
            logger.info(f"Initial Capital: {self.config.initial_capital:.2f}")
            logger.info(f"Final Capital: {final_capital:.2f}")
            logger.info(f"Total Return: {((final_capital - self.config.initial_capital) / self.config.initial_capital) * 100:.2f}%")
            
            return {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'benchmark_data': self.benchmark_data,
                'metrics': metrics,
                'final_capital': final_capital  # Add this for easy access
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise BacktestError(f"Backtest execution failed: {str(e)}") from e
    
    def _process_candles(self, initial_capital: float, initial_position: Optional[Dict[str, Any]],
                        benchmark_history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        """Process each candle in the backtest - FIX: Now returns final capital too"""
        capital = initial_capital
        position = initial_position
        equity_history = []
        
        for i in range(1, len(self.data)):
            prev_candle = self.data.iloc[i-1]
            current_candle = self.data.iloc[i]
            
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
        
        # FIX: Return both equity_history and final capital
        return equity_history, capital

    def _calculate_benchmark(self) -> List[Dict[str, Any]]:
        benchmark_start_price = self.data.iloc[0]['c']
        benchmark_shares = self.config.initial_capital / benchmark_start_price
        benchmark_values = benchmark_shares * self.data['c']
        return [
            {'timestamp': ts, 'value': val}
            for ts, val in zip(self.data.index, benchmark_values)
        ]
        
    # def _process_candles(self, initial_capital: float, initial_position: Optional[Dict[str, Any]],
    #                     benchmark_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """Process each candle in the backtest"""
    #     capital = initial_capital
    #     position = initial_position
    #     equity_history = []
        
    #     for i in range(1, len(self.data)):
    #         prev_candle = self.data.iloc[i-1]
    #         current_candle = self.data.iloc[i]
            
    #         # Track equity
    #         current_equity = self._calculate_current_equity(capital, position, current_candle)
    #         equity_history.append({
    #             'timestamp': current_candle.name,
    #             'equity': current_equity
    #         })
            
    #         # Track detailed information
    #         candle_tracking = self._create_candle_tracking(
    #             current_candle, prev_candle, capital, position, benchmark_history[i]['value']
    #         )
            
    #         # Check risk management exit
    #         capital, position = self._check_risk_exit(capital, position, prev_candle, current_candle, candle_tracking)
            
    #         # Check signal exit
    #         capital, position = self._check_signal_exit(capital, position, prev_candle, current_candle, candle_tracking)
            
    #         # Check new position entry
    #         position = self._check_position_entry(capital, position, prev_candle, current_candle, candle_tracking)
            
    #         self.detailed_tracking.append(candle_tracking)
        
    #     # Close final position if open
    #     if position and position['size'] > 0:
    #         capital = self._close_final_position(capital, position)
        
    #     return equity_history
    
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
            'signal': prev_candle['signal'],
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
        
        should_exit = (
            (position['type'] == 'long' and prev_candle['signal'] == -1) or
            (position['type'] == 'short' and prev_candle['signal'] == 1)
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
        if position or prev_candle['signal'] == 0:
            return position
        
        # For buy_sell strategy, only enter long positions
        if (self.strategy.config.strategy_type == 'buy_sell' and 
            prev_candle['signal'] == -1):
            return position
        
        # Calculate position size
        position_capital = capital * self.config.trade_size
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
            'exit_time': self.data.index[len(self.detailed_tracking)],  # Current timestamp
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
        last_candle = self.data.iloc[-1]
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
                self.trades, self.equity_curve, self.data, 
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
        symbol = self.data['symbol'].iloc[0] if 'symbol' in self.data.columns else 'unknown'
        
        detailed_tracking_file = os.path.join(
            self.output_dir, 
            f"{symbol}_detailed_tracking_{timestamp}.csv"
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
        symbol = self.data['symbol'].iloc[0] if 'symbol' in self.data.columns else 'unknown'
        
        metrics_file = os.path.join(
            self.output_dir, 
            f"{symbol}_performance_metrics_{timestamp}.csv"
        )
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Performance metrics saved to {metrics_file}")

# Performance Metrics Calculator
# class PerformanceMetricsCalculator:
#     """Calculate performance metrics for backtests"""
    
#     def __init__(self, trades: List[Dict[str, Any]], equity_curve: pd.DataFrame,
#                  data: pd.DataFrame, initial_capital: float, final_capital: float,
#                  config: BacktestConfig):
#         self.trades = trades
#         self.equity_curve = equity_curve
#         self.data = data
#         self.initial_capital = initial_capital
#         self.final_capital = final_capital
#         self.config = config
    
#     def calculate_all_metrics(self) -> Dict[str, Any]:
#         """Calculate all performance metrics"""
#         # Basic metrics
#         start_date = self.data.index[0]
#         end_date = self.data.index[-1]
#         period_delta = end_date - start_date
#         total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        
#         # Benchmark return
#         start_price = self.data.iloc[0]['c']
#         end_price = self.data.iloc[-1]['c']
#         benchmark_return = (end_price / start_price) - 1
        
#         # If no trades, return basic metrics
#         if not self.trades:
#             return self._get_basic_metrics(start_date, end_date, period_delta, 
#                                          total_return, benchmark_return)
        
#         # Calculate trade metrics
#         trade_metrics = self._calculate_trade_metrics()
        
#         # Calculate risk metrics
#         risk_metrics = self._calculate_risk_metrics(total_return, period_delta)
        
#         # Combine all metrics
#         metrics = {
#             'start': start_date,
#             'end': end_date,
#             'period': period_delta,
#             'start_value': self.initial_capital,
#             'end_value': self.final_capital,
#             'total_return': total_return,
#             'benchmark_return': benchmark_return,
#             **trade_metrics,
#             **risk_metrics
#         }
        
#         return metrics

# class PerformanceMetricsCalculator:
#     """Calculate performance metrics for backtests"""
    
#     def __init__(self, trades: List[Dict[str, Any]], equity_curve: pd.DataFrame,
#                  data: pd.DataFrame, initial_capital: float, final_capital: float,
#                  config: BacktestConfig):
#         self.trades = trades
#         self.equity_curve = equity_curve
#         self.data = data
#         self.initial_capital = initial_capital
#         self.final_capital = final_capital
#         self.config = config
    
#     def calculate_all_metrics(self) -> Dict[str, Any]:
#         """Calculate all performance metrics"""
#         # Basic metrics
#         start_date = self.data.index[0]
#         end_date = self.data.index[-1]
#         period_delta = end_date - start_date
        
#         # FIX: Ensure proper total return calculation
#         total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        
#         # Benchmark return
#         start_price = self.data.iloc[0]['c']
#         end_price = self.data.iloc[-1]['c']
#         benchmark_return = (end_price / start_price) - 1
        
#         # Log the calculation for debugging
#         logger.info(f"Performance calculation: Initial={self.initial_capital:.2f}, Final={self.final_capital:.2f}, Return={total_return*100:.2f}%")
        
#         # If no trades, return basic metrics
#         if not self.trades:
#             return self._get_basic_metrics(start_date, end_date, period_delta, 
#                                          total_return, benchmark_return)
        
#         # Calculate trade metrics
#         trade_metrics = self._calculate_trade_metrics()
        
#         # Calculate risk metrics
#         risk_metrics = self._calculate_risk_metrics(total_return, period_delta)
        
#         # Combine all metrics
#         metrics = {
#             'start': start_date,
#             'end': end_date,
#             'period': period_delta,
#             'start_value': self.initial_capital,
#             'end_value': self.final_capital,  # FIX: Use actual final capital
#             'total_return': total_return,
#             'benchmark_return': benchmark_return,
#             **trade_metrics,
#             **risk_metrics
#         }
        
#         return metrics
    
#     def _get_basic_metrics(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
#                           period_delta: pd.Timedelta, total_return: float,
#                           benchmark_return: float) -> Dict[str, Any]:
#         """Return basic metrics when no trades exist"""
#         return {
#             'start': start_date,
#             'end': end_date,
#             'period': period_delta,
#             'start_value': self.initial_capital,
#             'end_value': self.final_capital,
#             'total_return': total_return,
#             'benchmark_return': benchmark_return,
#             'max_gross_exposure': 0,
#             'total_fees_paid': 0,
#             'max_drawdown': 0,
#             'max_drawdown_duration': pd.Timedelta(0),
#             'total_trades': 0,
#             'total_closed_trades': 0,
#             'total_open_trades': 0,
#             'open_trade_pnl': 0,
#             'win_rate': 0,
#             'best_trade': 0,
#             'worst_trade': 0,
#             'avg_winning_trade': 0,
#             'avg_losing_trade': 0,
#             'avg_winning_trade_duration': pd.Timedelta(0),
#             'avg_losing_trade_duration': pd.Timedelta(0),
#             'profit_factor': 0,
#             'expectancy': 0,
#             'sharpe_ratio': 0,
#             'calmar_ratio': 0,
#             'omega_ratio': 0,
#             'sortino_ratio': 0
#         }
    
#     def _calculate_trade_metrics(self) -> Dict[str, Any]:
#         """Calculate trade-related metrics"""
#         trades_df = pd.DataFrame(self.trades)
        
#         num_trades = len(trades_df)
#         winning_trades = trades_df[trades_df['pnl'] > 0]
#         losing_trades = trades_df[trades_df['pnl'] <= 0]
        
#         win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
#         # Best and worst trades
#         best_trade_pct = winning_trades['pnl_pct'].max() * 100 if not winning_trades.empty else 0
#         worst_trade_pct = losing_trades['pnl_pct'].min() * 100 if not losing_trades.empty else 0
        
#         # Average trade performance
#         avg_winning_trade_pct = winning_trades['pnl_pct'].mean() * 100 if not winning_trades.empty else 0
#         avg_losing_trade_pct = losing_trades['pnl_pct'].mean() * 100 if not losing_trades.empty else 0
        
#         # Trade durations
#         avg_winning_duration, avg_losing_duration = self._calculate_trade_durations(
#             trades_df, winning_trades, losing_trades
#         )
        
#         # Profit factor
#         total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
#         total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
#         profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
#         # Expectancy
#         avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
#         avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
#         expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
#         # Max exposure
#         max_exposure = trades_df['size'].max() * trades_df['entry_price'].max() / self.initial_capital if not trades_df.empty else 0
        
#         return {
#             'max_gross_exposure': max_exposure,
#             'total_fees_paid': 0,  # No fees in this implementation
#             'total_trades': num_trades,
#             'total_closed_trades': num_trades,
#             'total_open_trades': 0,
#             'open_trade_pnl': 0,
#             'win_rate': win_rate,
#             'best_trade': best_trade_pct,
#             'worst_trade': worst_trade_pct,
#             'avg_winning_trade': avg_winning_trade_pct,
#             'avg_losing_trade': avg_losing_trade_pct,
#             'avg_winning_trade_duration': avg_winning_duration,
#             'avg_losing_trade_duration': avg_losing_duration,
#             'profit_factor': profit_factor,
#             'expectancy': expectancy,
#             'exit_reasons': trades_df['exit_reason'].value_counts().to_dict()
#         }
    
#     def _calculate_trade_durations(self, trades_df: pd.DataFrame, 
#                                   winning_trades: pd.DataFrame, 
#                                   losing_trades: pd.DataFrame) -> Tuple[pd.Timedelta, pd.Timedelta]:
#         """Calculate average trade durations"""
#         try:
#             # Ensure datetime types
#             if not pd.api.types.is_datetime64_dtype(trades_df['entry_time']):
#                 trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
#             if not pd.api.types.is_datetime64_dtype(trades_df['exit_time']):
#                 trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
#             trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
            
#             # Recalculate winning/losing trades with duration
#             winning_trades = trades_df[trades_df['pnl'] > 0].copy()
#             losing_trades = trades_df[trades_df['pnl'] <= 0].copy()
            
#             avg_winning_duration = winning_trades['duration'].mean() if not winning_trades.empty else pd.Timedelta(0)
#             avg_losing_duration = losing_trades['duration'].mean() if not losing_trades.empty else pd.Timedelta(0)
            
#             return avg_winning_duration, avg_losing_duration
            
#         except Exception as e:
#             logger.warning(f"Could not calculate trade durations: {e}")
#             return pd.Timedelta(0), pd.Timedelta(0)
    
#     def _calculate_risk_metrics(self, total_return: float, period_delta: pd.Timedelta) -> Dict[str, Any]:
#         """Calculate risk-related metrics"""
#         # Drawdown metrics
#         max_drawdown, max_drawdown_duration = self._calculate_drawdown_metrics()
        
#         # Return-based metrics
#         if len(self.equity_curve) > 1:
#             sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio = self._calculate_return_metrics(
#                 total_return, period_delta, max_drawdown
#             )
#         else:
#             sharpe_ratio = sortino_ratio = calmar_ratio = omega_ratio = 0
        
#         return {
#             'max_drawdown': max_drawdown,
#             'max_drawdown_duration': max_drawdown_duration,
#             'sharpe_ratio': sharpe_ratio,
#             'sortino_ratio': sortino_ratio,
#             'calmar_ratio': calmar_ratio,
#             'omega_ratio': omega_ratio
#         }
    
#     def _calculate_drawdown_metrics(self) -> Tuple[float, pd.Timedelta]:
#         """Calculate drawdown metrics"""
#         if self.equity_curve.empty:
#             return 0, pd.Timedelta(0)
        
#         equity = self.equity_curve['equity']
#         running_max = equity.cummax()
#         drawdown = (equity - running_max) / running_max
#         max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        
#         # Calculate drawdown duration
#         is_drawdown = (equity < running_max)
#         drawdown_periods = []
#         current_period = None
        
#         for i, in_drawdown in enumerate(is_drawdown):
#             if in_drawdown:
#                 if current_period is None:
#                     current_period = {'start': i}
#             else:
#                 if current_period is not None:
#                     current_period['end'] = i - 1
#                     drawdown_periods.append(current_period)
#                     current_period = None
        
#         # Handle case where still in drawdown at the end
#         if current_period is not None:
#             current_period['end'] = len(is_drawdown) - 1
#             drawdown_periods.append(current_period)
        
#         # Calculate maximum duration
#         max_duration = pd.Timedelta(0)
#         for period in drawdown_periods:
#             start_time = self.equity_curve.iloc[period['start']]['timestamp']
#             end_time = self.equity_curve.iloc[period['end']]['timestamp']
#             duration = end_time - start_time
#             max_duration = max(max_duration, duration)
        
#         return max_drawdown, max_duration
    
#     def _calculate_return_metrics(self, total_return: float, period_delta: pd.Timedelta,
#                                  max_drawdown: float) -> Tuple[float, float, float, float]:
#         """Calculate return-based risk metrics"""
#         equity = self.equity_curve['equity']
#         returns = equity.pct_change().dropna()
        
#         mean_return = returns.mean()
#         std_return = returns.std()
        
#         # Sharpe ratio
#         sharpe_ratio = (mean_return / std_return) * self.config.annualization_factor if std_return > 0 else 0
        
#         # Sortino ratio
#         negative_returns = returns[returns < 0]
#         downside_deviation = negative_returns.std()
#         sortino_ratio = (mean_return / downside_deviation) * self.config.annualization_factor if downside_deviation > 0 else 0
        
#         # Calmar ratio
#         years = period_delta.days / 365 if hasattr(period_delta, 'days') else 1
#         annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
#         calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
#         # Omega ratio
#         threshold = 0
#         omega_numerator = returns[returns > threshold].sum()
#         omega_denominator = abs(returns[returns < threshold].sum())
#         omega_ratio = omega_numerator / omega_denominator if omega_denominator > 0 else float('inf')
        
#         return sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio

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
        
        # FIX: Ensure proper total return calculation
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        
        # Benchmark return
        start_price = self.data.iloc[0]['c']
        end_price = self.data.iloc[-1]['c']
        benchmark_return = (end_price / start_price) - 1
        
        # Log the calculation for debugging
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
    
    # ... rest of the methods remain the same ...
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
    """Handles visualization of backtest results"""
    
    def __init__(self, data: pd.DataFrame, backtest_results: Dict[str, Any], 
                 output_dir: Optional[str] = None):
        self.data = data
        self.results = backtest_results
        self.output_dir = output_dir
        logger.info("Initialized BacktestVisualizer")
        
    def plot_price_and_signals(self) -> go.Figure:
        """Plot price chart with buy/sell signals"""
        try:
            # Sample data for performance if too large
            data_sample = self._sample_data_if_needed()
            
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.7, 0.3],
                subplot_titles=('Price Chart', 'MACD')
            )
            
            self._add_candlestick_chart(fig, data_sample)
            self._add_ema_lines(fig, data_sample)
            self._add_macd_indicators(fig, data_sample)
            self._add_trade_markers(fig)
            self._update_price_chart_layout(fig)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price chart: {str(e)}")
            raise BacktestError(f"Failed to create price chart: {str(e)}") from e
    
    def _sample_data_if_needed(self) -> pd.DataFrame:
        """Sample data if too large for performance"""
        if len(self.data) > 10000:
            return self.data.iloc[::int(len(self.data)/5000)]
        return self.data
    
    def _add_candlestick_chart(self, fig: go.Figure, data: pd.DataFrame) -> None:
        """Add candlestick chart to figure"""
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['o'],
                high=data['h'],
                low=data['l'],
                close=data['c'],
                name='OHLC'
            ),
            row=1, col=1
        )
    
    def _add_ema_lines(self, fig: go.Figure, data: pd.DataFrame) -> None:
        """Add EMA lines to price chart"""
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['ema_short'],
                name="Short EMA",
                line=dict(color='rgba(33, 150, 243, 0.7)', width=1.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['ema_long'],
                name="Long EMA",
                line=dict(color='rgba(255, 152, 0, 0.7)', width=1.5)
            ),
            row=1, col=1
        )
    
    def _add_macd_indicators(self, fig: go.Figure, data: pd.DataFrame) -> None:
        """Add MACD indicators to subplot"""
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['macd'],
                name='MACD Line',
                line=dict(color='blue', width=1.5)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['macd_signal'],
                name='Signal Line',
                line=dict(color='red', width=1.5)
            ),
            row=2, col=1
        )
        
        colors = np.where(data['macd_hist'] >= 0, 'green', 'red')
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['macd_hist'],
                name='MACD Histogram',
                marker_color=colors
            ),
            row=2, col=1
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
                    name='Equity',
                    line=dict(color='rgb(75, 192, 192)', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(75, 192, 192, 0.2)'
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
                title='Equity Curve and Drawdown',
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
    
    def save_trade_log(self, symbol: str, timeframe: str) -> None:
        """Save trade log to CSV"""
        if not self.output_dir:
            return
            
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            trades_df = pd.DataFrame(self.results['trades'])
            
            trades_file = os.path.join(
                self.output_dir, 
                f"{symbol}_{timeframe}_macd_trades_{timestamp}.csv"
            )
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Trade log saved to {trades_file}")
            
        except Exception as e:
            logger.error(f"Error saving trade log: {str(e)}")

# Walk Forward Tester Class
class WalkForwardTester:
    """Performs walk-forward testing of trading strategies"""
    
    def __init__(self, data_handler: DataHandler, config: WalkForwardConfig,
                 symbol: str = 'btcusd', timeframe: str = '10m', 
                 full_data: Optional[pd.DataFrame] = None):
        self.data_handler = data_handler
        self.config = config
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Load or use provided data
        if full_data is None:
            logger.info(f"Loading data for {symbol} on {timeframe} timeframe...")
            self.full_data = self._load_and_prepare_data()
        else:
            self.full_data = full_data
        
        # Setup output directory
        self.base_output_dir = self._setup_output_directory()
        
        # Initialize results storage
        self.results: Dict[str, Any] = {
            'period_results': [],
            'combined_equity_curve': None,
            'optimized_parameters': {},
            'final_metrics': {}
        }
        
        logger.info("Initialized WalkForwardTester")
    
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare data for walk-forward testing"""
        data = self.data_handler.load_data(self.symbol, self.timeframe, 'walk_forward')
        
        if self.timeframe == '10m':
            data = self.data_handler.resample_data(data, '10T')
        else:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")
        
        return data
    
    def _setup_output_directory(self) -> str:
        """Setup output directory for walk-forward results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join('output', f"{self.symbol}_{self.timeframe}_walkforward_{timestamp}")
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    
    def optimize_parameters(self, train_data: pd.DataFrame, 
                          parameter_grid: Dict[str, List[Any]]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Optimize strategy parameters using grid search"""
        try:
            logger.info(f"Optimizing parameters on training data ({len(train_data)} rows)...")
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_grid)
            
            # Create optimization directory
            opt_dir = self._create_optimization_directory()
            
            # Run optimization with parallel processing
            results = self._run_parallel_optimization(train_data, param_combinations, opt_dir)
            
            # Save and return best parameters
            return self._process_optimization_results(results, opt_dir)
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {str(e)}")
            raise BacktestError(f"Parameter optimization failed: {str(e)}") from e
    
    def _generate_parameter_combinations(self, parameter_grid: Dict[str, List[Any]]) -> List[Tuple[Any, ...]]:
        """Generate all valid parameter combinations"""
        param_combinations = list(itertools.product(
            parameter_grid['short_window'],
            parameter_grid['long_window'],
            parameter_grid['signal_window'],
            parameter_grid['strategy_type'],
            parameter_grid['trailing_stop']
        ))
        
        # Filter invalid combinations
        valid_combinations = [
            combo for combo in param_combinations
            if combo[0] < combo[1]  # short_window < long_window
        ]
        
        logger.info(f"Generated {len(valid_combinations)} valid parameter combinations")
        return valid_combinations
    
    def _create_optimization_directory(self) -> str:
        """Create directory for optimization results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        opt_dir = os.path.join(self.base_output_dir, f"optimization_{timestamp}")
        os.makedirs(opt_dir, exist_ok=True)
        return opt_dir
    
    def _run_parallel_optimization(self, train_data: pd.DataFrame, 
                                  param_combinations: List[Tuple[Any, ...]], 
                                  opt_dir: str) -> List[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
        """Run parameter optimization with parallel processing"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_params = {
                executor.submit(
                    self._run_single_optimization,
                    train_data.copy(),
                    *params,
                    opt_dir
                ): params
                for params in param_combinations
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_params), 
                              total=len(future_to_params),
                              desc="Testing parameter combinations"):
                params = future_to_params[future]
                try:
                    metrics = future.result()
                    results.append((params, metrics))
                except Exception as e:
                    logger.warning(f"Error with parameters {params}: {str(e)}")
        
        return results
    
    def _run_single_optimization(self, data: pd.DataFrame, short_window: int, 
                               long_window: int, signal_window: int, 
                               strategy_type: str, trailing_stop: float, 
                               output_dir: str) -> Dict[str, Any]:
        """Run a single backtest for parameter optimization"""
        try:
            # Create configurations
            strategy_config = StrategyConfig(
                short_window=short_window,
                long_window=long_window,
                signal_window=signal_window,
                strategy_type=strategy_type
            )
            
            risk_config = RiskConfig(trailing_stop=trailing_stop)
            backtest_config = BacktestConfig()
            
            # Create strategy and run backtest
            strategy = MACDStrategy(strategy_config)
            risk_manager = RiskManager(risk_config)
            
            data = strategy.calculate_indicators(data)
            data = strategy.generate_signals(data)
            
            backtest_engine = BacktestEngine(
                data, strategy, risk_manager, backtest_config, output_dir=None
            )
            
            results = backtest_engine.run_backtest()
            return results['metrics']
            
        except Exception as e:
            logger.error(f"Error in single optimization: {str(e)}")
            raise
    
    def _process_optimization_results(self, results: List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
                                    opt_dir: str) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Process optimization results and save to CSV"""
        if not results:
            raise ValueError("No valid parameter combinations found")
        
        # Sort by optimization metric
        metric_key = self.config.optimization_metric
        results.sort(key=lambda x: x[1].get(metric_key, 0), reverse=True)
        
        # Save all results
        self._save_optimization_results(results, opt_dir)
        
        # Return best parameters
        best_params, best_metrics = results[0]
        logger.info(f"Best parameters: {best_params} with {metric_key}: {best_metrics[metric_key]:.4f}")
        
        return best_params, best_metrics
    
    def _save_optimization_results(self, results: List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
                                 opt_dir: str) -> None:
        """Save optimization results to CSV"""
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
                'total_trades': metrics['total_trades']
            }
            for params, metrics in results
        ])
        
        opt_results_file = os.path.join(opt_dir, "optimization_results.csv")
        opt_results_df.to_csv(opt_results_file, index=False)
        logger.info(f"Optimization results saved to {opt_results_file}")
    
    def run_walk_forward_test(self, parameter_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Run complete walk-forward testing"""
        try:
            logger.info(f"Starting walk-forward testing for {self.symbol}")
            
            # Setup time windows
            periods = self._calculate_time_periods()
            all_period_results = []
            all_equity_curves = []
            
            # Process each period
            for period_count, (train_start, train_end, test_start, test_end) in enumerate(periods, 1):
                logger.info(f"Period {period_count}: Training {train_start} to {train_end}, Testing {test_start} to {test_end}")
                
                period_result = self._process_single_period(
                    period_count, train_start, train_end, test_start, test_end, parameter_grid
                )
                
                if period_result:
                    all_period_results.append(period_result)
                    
                    # Add equity curve
                    test_equity = period_result['test_results']['equity_curve'].copy()
                    test_equity['period'] = period_count
                    all_equity_curves.append(test_equity)
            
            # Combine results
            if all_equity_curves:
                return self._combine_and_save_results(all_period_results, all_equity_curves)
            else:
                logger.warning("No valid periods found in walk-forward test")
                return {}
                
        except Exception as e:
            logger.error(f"Error in walk-forward testing: {str(e)}")
            raise BacktestError(f"Walk-forward testing failed: {str(e)}") from e
    
    def _calculate_time_periods(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Calculate time periods for walk-forward testing"""
        start_date = self.full_data.index.min()
        end_date = self.full_data.index.max()
        
        training_period = pd.DateOffset(years=self.config.training_years)
        testing_period = pd.DateOffset(months=self.config.testing_months)
        step_period = pd.DateOffset(months=self.config.step_months)
        
        periods = []
        train_start = start_date
        
        while True:
            train_end = train_start + training_period
            test_start = train_end
            test_end = test_start + testing_period
            
            if test_end > end_date:
                break
                
            periods.append((train_start, train_end, test_start, test_end))
            train_start = train_start + step_period
        
        logger.info(f"Generated {len(periods)} time periods for walk-forward testing")
        return periods
    
    def _process_single_period(self, period_count: int, train_start: pd.Timestamp, 
                             train_end: pd.Timestamp, test_start: pd.Timestamp, 
                             test_end: pd.Timestamp, parameter_grid: Dict[str, List[Any]]) -> Optional[Dict[str, Any]]:
        """Process a single period in walk-forward testing"""
        try:
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
                logger.warning(f"Insufficient data for period {period_count}")
                return None
            
            # Create period directory
            period_dir = os.path.join(self.base_output_dir, f"period_{period_count}")
            os.makedirs(period_dir, exist_ok=True)
            
            # Optimize parameters
            best_params, best_metrics = self.optimize_parameters(train_data, parameter_grid)
            
            # Run test with best parameters
            test_results = self._run_test_with_parameters(test_data, best_params, period_dir)
            
            # Save period summary
            self._save_period_summary(period_count, best_params, best_metrics, test_results, period_dir)
            
            return {
                'period': period_count,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'best_params': best_params,
                'train_metrics': best_metrics,
                'test_results': test_results
            }
            
        except Exception as e:
            logger.error(f"Error processing period {period_count}: {str(e)}")
            return None
    
    def _run_test_with_parameters(self, data: pd.DataFrame, best_params: Tuple[Any, ...], 
                                output_dir: str) -> Dict[str, Any]:
        """Run backtest with specific parameters"""
        short_window, long_window, signal_window, strategy_type, trailing_stop = best_params
        
        # Create configurations
        strategy_config = StrategyConfig(
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
            strategy_type=strategy_type
        )
        
        risk_config = RiskConfig(trailing_stop=trailing_stop)
        backtest_config = BacktestConfig()
        
        # Run backtest
        strategy = MACDStrategy(strategy_config)
        risk_manager = RiskManager(risk_config)
        
        data = strategy.calculate_indicators(data)
        data = strategy.generate_signals(data)
        
        backtest_engine = BacktestEngine(
            data, strategy, risk_manager, backtest_config, output_dir
        )
        
        return backtest_engine.run_backtest()
    
    def _save_period_summary(self, period_count: int, best_params: Tuple[Any, ...], 
                           best_metrics: Dict[str, Any], test_results: Dict[str, Any], 
                           period_dir: str) -> None:
        """Save summary for a single period"""
        short_window, long_window, signal_window, strategy_type, trailing_stop = best_params
        
        period_summary = pd.DataFrame([{
            'period': period_count,
            'short_window': short_window,
            'long_window': long_window,
            'signal_window': signal_window,
            'strategy_type': strategy_type,
            'trailing_stop': trailing_stop,
            'train_return': best_metrics['total_return'],
            'train_sharpe': best_metrics['sharpe_ratio'],
            'test_return': test_results['metrics']['total_return'],
            'test_sharpe': test_results['metrics']['sharpe_ratio'],
            'test_win_rate': test_results['metrics']['win_rate'],
            'test_max_drawdown': test_results['metrics']['max_drawdown'],
            'test_trades': test_results['metrics']['total_trades']
        }])
        
        summary_file = os.path.join(period_dir, f"period_{period_count}_summary.csv")
        period_summary.to_csv(summary_file, index=False)
    
    def _combine_and_save_results(self, all_period_results: List[Dict[str, Any]], 
                                all_equity_curves: List[pd.DataFrame]) -> Dict[str, Any]:
        """Combine and save all walk-forward results"""
        # Combine equity curves
        combined_equity = pd.concat(all_equity_curves)
        
        # Calculate combined metrics
        combined_metrics = self._calculate_combined_metrics(combined_equity, all_period_results)
        
        # Save results
        results_dir = os.path.join(self.base_output_dir, "final_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save combined equity curve
        combined_equity_file = os.path.join(results_dir, "combined_equity_curve.csv")
        combined_equity.to_csv(combined_equity_file)
        
        # Save combined metrics
        combined_metrics_df = pd.DataFrame([combined_metrics])
        combined_metrics_file = os.path.join(results_dir, "combined_metrics.csv")
        combined_metrics_df.to_csv(combined_metrics_file, index=False)
        
        # Store results
        self.results = {
            'period_results': all_period_results,
            'combined_equity_curve': combined_equity,
            'final_metrics': combined_metrics
        }
        
        logger.info(f"Walk-forward testing completed. Results saved to {results_dir}")
        logger.info(f"Combined metrics: Return: {combined_metrics['total_return']*100:.2f}%, "
                   f"Sharpe: {combined_metrics['sharpe_ratio']:.2f}")
        
        return self.results
    
    def _calculate_combined_metrics(self, combined_equity: pd.DataFrame, 
                                  period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for combined equity curve"""
        # Basic return metrics
        initial_equity = combined_equity['equity'].iloc[0]
        final_equity = combined_equity['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Drawdown
        running_max = combined_equity['equity'].cummax()
        drawdown = (combined_equity['equity'] - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        
        # Risk metrics
        returns = combined_equity['equity'].pct_change().dropna()
        sharpe_ratio = 0
        if len(returns) > 0 and returns.std() > 0:
            periods_per_day = 144  # 10-minute intervals
            annualization_factor = np.sqrt(365 * periods_per_day)
            sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor
        
        # Trade metrics
        all_trades = []
        for period in period_results:
            all_trades.extend(period['test_results']['trades'])
        
        num_trades = len(all_trades)
        winning_trades = [trade for trade in all_trades if trade['pnl'] > 0]
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': num_trades,
            'win_rate': win_rate
        }

def run_walk_forward_analysis(
    symbol: str = 'btcusd', 
    timeframe: str = '10m',
    wf_config: Optional[WalkForwardConfig] = None
) -> Optional[Dict[str, Any]]:
    """
    Run a complete walk-forward analysis for a given symbol and timeframe.
    
    Args:
        symbol: Symbol to test
        timeframe: Timeframe to test
        wf_config: Walk-forward configuration
        
    Returns:
        Walk-forward analysis results
    """
    try:
        if wf_config is None:
            wf_config = WalkForwardConfig()
        
        logger.info(f"Running walk-forward analysis for {symbol.upper()} on {timeframe} timeframe")
        
        # Initialize data handler and walk-forward tester
        data_handler = DataHandler()
        wf_tester = WalkForwardTester(data_handler, wf_config, symbol=symbol, timeframe=timeframe)
        
        # Define parameter grid
        parameter_grid = {
            'short_window': [8, 12, 16, 20, 24],
            'long_window': [21, 26, 34, 40, 48],    
            'signal_window': [5, 9, 13, 17, 21],
            'strategy_type': ['buy_sell', 'reversal', 'buy_hold'],
            'trailing_stop': [0.01, 0.02, 0.03]
        }
        
        # Run walk-forward test
        results = wf_tester.run_walk_forward_test(parameter_grid)
        
        if results:
            logger.info("Walk-Forward Analysis Results:")
            logger.info(f"Total Return: {results['final_metrics']['total_return']*100:.2f}%")
            logger.info(f"Sharpe Ratio: {results['final_metrics']['sharpe_ratio']:.4f}")
            logger.info(f"Win Rate: {results['final_metrics']['win_rate']*100:.2f}%")
            logger.info(f"Max Drawdown: {results['final_metrics']['max_drawdown']*100:.2f}%")
            logger.info(f"Total Trades: {results['final_metrics']['total_trades']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in walk-forward analysis: {str(e)}")
        raise BacktestError(f"Walk-forward analysis failed: {str(e)}") from e

def create_comprehensive_backtest_report(
    symbol: str = 'btcusd',
    timeframe: str = '10m',
    strategy_config: Optional[StrategyConfig] = None,
    risk_config: Optional[RiskConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
    save_charts: bool = True
) -> Dict[str, Any]:
    """
    Create a comprehensive backtest report with all visualizations.
    
    Args:
        symbol: Symbol to test
        timeframe: Timeframe to test
        strategy_config: Strategy configuration
        risk_config: Risk management configuration
        backtest_config: Backtest configuration
        save_charts: Whether to save charts to files
        
    Returns:
        Complete backtest report with charts and metrics
    """
    try:
        logger.info(f"Creating comprehensive backtest report for {symbol}")
        
        # Run backtest
        results = run_macd_backtest(
            symbol=symbol,
            timeframe=timeframe,
            strategy_config=strategy_config,
            risk_config=risk_config,
            backtest_config=backtest_config
        )
        
        # Load data for visualization
        data_handler = DataHandler()
        data = data_handler.load_data(symbol, timeframe, 
                                     strategy_config.strategy_type if strategy_config else 'buy_sell')
        
        if timeframe == '10m':
            data = data_handler.resample_data(data, '10T')
        
        # Create strategy and calculate indicators
        if strategy_config is None:
            strategy_config = StrategyConfig()
        
        strategy = MACDStrategy(strategy_config)
        data = strategy.calculate_indicators(data)
        data = strategy.generate_signals(data)
        
        # Create visualizer
        visualizer = BacktestVisualizer(data, results, data_handler.output_strategy_dir)
        
        # Generate all charts
        price_chart = visualizer.plot_price_and_signals()
        equity_chart = visualizer.plot_equity_curve()
        summary_table = visualizer.create_summary_table()
        
        # Save charts if requested
        if save_charts and data_handler.output_strategy_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            price_chart_file = os.path.join(
                data_handler.output_strategy_dir, 
                f"{symbol}_price_chart_{timestamp}.html"
            )
            equity_chart_file = os.path.join(
                data_handler.output_strategy_dir, 
                f"{symbol}_equity_chart_{timestamp}.html"
            )
            
            price_chart.write_html(price_chart_file)
            equity_chart.write_html(equity_chart_file)
            
            # Save trade log and summary
            visualizer.save_trade_log(symbol, timeframe)
            
            logger.info(f"Charts saved to {data_handler.output_strategy_dir}")
        
        # Print summary
        logger.info("\nBacktest Summary:")
        logger.info("\n" + summary_table.to_string(index=False))
        
        return {
            'backtest_results': results,
            'data': data,
            'price_chart': price_chart,
            'equity_chart': equity_chart,
            'summary_table': summary_table,
            'output_directory': data_handler.output_strategy_dir
        }
        
    except Exception as e:
        logger.error(f"Error creating comprehensive report: {str(e)}")
        raise BacktestError(f"Failed to create report: {str(e)}") from e

def compare_strategies(
    symbol: str = 'btcusd',
    timeframe: str = '10m',
    strategy_configs: List[StrategyConfig] = None,
    risk_config: Optional[RiskConfig] = None,
    backtest_config: Optional[BacktestConfig] = None
) -> Dict[str, Any]:
    """
    Compare multiple strategy configurations.
    
    Args:
        symbol: Symbol to test
        timeframe: Timeframe to test
        strategy_configs: List of strategy configurations to compare
        risk_config: Risk management configuration
        backtest_config: Backtest configuration
        
    Returns:
        Comparison results
    """
    try:
        if strategy_configs is None:
            strategy_configs = [
                StrategyConfig(strategy_type='buy_hold'),
                StrategyConfig(strategy_type='buy_sell'),
                StrategyConfig(strategy_type='reversal')
            ]
        
        logger.info(f"Comparing {len(strategy_configs)} strategies for {symbol}")
        
        comparison_results = []
        
        for i, config in enumerate(strategy_configs):
            logger.info(f"Running strategy {i+1}/{len(strategy_configs)}: {config.strategy_type}")
            
            try:
                results = run_macd_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_config=config,
                    risk_config=risk_config,
                    backtest_config=backtest_config
                )
                
                comparison_results.append({
                    'strategy_config': config,
                    'results': results,
                    'metrics': results['metrics']
                })
                
            except Exception as e:
                logger.error(f"Error running strategy {config.strategy_type}: {str(e)}")
                continue
        
        # Create comparison table
        if comparison_results:
            comparison_table = _create_strategy_comparison_table(comparison_results)
            
            logger.info("\nStrategy Comparison:")
            logger.info("\n" + comparison_table.to_string(index=False))
            
            return {
                'comparison_results': comparison_results,
                'comparison_table': comparison_table
            }
        else:
            logger.warning("No successful strategy runs for comparison")
            return {}
            
    except Exception as e:
        logger.error(f"Error in strategy comparison: {str(e)}")
        raise BacktestError(f"Strategy comparison failed: {str(e)}") from e

def _create_strategy_comparison_table(comparison_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison table for multiple strategies"""
    comparison_data = []
    
    for result in comparison_results:
        config = result['strategy_config']
        metrics = result['metrics']
        
        comparison_data.append({
            'Strategy Type': config.strategy_type,
            'Short Window': config.short_window,
            'Long Window': config.long_window,
            'Signal Window': config.signal_window,
            'Total Return (%)': f"{metrics['total_return']*100:.2f}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
            'Win Rate (%)': f"{metrics['win_rate']*100:.2f}",
            'Max Drawdown (%)': f"{metrics['max_drawdown']*100:.2f}",
            'Total Trades': metrics['total_trades'],
            'Profit Factor': f"{metrics['profit_factor']:.2f}"
        })
    
    return pd.DataFrame(comparison_data)

# def optimize_single_strategy(
#     symbol: str = 'btcusd',
#     timeframe: str = '10m',
#     optimization_metric: str = 'sharpe_ratio',
#     risk_config: Optional[RiskConfig] = None,
#     backtest_config: Optional[BacktestConfig] = None
# ) -> Dict[str, Any]:
#     """
#     Optimize parameters for a single strategy using grid search.
    
#     Args:
#         symbol: Symbol to test
#         timeframe: Timeframe to test
#         optimization_metric: Metric to optimize for
#         risk_config: Risk management configuration
#         backtest_config: Backtest configuration
        
#     Returns:
#         Optimization results
#     """
#     try:
#         logger.info(f"Optimizing strategy parameters for {symbol}")
        
#         # Load data
#         data_handler = DataHandler()
#         data = data_handler.load_data(symbol, timeframe, 'optimization')
        
#         if timeframe == '10m':
#             data = data_handler.resample_data(data, '10T')
        
#         # Define parameter grid
#         parameter_combinations = []
#         for short_window in [8, 12, 16, 20]:
#             for long_window in [21, 26, 34, 40]:
#                 for signal_window in [5, 9, 13]:
#                     for strategy_type in ['buy_sell', 'reversal']:
#                         if short_window < long_window:
#                             parameter_combinations.append((
#                                 short_window, long_window, signal_window, strategy_type
#                             ))
        
#         logger.info(f"Testing {len(parameter_combinations)} parameter combinations")
        
#         # Run optimization
#         best_params = None
#         best_metric_value = float('-inf')
#         all_results = []
        
#         for params in tqdm(parameter_combinations, desc="Optimizing parameters"):
#             short_window, long_window, signal_window, strategy_type = params
            
#             try:
#                 strategy_config = StrategyConfig(
#                     short_window=short_window,
#                     long_window=long_window,
#                     signal_window=signal_window,
#                     strategy_type=strategy_type
#                 )
                
#                 results = run_macd_backtest(
#                     symbol=symbol,
#                     timeframe=timeframe,
#                     strategy_config=strategy_config,
#                     risk_config=risk_config,
#                     backtest_config=backtest_config
#                 )
                
#                 metric_value = results['metrics'].get(optimization_metric, float('-inf'))
                
#                 all_results.append({
#                     'parameters': params,
#                     'metrics': results['metrics'],
#                     'metric_value': metric_value
#                 })
                
#                 if metric_value > best_metric_value:
#                     best_metric_value = metric_value
#                     best_params = params
                    
#             except Exception as e:
#                 logger.warning(f"Error with parameters {params}: {str(e)}")
#                 continue
        
#         if best_params:
#             logger.info(f"Best parameters: {best_params}")
#             logger.info(f"Best {optimization_metric}: {best_metric_value:.4f}")
            
#             # Create optimization results table
#             optimization_table = _create_optimization_table(all_results, optimization_metric)
            
#             return {
#                 'best_parameters': best_params,
#                 'best_metric_value': best_metric_value,
#                 'all_results': all_results,
#                 'optimization_table': optimization_table
#             }
#         else:
#             logger.warning("No successful optimization runs")
#             return {}
            
#     except Exception as e:
#         logger.error(f"Error in strategy optimization: {str(e)}")
#         raise BacktestError(f"Strategy optimization failed: {str(e)}") from e

def optimize_single_strategy(
    symbol: str = 'btcusd',
    timeframe: str = '10m',
    optimization_metric: str = 'sharpe_ratio',
    risk_config: Optional[RiskConfig] = None,
    backtest_config: Optional[BacktestConfig] = None
) -> Dict[str, Any]:
    """
    Optimize parameters for a single strategy using grid search.
    
    Args:
        symbol: Symbol to test
        timeframe: Timeframe to test
        optimization_metric: Metric to optimize for (now supports more options)
        risk_config: Risk management configuration
        backtest_config: Backtest configuration
        
    Returns:
        Optimization results
    """
    try:
        # Validate optimization metric
        allowed_metrics = [
            'sharpe_ratio', 'total_return', 'profit_factor', 'win_rate', 
            'net_profit', 'avg_winner', 'avg_loser', 'expectancy', 
            'sortino_ratio', 'calmar_ratio', 'max_drawdown_inverse'
        ]
        
        if optimization_metric not in allowed_metrics:
            raise ValueError(f"Optimization metric must be one of: {', '.join(allowed_metrics)}")
        
        logger.info(f"Optimizing strategy parameters for {symbol} using {optimization_metric}")
        
        # Load data
        data_handler = DataHandler()
        data = data_handler.load_data(symbol, timeframe, 'optimization')
        
        if timeframe == '10m':
            data = data_handler.resample_data(data, '10T')
        
        # Define parameter grid
        parameter_combinations = []
        for short_window in [8, 12, 16, 20]:
            for long_window in [21, 26, 34, 40]:
                for signal_window in [5, 9, 13]:
                    for strategy_type in ['buy_sell', 'reversal']:
                        if short_window < long_window:
                            parameter_combinations.append((
                                short_window, long_window, signal_window, strategy_type
                            ))
        
        logger.info(f"Testing {len(parameter_combinations)} parameter combinations")
        
        # Run optimization
        best_params = None
        best_metric_value = float('-inf')
        all_results = []
        
        for params in tqdm(parameter_combinations, desc=f"Optimizing for {optimization_metric}"):
            short_window, long_window, signal_window, strategy_type = params
            
            try:
                strategy_config = StrategyConfig(
                    short_window=short_window,
                    long_window=long_window,
                    signal_window=signal_window,
                    strategy_type=strategy_type
                )
                
                results = run_macd_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_config=strategy_config,
                    risk_config=risk_config,
                    backtest_config=backtest_config
                )
                
                metric_value = results['metrics'].get(optimization_metric, float('-inf'))
                
                all_results.append({
                    'parameters': params,
                    'metrics': results['metrics'],
                    'metric_value': metric_value
                })
                
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Error with parameters {params}: {str(e)}")
                continue
        
        if best_params:
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best {optimization_metric}: {best_metric_value:.4f}")
            
            # Create optimization results table
            optimization_table = _create_optimization_table(all_results, optimization_metric)
            
            return {
                'best_parameters': best_params,
                'best_metric_value': best_metric_value,
                'optimization_metric': optimization_metric,
                'all_results': all_results,
                'optimization_table': optimization_table
            }
        else:
            logger.warning("No successful optimization runs")
            return {}
            
    except Exception as e:
        logger.error(f"Error in strategy optimization: {str(e)}")
        raise BacktestError(f"Strategy optimization failed: {str(e)}") from e


def _create_optimization_table(all_results: List[Dict[str, Any]], 
                              optimization_metric: str) -> pd.DataFrame:
    """Create optimization results table"""
    table_data = []
    
    # Sort by optimization metric
    sorted_results = sorted(all_results, key=lambda x: x['metric_value'], reverse=True)
    
    for result in sorted_results[:20]:  # Top 20 results
        params = result['parameters']
        metrics = result['metrics']
        
        table_data.append({
            'Short Window': params[0],
            'Long Window': params[1],
            'Signal Window': params[2],
            'Strategy Type': params[3],
            'Total Return (%)': f"{metrics['total_return']*100:.2f}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
            'Win Rate (%)': f"{metrics['win_rate']*100:.2f}",
            'Max Drawdown (%)': f"{metrics['max_drawdown']*100:.2f}",
            'Total Trades': metrics['total_trades'],
            f'{optimization_metric}': f"{result['metric_value']:.4f}"
        })
    
    return pd.DataFrame(table_data)

def run_macd_backtest(
    symbol: str = 'btcusd', 
    timeframe: str = '10m',
    strategy_config: Optional[StrategyConfig] = None,
    risk_config: Optional[RiskConfig] = None,
    backtest_config: Optional[BacktestConfig] = None
) -> Dict[str, Any]:
    """
    Run a complete MACD backtest with improved configuration management.
    
    Args:
        symbol: Symbol to test
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
        
        logger.info(f"Running MACD backtest for {symbol.upper()} on {timeframe} timeframe")
        logger.info(f"Strategy config: {strategy_config}")
        logger.info(f"Risk config: {risk_config}")
        
        # Load and prepare data
        data_handler = DataHandler()
        data = data_handler.load_data(symbol, timeframe, strategy_config.strategy_type)
        
        # Resample data if needed
        if timeframe == '10m':
            data = data_handler.resample_data(data, '10T')
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Initialize strategy and risk manager
        strategy = MACDStrategy(strategy_config)
        risk_manager = RiskManager(risk_config)
        
        # Calculate indicators and signals
        data = strategy.calculate_indicators(data)
        data = strategy.generate_signals(data)
        
        # Run backtest
        backtest_engine = BacktestEngine(
            data, strategy, risk_manager, backtest_config, 
            data_handler.output_strategy_dir
        )
        results = backtest_engine.run_backtest()
        
        logger.info("Backtest completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise BacktestError(f"Backtest failed: {str(e)}") 


# Example usage and main execution
def main():
    """Main function demonstrating various analysis types"""
    try:
        logger.info("Starting trading backtest analysis suite")
        
        # # Example 1: Basic backtest with custom configurations
        # logger.info("\n" + "="*50)
        # logger.info("Example 1: Basic MACD Backtest")
        # logger.info("="*50)
        
        # strategy_config = StrategyConfig(
        #     short_window=12,
        #     long_window=26,
        #     signal_window=9,
        #     strategy_type='buy_sell'
        # )
        
        # risk_config = RiskConfig(
        #     trailing_stop=0.02,
        #     position_size=1.0
        # )
        
        # backtest_config = BacktestConfig(
        #     initial_capital=10000,
        #     trade_size=1.0
        # )
        
        # basic_results = run_macd_backtest(
        #     symbol='btcusd',
        #     timeframe='10m',
        #     strategy_config=strategy_config,
        #     risk_config=risk_config,
        #     backtest_config=backtest_config
        # )
        
        # # Example 2: Comprehensive report with visualizations
        # logger.info("\n" + "="*50)
        # logger.info("Example 2: Comprehensive Backtest Report")
        # logger.info("="*50)
        
        # comprehensive_report = create_comprehensive_backtest_report(
        #     symbol='btcusd',
        #     timeframe='10m',
        #     strategy_config=strategy_config,
        #     risk_config=risk_config,
        #     backtest_config=backtest_config,
        #     save_charts=True
        # )
        
        # # Example 3: Strategy comparison
        # logger.info("\n" + "="*50)
        # logger.info("Example 3: Strategy Comparison")
        # logger.info("="*50)
        
        # strategy_configs = [
        #     StrategyConfig(strategy_type='buy_hold'),
        #     StrategyConfig(strategy_type='buy_sell'),
        #     StrategyConfig(strategy_type='reversal')
        # ]
        
        # comparison_results = compare_strategies(
        #     symbol='btcusd',
        #     timeframe='10m',
        #     strategy_configs=strategy_configs,
        #     risk_config=risk_config,
        #     backtest_config=backtest_config
        # )
        
        # # Example 4: Parameter optimization
        # logger.info("\n" + "="*50)
        # logger.info("Example 4: Parameter Optimization")
        # logger.info("="*50)
        
        # optimization_results = optimize_single_strategy(
        #     symbol='btcusd',
        #     timeframe='10m',
        #     # optimization_metric='sharpe_ratio',
        #     optimization_metric='win_rate',
        #     risk_config=risk_config,
        #     backtest_config=backtest_config
        # )
        
        # Example 5: Walk-forward analysis
        logger.info("\n" + "="*50)
        logger.info("Example 5: Walk-Forward Analysis")
        logger.info("="*50)
        
        wf_config = WalkForwardConfig(
            training_years=1,
            testing_months=6,
            step_months=6,
            optimization_metric='win_rate'
        )
        
        walkforward_results = run_walk_forward_analysis(
            symbol='btcusd',
            timeframe='10m',
            wf_config=wf_config
        )
        
        logger.info("\n" + "="*50)
        logger.info("All analyses completed successfully!")
        logger.info("="*50)
        
        return {
            'basic_results': basic_results,
            'comprehensive_report': comprehensive_report,
            'comparison_results': comparison_results,
            'optimization_results': optimization_results,
            'walkforward_results': walkforward_results
        }
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Run main analysis suite
    try:
        results = main()
        logger.info("Analysis suite completed successfully!")
    except Exception as e:
        logger.error(f"Analysis suite failed: {str(e)}")
        raise



# Example usage
if __name__ == "__main__":
    try:
        # Create custom configurations
        strategy_config = StrategyConfig(
            short_window=12,
            long_window=26,
            signal_window=9,
            strategy_type='buy_sell'
        )
        
        risk_config = RiskConfig(
            trailing_stop=0.02,
            position_size=1.0
        )
        
        backtest_config = BacktestConfig(
            initial_capital=10000,
            trade_size=1.0
        )
        
        # Run backtest
        results = run_macd_backtest(
            symbol='btcusd',
            timeframe='10m',
            strategy_config=strategy_config,
            risk_config=risk_config,
            backtest_config=backtest_config
        )
        
        # Print summary metrics
        metrics = results['metrics']
        logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise