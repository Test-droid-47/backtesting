#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PROFESSIONAL BACKTEST ENGINE v2.0                          ║
║                    Direct Model Loading - No Training Files                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "[%(asctime)s] [DEBUG] %(message)s" + reset,
        logging.INFO: cyan + "[%(asctime)s] [INFO] %(message)s" + reset,
        logging.WARNING: yellow + "[%(asctime)s] [WARNING] %(message)s" + reset,
        logging.ERROR: red + "[%(asctime)s] [ERROR] %(message)s" + reset,
        logging.CRITICAL: bold_red + "[%(asctime)s] [CRITICAL] %(message)s" + reset,
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

logger = logging.getLogger('BacktestEngine')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
logger.addHandler(console_handler)

file_handler = logging.FileHandler('backtest_detailed.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
logger.addHandler(file_handler)

# ============================================================================
# PROGRESS TRACKER
# ============================================================================
class ProgressTracker:
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"📊 {self.description}")
        print(f"{'='*60}")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"\n✅ {self.description} completed in {elapsed:.2f} seconds")
        print(f"{'='*60}\n")
    
    def update(self, step_name: str = None):
        self.current_step += 1
        percent = (self.current_step / self.total_steps) * 100
        bar_length = 40
        filled = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        sys.stdout.write(f'\r  [{bar}] {percent:5.1f}% ({self.current_step}/{self.total_steps})')
        if step_name:
            sys.stdout.write(f' - {step_name}')
        sys.stdout.flush()

# ============================================================================
# FILE DISCOVERY
# ============================================================================
class FileDiscovery:
    @staticmethod
    def find_models(base_path: str = '.') -> Dict[str, str]:
        models = {}
        search_paths = [base_path, os.path.join(base_path, 'models')]
        
        logger.info("🔍 Searching for model files...")
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            for file in os.listdir(search_path):
                full_path = os.path.join(search_path, file)
                if file.endswith('.keras'):
                    models['lstm'] = full_path
                    logger.info(f"  ✅ Found LSTM model: {file}")
                elif file.endswith('.pkl') and ('ensemble' in file.lower() or 'xgb' in file.lower()):
                    models['ensemble'] = full_path
                    logger.info(f"  ✅ Found Ensemble model: {file}")
                elif file.endswith('.pkl') and 'scaler' in file.lower():
                    models['scaler'] = full_path
                    logger.info(f"  ✅ Found Scaler: {file}")
                elif file.endswith('.json') and 'feature' in file.lower():
                    models['features'] = full_path
                    logger.info(f"  ✅ Found Features: {file}")
        
        return models
    
    @staticmethod
    def find_data(data_path: str = None) -> Dict[str, str]:
        data = {}
        search_paths = [data_path, '.', './data'] if data_path else ['.', './data']
        
        logger.info("🔍 Searching for data files...")
        
        for search_path in search_paths:
            if not search_path or not os.path.exists(search_path):
                continue
            if os.path.isfile(search_path) and search_path.endswith('.csv'):
                data['ohlcv'] = search_path
                logger.info(f"  ✅ Found OHLCV data: {os.path.basename(search_path)}")
            elif os.path.isdir(search_path):
                for file in os.listdir(search_path):
                    full_path = os.path.join(search_path, file)
                    if file.endswith('.csv'):
                        if 'ohlcv' in file.lower() or 'price' in file.lower():
                            data['ohlcv'] = full_path
                            logger.info(f"  ✅ Found OHLCV data: {file}")
                        elif 'fear' in file.lower() or 'greed' in file.lower():
                            data['fear_greed'] = full_path
                            logger.info(f"  ✅ Found Fear & Greed data: {file}")
        
        return data

# ============================================================================
# RESULTS VISUALIZER
# ============================================================================
class ResultsVisualizer:
    @staticmethod
    def display_results(result: Dict, elapsed_time: float):
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "BACKTEST RESULTS SUMMARY" + " " * 31 + "║")
        print("╠" + "═" * 78 + "╣")
        
        metrics = [
            ("Total Return", f"{result.get('total_return', 0):.2f}%", "🟢" if result.get('total_return', 0) > 0 else "🔴"),
            ("Sharpe Ratio", f"{result.get('sharpe', 0):.4f}", "🟢" if result.get('sharpe', 0) > 1 else "🟡" if result.get('sharpe', 0) > 0.5 else "🔴"),
            ("Max Drawdown", f"{result.get('max_drawdown', 0)*100:.2f}%", "🟢" if result.get('max_drawdown', 0) > -0.1 else "🟡" if result.get('max_drawdown', 0) > -0.2 else "🔴"),
            ("Win Rate", f"{result.get('win_rate', 0)*100:.1f}%", "🟢" if result.get('win_rate', 0) > 0.55 else "🟡" if result.get('win_rate', 0) > 0.45 else "🔴"),
            ("Total Trades", f"{result.get('total_trades', 0)}", "⚪"),
            ("Final Capital", f"${result.get('final_capital', 0):,.2f}", "🟢" if result.get('final_capital', 0) > 10000 else "🟡"),
        ]
        
        for metric, value, status in metrics:
            print(f"║   {status} {metric:<20}: {value:>20} {status} ║")
        
        print("╠" + "═" * 78 + "╣")
        
        sharpe = result.get('sharpe', 0)
        if sharpe > 1.5:
            verdict = "EXCELLENT - Ready for live trading! 🚀"
        elif sharpe > 1.0:
            verdict = "GOOD - Can proceed with caution ✅"
        elif sharpe > 0.5:
            verdict = "AVERAGE - Needs optimization ⚠️"
        elif result.get('total_return', 0) > 0:
            verdict = "POOR - Significant improvement needed 🔴"
        else:
            verdict = "UNPROFITABLE - Do NOT trade live ❌"
        
        print(f"║   🎯 {verdict:<70} ║")
        print("╠" + "═" * 78 + "╣")
        print("║" + f"⏱️  Backtest completed in {elapsed_time:.2f} seconds" + " " * (78 - 37 - len(f"{elapsed_time:.2f}")) + "║")
        print("╚" + "═" * 78 + "╝")
        
        with open('backtest_results.json', 'w') as f:
            json.dump({'timestamp': datetime.now(timezone.utc).isoformat(), 'results': result}, f, indent=2)
        logger.info("📁 Results saved to 'backtest_results.json'")

# ============================================================================
# MAIN BACKTEST RUNNER
# ============================================================================
class BacktestRunner:
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.start_time = None
        self.models = {}
        self.data = {}
        
    def _load_config(self, config_path: str = None) -> dict:
        paths_to_try = [config_path, 'config.json']
        default_config = {
            'symbol': 'BTC/USDT', 'timeframe': '1h', 'fee_rate': 0.001,
            'slippage': 0.0005, 'initial_capital': 10000, 'window': 60
        }
        for path in paths_to_try:
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        cfg = json.load(f)
                    logger.info(f"✅ Config loaded from {path}")
                    return {**default_config, **cfg}
                except Exception as e:
                    logger.warning(f"Failed to load config: {e}")
        logger.warning("⚠️ No config.json found. Using defaults.")
        return default_config
    
    def _discover_and_load_models(self, models_dir: str = None) -> bool:
        logger.info("=" * 60)
        logger.info("📦 MODEL LOADING PHASE")
        logger.info("=" * 60)
        
        discovered = FileDiscovery.find_models(models_dir or '.')
        if not discovered:
            logger.error("❌ No models found!")
            return False
        
        # Direct loading - NO PredictionModel class needed
        if 'lstm' in discovered:
            print(f"\n  Loading LSTM model...")
            try:
                from tensorflow.keras.models import load_model
                self.models['lstm'] = load_model(discovered['lstm'])
                logger.info(f"  ✅ LSTM model loaded: {os.path.basename(discovered['lstm'])}")
            except Exception as e:
                logger.error(f"  ❌ Failed to load LSTM: {e}")
        
        if 'ensemble' in discovered:
            print(f"\n  Loading Ensemble model...")
            try:
                import joblib
                self.models['ensemble'] = joblib.load(discovered['ensemble'])
                logger.info(f"  ✅ Ensemble model loaded: {os.path.basename(discovered['ensemble'])}")
            except Exception as e:
                logger.error(f"  ❌ Failed to load Ensemble: {e}")
        
        if 'scaler' in discovered:
            try:
                import joblib
                self.models['scaler'] = joblib.load(discovered['scaler'])
                logger.info(f"  ✅ Scaler loaded")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load scaler: {e}")
        
        return len(self.models) > 0
    
    def _discover_and_load_data(self, data_path: str = None) -> bool:
        logger.info("=" * 60)
        logger.info("📊 DATA LOADING PHASE")
        logger.info("=" * 60)
        
        discovered = FileDiscovery.find_data(data_path)
        if not discovered or 'ohlcv' not in discovered:
            logger.error("❌ No OHLCV data found!")
            return False
        
        print(f"\n  Loading OHLCV data...")
        try:
            self.data['df'] = pd.read_csv(discovered['ohlcv'])
            self.data['df']['timestamp'] = pd.to_datetime(self.data['df']['timestamp'], utc=True)
            logger.info(f"  ✅ Loaded {len(self.data['df'])} bars")
        except Exception as e:
            logger.error(f"  ❌ Failed to load OHLCV: {e}")
            return False
        
        if 'fear_greed' in discovered:
            try:
                fg_df = pd.read_csv(discovered['fear_greed'])
                fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'], utc=True)
                self.data['df']['date'] = self.data['df']['timestamp'].dt.date
                fg_df['date'] = fg_df['timestamp'].dt.date
                self.data['df'] = self.data['df'].merge(fg_df[['date', 'fear_greed']], on='date', how='left')
                self.data['df']['fear_greed'] = self.data['df']['fear_greed'].ffill().bfill().fillna(50)
                self.data['df'].drop('date', axis=1, inplace=True)
                logger.info(f"  ✅ Merged Fear & Greed data")
            except Exception as e:
                self.data['df']['fear_greed'] = 50
        
        return True
    
    def _prepare_features(self) -> bool:
    logger.info("=" * 60)
    logger.info("🔧 FEATURE PREPARATION")
    logger.info("=" * 60)
    
    df = self.data['df']
    
    # Load selected features from JSON
    features = None
    if 'features' in self.models:
        try:
            with open(self.models['features'], 'r') as f:
                feat_data = json.load(f)
            if isinstance(feat_data, dict) and 'selected_features' in feat_data:
                features = feat_data['selected_features']
            elif isinstance(feat_data, list):
                features = feat_data
            logger.info(f"  ✅ Loaded {len(features)} selected features from JSON")
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to load features: {e}")
    
    if not features:
        # Try to get features from scaler
        if 'scaler' in self.models:
            try:
                features = self.models['scaler'].feature_names_in_.tolist()
                logger.info(f"  ✅ Loaded {len(features)} features from scaler")
            except:
                features = ['open', 'high', 'low', 'close', 'volume']
                logger.warning(f"  ⚠️ Using default {len(features)} features")
    
    # Check which features are available in dataframe
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        logger.warning(f"  ⚠️ Missing {len(missing_features)} features: {missing_features[:5]}...")
    
    if not available_features:
        logger.error("  ❌ No available features found!")
        return False
    
    logger.info(f"  ✅ Using {len(available_features)} available features")
    
    # Prepare data
    data = df[available_features].copy()
    data = data.ffill().bfill().fillna(0)
    
    # Scale if scaler available
    if 'scaler' in self.models:
        try:
            # Scale only the columns scaler expects
            scaler_features = self.models['scaler'].feature_names_in_.tolist()
            common_features = [f for f in scaler_features if f in data.columns]
            
            if len(common_features) != len(scaler_features):
                logger.warning(f"  ⚠️ Scaler expects {len(scaler_features)} features, got {len(common_features)}")
                # Add missing columns with zeros
                for f in scaler_features:
                    if f not in data.columns:
                        data[f] = 0
                data = data[scaler_features]
            
            data_scaled = self.models['scaler'].transform(data)
            logger.info(f"  ✅ Data scaled")
            data = data_scaled
        except Exception as e:
            logger.warning(f"  ⚠️ Scaling failed: {e}, using raw data")
            data = data.values
    
    # Create sequences
    window = self.config.get('window', 60)
    if len(data) < window:
        logger.error(f"  ❌ Not enough data: {len(data)} < {window}")
        return False
    
    X = np.array([data[i-window:i] for i in range(window, len(data))])
    logger.info(f"  ✅ Created {len(X)} sequences (window={window}, features={data.shape[1]})")
    
    self.data['X'] = X
    self.data['window'] = window
    return True
    
    def _generate_predictions(self) -> bool:
        logger.info("=" * 60)
        logger.info("🤖 GENERATING PREDICTIONS")
        logger.info("=" * 60)
        
        df = self.data['df']
        X = self.data['X']
        window = self.data['window']
        
        # Initialize prediction columns
        df['pred_direction'] = 1
        df['pred_entry_quality'] = 0.5
        df['pred_position_size'] = 0.1
        
        if 'lstm' in self.models:
            print(f"\n  Running LSTM inference on {len(X)} samples...")
            try:
                lstm_out = self.models['lstm'].predict(X, verbose=0)
                # lstm_out is list: [price_pred, direction, entry_quality, exit_bar, position_size]
                if isinstance(lstm_out, list) and len(lstm_out) >= 3:
                    direction = np.argmax(lstm_out[1], axis=1)
                    entry_quality = lstm_out[2].flatten()
                    position_size = lstm_out[4].flatten() if len(lstm_out) > 4 else np.full(len(X), 0.1)
                    
                    df.iloc[window:, df.columns.get_loc('pred_direction')] = direction
                    df.iloc[window:, df.columns.get_loc('pred_entry_quality')] = entry_quality
                    df.iloc[window:, df.columns.get_loc('pred_position_size')] = position_size
                    logger.info(f"  ✅ LSTM predictions generated")
            except Exception as e:
                logger.warning(f"  ⚠️ LSTM prediction failed: {e}")
        
        # Ensemble predictions
        if 'ensemble' in self.models:
            print(f"\n  Running Ensemble inference...")
            try:
                probs = []
                for i in tqdm(range(len(df)), desc="  Ensemble", leave=False):
                    try:
                        row = df.iloc[i]
                        prob = self.models['ensemble'].predict_proba_bullish(row)
                    except:
                        prob = 0.5
                    probs.append(prob)
                df['ensemble_prob'] = probs
                logger.info(f"  ✅ Ensemble predictions generated")
            except Exception as e:
                df['ensemble_prob'] = 0.5
                logger.warning(f"  ⚠️ Ensemble failed: {e}")
        else:
            df['ensemble_prob'] = 0.5
        
        df['pred_trade_ok'] = df['pred_entry_quality'] > 0.35
        self.data['df'] = df
        return True
    
    def _run_backtest(self) -> Dict:
        logger.info("=" * 60)
        logger.info("💰 RUNNING BACKTEST")
        logger.info("=" * 60)
        
        df = self.data['df']
        capital = self.config.get('initial_capital', 10000)
        initial_capital = capital
        position = 0
        entry_price = 0
        trades = []
        portfolio = [capital]
        fee = self.config.get('fee_rate', 0.001)
        slippage = self.config.get('slippage', 0.0005)
        
        print(f"\n  Simulating trades on {len(df)} bars...")
        
        for i in tqdm(range(len(df)), desc="  Backtest progress", leave=False):
            price = df['close'].iloc[i]
            signal = df['pred_direction'].iloc[i]
            quality = df['pred_entry_quality'].iloc[i]
            
            # Entry logic
            if signal == 2 and position == 0 and quality > 0.5:
                buy_price = price * (1 + slippage)
                position = capital * 0.1 / buy_price
                entry_price = buy_price
                capital = 0
                trades.append({'type': 'buy', 'price': buy_price, 'bar': i})
            
            # Exit logic
            elif signal == 0 and position > 0:
                sell_price = price * (1 - slippage)
                capital = position * sell_price * (1 - fee)
                pnl = (sell_price - entry_price) / entry_price
                position = 0
                trades.append({'type': 'sell', 'price': sell_price, 'pnl': pnl, 'bar': i})
            
            portfolio.append(capital + position * price)
        
        # Calculate metrics
        final_value = portfolio[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate returns for Sharpe
        returns = []
        for i in range(1, len(portfolio)):
            ret = (portfolio[i] - portfolio[i-1]) / (portfolio[i-1] + 1e-10)
            returns.append(ret)
        
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 24) if returns else 0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio)
        drawdown = (np.array(portfolio) - peak) / (peak + 1e-10)
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        sell_trades = [t for t in trades if t['type'] == 'sell']
        wins = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        total_closed = len(sell_trades)
        win_rate = wins / total_closed if total_closed > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_closed,
            'final_capital': final_value,
            'portfolio': portfolio,
            'trades': trades
        }
    
    def run(self, models_dir: str = None, data_path: str = None) -> Dict[str, Any]:
        self.start_time = time.time()
        
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "PROFESSIONAL BACKTEST ENGINE v2.0" + " " * 28 + "║")
        print("║" + " " * 25 + "Direct Model Loading" + " " * 33 + "║")
        print("╚" + "═" * 78 + "╝")
        
        if not self._discover_and_load_models(models_dir):
            logger.error("❌ Backtest aborted: No models loaded")
            return {'error': 'No models loaded'}
        
        if not self._discover_and_load_data(data_path):
            logger.error("❌ Backtest aborted: No data loaded")
            return {'error': 'No data loaded'}
        
        if not self._prepare_features():
            logger.error("❌ Backtest aborted: Feature preparation failed")
            return {'error': 'Feature preparation failed'}
        
        if not self._generate_predictions():
            logger.error("❌ Backtest aborted: Prediction generation failed")
            return {'error': 'Prediction generation failed'}
        
        results = self._run_backtest()
        
        elapsed = time.time() - self.start_time
        ResultsVisualizer.display_results(results, elapsed)
        
        gc.collect()
        return results

def main():
    parser = argparse.ArgumentParser(description='Professional Backtest Engine')
    parser.add_argument('--models', type=str, default=None, help='Models directory')
    parser.add_argument('--data', type=str, default=None, help='Data file or directory')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    runner = BacktestRunner(config_path=args.config)
    result = runner.run(models_dir=args.models, data_path=args.data)
    
    return 0 if 'error' not in result else 1

if __name__ == '__main__':
    exit(main())
