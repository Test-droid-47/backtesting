#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PROFESSIONAL BACKTEST ENGINE v1.0                          ║
║                         Hedge Fund Level Backtesting                          ║
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
from pathlib import Path
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================
class CustomFormatter(logging.Formatter):
    """Custom log formatter with colors"""
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

# Setup logger
logger = logging.getLogger('BacktestEngine')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
logger.addHandler(console_handler)

# File handler for detailed logs
file_handler = logging.FileHandler('backtest_detailed.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
logger.addHandler(file_handler)

# ============================================================================
# PROGRESS BAR UTILITY
# ============================================================================
class ProgressTracker:
    """Professional progress tracking with percentages"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = None
        self.step_times = []
        
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
        step_start = time.time()
        percent = (self.current_step / self.total_steps) * 100
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        sys.stdout.write(f'\r  [{bar}] {percent:5.1f}% ({self.current_step}/{self.total_steps})')
        if step_name:
            sys.stdout.write(f' - {step_name}')
        sys.stdout.flush()
        
        self.step_times.append(time.time() - step_start)
    
    def log_step_time(self):
        avg_time = np.mean(self.step_times) if self.step_times else 0
        remaining = avg_time * (self.total_steps - self.current_step)
        if remaining > 0:
            sys.stdout.write(f' | ETA: {remaining:.1f}s')
        sys.stdout.flush()

# ============================================================================
# FILE DISCOVERY UTILITY
# ============================================================================
class FileDiscovery:
    """Auto-discover model and data files"""
    
    @staticmethod
    def find_models(base_path: str = '.') -> Dict[str, str]:
        """Auto-discover all model files"""
        models = {}
        search_paths = [base_path, os.path.join(base_path, 'models'), base_path]
        
        logger.info("🔍 Searching for model files...")
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
                
            for file in os.listdir(search_path):
                full_path = os.path.join(search_path, file)
                
                # LSTM model
                if file.endswith('.keras') and ('lstm' in file.lower() or 'model' in file.lower()):
                    models['lstm'] = full_path
                    logger.info(f"  ✅ Found LSTM model: {file}")
                
                # Ensemble model
                if file.endswith('.pkl') and ('ensemble' in file.lower() or 'xgb' in file.lower()):
                    models['ensemble'] = full_path
                    logger.info(f"  ✅ Found Ensemble model: {file}")
                
                # Regime detector
                if file.endswith('.json') and 'regime' in file.lower():
                    models['regime'] = full_path
                    logger.info(f"  ✅ Found Regime map: {file}")
                
                # Scaler
                if file.endswith('.pkl') and 'scaler' in file.lower():
                    models['scaler'] = full_path
                    logger.info(f"  ✅ Found Scaler: {file}")
                
                # Features
                if file.endswith('.json') and 'feature' in file.lower():
                    models['features'] = full_path
                    logger.info(f"  ✅ Found Features: {file}")
        
        return models
    
    @staticmethod
    def find_data(data_path: str = None) -> Dict[str, str]:
        """Auto-discover data files"""
        data = {}
        search_paths = [data_path, '.', './data', '../data'] if data_path else ['.', './data', '../data']
        
        logger.info("🔍 Searching for data files...")
        
        for search_path in search_paths:
            if not search_path or not os.path.exists(search_path):
                continue
                
            if os.path.isfile(search_path):
                # Single file provided
                if search_path.endswith('.csv'):
                    data['ohlcv'] = search_path
                    logger.info(f"  ✅ Found OHLCV data: {os.path.basename(search_path)}")
                continue
            
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
# BACKTEST RESULTS VISUALIZER
# ============================================================================
class ResultsVisualizer:
    """Professional results visualization"""
    
    @staticmethod
    def display_results(result: Dict, elapsed_time: float):
        """Display backtest results in professional format"""
        
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "BACKTEST RESULTS SUMMARY" + " " * 31 + "║")
        print("╠" + "═" * 78 + "╣")
        
        # Performance Metrics
        print("║" + " " * 5 + "📈 PERFORMANCE METRICS" + " " * 54 + "║")
        print("╠" + "─" * 78 + "╣")
        
        metrics = [
            ("Total Return", f"{result.get('total_return', 0):.2f}%", "🟢" if result.get('total_return', 0) > 0 else "🔴"),
            ("Sharpe Ratio", f"{result.get('sharpe', 0):.4f}", "🟢" if result.get('sharpe', 0) > 1 else "🟡" if result.get('sharpe', 0) > 0.5 else "🔴"),
            ("Sortino Ratio", f"{result.get('sortino', 0):.4f}", "🟢" if result.get('sortino', 0) > 1 else "🟡" if result.get('sortino', 0) > 0.5 else "🔴"),
            ("Max Drawdown", f"{result.get('max_drawdown', 0)*100:.2f}%", "🟢" if result.get('max_drawdown', 0) > -0.1 else "🟡" if result.get('max_drawdown', 0) > -0.2 else "🔴"),
            ("Win Rate", f"{result.get('win_rate', 0)*100:.1f}%", "🟢" if result.get('win_rate', 0) > 0.55 else "🟡" if result.get('win_rate', 0) > 0.45 else "🔴"),
            ("Profit Factor", f"{result.get('profit_factor', 0):.2f}", "🟢" if result.get('profit_factor', 0) > 1.5 else "🟡" if result.get('profit_factor', 0) > 1 else "🔴"),
            ("Total Trades", f"{result.get('total_trades', 0)}", "⚪"),
            ("Final Capital", f"${result.get('final_capital', 0):,.2f}", "🟢" if result.get('final_capital', 0) > 10000 else "🟡"),
        ]
        
        for metric, value, status in metrics:
            print(f"║   {status} {metric:<20}: {value:>20} {status} {' ' * (35 - len(metric))}║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Performance Assessment
        print("║" + " " * 5 + "🎯 PERFORMANCE ASSESSMENT" + " " * 52 + "║")
        print("╠" + "─" * 78 + "╣")
        
        sharpe = result.get('sharpe', 0)
        total_return = result.get('total_return', 0)
        
        if sharpe > 1.5 and total_return > 20:
            verdict = "EXCELLENT - Ready for live trading! 🚀"
            color = "🟢"
        elif sharpe > 1.0 and total_return > 10:
            verdict = "GOOD - Can proceed with caution ✅"
            color = "🟢"
        elif sharpe > 0.5 and total_return > 0:
            verdict = "AVERAGE - Needs optimization ⚠️"
            color = "🟡"
        elif total_return > 0:
            verdict = "POOR - Significant improvement needed 🔴"
            color = "🔴"
        else:
            verdict = "UNPROFITABLE - Do NOT trade live ❌"
            color = "🔴"
        
        print(f"║   {color} {verdict:<70} {color} ║")
        
        print("╠" + "═" * 78 + "╣")
        print("║" + f"⏱️  Backtest completed in {elapsed_time:.2f} seconds" + " " * (78 - 37 - len(f"{elapsed_time:.2f}")) + "║")
        print("╚" + "═" * 78 + "╝")
        
        # Save results to file
        ResultsVisualizer.save_results_json(result, elapsed_time)
    
    @staticmethod
    def save_results_json(result: Dict, elapsed_time: float):
        """Save results to JSON file"""
        output = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'duration_seconds': elapsed_time,
            'results': {
                'total_return': result.get('total_return', 0),
                'sharpe': result.get('sharpe', 0),
                'sortino': result.get('sortino', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'win_rate': result.get('win_rate', 0),
                'profit_factor': result.get('profit_factor', 0),
                'total_trades': result.get('total_trades', 0),
                'final_capital': result.get('final_capital', 0)
            }
        }
        
        with open('backtest_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
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
        self.results = {}
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration with fallback defaults"""
        paths_to_try = [
            config_path,
            os.path.join(os.path.dirname(__file__), 'config.json'),
            'config.json'
        ]
        
        default_config = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'fee_rate': 0.001,
            'slippage': 0.0005,
            'initial_capital': 10000,
            'kelly_cap': 0.25,
            'max_risk_per_trade': 0.02,
            'threshold_atr_multiplier': 0.5
        }
        
        for path in paths_to_try:
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        cfg = json.load(f)
                    logger.info(f"✅ Config loaded from {path}")
                    return {**default_config, **cfg}
                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")
        
        logger.warning("⚠️ No config.json found. Using defaults.")
        return default_config
    
    def _discover_and_load_models(self, models_dir: str = None) -> bool:
        """Auto-discover and load all models"""
        logger.info("=" * 60)
        logger.info("📦 MODEL LOADING PHASE")
        logger.info("=" * 60)
        
        # Discover models
        discovered = FileDiscovery.find_models(models_dir or '.')
        
        if not discovered:
            logger.error("❌ No models found! Please ensure models are in the correct location.")
            return False
        
        total_steps = len(discovered)
        step = 0
        
        # Import required modules
        try:
            from prediction_model import PredictionModel
            from ensemble_model import EnsembleModel
            from regime_detector import MarketRegimeDetector
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            return False
        
        # Load LSTM
        if 'lstm' in discovered:
            step += 1
            print(f"\n  [{step}/{total_steps}] Loading LSTM model...")
            try:
                self.models['lstm'] = PredictionModel(self.config)
                self.models['lstm'].load(discovered['lstm'])
                logger.info(f"  ✅ LSTM model loaded: {os.path.basename(discovered['lstm'])}")
            except Exception as e:
                logger.error(f"  ❌ Failed to load LSTM: {e}")
        
        # Load Ensemble
        if 'ensemble' in discovered:
            step += 1
            print(f"\n  [{step}/{total_steps}] Loading Ensemble model...")
            try:
                self.models['ensemble'] = EnsembleModel(self.config)
                self.models['ensemble'].load(discovered['ensemble'])
                logger.info(f"  ✅ Ensemble model loaded: {os.path.basename(discovered['ensemble'])}")
            except Exception as e:
                logger.error(f"  ❌ Failed to load Ensemble: {e}")
        
        # Load Regime Detector
        if 'regime' in discovered:
            step += 1
            print(f"\n  [{step}/{total_steps}] Loading Regime Detector...")
            try:
                self.models['regime'] = MarketRegimeDetector(self.config)
                self.models['regime'].load_map()
                logger.info(f"  ✅ Regime detector loaded")
            except Exception as e:
                logger.error(f"  ❌ Failed to load Regime detector: {e}")
        
        # Load Scaler
        if 'scaler' in discovered:
            try:
                import joblib
                self.models['scaler'] = joblib.load(discovered['scaler'])
                logger.info(f"  ✅ Scaler loaded")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load scaler: {e}")
        
        return len(self.models) > 0
    
    def _discover_and_load_data(self, data_path: str = None) -> bool:
        """Auto-discover and load data files"""
        logger.info("=" * 60)
        logger.info("📊 DATA LOADING PHASE")
        logger.info("=" * 60)
        
        # Discover data
        discovered = FileDiscovery.find_data(data_path)
        
        if not discovered or 'ohlcv' not in discovered:
            logger.error("❌ No OHLCV data found!")
            return False
        
        total_steps = len(discovered)
        step = 0
        
        # Load OHLCV
        step += 1
        print(f"\n  [{step}/{total_steps}] Loading OHLCV data...")
        try:
            self.data['df'] = pd.read_csv(discovered['ohlcv'])
            self.data['df']['timestamp'] = pd.to_datetime(self.data['df']['timestamp'], utc=True)
            logger.info(f"  ✅ Loaded {len(self.data['df'])} bars from {os.path.basename(discovered['ohlcv'])}")
        except Exception as e:
            logger.error(f"  ❌ Failed to load OHLCV: {e}")
            return False
        
        # Load Fear & Greed
        if 'fear_greed' in discovered:
            step += 1
            print(f"\n  [{step}/{total_steps}] Loading Fear & Greed data...")
            try:
                fg_df = pd.read_csv(discovered['fear_greed'])
                fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'], utc=True)
                
                # Merge with OHLCV
                self.data['df']['date'] = self.data['df']['timestamp'].dt.date
                fg_df['date'] = fg_df['timestamp'].dt.date
                self.data['df'] = self.data['df'].merge(fg_df[['date', 'fear_greed']], on='date', how='left')
                self.data['df']['fear_greed'] = self.data['df']['fear_greed'].ffill().bfill().fillna(50)
                self.data['df'].drop('date', axis=1, inplace=True)
                logger.info(f"  ✅ Merged Fear & Greed data")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load Fear & Greed: {e}")
                self.data['df']['fear_greed'] = 50
        
        return True
    
    def _build_features(self) -> bool:
        """Build technical features"""
        logger.info("=" * 60)
        logger.info("🔧 FEATURE ENGINEERING PHASE")
        logger.info("=" * 60)
        
        try:
            from feature_engine import FeatureEngine
            from smart_money import SmartMoneyEngine
            from alpha_factors import AlphaFactorEngine
            
            df = self.data['df']
            total_steps = 3
            step = 0
            
            # Step 1: Technical indicators
            step += 1
            print(f"\n  [{step}/{total_steps}] Adding technical indicators...")
            df = FeatureEngine.build_all(df)
            logger.info(f"  ✅ Technical indicators added")
            
            # Step 2: Smart Money Concepts
            step += 1
            print(f"\n  [{step}/{total_steps}] Adding Smart Money Concepts...")
            smc = SmartMoneyEngine(self.config)
            df = smc.build_all(df)
            logger.info(f"  ✅ SMC features added")
            
            # Step 3: Alpha Factors
            step += 1
            print(f"\n  [{step}/{total_steps}] Adding Alpha Factors...")
            alpha = AlphaFactorEngine(self.config)
            df = alpha.build_all(df)
            logger.info(f"  ✅ Alpha factors added")
            
            # Add regime if available
            if 'regime' in self.models:
                df = self.models['regime'].annotate(df)
                logger.info(f"  ✅ Regime detection added")
            else:
                df['regime'] = 0
            
            self.data['df'] = df
            logger.info(f"\n📊 Total features: {len(df.columns)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            return False
    
    def _generate_predictions(self) -> bool:
        """Generate predictions using loaded models"""
        logger.info("=" * 60)
        logger.info("🤖 PREDICTION GENERATION PHASE")
        logger.info("=" * 60)
        
        df = self.data['df']
        total_bars = len(df)
        
        print(f"\n  Generating predictions for {total_bars} bars...")
        
        try:
            if 'lstm' in self.models:
                # Use LSTM for predictions
                df = self.models['lstm'].predict_full(df)
                logger.info(f"  ✅ LSTM predictions generated")
            else:
                # Fallback predictions
                df['predicted_close'] = df['close']
                df['pred_direction'] = 1
                df['pred_entry_quality'] = 0.5
                df['pred_position_size'] = 0.1
                logger.warning(f"  ⚠️ No LSTM model, using fallback predictions")
            
            # Ensemble probabilities
            if 'ensemble' in self.models:
                ensemble_probs = []
                for i in tqdm(range(len(df)), desc="  Ensemble predictions", leave=False):
                    try:
                        prob = self.models['ensemble'].predict_proba_bullish(df.iloc[i])
                    except:
                        prob = 0.5
                    ensemble_probs.append(prob)
                df['ensemble_prob'] = ensemble_probs
                logger.info(f"  ✅ Ensemble predictions generated")
            else:
                df['ensemble_prob'] = 0.5
            
            df['pred_trade_ok'] = df['pred_entry_quality'] > 0.35
            self.data['df'] = df
            return True
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            return False
    
    def _run_backtest(self) -> Dict:
        """Execute backtest with realistic parameters"""
        logger.info("=" * 60)
        logger.info("💰 BACKTEST EXECUTION PHASE")
        logger.info("=" * 60)
        
        try:
            from backtest_engine import BacktestEngine
            
            engine = BacktestEngine(self.config)
            result = engine.run(self.data['df'], initial_capital=self.config.get('initial_capital', 10000))
            
            logger.info("  ✅ Backtest completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return {'error': str(e), 'success': False}
    
    def run(self, models_dir: str = None, data_path: str = None) -> Dict[str, Any]:
        """Main execution pipeline"""
        self.start_time = time.time()
        
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "PROFESSIONAL BACKTEST ENGINE" + " " * 31 + "║")
        print("║" + " " * 25 + "Hedge Fund Level" + " " * 36 + "║")
        print("╚" + "═" * 78 + "╝")
        
        # Phase 1: Load Models
        if not self._discover_and_load_models(models_dir):
            logger.error("❌ Backtest aborted: No models loaded")
            return {'error': 'No models loaded', 'success': False}
        
        # Phase 2: Load Data
        if not self._discover_and_load_data(data_path):
            logger.error("❌ Backtest aborted: No data loaded")
            return {'error': 'No data loaded', 'success': False}
        
        # Phase 3: Build Features
        if not self._build_features():
            logger.error("❌ Backtest aborted: Feature engineering failed")
            return {'error': 'Feature engineering failed', 'success': False}
        
        # Phase 4: Generate Predictions
        if not self._generate_predictions():
            logger.error("❌ Backtest aborted: Prediction generation failed")
            return {'error': 'Prediction generation failed', 'success': False}
        
        # Phase 5: Run Backtest
        results = self._run_backtest()
        
        # Display Results
        elapsed = time.time() - self.start_time
        ResultsVisualizer.display_results(results, elapsed)
        
        # Cleanup
        gc.collect()
        
        return results

def main():
    parser = argparse.ArgumentParser(
        description='Professional Backtest Engine - Hedge Fund Level',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py                          # Auto-discover all files
  python run_backtest.py --models ./models        # Specify models directory
  python run_backtest.py --data ./data/ohlcv.csv  # Specify data file
  python run_backtest.py --config custom_config.json
        """
    )
    parser.add_argument('--models', type=str, default=None, help='Models directory path')
    parser.add_argument('--data', type=str, default=None, help='Data file or directory path')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    runner = BacktestRunner(config_path=args.config)
    result = runner.run(models_dir=args.models, data_path=args.data)
    
    return 0 if 'error' not in result else 1

if __name__ == '__main__':
    exit(main())