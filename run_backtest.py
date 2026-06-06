#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('Backtest')

class BacktestRunner:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.start_time = None
        self.results = {}
        
    def _load_config(self, config_path: str = None) -> dict:
        paths_to_try = [
            config_path,
            os.path.join(os.path.dirname(__file__), 'config.json'),
            'config.json'
        ]
        
        for path in paths_to_try:
            if path and os.path.exists(path):
                with open(path, 'r') as f:
                    cfg = json.load(f)
                logger.info(f"Config loaded from {path}")
                return cfg
        
        logger.warning("No config.json found. Using defaults.")
        return {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'fee_rate': 0.001,
            'slippage': 0.0005,
            'initial_capital': 10000,
            'kelly_cap': 0.25,
            'max_risk_per_trade': 0.02,
            'threshold_atr_multiplier': 0.5
        }
    
    def load_models(self, models_dir: str = None) -> Dict:
        models_path = models_dir or os.path.join(os.path.dirname(__file__), 'models')
        
        if not os.path.exists(models_path):
            raise FileNotFoundError(f"Models directory not found: {models_path}")
        
        logger.info(f"Loading models from {models_path}")
        
        from prediction_model import PredictionModel
        from ensemble_model import EnsembleModel
        from regime_detector import MarketRegimeDetector
        
        models = {}
        
        lstm_path = os.path.join(models_path, 'lstm_model.keras')
        if os.path.exists(lstm_path):
            pred_model = PredictionModel(self.config)
            pred_model.load(lstm_path)
            models['lstm'] = pred_model
            logger.info("✅ LSTM model loaded")
        else:
            raise FileNotFoundError(f"LSTM model not found: {lstm_path}")
        
        ensemble_path = os.path.join(models_path, 'ensemble_model.pkl')
        if os.path.exists(ensemble_path):
            ensemble = EnsembleModel(self.config)
            ensemble.load(ensemble_path)
            models['ensemble'] = ensemble
            logger.info("✅ Ensemble model loaded")
        
        regime_path = os.path.join(models_path, 'regime_label_map.json')
        if os.path.exists(regime_path):
            regime_detector = MarketRegimeDetector(self.config)
            regime_detector.load_map()
            models['regime'] = regime_detector
            logger.info("✅ Regime detector loaded")
        
        return models
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        data_file = data_path or os.path.join(os.path.dirname(__file__), 'data', 'ohlcv_data.csv')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        logger.info(f"Loaded {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        return df
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building features...")
        
        from feature_engine import FeatureEngine
        from smart_money import SmartMoneyEngine
        from alpha_factors import AlphaFactorEngine
        
        df = FeatureEngine.build_all(df)
        df = SmartMoneyEngine(self.config).build_all(df)
        df = AlphaFactorEngine(self.config).build_all(df)
        
        logger.info(f"Features built. Total columns: {len(df.columns)}")
        return df
    
    def add_regime(self, df: pd.DataFrame, models: Dict) -> pd.DataFrame:
        if 'regime' in models:
            logger.info("Adding regime detection...")
            df = models['regime'].annotate(df)
        else:
            df['regime'] = 0
            df['regime_confidence'] = 1.0
        
        return df
    
    def generate_predictions(self, df: pd.DataFrame, models: Dict) -> pd.DataFrame:
        logger.info("Generating predictions...")
        
        if 'lstm' in models:
            df = models['lstm'].predict_full(df)
        else:
            df['predicted_close'] = df['close']
            df['pred_direction'] = 1
            df['pred_entry_quality'] = 0.5
            df['pred_position_size'] = 0.1
        
        if 'ensemble' in models:
            ensemble_probs = []
            for i in range(len(df)):
                try:
                    prob = models['ensemble'].predict_proba_bullish(df.iloc[i])
                except:
                    prob = 0.5
                ensemble_probs.append(prob)
            df['ensemble_prob'] = ensemble_probs
        else:
            df['ensemble_prob'] = 0.5
        
        df['pred_trade_ok'] = df['pred_entry_quality'] > 0.35
        
        return df
    
    def run_backtest(self, df: pd.DataFrame, models: Dict) -> Dict[str, Any]:
        logger.info("Running backtest...")
        
        from backtest_engine import BacktestEngine
        engine = BacktestEngine(self.config)
        
        result = engine.run(df, initial_capital=self.config.get('initial_capital', 10000))
        
        return result
    
    def save_report(self, result: Dict, output_path: str = None):
        report_path = output_path or os.path.join(os.path.dirname(__file__), 'backtest_results.json')
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'config': {
                'symbol': self.config.get('symbol'),
                'timeframe': self.config.get('timeframe'),
                'fee_rate': self.config.get('fee_rate'),
                'slippage': self.config.get('slippage'),
                'initial_capital': self.config.get('initial_capital')
            },
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
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        return report_path
    
    def print_summary(self, result: Dict):
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"Total Return:      {result.get('total_return', 0):.2f}%")
        print(f"Sharpe Ratio:      {result.get('sharpe', 0):.4f}")
        print(f"Sortino Ratio:     {result.get('sortino', 0):.4f}")
        print(f"Max Drawdown:      {result.get('max_drawdown', 0)*100:.2f}%")
        print(f"Win Rate:          {result.get('win_rate', 0)*100:.1f}%")
        print(f"Profit Factor:     {result.get('profit_factor', 0):.2f}")
        print(f"Total Trades:      {result.get('total_trades', 0)}")
        print(f"Final Capital:     ${result.get('final_capital', 0):,.2f}")
        print("=" * 70)
        
        if result.get('sharpe', 0) > 1.0:
            print("✅ STRATEGY IS PROFITABLE - Ready for live trading!")
        elif result.get('sharpe', 0) > 0.5:
            print("⚠️ STRATEGY IS OK - Consider optimization")
        else:
            print("❌ STRATEGY NEEDS IMPROVEMENT - Do not go live")
        print("=" * 70)
    
    def run(self, models_dir: str = None, data_path: str = None, output_path: str = None) -> Dict[str, Any]:
        self.start_time = time.time()
        
        print("=" * 70)
        print("STANDALONE BACKTEST")
        print("=" * 70)
        
        try:
            models = self.load_models(models_dir)
            df = self.load_data(data_path)
            df = self.build_features(df)
            df = self.add_regime(df, models)
            df = self.generate_predictions(df, models)
            result = self.run_backtest(df, models)
            
            self.save_report(result, output_path)
            self.print_summary(result)
            
            elapsed = time.time() - self.start_time
            logger.info(f"Backtest completed in {elapsed:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'success': False}

def main():
    parser = argparse.ArgumentParser(description='Standalone Backtest')
    parser.add_argument('--models', type=str, default=None, help='Models directory')
    parser.add_argument('--data', type=str, default=None, help='Data file path')
    parser.add_argument('--output', type=str, default=None, help='Output report path')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    
    args = parser.parse_args()
    
    runner = BacktestRunner(config_path=args.config)
    result = runner.run(
        models_dir=args.models,
        data_path=args.data,
        output_path=args.output
    )
    
    return 0 if 'error' not in result else 1

if __name__ == '__main__':
    exit(main())