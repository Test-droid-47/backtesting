#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PROFESSIONAL BACKTEST ENGINE v2.0                          ║
║                    Direct Model Loading - No Training Files                   ║
║              + MULTI-MODEL CONSENSUS (LSTM BOSS + HELPERS)                    ║
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
# LOGGING SETUP - ORIGINAL (NO CHANGE)
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

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('backtest_detailed.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
    logger.addHandler(file_handler)

# ============================================================================
# PROGRESS TRACKER - ORIGINAL (NO CHANGE)
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
# FILE DISCOVERY - ORIGINAL (NO CHANGE)
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
                # NEW: PPO Actor detection
                elif 'ppo_agent_actor' in file.lower() and file.endswith('.keras'):
                    models['ppo_actor'] = full_path
                    logger.info(f"  ✅ Found PPO Actor model: {file}")
                # NEW: PPO Critic detection
                elif 'ppo_agent_critic' in file.lower() and file.endswith('.keras'):
                    models['ppo_critic'] = full_path
                    logger.info(f"  ✅ Found PPO Critic model: {file}")

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
# RESULTS VISUALIZER - ORIGINAL (NO CHANGE)
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

    @staticmethod
    def save_results(result: Dict):
        try:
            serializable = {k: v for k, v in result.items() if k not in ('portfolio', 'trades')}
            serializable['portfolio'] = result.get('portfolio', [])
            serializable['trades'] = result.get('trades', [])
            with open('backtest_results.json', 'w') as f:
                json.dump({'timestamp': datetime.now(timezone.utc).isoformat(), 'results': serializable}, f, indent=2, default=str)
            logger.info("📁 Results saved to 'backtest_results.json'")
        except Exception as e:
            logger.warning(f"⚠️ Could not save results to JSON: {e}")

# ============================================================================
# CONSENSUS ENGINE (NEW - ADDED ONLY)
# ============================================================================
class MultiModelConsensus:
    """
    Multi-model consensus system
    LSTM = BOSS (final decision maker)
    Ensemble, PPO Actor, PPO Critic = Helpers (confirm/override)
    """
    
    def __init__(self):
        self.lstm_weight = 0.50      # BOSS
        self.ensemble_weight = 0.25   # Helper 1
        self.ppo_actor_weight = 0.15  # Helper 2
        self.ppo_critic_weight = 0.10 # Helper 3
        
        self.buy_threshold = 0.60
        self.sell_threshold = 0.40
        self.lstm_override_threshold = 0.80
        
        self.lstm_override_count = 0
        
    def get_lstm_signal(self, lstm_output) -> Tuple[int, float, float]:
        """Extract signal, confidence, buy_prob from LSTM output"""
        buy_prob = 0.5
        signal = 1  # Default HOLD
        confidence = 0.5
        
        try:
            if isinstance(lstm_output, (list, tuple)):
                if len(lstm_output) >= 3:
                    direction_probs = lstm_output[1]
                    if isinstance(direction_probs, np.ndarray):
                        if direction_probs.ndim == 2:
                            probs = direction_probs[0]
                        else:
                            probs = direction_probs
                    else:
                        probs = [0.33, 0.33, 0.34]
                    
                    if len(probs) >= 3:
                        buy_prob = probs[2]
                        sell_prob = probs[0]
                    else:
                        buy_prob = probs[0] if len(probs) > 0 else 0.5
                        sell_prob = 1 - buy_prob
                else:
                    buy_prob = float(lstm_output[0]) if len(lstm_output) > 0 else 0.5
                    sell_prob = 1 - buy_prob
            elif isinstance(lstm_output, np.ndarray):
                if lstm_output.ndim == 1:
                    if len(lstm_output) >= 3:
                        buy_prob = lstm_output[2]
                    else:
                        buy_prob = lstm_output[0]
                else:
                    buy_prob = float(np.mean(lstm_output[:, 2])) if lstm_output.shape[1] > 2 else 0.5
                sell_prob = 1 - buy_prob
            else:
                buy_prob = float(lstm_output) if isinstance(lstm_output, (int, float)) else 0.5
                sell_prob = 1 - buy_prob
                
            confidence = abs(buy_prob - sell_prob) * 2
            confidence = min(max(confidence, 0.3), 0.95)
            
            if buy_prob > self.buy_threshold:
                signal = 2  # BUY
            elif sell_prob > self.sell_threshold:
                signal = 0  # SELL
            else:
                signal = 1  # HOLD
                
        except Exception as e:
            logger.debug(f"LSTM signal extraction error: {e}")
            
        return signal, confidence, buy_prob
    
    def get_ensemble_signal(self, ensemble_pred) -> Tuple[int, float]:
        """Extract signal from ensemble model"""
        try:
            if ensemble_pred is None:
                return 1, 0.5
            prob = float(ensemble_pred) if isinstance(ensemble_pred, (int, float)) else 0.5
            confidence = abs(prob - 0.5) * 2
            
            if prob > self.buy_threshold:
                return 2, confidence
            elif prob < self.sell_threshold:
                return 0, confidence
            else:
                return 1, confidence
        except:
            return 1, 0.5
    
    def get_ppo_actor_signal(self, ppo_actor, row_data=None) -> Tuple[int, float]:
        """Extract signal from PPO Actor helper"""
        try:
            if ppo_actor is None:
                return 1, 0.5
            # Simple placeholder - actual implementation depends on PPO model
            return 1, 0.5
        except:
            return 1, 0.5
    
    def get_ppo_critic_signal(self, ppo_critic, row_data=None) -> Tuple[int, float]:
        """Extract signal from PPO Critic helper (risk assessment)"""
        try:
            if ppo_critic is None:
                return 1, 0.5
            # Simple placeholder - actual implementation depends on PPO model
            return 1, 0.5
        except:
            return 1, 0.5
    
    def calculate_consensus(self, lstm_signal, lstm_confidence, lstm_buy_prob,
                           ensemble_signal, ensemble_confidence,
                           ppo_actor_signal, ppo_actor_confidence,
                           ppo_critic_signal, ppo_critic_confidence) -> Dict:
        """
        Calculate weighted consensus with LSTM override capability
        Returns: {'signal': 0/1/2, 'confidence': float, 'lstm_override': bool}
        """
        
        # Weighted voting
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        # LSTM vote
        if lstm_signal == 2:
            buy_score += self.lstm_weight * lstm_confidence
        elif lstm_signal == 0:
            sell_score += self.lstm_weight * lstm_confidence
        else:
            buy_score += self.lstm_weight * 0.4
            sell_score += self.lstm_weight * 0.4
        total_weight += self.lstm_weight
        
        # Ensemble vote
        if ensemble_signal == 2:
            buy_score += self.ensemble_weight * ensemble_confidence
        elif ensemble_signal == 0:
            sell_score += self.ensemble_weight * ensemble_confidence
        else:
            buy_score += self.ensemble_weight * 0.3
            sell_score += self.ensemble_weight * 0.3
        total_weight += self.ensemble_weight
        
        # PPO Actor vote
        if ppo_actor_signal == 2:
            buy_score += self.ppo_actor_weight * ppo_actor_confidence
        elif ppo_actor_signal == 0:
            sell_score += self.ppo_actor_weight * ppo_actor_confidence
        else:
            buy_score += self.ppo_actor_weight * 0.3
            sell_score += self.ppo_actor_weight * 0.3
        total_weight += self.ppo_actor_weight
        
        # PPO Critic vote
        if ppo_critic_signal == 2:
            buy_score += self.ppo_critic_weight * ppo_critic_confidence
        elif ppo_critic_signal == 0:
            sell_score += self.ppo_critic_weight * ppo_critic_confidence
        else:
            buy_score += self.ppo_critic_weight * 0.3
            sell_score += self.ppo_critic_weight * 0.3
        total_weight += self.ppo_critic_weight
        
        # Normalize
        if total_weight > 0:
            consensus_score = buy_score / (buy_score + sell_score + 1e-10)
        else:
            consensus_score = 0.5
            
        # Check if models agree
        all_signals = [lstm_signal, ensemble_signal, ppo_actor_signal, ppo_critic_signal]
        agreement = 1 - (len(set(all_signals)) - 1) / 3
        
        # LSTM override logic
        lstm_override = False
        final_signal = 1
        final_confidence = consensus_score
        
        if lstm_confidence > self.lstm_override_threshold and agreement < 0.6:
            # LSTM overrides when high confidence and models disagree
            final_signal = lstm_signal
            final_confidence = lstm_confidence * 0.9
            lstm_override = True
            self.lstm_override_count += 1
        else:
            if consensus_score > self.buy_threshold:
                final_signal = 2
            elif consensus_score < self.sell_threshold:
                final_signal = 0
            else:
                final_signal = 1
                
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'consensus_score': consensus_score,
            'lstm_override': lstm_override,
            'agreement': agreement
        }
    
    def get_stats(self):
        return {'lstm_overrides': self.lstm_override_count}

# ============================================================================
# ORIGINAL FEATURE ENGINEERING (100% SAME - NO CHANGE)
# ============================================================================
class FeatureEngineer:

    @staticmethod
    def _safe(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan)

    @classmethod
    def calculate_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']

        logger.info("  ⚙️  Calculating features...")

        # ── TREND ────────────────────────────────────────────────────────────
        for p in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{p}'] = c.rolling(p).mean()
            df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()

        df['sma_ratio_7_21']   = cls._safe(df['sma_7']  / df['sma_21'])
        df['sma_ratio_21_50']  = cls._safe(df['sma_21'] / df['sma_50'])
        df['sma_ratio_50_200'] = cls._safe(df['sma_50'] / df['sma_200'])
        df['ema_ratio_7_21']   = cls._safe(df['ema_7']  / df['ema_21'])
        df['price_vs_sma50']   = cls._safe(c / df['sma_50'])
        df['price_vs_sma200']  = cls._safe(c / df['sma_200'])
        df['price_vs_ema21']   = cls._safe(c / df['ema_21'])

        # DPO (Detrended Price Oscillator) — period 14 & 20
        for p in [14, 20]:
            shift = p // 2 + 1
            df[f'dpo_{p}'] = cls._safe(c - c.rolling(p).mean().shift(shift))
            df[f'dpo_{p}_norm'] = cls._safe(df[f'dpo_{p}'] / c)

        # ── MOMENTUM ─────────────────────────────────────────────────────────
        for p in [3, 5, 10, 14, 20]:
            df[f'log_ret_{p}'] = cls._safe(np.log(c / c.shift(p)))

        df['log_ret_1']    = cls._safe(np.log(c / c.shift(1)))
        df['roc_5']        = cls._safe(c.pct_change(5))
        df['roc_10']       = cls._safe(c.pct_change(10))
        df['roc_20']       = cls._safe(c.pct_change(20))

        # RSI
        for p in [7, 14, 21]:
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(p).mean()
            loss = (-delta.clip(upper=0)).rolling(p).mean()
            rs = cls._safe(gain / (loss + 1e-10))
            df[f'rsi_{p}'] = 100 - (100 / (1 + rs))
            df[f'rsi_{p}_norm'] = df[f'rsi_{p}'] / 100.0

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df['macd']        = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist']   = df['macd'] - df['macd_signal']
        df['macd_norm']   = cls._safe(df['macd'] / (c + 1e-10))

        # Stochastic
        for p in [14, 21]:
            lo = l.rolling(p).min()
            hi = h.rolling(p).max()
            df[f'stoch_k_{p}'] = cls._safe(100 * (c - lo) / (hi - lo + 1e-10))
            df[f'stoch_d_{p}'] = df[f'stoch_k_{p}'].rolling(3).mean()

        # Williams %R
        df['williams_r_14'] = cls._safe(-100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-10))

        # CCI
        tp = (h + l + c) / 3
        df['cci_14'] = cls._safe((tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std() + 1e-10))
        df['cci_20'] = cls._safe((tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10))

        # ── VOLUME ───────────────────────────────────────────────────────────
        df['volume_sma_10']    = v.rolling(10).mean()
        df['volume_sma_20']    = v.rolling(20).mean()
        df['volume_ratio_10']  = cls._safe(v / (df['volume_sma_10'] + 1e-10))
        df['volume_ratio_20']  = cls._safe(v / (df['volume_sma_20'] + 1e-10))
        df['volume_log']       = cls._safe(np.log(v + 1))
        df['volume_change']    = v.pct_change()
        df['volume_change_5']  = v.pct_change(5)

        # OBV
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        df['obv']           = obv
        df['obv_sma_10']    = obv.rolling(10).mean()
        df['obv_ratio']     = cls._safe(obv / (df['obv_sma_10'] + 1e-10))

        # VWAP (rolling)
        typical = (h + l + c) / 3
        df['vwap_20']      = cls._safe((typical * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10))
        df['price_vs_vwap'] = cls._safe(c / (df['vwap_20'] + 1e-10))

        # MFI
        mf_raw = typical * v
        pos_mf = mf_raw.where(typical > typical.shift(1), 0).rolling(14).sum()
        neg_mf = mf_raw.where(typical < typical.shift(1), 0).rolling(14).sum()
        df['mfi_14'] = cls._safe(100 - (100 / (1 + pos_mf / (neg_mf + 1e-10))))

        # ── VOLATILITY ───────────────────────────────────────────────────────
        ret = c.pct_change()
        for p in [10, 20, 30]:
            df[f'volatility_{p}'] = ret.rolling(p).std()

        df['volatility_ratio'] = cls._safe(df['volatility_10'] / (df['volatility_30'] + 1e-10))

        # Bollinger Bands
        for p in [20]:
            mid = c.rolling(p).mean()
            std = c.rolling(p).std()
            df[f'bb_upper_{p}'] = mid + 2 * std
            df[f'bb_lower_{p}'] = mid - 2 * std
            df[f'bb_width_{p}'] = cls._safe((df[f'bb_upper_{p}'] - df[f'bb_lower_{p}']) / (mid + 1e-10))
            df[f'bb_pct_{p}']   = cls._safe((c - df[f'bb_lower_{p}']) / (df[f'bb_upper_{p}'] - df[f'bb_lower_{p}'] + 1e-10))

        # ATR
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        for p in [7, 14, 21]:
            df[f'atr_{p}']      = tr.rolling(p).mean()
            df[f'atr_{p}_norm'] = cls._safe(df[f'atr_{p}'] / (c + 1e-10))

        # Keltner Channel width
        kc_mid = c.ewm(span=20, adjust=False).mean()
        df['keltner_width'] = cls._safe(4 * df['atr_14'] / (kc_mid + 1e-10))

        # ── Z-SCORES ─────────────────────────────────────────────────────────
        for p in [10, 20, 50]:
            roll_mean = c.rolling(p).mean()
            roll_std  = c.rolling(p).std()
            df[f'close_zscore_{p}'] = cls._safe((c - roll_mean) / (roll_std + 1e-10))

        for p in [10, 20]:
            v_mean = v.rolling(p).mean()
            v_std  = v.rolling(p).std()
            df[f'volume_zscore_{p}'] = cls._safe((v - v_mean) / (v_std + 1e-10))

        # ── PRICE STRUCTURE ──────────────────────────────────────────────────
        df['hl_ratio']     = cls._safe((h - l) / (c + 1e-10))
        df['oc_ratio']     = cls._safe((c - o) / (h - l + 1e-10))
        df['upper_shadow'] = cls._safe((h - c.combine(o, max)) / (h - l + 1e-10))
        df['lower_shadow'] = cls._safe((c.combine(o, min) - l) / (h - l + 1e-10))
        df['gap']          = cls._safe((o - c.shift(1)) / (c.shift(1) + 1e-10))

        # ── TIME FEATURES ────────────────────────────────────────────────────
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            df['hour_sin']    = np.sin(2 * np.pi * ts.dt.hour / 24)
            df['hour_cos']    = np.cos(2 * np.pi * ts.dt.hour / 24)
            df['dow_sin']     = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
            df['dow_cos']     = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
            df['month_sin']   = np.sin(2 * np.pi * ts.dt.month / 12)
            df['month_cos']   = np.cos(2 * np.pi * ts.dt.month / 12)

        logger.info(f"  ✅ Feature engineering complete — {len(df.columns)} total columns")
        return df

# ============================================================================
# MAIN BACKTEST RUNNER (ORIGINAL + CONSENSUS INTEGRATION)
# ============================================================================
class BacktestRunner:

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.start_time = None
        self.models = {}
        self.data = {}
        self.consensus = MultiModelConsensus()  # NEW: Consensus engine

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
                
        # NEW: Load PPO Actor if found
        if 'ppo_actor' in discovered:
            print(f"\n  Loading PPO Actor model...")
            try:
                from tensorflow.keras.models import load_model
                self.models['ppo_actor'] = load_model(discovered['ppo_actor'])
                logger.info(f"  ✅ PPO Actor model loaded: {os.path.basename(discovered['ppo_actor'])}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load PPO Actor: {e}")
                
        # NEW: Load PPO Critic if found
        if 'ppo_critic' in discovered:
            print(f"\n  Loading PPO Critic model...")
            try:
                from tensorflow.keras.models import load_model
                self.models['ppo_critic'] = load_model(discovered['ppo_critic'])
                logger.info(f"  ✅ PPO Critic model loaded: {os.path.basename(discovered['ppo_critic'])}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load PPO Critic: {e}")

        if 'scaler' in discovered:
            try:
                import joblib
                self.models['scaler'] = joblib.load(discovered['scaler'])
                logger.info(f"  ✅ Scaler loaded")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load scaler: {e}")

        if 'features' in discovered:
            self.models['features'] = discovered['features']

        return len(self.models) > 0

    def _parse_timestamp(self, series: pd.Series) -> pd.Series:
        try:
            parsed = pd.to_datetime(series, utc=True, infer_datetime_format=True)
            return parsed
        except Exception:
            pass
        try:
            parsed = pd.to_datetime(series, unit='ms', utc=True)
            return parsed
        except Exception:
            pass
        try:
            parsed = pd.to_datetime(series, unit='s', utc=True)
            return parsed
        except Exception:
            pass
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize('UTC')
        else:
            parsed = parsed.dt.tz_convert('UTC')
        return parsed

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
            df = pd.read_csv(discovered['ohlcv'])
            df.columns = [c.strip().lower() for c in df.columns]

            if 'timestamp' in df.columns:
                df['timestamp'] = self._parse_timestamp(df['timestamp'])
                logger.info(f"  ✅ Timestamp parsed — dtype: {df['timestamp'].dtype}")
            else:
                logger.warning("  ⚠️ No 'timestamp' column found in OHLCV data")

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                logger.error(f"  ❌ OHLCV data missing required columns: {missing_cols}")
                return False

            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=['close'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.data['df'] = df
            logger.info(f"  ✅ Loaded {len(df)} bars")
        except Exception as e:
            logger.error(f"  ❌ Failed to load OHLCV: {e}")
            return False

        if 'fear_greed' in discovered:
            try:
                fg_raw = pd.read_csv(discovered['fear_greed'], header=None, nrows=1)
                first_val = str(fg_raw.iloc[0, 0])
                try:
                    pd.to_datetime(first_val)
                    fg_df = pd.read_csv(discovered['fear_greed'], header=None)
                    fg_df.columns = [f'col_{i}' for i in range(len(fg_df.columns))]
                    logger.info(f"  ℹ️ F&G CSV has no header — inferring columns")
                except Exception:
                    fg_df = pd.read_csv(discovered['fear_greed'])
                    fg_df.columns = [c.strip().lower() for c in fg_df.columns]

                ts_col  = next((c for c in fg_df.columns if any(k in str(c).lower() for k in ('time', 'date', 'col_0'))), None)
                val_col = next((c for c in fg_df.columns if any(k in str(c).lower() for k in ('fear', 'greed', 'value', 'col_1'))), None)

                if ts_col is None or val_col is None:
                    raise ValueError(f"Cannot identify timestamp/value columns in F&G CSV. Columns: {fg_df.columns.tolist()}")

                fg_df[ts_col] = self._parse_timestamp(fg_df[ts_col])
                fg_df['_date'] = fg_df[ts_col].dt.normalize()
                fg_df = fg_df[[ts_col, '_date', val_col]].rename(columns={val_col: 'fear_greed'})
                fg_df['fear_greed'] = pd.to_numeric(fg_df['fear_greed'], errors='coerce')
                fg_df = fg_df.dropna(subset=['fear_greed']).drop_duplicates(subset=['_date'])

                self.data['df']['_date'] = self.data['df']['timestamp'].dt.normalize()
                self.data['df'] = self.data['df'].merge(fg_df[['_date', 'fear_greed']], on='_date', how='left')
                self.data['df']['fear_greed'] = self.data['df']['fear_greed'].ffill().bfill().fillna(50)
                self.data['df'].drop(columns=['_date'], inplace=True)
                logger.info(f"  ✅ Merged Fear & Greed data")
            except Exception as e:
                logger.warning(f"  ⚠️ Fear & Greed merge failed: {e}. Defaulting to 50.")
                self.data['df']['fear_greed'] = 50

        return True

    def _prepare_features(self) -> bool:
        logger.info("=" * 60)
        logger.info("🔧 FEATURE PREPARATION")
        logger.info("=" * 60)

        self.data['df'] = FeatureEngineer.calculate_all(self.data['df'])
        df = self.data['df']

        # Aliases — training mein short names use hue hon to yahan map karo
        aliases = {
            'dpo':    'dpo_14',
            'dpo_norm': 'dpo_14_norm',
            'ema_20': 'ema_21',
            'sma_20': 'sma_21',
            'atr':    'atr_14',
            'atr_norm': 'atr_14_norm',
            'rsi':    'rsi_14',
            'stoch_k': 'stoch_k_14',
            'stoch_d': 'stoch_d_14',
            'cci':    'cci_14',
            'bb_width': 'bb_width_20',
            'bb_pct':   'bb_pct_20',
            'volatility': 'volatility_20',
            'close_zscore': 'close_zscore_20',
            'volume_zscore': 'volume_zscore_20',
        }
        for alias, source in aliases.items():
            if alias not in df.columns and source in df.columns:
                df[alias] = df[source]

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
            if 'scaler' in self.models:
                try:
                    features = self.models['scaler'].feature_names_in_.tolist()
                    logger.info(f"  ✅ Loaded {len(features)} features from scaler")
                except Exception:
                    features = ['open', 'high', 'low', 'close', 'volume']
                    logger.warning(f"  ⚠️ Using default {len(features)} features")
            else:
                features = ['open', 'high', 'low', 'close', 'volume']
                logger.warning(f"  ⚠️ No feature source found. Using default {len(features)} features")

        available_features = [f for f in features if f in df.columns]
        missing_features   = [f for f in features if f not in df.columns]

        if missing_features:
            logger.warning(f"  ⚠️ Missing {len(missing_features)} features: {missing_features[:10]}...")

        if not available_features:
            logger.error("  ❌ No available features found in DataFrame!")
            return False

        logger.info(f"  ✅ Using {len(available_features)} / {len(features)} selected features")

        data = df[available_features].copy()
        data = data.ffill().bfill().fillna(0)

        if 'scaler' in self.models:
            try:
                scaler_features = self.models['scaler'].feature_names_in_.tolist()
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
        else:
            data = data.values

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

        # Original prediction columns
        df['pred_direction']    = 1
        df['pred_entry_quality'] = 0.5
        df['pred_position_size'] = 0.1
        
        # NEW: Columns for consensus system
        df['lstm_signal'] = 1
        df['lstm_confidence'] = 0.5
        df['lstm_buy_prob'] = 0.5
        df['ensemble_signal'] = 1
        df['ensemble_confidence'] = 0.5
        df['consensus_signal'] = 1
        df['consensus_confidence'] = 0.5

        # LSTM predictions (BOSS)
        if 'lstm' in self.models:
            print(f"\n  Running LSTM inference on {len(X)} samples...")
            try:
                lstm_out = self.models['lstm'].predict(X, verbose=0)
                
                for i, idx in enumerate(range(window, len(df))):
                    output = lstm_out[i] if len(lstm_out) > i else lstm_out
                    signal, confidence, buy_prob = self.consensus.get_lstm_signal(output)
                    
                    df.at[idx, 'lstm_signal'] = signal
                    df.at[idx, 'lstm_confidence'] = confidence
                    df.at[idx, 'lstm_buy_prob'] = buy_prob
                    
                    # Also set original fields for compatibility
                    if signal == 2:
                        df.at[idx, 'pred_direction'] = 1
                        df.at[idx, 'pred_entry_quality'] = confidence
                    elif signal == 0:
                        df.at[idx, 'pred_direction'] = 0
                        df.at[idx, 'pred_entry_quality'] = confidence
                    else:
                        df.at[idx, 'pred_direction'] = 1
                        df.at[idx, 'pred_entry_quality'] = 0.3
                        
                logger.info(f"  ✅ LSTM predictions generated")
            except Exception as e:
                logger.warning(f"  ⚠️ LSTM prediction failed: {e}")

        # Ensemble predictions (Helper 1)
        if 'ensemble' in self.models:
            print(f"\n  Running Ensemble inference...")
            try:
                for i in tqdm(range(len(df)), desc="  Ensemble", leave=False):
                    try:
                        row = df.iloc[i]
                        if hasattr(self.models['ensemble'], 'predict_proba_bullish'):
                            prob = self.models['ensemble'].predict_proba_bullish(row)
                        else:
                            prob = 0.5
                        signal, confidence = self.consensus.get_ensemble_signal(prob)
                        df.at[i, 'ensemble_signal'] = signal
                        df.at[i, 'ensemble_confidence'] = confidence
                    except Exception:
                        df.at[i, 'ensemble_signal'] = 1
                        df.at[i, 'ensemble_confidence'] = 0.5
                logger.info(f"  ✅ Ensemble predictions generated")
            except Exception as e:
                logger.warning(f"  ⚠️ Ensemble failed: {e}")
                df['ensemble_signal'] = 1
                df['ensemble_confidence'] = 0.5
        else:
            df['ensemble_signal'] = 1
            df['ensemble_confidence'] = 0.5

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
            if price <= 0 or np.isnan(price):
                portfolio.append(portfolio[-1])
                continue

            # Get individual model signals
            lstm_signal = int(df['lstm_signal'].iloc[i])
            lstm_confidence = float(df['lstm_confidence'].iloc[i])
            lstm_buy_prob = float(df['lstm_buy_prob'].iloc[i])
            
            ensemble_signal = int(df['ensemble_signal'].iloc[i])
            ensemble_confidence = float(df['ensemble_confidence'].iloc[i])
            
            # PPO signals (if models available)
            ppo_actor_signal = 1
            ppo_actor_confidence = 0.5
            ppo_critic_signal = 1
            ppo_critic_confidence = 0.5
            
            if 'ppo_actor' in self.models:
                # Simple placeholder - can be enhanced based on PPO model output
                ppo_actor_signal = lstm_signal  # Default to LSTM
                ppo_actor_confidence = lstm_confidence * 0.8
                
            if 'ppo_critic' in self.models:
                ppo_critic_signal = lstm_signal
                ppo_critic_confidence = lstm_confidence * 0.7
            
            # Calculate consensus using all models
            consensus = self.consensus.calculate_consensus(
                lstm_signal, lstm_confidence, lstm_buy_prob,
                ensemble_signal, ensemble_confidence,
                ppo_actor_signal, ppo_actor_confidence,
                ppo_critic_signal, ppo_critic_confidence
            )
            
            df.at[i, 'consensus_signal'] = consensus['signal']
            df.at[i, 'consensus_confidence'] = consensus['confidence']
            
            # Get signal and quality (maintain original interface)
            signal = consensus['signal']
            quality = consensus['confidence']
            
            # Convert consensus signal to original format (1=buy, 0=sell)
            if signal == 2:  # BUY
                final_signal = 1
                final_quality = quality
            elif signal == 0:  # SELL
                final_signal = 0
                final_quality = quality
            else:  # HOLD
                final_signal = 1 if position == 0 else 0
                final_quality = 0.3

            # ORIGINAL TRADING LOGIC (UNCHANGED)
            if final_signal == 1 and position == 0 and final_quality >= 0.5:
                buy_price = price * (1 + slippage)
                position  = capital / buy_price * (1 - fee)
                entry_price = buy_price
                capital = 0
                trades.append({'type': 'buy', 'price': buy_price, 'bar': i, 'consensus_score': consensus['consensus_score'], 'lstm_override': consensus['lstm_override']})

            elif final_signal == 0 and position > 0:
                sell_price = price * (1 - slippage)
                capital    = position * sell_price * (1 - fee)
                pnl        = (sell_price - entry_price) / entry_price
                position   = 0
                trades.append({'type': 'sell', 'price': sell_price, 'pnl': pnl, 'bar': i})

            portfolio.append(capital + position * price)

        if position > 0:
            last_price = df['close'].iloc[-1]
            sell_price = last_price * (1 - slippage)
            final_capital = position * sell_price * (1 - fee)
            pnl = (sell_price - entry_price) / entry_price
            trades.append({'type': 'sell', 'price': sell_price, 'pnl': pnl, 'bar': len(df) - 1, 'forced': True})
            portfolio[-1] = final_capital

        final_value  = portfolio[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100

        returns = []
        for i in range(1, len(portfolio)):
            ret = (portfolio[i] - portfolio[i-1]) / (portfolio[i-1] + 1e-10)
            returns.append(ret)

        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 24) if returns else 0

        peak        = np.maximum.accumulate(portfolio)
        drawdown    = (np.array(portfolio) - peak) / (peak + 1e-10)
        max_drawdown = drawdown.min()

        sell_trades  = [t for t in trades if t['type'] == 'sell']
        wins         = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        total_closed = len(sell_trades)
        win_rate     = wins / total_closed if total_closed > 0 else 0
        
        # Add consensus stats to results
        consensus_stats = self.consensus.get_stats()

        return {
            'total_return': total_return,
            'sharpe':       sharpe,
            'max_drawdown': max_drawdown,
            'win_rate':     win_rate,
            'total_trades': total_closed,
            'final_capital': final_value,
            'lstm_overrides': consensus_stats.get('lstm_overrides', 0),
            'portfolio':    portfolio,
            'trades':       trades
        }

    def run(self, models_dir: str = None, data_path: str = None) -> Dict[str, Any]:
        self.start_time = time.time()

        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "PROFESSIONAL BACKTEST ENGINE v2.0" + " " * 28 + "║")
        print("║" + " " * 18 + "Multi-Model Consensus (LSTM BOSS)" + " " * 27 + "║")
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
        ResultsVisualizer.save_results(results)
        
        # Also print consensus-specific stats
        if results.get('lstm_overrides', 0) > 0:
            print(f"\n  🔵 LSTM Overrode helpers {results['lstm_overrides']} times")

        gc.collect()
        return results

def main():
    parser = argparse.ArgumentParser(description='Professional Backtest Engine')
    parser.add_argument('--models', type=str, default=None, help='Models directory')
    parser.add_argument('--data',   type=str, default=None, help='Data file or directory')
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