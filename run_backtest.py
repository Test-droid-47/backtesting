#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         PROFESSIONAL MULTI-MODEL CONSENSUS BACKTEST ENGINE v3.0               ║
║              LSTM as BOSS + Ensemble + PPO as Helpers                         ║
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
from collections import deque
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
    blue = "\x1b[34;20m"
    magenta = "\x1b[35;20m"
    bold_red = "\x1b[31;1m"
    bold_green = "\x1b[32;1m"
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

logger = logging.getLogger('ConsensusEngine')
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('consensus_backtest.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
    logger.addHandler(file_handler)

# ============================================================================
# CONSENSUS DECISION ENGINE
# ============================================================================
class ConsensusEngine:
    """Multi-model consensus system with LSTM as BOSS"""
    
    # Model weights (LSTM gets highest priority)
    MODEL_WEIGHTS = {
        'lstm': 0.45,      # BOSS - final decision maker
        'ensemble': 0.25,  # Helper 1 - strong confirmator
        'ppo_actor': 0.15, # Helper 2 - action specialist
        'ppo_critic': 0.15 # Helper 3 - value/risk assessor
    }
    
    # Decision thresholds
    BUY_THRESHOLD = 0.65
    SELL_THRESHOLD = 0.35
    STRONG_BUY = 0.75
    STRONG_SELL = 0.25
    HOLD_ZONE = (0.35, 0.65)
    
    # Agreement levels
    AGREEMENT_HIGH = 0.80
    AGREEMENT_MEDIUM = 0.60
    
    def __init__(self):
        self.decision_history = deque(maxlen=100)
        self.consensus_history = deque(maxlen=100)
        self.lstm_override_count = 0
        self.helper_influence_count = 0
        
    def get_lstm_decision(self, lstm_output) -> Tuple[int, float]:
        """
        LSTM (BOSS) decides:
        Returns: (signal, confidence)
        signal: 0=Sell, 1=Hold, 2=Buy
        """
        if isinstance(lstm_output, (list, tuple)):
            # Case 1: LSTM outputs multiple heads
            if len(lstm_output) >= 3:
                # Assuming format: [direction_probs, entry_quality, position_size, ...]
                direction_probs = lstm_output[1] if len(lstm_output) > 1 else lstm_output[0]
                entry_quality = lstm_output[2] if len(lstm_output) > 2 else 0.5
                
                if isinstance(direction_probs, np.ndarray):
                    if direction_probs.ndim == 2:
                        probs = direction_probs[0] if len(direction_probs) > 0 else [0.33, 0.33, 0.34]
                    else:
                        probs = direction_probs
                else:
                    probs = [0.33, 0.33, 0.34]
                
                # Convert to (buy, hold, sell) probabilities
                if len(probs) >= 3:
                    buy_prob = probs[2] if len(probs) > 2 else probs[1]
                    sell_prob = probs[0]
                    hold_prob = probs[1] if len(probs) > 1 else 0.33
                else:
                    buy_prob = float(probs[0]) if len(probs) > 0 else 0.5
                    sell_prob = 1 - buy_prob
                    hold_prob = 0.33
                    
            else:
                buy_prob = float(lstm_output[0]) if len(lstm_output) > 0 else 0.5
                sell_prob = 1 - buy_prob
                hold_prob = 0.33
                entry_quality = 0.5
                
        elif isinstance(lstm_output, np.ndarray):
            if lstm_output.ndim == 1:
                if len(lstm_output) >= 3:
                    buy_prob = lstm_output[2]
                    sell_prob = lstm_output[0]
                    hold_prob = lstm_output[1]
                else:
                    buy_prob = lstm_output[0]
                    sell_prob = 1 - buy_prob
                    hold_prob = 0.33
            else:
                buy_prob = float(np.mean(lstm_output[:, 2])) if lstm_output.shape[1] > 2 else 0.5
                sell_prob = 1 - buy_prob
                hold_prob = 0.33
            entry_quality = 0.5
        else:
            buy_prob = float(lstm_output) if isinstance(lstm_output, (int, float)) else 0.5
            sell_prob = 1 - buy_prob
            hold_prob = 0.33
            entry_quality = 0.5
            
        # Calculate confidence from probability spread
        confidence = abs(buy_prob - sell_prob) * 2
        confidence = min(max(confidence, 0.3), 0.95)
        
        # Determine signal
        if buy_prob > self.BUY_THRESHOLD:
            signal = 2  # BUY
        elif sell_prob > self.SELL_THRESHOLD:
            signal = 0  # SELL
        else:
            signal = 1  # HOLD
            
        return signal, confidence, buy_prob, entry_quality
    
    def get_helper_decisions(self, ensemble_pred: float, ppo_actor: Any, ppo_critic: Any) -> Dict:
        """Get decisions from helper models"""
        helper_votes = {
            'ensemble': {'signal': 1, 'confidence': 0.5, 'value': 0.5},
            'ppo_actor': {'signal': 1, 'confidence': 0.5, 'value': 0.5},
            'ppo_critic': {'signal': 1, 'confidence': 0.5, 'value': 0.5}
        }
        
        # Ensemble Helper (classification)
        if ensemble_pred is not None:
            ensemble_value = float(ensemble_pred) if isinstance(ensemble_pred, (int, float)) else 0.5
            helper_votes['ensemble']['value'] = ensemble_value
            helper_votes['ensemble']['confidence'] = abs(ensemble_value - 0.5) * 2
            
            if ensemble_value > self.BUY_THRESHOLD:
                helper_votes['ensemble']['signal'] = 2
            elif ensemble_value < self.SELL_THRESHOLD:
                helper_votes['ensemble']['signal'] = 0
            else:
                helper_votes['ensemble']['signal'] = 1
        
        # PPO Actor Helper (action specialist)
        if ppo_actor is not None:
            try:
                if hasattr(ppo_actor, 'predict'):
                    actor_out = ppo_actor.predict(0)  # Placeholder
                    if isinstance(actor_out, (list, tuple, np.ndarray)):
                        actor_val = float(actor_out[0]) if len(actor_out) > 0 else 0.5
                    else:
                        actor_val = float(actor_out)
                else:
                    actor_val = 0.5
                    
                helper_votes['ppo_actor']['value'] = actor_val
                helper_votes['ppo_actor']['confidence'] = abs(actor_val - 0.5) * 1.5
                
                if actor_val > self.BUY_THRESHOLD + 0.05:
                    helper_votes['ppo_actor']['signal'] = 2
                elif actor_val < self.SELL_THRESHOLD - 0.05:
                    helper_votes['ppo_actor']['signal'] = 0
                else:
                    helper_votes['ppo_actor']['signal'] = 1
            except Exception as e:
                logger.debug(f"PPO Actor prediction failed: {e}")
        
        # PPO Critic Helper (value/risk assessor)
        if ppo_critic is not None:
            try:
                if hasattr(ppo_critic, 'predict'):
                    critic_out = ppo_critic.predict(0)
                    if isinstance(critic_out, (list, tuple, np.ndarray)):
                        critic_val = float(critic_out[0]) if len(critic_out) > 0 else 0.5
                    else:
                        critic_val = float(critic_out)
                else:
                    critic_val = 0.5
                    
                helper_votes['ppo_critic']['value'] = critic_val
                helper_votes['ppo_critic']['confidence'] = abs(critic_val - 0.5) * 1.5
                
                # Critic provides risk assessment
                if critic_val > 0.7:
                    helper_votes['ppo_critic']['signal'] = 2  # Low risk, good entry
                elif critic_val < 0.3:
                    helper_votes['ppo_critic']['signal'] = 0  # High risk
                else:
                    helper_votes['ppo_critic']['signal'] = 1
            except Exception as e:
                logger.debug(f"PPO Critic prediction failed: {e}")
                
        return helper_votes
    
    def calculate_consensus(self, lstm_signal: int, lstm_confidence: float, 
                           lstm_buy_prob: float, helper_votes: Dict) -> Dict:
        """Calculate weighted consensus decision"""
        
        # Weighted voting calculation
        weighted_buy_score = 0
        weighted_sell_score = 0
        total_weight = 0
        
        # LSTM (BOSS) vote
        lstm_weight = self.MODEL_WEIGHTS['lstm']
        if lstm_signal == 2:  # BUY
            weighted_buy_score += lstm_weight * lstm_confidence
        elif lstm_signal == 0:  # SELL
            weighted_sell_score += lstm_weight * lstm_confidence
        else:  # HOLD
            weighted_buy_score += lstm_weight * 0.5
            weighted_sell_score += lstm_weight * 0.5
        total_weight += lstm_weight
        
        # Helper votes
        for model_name, weight in self.MODEL_WEIGHTS.items():
            if model_name == 'lstm':
                continue
                
            if model_name in helper_votes:
                vote = helper_votes[model_name]
                model_weight = weight
                
                if vote['signal'] == 2:
                    weighted_buy_score += model_weight * vote['confidence']
                elif vote['signal'] == 0:
                    weighted_sell_score += model_weight * vote['confidence']
                else:
                    # Hold = neutral
                    weighted_buy_score += model_weight * 0.3
                    weighted_sell_score += model_weight * 0.3
                total_weight += model_weight
        
        # Normalize scores
        if total_weight > 0:
            buy_score = weighted_buy_score / total_weight
            sell_score = weighted_sell_score / total_weight
        else:
            buy_score = 0.5
            sell_score = 0.5
            
        # Final consensus score (normalized to 0-1, where 1=strong buy)
        consensus_score = buy_score / (buy_score + sell_score + 1e-10)
        consensus_score = min(max(consensus_score, 0), 1)
        
        # Agreement level between models
        all_signals = [lstm_signal] + [v['signal'] for k, v in helper_votes.items()]
        agreement = 1 - (len(set(all_signals)) - 1) / 3  # 1 if all agree, 0 if all different
        
        # Final decision with override logic
        final_signal = 1  # Default HOLD
        final_confidence = consensus_score
        
        # LSTM Override when confidence is high
        if lstm_confidence > 0.8 and agreement < 0.5:
            # BOSS overrides disagreement
            final_signal = lstm_signal
            final_confidence = lstm_confidence * 0.9
            self.lstm_override_count += 1
            logger.debug(f"LSTM OVERRIDE: Disagreement={1-agreement:.2f}, LSTM confidence={lstm_confidence:.2f}")
        else:
            # Normal consensus
            if consensus_score > self.BUY_THRESHOLD:
                final_signal = 2  # BUY
            elif consensus_score < self.SELL_THRESHOLD:
                final_signal = 0  # SELL
            else:
                final_signal = 1  # HOLD
                
            if agreement > self.AGREEMENT_HIGH:
                final_confidence = min(consensus_score * 1.2, 0.95)  # Boost confidence
            elif agreement < self.AGREEMENT_MEDIUM:
                final_confidence = consensus_score * 0.7  # Reduce confidence on disagreement
                
        # Position sizing based on confidence
        if final_signal == 2:
            position_size = min(0.25 * final_confidence, 0.20)  # Max 20% per trade
        elif final_signal == 0:
            position_size = 0  # Full exit on sell signal
        else:
            position_size = 0
            
        # Track history
        self.decision_history.append(final_signal)
        self.consensus_history.append(consensus_score)
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'position_size': position_size,
            'consensus_score': consensus_score,
            'agreement': agreement,
            'lstm_override': lstm_confidence > 0.8 and agreement < 0.5,
            'lstm_signal': lstm_signal,
            'lstm_confidence': lstm_confidence,
            'lstm_buy_prob': lstm_buy_prob,
            'helper_votes': helper_votes
        }
    
    def get_risk_adjustment(self, consecutive_losses: int, drawdown: float) -> float:
        """Dynamic risk adjustment based on performance"""
        risk_multiplier = 1.0
        
        if consecutive_losses >= 3:
            risk_multiplier *= 0.5
        elif consecutive_losses >= 2:
            risk_multiplier *= 0.75
            
        if drawdown < -0.15:  # >15% drawdown
            risk_multiplier *= 0.5
        elif drawdown < -0.10:
            risk_multiplier *= 0.7
        elif drawdown < -0.05:
            risk_multiplier *= 0.9
            
        return min(max(risk_multiplier, 0.25), 1.0)
    
    def get_stats(self) -> Dict:
        """Get consensus engine statistics"""
        if len(self.decision_history) == 0:
            return {}
            
        return {
            'lstm_overrides': self.lstm_override_count,
            'avg_consensus': np.mean(self.consensus_history) if self.consensus_history else 0,
            'avg_agreement': 1 - (len(set(self.decision_history)) - 1) / 3 if self.decision_history else 0
        }

# ============================================================================
# FEATURE ENGINEERING (Enhanced)
# ============================================================================
class FeatureEngineer:
    """Enhanced feature engineering for multi-model input"""

    @staticmethod
    def _safe(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan)

    @classmethod
    def calculate_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']

        logger.info("  ⚙️  Calculating comprehensive features...")

        # ── TREND INDICATORS ────────────────────────────────────────────────
        for p in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{p}'] = c.rolling(p).mean()
            df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()

        df['sma_ratio_7_21'] = cls._safe(df['sma_7'] / df['sma_21'])
        df['sma_ratio_21_50'] = cls._safe(df['sma_21'] / df['sma_50'])
        df['price_vs_sma50'] = cls._safe(c / df['sma_50'])
        df['price_vs_sma200'] = cls._safe(c / df['sma_200'])

        # ── MOMENTUM OSCILLATORS ────────────────────────────────────────────
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
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_norm'] = cls._safe(df['macd'] / (c + 1e-10))

        # Stochastic
        for p in [14, 21]:
            lo = l.rolling(p).min()
            hi = h.rolling(p).max()
            df[f'stoch_k_{p}'] = cls._safe(100 * (c - lo) / (hi - lo + 1e-10))
            df[f'stoch_d_{p}'] = df[f'stoch_k_{p}'].rolling(3).mean()

        # CCI
        tp = (h + l + c) / 3
        df['cci_14'] = cls._safe((tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std() + 1e-10))

        # ── VOLUME INDICATORS ───────────────────────────────────────────────
        df['volume_sma_20'] = v.rolling(20).mean()
        df['volume_ratio'] = cls._safe(v / (df['volume_sma_20'] + 1e-10))
        df['obv'] = (np.sign(c.diff()) * v).fillna(0).cumsum()
        
        # VWAP
        typical = (h + l + c) / 3
        df['vwap_20'] = cls._safe((typical * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10))
        df['price_vs_vwap'] = cls._safe(c / (df['vwap_20'] + 1e-10))

        # ── VOLATILITY ──────────────────────────────────────────────────────
        ret = c.pct_change()
        for p in [10, 20, 30]:
            df[f'volatility_{p}'] = ret.rolling(p).std()

        # Bollinger Bands
        mid = c.rolling(20).mean()
        std = c.rolling(20).std()
        df['bb_upper'] = mid + 2 * std
        df['bb_lower'] = mid - 2 * std
        df['bb_width'] = cls._safe((df['bb_upper'] - df['bb_lower']) / (mid + 1e-10))
        df['bb_pct'] = cls._safe((c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10))

        # ATR
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_norm'] = cls._safe(df['atr_14'] / (c + 1e-10))

        # ── PRICE ACTION ────────────────────────────────────────────────────
        df['hl_ratio'] = cls._safe((h - l) / (c + 1e-10))
        df['body_size'] = cls._safe(abs(c - o) / (h - l + 1e-10))
        df['upper_wick'] = cls._safe((h - c.combine(o, max)) / (h - l + 1e-10))
        df['lower_wick'] = cls._safe((c.combine(o, min) - l) / (h - l + 1e-10))

        # ── TIME FEATURES ───────────────────────────────────────────────────
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            df['hour'] = ts.dt.hour
            df['dayofweek'] = ts.dt.dayofweek
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        logger.info(f"  ✅ Feature engineering complete — {len(df.columns)} total columns")
        return df

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
                
                if 'lstm' in file.lower() and file.endswith('.keras'):
                    models['lstm'] = full_path
                    logger.info(f"  ✅ Found LSTM (BOSS): {file}")
                elif 'ensemble' in file.lower() and file.endswith('.pkl'):
                    models['ensemble'] = full_path
                    logger.info(f"  ✅ Found Ensemble (Helper): {file}")
                elif 'ppo_agent_actor' in file.lower() and file.endswith('.keras'):
                    models['ppo_actor'] = full_path
                    logger.info(f"  ✅ Found PPO Actor (Helper): {file}")
                elif 'ppo_agent_critic' in file.lower() and file.endswith('.keras'):
                    models['ppo_critic'] = full_path
                    logger.info(f"  ✅ Found PPO Critic (Helper): {file}")
                elif 'scaler' in file.lower() and file.endswith('.pkl'):
                    models['scaler'] = full_path
                    logger.info(f"  ✅ Found Scaler: {file}")

        return models

# ============================================================================
# MAIN BACKTEST ENGINE
# ============================================================================
class ConsensusBacktestEngine:
    """Multi-model consensus backtest engine"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.models = {}
        self.data = {}
        self.consensus_engine = ConsensusEngine()
        self.start_time = None
        
    def _load_config(self, config_path: str = None) -> dict:
        default_config = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'fee_rate': 0.01,
            'slippage': 0.05,
            'initial_capital': 100,
            'window': 60,
            'max_position_pct': 0.10,  # Max 10% per trade
            'min_consensus_threshold': 0.55,
            'stop_loss_pct': 0.05,  # 5% stop loss
            'take_profit_pct': 0.10,  # 10% take profit
            'max_consecutive_losses': 3
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                default_config.update(cfg)
                logger.info(f"✅ Config loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
                
        return default_config
    
    def load_models(self, models_dir: str = None) -> bool:
        """Load all 5 models"""
        logger.info("=" * 70)
        logger.info("📦 MULTI-MODEL LOADING PHASE")
        logger.info("=" * 70)
        
        discovered = FileDiscovery.find_models(models_dir or '.')
        
        if not discovered:
            logger.error("❌ No models found!")
            return False
            
        # Load LSTM (BOSS)
        if 'lstm' in discovered:
            try:
                from tensorflow.keras.models import load_model
                self.models['lstm'] = load_model(discovered['lstm'])
                logger.info(f"  ✅ LSTM BOSS loaded: {os.path.basename(discovered['lstm'])}")
            except Exception as e:
                logger.error(f"  ❌ Failed to load LSTM: {e}")
                return False
                
        # Load Ensemble Helper
        if 'ensemble' in discovered:
            try:
                import joblib
                self.models['ensemble'] = joblib.load(discovered['ensemble'])
                logger.info(f"  ✅ Ensemble Helper loaded: {os.path.basename(discovered['ensemble'])}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load Ensemble: {e}")
                
        # Load PPO Actor Helper
        if 'ppo_actor' in discovered:
            try:
                from tensorflow.keras.models import load_model
                self.models['ppo_actor'] = load_model(discovered['ppo_actor'])
                logger.info(f"  ✅ PPO Actor Helper loaded: {os.path.basename(discovered['ppo_actor'])}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load PPO Actor: {e}")
                
        # Load PPO Critic Helper
        if 'ppo_critic' in discovered:
            try:
                from tensorflow.keras.models import load_model
                self.models['ppo_critic'] = load_model(discovered['ppo_critic'])
                logger.info(f"  ✅ PPO Critic Helper loaded: {os.path.basename(discovered['ppo_critic'])}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load PPO Critic: {e}")
                
        # Load Scaler
        if 'scaler' in discovered:
            try:
                import joblib
                self.models['scaler'] = joblib.load(discovered['scaler'])
                logger.info(f"  ✅ Scaler loaded")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load scaler: {e}")
                
        return 'lstm' in self.models
    
    def load_data(self, data_path: str = None) -> bool:
        """Load and prepare data"""
        logger.info("=" * 70)
        logger.info("📊 DATA LOADING PHASE")
        logger.info("=" * 70)
        
        if not data_path or not os.path.exists(data_path):
            logger.error("❌ No data file provided or file not found!")
            return False
            
        try:
            df = pd.read_csv(data_path)
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"  ✅ Timestamp parsed")
                
            # Validate required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.error(f"  ❌ Missing columns: {missing}")
                return False
                
            # Convert to numeric
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df.dropna(subset=['close'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            self.data['df'] = df
            logger.info(f"  ✅ Loaded {len(df)} bars")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Failed to load data: {e}")
            return False
            
    def prepare_features(self) -> bool:
        """Prepare features for all models"""
        logger.info("=" * 70)
        logger.info("🔧 FEATURE PREPARATION")
        logger.info("=" * 70)
        
        df = FeatureEngineer.calculate_all(self.data['df'])
        
        # Define core features
        core_features = [
            'close', 'open', 'high', 'low', 'volume',
            'rsi_14', 'macd', 'macd_hist', 'stoch_k_14', 'stoch_d_14',
            'cci_14', 'atr_14', 'atr_norm', 'bb_width', 'bb_pct',
            'volatility_20', 'volume_ratio', 'price_vs_vwap',
            'hl_ratio', 'body_size', 'upper_wick', 'lower_wick',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        ]
        
        # Use available features
        available = [f for f in core_features if f in df.columns]
        
        if not available:
            logger.error("  ❌ No features available!")
            return False
            
        # Handle missing values
        data = df[available].copy()
        data = data.ffill().bfill().fillna(0)
        
        # Scale if scaler available
        if 'scaler' in self.models:
            try:
                scaler_features = self.models['scaler'].feature_names_in_.tolist()
                for f in scaler_features:
                    if f not in data.columns:
                        data[f] = 0
                data = data[scaler_features]
                data_scaled = self.models['scaler'].transform(data)
                logger.info(f"  ✅ Data scaled with {len(scaler_features)} features")
                data = data_scaled
            except Exception as e:
                logger.warning(f"  ⚠️ Scaling failed: {e}, using raw data")
                data = data.values
        else:
            data = data.values
            
        # Create sequences
        window = self.config.get('window', 60)
        if len(data) < window:
            logger.error(f"  ❌ Not enough data: {len(data)} < {window}")
            return False
            
        X = np.array([data[i-window:i] for i in range(window, len(data))])
        
        self.data['X'] = X
        self.data['window'] = window
        self.data['feature_names'] = available
        
        logger.info(f"  ✅ Created {len(X)} sequences (window={window}, features={data.shape[1]})")
        return True
    
    def generate_predictions(self) -> bool:
        """Generate predictions from all models"""
        logger.info("=" * 70)
        logger.info("🤖 MULTI-MODEL PREDICTION PHASE")
        logger.info("=" * 70)
        
        df = self.data['df']
        X = self.data['X']
        window = self.data['window']
        
        # Initialize prediction columns
        df['lstm_signal'] = 1
        df['lstm_confidence'] = 0.5
        df['lstm_buy_prob'] = 0.5
        df['ensemble_pred'] = 0.5
        df['consensus_signal'] = 1
        df['consensus_confidence'] = 0.5
        df['position_size'] = 0
        
        # Generate LSTM (BOSS) predictions
        if 'lstm' in self.models:
            print(f"\n  🦁 LSTM BOSS: Generating predictions on {len(X)} samples...")
            try:
                lstm_outputs = self.models['lstm'].predict(X, verbose=0)
                
                for i, idx in enumerate(range(window, len(df))):
                    lstm_out = lstm_outputs[i] if len(lstm_outputs) > i else lstm_outputs
                    signal, confidence, buy_prob, _ = self.consensus_engine.get_lstm_decision(lstm_out)
                    
                    df.at[idx, 'lstm_signal'] = signal
                    df.at[idx, 'lstm_confidence'] = confidence
                    df.at[idx, 'lstm_buy_prob'] = buy_prob
                    
                logger.info(f"  ✅ LSTM predictions generated")
            except Exception as e:
                logger.error(f"  ❌ LSTM prediction failed: {e}")
                return False
                
        # Generate Ensemble Helper predictions
        if 'ensemble' in self.models:
            print(f"\n  🤝 Ensemble Helper: Generating predictions...")
            try:
                for i in tqdm(range(len(df)), desc="  Ensemble", leave=False):
                    try:
                        row = df.iloc[i]
                        # Assuming ensemble has predict_proba_bullish method
                        if hasattr(self.models['ensemble'], 'predict_proba_bullish'):
                            prob = self.models['ensemble'].predict_proba_bullish(row)
                        else:
                            prob = 0.5
                        df.at[i, 'ensemble_pred'] = float(prob)
                    except:
                        df.at[i, 'ensemble_pred'] = 0.5
                logger.info(f"  ✅ Ensemble predictions generated")
            except Exception as e:
                logger.warning(f"  ⚠️ Ensemble failed: {e}")
                df['ensemble_pred'] = 0.5
        else:
            df['ensemble_pred'] = 0.5
            
        self.data['df'] = df
        return True
    
    def run_backtest(self) -> Dict:
        """Run backtest with consensus decisions"""
        logger.info("=" * 70)
        logger.info("💰 RUNNING CONSENSUS BACKTEST")
        logger.info("=" * 70)
        
        df = self.data['df']
        capital = self.config['initial_capital']
        initial_capital = capital
        position = 0
        entry_price = 0
        entry_idx = 0
        trades = []
        portfolio = [capital]
        
        fee = self.config['fee_rate']
        slippage = self.config['slippage']
        stop_loss_pct = self.config['stop_loss_pct']
        take_profit_pct = self.config['take_profit_pct']
        
        consecutive_losses = 0
        win_streak = 0
        
        print(f"\n  Running consensus backtest on {len(df)} bars...")
        
        for i in tqdm(range(len(df)), desc="  Backtest", leave=False):
            price = df['close'].iloc[i]
            if price <= 0 or np.isnan(price):
                portfolio.append(portfolio[-1])
                continue
                
            # Get model predictions
            lstm_signal = int(df['lstm_signal'].iloc[i])
            lstm_confidence = float(df['lstm_confidence'].iloc[i])
            lstm_buy_prob = float(df['lstm_buy_prob'].iloc[i])
            ensemble_pred = float(df['ensemble_pred'].iloc[i])
            
            # Get helper decisions
            helper_votes = self.consensus_engine.get_helper_decisions(
                ensemble_pred, 
                self.models.get('ppo_actor'), 
                self.models.get('ppo_critic')
            )
            
            # Calculate consensus
            consensus = self.consensus_engine.calculate_consensus(
                lstm_signal, lstm_confidence, lstm_buy_prob, helper_votes
            )
            
            # Apply risk adjustment
            current_dd = (portfolio[-1] - max(portfolio[:i+1])) / (max(portfolio[:i+1]) + 1e-10)
            risk_mult = self.consensus_engine.get_risk_adjustment(consecutive_losses, current_dd)
            
            final_signal = consensus['signal']
            final_confidence = consensus['confidence'] * risk_mult
            
            # Store consensus results
            df.at[i, 'consensus_signal'] = final_signal
            df.at[i, 'consensus_confidence'] = final_confidence
            
            # Check stop loss / take profit if in position
            if position > 0:
                pnl_pct = (price - entry_price) / entry_price
                
                # Stop loss hit
                if pnl_pct <= -stop_loss_pct:
                    sell_price = price * (1 - slippage)
                    capital = position * sell_price * (1 - fee)
                    pnl = (sell_price - entry_price) / entry_price
                    
                    trades.append({
                        'type': 'sell', 'price': sell_price, 'pnl': pnl,
                        'reason': 'stop_loss', 'bar': i
                    })
                    position = 0
                    consecutive_losses += 1
                    win_streak = 0
                    
                # Take profit hit
                elif pnl_pct >= take_profit_pct:
                    sell_price = price * (1 - slippage)
                    capital = position * sell_price * (1 - fee)
                    pnl = (sell_price - entry_price) / entry_price
                    
                    trades.append({
                        'type': 'sell', 'price': sell_price, 'pnl': pnl,
                        'reason': 'take_profit', 'bar': i
                    })
                    position = 0
                    consecutive_losses = 0
                    win_streak += 1
                    
            # Entry signal
            if final_signal == 2 and position == 0 and final_confidence >= self.config['min_consensus_threshold']:
                position_size_pct = consensus['position_size'] * risk_mult
                position_size_pct = min(position_size_pct, self.config['max_position_pct'])
                
                buy_price = price * (1 + slippage)
                position = capital * position_size_pct / buy_price * (1 - fee)
                capital -= capital * position_size_pct
                entry_price = buy_price
                entry_idx = i
                
                trades.append({
                    'type': 'buy', 'price': buy_price, 'size_pct': position_size_pct,
                    'confidence': final_confidence, 'consensus_score': consensus['consensus_score'],
                    'agreement': consensus['agreement'], 'bar': i
                })
                
            # Exit signal
            elif final_signal == 0 and position > 0:
                sell_price = price * (1 - slippage)
                capital = position * sell_price * (1 - fee)
                pnl = (sell_price - entry_price) / entry_price
                
                trades.append({
                    'type': 'sell', 'price': sell_price, 'pnl': pnl,
                    'reason': 'consensus_exit', 'bar': i
                })
                position = 0
                
                if pnl > 0:
                    consecutive_losses = 0
                    win_streak += 1
                else:
                    consecutive_losses += 1
                    win_streak = 0
                    
            portfolio.append(capital + position * price)
            
        # Close any open position
        if position > 0:
            last_price = df['close'].iloc[-1]
            sell_price = last_price * (1 - slippage)
            final_capital = position * sell_price * (1 - fee)
            pnl = (sell_price - entry_price) / entry_price
            trades.append({'type': 'sell', 'price': sell_price, 'pnl': pnl, 'forced': True})
            portfolio[-1] = final_capital
            
        # Calculate metrics
        final_value = portfolio[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        returns = []
        for i in range(1, len(portfolio)):
            ret = (portfolio[i] - portfolio[i-1]) / (portfolio[i-1] + 1e-10)
            returns.append(ret)
            
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 24) if returns else 0
        
        peak = np.maximum.accumulate(portfolio)
        drawdown = (np.array(portfolio) - peak) / (peak + 1e-10)
        max_drawdown = drawdown.min()
        
        sell_trades = [t for t in trades if t['type'] == 'sell' and 'pnl' in t]
        wins = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        total_closed = len(sell_trades)
        win_rate = wins / total_closed if total_closed > 0 else 0
        
        # Profit factor
        gross_profit = sum([t['pnl'] for t in sell_trades if t.get('pnl', 0) > 0])
        gross_loss = abs(sum([t['pnl'] for t in sell_trades if t.get('pnl', 0) < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Get consensus stats
        consensus_stats = self.consensus_engine.get_stats()
        
        results = {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_closed,
            'final_capital': final_value,
            'lstm_overrides': consensus_stats.get('lstm_overrides', 0),
            'avg_consensus': consensus_stats.get('avg_consensus', 0),
            'portfolio': portfolio,
            'trades': trades
        }
        
        self.data['df'] = df
        return results
    
    def display_results(self, results: Dict, elapsed: float):
        """Display comprehensive results"""
        print("\n" + "╔" + "═" * 80 + "╗")
        print("║" + " " * 22 + "CONSENSUS BACKTEST RESULTS" + " " * 33 + "║")
        print("╠" + "═" * 80 + "╣")
        
        metrics = [
            ("Total Return", f"{results.get('total_return', 0):.2f}%", 
             "🟢" if results.get('total_return', 0) > 20 else "🟡" if results.get('total_return', 0) > 0 else "🔴"),
            ("Sharpe Ratio", f"{results.get('sharpe', 0):.4f}",
             "🟢" if results.get('sharpe', 0) > 1.5 else "🟡" if results.get('sharpe', 0) > 0.8 else "🔴"),
            ("Max Drawdown", f"{results.get('max_drawdown', 0)*100:.2f}%",
             "🟢" if results.get('max_drawdown', 0) > -0.1 else "🟡" if results.get('max_drawdown', 0) > -0.2 else "🔴"),
            ("Win Rate", f"{results.get('win_rate', 0)*100:.1f}%",
             "🟢" if results.get('win_rate', 0) > 0.55 else "🟡" if results.get('win_rate', 0) > 0.45 else "🔴"),
            ("Profit Factor", f"{results.get('profit_factor', 0):.2f}",
             "🟢" if results.get('profit_factor', 0) > 1.5 else "🟡" if results.get('profit_factor', 0) > 1.0 else "🔴"),
            ("Total Trades", f"{results.get('total_trades', 0)}", "⚪"),
            ("Final Capital", f"${results.get('final_capital', 0):,.2f}", "🟢"),
            ("LSTM Overrides", f"{results.get('lstm_overrides', 0)}", "🔵"),
        ]
        
        for metric, value, status in metrics:
            print(f"║   {status} {metric:<20}: {value:>25} {status} ║")
            
        print("╠" + "═" * 80 + "╣")
        
        # Final verdict
        sharpe = results.get('sharpe', 0)
        win_rate = results.get('win_rate', 0)
        total_return = results.get('total_return', 0)
        
        if sharpe > 1.5 and win_rate > 0.55 and total_return > 20:
            verdict = "EXCELLENT - Multi-model consensus working perfectly! 🚀"
        elif sharpe > 1.0 and win_rate > 0.5:
            verdict = "GOOD - Consensus system is effective ✅"
        elif sharpe > 0.5 or total_return > 0:
            verdict = "AVERAGE - Consider adjusting model weights ⚠️"
        else:
            verdict = "POOR - Check model quality or consensus logic ❌"
            
        print(f"║   🎯 {verdict:<74} ║")
        print("╠" + "═" * 80 + "╣")
        print(f"║   ⏱️  Backtest completed in {elapsed:.2f} seconds" + " " * (80 - 37 - len(f"{elapsed:.2f}")) + "║")
        print("╚" + "═" * 80 + "╝")
        
    def save_results(self, results: Dict):
        """Save results to JSON"""
        try:
            serializable = {k: v for k, v in results.items() if k not in ('portfolio', 'trades')}
            serializable['portfolio'] = [float(x) for x in results.get('portfolio', [])]
            serializable['trades'] = results.get('trades', [])
            
            output = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'config': self.config,
                'results': serializable
            }
            
            with open('consensus_backtest_results.json', 'w') as f:
                json.dump(output, f, indent=2, default=str)
                
            logger.info("📁 Results saved to 'consensus_backtest_results.json'")
        except Exception as e:
            logger.warning(f"⚠️ Could not save results: {e}")
            
    def run(self, models_dir: str = None, data_path: str = None) -> Dict:
        """Main execution pipeline"""
        self.start_time = time.time()
        
        print("\n" + "╔" + "═" * 80 + "╗")
        print("║" + " " * 15 + "PROFESSIONAL CONSENSUS BACKTEST ENGINE v3.0" + " " * 21 + "║")
        print("║" + " " * 25 + "LSTM as BOSS + Ensemble + PPO as Helpers" + " " * 18 + "║")
        print("╚" + "═" * 80 + "╝")
        
        # Step 1: Load all models
        if not self.load_models(models_dir):
            logger.error("❌ Backtest aborted: No models loaded")
            return {'error': 'No models loaded'}
            
        # Step 2: Load data
        if not self.load_data(data_path):
            logger.error("❌ Backtest aborted: No data loaded")
            return {'error': 'No data loaded'}
            
        # Step 3: Prepare features
        if not self.prepare_features():
            logger.error("❌ Backtest aborted: Feature preparation failed")
            return {'error': 'Feature preparation failed'}
            
        # Step 4: Generate predictions
        if not self.generate_predictions():
            logger.error("❌ Backtest aborted: Prediction generation failed")
            return {'error': 'Prediction generation failed'}
            
        # Step 5: Run backtest
        results = self.run_backtest()
        
        # Step 6: Display and save
        elapsed = time.time() - self.start_time
        self.display_results(results, elapsed)
        self.save_results(results)
        
        # Cleanup
        gc.collect()
        return results
# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Multi-Model Consensus Backtest Engine')
    parser.add_argument('--models', type=str, default='./models', help='Models directory')
    parser.add_argument('--data', type=str, required=True, help='Data CSV file path')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    engine = ConsensusBacktestEngine(config_path=args.config)
    result = engine.run(models_dir=args.models, data_path=args.data)
    
    return 0 if 'error' not in result else 1

if __name__ == '__main__':
    exit(main())