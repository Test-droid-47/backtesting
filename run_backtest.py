#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              PROFESSIONAL BACKTEST ENGINE v3.1                                ║
║              5-Model Ensemble — LSTM as Boss, Ensemble as Co-Decision Maker   ║
║              PPO Actor/Critic Optional (Configurable)                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Signal Flow:
  LSTM (Boss)  →  raw signal: BUY / SELL / HOLD
  Ensemble     →  bullish probability (co-decision maker)
  PPO Actor    →  action probability confirmation (OPTIONAL, disabled by default)
  PPO Critic   →  state value confidence (OPTIONAL, disabled by default)
  ─────────────────────────────────────────────────────────────
  FinalSignal  =  LSTM signal confirmed by Ensemble (and optional helpers)
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
# CONSTANTS
# ============================================================================
SIGNAL_BUY  = 1
SIGNAL_SELL = 0
SIGNAL_HOLD = 2

# ============================================================================
# LOGGING
# ============================================================================
class CustomFormatter(logging.Formatter):
    grey     = "\x1b[38;20m"
    yellow   = "\x1b[33;20m"
    red      = "\x1b[31;20m"
    cyan     = "\x1b[36;20m"
    bold_red = "\x1b[31;1m"
    reset    = "\x1b[0m"
    FORMATS  = {
        logging.DEBUG:    grey     + "[%(asctime)s] [DEBUG] %(message)s"    + reset,
        logging.INFO:     cyan     + "[%(asctime)s] [INFO] %(message)s"     + reset,
        logging.WARNING:  yellow   + "[%(asctime)s] [WARNING] %(message)s"  + reset,
        logging.ERROR:    red      + "[%(asctime)s] [ERROR] %(message)s"    + reset,
        logging.CRITICAL: bold_red + "[%(asctime)s] [CRITICAL] %(message)s" + reset,
    }
    def format(self, record):
        fmt = logging.Formatter(self.FORMATS.get(record.levelno), datefmt='%Y-%m-%d %H:%M:%S')
        return fmt.format(record)

logger = logging.getLogger('BacktestEngine')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    fh = logging.FileHandler('backtest_detailed.log')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

# ============================================================================
# FILE DISCOVERY
# ============================================================================
class FileDiscovery:
    @staticmethod
    def find_models(base_path: str = '.') -> Dict[str, str]:
        found = {}
        search = [base_path, os.path.join(base_path, 'models')]
        logger.info("🔍 Searching for model files...")
        for sp in search:
            if not os.path.exists(sp):
                continue
            for f in os.listdir(sp):
                fp = os.path.join(sp, f)
                fl = f.lower()
                if fl == 'lstm_model.keras':
                    found['lstm'] = fp
                    logger.info(f"  ✅ LSTM (Boss): {f}")
                elif fl == 'ppo_agent_actor.keras':
                    found['ppo_actor'] = fp
                    logger.info(f"  ✅ PPO Actor:   {f}")
                elif fl == 'ppo_agent_critic.keras':
                    found['ppo_critic'] = fp
                    logger.info(f"  ✅ PPO Critic:  {f}")
                elif fl == 'ensemble_model.pkl':
                    found['ensemble'] = fp
                    logger.info(f"  ✅ Ensemble:    {f}")
                elif fl == 'scaler.pkl':
                    found['scaler'] = fp
                    logger.info(f"  ✅ Scaler:      {f}")
                elif f.endswith('.json') and 'feature' in fl:
                    found['features'] = fp
                    logger.info(f"  ✅ Features:    {f}")
        return found

    @staticmethod
    def find_data(data_path: str = None) -> Dict[str, str]:
        found = {}
        search = [data_path, '.', './data'] if data_path else ['.', './data']
        logger.info("🔍 Searching for data files...")
        for sp in search:
            if not sp or not os.path.exists(sp):
                continue
            if os.path.isfile(sp) and sp.endswith('.csv'):
                found['ohlcv'] = sp
                logger.info(f"  ✅ OHLCV: {os.path.basename(sp)}")
            elif os.path.isdir(sp):
                for f in os.listdir(sp):
                    fp = os.path.join(sp, f)
                    fl = f.lower()
                    if f.endswith('.csv'):
                        if 'ohlcv' in fl or 'price' in fl:
                            found['ohlcv'] = fp
                            logger.info(f"  ✅ OHLCV: {f}")
                        elif 'fear' in fl or 'greed' in fl:
                            found['fear_greed'] = fp
                            logger.info(f"  ✅ Fear & Greed: {f}")
        return found

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
class FeatureEngineer:
    @staticmethod
    def _safe(s: pd.Series) -> pd.Series:
        return s.replace([np.inf, -np.inf], np.nan)

    @classmethod
    def calculate_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
        logger.info("  ⚙️  Calculating features...")

        # TREND
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
        for p in [14, 20]:
            shift = p // 2 + 1
            df[f'dpo_{p}']      = cls._safe(c - c.rolling(p).mean().shift(shift))
            df[f'dpo_{p}_norm'] = cls._safe(df[f'dpo_{p}'] / c)

        # MOMENTUM
        for p in [1, 3, 5, 10, 14, 20]:
            df[f'log_ret_{p}'] = cls._safe(np.log(c / c.shift(p)))
        for p in [5, 10, 20]:
            df[f'roc_{p}'] = cls._safe(c.pct_change(p))
        for p in [7, 14, 21]:
            delta = c.diff()
            gain  = delta.clip(lower=0).rolling(p).mean()
            loss  = (-delta.clip(upper=0)).rolling(p).mean()
            rs = cls._safe(gain / (loss + 1e-10))
            df[f'rsi_{p}']      = 100 - (100 / (1 + rs))
            df[f'rsi_{p}_norm'] = df[f'rsi_{p}'] / 100.0
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df['macd']        = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist']   = df['macd'] - df['macd_signal']
        df['macd_norm']   = cls._safe(df['macd'] / (c + 1e-10))
        for p in [14, 21]:
            lo = l.rolling(p).min(); hi = h.rolling(p).max()
            df[f'stoch_k_{p}'] = cls._safe(100 * (c - lo) / (hi - lo + 1e-10))
            df[f'stoch_d_{p}'] = df[f'stoch_k_{p}'].rolling(3).mean()
        df['williams_r_14'] = cls._safe(-100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-10))
        tp = (h + l + c) / 3
        df['cci_14'] = cls._safe((tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std() + 1e-10))
        df['cci_20'] = cls._safe((tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10))

        # VOLUME
        df['volume_sma_10']   = v.rolling(10).mean()
        df['volume_sma_20']   = v.rolling(20).mean()
        df['volume_ratio_10'] = cls._safe(v / (df['volume_sma_10'] + 1e-10))
        df['volume_ratio_20'] = cls._safe(v / (df['volume_sma_20'] + 1e-10))
        df['volume_log']      = cls._safe(np.log(v + 1))
        df['volume_change']   = v.pct_change()
        df['volume_change_5'] = v.pct_change(5)
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        df['obv']         = obv
        df['obv_sma_10']  = obv.rolling(10).mean()
        df['obv_ratio']   = cls._safe(obv / (df['obv_sma_10'] + 1e-10))
        df['vwap_20']      = cls._safe((tp * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10))
        df['price_vs_vwap'] = cls._safe(c / (df['vwap_20'] + 1e-10))
        mf_raw = tp * v
        pos_mf = mf_raw.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf_raw.where(tp < tp.shift(1), 0).rolling(14).sum()
        df['mfi_14'] = cls._safe(100 - (100 / (1 + pos_mf / (neg_mf + 1e-10))))

        # VOLATILITY
        ret = c.pct_change()
        for p in [10, 20, 30]:
            df[f'volatility_{p}'] = ret.rolling(p).std()
        df['volatility_ratio'] = cls._safe(df['volatility_10'] / (df['volatility_30'] + 1e-10))
        mid = c.rolling(20).mean(); std = c.rolling(20).std()
        df['bb_upper_20'] = mid + 2 * std
        df['bb_lower_20'] = mid - 2 * std
        df['bb_width_20'] = cls._safe((df['bb_upper_20'] - df['bb_lower_20']) / (mid + 1e-10))
        df['bb_pct_20']   = cls._safe((c - df['bb_lower_20']) / (df['bb_upper_20'] - df['bb_lower_20'] + 1e-10))
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        for p in [7, 14, 21]:
            df[f'atr_{p}']      = tr.rolling(p).mean()
            df[f'atr_{p}_norm'] = cls._safe(df[f'atr_{p}'] / (c + 1e-10))
        kc_mid = c.ewm(span=20, adjust=False).mean()
        df['keltner_width'] = cls._safe(4 * df['atr_14'] / (kc_mid + 1e-10))

        # Z-SCORES
        for p in [10, 20, 50]:
            rm = c.rolling(p).mean(); rs = c.rolling(p).std()
            df[f'close_zscore_{p}'] = cls._safe((c - rm) / (rs + 1e-10))
        for p in [10, 20]:
            vm = v.rolling(p).mean(); vs = v.rolling(p).std()
            df[f'volume_zscore_{p}'] = cls._safe((v - vm) / (vs + 1e-10))

        # PRICE STRUCTURE
        df['hl_ratio']     = cls._safe((h - l) / (c + 1e-10))
        df['oc_ratio']     = cls._safe((c - o) / (h - l + 1e-10))
        df['upper_shadow'] = cls._safe((h - c.combine(o, max)) / (h - l + 1e-10))
        df['lower_shadow'] = cls._safe((c.combine(o, min) - l) / (h - l + 1e-10))
        df['gap']          = cls._safe((o - c.shift(1)) / (c.shift(1) + 1e-10))

        # TIME
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            df['hour_sin']  = np.sin(2 * np.pi * ts.dt.hour / 24)
            df['hour_cos']  = np.cos(2 * np.pi * ts.dt.hour / 24)
            df['dow_sin']   = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
            df['dow_cos']   = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * ts.dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * ts.dt.month / 12)

        # ALIASES — short names training mein use hue to map karo
        aliases = {
            'dpo': 'dpo_14', 'dpo_norm': 'dpo_14_norm',
            'ema_20': 'ema_21', 'sma_20': 'sma_21',
            'atr': 'atr_14', 'atr_norm': 'atr_14_norm',
            'rsi': 'rsi_14', 'stoch_k': 'stoch_k_14', 'stoch_d': 'stoch_d_14',
            'cci': 'cci_14', 'bb_width': 'bb_width_20', 'bb_pct': 'bb_pct_20',
            'volatility': 'volatility_20', 'close_zscore': 'close_zscore_20',
            'volume_zscore': 'volume_zscore_20', 'log_ret_3': 'log_ret_3',
        }
        for alias, src in aliases.items():
            if alias not in df.columns and src in df.columns:
                df[alias] = df[src]

        logger.info(f"  ✅ Feature engineering done — {len(df.columns)} total columns")
        return df

# ============================================================================
# SIGNAL ENGINE — LSTM Boss + Ensemble Co-Decision Maker + Optional Helpers
# ============================================================================
class SignalEngine:
    """
    LSTM is the primary decision maker.
    Ensemble acts as co-decision maker (confirmation).
    PPO Actor & Critic are optional helpers that can be enabled/disabled via config.
    
    Signal logic:
      - LSTM provides direction: BUY / SELL / HOLD
      - Ensemble provides bullish probability
      - If LSTM says BUY: require ensemble bullish probability >= CONFIDENCE_THRESHOLD
      - If LSTM says SELL: require ensemble bullish probability <= (1 - CONFIDENCE_THRESHOLD)
      - If LSTM says HOLD: output HOLD
      - Optional PPO helpers can further gate the signal if enabled
    """

    CONFIDENCE_THRESHOLD = 0.55  # Ensemble confirmation threshold
    GATE_THRESHOLD = 0.35  # Legacy gate threshold (for PPO helpers if enabled)

    def __init__(self, models: Dict, config: Dict):
        self.models = models
        self.config = config
        # Check if PPO helpers are enabled (default: False)
        self.ppo_enabled = config.get('enable_ppo_helpers', False)
        if self.ppo_enabled:
            logger.info("  🟢 PPO Helpers (Actor/Critic) are ENABLED")
        else:
            logger.info("  🔴 PPO Helpers (Actor/Critic) are DISABLED (using only LSTM + Ensemble)")

    # ── LSTM (Boss) with FIXED signal mapping ─────────────────────────────────────────
    def _run_lstm(self, X: np.ndarray, window: int, n_bars: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (direction[n_bars], quality[n_bars]) arrays.
        
        FIX: Properly maps LSTM outputs to BUY(1)/SELL(0)/HOLD(2)
        The model outputs [SELL_prob, BUY_prob, HOLD_prob] (standard order)
        """
        direction = np.full(n_bars, SIGNAL_HOLD, dtype=int)
        quality   = np.full(n_bars, 0.5)

        if 'lstm' not in self.models:
            logger.warning("  ⚠️ LSTM not loaded — all signals will be HOLD")
            return direction, quality

        print(f"\n  🧠 LSTM (Boss) inference on {len(X)} sequences...")
        try:
            out = self.models['lstm'].predict(X, verbose=0)

            if isinstance(out, list):
                # Multi-output: [price, direction_probs, quality, exit_bar, pos_size, ...]
                if len(out) >= 2:
                    # out[1] shape: (batch, 3) -> [SELL_prob, BUY_prob, HOLD_prob] (standard)
                    probs = out[1]
                    if probs.shape[1] >= 3:
                        # Get class with highest probability
                        raw_dirs = np.argmax(probs[:, :3], axis=1)
                        # Map: if argmax=0 -> SELL(0), argmax=1 -> BUY(1), argmax=2 -> HOLD(2)
                        # This mapping is correct if model output order is [SELL, BUY, HOLD]
                        direction[window:] = raw_dirs
                        
                        # DEBUG: Print first few predictions
                        logger.debug(f"  LSTM direction sample (first 10 valid): {raw_dirs[:10]}")
                        unique, counts = np.unique(raw_dirs, return_counts=True)
                        logger.info(f"  LSTM direction distribution: {dict(zip(unique, counts))}")
                        
                    elif probs.shape[1] == 2:
                        # Binary output: [sell_prob, buy_prob]
                        direction[window:] = np.argmax(probs, axis=1)
                    elif probs.shape[1] == 1:
                        vals = probs.flatten()
                        direction[window:] = np.where(vals >= 0.5, SIGNAL_BUY, SIGNAL_SELL)
                    
                    if len(out) >= 3:
                        quality[window:] = np.clip(out[2].flatten(), 0, 1)
                logger.info(f"  ✅ LSTM multi-output parsed shape={[o.shape for o in out]}")

            elif isinstance(out, np.ndarray):
                if out.ndim == 2:
                    if out.shape[1] >= 3:
                        raw_dirs = np.argmax(out[:, :3], axis=1)
                        direction[window:] = raw_dirs
                        logger.info(f"  LSTM direction distribution: {np.unique(raw_dirs, return_counts=True)}")
                        quality[window:] = np.max(out[:, :3], axis=1)
                    elif out.shape[1] == 2:
                        direction[window:] = np.argmax(out, axis=1)
                        quality[window:] = np.max(out, axis=1)
                    elif out.shape[1] == 1:
                        vals = out.flatten()
                        direction[window:] = np.where(vals >= 0.5, SIGNAL_BUY, SIGNAL_SELL)
                        quality[window:] = np.where(vals >= 0.5, vals, 1 - vals)
                elif out.ndim == 1:
                    direction[window:] = np.where(out >= 0.5, SIGNAL_BUY, SIGNAL_SELL)
                    quality[window:] = np.where(out >= 0.5, out, 1 - out)
                logger.info(f"  ✅ LSTM ndarray parsed shape={out.shape}")
        except Exception as e:
            logger.error(f"  ❌ LSTM inference failed: {e}")

        return direction, quality

    # ── PPO Actor (Optional) ────────────────────────────────────────────────────────────
    def _run_ppo_actor(self, X: np.ndarray) -> np.ndarray:
        """Returns bullish probability array [0,1] for each sequence."""
        if not self.ppo_enabled or 'ppo_actor' not in self.models:
            return np.full(len(X), 0.5)
        print(f"\n  🎭 PPO Actor inference...")
        try:
            out = self.models['ppo_actor'].predict(X, verbose=0)
            if isinstance(out, np.ndarray):
                if out.ndim == 2 and out.shape[1] >= 2:
                    probs = out[:, 1]          # prob of BUY action
                elif out.ndim == 2 and out.shape[1] == 1:
                    probs = out.flatten()
                else:
                    probs = out.flatten()
                probs = np.clip(probs, 0, 1)
                logger.info(f"  ✅ PPO Actor done mean_bull={probs.mean():.3f}")
                return probs
        except Exception as e:
            logger.warning(f"  ⚠️ PPO Actor failed: {e}")
        return np.full(len(X), 0.5)

    # ── PPO Critic (Optional) ───────────────────────────────────────────────────────────
    def _run_ppo_critic(self, X: np.ndarray) -> np.ndarray:
        """Returns state-value array normalized to [0,1] as confidence."""
        if not self.ppo_enabled or 'ppo_critic' not in self.models:
            return np.full(len(X), 0.5)
        print(f"\n  🧑‍⚖️ PPO Critic inference...")
        try:
            out = self.models['ppo_critic'].predict(X, verbose=0)
            vals = out.flatten() if isinstance(out, np.ndarray) else np.array(out).flatten()
            # Normalize raw state values to [0,1] via sigmoid
            confidence = 1 / (1 + np.exp(-vals))
            logger.info(f"  ✅ PPO Critic done mean_conf={confidence.mean():.3f}")
            return confidence
        except Exception as e:
            logger.warning(f"  ⚠️ PPO Critic failed: {e}")
        return np.full(len(X), 0.5)

    # ── Ensemble (Co-Decision Maker) ─────────────────────────────────────────────────────
    def _run_ensemble(self, df: pd.DataFrame, features_for_ensemble: List[str]) -> np.ndarray:
        """Returns bullish probability per bar via ensemble model.
        
        FIX: Ensemble now properly extracts features that match its training.
        Uses the same features that were used during training (from scaler or feature list).
        """
        if 'ensemble' not in self.models:
            return np.full(len(df), 0.5)
        print(f"\n  🤝 Ensemble inference ({len(df)} bars)...")
        
        # Determine which features to use for ensemble
        available_features = [f for f in features_for_ensemble if f in df.columns]
        if not available_features:
            logger.warning("  ⚠️ No matching features for ensemble, using all numeric columns")
            available_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"  Ensemble using {len(available_features)} features")
        
        # Prepare data
        X_ens = df[available_features].fillna(0).values
        
        probs = np.full(len(df), 0.5)
        mdl = self.models['ensemble']
        
        try:
            # Try predict_proba first
            if hasattr(mdl, 'predict_proba'):
                raw = mdl.predict_proba(X_ens)
                if raw.ndim == 2 and raw.shape[1] >= 2:
                    probs = raw[:, 1]
                elif raw.ndim == 1:
                    probs = raw
                else:
                    probs = raw.flatten()
                logger.info(f"  ✅ Ensemble predict_proba done mean_bull={probs.mean():.3f}")
            elif hasattr(mdl, 'predict'):
                raw = mdl.predict(X_ens)
                probs = np.clip(raw.flatten(), 0, 1)
                logger.info(f"  ✅ Ensemble predict done mean_bull={probs.mean():.3f}")
            else:
                logger.warning(f"  ⚠️ Ensemble has no predict/predict_proba method")
        except Exception as e:
            logger.warning(f"  ⚠️ Ensemble inference failed: {e}")
        
        # Clip and ensure valid range
        probs = np.clip(probs, 0, 1)
        return probs

    # ── Helper Gate (Optional PPO helpers) ───────────────────────────────────────────────
    def _helper_score(self, actor_p: float, critic_c: float, ens_p: float, lstm_signal: int) -> float:
        """
        Weighted vote from helpers (if enabled).
        For BUY signal → helpers should show high bullish prob.
        For SELL signal → helpers should show low bullish prob.
        For HOLD → no gate needed, pass through.
        """
        if not self.ppo_enabled:
            # If PPO disabled, helper score is just ensemble probability
            if lstm_signal == SIGNAL_HOLD:
                return 1.0
            elif lstm_signal == SIGNAL_BUY:
                return ens_p
            else:  # SELL
                return 1.0 - ens_p
        
        # PPO enabled: weighted vote from all three helpers
        if lstm_signal == SIGNAL_HOLD:
            return 1.0
        
        # Weighted: Ensemble 50%, Actor 25%, Critic 25% (when PPO enabled)
        raw_score = 0.50 * ens_p + 0.25 * actor_p + 0.25 * critic_c
        
        if lstm_signal == SIGNAL_SELL:
            raw_score = 1.0 - raw_score
        
        return float(np.clip(raw_score, 0, 1))

    # ── Main Generate with FIXED logic ───────────────────────────────────────────────────
    def generate(self, df: pd.DataFrame, X: np.ndarray, window: int, 
                 ensemble_features: List[str]) -> pd.DataFrame:
        n = len(df)

        # 1. LSTM Boss
        lstm_dir, lstm_qual = self._run_lstm(X, window, n)

        # 2. Ensemble (Co-Decision Maker) - always used
        ens_probs = self._run_ensemble(df, ensemble_features)
        
        # Pad ensemble to full length
        ens_full = np.full(n, 0.5)
        ens_full[window:] = ens_probs[window:] if len(ens_probs) >= n else ens_probs

        # 3. PPO Helpers (optional)
        actor_full = np.full(n, 0.5)
        critic_full = np.full(n, 0.5)
        
        if self.ppo_enabled:
            actor_probs = self._run_ppo_actor(X)
            critic_conf = self._run_ppo_critic(X)
            actor_full[window:] = actor_probs if len(actor_probs) >= n else actor_probs
            critic_full[window:] = critic_conf if len(critic_conf) >= n else critic_conf

        # 4. Apply signal logic with Ensemble confirmation
        final_signal = np.full(n, SIGNAL_HOLD, dtype=int)
        final_quality = np.full(n, 0.0)
        helper_scores = np.zeros(n)
        
        for i in range(n):
            lstm_sig = int(lstm_dir[i])
            ens_prob = float(ens_full[i])
            
            # Determine helper score (respects ppo_enabled flag)
            if self.ppo_enabled:
                hs = self._helper_score(
                    actor_p=float(actor_full[i]),
                    critic_c=float(critic_full[i]),
                    ens_p=ens_prob,
                    lstm_signal=lstm_sig
                )
            else:
                # PPO disabled: helper score is ensemble confirmation
                if lstm_sig == SIGNAL_HOLD:
                    hs = 1.0
                elif lstm_sig == SIGNAL_BUY:
                    hs = ens_prob
                else:  # SELL
                    hs = 1.0 - ens_prob
            
            helper_scores[i] = hs
            
            # Apply ensemble confirmation gate
            if lstm_sig == SIGNAL_BUY:
                # Require ensemble bullish probability >= CONFIDENCE_THRESHOLD
                if ens_prob >= self.CONFIDENCE_THRESHOLD:
                    final_signal[i] = SIGNAL_BUY
                    final_quality[i] = lstm_qual[i] * ens_prob
                else:
                    final_signal[i] = SIGNAL_HOLD
                    final_quality[i] = lstm_qual[i] * ens_prob * 0.5
                    
            elif lstm_sig == SIGNAL_SELL:
                # Require ensemble bullish probability <= (1 - CONFIDENCE_THRESHOLD)
                if ens_prob <= (1.0 - self.CONFIDENCE_THRESHOLD):
                    final_signal[i] = SIGNAL_SELL
                    final_quality[i] = lstm_qual[i] * (1.0 - ens_prob)
                else:
                    final_signal[i] = SIGNAL_HOLD
                    final_quality[i] = lstm_qual[i] * (1.0 - ens_prob) * 0.5
                    
            else:  # HOLD
                final_signal[i] = SIGNAL_HOLD
                final_quality[i] = lstm_qual[i]

        # Add columns to dataframe
        df = df.copy()
        df['lstm_signal'] = lstm_dir
        df['lstm_quality'] = lstm_qual
        df['ensemble_prob'] = ens_full
        if self.ppo_enabled:
            df['ppo_actor_prob'] = actor_full
            df['ppo_critic_conf'] = critic_full
        df['helper_score'] = helper_scores
        df['final_signal'] = final_signal
        df['final_quality'] = final_quality

        buys = (final_signal == SIGNAL_BUY).sum()
        sells = (final_signal == SIGNAL_SELL).sum()
        holds = (final_signal == SIGNAL_HOLD).sum()
        blocked_buys = ((lstm_dir == SIGNAL_BUY) & (final_signal == SIGNAL_HOLD) & (ens_full < self.CONFIDENCE_THRESHOLD)).sum()
        blocked_sells = ((lstm_dir == SIGNAL_SELL) & (final_signal == SIGNAL_HOLD) & (ens_full > (1.0 - self.CONFIDENCE_THRESHOLD))).sum()

        logger.info(f"  ✅ Signal summary — BUY:{buys}  SELL:{sells}  HOLD:{holds}")
        logger.info(f"  📊 Ensemble blocked BUYs:{blocked_buys}  blocked SELLs:{blocked_sells}")
        return df

# ============================================================================
# RESULTS VISUALIZER
# ============================================================================
class ResultsVisualizer:
    @staticmethod
    def display_results(result: Dict, elapsed_time: float):
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " BACKTEST RESULTS SUMMARY".center(78) + "║")
        print("╠" + "═" * 78 + "╣")
        metrics = [
            ("Total Return",  f"{result.get('total_return', 0):.2f}%",
             "🟢" if result.get('total_return', 0) > 0 else "🔴"),
            ("Sharpe Ratio",  f"{result.get('sharpe', 0):.4f}",
             "🟢" if result.get('sharpe', 0) > 1 else "🟡" if result.get('sharpe', 0) > 0.5 else "🔴"),
            ("Sortino Ratio", f"{result.get('sortino', 0):.4f}",
             "🟢" if result.get('sortino', 0) > 1 else "🟡" if result.get('sortino', 0) > 0.5 else "🔴"),
            ("Max Drawdown",  f"{result.get('max_drawdown', 0)*100:.2f}%",
             "🟢" if result.get('max_drawdown', 0) > -0.1 else "🟡" if result.get('max_drawdown', 0) > -0.2 else "🔴"),
            ("Win Rate",      f"{result.get('win_rate', 0)*100:.1f}%",
             "🟢" if result.get('win_rate', 0) > 0.55 else "🟡" if result.get('win_rate', 0) > 0.45 else "🔴"),
            ("Profit Factor", f"{result.get('profit_factor', 0):.2f}",
             "🟢" if result.get('profit_factor', 0) > 1.5 else "🟡" if result.get('profit_factor', 0) > 1 else "🔴"),
            ("Total Trades",  f"{result.get('total_trades', 0)}", "⚪"),
            ("Avg Trade PnL", f"{result.get('avg_pnl', 0)*100:.2f}%", "⚪"),
            ("Final Capital", f"${result.get('final_capital', 0):,.2f}",
             "🟢" if result.get('final_capital', 0) > 10000 else "🟡"),
        ]
        for name, val, icon in metrics:
            print(f"║  {icon} {name:<22}: {val:>18}  {icon} ║")
        print("╠" + "═" * 78 + "╣")
        sharpe = result.get('sharpe', 0)
        if sharpe > 1.5:   verdict = "EXCELLENT — Ready for live trading! 🚀"
        elif sharpe > 1.0: verdict = "GOOD — Can proceed with caution ✅"
        elif sharpe > 0.5: verdict = "AVERAGE — Needs optimization ⚠️"
        elif result.get('total_return', 0) > 0: verdict = "POOR — Significant improvement needed 🔴"
        else:              verdict = "UNPROFITABLE — Do NOT trade live ❌"
        print(f"║  🎯 {verdict:<72} ║")
        print("╠" + "═" * 78 + "╣")
        pad = 78 - 35 - len(f"{elapsed_time:.2f}")
        print("║" + f"  ⏱️  Backtest completed in {elapsed_time:.2f} seconds" + " " * pad + "║")
        print("╚" + "═" * 78 + "╝")

    @staticmethod
    def save_results(result: Dict):
        try:
            out = {k: v for k, v in result.items() if k not in ('portfolio', 'trades')}
            out['portfolio'] = result.get('portfolio', [])
            out['trades']    = result.get('trades', [])
            with open('backtest_results.json', 'w') as f:
                json.dump({'timestamp': datetime.now(timezone.utc).isoformat(), 'results': out},
                          f, indent=2, default=str)
            logger.info("📁 Results saved → backtest_results.json")
        except Exception as e:
            logger.warning(f"⚠️ Could not save results: {e}")

# ============================================================================
# BACKTEST RUNNER
# ============================================================================
class BacktestRunner:

    def __init__(self, config_path: str = None):
        self.config     = self._load_config(config_path)
        self.models     = {}
        self.data       = {}
        self.start_time = None

    # ── Config with PPO enable/disable switch ───────────────────────────────────────────
    def _load_config(self, path: str = None) -> dict:
        defaults = {
            'symbol': 'BTC/USDT', 'timeframe': '1h',
            'fee_rate': 0.001, 'slippage': 0.0005,
            'initial_capital': 10000, 'window': 60,
            'max_position_pct': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'enable_ppo_helpers': False,  # NEW: Disable PPO by default
            'ensemble_confidence_threshold': 0.55,  # NEW: Configurable threshold
        }
        for p in [path, 'config.json']:
            if p and os.path.exists(p):
                try:
                    cfg = json.load(open(p))
                    logger.info(f"✅ Config loaded: {p}")
                    if 'enable_ppo_helpers' in cfg:
                        logger.info(f"   PPO Helpers enabled: {cfg['enable_ppo_helpers']}")
                    merged = {**defaults, **cfg}
                    return merged
                except Exception as e:
                    logger.warning(f"Config load failed: {e}")
        logger.warning("⚠️ No config.json — using defaults (PPO helpers DISABLED)")
        return defaults

    # ── Timestamp parser ─────────────────────────────────────────────────────
    def _parse_timestamp(self, series: pd.Series) -> pd.Series:
        for kwargs in [
            {'utc': True, 'infer_datetime_format': True},
            {'unit': 'ms', 'utc': True},
            {'unit': 's',  'utc': True},
        ]:
            try:
                return pd.to_datetime(series, **kwargs)
            except Exception:
                pass
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize('UTC')
        else:
            parsed = parsed.dt.tz_convert('UTC')
        return parsed

    # ── Model loading ────────────────────────────────────────────────────────
    def _discover_and_load_models(self, models_dir: str = None) -> bool:
        logger.info("=" * 60)
        logger.info("📦 MODEL LOADING PHASE")
        logger.info("=" * 60)

        found = FileDiscovery.find_models(models_dir or '.')
        if not found:
            logger.error("❌ No models found!")
            return False

        # LSTM (Boss) - mandatory
        if 'lstm' in found:
            try:
                from tensorflow.keras.models import load_model
                self.models['lstm'] = load_model(found['lstm'])
                logger.info(f"  ✅ LSTM Boss loaded")
            except Exception as e:
                logger.error(f"  ❌ LSTM load failed: {e}")
                return False
        else:
            logger.error("❌ LSTM model not found — mandatory")
            return False

        # PPO Actor (optional, load only if exists)
        if 'ppo_actor' in found:
            try:
                from tensorflow.keras.models import load_model
                self.models['ppo_actor'] = load_model(found['ppo_actor'])
                logger.info(f"  ✅ PPO Actor loaded (optional)")
            except Exception as e:
                logger.warning(f"  ⚠️ PPO Actor load failed: {e}")

        # PPO Critic (optional)
        if 'ppo_critic' in found:
            try:
                from tensorflow.keras.models import load_model
                self.models['ppo_critic'] = load_model(found['ppo_critic'])
                logger.info(f"  ✅ PPO Critic loaded (optional)")
            except Exception as e:
                logger.warning(f"  ⚠️ PPO Critic load failed: {e}")

        # Ensemble - mandatory
        if 'ensemble' in found:
            try:
                import joblib
                self.models['ensemble'] = joblib.load(found['ensemble'])
                logger.info(f"  ✅ Ensemble loaded")
            except Exception as e:
                logger.warning(f"  ⚠️ Ensemble load failed: {e}")

        # Scaler
        if 'scaler' in found:
            try:
                import joblib
                self.models['scaler'] = joblib.load(found['scaler'])
                logger.info(f"  ✅ Scaler loaded")
                # Extract feature names from scaler
                if hasattr(self.models['scaler'], 'feature_names_in_'):
                    self.models['scaler_features'] = list(self.models['scaler'].feature_names_in_)
                    logger.info(f"  📊 Scaler expects {len(self.models['scaler_features'])} features")
            except Exception as e:
                logger.warning(f"  ⚠️ Scaler load failed: {e}")

        if 'features' in found:
            self.models['features'] = found['features']

        # Load feature list from JSON
        self.models['ensemble_features'] = self._get_ensemble_features()
        
        loaded = ['lstm', 'ensemble'] + [k for k in ('ppo_actor', 'ppo_critic') if k in self.models]
        logger.info(f"  📊 Loaded models: {loaded}")
        return True
    
    def _get_ensemble_features(self) -> List[str]:
        """Extract features for ensemble from scaler or feature JSON."""
        features = []
        
        # First priority: features from scaler (what model was trained on)
        if 'scaler' in self.models and hasattr(self.models['scaler'], 'feature_names_in_'):
            features = list(self.models['scaler'].feature_names_in_)
            logger.info(f"  ✅ Using features from scaler: {len(features)} features")
            return features
        
        # Second priority: features from JSON file
        if 'features' in self.models:
            try:
                with open(self.models['features'], 'r') as f:
                    fd = json.load(f)
                if isinstance(fd, dict):
                    if 'selected_features' in fd:
                        features = fd['selected_features']
                    elif 'features' in fd:
                        features = fd['features']
                elif isinstance(fd, list):
                    features = fd
                if features:
                    logger.info(f"  ✅ Using features from JSON: {len(features)} features")
                    return features
            except Exception as e:
                logger.warning(f"  ⚠️ Feature JSON load failed: {e}")
        
        # Fallback: minimal feature set
        features = ['open', 'high', 'low', 'close', 'volume']
        logger.warning(f"  ⚠️ Using fallback features: {features}")
        return features

    # ── Data loading ─────────────────────────────────────────────────────────
    def _discover_and_load_data(self, data_path: str = None) -> bool:
        logger.info("=" * 60)
        logger.info("📊 DATA LOADING PHASE")
        logger.info("=" * 60)

        found = FileDiscovery.find_data(data_path)
        if 'ohlcv' not in found:
            logger.error("❌ No OHLCV data found!")
            return False

        try:
            df = pd.read_csv(found['ohlcv'])
            df.columns = [c.strip().lower() for c in df.columns]
            if 'timestamp' in df.columns:
                df['timestamp'] = self._parse_timestamp(df['timestamp'])
                logger.info(f"  ✅ Timestamp dtype: {df['timestamp'].dtype}")
            required = ['open', 'high', 'low', 'close', 'volume']
            missing  = [c for c in required if c not in df.columns]
            if missing:
                logger.error(f"  ❌ Missing OHLCV columns: {missing}")
                return False
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['close'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.data['df'] = df
            logger.info(f"  ✅ Loaded {len(df)} bars")
        except Exception as e:
            logger.error(f"  ❌ OHLCV load failed: {e}")
            return False

        if 'fear_greed' in found:
            try:
                fg_raw = pd.read_csv(found['fear_greed'], header=None, nrows=1)
                first  = str(fg_raw.iloc[0, 0])
                try:
                    pd.to_datetime(first)
                    fg_df = pd.read_csv(found['fear_greed'], header=None)
                    fg_df.columns = [f'col_{i}' for i in range(len(fg_df.columns))]
                    logger.info("  ℹ️ F&G CSV — no header, inferring columns")
                except Exception:
                    fg_df = pd.read_csv(found['fear_greed'])
                    fg_df.columns = [c.strip().lower() for c in fg_df.columns]

                ts_col  = next((c for c in fg_df.columns if any(k in str(c).lower() for k in ('time','date','col_0'))), None)
                val_col = next((c for c in fg_df.columns if any(k in str(c).lower() for k in ('fear','greed','value','col_1'))), None)

                if ts_col is None or val_col is None:
                    raise ValueError(f"Cannot identify columns in F&G CSV: {fg_df.columns.tolist()}")

                fg_df[ts_col]       = self._parse_timestamp(fg_df[ts_col])
                fg_df['_date']      = fg_df[ts_col].dt.normalize()
                fg_df['fear_greed'] = pd.to_numeric(fg_df[val_col], errors='coerce')
                fg_df = fg_df.dropna(subset=['fear_greed']).drop_duplicates(subset=['_date'])

                self.data['df']['_date'] = self.data['df']['timestamp'].dt.normalize()
                self.data['df'] = self.data['df'].merge(fg_df[['_date', 'fear_greed']], on='_date', how='left')
                self.data['df']['fear_greed'] = self.data['df']['fear_greed'].ffill().bfill().fillna(50)
                self.data['df'].drop(columns=['_date'], inplace=True)
                logger.info("  ✅ Fear & Greed merged")
            except Exception as e:
                logger.warning(f"  ⚠️ F&G merge failed: {e} — defaulting to 50")
                self.data['df']['fear_greed'] = 50

        return True

    # ── Feature preparation with AUTO feature alignment ───────────────────────────────────
    def _prepare_features(self) -> bool:
        logger.info("=" * 60)
        logger.info("🔧 FEATURE PREPARATION")
        logger.info("=" * 60)

        # Calculate all features (115 columns)
        self.data['df'] = FeatureEngineer.calculate_all(self.data['df'])
        df = self.data['df']

        # Get features that scaler expects (what model was trained on)
        scaler_features = self.models.get('scaler_features', [])
        ensemble_features = self.models.get('ensemble_features', [])
        
        # Determine which features to use for LSTM/PPO (scaler features)
        if scaler_features:
            # Use exactly the features scaler was trained on
            available_for_scaler = [f for f in scaler_features if f in df.columns]
            missing_from_scaler = [f for f in scaler_features if f not in df.columns]
            
            if missing_from_scaler:
                logger.warning(f"  ⚠️ Missing {len(missing_from_scaler)} features expected by scaler: {missing_from_scaler[:10]}")
                # Add missing columns with zeros
                for f in missing_from_scaler:
                    df[f] = 0
                available_for_scaler = scaler_features
            
            logger.info(f"  ✅ Using {len(available_for_scaler)} features for scaling (as per scaler training)")
            data_for_scaler = df[available_for_scaler].copy().ffill().bfill().fillna(0)
            
            # Apply scaling
            if 'scaler' in self.models:
                try:
                    # Ensure column order matches scaler's training
                    scaler_cols = list(self.models['scaler'].feature_names_in_)
                    for col in scaler_cols:
                        if col not in data_for_scaler.columns:
                            data_for_scaler[col] = 0
                    data_for_scaler = data_for_scaler[scaler_cols]
                    scaled_data = self.models['scaler'].transform(data_for_scaler)
                    logger.info("  ✅ Data scaled successfully")
                except Exception as e:
                    logger.warning(f"  ⚠️ Scaling failed: {e}")
                    scaled_data = data_for_scaler.values
            else:
                scaled_data = data_for_scaler.values
        else:
            # No scaler available, use all numeric columns
            logger.warning("  ⚠️ No scaler features available, using all numeric columns")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            data_for_scaler = df[numeric_cols].copy().ffill().bfill().fillna(0)
            scaled_data = data_for_scaler.values
            available_for_scaler = numeric_cols
            logger.info(f"  ✅ Using {len(available_for_scaler)} numeric columns")

        # Create sequences for LSTM/PPO
        window = self.config.get('window', 60)
        if len(scaled_data) < window:
            logger.error(f"  ❌ Not enough data: {len(scaled_data)} < {window}")
            return False

        X = np.array([scaled_data[i-window:i] for i in range(window, len(scaled_data))])
        logger.info(f"  ✅ Sequences shape: {X.shape}")
        logger.info(f"  ✅ Features per timestep: {X.shape[2]}")

        self.data['X'] = X
        self.data['window'] = window
        self.data['scaler_features'] = available_for_scaler
        self.data['ensemble_features'] = ensemble_features
        
        return True

    # ── Signal generation ────────────────────────────────────────────────────
    def _generate_signals(self) -> bool:
        logger.info("=" * 60)
        logger.info("🤖 SIGNAL GENERATION — LSTM Boss + Ensemble Co-Decision")
        logger.info("=" * 60)

        # Pass confidence threshold from config
        SignalEngine.CONFIDENCE_THRESHOLD = self.config.get('ensemble_confidence_threshold', 0.55)
        logger.info(f"  🎯 Ensemble confidence threshold: {SignalEngine.CONFIDENCE_THRESHOLD}")
        
        engine = SignalEngine(self.models, self.config)
        self.data['df'] = engine.generate(
            df=self.data['df'],
            X=self.data['X'],
            window=self.data['window'],
            ensemble_features=self.data.get('ensemble_features', [])
        )
        return True

    # ── Backtest loop ────────────────────────────────────────────────────────
    def _run_backtest(self) -> Dict:
        logger.info("=" * 60)
        logger.info("💰 RUNNING BACKTEST")
        logger.info("=" * 60)

        df              = self.data['df']
        initial_capital = self.config.get('initial_capital', 10000)
        fee             = self.config.get('fee_rate', 0.001)
        slippage        = self.config.get('slippage', 0.0005)
        sl_pct          = self.config.get('stop_loss_pct', 0.05)
        tp_pct          = self.config.get('take_profit_pct', 0.10)

        capital     = initial_capital
        position    = 0.0
        entry_price = 0.0
        stop_loss   = 0.0
        take_profit = 0.0
        trades      = []
        portfolio   = [capital]

        print(f"\n  Simulating {len(df)} bars  SL={sl_pct*100:.1f}%  TP={tp_pct*100:.1f}%")

        for i in tqdm(range(len(df)), desc="  Backtest", leave=False):
            price   = float(df['close'].iloc[i])
            if price <= 0 or np.isnan(price):
                portfolio.append(portfolio[-1])
                continue

            signal  = int(df['final_signal'].iloc[i])
            quality = float(df['final_quality'].iloc[i])
            h_score = float(df['helper_score'].iloc[i])

            # ── Stop Loss / Take Profit check ─────────────────────────────
            if position > 0:
                if price <= stop_loss:
                    sell_p  = stop_loss * (1 - slippage)
                    capital = position * sell_p * (1 - fee)
                    pnl     = (sell_p - entry_price) / entry_price
                    trades.append({'type': 'sell', 'price': sell_p, 'pnl': pnl, 'bar': i, 'reason': 'stop_loss'})
                    position = 0.0
                    portfolio.append(capital)
                    continue
                elif price >= take_profit:
                    sell_p  = take_profit * (1 - slippage)
                    capital = position * sell_p * (1 - fee)
                    pnl     = (sell_p - entry_price) / entry_price
                    trades.append({'type': 'sell', 'price': sell_p, 'pnl': pnl, 'bar': i, 'reason': 'take_profit'})
                    position = 0.0
                    portfolio.append(capital)
                    continue

            # ── Entry ─────────────────────────────────────────────────────
            if signal == SIGNAL_BUY and position == 0 and quality >= 0.5:
                buy_p    = price * (1 + slippage)
                pos_frac = min(quality, self.config.get('max_position_pct', 1.0))
                position = (capital * pos_frac) / buy_p * (1 - fee)
                capital  = capital * (1 - pos_frac)
                entry_price = buy_p
                stop_loss   = entry_price * (1 - sl_pct)
                take_profit = entry_price * (1 + tp_pct)
                trades.append({
                    'type': 'buy', 'price': buy_p, 'bar': i,
                    'quality': quality, 'helper_score': h_score,
                    'stop_loss': stop_loss, 'take_profit': take_profit
                })

            # ── Exit (SELL signal) ────────────────────────────────────────
            elif signal == SIGNAL_SELL and position > 0:
                sell_p  = price * (1 - slippage)
                capital = capital + position * sell_p * (1 - fee)
                pnl     = (sell_p - entry_price) / entry_price
                trades.append({'type': 'sell', 'price': sell_p, 'pnl': pnl, 'bar': i, 'reason': 'signal'})
                position = 0.0

            portfolio.append(capital + position * price)

        # Force close open position
        if position > 0:
            last_p  = float(df['close'].iloc[-1])
            sell_p  = last_p * (1 - slippage)
            capital = capital + position * sell_p * (1 - fee)
            pnl     = (sell_p - entry_price) / entry_price
            trades.append({'type': 'sell', 'price': sell_p, 'pnl': pnl,
                           'bar': len(df)-1, 'reason': 'forced_close'})
            portfolio[-1] = capital

        # ── Metrics ───────────────────────────────────────────────────────
        final_value  = portfolio[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100

        rets = np.diff(portfolio) / (np.array(portfolio[:-1]) + 1e-10)
        sharpe  = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(365 * 24)) if len(rets) else 0
        neg     = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(neg) + 1e-10) * np.sqrt(365 * 24)) if len(neg) else sharpe

        peak        = np.maximum.accumulate(portfolio)
        drawdown    = (np.array(portfolio) - peak) / (peak + 1e-10)
        max_dd      = float(drawdown.min())

        sell_trades = [t for t in trades if t['type'] == 'sell']
        pnls        = [t.get('pnl', 0) for t in sell_trades]
        wins        = [p for p in pnls if p > 0]
        losses      = [p for p in pnls if p <= 0]
        win_rate    = len(wins) / len(pnls) if pnls else 0
        avg_pnl     = float(np.mean(pnls)) if pnls else 0
        gross_win   = sum(wins)
        gross_loss  = abs(sum(losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')

        reasons = {}
        for t in sell_trades:
            r = t.get('reason', 'signal')
            reasons[r] = reasons.get(r, 0) + 1

        logger.info(f"  📊 Exit reasons: {reasons}")

        return {
            'total_return':  total_return,
            'sharpe':        sharpe,
            'sortino':       sortino,
            'max_drawdown':  max_dd,
            'win_rate':      win_rate,
            'profit_factor': profit_factor,
            'avg_pnl':       avg_pnl,
            'total_trades':  len(sell_trades),
            'final_capital': final_value,
            'exit_reasons':  reasons,
            'portfolio':     portfolio,
            'trades':        trades,
        }

    # ── Main run ─────────────────────────────────────────────────────────────
    def run(self, models_dir: str = None, data_path: str = None) -> Dict[str, Any]:
        self.start_time = time.time()

        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " PROFESSIONAL BACKTEST ENGINE v3.1".center(78) + "║")
        print("║" + " LSTM Boss + Ensemble Co-Decision + Optional PPO Helpers".center(78) + "║")
        print("╚" + "═" * 78 + "╝")

        steps = [
            (self._discover_and_load_models, (models_dir,), "No models loaded"),
            (self._discover_and_load_data,   (data_path,),  "No data loaded"),
            (self._prepare_features,         (),            "Feature preparation failed"),
            (self._generate_signals,         (),            "Signal generation failed"),
        ]
        for fn, args, err_msg in steps:
            if not fn(*args):
                logger.error(f"❌ Aborted: {err_msg}")
                return {'error': err_msg}

        results = self._run_backtest()
        elapsed = time.time() - self.start_time
        ResultsVisualizer.display_results(results, elapsed)
        ResultsVisualizer.save_results(results)
        gc.collect()
        return results

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Backtest Engine v3.1 — LSTM + Ensemble with Optional PPO')
    parser.add_argument('--models',  default=None, help='Models directory')
    parser.add_argument('--data',    default=None, help='Data file or directory')
    parser.add_argument('--config',  default=None, help='Config JSON path')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    runner = BacktestRunner(config_path=args.config)
    result = runner.run(models_dir=args.models, data_path=args.data)
    return 0 if 'error' not in result else 1

if __name__ == '__main__':
    exit(main())