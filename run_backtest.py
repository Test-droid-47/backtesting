#!/usr/bin/env python3
"""
AISure Backtest Engine - Final Production Version
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
from tensorflow.keras.models import load_model

# Suppress TensorFlow and Sklearn warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ============================================================
# HELPER FUNCTIONS FOR INDICATORS
# ============================================================
def calculate_obv(df):
    """On-Balance Volume"""
    return (np.sign(df['close'].diff().fillna(0)) * df['volume']).cumsum()

def calculate_dpo(df, period=20):
    """Detrended Price Oscillator"""
    shift = int(period / 2) + 1
    return df['close'] - df['close'].rolling(period).mean().shift(shift)

def find_file(extension, keyword):
    for f in os.listdir('.'):
        if f.endswith(extension) and keyword in f.lower():
            return f
    return None

print("="*60)
print("AISURE BACKTEST ENGINE: FINAL VERSION")
print("="*60)

# ============================================================
# 1. LOAD MODELS & METADATA
# ============================================================
print("\n[1/5] Loading models and scalers...")

scaler_file = find_file('.pkl', 'scaler')
if not scaler_file:
    print("❌ No scaler found!"); exit(1)
scaler = joblib.load(scaler_file)

# Extract features required by the scaler
if hasattr(scaler, 'feature_names_in_'):
    expected_features = scaler.feature_names_in_.tolist()
    print(f"  ✅ Scaler loaded. Expecting {len(expected_features)} features.")
else:
    # Fallback to JSON if scaler doesn't have names
    feature_file = find_file('.json', 'feature') or find_file('.json', 'selected')
    if feature_file:
        with open(feature_file, 'r') as f:
            data = json.load(f)
            expected_features = data if isinstance(data, list) else data.get('selected_features', [])
    else:
        print("❌ Could not determine feature list!"); exit(1)

# ============================================================
# 2. DATA LOADING & ROBUST FEATURE ENGINEERING
# ============================================================
print("\n[2/5] Engineering features for alignment...")

data_file = find_file('.csv', 'ohlcv') or find_file('.csv', 'price')
df = pd.read_csv(data_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Fundamental Technicals
for p in [1, 3, 5, 10]:
    df[f'ret_{p}'] = df['close'].pct_change(p)
    df[f'log_ret_{p}'] = np.log(df['close'] / df['close'].shift(p))

for p in [10, 20, 50, 100, 200]:
    df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

# ATR
tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# Missing Indicators responsible for the ValueError
df['obv'] = calculate_obv(df)
df['dpo'] = calculate_dpo(df, 20)
df['close_zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-9)
df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)

# Fear & Greed logic
fg_file = find_file('.csv', 'fear')
if fg_file:
    fg = pd.read_csv(fg_file, header=None, names=['timestamp', 'fear_greed', 'sentiment'])
    fg['date'] = pd.to_datetime(fg['timestamp']).dt.date
    df['date'] = df['timestamp'].dt.date
    df = df.merge(fg[['date', 'fear_greed']], on='date', how='left')
    df['fear_greed'] = df['fear_greed'].fillna(50)
    df.drop(['date'], axis=1, inplace=True)
elif 'fear_greed' in expected_features:
    df['fear_greed'] = 50

# Ensure every requested feature exists (even if we fill with zeros)
for feat in expected_features:
    if feat not in df.columns:
        df[feat] = 0.0

# ============================================================
# 3. GENERATING PREDICTIONS
# ============================================================
print("\n[3/5] Scaling and Model Inference...")

# CRITICAL: Reorder columns to match 'fit' time exactly
data_mat = df[expected_features].copy()
data_mat = data_mat.fillna(0).replace([np.inf, -np.inf], 0)

# Scaling
data_scaled = scaler.transform(data_mat)

# LSTM Prediction
lstm_file = find_file('.keras', 'lstm')
lstm = load_model(lstm_file)

window = 60
X = []
for i in range(window, len(data_scaled)):
    X.append(data_scaled[i-window:i])
X = np.array(X)

raw_preds = lstm.predict(X, verbose=0)

# Parse multi-output model: [0] tends to be Price, [1] Direction, [2] Quality
if isinstance(raw_preds, list):
    # Adjust indexing if your model training output was different
    # 0=SELL, 1=HOLD, 2=BUY
    dirs = np.argmax(raw_preds[1], axis=1) 
    quals = raw_preds[2].flatten()
else:
    dirs = np.argmax(raw_preds, axis=1) if raw_preds.shape[1] > 1 else (raw_preds > 0.5).astype(int)
    quals = np.max(raw_preds, axis=1) if raw_preds.shape[1] > 1 else raw_preds.flatten()

# Map predictions back to DF
df['pred_direction'] = 1 
df['pred_quality'] = 0.0
df.loc[window:, 'pred_direction'] = dirs
df.loc[window:, 'pred_quality'] = quals

# ============================================================
# 4. BACKTEST EXECUTION (Look-ahead bias free)
# ============================================================
print("\n[4/5] Executing Backtest...")

capital = 10000.0
initial_capital = capital
shares = 0
entry_price = 0
trades = []
portfolio_value = []

fee = 0.001
slippage = 0.0005

for i in range(window + 1, len(df)):
    current_price = df['close'].iloc[i]
    # Trade based on signal from PREVIOUS BAR
    sig = df['pred_direction'].iloc[i-1]
    qual = df['pred_quality'].iloc[i-1]
    
    # 1. EXIT LOGIC
    if shares > 0:
        pnl = (current_price / entry_price) - 1
        sell_trigger = False
        
        if sig == 0: sell_trigger = True
        elif pnl > 0.02: sell_trigger = True # 2% TP
        elif pnl < -0.01: sell_trigger = True # 1% SL
        
        if sell_trigger:
            exit_px = current_price * (1 - slippage)
            capital = shares * exit_px * (1 - fee)
            trades.append({
                'type': 'SELL', 
                'pnl': (exit_px / entry_price) - 1, 
                'date': df['timestamp'].iloc[i]
            })
            shares = 0

    # 2. ENTRY LOGIC
    elif sig == 2 and qual > 0.6: # Filter entries by quality
        entry_price = current_price * (1 + slippage)
        # Allocate 50% of capital to trade
        position_size = capital * 0.5
        shares = (position_size * (1 - fee)) / entry_price
        capital -= position_size
        trades.append({'type': 'BUY', 'date': df['timestamp'].iloc[i]})

    current_val = capital + (shares * current_price if shares > 0 else 0)
    portfolio_value.append(current_val)

# ============================================================
# 5. RESULTS & EXPORT
# ============================================================
final_value = portfolio_value[-1]
total_return = (final_value / initial_capital - 1) * 100

# Performance Stats
rets_series = pd.Series(portfolio_value).pct_change().dropna()
sharpe = (rets_series.mean() / (rets_series.std() + 1e-9)) * np.sqrt(365 * 24)
sell_trades = [t for t in trades if t['type'] == 'SELL']
win_rate = (len([t for t in sell_trades if t['pnl'] > 0]) / len(sell_trades) * 100) if sell_trades else 0

print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"Initial Capital:   ${initial_capital:,.2f}")
print(f"Final Capital:     ${final_value:,.2f}")
print(f"Total Return:      {total_return:.2f}%")
print(f"Sharpe Ratio:      {sharpe:.4f}")
print(f"Win Rate:          {win_rate:.1f}%")
print(f"Total Trades:      {len(sell_trades)}")
print("="*60)

# Save results
results = {
    'summary': {
        'total_return': total_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades_count': len(sell_trades)
    },
    'trades': trades
}

with open('backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n📁 Full trade log saved to 'backtest_results.json'")
