import warnings
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import yfinance as yf
import math
import requests
import re
import time
from io import StringIO, BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 0. PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="RSI Scans & Backtester", layout="wide", page_icon="📈")

# --- 1. GLOBAL DATA LOADING & UTILITIES ---

# --- CONSTANTS ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA8_PERIOD = 8
EMA21_PERIOD = 21

def parse_periods(periods_str):
    """Parses a comma-separated string into a list of integers."""
    try:
        # Split by comma, strip whitespace, convert to int, filter valid numbers, sort unique
        p_list = sorted(list(set([int(x.strip()) for x in periods_str.split(',') if x.strip().isdigit()])))
        if not p_list:
            return [10, 30, 60, 90, 180]
        return p_list
    except:
        return [10, 30, 60, 90, 180]

def add_technicals(df):
    """
    Centralized technical indicator calculation.
    Adds RSI (14), EMA (8, 21), and SMA (200) if they don't exist.
    """
    if df is None or df.empty: return df
    
    # 1. Identify Close Column
    cols = df.columns
    close_col = next((c for c in ['Price', 'Close', 'CLOSE'] if c in cols), None)
    
    if not close_col: return df

    # 2. RSI Calculation
    if not any(x in cols for x in ['RSI', 'RSI_14', 'RSI14']):
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_14'] = df['RSI']
    
    # 3. EMA/SMA Calculation
    if not any(x in cols for x in ['EMA8', 'EMA_8']):
        df['EMA_8'] = df[close_col].ewm(span=8, adjust=False).mean()
        
    if not any(x in cols for x in ['EMA21', 'EMA_21']):
        df['EMA_21'] = df[close_col].ewm(span=21, adjust=False).mean()
        
    if not any(x in cols for x in ['SMA200', 'SMA_200']):
        if len(df) >= 200:
            df['SMA_200'] = df[close_col].rolling(window=200).mean()
            
    return df

@st.cache_data(ttl=300)
def get_stock_indicators(sym: str):
    try:
        ticker_obj = yf.Ticker(sym)
        h_full = ticker_obj.history(period="2y", interval="1d")
        
        if len(h_full) == 0: return None, None, None, None, None
        
        # Centralized Calc
        h_full = add_technicals(h_full)
        
        sma200 = float(h_full["SMA_200"].iloc[-1]) if "SMA_200" in h_full.columns else None
        
        h_recent = h_full.iloc[-60:].copy() if len(h_full) > 60 else h_full.copy()
        if len(h_recent) == 0: return None, None, None, None, None
        
        spot_val = float(h_recent["Close"].iloc[-1])
        # Handle variations (Helper adds EMA_8, but existing files might have EMA8)
        ema8 = float(h_recent.get("EMA_8", h_recent.get("EMA8")).iloc[-1])
        ema21 = float(h_recent.get("EMA_21", h_recent.get("EMA21")).iloc[-1])
        
        return spot_val, ema8, ema21, sma200, h_full
    except: 
        return None, None, None, None, None

def get_table_height(df, max_rows=30):
    row_count = len(df)
    if row_count == 0:
        return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5

def get_confirmed_gdrive_data(url):
    try:
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
        
        if not file_id: return None
            
        download_url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(download_url, params={'id': file_id}, stream=True)
        
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_token = value
                break
        
        if not confirm_token:
            match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
            if match: confirm_token = match.group(1)

        if confirm_token:
            response = session.get(download_url, params={'id': file_id, 'confirm': confirm_token}, stream=True)
        
        if response.text.strip().startswith("<!DOCTYPE html>"): return "HTML_ERROR"
            
        return StringIO(response.text)
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_parquet_config():
    """
    Loads dataset configuration. 
    Priority 1: Text file in Google Drive (URL defined in secrets as URL_PARQUET_LIST)
    Priority 2: String in secrets (PARQUET_CONFIG)
    """
    config = {}
    
    # 1. Try loading from Google Drive Text File
    url_list = st.secrets.get("URL_PARQUET_LIST", "")
    if url_list:
        try:
            buffer = get_confirmed_gdrive_data(url_list)
            if buffer and buffer != "HTML_ERROR":
                content = buffer.getvalue()
                lines = content.strip().split('\n')
                for line in lines:
                    if not line.strip(): continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        name = re.sub(r'\\s*', '', name)
                        key = parts[1].strip()
                        config[name] = key
        except Exception as e:
            print(f"Error loading external config: {e}")

    # 2. Fallback to secrets string
    if not config:
        try:
            raw_config = st.secrets.get("PARQUET_CONFIG", "")
            if raw_config:
                lines = raw_config.strip().split('\n')
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        key = parts[1].strip()
                        config[name] = key
        except Exception:
            pass
    
    if not config:
        st.error("⛔ CRITICAL ERROR: No dataset configuration found. Please check 'URL_PARQUET_LIST' in your secrets.toml.")
        st.stop()
        
    return config

DATA_KEYS_PARQUET = get_parquet_config()

def get_gdrive_binary_data(url):
    try:
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
        
        if not file_id: return None
            
        download_url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(download_url, params={'id': file_id}, stream=True)
        
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_token = value
                break
        
        if not confirm_token:
            if b"<!DOCTYPE html>" in response.content[:200]:
                match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
                if match: confirm_token = match.group(1)

        if confirm_token:
            response = session.get(download_url, params={'id': file_id, 'confirm': confirm_token}, stream=True)
            
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Download Error: {e}")
        return None

@st.cache_data(ttl=900)
def load_parquet_and_clean(url_key):
    url = st.secrets.get(url_key)
    if not url: return None
    
    try:
        buffer = get_gdrive_binary_data(url)
        if not buffer: return None
        
        df = pd.read_parquet(buffer, engine='pyarrow')
        
        rename_map = {
            "RSI14": "RSI",
            "W_RSI14": "W_RSI",
            "W_EMA8": "W_EMA8",
            "W_EMA21": "W_EMA21",
            "EMA8": "EMA8",
            "EMA21": "EMA21"
        }
        
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        df.rename(columns=actual_rename, inplace=True)
        
        for col in df.columns:
            c_up = col.upper()
            if any(x in c_up for x in ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOL', 'RSI', 'EMA', 'SMA']):
                try:
                    df[col] = df[col].astype('float64')
                except Exception:
                    pass
        
        date_cols = [c for c in df.columns if "DATE" in c.upper()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            
        return df
    except Exception as e:
        st.error(f"Error loading {url_key}: {e}")
        return None

@st.cache_data(ttl=3600)
def load_ticker_map():
    try:
        url = st.secrets.get("URL_TICKER_MAP")
        if not url: return {}

        buffer = get_confirmed_gdrive_data(url)
        if buffer and buffer != "HTML_ERROR":
            df = pd.read_csv(buffer)
            if len(df.columns) >= 2:
                return dict(zip(df.iloc[:, 0].astype(str).str.strip().str.upper(), 
                              df.iloc[:, 1].astype(str).str.strip()))
    except Exception:
        pass
    return {}

@st.cache_data(ttl=300)
def get_ticker_technicals(ticker: str, mapping: dict):
    if not mapping or ticker not in mapping:
        return None
    file_id = mapping[ticker]
    file_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    buffer = get_confirmed_gdrive_data(file_url)
    if buffer and buffer != "HTML_ERROR":
        try:
            df = pd.read_csv(buffer)
            df.columns = [c.strip().upper() for c in df.columns]
            return df
        except:
            return None
    return None

def calculate_optimal_signal_stats(history_indices, price_array, current_idx, signal_type='Bullish', timeframe='Daily', periods_input=None, optimize_for='PF'):
    """
    Vectorized calculation of forward returns for multiple periods.
    optimize_for: 'PF' (Profit Factor) or 'SQN' (System Quality Number)
    """
    # 1. Filter for valid historical indices
    hist_arr = np.array(history_indices)
    valid_mask = hist_arr < current_idx
    valid_indices = hist_arr[valid_mask]
    
    if len(valid_indices) == 0:
        return None

    # Handle Periods
    if periods_input is None:
        periods = np.array([10, 30, 60, 90, 180])
    else:
        periods = np.array(periods_input)

    total_len = len(price_array)
    unit = 'w' if timeframe.lower() == 'weekly' else 'd'

    # 2. Vectorized Exit Index Calculation
    exit_indices_matrix = valid_indices[:, None] + periods[None, :]
    
    # 3. Create Validity Mask
    valid_exits_mask = exit_indices_matrix < total_len
    
    # 4. Fetch Prices Safely
    safe_exit_indices = np.clip(exit_indices_matrix, 0, total_len - 1)
    
    entry_prices = price_array[valid_indices]
    exit_prices_matrix = price_array[safe_exit_indices]
    
    # 5. Calculate Returns Matrix
    raw_returns_matrix = (exit_prices_matrix - entry_prices[:, None]) / entry_prices[:, None]
    
    if signal_type == 'Bearish':
        strat_returns_matrix = -raw_returns_matrix
    else:
        strat_returns_matrix = raw_returns_matrix

    # 6. Calculate Stats per Period
    best_score = -999.0
    best_stats = None
    
    for i, p in enumerate(periods):
        col_mask = valid_exits_mask[:, i]
        period_returns = strat_returns_matrix[col_mask, i]
        
        if len(period_returns) == 0:
            continue
            
        wins = period_returns[period_returns > 0]
        losses = period_returns[period_returns < 0]
        
        gross_win = np.sum(wins)
        gross_loss = np.abs(np.sum(losses))
        
        if gross_loss == 0:
            pf = 999.0 if gross_win > 0 else 0.0
        else:
            pf = gross_win / gross_loss
            
        n = len(period_returns)
        win_rate = (len(wins) / n) * 100
        avg_ret = np.mean(period_returns) * 100
        
        # --- SQN Calculation ---
        # Standard Deviation requires at least 2 data points generally, 
        # but numpy will return 0.0 for 1 data point (ddof=0 default).
        std_dev = np.std(period_returns)
        
        if std_dev > 0 and n > 0:
            sqn = (np.mean(period_returns) / std_dev) * np.sqrt(n)
        else:
            sqn = 0.0
        
        # Determine Score based on optimization metric
        current_score = pf if optimize_for == 'PF' else sqn
        
        if current_score > best_score:
            best_score = current_score
            best_stats = {
                "Best Period": f"{p}{unit}",
                "Profit Factor": pf,
                "Win Rate": win_rate,
                "EV": avg_ret,
                "N": n,
                "SQN": sqn
            }
            
    return best_stats

@st.cache_data(ttl=86400)
def fetch_yahoo_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10y")
        if df.empty: return None
        
        df = df.reset_index()
        date_col_name = df.columns[0]
        df = df.rename(columns={date_col_name: "DATE"})
        
        if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
            df["DATE"] = pd.to_datetime(df["DATE"])
        if df["DATE"].dt.tz is not None:
            df["DATE"] = df["DATE"].dt.tz_localize(None)
            
        df = df.rename(columns={"Close": "CLOSE", "Volume": "VOLUME", "High": "HIGH", "Low": "LOW", "Open": "OPEN"})
        df.columns = [c.upper() for c in df.columns]
        
        # Centralized Calc
        df = add_technicals(df)
        
        return df
    except Exception:
        return None

@st.cache_data(ttl=86400)
def fetch_benchmark_data():
    """Fetches SPY and QQQ data for benchmarking. Priority: PARQUET_MACRO -> yfinance"""
    spy_df = pd.DataFrame()
    qqq_df = pd.DataFrame()

    # 1. Try Parquet from SECRETS if available
    macro_url = st.secrets.get("PARQUET_MACRO")
    if macro_url:
        try:
            buffer = get_gdrive_binary_data(macro_url)
            if buffer:
                macro_df = pd.read_parquet(buffer, engine='pyarrow')
                # Clean columns: Strip, upper, remove special chars
                macro_df.columns = [c.strip().replace(' ', '').replace('-', '').upper() for c in macro_df.columns]
                
                # Identify key columns
                t_col = next((c for c in macro_df.columns if c in ['TICKER', 'SYMBOL']), None)
                d_col = next((c for c in macro_df.columns if 'DATE' in c), None)
                # Prefer normal Close, fallback to Adjusted if needed, exclude Weekly
                c_col = next((c for c in macro_df.columns if 'CLOSE' in c and 'W_' not in c), None)

                if t_col and d_col and c_col:
                    # Ensure Date is datetime and timezone-naive
                    macro_df[d_col] = pd.to_datetime(macro_df[d_col])
                    if macro_df[d_col].dt.tz is not None:
                        macro_df[d_col] = macro_df[d_col].dt.tz_localize(None)
                    
                    # Process SPY
                    spy_raw = macro_df[macro_df[t_col] == 'SPY'].copy()
                    if not spy_raw.empty:
                        spy_df = spy_raw[[d_col, c_col]].rename(columns={c_col: 'CLOSE'}).set_index(d_col).sort_index()
                    
                    # Process QQQ
                    qqq_raw = macro_df[macro_df[t_col] == 'QQQ'].copy()
                    if not qqq_raw.empty:
                        qqq_df = qqq_raw[[d_col, c_col]].rename(columns={c_col: 'CLOSE'}).set_index(d_col).sort_index()
        except Exception as e:
            print(f"Macro Parquet Load Error: {e}")

    # 2. Fallback to yfinance if parquet failed or data missing
    if spy_df.empty or qqq_df.empty:
        try:
            data = yf.download("SPY QQQ", period="10y", group_by='ticker', auto_adjust=True, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    # yfinance might return different structures depending on version
                    if 'SPY' in data.columns.levels[0]:
                        spy_df = data['SPY'].reset_index()
                    if 'QQQ' in data.columns.levels[0]:
                        qqq_df = data['QQQ'].reset_index()
                except KeyError:
                    pass
            
            # Standardize fallback data
            for d in [spy_df, qqq_df]:
                if not d.empty:
                    d.columns = [c.upper() for c in d.columns]
                    d_col = next((c for c in d.columns if "DATE" in c), "DATE")
                    d[d_col] = pd.to_datetime(d[d_col])
                    if d[d_col].dt.tz is not None:
                        d[d_col] = d[d_col].dt.tz_localize(None)
                    d.set_index(d_col, inplace=True)
                    d.sort_index(inplace=True)
                    
        except Exception as e:
            print(f"Benchmark yfinance error: {e}")

    return spy_df, qqq_df

def prepare_data(df):
    # Standardize column names (removes spaces, dashes, converts to UPPER)
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    cols = df.columns
    date_col = next((c for c in cols if 'DATE' in c), None)
    close_col = next((c for c in cols if 'CLOSE' in c and 'W_' not in c), None)
    vol_col = next((c for c in cols if ('VOL' in c or 'VOLUME' in c) and 'W_' not in c), None)
    high_col = next((c for c in cols if 'HIGH' in c and 'W_' not in c), None)
    low_col = next((c for c in cols if 'LOW' in c and 'W_' not in c), None)
    open_col = next((c for c in cols if 'OPEN' in c and 'W_' not in c), None) # ADDED OPEN
    
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # --- BUILD DAILY ---
    d_rsi = next((c for c in cols if c in ['RSI', 'RSI14'] and 'W_' not in c), 'RSI')
    d_ema8 = next((c for c in cols if c == 'EMA8'), 'EMA8')
    d_ema21 = next((c for c in cols if c == 'EMA21'), 'EMA21')

    needed_cols = [close_col, vol_col, high_col, low_col]
    if open_col: needed_cols.append(open_col) # Include Open
    
    if d_rsi in df.columns: needed_cols.append(d_rsi)
    if d_ema8 in df.columns: needed_cols.append(d_ema8)
    if d_ema21 in df.columns: needed_cols.append(d_ema21)
    
    df_d = df[needed_cols].copy()
    
    rename_dict = {close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low'}
    if open_col: rename_dict[open_col] = 'Open' # Rename Open
    if d_rsi in df_d.columns: rename_dict[d_rsi] = 'RSI'
    if d_ema8 in df_d.columns: rename_dict[d_ema8] = 'EMA8'
    if d_ema21 in df_d.columns: rename_dict[d_ema21] = 'EMA21'
    
    df_d.rename(columns=rename_dict, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    
    df_d = add_technicals(df_d)
    df_d = df_d.dropna(subset=['Price', 'RSI'])
    
    # --- BUILD WEEKLY ---
    w_close, w_vol = 'W_CLOSE', 'W_VOLUME'
    w_high, w_low = 'W_HIGH', 'W_LOW'
    
    w_rsi_source = next((c for c in cols if c in ['W_RSI', 'W_RSI14']), None)
    w_ema8_source = next((c for c in cols if c in ['W_EMA8']), None)
    w_ema21_source = next((c for c in cols if c in ['W_EMA21']), None)
    
    if all(c in df.columns for c in [w_close, w_vol, w_high, w_low]):
        cols_w = [w_close, w_vol, w_high, w_low]
        if w_rsi_source: cols_w.append(w_rsi_source)
        if w_ema8_source: cols_w.append(w_ema8_source)
        if w_ema21_source: cols_w.append(w_ema21_source)
        
        df_w = df[cols_w].copy()
        
        w_rename = {w_close: 'Price', w_vol: 'Volume', w_high: 'High', w_low: 'Low'}
        if w_rsi_source: w_rename[w_rsi_source] = 'RSI'
        if w_ema8_source: w_rename[w_ema8_source] = 'EMA8'
        if w_ema21_source: w_rename[w_ema21_source] = 'EMA21'
        
        df_w.rename(columns=w_rename, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
        
        df_w = add_technicals(df_w)
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else: 
        df_w = None
        
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe, min_n=0, periods_input=None, optimize_for='PF'):
    divergences = []
    n_rows = len(df_tf)
    
    if n_rows < DIVERGENCE_LOOKBACK + 1: return divergences
    
    rsi_vals = df_tf['RSI'].values
    low_vals = df_tf['Low'].values
    high_vals = df_tf['High'].values
    vol_vals = df_tf['Volume'].values
    vol_sma_vals = df_tf['VolSMA'].values
    close_vals = df_tf['Price'].values
    
    def get_date_str(idx, fmt='%Y-%m-%d'): 
        ts = df_tf.index[idx]
        if timeframe.lower() == 'weekly': 
             return df_tf.iloc[idx]['ChartDate'].strftime(fmt)
        return ts.strftime(fmt)
    
    # ---------------------------------------------------------
    # PASS 1: VECTORIZED PRE-CHECK
    # ---------------------------------------------------------
    roll_low_min = pd.Series(low_vals).shift(1).rolling(window=DIVERGENCE_LOOKBACK).min().values
    roll_high_max = pd.Series(high_vals).shift(1).rolling(window=DIVERGENCE_LOOKBACK).max().values
    
    is_new_low = (low_vals < roll_low_min)
    is_new_high = (high_vals > roll_high_max)
    
    valid_mask = np.zeros(n_rows, dtype=bool)
    valid_mask[DIVERGENCE_LOOKBACK:] = True
    
    candidate_indices = np.where(valid_mask & (is_new_low | is_new_high))[0]
    
    bullish_signal_indices = []
    bearish_signal_indices = []
    potential_signals = [] 

    # ---------------------------------------------------------
    # PASS 2: SCAN CANDIDATES
    # ---------------------------------------------------------
    for i in candidate_indices:
        p2_rsi = rsi_vals[i]
        p2_vol = vol_vals[i]
        p2_volsma = vol_sma_vals[i]
        
        lb_start = i - DIVERGENCE_LOOKBACK
        lb_rsi = rsi_vals[lb_start:i]
        
        is_vol_high = int(p2_vol > (p2_volsma * 1.5)) if not np.isnan(p2_volsma) else 0
        
        # Bullish Divergence
        if is_new_low[i]:
            p1_idx_rel = np.argmin(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            
            if p2_rsi > (p1_rsi + RSI_DIFF_THRESHOLD):
                idx_p1_abs = lb_start + p1_idx_rel
                subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                if not np.any(subset_rsi > 50): 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi <= p1_rsi): valid = False
                    
                    if valid:
                        bullish_signal_indices.append(i)
                        potential_signals.append({"index": i, "type": "Bullish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})
        
        # Bearish Divergence
        elif is_new_high[i]:
            p1_idx_rel = np.argmax(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            
            if p2_rsi < (p1_rsi - RSI_DIFF_THRESHOLD):
                idx_p1_abs = lb_start + p1_idx_rel
                subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                if not np.any(subset_rsi < 50): 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi >= p1_rsi): valid = False
                    
                    if valid:
                        bearish_signal_indices.append(i)
                        potential_signals.append({"index": i, "type": "Bearish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})

    # ---------------------------------------------------------
    # PASS 3: REPORT & BACKTEST
    # ---------------------------------------------------------
    display_threshold_idx = n_rows - SIGNAL_LOOKBACK_PERIOD
    
    for sig in potential_signals:
        i = sig["index"]
        if i < display_threshold_idx: continue

        s_type = sig["type"]
        idx_p1_abs = sig["p1_idx"]
        
        tags = []
        latest_row = df_tf.iloc[-1]
        last_price = latest_row['Price']
        last_ema8 = latest_row.get('EMA8') 
        last_ema21 = latest_row.get('EMA21')

        def is_valid(val): return val is not None and not pd.isna(val)

        if s_type == 'Bullish':
            if is_valid(last_ema8) and last_price >= last_ema8: tags.append(f"EMA{EMA8_PERIOD}")
            if is_valid(last_ema21) and last_price >= last_ema21: tags.append(f"EMA{EMA21_PERIOD}")
        else: 
            if is_valid(last_ema8) and last_price <= last_ema8: tags.append(f"EMA{EMA8_PERIOD}")
            if is_valid(last_ema21) and last_price <= last_ema21: tags.append(f"EMA{EMA8_PERIOD}")
            
        if sig["vol_high"]: tags.append("V_HI")
        if vol_vals[i] > vol_vals[idx_p1_abs]: tags.append("V_GROW")
        
        sig_date_iso = get_date_str(i, '%Y-%m-%d')
        date_display = f"{get_date_str(idx_p1_abs, '%b %d')} → {get_date_str(i, '%b %d')}"
        rsi_display = f"{int(round(rsi_vals[idx_p1_abs]))} {'↗' if rsi_vals[i] > rsi_vals[idx_p1_abs] else '↘'} {int(round(rsi_vals[i]))}"
        
        price_p1 = low_vals[idx_p1_abs] if s_type=='Bullish' else high_vals[idx_p1_abs]
        price_p2 = low_vals[i] if s_type=='Bullish' else high_vals[i]
        price_display = f"${price_p1:,.2f} ↗ ${price_p2:,.2f}" if price_p2 > price_p1 else f"${price_p1:,.2f} ↘ ${price_p2:,.2f}"

        hist_list = bullish_signal_indices if s_type == 'Bullish' else bearish_signal_indices
        best_stats = calculate_optimal_signal_stats(hist_list, close_vals, i, signal_type=s_type, timeframe=timeframe, periods_input=periods_input, optimize_for=optimize_for)
        
        if best_stats is None:
             best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0}
        
        if best_stats["N"] < min_n: continue

        # --- EV PRICE CALCULATION ---
        ev_val = best_stats['EV']
        sig_close = close_vals[i]
        
        # New Rule: If N=0, EV Target is 0
        if best_stats['N'] == 0:
            ev_price = 0.0
        else:
            if s_type == 'Bullish':
                ev_price = sig_close * (1 + (ev_val / 100.0))
            else:
                ev_price = sig_close * (1 - (ev_val / 100.0))

        divergences.append({
            'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe, 
            'Tags': tags, 'Signal_Date_ISO': sig_date_iso, 'Date_Display': date_display,
            'RSI_Display': rsi_display, 'Price_Display': price_display, 'Last_Close': f"${latest_row['Price']:,.2f}", 
            'Best Period': best_stats['Best Period'], 'Profit Factor': best_stats['Profit Factor'],
            'Win Rate': best_stats['Win Rate'], 'EV': best_stats['EV'], 
            'EV Target': ev_price, 
            'N': best_stats['N'],
            'SQN': best_stats.get('SQN', 0.0)
        })
            
    return divergences

def find_rsi_percentile_signals(df, ticker, pct_low=0.10, pct_high=0.90, min_n=1, filter_date=None, timeframe='Daily', periods_input=None, optimize_for='SQN', backtest_mode=False, min_pf=0.0, min_wr=0.0, min_ev=0.0, min_sqn=0.0):
    signals = []
    if len(df) < 200: return signals
    
    # 1. Identify ALL Signal Indices in full history (needed for percentiles)
    cutoff = df.index.max() - timedelta(days=365*10)
    hist_df = df[df.index >= cutoff].copy()
    
    if hist_df.empty: return signals
    
    p10 = hist_df['RSI'].quantile(pct_low)
    p90 = hist_df['RSI'].quantile(pct_high)
    
    rsi_series = hist_df['RSI']
    rsi_vals = rsi_series.values 
    price_vals = hist_df['Price'].values
    
    # Identify Signal Indices
    prev_rsi = rsi_series.shift(1)
    
    bull_mask = (prev_rsi < p10) & (rsi_series >= (p10 + 1.0))
    bear_mask = (prev_rsi > p90) & (rsi_series <= (p90 - 1.0))
    
    bullish_signal_indices = np.where(bull_mask)[0].tolist()
    bearish_signal_indices = np.where(bear_mask)[0].tolist()
            
    # 2. Filter and Optimize
    latest_close = df['Price'].iloc[-1] 
    all_indices = sorted(bullish_signal_indices + bearish_signal_indices)
    
    for i in all_indices:
        curr_row = hist_df.iloc[i]
        curr_date = curr_row.name.date()
        
        # --- VIEW MODE: Only latest dates ---
        if not backtest_mode:
            if filter_date and curr_date < filter_date:
                continue
        
        is_bullish = i in bullish_signal_indices
        s_type = 'Bullish' if is_bullish else 'Bearish'
        thresh_val = p10 if is_bullish else p90
        curr_rsi_val = rsi_vals[i]
        
        hist_list = bullish_signal_indices if is_bullish else bearish_signal_indices
        best_stats = calculate_optimal_signal_stats(hist_list, price_vals, i, signal_type=s_type, timeframe=timeframe, periods_input=periods_input, optimize_for=optimize_for)
        
        if best_stats is None:
             best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0, "SQN": 0.0}
             
        # --- FILTERS ---
        if best_stats["N"] < min_n: continue
        if best_stats["Profit Factor"] < min_pf: continue
        if best_stats["Win Rate"] < min_wr: continue
        if best_stats["EV"] < min_ev: continue
        if best_stats.get("SQN", 0) < min_sqn: continue
            
        rsi_disp = f"{thresh_val:.0f} ↗ {curr_rsi_val:.0f}" if is_bullish else f"{thresh_val:.0f} ↘ {curr_rsi_val:.0f}"
        action_str = "Leaving Low" if is_bullish else "Leaving High"
        
        # --- NEW EV TARGET CALCULATION ---
        ev_val = best_stats['EV']
        sig_close = curr_row['Price']
        
        # New Rule: If N=0, EV Target is 0
        if best_stats['N'] == 0:
            ev_price = 0.0
        else:
            if is_bullish:
                ev_price = sig_close * (1 + (ev_val / 100.0))
            else:
                ev_price = sig_close * (1 - (ev_val / 100.0))

        sig_data = {
            'Ticker': ticker,
            'Date': curr_row.name.strftime('%b %d'),
            'Date_Obj': curr_date,
            'Action': action_str,
            'RSI_Display': rsi_disp,
            'Signal_Price': f"${sig_close:,.2f}",
            'Last_Close': f"${latest_close:,.2f}", 
            'Signal_Type': s_type,
            'Best Period': best_stats['Best Period'],
            'Profit Factor': best_stats['Profit Factor'],
            'Win Rate': best_stats['Win Rate'],
            'EV': best_stats['EV'],
            'EV Target': ev_price,
            'N': best_stats['N'],
            'SQN': best_stats.get('SQN', 0.0)
        }

        # --- BACKTEST MODE LOGIC ---
        if backtest_mode:
            # "Buy at open the trading day after the signal date"
            # "Hold for trading days chosen by optimal time period"
            # "Sell at close the last day of optimal time period"
            
            # 1. Parse Period
            best_period_str = best_stats['Best Period']
            try:
                p_val = int(re.search(r'\d+', str(best_period_str)).group())
            except:
                p_val = 0
            
            if p_val > 0:
                # 2. Identify Indices
                # Signal is at 'i'. Entry is 'i+1'. Exit is 'i+1+p_val'?
                # "Hold for the trading days chosen"
                # If I hold for 1 day, I enter T+1 Open, exit T+1 Close? Or T+2 Close?
                # Standard convention: Hold period X starts from entry.
                # So Entry at i+1. Exit at i+1+p_val.
                entry_idx = i + 1
                exit_idx = i + 1 + p_val + 1

                if exit_idx < len(hist_df):
                    # Check validity of 'Open' column
                    has_open = 'Open' in hist_df.columns
                    
                    entry_date = hist_df.index[entry_idx]
                    exit_date = hist_df.index[exit_idx]
                    
                    entry_price = hist_df['Open'].iloc[entry_idx] if has_open else hist_df['Price'].iloc[entry_idx]
                    exit_price = hist_df['Price'].iloc[exit_idx]
                    
                    # Calculate Trade Return
                    trade_ret = (exit_price - entry_price) / entry_price
                    if s_type == 'Bearish': trade_ret = -trade_ret
                    
                    sig_data['BT_Entry_Date'] = entry_date
                    sig_data['BT_Exit_Date'] = exit_date
                    sig_data['BT_Return'] = trade_ret
                    sig_data['BT_Hold_Days'] = p_val
                else:
                    # Period not yet passed
                    sig_data['BT_Return'] = None
            else:
                sig_data['BT_Return'] = None
        
        signals.append(sig_data)
            
    return signals

# --- 2. APP MODULES ---

def run_rsi_scanner_app(df_global):
    st.title("📈 RSI Scanner")
    
    st.markdown("""
        <style>
        .top-note { color: #888888; font-size: 14px; margin-bottom: 2px; font-family: inherit; }
        .footer-header { color: #31333f; margin-top: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: 5px; font-weight: bold; }
        [data-testid="stDataFrame"] th { font-weight: 900 !important; }
        </style>
        """, unsafe_allow_html=True)
    
    # --- Session State Init ---
    if 'saved_rsi_div_min_n' not in st.session_state: st.session_state.saved_rsi_div_min_n = 0
    if 'saved_rsi_div_periods' not in st.session_state: st.session_state.saved_rsi_div_periods = "10,30,60,90,180"
    if 'saved_rsi_div_opt' not in st.session_state: st.session_state.saved_rsi_div_opt = "Profit Factor" # Default PF
    
    if 'saved_rsi_pct_low' not in st.session_state: st.session_state.saved_rsi_pct_low = 10
    if 'saved_rsi_pct_high' not in st.session_state: st.session_state.saved_rsi_pct_high = 90
    if 'saved_rsi_pct_show' not in st.session_state: st.session_state.saved_rsi_pct_show = "Everything"
    if 'saved_rsi_pct_opt' not in st.session_state: st.session_state.saved_rsi_pct_opt = "SQN" # Default SQN
    
    # We'll set the default date dynamically below, but init here to avoid errors
    if 'saved_rsi_pct_date' not in st.session_state: st.session_state.saved_rsi_pct_date = None
    if 'saved_rsi_pct_min_n' not in st.session_state: st.session_state.saved_rsi_pct_min_n = 5
    if 'saved_rsi_pct_periods' not in st.session_state: st.session_state.saved_rsi_pct_periods = "10,30,60,90,180"

    # New Filter States
    if 'saved_rsi_min_pf' not in st.session_state: st.session_state.saved_rsi_min_pf = 2.0
    if 'saved_rsi_min_wr' not in st.session_state: st.session_state.saved_rsi_min_wr = 60
    if 'saved_rsi_min_ev' not in st.session_state: st.session_state.saved_rsi_min_ev = 5.0
    if 'saved_rsi_min_sqn' not in st.session_state: st.session_state.saved_rsi_min_sqn = 3.0

    def save_rsi_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]
        
    dataset_map = DATA_KEYS_PARQUET
    options = list(dataset_map.keys())
    
    # Helper to map dropdown text to function codes
    OPT_MAP = {"Profit Factor": "PF", "SQN": "SQN"}

    # --- TABS SETUP (MODIFIED: Only Percentile Backtester Visible) ---
    tab_pct_bt, = st.tabs(["📊 Percentile Backtester"])

    with tab_pct_bt:
        st.markdown("### 📊 Strategy Backtester")
        
        # --- INPUTS ---
        data_option_bt = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_bt_pills_new")
        
        # ROW 1 (6 Cols)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: in_low_bt = st.number_input("RSI Low", min_value=1, max_value=49, value=st.session_state.saved_rsi_pct_low, step=1, key="rsi_bt_low")
        with c2: in_high_bt = st.number_input("RSI High", min_value=51, max_value=99, value=st.session_state.saved_rsi_pct_high, step=1, key="rsi_bt_high")
        
        show_opts = ["Everything", "Leaving High", "Leaving Low"]
        curr_show = st.session_state.saved_rsi_pct_show
        idx_show = show_opts.index(curr_show) if curr_show in show_opts else 0
        with c3: show_filter_bt = st.selectbox("Show", show_opts, index=idx_show, key="rsi_bt_show")
        
        with c4: bt_start_date = st.date_input("Start", value=date(2020, 1, 1), key="rsi_bt_start")
        with c5: bt_end_date = st.date_input("End", value=date(2025, 12, 31), key="rsi_bt_end")
        
        with c6:
             curr_pct_opt = st.session_state.saved_rsi_pct_opt
             idx_pct_opt = ["Profit Factor", "SQN"].index(curr_pct_opt) if curr_pct_opt in ["Profit Factor", "SQN"] else 1
             opt_mode_bt = st.selectbox("Optimize By", ["Profit Factor", "SQN"], index=idx_pct_opt, key="rsi_bt_opt")

        # ROW 2 (6 Cols)
        r2_c1, r2_c2, r2_c3, r2_c4, r2_c5, r2_c6 = st.columns(6)
        
        with r2_c1: periods_str_bt = st.text_input("Test Periods", value=st.session_state.saved_rsi_pct_periods, key="rsi_bt_periods")
        with r2_c2: min_pf_bt = st.number_input("Min PF", value=st.session_state.saved_rsi_min_pf, step=0.1, key="rsi_bt_min_pf")
        with r2_c3: min_wr_bt = st.number_input("Min WR %", value=float(st.session_state.saved_rsi_min_wr), step=5.0, key="rsi_bt_min_wr")
        with r2_c4: min_ev_bt = st.number_input("Min EV %", value=st.session_state.saved_rsi_min_ev, step=0.5, key="rsi_bt_min_ev")
        with r2_c5: min_sqn_bt = st.number_input("Min SQN", value=st.session_state.saved_rsi_min_sqn, step=0.5, key="rsi_bt_min_sqn")
        with r2_c6: min_n_bt = st.number_input("Min N", min_value=0, value=st.session_state.saved_rsi_pct_min_n, step=1, key="rsi_bt_min_n")

        periods_bt = parse_periods(periods_str_bt)
        pct_opt_code_bt = OPT_MAP[opt_mode_bt]
        
        if data_option_bt:
            try:
                key = dataset_map[data_option_bt]
                master = load_parquet_and_clean(key)
                
                if master is not None and not master.empty:
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    all_tickers = sorted(master[t_col].unique())

                    # --- 1. RUN BACKTEST LOGIC ---
                    if st.button("Run Backtest Strategy", type="primary"):
                        with st.spinner("Caching Benchmark Data..."):
                            spy_df, qqq_df = fetch_benchmark_data()
                        
                        raw_signals_bt = []
                        progress_bar = st.progress(0, text="Backtesting signals...")
                        grouped = master.groupby(t_col)
                        grouped_list = list(grouped)
                        total_groups = len(grouped_list)
                        
                        # Use User Selected Dates
                        start_cutoff = bt_start_date
                        end_cutoff = bt_end_date

                        for i, (ticker, group) in enumerate(grouped_list):
                            d_d, _ = prepare_data(group.copy())
                            if d_d is not None:
                                # Get ALL signals first, then we filter by date for the backtest range
                                # PASS NEW FILTERS HERE
                                sigs = find_rsi_percentile_signals(
                                    d_d, ticker, 
                                    pct_low=in_low_bt/100.0, pct_high=in_high_bt/100.0, 
                                    min_n=min_n_bt, timeframe='Daily', periods_input=periods_bt, 
                                    optimize_for=pct_opt_code_bt, backtest_mode=True,
                                    min_pf=min_pf_bt, min_wr=min_wr_bt, min_ev=min_ev_bt, min_sqn=min_sqn_bt
                                )
                                
                                for s in sigs:
                                    # Filter by Action User Input
                                    if show_filter_bt == "Leaving High" and s['Signal_Type'] != "Bearish": continue
                                    if show_filter_bt == "Leaving Low" and s['Signal_Type'] != "Bullish": continue

                                    # Filter by Date Range (User Selected)
                                    s_date = s['Date_Obj']
                                    if s_date < start_cutoff or s_date > end_cutoff:
                                        continue
                                    
                                    # Only include signals where optimal period passed
                                    if s.get('BT_Return') is not None:
                                        # Lookup Benchmark Returns for same period
                                        entry_d = s['BT_Entry_Date']
                                        exit_d = s['BT_Exit_Date']
                                        
                                        def get_bm_ret(bm_df):
                                            try:
                                                # Ensure timezone naive for comparison
                                                e_d = entry_d.replace(tzinfo=None) if hasattr(entry_d, 'tzinfo') else entry_d
                                                x_d = exit_d.replace(tzinfo=None) if hasattr(exit_d, 'tzinfo') else exit_d

                                                # Find closest dates if exact match missing
                                                # Use searchsorted/get_indexer logic or simple loc if full
                                                # Benchmark is indexed by date
                                                if e_d in bm_df.index and x_d in bm_df.index:
                                                    p1 = bm_df.loc[e_d]['CLOSE']
                                                    p2 = bm_df.loc[x_d]['CLOSE']
                                                    return (p2 - p1) / p1
                                                else:
                                                    # Try nearest
                                                    p1 = bm_df.iloc[bm_df.index.get_indexer([e_d], method='nearest')[0]]['CLOSE']
                                                    p2 = bm_df.iloc[bm_df.index.get_indexer([x_d], method='nearest')[0]]['CLOSE']
                                                    return (p2 - p1) / p1
                                            except:
                                                return 0.0
                                        
                                        s['SPY_Ret'] = get_bm_ret(spy_df)
                                        s['QQQ_Ret'] = get_bm_ret(qqq_df)
                                        
                                        raw_signals_bt.append(s)

                            if i % 10 == 0 or i == total_groups - 1: progress_bar.progress((i + 1) / total_groups)
                        
                        progress_bar.empty()
                        
                        # Store results in session state
                        if raw_signals_bt:
                            st.session_state.bt_results_df = pd.DataFrame(raw_signals_bt)
                        else:
                            st.session_state.bt_results_df = pd.DataFrame()

                    # --- DISPLAY RESULTS (From Session State) ---
                    if 'bt_results_df' in st.session_state and not st.session_state.bt_results_df.empty:
                        df_res_full = st.session_state.bt_results_df
                        
                        # Need to re-fetch benchmark strictly for the Buy&Hold section (since it uses separate start/end dates logic)
                        # Since fetch_benchmark_data is cached, this is effectively instant
                        spy_df, qqq_df = fetch_benchmark_data() 

                        # --- FILTER SECTION ---
                        f_col1, f_col2 = st.columns([1, 3])
                        with f_col1:
                            ticker_input_bt = st.text_input("Ticker Filter (blank=all)", key="rsi_bt_ticker_input_filter").strip().upper()
                        
                        # Apply Filter
                        if ticker_input_bt:
                            df_res = df_res_full[df_res_full['Ticker'] == ticker_input_bt].copy()
                        else:
                            df_res = df_res_full.copy()

                        with f_col2:
                            # Spacer
                            st.write("")
                            st.write("")
                            # CSV Download (Downloads the filtered view)
                            csv = df_res.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Signals",
                                data=csv,
                                file_name='rsi_percentile_backtest.csv',
                                mime='text/csv',
                            )

                        if df_res.empty:
                            st.warning(f"No signals found for ticker: {ticker_input_bt}")
                        else:
                            # --- CALCULATE ACTIVE TRADES STATS ---
                            # Logic: +1 to count on Entry Day, -1 to count on (Exit Day + 1)
                            # Ensure datetime types
                            df_res['BT_Entry_Date'] = pd.to_datetime(df_res['BT_Entry_Date'])
                            df_res['BT_Exit_Date'] = pd.to_datetime(df_res['BT_Exit_Date'])
                            
                            entry_counts = df_res['BT_Entry_Date'].value_counts()
                            # We subtract 1 the day AFTER exit, because we hold through the exit close
                            exit_counts = (df_res['BT_Exit_Date'] + timedelta(days=1)).value_counts()
                            
                            # Merge into daily events timeline
                            events_df = pd.DataFrame({'Entries': entry_counts, 'Exits': exit_counts}).fillna(0)
                            events_df.sort_index(inplace=True)
                            
                            # Reindex to fill gaps (including weekends/holidays to represent capital tied up duration)
                            if not events_df.empty:
                                full_idx = pd.date_range(start=events_df.index.min(), end=events_df.index.max())
                                events_df = events_df.reindex(full_idx).fillna(0)
                            
                            events_df['Net_Change'] = events_df['Entries'] - events_df['Exits']
                            events_df['Active_Count'] = events_df['Net_Change'].cumsum()
                            
                            avg_active = events_df['Active_Count'].mean()
                            max_active = events_df['Active_Count'].max()
                            if max_active == 0: max_active = 1 # Avoid div by zero

                            # --- FORMATTING HELPERS ---
                            def fmt_pct(x):
                                """Formats 0.10 as 10%, -0.05 as (5%), 10.0 as 1,000%"""
                                if not isinstance(x, (float, int)) or pd.isna(x): return x
                                if x < 0: return f"({abs(x):,.0%})"
                                return f"{x:,.0%}"

                            def color_ret(val):
                                """Light green for positive, light red for negative"""
                                if not isinstance(val, (float, int)) or pd.isna(val): return ''
                                color = '#e6f4ea' if val >= 0 else '#fce8e6' 
                                return f'background-color: {color}; color: #000000;'

                            # Layout Columns
                            res_col1, res_col2 = st.columns([1, 1], gap="large")

                            with res_col1:
                                # --- TABLE 1: TICKER SUMMARY ---
                                st.subheader("Results by Ticker")
                                
                                # Methodology Expander
                                with st.expander("ℹ️ Strategy Methodology"):
                                    st.markdown("""
                                    **Strategy Execution:**
                                    1. **Signal:** Triggered by RSI Percentile criteria.
                                    2. **Buy:** At **Open** on the trading day *after* the signal (T+1).
                                    3. **Hold:** For the optimal period determined by historical data.
                                    4. **Sell:** At **Close** on the last day of that period.
                                    """)
                                
                                # Calculate Ticker-Specific Max Active & ROI
                                # Need to do this carefully. Groupby alone doesn't give daily active timeline.
                                # For "Results by Ticker", we want Effective ROI = Sum(Ret) / Ticker_Max_Active
                                
                                results_by_ticker = []
                                for t, t_df in df_res.groupby('Ticker'):
                                    # Calculate Max Active for THIS ticker
                                    t_entries = t_df['BT_Entry_Date'].value_counts()
                                    t_exits = (t_df['BT_Exit_Date'] + timedelta(days=1)).value_counts()
                                    t_tl = pd.DataFrame({'E': t_entries, 'X': t_exits}).fillna(0).sort_index()
                                    # Don't need full reindex for just max, cumsum works on sorted unique dates
                                    t_tl['Net'] = t_tl['E'] - t_tl['X']
                                    t_active = t_tl['Net'].cumsum().max()
                                    if t_active <= 0: t_active = 1
                                    
                                    sum_ret = t_df['BT_Return'].sum()
                                    eff_roi = sum_ret / t_active
                                    
                                    results_by_ticker.append({
                                        'Ticker': t,
                                        'N_Trades': len(t_df),
                                        'Avg_Hold': t_df['BT_Hold_Days'].mean(),
                                        'Effective_ROI': eff_roi
                                    })
                                
                                agg_ticker = pd.DataFrame(results_by_ticker)
                                
                                # Sort by return high to low for better visibility
                                if not agg_ticker.empty:
                                    agg_ticker = agg_ticker.sort_values(by="Effective_ROI", ascending=False)
                                
                                st.dataframe(
                                    agg_ticker.style
                                    .format({
                                        "Avg_Hold": "{:.1f} d", 
                                        "Effective_ROI": fmt_pct
                                    })
                                    .map(color_ret, subset=["Effective_ROI"]),
                                    column_config={
                                        "Ticker": "Ticker",
                                        "N_Trades": "N",
                                        "Avg_Hold": "Avg Hold Time",
                                        "Effective_ROI": st.column_config.TextColumn("Effective ROI", help="Total Return / Max Active Trades")
                                    },
                                    hide_index=True,
                                    use_container_width=True,
                                    height=get_table_height(agg_ticker, 15)
                                )

                            with res_col2:
                                # --- NEW TABLE: ALL TICKERS SUMMARY ---
                                st.subheader("Results Summary")
                                
                                # Calculate Overall Effective ROI
                                # Effective ROI = Total Strategy Return / Max Active Trades
                                # If Max Active is 0 (no trades), ROI is 0
                                overall_eff_roi = df_res['BT_Return'].sum() / max_active if max_active > 0 else 0.0

                                agg_all = pd.DataFrame([{
                                    "N_Trades": len(df_res),
                                    "Avg_Hold": df_res['BT_Hold_Days'].mean(),
                                    "Effective_ROI": overall_eff_roi,
                                    "Avg_Active": avg_active,
                                    "Max_Active": max_active
                                }])
                                
                                st.dataframe(
                                    agg_all.style
                                    .format({
                                        "Avg_Hold": "{:.1f} d", 
                                        "Effective_ROI": fmt_pct,
                                        "N_Trades": "{:,}",
                                        "Avg_Active": "{:,.0f}",
                                        "Max_Active": "{:,.0f}"
                                    })
                                    .map(color_ret, subset=["Effective_ROI"]),
                                    column_config={
                                        "N_Trades": "Total N",
                                        "Avg_Hold": "Avg Hold Time",
                                        "Effective_ROI": st.column_config.TextColumn("Effective ROI", help="Total Return / Max Active. Represents return on peak capital required."),
                                        "Avg_Active": "Avg Active",
                                        "Max_Active": "Max Active"
                                    },
                                    hide_index=True,
                                    use_container_width=True
                                )
                                st.caption(f"ℹ️ **Max Active:** Peak number of simultaneous trades. You would need to split capital into **{int(max_active)} units** (approx {100/max_active:.1f}% each) to take every signal without running out of cash.")
                                st.caption(f"ℹ️ **Avg Active:** Average number of simultaneous trades. On a typical day, you are holding **{avg_active:,.0f} positions**.")

                                # --- PRE-CALCULATE BUY & HOLD FOR BENCHMARK REFERENCE ---
                                def get_bh_return(bm_df, start_d, end_d):
                                    try:
                                        # Ensure naive
                                        start_d = datetime(start_d.year, start_d.month, start_d.day)
                                        end_d = datetime(end_d.year, end_d.month, end_d.day)
                                        
                                        # Handle empty DF
                                        if bm_df.empty: return 0.0

                                        idx_s = bm_df.index.get_indexer([start_d], method='nearest')[0]
                                        idx_e = bm_df.index.get_indexer([end_d], method='nearest')[0]
                                        
                                        p_s = bm_df.iloc[idx_s]['CLOSE']
                                        p_e = bm_df.iloc[idx_e]['CLOSE']
                                        
                                        return (p_e - p_s) / p_s
                                    except:
                                        return 0.0

                                spy_bh = get_bh_return(spy_df, bt_start_date, bt_end_date)
                                qqq_bh = get_bh_return(qqq_df, bt_start_date, bt_end_date)

                                # --- PORTFOLIO FEASIBILITY CALCULATOR ---
                                with st.expander("🧮 Portfolio Feasibility Calculator", expanded=True):
                                    pf_col1, pf_col2 = st.columns(2)
                                    with pf_col1:
                                        pf_amount = st.number_input("Portfolio Size ($)", value=1000000, step=100000, format="%d")
                                    with pf_col2:
                                        # Dropdown for Benchmark
                                        bench_choice = st.selectbox("Target Benchmark", ["SPY", "QQQ"])
                                        # Select appropriate return
                                        selected_bm_ret = spy_bh if bench_choice == "SPY" else qqq_bh
                                        st.caption(f"Target to Beat: **{selected_bm_ret:.1%}** ({bench_choice} Buy & Hold)")

                                    strat_total_return_mult = df_res['BT_Return'].sum() # e.g., 368.17 = 36817%
                                    
                                    st.markdown("---")
                                    
                                    if strat_total_return_mult > 0 and max_active > 0:
                                        # 1. Calculate Target Profit (Matching the Index)
                                        target_profit_dollars = pf_amount * selected_bm_ret
                                        
                                        # 2. Calculate Required Unit Size to hit that target
                                        # Equation: Unit_Size * Strategy_Mult = Target_Profit
                                        req_unit_size = target_profit_dollars / strat_total_return_mult
                                        
                                        # 3. Calculate Peak Capital Usage
                                        req_peak_capital = req_unit_size * max_active
                                        
                                        # 4. Utilization
                                        utilization = req_peak_capital / pf_amount
                                        
                                        cal_c1, cal_c2 = st.columns(2)
                                        cal_c1.metric("Required Unit Size", f"${req_unit_size:,.0f}", help="Bet size needed per trade to generate enough profit to match the benchmark")
                                        cal_c2.metric("Peak Capital Utilization", f"{utilization:.1%}", help=f"You need ${req_peak_capital:,.0f} ({utilization:.1%}) of your portfolio to execute this. The rest sits in cash.")
                                        
                                        if req_peak_capital > pf_amount:
                                            shortfall = req_peak_capital - pf_amount
                                            st.error(f"❌ **Busted (Liquidity Crisis)**")
                                            st.markdown(f"To match {bench_choice}'s return, you need to bet **${req_unit_size:,.0f}** per trade.")
                                            st.markdown(f"However, with **{int(max_active)} active trades** at peak, you would need **${req_peak_capital:,.0f}**, which exceeds your portfolio.")
                                        else:
                                            st.success(f"✅ **Feasible & Efficient**")
                                            st.markdown(f"You can match the {bench_choice} return by allocating just **{utilization:.1%}** of your capital.")
                                            st.markdown(f"**The Advantage:** You achieve the same return as the index, but **{(1-utilization):.1%}** of your cash (${(pf_amount - req_peak_capital):,.0f}) remains risk-free in the bank.")

                                    else:
                                        if strat_total_return_mult <= 0:
                                            st.warning("Strategy has negative total return. Cannot beat a positive benchmark.")
                                        else:
                                            st.warning("No active trades detected.")

                                # --- TABLE 2: BENCHMARK COMPARISON ---
                                st.subheader("Annual Benchmark Comparison")
                                
                                with st.expander("ℹ️ Methodology"):
                                    st.markdown("""
                                    **"Apples-to-Apples" Comparison:**
                                    * **Same Dates:** The Index (SPY/QQQ) is "bought" on the exact same Entry Date and "sold" on the exact same Exit Date as the strategy signal.
                                    * **Same Position Size:** Returns are calculated assuming an equal dollar amount allocated to every trade.
                                    * **Effective ROI:** Calculates the return on the *actual capital required* to trade the strategy (accounting for peak congestion), rather than just summing up percentages.
                                    """)

                                # Add Year Column based on EXIT date
                                df_res['Exit_Year'] = pd.to_datetime(df_res['BT_Exit_Date']).dt.year
                                df_res['Start_Year'] = pd.to_datetime(df_res['BT_Entry_Date']).dt.year
                                
                                # Determine years from user selection
                                start_yr = bt_start_date.year
                                end_yr = bt_end_date.year
                                years = range(start_yr, end_yr + 1)
                                
                                bm_rows = []
                                
                                # Calculate Annual Rows
                                for y in years:
                                    # Trades ending in Year y (Realized P&L)
                                    ended_in_y = df_res[df_res['Exit_Year'] == y]
                                    started_in_y = df_res[df_res['Start_Year'] == y]
                                    
                                    strat_ret_sum = ended_in_y['BT_Return'].sum()
                                    spy_ret = ended_in_y['SPY_Ret'].sum()
                                    qqq_ret = ended_in_y['QQQ_Ret'].sum()
                                    
                                    # Calculate Max Active specifically for this year
                                    # Filter the global events_df for dates within year Y
                                    max_active_y = 0
                                    if not events_df.empty:
                                        # events_df index is DatetimeIndex
                                        events_y = events_df[events_df.index.year == y]
                                        if not events_y.empty:
                                            max_active_y = events_y['Active_Count'].max()
                                    
                                    # Effective ROI = Total Return / Max Active Units
                                    eff_roi = strat_ret_sum / max_active_y if max_active_y > 0 else 0.0
                                    
                                    # Adjust Benchmark Returns to Effective ROI as well
                                    eff_spy = spy_ret / max_active_y if max_active_y > 0 else 0.0
                                    eff_qqq = qqq_ret / max_active_y if max_active_y > 0 else 0.0
                                    
                                    bm_rows.append({
                                        "Year": str(y),
                                        "Effective ROI": eff_roi,
                                        "SPY Return": eff_spy,
                                        "QQQ Return": eff_qqq,
                                        "N_Start": len(started_in_y),
                                        "N_End": len(ended_in_y),
                                        "Max_Active": max_active_y
                                    })
                                
                                # Calculate Overall Row (Last)
                                ov_strat_sum = df_res['BT_Return'].sum()
                                ov_spy_sum = df_res['SPY_Ret'].sum()
                                ov_qqq_sum = df_res['QQQ_Ret'].sum()
                                
                                # Overall Max Active was calculated earlier as 'max_active' in the Results Summary section
                                ov_eff_roi = ov_strat_sum / max_active if max_active > 0 else 0.0
                                ov_eff_spy = ov_spy_sum / max_active if max_active > 0 else 0.0
                                ov_eff_qqq = ov_qqq_sum / max_active if max_active > 0 else 0.0
                                
                                bm_rows.append({
                                    "Year": "Overall",
                                    "Effective ROI": ov_eff_roi,
                                    "SPY Return": ov_eff_spy,
                                    "QQQ Return": ov_eff_qqq,
                                    "N_Start": len(df_res),
                                    "N_End": len(df_res),
                                    "Max_Active": max_active
                                })

                                bm_df = pd.DataFrame(bm_rows)
                                
                                def highlight_bm(row):
                                    styles = [''] * len(row)
                                    if row['Year'] == "Overall":
                                        return ['font-weight: bold; background-color: #f0f2f6; border-top: 2px solid #ccc'] * len(row)
                                    return styles

                                st.dataframe(
                                    bm_df.style
                                    .apply(highlight_bm, axis=1)
                                    .format({
                                        "Effective ROI": fmt_pct,
                                        "SPY Return": fmt_pct,
                                        "QQQ Return": fmt_pct,
                                        "N_Start": "{:,}",
                                        "N_End": "{:,}",
                                        "Max_Active": "{:,.0f}"
                                    }),
                                    hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        "N_Start": st.column_config.TextColumn("N Start"),
                                        "N_End": st.column_config.TextColumn("N End"),
                                        "Max_Active": st.column_config.TextColumn("Max Active", help="Peak number of simultaneous trades in this period"),
                                        "Effective ROI": st.column_config.TextColumn("Effective ROI", help="Total Strategy Return / Max Active Trades. Represents return on capital required.")
                                    }
                                )

                                # --- BUY & HOLD SECTION ---
                                st.markdown("##### 🏛️ Passive Buy & Hold (Full Period)")
                                bh_cols = st.columns(2)
                                
                                # (Calculated above for feasibility calc, reused here)
                                bh_cols[0].metric("SPY Buy & Hold", f"{spy_bh:.1%}")
                                bh_cols[1].metric("QQQ Buy & Hold", f"{qqq_bh:.1%}")
                                
                                st.caption(f"Methodology: Assumes a single buy on {bt_start_date.strftime('%b %d, %Y')} and sell on {bt_end_date.strftime('%b %d, %Y')}.")

                            # Invisible Buffer
                            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
                        
                    elif 'bt_results_df' in st.session_state:
                        st.warning("No signals found matching criteria in the backtest period.")

            except Exception as e: st.error(f"Analysis failed: {e}")

    # with tab_bot: (HIDDEN)
    if False:
        st.markdown('<div class="light-note" style="margin-bottom: 15px;">ℹ️ If this is buggy, just go back to the RSI Divergences tab and back here and it will work.</div>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ Page Notes: Backtester Logic"):
            st.markdown("""
            * **Data Source**: Unlike the Divergences and Percentile tabs (which use limited ~10yr history files), this tab pulls **Complete Price History** via Yahoo Finance or the full Ticker Map file.
            * **Methodology**: Calculates forward returns for all historical periods matching the criteria.
            * **Metrics**:
                * **Profit Factor**: Gross Wins / Gross Losses.
                * **Win Rate**: Percentage of trades that closed positive.
                * **EV**: Average Return % per trade.
            """)

        c_left, c_right = st.columns([1, 6])
        
        with c_left:
            ticker = st.text_input("Ticker", value="NFLX", help="Enter a symbol (e.g., TSLA, NVDA)", key="rsi_bt_ticker_input").strip().upper()
            lookback_years = st.number_input("Lookback Years", min_value=1, max_value=10, value=10)
            rsi_tol = st.number_input("RSI Tolerance", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            rsi_metric_container = st.empty()
        
        if ticker:
            ticker_map = load_ticker_map()
            
            with st.spinner(f"Crunching numbers for {ticker}..."):
                df = get_ticker_technicals(ticker, ticker_map)
                
                if df is None or df.empty:
                    df = fetch_yahoo_data(ticker)
                
                if df is None or df.empty:
                    st.error(f"Sorry, data could not be retrieved for {ticker} (neither via Drive nor Yahoo Finance).")
                else:
                    df.columns = [c.strip().upper() for c in df.columns]
                    
                    date_col = next((c for c in df.columns if 'DATE' in c), None)
                    close_col = next((c for c in df.columns if 'CLOSE' in c), None)
                    rsi_priority = ['RSI14', 'RSI', 'RSI_14']
                    rsi_col = next((c for c in rsi_priority if c in df.columns), None)
                    
                    if not rsi_col:
                        rsi_col = next((c for c in df.columns if 'RSI' in c and 'W_' not in c), None)

                    if not all([date_col, close_col]):
                        st.error("Data source missing Date or Close columns.")
                    else:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.sort_values(by=date_col).reset_index(drop=True)

                        if not rsi_col:
                            delta = df[close_col].diff()
                            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                            rs = gain / loss
                            df['RSI'] = 100 - (100 / (1 + rs))
                            rsi_col = 'RSI'

                        cutoff_date = df[date_col].max() - timedelta(days=365*lookback_years)
                        df = df[df[date_col] >= cutoff_date].copy().reset_index(drop=True) 

                        current_row = df.iloc[-1]
                        current_rsi = current_row[rsi_col]
                        
                        rsi_metric_container.markdown(f"""<div style="margin-top: 10px; font-size: 0.9rem; color: #666;">Current RSI</div><div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 15px;">{current_rsi:.2f}</div>""", unsafe_allow_html=True)
                        
                        rsi_min = current_rsi - rsi_tol
                        rsi_max = current_rsi + rsi_tol
                        
                        hist_df = df.iloc[:-1].copy()
                        matches = hist_df[(hist_df[rsi_col] >= rsi_min) & (hist_df[rsi_col] <= rsi_max)].copy()
                        
                        full_close = df[close_col].values
                        match_indices = matches.index.values
                        total_len = len(full_close)

                        results = []
                        periods = [1, 3, 5, 7, 10, 14, 30, 60, 90, 180]
                        
                        for p in periods:
                            valid_indices = match_indices[match_indices + p < total_len]
                            
                            if len(valid_indices) == 0:
                                results.append({"Days": p, "Win Rate": np.nan, "EV": np.nan, "Count": 0, "Profit Factor": np.nan})
                                continue
                                
                            entry_prices = full_close[valid_indices]
                            exit_prices = full_close[valid_indices + p]
                            
                            returns = (exit_prices - entry_prices) / entry_prices
                            
                            wins = returns[returns > 0]
                            losses = returns[returns < 0]
                            gross_win = np.sum(wins)
                            gross_loss = np.abs(np.sum(losses))
                            
                            pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
                            
                            win_rate = np.mean(returns > 0) * 100
                            avg_ret = np.mean(returns) * 100
                            
                            results.append({
                                "Days": p, 
                                "Profit Factor": pf, 
                                "Win Rate": win_rate, 
                                "EV": avg_ret, 
                                "Count": len(valid_indices)
                            })

                        res_df = pd.DataFrame(results)

                        with c_right:
                            if matches.empty:
                                st.warning(f"No historical periods found where RSI was between {rsi_min:.2f} and {rsi_max:.2f}.")
                            else:
                                def highlight_best(row):
                                    days = row['Days']
                                    if days <= 20: threshold = 30
                                    elif days <= 60: threshold = 20
                                    else: threshold = 10
                                    
                                    condition = (row['Count'] >= threshold) and (row['Win Rate'] > 75)
                                    color = 'background-color: rgba(144, 238, 144, 0.2)' if condition else ''
                                    return [color] * len(row)

                                def highlight_ret(val):
                                    if val is None or pd.isna(val): return ''
                                    if not isinstance(val, (int, float)): return ''
                                    color = '#71d28a' if val > 0 else '#f29ca0'
                                    return f'color: {color}; font-weight: bold;'
                                
                                format_func = lambda x: f"{x:+.2f}%" if pd.notnull(x) else "—"
                                format_wr = lambda x: f"{x:.1f}%" if pd.notnull(x) else "—"
                                format_pf = lambda x: f"{x:.2f}" if pd.notnull(x) else "—"

                                st.dataframe(
                                    res_df.style
                                    .format({"Win Rate": format_wr, "EV": format_func, "Profit Factor": format_pf})
                                    .map(highlight_ret, subset=["EV"])
                                    .apply(highlight_best, axis=1)
                                    .set_table_styles([dict(selector="th", props=[("font-weight", "bold"), ("background-color", "#f0f2f6")])]),
                                    use_container_width=False,
                                    column_config={
                                        "Days": st.column_config.NumberColumn("Days", width=60),
                                        "Profit Factor": st.column_config.NumberColumn("Profit Factor", width=80),
                                        "Win Rate": st.column_config.TextColumn("Win Rate", width=80),
                                        "EV": st.column_config.TextColumn("EV", width=80),
                                        "Count": st.column_config.NumberColumn("Count", width=60)
                                    },
                                    hide_index=True
                                )

                        st.markdown("<br><br><br>", unsafe_allow_html=True)
                
st.markdown("""<style>
.block-container{padding-top:3.5rem;padding-bottom:1rem;}
.zones-panel{padding:14px 0; border-radius:10px;}
.zone-row{display:flex; align-items:center; gap:10px; margin:8px 0;}
.zone-label{width:90px; font-weight:700; text-align:right; flex-shrink: 0; font-size: 13px;}
.zone-wrapper{
    flex-grow: 1; 
    position: relative; 
    height: 24px; 
    background-color: rgba(0,0,0,0.03);
    border-radius: 4px;
    overflow: hidden;
}
.zone-bar{
    position: absolute;
    left: 0; 
    top: 0; 
    bottom: 0; 
    z-index: 1;
    border-radius: 3px;
    opacity: 0.65;
}
.zone-bull{background-color: #71d28a;}
.zone-bear{background-color: #f29ca0;}
.zone-value{
    position: absolute;
    right: 8px;
    top: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    z-index: 2;
    font-size: 12px; 
    font-weight: 700;
    color: #1f1f1f;
    white-space: nowrap;
    text-shadow: 0 0 4px rgba(255,255,255,0.8);
}
.price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; }
.price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: #66b7ff; opacity: 0.4; }
.price-badge { background: rgba(102, 183, 255, 0.1); color: #66b7ff; border: 1px solid rgba(102, 183, 255, 0.5); border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; white-space: nowrap; margin: 0 12px; z-index: 1; }
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
.badge{background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.2); border-radius:18px; padding:6px 10px; font-weight:700}
.price-badge-header{background: rgba(102, 183, 255, 0.1); border: 1px solid #66b7ff; border-radius:18px; padding:6px 10px; font-weight:800}
.light-note { opacity: 0.7; font-size: 14px; margin-bottom: 10px; }

</style>""", unsafe_allow_html=True)

try:
    # Use empty dataframe as placeholder since db features were removed
    df_placeholder = pd.DataFrame()
    run_rsi_scanner_app(df_placeholder)
    
except Exception as e: 
    st.error(f"Error initializing dashboard: {e}")
