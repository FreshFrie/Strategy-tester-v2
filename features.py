import pandas as pd, numpy as np, yaml

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    tr = pd.concat([
        df['high']-df['low'],
        (df['high']-df['close'].shift()).abs(),
        (df['low']-df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def zscore(s, w=30):
    r = s.rolling(w)
    return (s - r.mean())/(r.std(ddof=0)+1e-12)

def load_sessions(path="configs/sessions.yaml"):
    with open(path, "r") as f: return yaml.safe_load(f)

def tag_sessions_utc_minus_4(df, sessions_cfg):
    df = df.copy()
    t = pd.to_datetime(df['time'])

    # Work in numeric minutes-since-midnight to avoid slow string operations
    minutes = t.dt.hour * 60 + t.dt.minute

    def parse_minute(hhmm: str) -> int:
        # expects 'HH:MM'
        parts = hhmm.split(":")
        return int(parts[0]) * 60 + int(parts[1])

    def in_window(start: str, end: str):
        smin = parse_minute(start)
        emin = parse_minute(end)
        return (minutes >= smin) & (minutes < emin)

    fx = (sessions_cfg or {}).get('fx', {})
    lon = (fx.get('london_open') or [{'start':'03:00','end':'04:00'}])[0]
    ny  = (fx.get('ny_open')     or [{'start':'08:00','end':'09:00'}])[0]
    ovl = (fx.get('overlap')     or [{'start':'08:00','end':'11:00'}])[0]

    df['in_london_open'] = in_window(lon['start'], lon['end'])
    df['in_ny_open']     = in_window(ny['start'],  ny['end'])
    df['in_overlap']     = in_window(ovl['start'], ovl['end'])
    return df

def compute_opening_range(df, minutes=12, flag_col='in_london_open'):
    # for each day, build OR only for bars within the first `minutes` of the flagged window start
    df = df.copy()
    t = pd.to_datetime(df['time'])
    df['date'] = t.dt.date
    df['minute'] = t.dt.minute
    df['OR_high'] = np.nan; df['OR_low'] = np.nan; df['OR_mid'] = np.nan; df['in_open'] = False
    # find window starts per day
    for d, g in df.groupby('date'):
        idx = g.index[g[flag_col]].tolist()
        if not idx: continue
        # window start is first bar with flag true
        start_i = idx[0]
        start_time = t.loc[start_i]
        mask = (t>=start_time) & (t< start_time + pd.Timedelta(minutes=minutes))
        or_high = df.loc[mask, 'high'].max()
        or_low  = df.loc[mask, 'low'].min()
        df.loc[mask, 'OR_high'] = or_high
        df.loc[mask, 'OR_low']  = or_low
        df.loc[mask, 'OR_mid']  = (or_high+or_low)/2.0
        df.loc[mask, 'in_open'] = True
        # prior session highs/lows (yesterday’s open window)
        prev = df[(df['date']<d) & (df['in_open'])]
        if not prev.empty:
            df.loc[mask, 'prior_session_high'] = prev['OR_high'].dropna().iloc[-1]
            df.loc[mask, 'prior_session_low']  = prev['OR_low'].dropna().iloc[-1]
    return df

def add_core_features(df, params, sessions_cfg):
    df = df.copy()
    df['atr']   = atr(df, n=params['ATR_N'])
    df['vol_z'] = zscore(df['volume'], w=params.get('vol_z_period',30))
    df['ema_f'] = ema(df['close'], params['EMA_fast'])
    df['ema_s'] = ema(df['close'], params['EMA_slow'])
    df = tag_sessions_utc_minus_4(df, sessions_cfg)
    df = compute_opening_range(df, minutes=params.get('or_window_min',12), flag_col='in_london_open')
    return df
