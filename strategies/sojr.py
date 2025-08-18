import numpy as np

def detect_sojr(row, params):
    # needs columns: in_open, OR_high, OR_low, OR_mid, prior_session_high/low, atr, ema_f, ema_s, open/high/low/close
    if not bool(row.get('in_open', False)): return None
    OR_h, OR_l, OR_mid = row['OR_high'], row['OR_low'], row['OR_mid']
    if np.isnan(OR_h) or np.isnan(OR_l): return None
    atrN = row['atr']

    body = abs(row['close']-row['open'])
    wu = row['high']-max(row['open'], row['close'])
    wd = min(row['open'], row['close'])-row['low']
    wbr_up = wu/max(body,1e-12); wbr_dn = wd/max(body,1e-12)

    # sweep high then re-entry
    if row['high'] > max(OR_h, row.get('prior_session_high', OR_h)) + params['sweep_atr_mult']*atrN and wbr_up >= params['wick_min']:
        if row['close'] < OR_h and row['ema_f'] < row['ema_s']:
            return {'side': -1, 'event': 'sojr_sweep_high_reversal_short'}
    # sweep low then re-entry
    if row['low'] < min(OR_l, row.get('prior_session_low', OR_l)) - params['sweep_atr_mult']*atrN and wbr_dn >= params['wick_min']:
        if row['close'] > OR_l and row['ema_f'] > row['ema_s']:
            return {'side': +1, 'event': 'sojr_sweep_low_reversal_long'}
    return None
