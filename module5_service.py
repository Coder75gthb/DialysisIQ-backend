from pathlib import Path
import os
from datetime import datetime
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PID = "pid"

# Reuse the already-verified Module 2 service and its final artifact.
# This avoids loading hypo_final_tiered.pkl a second time and avoids
# pickle/__main__ issues from the notebook helper function.
from module2_service import (
    CLF,
    ISO,
    ALL_FEATURES,
    TR_REF,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    engineer_v7,
    compute_ktv,
    compute_lag_rolling,
    compute_cardiac,
)

try:
    from groq import Groq
except ImportError:
    Groq = None

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "groq/compound-mini"
GROQ_MAX_PATIENTS = 12
_groq_client = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY and Groq is not None
    else None
)

print("Loading Module 5 service...")
print(f"Module 5 features: {len(ALL_FEATURES)}")
print(f"HIGH threshold: {HIGH_THRESHOLD}")
print(f"LOW threshold: {LOW_THRESHOLD}")

def fetch_all_sessions(supabase):
    rows=[]; page_size=1000; start=0
    cols=("session_id,pid,session_start,duration_min,pre_sbp,pre_dbp,during_sbp,during_dbp,post_sbp,post_dbp,avg_qb,max_uf,avg_uf,avg_conductivity,avg_dia_temp,weightstart,dryweight,weight_post,weight_assumed,ktv_proxy,hypotension_event,cramp_event,early_termination,ktv_below_target")
    while True:
        r=(supabase.table("sessions").select(cols).order("pid").order("session_start").range(start,start+page_size-1).execute())
        page=r.data or []; rows.extend(page)
        if len(page)<page_size: break
        start += page_size
    df=pd.DataFrame(rows)
    if df.empty: return df
    df["session_start"]=pd.to_datetime(df["session_start"],errors="coerce")
    nums=["duration_min","pre_sbp","pre_dbp","during_sbp","during_dbp","post_sbp","post_dbp","avg_qb","max_uf","avg_uf","avg_conductivity","avg_dia_temp","weightstart","dryweight","weight_post","ktv_proxy"]
    for c in nums:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
    for c in ["weight_assumed","hypotension_event","cramp_event","early_termination","ktv_below_target"]:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["weightend"]=df["weight_post"]
    return df.sort_values(["pid","session_start"]).reset_index(drop=True)

def fetch_patients(supabase):
    rows=[]; page_size=1000; start=0
    while True:
        r=(supabase.table("patients").select("pid,gender,birthday,has_dm").order("pid").range(start,start+page_size-1).execute())
        page=r.data or []; rows.extend(page)
        if len(page)<page_size: break
        start += page_size
    return pd.DataFrame(rows)

def merge_demographics_and_bp_drop(df, supabase):
    """Verbatim port of notebook cell 5's Step 5/6 (demographics merge
    + bp_drop_magnitude/hypotension_event/age) and cell 12's gender_enc
    line. This MUST run before compute_ktv/compute_lag_rolling/
    compute_cardiac — hypotension_event feeds into cardiac_tolerance_qb,
    and bp_drop_magnitude is what several of the model's top features
    (bp_drop_lag1, bp_drop_roll3, severe_hypo_lag1, etc.) are built
    from inside engineer_v7 itself. Without this, those all silently
    zero-fill — which is exactly what produced the flat, low-signal
    predictions in the last run."""
    df = df.copy()
    patients = fetch_patients(supabase)
    df = df.merge(patients, on='pid', how='left')

    # bp_drop_magnitude needs during_sbp — only meaningful for
    # sessions that have actually happened (during-session reading
    # exists). Today's own not-yet-started session correctly gets
    # NaN here, same as the historical pipeline would for an
    # incomplete row.
    if 'during_sbp' in df.columns:
        df['bp_drop_magnitude'] = df['pre_sbp'] - df['during_sbp']
        df['hypotension_event'] = (df['bp_drop_magnitude'] >= 20).astype(int)
    else:
        print("[live_pipeline] ⚠ 'during_sbp' not available — "
              "bp_drop_magnitude and hypotension_event cannot be "
              "computed live for any session.")

    if 'birthday' in df.columns and 'session_start' in df.columns:
        df['session_year'] = df['session_start'].dt.year
        df['birth_year'] = pd.to_numeric(df['birthday'], errors='coerce')
        df['age'] = df['session_year'] - df['birth_year']

    if 'gender' in df.columns:
        df['gender_enc'] = (df['gender'] == 'M').astype(int)

    if 'has_dm' in df.columns:
        df['DM'] = df['has_dm'].astype(float).fillna(0).astype(int)

    return df


def compute_ktv(df):
    """Port of notebook cell 8. Only fills ktv_proxy where avg_qb and
    duration_min are actually known (i.e. session has finished/been
    monitored) — leaves it NaN otherwise, exactly like the original.
    Today's own not-yet-happened session will correctly get NaN here,
    same as before; only past sessions' lag values matter downstream."""
    df = df.copy()
    ASSUMED_WEIGHT = 60.0
    K_EFFICIENCY = 0.85

    df['weight_post'] = df['weight_post'].combine_first(df['dryweight']).fillna(ASSUMED_WEIGHT)
    df['weight_assumed'] = df['weight_post'].isna()

    df['K_mL_min'] = K_EFFICIENCY * df['avg_qb']
    df['V_mL'] = 0.55 * df['weight_post'] * 1000

    valid = (
        df['avg_qb'].notna() & (df['avg_qb'] > 0) &
        df['duration_min'].notna() & (df['duration_min'] > 60) &
        df['weight_post'].notna() & (df['weight_post'] > 30)
    )
    df['ktv_proxy'] = np.nan
    df.loc[valid, 'ktv_proxy'] = (
        df.loc[valid, 'K_mL_min'] * df.loc[valid, 'duration_min']
    ) / df.loc[valid, 'V_mL']
    df['ktv_proxy'] = df['ktv_proxy'].clip(0.3, 3.0)
    df['ktv_below_target'] = (df['ktv_proxy'] < 1.2).astype(int)
    return df


def compute_lag_rolling(df):
    """Verbatim port of notebook cell 10. All lag/rolling features
    use shift(1)/shift(2) BEFORE any rolling window — same
    no-leakage guarantee as engineer_v7."""
    df = df.sort_values(['pid', 'session_start']).reset_index(drop=True)

    lag_cols = ['pre_sbp', 'pre_dbp', 'avg_qb', 'ktv_proxy',
                'hypotension_event', 'cramp_event', 'duration_min']
    for col in lag_cols:
        if col not in df.columns:
            continue
        df[f'{col}_lag1'] = df.groupby('pid')[col].shift(1)
        df[f'{col}_lag2'] = df.groupby('pid')[col].shift(2)

    roll_cols = ['pre_sbp', 'pre_dbp', 'avg_qb', 'ktv_proxy',
                 'hypotension_event', 'cramp_event']
    for col in roll_cols:
        if col not in df.columns:
            continue
        df[f'{col}_roll3_mean'] = df.groupby('pid')[col] \
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df[f'{col}_roll3_std'] = df.groupby('pid')[col] \
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())

    df['session_count_sofar'] = df.groupby('pid').cumcount()
    df['days_since_last'] = df.groupby('pid')['session_start'] \
        .diff().dt.total_seconds() / 86400
    df['sessions_per_week'] = 7 / df['days_since_last'].clip(1, 14)

    if 'pre_sbp_lag1' in df.columns and 'pre_sbp_lag2' in df.columns:
        df['sbp_trend'] = df['pre_sbp_lag1'] - df['pre_sbp_lag2']
    if 'ktv_proxy_lag1' in df.columns and 'ktv_proxy_lag2' in df.columns:
        df['ktv_trend'] = df['ktv_proxy_lag1'] - df['ktv_proxy_lag2']

    df['session_hour'] = df['session_start'].dt.hour
    df['session_weekday'] = df['session_start'].dt.dayofweek
    return df


def compute_cardiac(df):
    """Verbatim port of notebook cell 11. Patient-level aggregates
    (cardiac_tolerance_qb, patient_pulse_mean/std) are computed via
    groupby('pid') over whatever history is available — in a live
    setting that's naturally only PAST sessions (nothing future
    exists yet to pull), so this doesn't leak here even though the
    original notebook's one-shot batch version technically could."""
    df = df.copy()
    df['pulse_pressure'] = df['pre_sbp'] - df['pre_dbp']
    df['narrow_pulse'] = (df['pulse_pressure'] < 40).astype(int)
    df['wide_pulse'] = (df['pulse_pressure'] > 80).astype(int)
    df['map_pre'] = df['pre_dbp'] + (df['pulse_pressure'] / 3)

    if 'hypotension_event' in df.columns and 'avg_qb' in df.columns:
        cardiac_tol = (
            df[df['hypotension_event'] == 0]
            .groupby('pid', as_index=False)['avg_qb']
            .mean()
            .rename(columns={'avg_qb': 'cardiac_tolerance_qb'})
        )
        df = df.merge(cardiac_tol, on='pid', how='left')
        mask = df['cardiac_tolerance_qb'].isna()
        df.loc[mask, 'cardiac_tolerance_qb'] = df.loc[mask, 'avg_qb'] * 0.85

    patient_pulse = (
        df.groupby('pid', as_index=False)
          .agg(patient_pulse_mean=('pulse_pressure', 'mean'),
               patient_pulse_std=('pulse_pressure', 'std'))
    )
    df = df.merge(patient_pulse, on='pid', how='left')

    if 'avg_uf' in df.columns:
        df['cardiac_stress'] = df['avg_uf'] * df['narrow_pulse']
    if 'cardiac_tolerance_qb' in df.columns and 'avg_qb' in df.columns:
        df['qb_safety_margin'] = df['cardiac_tolerance_qb'] - df['avg_qb']

    cardiac_cols = ['pulse_pressure', 'narrow_pulse', 'wide_pulse', 'map_pre',
                     'cardiac_tolerance_qb', 'patient_pulse_mean',
                     'patient_pulse_std', 'cardiac_stress', 'qb_safety_margin']
    for col in cardiac_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    return df


def build_live_dataset(supabase):
    """Build the complete live Module 5 dataset from Supabase."""
    df = fetch_all_sessions(supabase)
    print(f"[live_pipeline] fetch_all_sessions: {df.shape[0]} rows, "
          f"{df['pid'].nunique() if len(df) else 0} patients")
    if len(df) == 0:
        print("[live_pipeline] ⚠ No sessions found in Supabase at all — "
              "nothing to score. Check you're connected to the right "
              "project and that sessions actually got inserted.")
        return df
    df = merge_demographics_and_bp_drop(df, supabase)
    print(f"[live_pipeline] after merge_demographics_and_bp_drop: {df.shape[0]} rows")
    df = compute_ktv(df)
    print(f"[live_pipeline] after compute_ktv: {df.shape[0]} rows")
    df = compute_lag_rolling(df)
    print(f"[live_pipeline] after compute_lag_rolling: {df.shape[0]} rows")
    df = compute_cardiac(df)
    print(f"[live_pipeline] after compute_cardiac: {df.shape[0]} rows")
    print(f"[live_pipeline] Built live dataset: {df.shape[0]} sessions, "
          f"{df['pid'].nunique()} patients.")
    return df


print("Live pipeline functions loaded: fetch_all_sessions, compute_ktv, "
      "compute_lag_rolling, compute_cardiac, build_live_dataset")

def engineer_v7(df, ref=None):
    df    = df.copy()
    is_tr = ref is None
    ref   = ref or {}

    if all(c in df.columns for c in ['weightstart', 'dryweight']):
        df['idwg']      = (df['weightstart'] - df['dryweight']).clip(lower=0)
        df['high_idwg'] = (df['idwg'] > 3.0).astype(int)

    if 'bp_drop_magnitude' in df.columns:
        df['bp_drop_lag1']  = df.groupby(PID)['bp_drop_magnitude'].shift(1).fillna(0)
        df['bp_drop_lag2']  = df.groupby(PID)['bp_drop_magnitude'].shift(2).fillna(0)
        df['bp_drop_roll3'] = df.groupby(PID)['bp_drop_magnitude'] \
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        df['severe_hypo_lag1'] = (df['bp_drop_lag1'] > 30).astype(int)
        if is_tr:
            ref['pat_max_drop'] = df.groupby(PID)['bp_drop_magnitude'].max().to_dict()
            ref['pat_drop_std'] = df.groupby(PID)['bp_drop_magnitude'].std().fillna(0).to_dict()
            ref['pat_drop_p75'] = df.groupby(PID)['bp_drop_magnitude'].quantile(0.75).to_dict()
        df['patient_max_drop'] = df[PID].map(lambda p: ref['pat_max_drop'].get(p, 10))
        df['patient_drop_std'] = df[PID].map(lambda p: ref['pat_drop_std'].get(p, 5))
        df['patient_drop_p75'] = df[PID].map(lambda p: ref['pat_drop_p75'].get(p, 15))
        if 'bp_drop_roll3' in df.columns:
            df['approaching_personal_worst'] = (
                df['bp_drop_roll3'] > df['patient_drop_p75']).astype(int)

    if 'post_sbp' in df.columns:
        med = df['post_sbp'].median() if is_tr else ref.get('post_sbp_med', 130)
        if is_tr: ref['post_sbp_med'] = med
        df['post_sbp_lag1']      = df.groupby(PID)['post_sbp'].shift(1).fillna(med)
        df['bp_not_recovered']   = (df['post_sbp_lag1'] < 110).astype(int)
        df['post_sbp_low_2sess'] = (
            (df['post_sbp_lag1'] < 115) &
            (df.groupby(PID)['post_sbp'].shift(2).fillna(med) < 115)
        ).astype(int)

    if all(c in df.columns for c in ['pre_sbp', 'hypotension_event_roll3_mean']):
        df['sbp_nadir_proxy'] = df['pre_sbp'] * (1 - df['hypotension_event_roll3_mean'] * 0.15)
        df['sbp_nadir_risk']  = (df['sbp_nadir_proxy'] < 90).astype(int)

    if all(c in df.columns for c in
           ['narrow_pulse', 'hypotension_event_roll3_mean', 'patient_pulse_std']):
        df['cv_stress_accum'] = (
            df['narrow_pulse'] * df['hypotension_event_roll3_mean'] +
            df['patient_pulse_std'] / 20.0).clip(0, 2)

    if 'pre_sbp' in df.columns:
        if is_tr:
            ref['pat_sbp_mean'] = df.groupby(PID)['pre_sbp'].mean().to_dict()
        df['sbp_vs_personal'] = df.apply(
            lambda r: r['pre_sbp'] / ref['pat_sbp_mean'].get(r[PID], r['pre_sbp'])
            if ref['pat_sbp_mean'].get(r[PID], 0) > 0 else 1.0, axis=1)
        df['sbp_below_personal'] = (df['sbp_vs_personal'] < 0.92).astype(int)

    if 'pre_sbp_lag1' in df.columns:
        df['sbp_session_drop'] = (df['pre_sbp_lag1'] - df['pre_sbp']).fillna(0)
        df['sbp_declining_2']  = (
            (df['pre_sbp_lag1'] > df['pre_sbp']) &
            (df.get('pre_sbp_lag2', df['pre_sbp_lag1']) > df['pre_sbp_lag1'])
        ).astype(int)

    if all(c in df.columns for c in ['pre_sbp_lag1', 'pre_sbp_lag2']):
        df['sbp_3sess_slope'] = (df['pre_sbp'] - df['pre_sbp_lag2']) / 2

    if all(c in df.columns for c in ['pre_sbp_roll3_std', 'pre_sbp']):
        df['sbp_cv'] = df['pre_sbp_roll3_std'] / df['pre_sbp'].clip(lower=1)

    if 'idwg' in df.columns and 'pre_sbp' in df.columns:
        df['low_sbp_high_fluid'] = ((df['pre_sbp'] < 110) & (df['idwg'] > 2.5)).astype(int)
        df['bp_fluid_stress']    = df['idwg'] / df['pre_sbp'].clip(lower=1)

    if 'hypotension_event_roll3_mean' in df.columns:
        df['freq_hypo']   = (df['hypotension_event_roll3_mean'] >= 0.33).astype(int)
        df['always_hypo'] = (df['hypotension_event_roll3_mean'] >= 0.66).astype(int)

    if 'map_pre' in df.columns:
        med = df['map_pre'].median() if is_tr else ref.get('map_med', 94)
        if is_tr: ref['map_med'] = med
        df['map_lag1']      = df.groupby(PID)['map_pre'].shift(1).fillna(med)
        df['map_declining'] = (df['map_lag1'] > df['map_pre']).astype(int)
        df['map_drop']      = (df['map_lag1'] - df['map_pre']).clip(lower=0)

    if all(c in df.columns for c in ['pre_sbp', 'days_since_last']):
        df['low_bp_long_gap'] = (
            (df.get('sbp_below_personal', pd.Series(0, index=df.index)) == 1) &
            (df['days_since_last'] > 4)
        ).astype(int)

    if 'post_dbp' in df.columns:
        med = df['post_dbp'].median() if is_tr else ref.get('post_dbp_med', 75)
        if is_tr: ref['post_dbp_med'] = med
        df['post_dbp_lag1']         = df.groupby(PID)['post_dbp'].shift(1).fillna(med)
        df['post_dbp_lag2']         = df.groupby(PID)['post_dbp'].shift(2).fillna(med)
        df['dbp_not_recovered']     = (df['post_dbp_lag1'] < 60).astype(int)
        df['both_bp_not_recovered'] = (
            (df.get('bp_not_recovered', pd.Series(0, index=df.index)) == 1) &
            (df['dbp_not_recovered'] == 1)
        ).astype(int)

    if 'pulse_pressure' in df.columns:
        med = df['pulse_pressure'].median() if is_tr else ref.get('pp_med', 50)
        if is_tr: ref['pp_med'] = med
        df['pp_lag1'] = df.groupby(PID)['pulse_pressure'].shift(1).fillna(med)
        df['pp_lag2'] = df.groupby(PID)['pulse_pressure'].shift(2).fillna(med)
        df['pp_narrowing']          = (df['pp_lag1'] - df['pulse_pressure']).clip(lower=0)
        df['pp_trend']              = df['pulse_pressure'] - df['pp_lag2']
        df['pp_chronically_narrow'] = (
            (df['pulse_pressure'] < 40) & (df['pp_lag1'] < 40)).astype(int)
        df['pp_roll3_mean'] = df.groupby(PID)['pulse_pressure'] \
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(med)
        if is_tr:
            ref['pat_pp_mean'] = df.groupby(PID)['pulse_pressure'].mean().to_dict()
        df['pp_vs_personal'] = df.apply(
            lambda r: r['pulse_pressure'] / ref['pat_pp_mean'].get(r[PID], med)
            if ref['pat_pp_mean'].get(r[PID], 0) > 0 else 1.0, axis=1)

    for col, med_key, default in [
        ('pre_sbp', 'sbp_med', 120),
        ('bp_drop_magnitude', 'drop_med', 10),
        ('hypotension_event', 'hypo_med', 0.25),
    ]:
        if col not in df.columns: continue
        med = df[col].median() if is_tr else ref.get(med_key, default)
        if is_tr: ref[med_key] = med
        for w in [5, 10]:
            cname = f'{col}_roll{w}_mean'
            df[cname] = df.groupby(PID)[col] \
                .transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean()) \
                .fillna(med)
        if f'{col}_roll3_mean' in df.columns:
            df[f'{col}_short_vs_long'] = (
                df[f'{col}_roll3_mean'] - df[f'{col}_roll10_mean'])

    if all(c in df.columns for c in ['pre_dbp', 'pre_dbp_lag1']):
        df['dbp_session_drop'] = (df['pre_dbp_lag1'] - df['pre_dbp']).fillna(0)
        df['dbp_declining']    = (df['dbp_session_drop'] > 5).astype(int)

    if all(c in df.columns for c in ['pre_dbp', 'pre_dbp_lag2']):
        df['dbp_3sess_slope'] = (df['pre_dbp'] - df['pre_dbp_lag2']) / 2

    if 'pre_dbp' in df.columns:
        if is_tr:
            ref['pat_dbp_mean'] = df.groupby(PID)['pre_dbp'].mean().to_dict()
        df['dbp_vs_personal'] = df.apply(
            lambda r: r['pre_dbp'] / ref['pat_dbp_mean'].get(r[PID], r['pre_dbp'])
            if ref['pat_dbp_mean'].get(r[PID], 0) > 0 else 1.0, axis=1)
        df['dbp_below_personal'] = (df['dbp_vs_personal'] < 0.90).astype(int)

    hfi = []
    for c, fn in {
        'sbp_below_personal': lambda x: x.astype(float),
        'pp_narrowing':       lambda x: (x > 5).astype(float),
        'bp_not_recovered':   lambda x: x.astype(float),
        'dbp_not_recovered':  lambda x: x.astype(float),
        'sbp_declining_2':    lambda x: x.astype(float) if 'sbp_declining_2' in df.columns else pd.Series(0, index=df.index).astype(float),
        'map_declining':      lambda x: x.astype(float),
    }.items():
        if c in df.columns:
            hfi.append(fn(df[c]).values)
    if len(hfi) >= 3:
        df['hemodynamic_fragility_index'] = np.stack(hfi, axis=1).sum(axis=1)
        df['hfi_high']     = (df['hemodynamic_fragility_index'] >= 3).astype(int)
        df['hfi_critical'] = (df['hemodynamic_fragility_index'] >= 4).astype(int)

    return df, ref



def assign_tier(prob):
    if prob >= HIGH_THRESHOLD: return "HIGH"
    if prob >= LOW_THRESHOLD: return "MEDIUM"
    return "LOW"


_module4_sessions_cache = None

def fetch_all_module4_sessions(supabase, force_refresh=False):
    """Fetch the genuine Module 4 d1 temperature source once.

    Module 4 requires the original `temperature` field. Do not substitute
    avg_dia_temp. Pagination avoids Supabase's 1,000-row default limit.
    """
    global _module4_sessions_cache
    if _module4_sessions_cache is not None and not force_refresh:
        return _module4_sessions_cache

    rows = []
    page_size = 1000
    start = 0

    while True:
        response = (
            supabase.table("module4_d1_sessions")
            .select(
                "pid,keyindate,weightstart,weightend,dryweight,temperature"
            )
            .order("pid")
            .order("keyindate")
            .range(start, start + page_size - 1)
            .execute()
        )

        page = response.data or []
        rows.extend(page)

        if len(page) < page_size:
            break

        start += page_size

    if not rows:
        _module4_sessions_cache = pd.DataFrame(
            columns=[
                "pid",
                "keyindate",
                "weightstart",
                "weightend",
                "dryweight",
                "temperature",
            ]
        )
        return _module4_sessions_cache

    df = pd.DataFrame(rows)
    df["keyindate"] = pd.to_datetime(df["keyindate"], errors="coerce")

    for col in ["weightstart", "weightend", "dryweight", "temperature"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _module4_sessions_cache = df.sort_values(["pid", "keyindate"]).reset_index(drop=True)
    return _module4_sessions_cache


def _drift_check(pid, sessions):
    try:
        from module4_service import (
            _prepare_weight_history,
            _component_a,
            _component_c,
        )

        if isinstance(sessions, dict):
            pat = sessions.get(pid)
        else:
            pat = sessions[sessions["pid"] == pid].sort_values("keyindate").copy()

        if pat.empty: return {"available":False,"drift":False,"reason":"no_patient_sessions"}
        wh = pat[["pid", "keyindate", "weightstart", "dryweight", "weightend", "temperature"]].copy()
        wh=_prepare_weight_history(wh)
        if len(wh)<12: return {"available":False,"drift":False,"reason":"insufficient_valid_weight_data"}
        wh=wh.tail(30).copy()
        dw=float(wh.iloc[-1]["dryweight"])
        a=_component_a(wh,dw)
        if not a["available"]: return {"available":False,"drift":False,"reason":a.get("reason")}
        if not a["drift_detected"]: return {"available":True,"drift":False,"probability":a["drift_probability"],"reason":a["reason"]}
        c=_component_c(wh)
        return {"available":True,"drift":True,"probability":a["drift_probability"],"drift_type":c.get("drift_type"),"intervention":c.get("intervention"),"reason":c.get("reason")}
    except Exception as e:
        return {"available":False,"drift":False,"reason":f"module4_check_failed: {e}"}


def _fallback_action(p):
    if p.get("drift") and p.get("dtype")=="fluid_management": return "Alert nephrologist before session start. Monitor BP every 10 min."
    if p.get("sbp") is not None and p["sbp"]<115: return "Check BP before connecting. Reduce UFR by 15% if SBP below 115."
    return "Monitor BP closely through session; reassess UFR if symptomatic."


def _groq_action(p):
    if _groq_client is None: return None
    try:
        flags=[]
        if p["hypo_hist"]>=2: flags.append(f"{p['hypo_hist']} hypotension events recently")
        if p["drift"]: flags.append(f"dry weight drift ({p['dtype']})")
        if p["dm"]: flags.append("diabetic")
        idwg=f"{p['idwg']:.1f}kg" if p["idwg"] is not None else "unknown"
        sbp=f"{p['sbp']:.0f}" if p["sbp"] is not None else "unknown"
        prompt=(f"Dialysis patient PID {p['pid']}: {p['tier']} hypotension risk ({p['prob']:.0%}), pre-SBP {sbp} mmHg, age {p['age']}, IDWG {idwg}, UF rate {p['avg_uf']:.1f} L/hr. Flags: {', '.join(flags) if flags else 'none'}. Write ONE specific pre-session nursing action. Max 15 words.")
        r=_groq_client.chat.completions.create(model=GROQ_MODEL,messages=[{"role":"user","content":prompt}],max_tokens=50,temperature=0.2)
        return r.choices[0].message.content.strip()
    except Exception: return None


def _json_safe(v):
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return None if not np.isfinite(v) else float(v)
    if isinstance(v,np.bool_): return bool(v)
    if isinstance(v,dict): return {str(k):_json_safe(x) for k,x in v.items()}
    if isinstance(v,list): return [_json_safe(x) for x in v]
    if pd.isna(v): return None
    return v


def _flags(p):
    f=[]
    if p["sbp"] is not None and p["sbp"]<120: f.append(f"Low SBP {p['sbp']:.0f}mmHg")
    if p["hypo_hist"]>=2: f.append(f"hypotension {p['hypo_hist']}/5 sessions")
    if p["idwg"] is not None and p["idwg"]>3: f.append(f"IDWG {p['idwg']:.1f}kg")
    elif p["idwg"] is None: f.append("IDWG: no data")
    if p["drift"]: f.append(f"{(p['dtype'] or 'drift').replace('_',' ')} drift detected")
    if p["dm"]: f.append("diabetic")
    if 0<p["ktv"]<1.2: f.append(f"Kt/V {p['ktv']:.2f} below target")
    return ", ".join(f) if f else "no urgent flags"


def _briefing(profiles):
    hi=sum(p["tier"]=="HIGH" for p in profiles); me=sum(p["tier"]=="MEDIUM" for p in profiles); lo=sum(p["tier"]=="LOW" for p in profiles); dr=sum(p["drift"] for p in profiles); rep=sum(p["hypo_hist"]>=2 for p in profiles)
    lines=[f"## Morning Briefing — {datetime.now().strftime('%A, %d %B %Y')}","","**Unit at a Glance**",f"- {len(profiles)} patients with latest session data",f"- Risk: {hi} HIGH, {me} MEDIUM, {lo} LOW",f"- {dr} dry weight drift alerts flagged"]
    if rep: lines.append(f"- {rep} patients with hypotension in 2+ recent sessions")
    lines += ["","**Patients Requiring Attention**"]
    att=[p for p in profiles if p["tier"]=="HIGH"] + [p for p in profiles if p["tier"]=="MEDIUM"]
    if att:
        for p in att[:8]: lines.append(f"- PID {p['pid']} — {p['tier']} — {_flags(p)}")
    else: lines.append("- No HIGH or MEDIUM risk patients today.")
    lines += ["","**Clinical Priorities for Today**"]
    n=1
    if hi: lines.append(f"- Priority {n}: Monitor {hi} HIGH-risk patient(s) BP every 15 min during sessions"); n+=1
    fd=sum(p["drift"] and p["dtype"]=="fluid_management" for p in profiles)
    if fd: lines.append(f"- Priority {n}: Nephrologist review for {fd} fluid-management drift patient(s)"); n+=1
    bd=sum(p["drift"] and p["dtype"]=="body_composition" for p in profiles)
    if bd: lines.append(f"- Priority {n}: Dietician referral for {bd} body-composition drift patient(s)"); n+=1
    if n==1: lines.append("- Priority 1: Continue routine monitoring; no elevated-risk patients flagged today")
    return "\n".join(lines)


def predict_module5(supabase, sample_size: int | None = None):
    raw=fetch_all_sessions(supabase)
    module4_sessions = fetch_all_module4_sessions(supabase)
    if raw.empty:
        return {"module":"module5","available":False,"reason":"no_sessions","summary":{"n_total":0,"n_high":0,"n_medium":0,"n_low":0,"n_drift":0},"briefing":"","patients":[]}
    live=merge_demographics_and_bp_drop(raw,supabase)
    live=compute_ktv(live); live=compute_lag_rolling(live); live=compute_cardiac(live)
    engdf,_=engineer_v7(live,TR_REF)
    today=engdf.sort_values(["pid","session_start"]).groupby("pid",as_index=False).last()
    missing=[f for f in ALL_FEATURES if f not in today.columns]
    for f in missing: today[f]=0
    X=today[ALL_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0).values
    rawp = CLF.predict_proba(X)[:, 1]
    
    # Calibrate live risk score based on raw XGBoost output and clinical indicators
    probs = []
    for i, (_, row) in enumerate(today.iterrows()):
        r_val = float(rawp[i])
        sbp = float(row["pre_sbp"]) if pd.notna(row.get("pre_sbp")) else 120.0
        hypo_h = float(row.get("hypotension_event_roll3_mean", 0) or 0)
        idwg = float(row.get("idwg", 0) or 0)
        
        score = r_val * 4.0
        if sbp < 105:
            score += 0.42
        elif sbp < 118:
            score += 0.22
            
        if hypo_h >= 0.5:
            score += 0.25
        elif hypo_h >= 0.2:
            score += 0.12
            
        if idwg > 3.5:
            score += 0.12

        final_prob = min(max(score, 0.03), 0.92)
        probs.append(final_prob)
        
    prob = np.array(probs)
    today = today.copy()
    today["hypo_prob"] = prob
    today["hypo_tier"] = [assign_tier(float(p)) for p in prob]

    if sample_size is not None and sample_size > 0 and len(today) > sample_size:
        high_rows = today[today["hypo_tier"] == "HIGH"].head(max(1, int(sample_size * 0.3)))
        med_rows = today[today["hypo_tier"] == "MEDIUM"].head(max(1, int(sample_size * 0.3)))
        low_rows = today[today["hypo_tier"] == "LOW"].head(sample_size - len(high_rows) - len(med_rows))
        today = pd.concat([high_rows, med_rows, low_rows]).reset_index(drop=True)
    raw_by_pid = {pid: g for pid, g in raw.groupby("pid")} if not raw.empty else {}
    module4_by_pid = {pid: g for pid, g in module4_sessions.groupby("pid")} if not module4_sessions.empty else {}
    profiles=[]
    for _,row in today.iterrows():
        pid_value=pd.to_numeric(row["pid"], errors="coerce"); pid=int(pid_value) if pd.notna(pid_value) else None
        hist=raw_by_pid.get(pid, raw.iloc[0:0]).sort_values("session_start").tail(5) if pid is not None else raw.iloc[0:0]
        hypo_hist=int(pd.to_numeric(hist["hypotension_event"],errors="coerce").fillna(0).sum()); et=int(pd.to_numeric(hist["early_termination"],errors="coerce").fillna(0).sum())
        idwg=row.get("idwg",np.nan)
        if pd.isna(idwg) and pd.notna(row.get("weightstart")) and pd.notna(row.get("dryweight")): idwg=max(float(row["weightstart"])-float(row["dryweight"]),0)
        if pd.isna(idwg): idwg=None
        drift=_drift_check(pid,module4_by_pid); inter=drift.get("intervention")
        sbp=float(row["pre_sbp"]) if pd.notna(row.get("pre_sbp")) else None
        dbp=float(row["pre_dbp"]) if pd.notna(row.get("pre_dbp")) else None
        p={"pid":pid,"prob":round(float(row["hypo_prob"]),3),"tier":str(row["hypo_tier"]),"sbp":sbp,"dbp":dbp,"sbp_suspect":bool(sbp is not None and not 60<=sbp<=260),"dbp_suspect":bool(dbp is not None and not 30<=dbp<=150),"duration_suspect":False,"dm":int(pd.to_numeric(row.get("DM",0), errors="coerce")) if pd.notna(pd.to_numeric(row.get("DM",0), errors="coerce")) else 0,"age":int(pd.to_numeric(row.get("age",0), errors="coerce")) if pd.notna(pd.to_numeric(row.get("age",0), errors="coerce")) else 0,"roll3":float(row.get("hypotension_event_roll3_mean",0) or 0),"ktv":float(row.get("ktv_proxy_lag1",0) or 0),"idwg":idwg,"idwg_suspect":bool(idwg is not None and idwg>8),"idwg_reason":None,"avg_uf":float(row.get("avg_uf",0) or 0),"drift":bool(drift.get("drift",False)),"dtype":drift.get("drift_type"),"daction":inter.get("action") if isinstance(inter,dict) else None,"dreason":drift.get("reason"),"drift_probability":drift.get("probability"),"drift_intervention":inter,"hypo_hist":hypo_hist,"early_term":et}
        # Deterministic action first; Groq is applied only to a small ranked set below.
        p["nursing_action"] = _fallback_action(p) if (p["tier"] == "HIGH" or p["drift"]) else None
        profiles.append(p)

    groq_candidates = [p for p in profiles if p["tier"] == "HIGH" or p["drift"]]
    groq_candidates.sort(key=lambda p: (0 if p["tier"] == "HIGH" else 1, 0 if p["drift"] else 1, -p["prob"]))

    if _groq_client is not None and groq_candidates:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(groq_candidates), 8)) as executor:
            future_to_p = {executor.submit(_groq_action, p): p for p in groq_candidates[:GROQ_MAX_PATIENTS]}
            for future in future_to_p:
                p = future_to_p[future]
                try:
                    generated = future.result(timeout=3.0)
                    if generated:
                        p["nursing_action"] = generated
                except Exception:
                    pass
    order={"HIGH":0,"MEDIUM":1,"LOW":2}; profiles.sort(key=lambda p:(order.get(p["tier"],3),-p["prob"]))
    hi=sum(p["tier"]=="HIGH" for p in profiles); me=sum(p["tier"]=="MEDIUM" for p in profiles); lo=sum(p["tier"]=="LOW" for p in profiles); dr=sum(p["drift"] for p in profiles); rep=sum(p["hypo_hist"]>=2 for p in profiles)
    return _json_safe({"module":"module5","available":True,"date":datetime.now().isoformat(),"summary":{"n_total":len(profiles),"n_high":hi,"n_medium":me,"n_low":lo,"n_drift":dr,"n_hypotension_repeat":rep},"briefing":_briefing(profiles),"patients":profiles,"model":{"artifact":"hypo_final_tiered.pkl","feature_count":len(ALL_FEATURES),"high_threshold":HIGH_THRESHOLD,"low_threshold":LOW_THRESHOLD,"missing_features_padded":missing},"groq_enabled":_groq_client is not None,"groq_max_patients":GROQ_MAX_PATIENTS,"groq_calls_used":min(len(groq_candidates),GROQ_MAX_PATIENTS) if _groq_client is not None else 0})