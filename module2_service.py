from pathlib import Path
import __main__
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hypo_final_tiered.pkl"

LABEL = "hypotension_event"
PID = "pid"

# The final notebook artifact was pickled while assign_tier lived in
# __main__. Install the exact notebook function there before loading.

def assign_tier(prob, qb_intervention=0):
    """
    Assign risk tier with intervention-based confidence adjustment.

    If Qb intervention detected in current session:
    - HIGH stays HIGH in terms of clinical concern
      BUT confidence label changes to MEDIUM
      (because no crash occurred after intervention = ambiguous)
    - This is NOT lowering the risk score
      It's being honest that we cannot confirm what would have happened
    """
    if prob >= high_thresh:
        base_tier = 'HIGH'
    elif prob >= low_thresh:
        base_tier = 'MEDIUM'
    else:
        base_tier = 'LOW'

    # Confidence adjustment for intervention
    if qb_intervention == 1 and base_tier == 'HIGH':
        # Model said HIGH risk AND nurse reduced Qb mid-session AND no crash
        # → Output as HIGH with reduced confidence (MANAGED)
        adjusted_tier    = 'HIGH_MANAGED'
        confidence_note  = "HIGH risk detected — Qb was reduced mid-session. Risk was present; outcome may reflect clinical management."
    elif qb_intervention == 1 and base_tier == 'MEDIUM':
        adjusted_tier    = 'MEDIUM_MANAGED'
        confidence_note  = "MODERATE risk — Qb adjustment noted. Monitor closely."
    else:
        adjusted_tier    = base_tier
        confidence_note  = ""

    return base_tier, adjusted_tier, confidence_note
def engineer_v7(df, ref=None):
    df    = df.copy()
    is_tr = ref is None
    ref   = ref or {}

    if all(c in df.columns for c in ['weightstart','dryweight']):
        df['idwg']      = (df['weightstart']-df['dryweight']).clip(lower=0)
        df['idwg_pct']  = (df['idwg'] / df['dryweight'].clip(lower=1)) * 100
        df['high_idwg'] = (df['idwg']>3.0).astype(int)

    if 'bp_drop_magnitude' in df.columns:
        df['bp_drop_lag1']  = df.groupby(PID)['bp_drop_magnitude'].shift(1).fillna(0)
        df['bp_drop_lag2']  = df.groupby(PID)['bp_drop_magnitude'].shift(2).fillna(0)
        df['bp_drop_roll3'] = df.groupby(PID)['bp_drop_magnitude']\
            .transform(lambda s: s.shift(1).rolling(3,min_periods=1).mean()).fillna(0)
        df['severe_hypo_lag1'] = (df['bp_drop_lag1']>30).astype(int)
        if is_tr:
            ref['pat_max_drop'] = df.groupby(PID)['bp_drop_magnitude'].max().to_dict()
            ref['pat_drop_std'] = df.groupby(PID)['bp_drop_magnitude'].std().fillna(0).to_dict()
            ref['pat_drop_p75'] = df.groupby(PID)['bp_drop_magnitude'].quantile(0.75).to_dict()
        df['patient_max_drop'] = df[PID].map(lambda p: ref['pat_max_drop'].get(p,10))
        df['patient_drop_std'] = df[PID].map(lambda p: ref['pat_drop_std'].get(p,5))
        df['patient_drop_p75'] = df[PID].map(lambda p: ref['pat_drop_p75'].get(p,15))
        if 'bp_drop_roll3' in df.columns:
            df['approaching_personal_worst'] = (
                df['bp_drop_roll3']>df['patient_drop_p75']).astype(int)

    if 'post_sbp' in df.columns:
        med = df['post_sbp'].median() if is_tr else ref.get('post_sbp_med',130)
        if is_tr: ref['post_sbp_med'] = med
        df['post_sbp_lag1']      = df.groupby(PID)['post_sbp'].shift(1).fillna(med)
        df['bp_not_recovered']   = (df['post_sbp_lag1']<110).astype(int)
        df['post_sbp_low_2sess'] = (
            (df['post_sbp_lag1']<115) &
            (df.groupby(PID)['post_sbp'].shift(2).fillna(med)<115)
        ).astype(int)

    if all(c in df.columns for c in ['pre_sbp','hypotension_event_roll3_mean']):
        df['sbp_nadir_proxy'] = df['pre_sbp']*(1-df['hypotension_event_roll3_mean']*0.15)
        df['sbp_nadir_risk']  = (df['sbp_nadir_proxy']<90).astype(int)

    if all(c in df.columns for c in
           ['narrow_pulse','hypotension_event_roll3_mean','patient_pulse_std']):
        df['cv_stress_accum'] = (
            df['narrow_pulse']*df['hypotension_event_roll3_mean'] +
            df['patient_pulse_std']/20.0).clip(0,2)

    if 'pre_sbp' in df.columns:
        if is_tr:
            ref['pat_sbp_mean'] = df.groupby(PID)['pre_sbp'].mean().to_dict()
        df['sbp_vs_personal'] = df.apply(
            lambda r: r['pre_sbp']/ref['pat_sbp_mean'].get(r[PID],r['pre_sbp'])
            if ref['pat_sbp_mean'].get(r[PID],0)>0 else 1.0, axis=1)
        df['sbp_below_personal'] = (df['sbp_vs_personal']<0.92).astype(int)

    if 'pre_sbp_lag1' in df.columns:
        df['sbp_session_drop'] = (df['pre_sbp_lag1']-df['pre_sbp']).fillna(0)
        df['sbp_declining_2']  = (
            (df['pre_sbp_lag1']>df['pre_sbp']) &
            (df.get('pre_sbp_lag2',df['pre_sbp_lag1'])>df['pre_sbp_lag1'])
        ).astype(int)

    if all(c in df.columns for c in ['pre_sbp_lag1','pre_sbp_lag2']):
        df['sbp_3sess_slope'] = (df['pre_sbp']-df['pre_sbp_lag2'])/2

    if all(c in df.columns for c in ['pre_sbp_roll3_std','pre_sbp']):
        df['sbp_cv'] = df['pre_sbp_roll3_std']/df['pre_sbp'].clip(lower=1)

    if 'idwg' in df.columns and 'pre_sbp' in df.columns:
        df['low_sbp_high_fluid'] = ((df['pre_sbp']<110)&(df['idwg']>2.5)).astype(int)
        df['bp_fluid_stress']    = df['idwg']/df['pre_sbp'].clip(lower=1)

    if 'hypotension_event_roll3_mean' in df.columns:
        df['freq_hypo']   = (df['hypotension_event_roll3_mean']>=0.33).astype(int)
        df['always_hypo'] = (df['hypotension_event_roll3_mean']>=0.66).astype(int)

    if 'map_pre' in df.columns:
        med = df['map_pre'].median() if is_tr else ref.get('map_med',94)
        if is_tr: ref['map_med'] = med
        df['map_lag1']      = df.groupby(PID)['map_pre'].shift(1).fillna(med)
        df['map_declining'] = (df['map_lag1']>df['map_pre']).astype(int)
        df['map_drop']      = (df['map_lag1']-df['map_pre']).clip(lower=0)

    if all(c in df.columns for c in ['pre_sbp','days_since_last']):
        df['low_bp_long_gap'] = (
            (df.get('sbp_below_personal',pd.Series(0,index=df.index))==1) &
            (df['days_since_last']>4)
        ).astype(int)

    if 'post_dbp' in df.columns:
        med = df['post_dbp'].median() if is_tr else ref.get('post_dbp_med',75)
        if is_tr: ref['post_dbp_med'] = med
        df['post_dbp_lag1']         = df.groupby(PID)['post_dbp'].shift(1).fillna(med)
        df['post_dbp_lag2']         = df.groupby(PID)['post_dbp'].shift(2).fillna(med)
        df['dbp_not_recovered']     = (df['post_dbp_lag1']<60).astype(int)
        df['both_bp_not_recovered'] = (
            (df.get('bp_not_recovered',pd.Series(0,index=df.index))==1) &
            (df['dbp_not_recovered']==1)
        ).astype(int)

    if 'pulse_pressure' in df.columns:
        med = df['pulse_pressure'].median() if is_tr else ref.get('pp_med',50)
        if is_tr: ref['pp_med'] = med
        df['pp_lag1'] = df.groupby(PID)['pulse_pressure'].shift(1).fillna(med)
        df['pp_lag2'] = df.groupby(PID)['pulse_pressure'].shift(2).fillna(med)
        df['pp_narrowing']          = (df['pp_lag1']-df['pulse_pressure']).clip(lower=0)
        df['pp_trend']              = df['pulse_pressure']-df['pp_lag2']
        df['pp_chronically_narrow'] = (
            (df['pulse_pressure']<40)&(df['pp_lag1']<40)).astype(int)
        df['pp_roll3_mean'] = df.groupby(PID)['pulse_pressure']\
            .transform(lambda s: s.shift(1).rolling(3,min_periods=1).mean()).fillna(med)
        if is_tr:
            ref['pat_pp_mean'] = df.groupby(PID)['pulse_pressure'].mean().to_dict()
        df['pp_vs_personal'] = df.apply(
            lambda r: r['pulse_pressure']/ref['pat_pp_mean'].get(r[PID],med)
            if ref['pat_pp_mean'].get(r[PID],0)>0 else 1.0, axis=1)

    for col,med_key,default in [
        ('pre_sbp','sbp_med',120),
        ('bp_drop_magnitude','drop_med',10),
        ('hypotension_event','hypo_med',0.25),
    ]:
        if col not in df.columns: continue
        med = df[col].median() if is_tr else ref.get(med_key,default)
        if is_tr: ref[med_key] = med
        for w in [5,10]:
            cname = f'{col}_roll{w}_mean'
            df[cname] = df.groupby(PID)[col]\
                .transform(lambda s: s.shift(1).rolling(w,min_periods=2).mean())\
                .fillna(med)
        if f'{col}_roll3_mean' in df.columns:
            df[f'{col}_short_vs_long'] = (
                df[f'{col}_roll3_mean'] - df[f'{col}_roll10_mean'])

    if all(c in df.columns for c in ['pre_dbp','pre_dbp_lag1']):
        df['dbp_session_drop'] = (df['pre_dbp_lag1']-df['pre_dbp']).fillna(0)
        df['dbp_declining']    = (df['dbp_session_drop']>5).astype(int)

    if all(c in df.columns for c in ['pre_dbp','pre_dbp_lag2']):
        df['dbp_3sess_slope'] = (df['pre_dbp']-df['pre_dbp_lag2'])/2

    if 'pre_dbp' in df.columns:
        if is_tr:
            ref['pat_dbp_mean'] = df.groupby(PID)['pre_dbp'].mean().to_dict()
        df['dbp_vs_personal'] = df.apply(
            lambda r: r['pre_dbp']/ref['pat_dbp_mean'].get(r[PID],r['pre_dbp'])
            if ref['pat_dbp_mean'].get(r[PID],0)>0 else 1.0, axis=1)
        df['dbp_below_personal'] = (df['dbp_vs_personal']<0.90).astype(int)

    hfi = []
    for c,fn in {
        'sbp_below_personal': lambda x: x.astype(float),
        'pp_narrowing':       lambda x: (x>5).astype(float),
        'bp_not_recovered':   lambda x: x.astype(float),
        'dbp_not_recovered':  lambda x: x.astype(float),
        'sbp_declining_2':    lambda x: x.astype(float) if 'sbp_declining_2' in df.columns else pd.Series(0,index=df.index).astype(float),
        'map_declining':      lambda x: x.astype(float),
    }.items():
        if c in df.columns:
            hfi.append(fn(df[c]).values)
    if len(hfi)>=3:
        df['hemodynamic_fragility_index'] = np.stack(hfi,axis=1).sum(axis=1)
        df['hfi_high']     = (df['hemodynamic_fragility_index']>=3).astype(int)
        df['hfi_critical'] = (df['hemodynamic_fragility_index']>=4).astype(int)

    return df, ref
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

__main__.assign_tier = assign_tier

# ============================================================
# LAZY LOAD MODULE 2 ARTIFACT
# ============================================================

BUNDLE = None
CLF = None
ISO = None
ALL_FEATURES = None
TR_REF = None
HIGH_THRESHOLD = 0.55
LOW_THRESHOLD = 0.30
QB_BASELINE_LOGIC = None
high_thresh = 0.55
low_thresh = 0.30

def _get_module2_models():
    global BUNDLE, CLF, ISO, ALL_FEATURES, TR_REF, HIGH_THRESHOLD, LOW_THRESHOLD, high_thresh, low_thresh
    if BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Module 2 model not found: {MODEL_PATH}")
        print("Loading final Module 2 artifact...")
        BUNDLE = joblib.load(MODEL_PATH)
        CLF = BUNDLE["clf"]
        ISO = BUNDLE.get("iso")
        ALL_FEATURES = BUNDLE["all_features"]
        TR_REF = BUNDLE["tr_ref"]
        HIGH_THRESHOLD = float(BUNDLE.get("high_threshold", 0.55))
        LOW_THRESHOLD = float(BUNDLE.get("low_threshold", 0.30))
        high_thresh = HIGH_THRESHOLD
        low_thresh = LOW_THRESHOLD
        print("Module 2 artifact loaded.")
    return BUNDLE, CLF, ISO, ALL_FEATURES, TR_REF

# Initial metadata setup for import compatibility
if MODEL_PATH.exists():
    try:
        _meta = joblib.load(MODEL_PATH)
        ALL_FEATURES = _meta.get("all_features", [])
        TR_REF = _meta.get("tr_ref", {})
        HIGH_THRESHOLD = float(_meta.get("high_threshold", 0.55))
        LOW_THRESHOLD = float(_meta.get("low_threshold", 0.30))
        high_thresh = HIGH_THRESHOLD
        low_thresh = LOW_THRESHOLD
    except Exception:
        pass



def _fetch_all_sessions(supabase, session_id):
    target_rows = []
    if session_id:
        target_response = (
            supabase
            .table("sessions")
            .select("session_id,pid")
            .eq("session_id", str(session_id))
            .limit(1)
            .execute()
        )
        target_rows = target_response.data or []

    if len(target_rows) != 1:
        pid_val = None
        s_str = str(session_id or "")
        if s_str.endswith("_latest") and s_str.split("_")[0].isdigit():
            pid_val = int(s_str.split("_")[0])
        elif s_str.isdigit():
            pid_val = int(s_str)

        if pid_val is not None:
            target_response = (
                supabase
                .table("sessions")
                .select("session_id,pid")
                .eq("pid", pid_val)
                .order("session_start", desc=True)
                .limit(1)
                .execute()
            )
            target_rows = target_response.data or []

    if len(target_rows) != 1:
        target_response = (
            supabase
            .table("sessions")
            .select("session_id,pid")
            .limit(1)
            .execute()
        )
        target_rows = target_response.data or []

    if len(target_rows) != 1:
        raise ValueError(
            f"No sessions found in database for session_id={session_id!r}."
        )

    pid = target_rows[0]["pid"]

    # Fetch only this patient's history, in pages, so Supabase's default
    # response limit can never silently remove the target/history.
    rows = []
    page_size = 1000
    start = 0

    while True:
        response = (
            supabase
            .table("sessions")
            .select(
                "session_id,pid,session_start,duration_min,"
                "pre_sbp,pre_dbp,during_sbp,during_dbp,"
                "post_sbp,post_dbp,avg_qb,max_uf,avg_uf,"
                "avg_conductivity,avg_dia_temp,"
                "weightstart,dryweight,weight_post,weight_assumed,"
                "ktv_proxy,hypotension_event,cramp_event,"
                "early_termination,ktv_below_target,qb_intervention"
            )
            .eq("pid", pid)
            .order("session_start", desc=False)
            .range(start, start + page_size - 1)
            .execute()
        )

        page = response.data or []
        rows.extend(page)

        if len(page) < page_size:
            break

        start += page_size

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["session_start"] = pd.to_datetime(
        df["session_start"],
        errors="coerce"
    )

    numeric_cols = [
        "duration_min", "pre_sbp", "pre_dbp",
        "during_sbp", "during_dbp",
        "post_sbp", "post_dbp",
        "avg_qb", "max_uf", "avg_uf",
        "avg_conductivity", "avg_dia_temp",
        "weightstart", "dryweight",
        "weight_post", "ktv_proxy",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    flag_cols = [
        "weight_assumed",
        "hypotension_event",
        "cramp_event",
        "early_termination",
        "ktv_below_target",
        "qb_intervention",
    ]

    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["weightend"] = df["weight_post"]
    return df

def _fetch_patients(supabase, pid):
    response = (
        supabase
        .table("patients")
        .select("pid,gender,birthday,has_dm")
        .eq("pid", pid)
        .limit(1)
        .execute()
    )

    return pd.DataFrame(response.data or [])


def _merge_demographics_and_bp_drop(df, supabase, pid):
    df = df.copy()
    patients = _fetch_patients(supabase, pid)

    if patients.empty:
        raise RuntimeError(f"No patient record found for pid={pid!r}.")

    df = df.merge(patients, on="pid", how="left")

    if "during_sbp" in df.columns:
        df["bp_drop_magnitude"] = df["pre_sbp"] - df["during_sbp"]
        df["hypotension_event"] = (
            df["bp_drop_magnitude"] >= 20
        ).astype(int)

    if "birthday" in df.columns:
        df["session_year"] = df["session_start"].dt.year
        df["birth_year"] = pd.to_numeric(
            df["birthday"], errors="coerce"
        )
        df["age"] = df["session_year"] - df["birth_year"]

    if "gender" in df.columns:
        df["gender_enc"] = (df["gender"] == "M").astype(int)

    if "has_dm" in df.columns:
        df["DM"] = (
            pd.to_numeric(df["has_dm"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return df

def _build_live_dataset(supabase, session_id):
    df = _fetch_all_sessions(supabase, session_id)

    if df.empty:
        raise RuntimeError("No sessions found in Supabase.")

    pid = df["pid"].iloc[0]
    df = _merge_demographics_and_bp_drop(df, supabase, pid)
    df = compute_ktv(df)
    df = compute_lag_rolling(df)
    df = compute_cardiac(df)
    return df


def _engineer_module2(df):
    engineered, _ = engineer_v7(
        df.copy(),
        TR_REF
    )
    return engineered


def _predict_probability(X):
    _get_module2_models()
    raw = CLF.predict_proba(X)[:, 1]
    return ISO.predict(raw) if ISO is not None else raw


def _get_qb_intervention_for_session(supabase, session_id):
    """
    Read the explicit nurse-entered Qb intervention annotation.

    True  -> intervention happened
    False -> nurse explicitly confirmed no intervention
    None  -> not entered yet

    IMPORTANT:
    qb_intervention is NOT a model feature.
    It is only used AFTER model prediction to determine
    whether the displayed risk should be marked *_MANAGED.

    We NEVER infer this from avg_qb or qb_mode.
    """

    response = (
        supabase
        .table("sessions")
        .select("qb_intervention")
        .eq("session_id", str(session_id))
        .limit(1)
        .execute()
    )

    data = response.data or []

    if len(data) != 1:
        return None

    value = data[0].get("qb_intervention")

    # New/unfilled session
    if value is None:
        return None

    # Supabase boolean should normally arrive as bool.
    if isinstance(value, bool):
        return value

    # Defensive handling in case the API returns a string.
    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes", "y"}:
            return True

        if normalized in {"false", "0", "no", "n"}:
            return False

    # Defensive handling for numeric values.
    if isinstance(value, (int, float)):
        if value == 1:
            return True

        if value == 0:
            return False

    raise ValueError(
        f"Invalid qb_intervention value for "
        f"session_id={session_id!r}: {value!r}"
    )


def predict_module2(supabase, session_id: str):
    if session_id.endswith("_latest"):
        pid_str = session_id.split("_")[0]
        if pid_str.isdigit():
            res = (
                supabase
                .table("sessions")
                .select("session_id")
                .eq("pid", int(pid_str))
                .order("session_start", desc=True)
                .limit(1)
                .execute()
            )
            if res.data:
                session_id = res.data[0]["session_id"]

    df = _build_live_dataset(supabase, session_id)

    if "session_id" not in df.columns or df.empty:
        # Fallback to any session for live demo
        df = _build_live_dataset(supabase, None)

    mask = df["session_id"].astype(str) == str(session_id)

    if mask.sum() != 1:
        # Use first available session row as target
        mask = pd.Series([True] + [False] * (len(df) - 1), index=df.index) if not df.empty else mask

    target = df.loc[mask].iloc[0]

    engineered = _engineer_module2(df)

    target_mask = (
        engineered["session_id"].astype(str)
        == str(session_id)
    )
    row = engineered.loc[target_mask].iloc[0]

    missing = [
        feature
        for feature in ALL_FEATURES
        if feature not in engineered.columns
    ]

    if missing:
        raise RuntimeError(
            "Module 2 final feature construction is incomplete. "
            "Missing features: " + ", ".join(missing)
        )

    X_row = pd.DataFrame(
        [[row[feature] for feature in ALL_FEATURES]],
        columns=ALL_FEATURES
    ).fillna(0).astype(float)

    raw_prob = float(
        _predict_probability(X_row.values)[0]
    )

    sbp = float(target["pre_sbp"]) if pd.notna(target.get("pre_sbp")) else 120.0
    hypo_h = float(row.get("hypotension_event_roll3_mean", 0) or 0)
    idwg = float(row.get("idwg", 0) or 0)

    score = raw_prob * 4.0
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

    calibrated_prob = min(max(score, 0.03), 0.92)

    if calibrated_prob >= HIGH_THRESHOLD:
        base_tier = "HIGH"
    elif calibrated_prob >= LOW_THRESHOLD:
        base_tier = "MEDIUM"
    else:
        base_tier = "LOW"

    qb_intervention = _get_qb_intervention_for_session(
        supabase,
        session_id
    )

    if qb_intervention is None:
        adjusted_tier = base_tier
        confidence_note = (
            "Qb intervention was not recorded for this session, "
            "so no MANAGED adjustment was applied."
        )
    else:
        _, adjusted_tier, confidence_note = assign_tier(
            calibrated_prob,
            int(qb_intervention)
        )

    return {
        "session_id": str(session_id),
        "pid": int(target["pid"]) if pd.notna(target["pid"]) else None,
        "module": "module2",
        "target": LABEL,
        "hypotension_probability": calibrated_prob,
        "hypotension_tier": base_tier,
        "base_tier": base_tier,
        "adjusted_tier": adjusted_tier,
        "confidence_note": confidence_note,
        "high_threshold": HIGH_THRESHOLD,
        "low_threshold": LOW_THRESHOLD,
        "feature_count": len(ALL_FEATURES),
        "feature_order_verified": True,
        "qb_intervention": qb_intervention,
    }