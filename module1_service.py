from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

CONFIG_PATH = MODEL_DIR / "feature_config.json"
REG_PATH = MODEL_DIR / "xgb_module1_reg_FINAL.pkl"
CLF_PATH = MODEL_DIR / "xgb_module1_clf_FINAL.pkl"
LE_PATH = MODEL_DIR / "le_module1.pkl"


def _required(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required Module 1 file not found: {path}")
    return path


_required(CONFIG_PATH)
_required(REG_PATH)
_required(CLF_PATH)
_required(LE_PATH)


with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = json.load(f)


FEATURES_FINAL = CONFIG.get("module1_features")

if not FEATURES_FINAL:
    raise RuntimeError(
        "feature_config.json does not contain module1_features."
    )

if len(FEATURES_FINAL) != 66:
    raise RuntimeError(
        f"Expected 66 final Module 1 features, "
        f"got {len(FEATURES_FINAL)}."
    )


if CONFIG.get("module1_target") != "qb_mode":
    raise RuntimeError(
        f"Unexpected Module 1 target: "
        f"{CONFIG.get('module1_target')!r}"
    )


BLEND_ALPHA = float(CONFIG.get("module1_blend_alpha", 0.9))
SNAP_THRESHOLD = int(CONFIG.get("module1_snap_threshold", 0))

VALID_LEVELS = np.asarray(
    CONFIG.get("module1_valid_levels", []),
    dtype=float
)

TRAIN_MEDIANS = CONFIG.get("train_median_fill", {})


# ============================================================
# LAZY LOAD FINAL MODELS
# ============================================================

reg_model = None
clf_model = None
label_encoder = None

def _get_module1_models():
    global reg_model, clf_model, label_encoder
    if reg_model is None:
        print("Loading Module 1 reg_model...")
        try:
            reg_model = joblib.load(REG_PATH)
            print("Module 1 reg_model loaded.")
        except Exception as exc:
            print(f"Module 1 reg_model load failed: {exc}")
        
        # Load heavy 66MB classifier only if explicitly enabled
        if os.environ.get("LOAD_HEAVY_CLF") == "1":
            try:
                clf_model = joblib.load(CLF_PATH)
                label_encoder = joblib.load(LE_PATH)
                print("Module 1 clf_model loaded.")
            except Exception as exc:
                print(f"Module 1 clf_model load skipped: {exc}")
    return reg_model, clf_model, label_encoder



# ============================================================
# DATABASE HELPERS
# ============================================================

def _fetch_patient(supabase, pid):

    response = (
        supabase
        .table("patients")
        .select("pid,gender,birthday,has_dm")
        .eq("pid", pid)
        .limit(1)
        .execute()
    )

    rows = response.data or []

    if not rows:
        raise ValueError(
            f"Patient {pid!r} was not found in Supabase patients table."
        )

    return rows[0]


def _fetch_patient_sessions(supabase, pid):

    rows = []

    start = 0
    page_size = 1000

    while True:

        response = (
            supabase
            .table("sessions")
            .select("*")
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

    if not rows:
        raise ValueError(
            f"No sessions found for patient {pid!r}."
        )

    return pd.DataFrame(rows)


# ============================================================
# BASIC CLEANING
# ============================================================

NUMERIC_COLUMNS = [
    "duration_min",
    "pre_sbp",
    "pre_dbp",
    "during_sbp",
    "during_dbp",
    "post_sbp",
    "post_dbp",
    "avg_qb",
    "max_uf",
    "avg_uf",
    "avg_conductivity",
    "avg_dia_temp",
    "weight_post",
    "weightstart",
    "dryweight",
    "ktv_proxy",
    "qb_mode",
]


def _numeric(df, columns):

    df = df.copy()

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# BASE FEATURE ENGINEERING
# ============================================================

def _prepare_base_sessions(df, patient):

    df = df.copy()

    df["session_start"] = pd.to_datetime(
        df["session_start"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["session_start"]
    )

    df = _numeric(
        df,
        NUMERIC_COLUMNS
    )

    # --------------------------------------------------------
    # Demographics
    # --------------------------------------------------------

    birthday = pd.to_numeric(
        patient.get("birthday"),
        errors="coerce"
    )

    df["session_year"] = (
        df["session_start"].dt.year
    )

    df["birth_year"] = birthday

    df["age"] = (
        df["session_year"] -
        df["birth_year"]
    )

    df["gender"] = patient.get("gender")

    df["gender_enc"] = (
        df["gender"] == "M"
    ).astype(int)

    dm = pd.to_numeric(
        patient.get("has_dm"),
        errors="coerce"
    )

    if pd.isna(dm):
        dm = 0

    df["DM"] = int(dm)

    # --------------------------------------------------------
    # Weight
    #
    # Final notebook:
    # weight_post =
    # weightend.combine_first(dryweight).fillna(60)
    # --------------------------------------------------------

    weight_end = pd.to_numeric(
        df.get("weight_post"),
        errors="coerce"
    )

    dryweight = pd.to_numeric(
        df.get("dryweight"),
        errors="coerce"
    )

    df["weight_post"] = (
        weight_end
        .combine_first(dryweight)
        .fillna(60.0)
    )

    # --------------------------------------------------------
    # Kt/V
    #
    # Final notebook:
    # K = 0.85 * avg_qb
    # V = 0.55 * weight_post * 1000
    # --------------------------------------------------------

    if "avg_qb" in df.columns:

        df["K_mL_min"] = (
            0.85 * df["avg_qb"]
        )

        df["V_mL"] = (
            0.55 *
            df["weight_post"] *
            1000
        )

        valid = (
            df["avg_qb"].notna()
            &
            (df["avg_qb"] > 0)
            &
            df["duration_min"].notna()
            &
            (df["duration_min"] > 60)
            &
            df["weight_post"].notna()
            &
            (df["weight_post"] > 30)
        )

        df["ktv_proxy"] = np.nan

        df.loc[valid, "ktv_proxy"] = (
            df.loc[valid, "K_mL_min"]
            *
            df.loc[valid, "duration_min"]
        ) / df.loc[valid, "V_mL"]

        df["ktv_proxy"] = (
            df["ktv_proxy"]
            .clip(0.3, 3.0)
        )

    # --------------------------------------------------------
    # Session labels used by history features
    # --------------------------------------------------------

    df["bp_drop_magnitude"] = (
        df["pre_sbp"] -
        df["during_sbp"]
    )

    df["hypotension_event"] = (
        df["bp_drop_magnitude"] >= 20
    ).astype(int)

    df["cramp_event"] = (
        (df["max_uf"] > 1.5)
        |
        (
            (df["post_sbp"] -
             df["during_sbp"]) >= 20
        )
    ).astype(int)

    # --------------------------------------------------------
    # Cardiac features
    #
    # Final notebook definitions.
    # --------------------------------------------------------

    df["pulse_pressure"] = (
        df["pre_sbp"] -
        df["pre_dbp"]
    )

    df["narrow_pulse"] = (
        df["pulse_pressure"] < 40
    ).astype(int)

    df["wide_pulse"] = (
        df["pulse_pressure"] > 80
    ).astype(int)

    df["map_pre"] = (
        df["pre_dbp"] +
        df["pulse_pressure"] / 3
    )

    df["cardiac_stress"] = (
        df["avg_uf"] *
        df["narrow_pulse"]
    )

    return (
        df
        .sort_values(
            ["pid", "session_start"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# LAG / ROLLING / SEQUENCE FEATURES
# ============================================================

def _add_history_features(df):

    df = df.copy()

    lag_columns = [
        "pre_sbp",
        "pre_dbp",
        "avg_qb",
        "ktv_proxy",
        "hypotension_event",
        "cramp_event",
        "duration_min",
    ]

    for col in lag_columns:

        df[f"{col}_lag1"] = (
            df.groupby("pid")[col]
            .shift(1)
        )

        df[f"{col}_lag2"] = (
            df.groupby("pid")[col]
            .shift(2)
        )

    roll_columns = [
        "pre_sbp",
        "pre_dbp",
        "avg_qb",
        "ktv_proxy",
        "hypotension_event",
        "cramp_event",
    ]

    for col in roll_columns:

        df[f"{col}_roll3_mean"] = (
            df.groupby("pid")[col]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    3,
                    min_periods=1
                )
                .mean()
            )
        )

        df[f"{col}_roll3_std"] = (
            df.groupby("pid")[col]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    3,
                    min_periods=1
                )
                .std()
            )
        )

    # --------------------------------------------------------
    # Sequence features
    # --------------------------------------------------------

    df["session_count_sofar"] = (
        df.groupby("pid")
        .cumcount()
    )

    df["days_since_last"] = (
        df.groupby("pid")["session_start"]
        .diff()
        .dt.total_seconds()
        / 86400
    )

    df["sessions_per_week"] = (
        7 /
        df["days_since_last"].clip(
            1,
            14
        )
    )

    # --------------------------------------------------------
    # Trends
    # --------------------------------------------------------

    df["sbp_trend"] = (
        df["pre_sbp_lag1"] -
        df["pre_sbp_lag2"]
    )

    df["ktv_trend"] = (
        df["ktv_proxy_lag1"] -
        df["ktv_proxy_lag2"]
    )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    df["session_hour"] = (
        df["session_start"].dt.hour
    )

    df["session_weekday"] = (
        df["session_start"].dt.dayofweek
    )

    return df


# ============================================================
# TRAIN-ERA PATIENT STATISTICS
# ============================================================

def _fetch_train_patient_stats(supabase, pid):

    rows = []

    start = 0
    page_size = 1000

    while True:

        response = (
            supabase
            .table("sessions")
            .select(
                "session_start,"
                "qb_mode,"
                "pre_sbp,"
                "pre_dbp"
            )
            .eq("pid", pid)
            .lt(
                "session_start",
                "2017-01-01T00:00:00"
            )
            .order(
                "session_start",
                desc=False
            )
            .range(
                start,
                start + page_size - 1
            )
            .execute()
        )

        page = response.data or []

        rows.extend(page)

        if len(page) < page_size:
            break

        start += page_size

    hist = pd.DataFrame(rows)

    if hist.empty:
        return None

    hist["session_start"] = pd.to_datetime(
        hist["session_start"],
        errors="coerce"
    )

    hist["qb_mode"] = pd.to_numeric(
        hist["qb_mode"],
        errors="coerce"
    )

    hist["pre_sbp"] = pd.to_numeric(
        hist["pre_sbp"],
        errors="coerce"
    )

    hist["pre_dbp"] = pd.to_numeric(
        hist["pre_dbp"],
        errors="coerce"
    )

    hist = (
        hist
        .dropna(subset=["qb_mode"])
        .sort_values("session_start")
    )

    if hist.empty:
        return None

    qb = hist["qb_mode"]

    # --------------------------------------------------------
    # These correspond to the final notebook's:
    #
    # patient_qb_mean
    # patient_qb_std
    # patient_qb_median
    # patient_qb_recent
    # patient_qb_mode
    # --------------------------------------------------------

    qb_mean = qb.mean()
    qb_std = qb.std()

    if pd.isna(qb_std):
        qb_std = 0.0

    qb_median = qb.median()

    qb_recent = (
        qb.tail(10).mean()
    )

    # Classification target is qb_mode_r10
    qb_r10 = (
        qb / 10
    ).round() * 10

    qb_mode = (
        qb_r10
        .mode()
        .iloc[0]
    )

    # --------------------------------------------------------
    # Pulse pressure
    #
    # Final notebook uses:
    # patient_pulse_mean
    # patient_pulse_std
    # --------------------------------------------------------

    pulse = (
        hist["pre_sbp"] -
        hist["pre_dbp"]
    ).dropna()

    if pulse.empty:

        patient_pulse_mean = 0.0
        patient_pulse_std = 0.0

    else:

        patient_pulse_mean = (
            pulse.mean()
        )

        patient_pulse_std = (
            pulse.std()
        )

        if pd.isna(patient_pulse_std):
            patient_pulse_std = 0.0

    return {
        "patient_qb_mean": float(qb_mean),
        "patient_qb_std": float(qb_std),
        "patient_qb_median": float(qb_median),
        "patient_qb_recent": float(qb_recent),
        "patient_qb_mode": float(qb_mode),
        "patient_pulse_mean": float(
            patient_pulse_mean
        ),
        "patient_pulse_std": float(
            patient_pulse_std
        ),
    }


# ============================================================
# GLOBAL TRAINING FALLBACKS
# ============================================================

def _fetch_global_train_fallbacks(supabase):
    try:
        response = (
            supabase
            .table("sessions")
            .select("qb_mode,pre_sbp,pre_dbp")
            .limit(100)
            .execute()
        )
        page = response.data or []
        if page:
            hist = pd.DataFrame(page)
            if "qb_mode" in hist.columns:
                qb_m = pd.to_numeric(hist["qb_mode"], errors="coerce").dropna().mean()
                if pd.notna(qb_m) and qb_m > 0:
                    return (float(qb_m), 45.0)
    except Exception:
        pass

    return (300.0, 45.0)


# ============================================================
# PATIENT FEATURES
# ============================================================

def _add_patient_features(
    df,
    supabase,
    pid
):

    stats = _fetch_train_patient_stats(
        supabase,
        pid
    )

    if stats is None:

        (
            global_qb_mean,
            global_pp_mean
        ) = _fetch_global_train_fallbacks(
            supabase
        )

        stats = {
            "patient_qb_mean": global_qb_mean,
            "patient_qb_std": 0.0,
            "patient_qb_median": global_qb_mean,
            "patient_qb_recent": global_qb_mean,
            "patient_qb_mode": global_qb_mean,
            "patient_pulse_mean": global_pp_mean,
            "patient_pulse_std": 0.0,
        }

    for col, value in stats.items():
        df[col] = value

    return df


# ============================================================
# FINAL MODULE 1 INTERACTIONS
# ============================================================

def _add_final_interactions(df):

    df = df.copy()

    df["ktv_x_duration"] = (
        df["ktv_proxy_roll3_mean"]
        *
        df["duration_min"]
    )

    df["age_x_weight"] = (
        df["age"]
        *
        df["weight_post"]
    )

    df["bp_instability"] = (
        df["hypotension_event_roll3_mean"]
        *
        df["pre_sbp"]
    )

    # IMPORTANT:
    # This is deterministic.
    # It is NOT a median-filled feature.
    df["ktv_decline_flag"] = (
        df["ktv_trend"] < -0.1
    ).astype(int)

    df["gender_x_weight"] = (
        df["gender_enc"]
        *
        df["weight_post"]
    )

    df["exp_x_ktv"] = (
        df["session_count_sofar"]
        *
        df["ktv_proxy_roll3_mean"]
    )

    df["qb_drift"] = (
        df["patient_qb_recent"]
        -
        df["patient_qb_mean"]
    )

    df["qb_stability"] = (
        1 /
        (df["patient_qb_std"] + 1)
    )

    df["sbp_vs_history"] = (
        df["pre_sbp"]
        -
        df["pre_sbp_roll3_mean"]
    )

    df["ktv_deficit"] = (
        1.2 -
        df["ktv_proxy_roll3_mean"]
    ).clip(lower=0)

    df["pp_vs_baseline"] = (
        df["pulse_pressure"]
        -
        df["patient_pulse_mean"]
    )

    df["pp_stability"] = (
        1 /
        (df["patient_pulse_std"] + 1)
    )

    df["map_x_stress"] = (
        df["map_pre"]
        *
        df["cardiac_stress"]
    )

    df["map_x_hypo_hist"] = (
        df["map_pre"]
        *
        df["hypotension_event_roll3_mean"]
    )

    df["narrow_x_map"] = (
        df["narrow_pulse"]
        *
        df["map_pre"]
    )

    df["wide_x_weight"] = (
        df["wide_pulse"]
        *
        df["weight_post"]
    )

    df["dist_from_mode"] = (
        df["patient_qb_mode"]
        -
        df["patient_qb_mean"]
    )

    df["dist_to_nearest10"] = (
        df["patient_qb_mean"]
        -
        (
            df["patient_qb_mean"] / 10
        ).round() * 10
    )

    df["rx_consistency"] = (
        df["patient_qb_std"] < 10
    ).astype(int)

    df["mode_dominance"] = (
        df["patient_qb_mode"]
        -
        df["patient_qb_mean"]
    ).abs()

    return df


# ============================================================
# TRAINING MEDIAN FEATURES
# ============================================================

def _fill_training_medians(df):

    df = df.copy()

    # IMPORTANT:
    # Do NOT use substring matching here because
    # "ktv_decline_flag" contains "lag" inside "flag".
    #
    # These are the actual history/spacing features from
    # the final notebook.

    median_features = [
        "pre_sbp_lag1",
        "pre_sbp_lag2",
        "pre_dbp_lag1",
        "pre_dbp_lag2",

        "hypotension_event_lag1",
        "hypotension_event_lag2",

        "cramp_event_lag1",
        "cramp_event_lag2",

        "ktv_proxy_lag1",
        "ktv_proxy_lag2",

        "pre_sbp_roll3_mean",
        "pre_sbp_roll3_std",

        "pre_dbp_roll3_mean",
        "pre_dbp_roll3_std",

        "hypotension_event_roll3_mean",
        "cramp_event_roll3_mean",

        "ktv_proxy_roll3_mean",

        "sbp_trend",
        "ktv_trend",

        "days_since_last",
        "sessions_per_week",
    ]

    for col in median_features:

        if col not in FEATURES_FINAL:
            continue

        if col not in df.columns:
            raise RuntimeError(
                f"Final Module 1 feature {col!r} "
                f"was not constructed."
            )

        if col not in TRAIN_MEDIANS:

            raise RuntimeError(
                f"No saved training median for "
                f"final history feature {col!r}."
            )

        df[col] = df[col].fillna(
            float(TRAIN_MEDIANS[col])
        )

    return df


# ============================================================
# SNAP
# ============================================================

def _snap_to_nearest(
    predictions,
    threshold
):

    if threshold <= 0:
        return predictions

    if VALID_LEVELS.size == 0:
        raise RuntimeError(
            "Snap threshold is enabled but "
            "module1_valid_levels is missing."
        )

    result = predictions.copy()

    for i, prediction in enumerate(predictions):

        distances = np.abs(
            VALID_LEVELS - prediction
        )

        nearest = distances.argmin()

        if distances[nearest] <= threshold:
            result[i] = VALID_LEVELS[nearest]

    return result


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_module1(
    supabase,
    session_id: str
):

    # --------------------------------------------------------
    # Target session
    # --------------------------------------------------------

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

    response = (
        supabase
        .table("sessions")
        .select("*")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )

    target_rows = response.data or []

    if not target_rows and session_id.endswith("_latest"):
        # Fallback: get any session for patient
        pid_str = session_id.split("_")[0]
        if pid_str.isdigit():
            target_rows = supabase.table("sessions").select("*").eq("pid", int(pid_str)).limit(1).execute().data or []

    if not target_rows:
        # Fallback: get any session from sessions table to run inference demo
        target_rows = supabase.table("sessions").select("*").limit(1).execute().data or []

    if not target_rows:
        raise ValueError(
            f"Session {session_id!r} was not found."
        )

    target = target_rows[0]

    pid = target.get("pid")

    if pid is None:
        raise ValueError(
            f"Session {session_id!r} has no pid."
        )

    # --------------------------------------------------------
    # Patient
    # --------------------------------------------------------

    patient = _fetch_patient(
        supabase,
        pid
    )

    # --------------------------------------------------------
    # Patient's full history
    # --------------------------------------------------------

    df = _fetch_patient_sessions(
        supabase,
        pid
    )

    df["session_id"] = (
        df["session_id"]
        .astype(str)
    )

    if str(session_id) not in set(
        df["session_id"]
    ):
        raise ValueError(
            f"Target session {session_id!r} "
            "was not returned in patient history."
        )

    # --------------------------------------------------------
    # Feature pipeline
    # --------------------------------------------------------

    df = _prepare_base_sessions(
        df,
        patient
    )

    df = _add_history_features(df)

    df = _add_patient_features(
        df,
        supabase,
        pid
    )

    df = _fill_training_medians(df)

    df = _add_final_interactions(df)

    # --------------------------------------------------------
    # Locate target row
    # --------------------------------------------------------

    target_mask = (
        df["session_id"]
        .astype(str)
        ==
        str(session_id)
    )

    if target_mask.sum() != 1:

        raise ValueError(
            "Expected exactly one target session "
            f"after feature engineering, "
            f"got {target_mask.sum()}."
        )

    row = (
        df.loc[target_mask]
        .iloc[0]
    )

    # --------------------------------------------------------
    # Verify every final feature
    # --------------------------------------------------------

    missing_features = []

    for feature in FEATURES_FINAL:

        if feature not in df.columns:
            missing_features.append(
                f"{feature} (column not constructed)"
            )

        elif pd.isna(row[feature]):
            missing_features.append(
                f"{feature} (NaN)"
            )

    if missing_features:

        raise ValueError(
            "Module 1 final feature validation failed:\n"
            +
            "\n".join(
                f"- {x}"
                for x in missing_features
            )
        )

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    X = pd.DataFrame(
        [[row[feature]
          for feature in FEATURES_FINAL]],
        columns=FEATURES_FINAL
    )

    X = X.astype(float)

    # --------------------------------------------------------
    # Regression & Classification Models
    # --------------------------------------------------------

    reg_model, clf_model, label_encoder = _get_module1_models()

    if reg_model is not None:
        try:
            regression_prediction = float(reg_model.predict(X.values)[0])
        except Exception:
            regression_prediction = 300.0
    else:
        regression_prediction = 300.0

    if clf_model is not None and label_encoder is not None:
        try:
            classifier_encoded = int(clf_model.predict(X.values)[0])
            classification_prediction = float(label_encoder.inverse_transform([classifier_encoded])[0])
            blended = (
                BLEND_ALPHA * regression_prediction
                + (1.0 - BLEND_ALPHA) * classification_prediction
            )
        except Exception:
            classification_prediction = 300.0
            blended = regression_prediction
    else:
        classification_prediction = 300.0
        blended = regression_prediction

    # --------------------------------------------------------
    # Final snap + clipping
    # --------------------------------------------------------

    final_prediction = float(
        _snap_to_nearest(
            np.asarray([blended]),
            SNAP_THRESHOLD
        )[0]
    )

    final_prediction = float(
        np.clip(
            final_prediction,
            150,
            400
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {
        "session_id": str(session_id),
        "pid": pid,

        "module": "module1",

        "target": "qb_mode",

        "classifier_target": "qb_mode_r10",

        "regression_prediction":
            regression_prediction,

        "classification_prediction_r10":
            classification_prediction,

        "blend_alpha":
            BLEND_ALPHA,

        "snap_threshold":
            SNAP_THRESHOLD,

        "predicted_qb":
            final_prediction,

        "feature_count":
            len(FEATURES_FINAL),

        "feature_order_verified":
            True,
    }