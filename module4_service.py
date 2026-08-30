from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

COMP_A_MODEL = MODEL_DIR / "module4_compA_clf.pkl"
COMP_A_FEATURES = MODEL_DIR / "module4_compA_features.pkl"
DIRECTION_MODEL = MODEL_DIR / "module4_direction_clf_final.joblib"
DIRECTION_FEATURES = MODEL_DIR / "module4_direction_features_final.joblib"
COMP_C_MODEL = MODEL_DIR / "module4_drifttype_clf_v8.joblib"
COMP_C_LE = MODEL_DIR / "module4_drifttype_le_v8.joblib"
COMP_C_FEATURES = MODEL_DIR / "module4_drifttype_features_v8.joblib"
INTERVENTIONS = MODEL_DIR / "module4_interventions_v8.joblib"

for path in [
    COMP_A_MODEL, COMP_A_FEATURES, DIRECTION_MODEL,
    DIRECTION_FEATURES, COMP_C_MODEL, COMP_C_LE,
    COMP_C_FEATURES, INTERVENTIONS
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Module 4 artifact: {path}")

COMP_A = joblib.load(COMP_A_MODEL)
COMP_A_FEATURES_LIST = list(joblib.load(COMP_A_FEATURES))
DIRECTION = joblib.load(DIRECTION_MODEL)
DIRECTION_FEATURES_LIST = list(joblib.load(DIRECTION_FEATURES))
COMP_C = joblib.load(COMP_C_MODEL)
COMP_C_LE = joblib.load(COMP_C_LE)
COMP_C_FEATURES_LIST = list(joblib.load(COMP_C_FEATURES))
INTERVENTION_MAP = joblib.load(INTERVENTIONS)

DRIFT_THRESHOLD = 0.40
MIN_WINDOW = 12
WINDOW_SIZE = 30
RECENT_N = 8

print("Loading Module 4 final artifacts...")
print(f"Component A features: {len(COMP_A_FEATURES_LIST)}")
print(f"Component C features: {len(COMP_C_FEATURES_LIST)}")
print(f"Direction features: {len(DIRECTION_FEATURES_LIST)}")
print(f"Drift threshold: {DRIFT_THRESHOLD}")
print(f"Drift types: {list(COMP_C_LE.classes_)}")
print("Module 4 artifacts loaded.")


def _fetch_patient_sessions(supabase, pid: int) -> pd.DataFrame:
    rows = []
    page_size = 1000
    start = 0

    while True:
        response = (
            supabase.table("module4_d1_sessions")
            .select(
                "pid,keyindate,weightstart,weightend,dryweight,temperature"
            )
            .eq("pid", pid)
            .order("keyindate", desc=False)
            .range(start, start + page_size - 1)
            .execute()
        )
        page = response.data or []
        rows.extend(page)

        if len(page) < page_size:
            break
        start += page_size

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["keyindate"] = pd.to_datetime(df["keyindate"], errors="coerce")

    for col in ["weightstart", "weightend", "dryweight", "temperature"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("keyindate").reset_index(drop=True)


def _prepare_weight_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(
        subset=["weightstart", "weightend", "dryweight"]
    ).copy()

    if df.empty:
        return df

    df["idwg"] = (
        df["weightstart"] - df["dryweight"]
    ).clip(lower=0)

    df["fluid_removed"] = (
        df["weightstart"] - df["weightend"]
    ).clip(lower=0)

    df["completion_ratio"] = np.where(
        df["idwg"] > 0,
        (
            df["fluid_removed"] / df["idwg"]
        ).clip(0, 1.5),
        1.0,
    )

    df["post_excess"] = (
        df["weightend"] - df["dryweight"]
    )

    return df


def _window_features_ab(window, current_dw):
    n = len(window)
    if n < MIN_WINDOW:
        return {}

    w = window.reset_index(drop=True)
    recent = w.iloc[2 * n // 3:]
    early = w.iloc[: n // 3]

    idwg_values = (
        w["idwg"].fillna(w["idwg"].mean()).values
    )
    mean_fluid_removed = w["fluid_removed"].mean()

    features = {
        "mean_idwg": w["idwg"].mean(),
        "std_idwg": w["idwg"].std(),
        "trend_idwg": (
            np.polyfit(range(n), idwg_values, 1)[0]
            if n > 2 else 0.0
        ),
        "max_idwg_in_window": w["idwg"].max(),
        "recent_mean_idwg": recent["idwg"].mean(),
        "early_mean_idwg": early["idwg"].mean(),
        "idwg_acceleration": (
            recent["idwg"].mean() - early["idwg"].mean()
        ),
        "mean_completion_ratio": (
            w["completion_ratio"].clip(0, 3).mean()
        ),
        "pct_under_removed": (
            w["completion_ratio"] < 0.85
        ).mean(),
        "max_completion_deficit": (
            1 - w["completion_ratio"].clip(0, 3)
        ).max(),
        "recent_completion_ratio": (
            recent["completion_ratio"].clip(0, 3).mean()
        ),
        "early_completion_ratio": (
            early["completion_ratio"].clip(0, 3).mean()
        ),
        "completion_acceleration": (
            recent["completion_ratio"].clip(0, 3).mean()
            - early["completion_ratio"].clip(0, 3).mean()
        ),
        "std_weightstart": w["weightstart"].std(),
        "mean_fluid_removed": mean_fluid_removed,
        "cv_fluid_removed": (
            w["fluid_removed"].std() / mean_fluid_removed
            if mean_fluid_removed else np.nan
        ),
        "mean_weightend_minus_dryweight": (
            w["weightend"] - current_dw
        ).mean(),
        "std_weightend_minus_dryweight": (
            w["weightend"] - current_dw
        ).std(),
        "weightstart_to_dryweight_ratio": (
            w["weightstart"].mean() / current_dw
            if current_dw else np.nan
        ),
        "rolling_std_idwg_last6": w["idwg"].tail(6).std(),
        "rolling_std_completion_last6": (
            w["completion_ratio"].clip(0, 3).tail(6).std()
        ),
        "range_weightend_minus_dryweight": (
            (w["weightend"] - current_dw).max()
            - (w["weightend"] - current_dw).min()
        ),
        "pct_sessions_above_dryweight": (
            w["weightend"] > current_dw
        ).mean(),
        "mean_temperature": w["temperature"].mean(),
        "std_temperature": w["temperature"].std(),
    }

    features.update({
        "mean_idwg_x_completion": (
            w["idwg"] * w["completion_ratio"].clip(0, 3)
        ).mean(),
        "n_sessions_in_window": n,
        "n_sessions_above_target_idwg": (
            w["idwg"] > 2.5
        ).sum(),
        "max_consecutive_under_removed": (
            (w["completion_ratio"] < 0.85)
            .astype(int)
            .groupby(
                (w["completion_ratio"] >= 0.85)
                .astype(int)
                .cumsum()
            )
            .sum()
            .max()
        ),
    })
    return features


def _window_features_c(window):
    n = len(window)
    if n < MIN_WINDOW:
        return {}

    half = n // 2
    early = window.iloc[:half]
    late = window.iloc[half:]
    recent = window.tail(RECENT_N)

    fr_all = window["fluid_removed"].values
    cr_all = window["completion_ratio"].values
    post_all = window["post_excess"].values

    fr_recent = recent["fluid_removed"].values
    cr_recent = recent["completion_ratio"].values
    post_recent = recent["post_excess"].values

    fr_early = early["fluid_removed"].values
    fr_late = late["fluid_removed"].values
    cr_early = early["completion_ratio"].values
    cr_late = late["completion_ratio"].values

    idwg_all = window["idwg"].values
    eff_all = fr_all / (idwg_all + 1e-6)
    eff_recent = fr_recent / (recent["idwg"].values + 1e-6)
    eff_early = fr_early / (early["idwg"].values + 1e-6)
    eff_late = fr_late / (late["idwg"].values + 1e-6)

    return {
        "mean_fr": np.mean(fr_all),
        "std_fr": np.std(fr_all),
        "recent_mean_fr": np.mean(fr_recent),
        "fr_trend": np.mean(fr_late) - np.mean(fr_early),
        "fr_slope": float(np.polyfit(range(n), fr_all, 1)[0]),
        "mean_completion": np.mean(cr_all),
        "std_completion": np.std(cr_all),
        "recent_completion": np.mean(cr_recent),
        "pct_under_removed": np.mean(cr_all < 0.85),
        "max_deficit": float(np.max(1 - cr_all)),
        "completion_trend": np.mean(cr_late) - np.mean(cr_early),
        "rolling_std_cr6": float(
            window["completion_ratio"].tail(6).std()
        ),
        "mean_post_excess": np.mean(post_all),
        "std_post_excess": np.std(post_all),
        "recent_post_excess": np.mean(post_recent),
        "pct_above_target": np.mean(post_all > 0.3),
        "post_excess_trend": (
            np.mean(post_all[half:]) -
            np.mean(post_all[:half])
        ),
        "max_post_excess": float(np.max(post_all)),
        "mean_efficiency": np.mean(eff_all),
        "std_efficiency": np.std(eff_all),
        "recent_efficiency": np.mean(eff_recent),
        "efficiency_trend": (
            np.mean(eff_late) - np.mean(eff_early)
        ),
        "pct_low_efficiency": np.mean(eff_all < 0.80),
        "fr_completion_div": (
            float(np.polyfit(range(n), fr_all, 1)[0])
            - float(np.polyfit(range(n), cr_all, 1)[0])
        ),
        "n_sessions": n,
    }


def _model_frame(features, feature_list):
    missing = [
        f for f in feature_list
        if f not in features
    ]
    if missing:
        raise RuntimeError(
            "Module 4 feature construction is incomplete. "
            f"Missing features: {missing}"
        )

    X = pd.DataFrame([features])[feature_list]

    return (
        X
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _component_a(weight_df, current_dw):
    window = weight_df.tail(WINDOW_SIZE).copy()

    if len(window) < MIN_WINDOW:
        return {
            "available": False,
            "drift_probability": None,
            "drift_detected": False,
            "reason": "insufficient_valid_weight_data",
        }

    features = _window_features_ab(
        window,
        current_dw
    )

    X = _model_frame(
        features,
        COMP_A_FEATURES_LIST
    )

    probability = float(
        COMP_A.predict_proba(X)[0, 1]
    )

    return {
        "available": True,
        "drift_probability": probability,
        "drift_detected": bool(
            probability >= DRIFT_THRESHOLD
        ),
        "reason": (
            "ok"
            if probability >= DRIFT_THRESHOLD
            else "below_drift_threshold"
        ),
    }


def _component_b(weight_df, current_dw):
    if "temperature" not in weight_df.columns:
        return {
            "available": False,
            "direction": None,
            "confidence": None,
            "reason": "missing_training_field_temperature",
            "required_database_field": "temperature",
        }

    window = weight_df.tail(WINDOW_SIZE).copy()

    if len(window) < MIN_WINDOW:
        return {
            "available": False,
            "direction": None,
            "confidence": None,
            "reason": "insufficient_valid_weight_data",
        }

    features = _window_features_ab(
        window,
        current_dw
    )

    needs_temperature = (
        "mean_temperature" in DIRECTION_FEATURES_LIST
        or "std_temperature" in DIRECTION_FEATURES_LIST
    )

    if needs_temperature and window["temperature"].notna().sum() == 0:
        return {
            "available": False,
            "direction": None,
            "confidence": None,
            "reason": "temperature_unavailable_for_window",
        }

    X = _model_frame(
        features,
        DIRECTION_FEATURES_LIST
    )

    probability_up = float(
        DIRECTION.predict_proba(X)[0, 1]
    )

    direction = (
        "UP"
        if probability_up >= 0.5
        else "DOWN"
    )

    confidence = (
        probability_up
        if direction == "UP"
        else 1.0 - probability_up
    )

    return {
        "available": True,
        "direction": direction,
        "confidence": float(confidence),
        "probability_up": probability_up,
        "reason": "ok",
    }


def _component_c(weight_df):
    window = weight_df.tail(WINDOW_SIZE).copy()

    if len(window) < MIN_WINDOW:
        return {
            "available": False,
            "drift_type": None,
            "reason": "insufficient_valid_weight_data",
            "intervention": None,
        }

    features = _window_features_c(window)

    X = _model_frame(
        features,
        COMP_C_FEATURES_LIST
    )

    encoded = COMP_C.predict(X)[0]

    drift_type = str(
        COMP_C_LE.inverse_transform([encoded])[0]
    )

    intervention = INTERVENTION_MAP.get(
        drift_type
    )

    return {
        "available": True,
        "drift_type": drift_type,
        "reason": "ok",
        "intervention": intervention,
    }


def predict_module4(supabase, pid: int):
    history = _fetch_patient_sessions(
        supabase,
        pid
    )

    if history.empty:
        raise ValueError(
            f"No d1 sessions found for pid={pid}."
        )

    weight_history = _prepare_weight_history(
        history
    )

    if weight_history.empty:
        return {
            "module": "module4",
            "pid": int(pid),
            "available": False,
            "reason": (
                "no sessions with weightstart, "
                "weightend and dryweight"
            ),
            "component_a": None,
            "component_b_direction": None,
            "component_c_type": None,
            "intervention": None,
        }

    current_dw = float(
        weight_history.iloc[-1]["dryweight"]
    )

    component_a = _component_a(
        weight_history,
        current_dw
    )

    if component_a["drift_detected"]:
        component_b = _component_b(
            weight_history,
            current_dw
        )
        component_c = _component_c(
            weight_history
        )
    else:
        component_b = {
            "available": True,
            "direction": None,
            "confidence": None,
            "reason": "not_run_component_a_not_flagged",
        }
        component_c = {
            "available": True,
            "drift_type": None,
            "reason": "not_run_component_a_not_flagged",
            "intervention": None,
        }

    return {
        "module": "module4",
        "pid": int(pid),
        "available": True,
        "sessions_available": int(len(weight_history)),
        "sessions_used": int(
            min(len(weight_history), WINDOW_SIZE)
        ),
        "current_dryweight": current_dw,
        "component_a": component_a,
        "component_b_direction": component_b,
        "component_c_type": component_c,
        "intervention": component_c.get("intervention"),
        "artifact_feature_counts": {
            "component_a": len(COMP_A_FEATURES_LIST),
            "component_b_direction": len(DIRECTION_FEATURES_LIST),
            "component_c_type": len(COMP_C_FEATURES_LIST),
        },
    }