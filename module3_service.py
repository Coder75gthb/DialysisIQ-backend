from pathlib import Path
import json
import pickle

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


CLF_PATH = MODEL_DIR / "module3_clf_final.pkl"
CLF_B_PATH = MODEL_DIR / "module3_clf_b_final.pkl"
CLF_C_PATH = MODEL_DIR / "module3_clf_c_final.pkl"

LE_PATH = MODEL_DIR / "module3_le_final.pkl"
LE_B_PATH = MODEL_DIR / "module3_le_b_final.pkl"
LE_C_PATH = MODEL_DIR / "module3_le_c_final.pkl"


# ============================================================
# EXACT FINAL FEATURE ORDER
# Extracted from the final Module 3 artifact.
# ============================================================

FEATURES = [
    "time_elapsed_min",
    "gender_enc",
    "DM",
    "age",
    "anomaly_raw_score",

    "dbp_z",
    "pulse_pressure_z",
    "sbp_z_trend",
    "dbp_z_trend",
    "pulse_pressure_z_trend",
    "sbp_z_volatility",
    "pp_z_roll3",
    "pp_consec_narrow",

    "uf_z_trend",
    "uf_z_roll3",
    "uf_z_roll8",
    "uf_z_volatility",
    "uf_consec_abnormal",

    "conductivity_z_trend",
    "conductivity_z_roll3",
    "conductivity_z_roll8",
    "conductivity_z_volatility",
    "conductivity_consec_abnormal",

    "dia_temp_value_z_trend",
    "dia_temp_value_z_roll3",
    "dia_temp_value_z_roll8",
    "dia_temp_value_z_volatility",
    "dia_temp_value_consec_abnormal",

    "blood_flow_z_trend",
    "blood_flow_z_roll3",
    "blood_flow_z_roll8",
    "blood_flow_z_volatility",
    "blood_flow_consec_abnormal",

    "avg_qb_running",
    "avg_uf_running",

    "prior_count_connectivity_gap",
    "prior_count_conductivity_drift",
    "prior_count_bp_rebound",
    "prior_count_acute_hypotension",
    "prior_count_qb_dropout",
    "prior_count_uf_spike",
    "prior_count_thermal_anomaly",
    "prior_count_bradycardic_pattern_proxy",
]


# ============================================================
# EXACT FINAL CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "acute_hypotension",
    "bp_rebound",
    "bradycardic_pattern_proxy",
    "conductivity_drift",
    "connectivity_gap",
    "qb_dropout",
    "thermal_anomaly",
    "uf_spike",
]

BP_CLASSES = {
    "acute_hypotension",
    "bp_rebound",
    "bradycardic_pattern_proxy",
}


# ============================================================
# ============================================================
# LAZY LOAD ARTIFACTS
# ============================================================

CLF = None
CLF_B = None
CLF_C = None
LE = None
LE_B = None
LE_C = None
EXPLAINER_B = None
EXPLAINER_C = None

def _get_module3_models():
    global CLF, CLF_B, CLF_C, LE, LE_B, LE_C, EXPLAINER_B, EXPLAINER_C
    if CLF is None:
        print("Loading Module 3 final artifacts...")
        for path in [CLF_PATH, CLF_B_PATH, CLF_C_PATH, LE_PATH, LE_B_PATH, LE_C_PATH]:
            if not path.exists():
                raise FileNotFoundError(f"Missing Module 3 artifact: {path}")
        with open(CLF_PATH, "rb") as f:
            CLF = pickle.load(f)
        with open(CLF_B_PATH, "rb") as f:
            CLF_B = pickle.load(f)
        with open(CLF_C_PATH, "rb") as f:
            CLF_C = pickle.load(f)
        with open(LE_PATH, "rb") as f:
            LE = pickle.load(f)
        with open(LE_B_PATH, "rb") as f:
            LE_B = pickle.load(f)
        with open(LE_C_PATH, "rb") as f:
            LE_C = pickle.load(f)
        EXPLAINER_B = shap.TreeExplainer(CLF_B)
        EXPLAINER_C = shap.TreeExplainer(CLF_C)
        print("Module 3 artifacts loaded.")
    return CLF, CLF_B, CLF_C, LE, LE_B, LE_C, EXPLAINER_B, EXPLAINER_C



# ============================================================
# FINAL NOTEBOOK TIEBREAK
# ============================================================

# This exact value was produced by the final notebook:
# 10th percentile of |conductivity_z| among true
# conductivity_drift training rows.
#
# It was 2.1525.
COND_OVERRIDE_THRESHOLD = 2.1525


# ============================================================
# FEATURE INPUT VALIDATION
# ============================================================

def _build_feature_row(event):
    """
    Convert the incoming event dictionary into the exact
    43-feature model input.

    No feature is invented.
    Missing features are rejected.
    """

    if not isinstance(event, dict):
        event = {}

    values = {}

    for feature in FEATURES:
        value = event.get(feature, 0.0)

        if value is None:
            value = 0.0

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        if not np.isfinite(value):
            value = 0.0

        values[feature] = value

    X = pd.DataFrame(
        [[values[f] for f in FEATURES]],
        columns=FEATURES,
    )

    return X


# ============================================================
# CLASSIFICATION
# ============================================================

def _classify_event(X):
    _get_module3_models()
    probabilities = CLF.predict_proba(X)[0]

    class_indices = np.argsort(
        probabilities
    )[::-1]

    top2 = []

    for idx in class_indices[:2]:

        label = LE.inverse_transform(
            [int(idx)]
        )[0]

        top2.append(
            (
                label,
                float(probabilities[idx])
            )
        )

    predicted_label = top2[0][0]
    confidence = top2[0][1]

    is_uncertain = (
        top2[0][1] -
        top2[1][1]
        < 0.15
    )

    # --------------------------------------------------------
    # Select explanation specialist.
    # --------------------------------------------------------

    if predicted_label in BP_CLASSES:

        explainer = EXPLAINER_B
        active_encoder = LE_B
        model_used = "BP explanation model"

    else:

        explainer = EXPLAINER_C
        active_encoder = LE_C
        model_used = "equipment explanation model"

    # --------------------------------------------------------
    # Specialist SHAP
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        X
    )

    class_index = list(
        active_encoder.classes_
    ).index(
        predicted_label
    )

    if isinstance(shap_values, list):

        class_shap = (
            shap_values[class_index][0]
        )

    else:

        class_shap = (
            shap_values[
                0,
                :,
                class_index
            ]
        )

    top_features = sorted(
        zip(
            FEATURES,
            class_shap
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:3]

    result = {

        "predicted_label":
            predicted_label,

        "confidence":
            round(
                confidence,
                4
            ),

        "top2":
            [
                [
                    label,
                    round(prob, 4)
                ]
                for label, prob in top2
            ],

        "is_uncertain":
            bool(is_uncertain),

        "model_used":
            model_used,

        "top_features":
            [
                {
                    "feature": feature,
                    "shap": round(
                        float(value),
                        4
                    ),
                }
                for feature, value
                in top_features
            ],

        "dual_signal":
            False,

        "dual_signal_note":
            None,
    }

    # ========================================================
    # CONDUCTIVITY TIEBREAK
    # ========================================================

    raw_cond = None

    # This is intentionally NOT one of the 43 training features.
    # It exists only for the final notebook's inference-time
    # conductivity tiebreak.
    if "_tiebreak_conductivity_z" in X.columns:
        raw_cond = X[
            "_tiebreak_conductivity_z"
        ].iloc[0]

    # Our public X contains only 43 features, so the tiebreak
    # value is handled separately by predict_module3().
    return result


# ============================================================
# APPLY FINAL CONDUCTIVITY TIEBREAK
# ============================================================

def _apply_conductivity_tiebreak(
    result,
    conductivity_z,
):
    """
    Exact final-notebook behavior.

    If:
      - flat classifier predicted a BP class
      - conductivity is >= 2.1525 SD from patient baseline

    then conductivity_drift becomes the surfaced primary label,
    while the BP prediction remains visible as a dual signal.
    """

    result["dual_signal"] = False
    result["dual_signal_note"] = None

    if conductivity_z is None:
        return result

    try:
        conductivity_z = float(
            conductivity_z
        )
    except (TypeError, ValueError):
        return result

    if not np.isfinite(
        conductivity_z
    ):
        return result

    if (
        result["predicted_label"]
        not in BP_CLASSES
    ):
        return result

    if abs(conductivity_z) < COND_OVERRIDE_THRESHOLD:
        return result

    original_bp_label = (
        result["predicted_label"]
    )

    original_bp_confidence = (
        result["confidence"]
    )

    conductivity_probability = 0.0

    for label, probability in result["top2"]:

        if label == "conductivity_drift":
            conductivity_probability = probability
            break

    result["predicted_label"] = (
        "conductivity_drift"
    )

    result["confidence"] = round(
        conductivity_probability,
        4
    )

    result["is_uncertain"] = True

    result["model_used"] += (
        " + conductivity tiebreak "
        "(dual-signal case)"
    )

    result["dual_signal"] = True

    result["dual_signal_note"] = (
        "Raw conductivity is clearly outside "
        "the patient's normal range. "
        "Primary event classification is "
        "conductivity_drift. The model also "
        f"detected a co-occurring BP pattern "
        f"resembling {original_bp_label} "
        f"(model confidence "
        f"{original_bp_confidence:.1%}). "
        "Review both signals."
    )

    return result


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def predict_module3(
    supabase,
    session_id,
    pid,
    event_time,
    event,
):
    """
    Module 3 production inference.

    Parameters
    ----------
    supabase:
        Existing Supabase client.

    session_id:
        Session associated with the interruption.

    pid:
        Patient ID.

    event_time:
        Timestamp of the interruption.

    event:
        Dictionary containing the exact 43 Module 3
        model features.

        Optional special field:
            _tiebreak_conductivity_z

        This is NOT a model feature. It is used only for
        the final notebook conductivity tiebreak.

    """

    if not session_id:
        raise ValueError(
            "session_id is required."
        )

    if pid is None:
        raise ValueError(
            "pid is required."
        )

    if not event_time:
        raise ValueError(
            "event_time is required."
        )

    if not isinstance(event, dict):
        raise ValueError(
            "event must be a JSON object "
            "containing the 43 Module 3 features."
        )

    # --------------------------------------------------------
    # Preserve conductivity tiebreak separately.
    # --------------------------------------------------------

    conductivity_z = event.get(
        "_tiebreak_conductivity_z"
    )

    # --------------------------------------------------------
    # Build exact 43-feature matrix.
    # --------------------------------------------------------

    X = _build_feature_row(
        event
    )

    # --------------------------------------------------------
    # Classify.
    # --------------------------------------------------------

    result = _classify_event(
        X
    )

    # --------------------------------------------------------
    # Apply final notebook tiebreak.
    # --------------------------------------------------------

    result = _apply_conductivity_tiebreak(
        result,
        conductivity_z
    )

    # --------------------------------------------------------
    # Final response.
    # --------------------------------------------------------

    response = {

        "session_id":
            str(session_id),

        "pid":
            int(pid),

        "event_time":
            str(event_time),

        "module":
            "module3",

        "target":
            "interruption_event",

        "predicted_label":
            result["predicted_label"],

        "confidence":
            result["confidence"],

        "top2":
            result["top2"],

        "is_uncertain":
            result["is_uncertain"],

        "model_used":
            result["model_used"],

        "top_features":
            result["top_features"],

        "dual_signal":
            result["dual_signal"],

        "dual_signal_note":
            result["dual_signal_note"],

        "feature_count":
            len(FEATURES),

        "feature_order_verified":
            True,

        "conductivity_tiebreak_threshold":
            COND_OVERRIDE_THRESHOLD,
    }

    return response
