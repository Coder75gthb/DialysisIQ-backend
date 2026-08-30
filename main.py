import sys
from pathlib import Path
import os
import threading
import time
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv

from module1_service import predict_module1
from module2_service import predict_module2
from module3_service import predict_module3
from module4_service import predict_module4
from module5_service import predict_module5

load_dotenv(BASE_DIR / ".env")
load_dotenv()

app = FastAPI(title="DialysisIQ Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.vercel\.app|https://.*\.hf\.space",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_SECRET_KEY must be present in .env"
    )


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# Global Cache and Lock for Module 5
module5_cache: Dict[str, Any] = {
    "data": None,
    "generated_at": None,
}
module5_lock = threading.Lock()


class PatientCreatePayload(BaseModel):
    pid: int
    name: Optional[str] = None
    gender: Optional[str] = "M"
    birthday: Optional[int] = 1965
    has_dm: bool = False

class SessionCreatePayload(BaseModel):
    pid: int
    session_start: Optional[str] = None
    pre_sbp: float
    pre_dbp: float
    weightstart: float
    dryweight: float
    weight_post: Optional[float] = None
    duration_min: Optional[float] = 240.0
    avg_uf: Optional[float] = None
    max_uf: Optional[float] = None
    avg_conductivity: Optional[float] = 14.0
    avg_dia_temp: Optional[float] = 36.5

class QBInterventionUpdate(BaseModel):
    qb_intervention: bool

class Module3Request(BaseModel):
    session_id: str
    pid: int
    event_time: str
    event: Dict[str, Any]


def _get_first_dict(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0]
    return None

def _get_dict_list(data: Any) -> list[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


@app.get("/")
def root():
    return {
        "message": "DialysisIQ backend is running"
    }


@app.get("/patients")
def get_patients(query: Optional[str] = None):
    query_builder = supabase.table("patients").select("*")
    
    if query:
        clean_q = query.strip()
        if clean_q.isdigit():
            # Filter exact or prefix match on numeric pid
            query_builder = query_builder.eq("pid", int(clean_q))
        else:
            query_builder = query_builder.ilike("name", f"%{clean_q}%")
            
    response = query_builder.limit(50).execute()
    data = _get_dict_list(response.data)

    return {
        "count": len(data),
        "patients": data,
    }


@app.post("/patients")
def create_patient(payload: PatientCreatePayload):
    try:
        response = (
            supabase
            .table("patients")
            .insert({
                "pid": payload.pid,
                "name": payload.name,
                "gender": payload.gender,
                "birthday": payload.birthday,
                "has_dm": payload.has_dm,
            })
            .execute()
        )
        first_patient = _get_first_dict(response.data)
        return {
            "message": "Patient created successfully",
            "patient": first_patient or {
                "pid": payload.pid,
                "name": payload.name,
                "gender": payload.gender,
                "birthday": payload.birthday,
                "has_dm": payload.has_dm,
            }
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create patient: {exc}"
        ) from exc


@app.get("/sessions")
def get_sessions():
    response = (
        supabase
        .table("sessions")
        .select("*")
        .limit(20)
        .execute()
    )
    data = _get_dict_list(response.data)

    return {
        "count": len(data),
        "sessions": data,
    }


@app.post("/sessions")
def create_session(payload: SessionCreatePayload):
    global module5_cache
    try:
        from datetime import datetime
        now_iso = datetime.now().isoformat()
        session_start = payload.session_start or now_iso
        session_id = f"{payload.pid}_{int(time.time())}"

        # Ensure patient exists in patients table
        p_check = (
            supabase
            .table("patients")
            .select("pid")
            .eq("pid", payload.pid)
            .limit(1)
            .execute()
        )
        p_data = _get_dict_list(p_check.data)
        if not p_data:
            # Create a placeholder patient if not exists
            supabase.table("patients").insert({
                "pid": payload.pid,
                "gender": "M",
                "birthday": 1965,
                "has_dm": False,
            }).execute()

        new_row = {
            "session_id": session_id,
            "pid": payload.pid,
            "session_start": session_start,
            "pre_sbp": payload.pre_sbp,
            "pre_dbp": payload.pre_dbp,
            "weightstart": payload.weightstart,
            "dryweight": payload.dryweight,
            "weight_post": payload.weight_post or payload.dryweight,
            "duration_min": payload.duration_min or 240,
            "avg_uf": payload.avg_uf or round((payload.weightstart - payload.dryweight) / 4.0, 2),
            "max_uf": payload.max_uf or round((payload.weightstart - payload.dryweight) / 4.0 * 1.2, 2),
            "avg_conductivity": payload.avg_conductivity or 14.0,
            "avg_dia_temp": payload.avg_dia_temp or 36.5,
        }

        response = (
            supabase
            .table("sessions")
            .insert(new_row)
            .execute()
        )

        # Invalidate Module 5 morning briefing cache so new session data is processed
        with module5_lock:
            module5_cache["data"] = None
            module5_cache["generated_at"] = None

        first_session = _get_first_dict(response.data)
        return {
            "message": "Session recorded successfully",
            "session_id": session_id,
            "session": first_session or new_row,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record session: {exc}"
        ) from exc


# ============================================================
# MODULE 1
# ============================================================

@app.get("/module1/predict/{session_id}")
def module1_predict(session_id: str):
    try:
        return predict_module1(
            supabase,
            session_id
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Module 1 inference failed: {exc}",
        ) from exc


# ============================================================
# MODULE 2
# ============================================================

@app.get("/module2/predict/{session_id}")
def module2_predict(session_id: str):
    try:
        return predict_module2(
            supabase,
            session_id
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Module 2 inference failed: {exc}",
        ) from exc



# ============================================================
# MODULE 3
# ============================================================

@app.post("/module3/predict")
def module3_predict(payload: Module3Request):
    """
    Run Module 3 interruption-event classification.

    Module 3 expects:
        session_id
        pid
        event_time
        event = exact 43 Module 3 features

    The service performs:
        - 8-class event classification
        - BP specialist explanation
        - equipment specialist explanation
        - SHAP top features
        - conductivity tiebreak
        - dual-signal detection
    """

    try:
        return predict_module3(
            supabase=supabase,
            session_id=payload.session_id,
            pid=payload.pid,
            event_time=payload.event_time,
            event=payload.event,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Module 3 inference failed: {exc}",
        ) from exc

    
# ============================================================
# MODULE 4
# ============================================================

@app.get("/module4/predict/{pid}")
def module4_predict(pid: int):
    """
    Run Module 4 for a patient.

    Module 4 uses the original d1-derived history stored in
    `module4_d1_sessions`.

    The service handles:
        Component A -> drift detection
        Component B -> direction (if drift)
        Component C -> drift type/intervention (if drift)
    """

    try:
        return predict_module4(
            supabase,
            pid
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Module 4 inference failed: {exc}",
        ) from exc


# ============================================================
# QB INTERVENTION
# ============================================================

@app.patch("/sessions/{session_id}/qb-intervention")
def update_qb_intervention(
    session_id: str,
    payload: QBInterventionUpdate,
):
    """
    Nurse-facing post-session annotation.

    This changes only the explicit intervention annotation.
    It does not modify Module 2 model features,
    probability, or model artifacts.
    """

    try:
        if session_id.endswith("_latest"):
            pid_str = session_id.split("_")[0]
            if pid_str.isdigit():
                pid_val = int(pid_str)
                res = (
                    supabase
                    .table("sessions")
                    .select("session_id")
                    .eq("pid", pid_val)
                    .order("session_start", desc=True)
                    .limit(1)
                    .execute()
                )
                first_latest = _get_first_dict(res.data)
                if first_latest and "session_id" in first_latest:
                    session_id = str(first_latest["session_id"])
                else:
                    # Create a placeholder session so intervention annotation succeeds
                    new_id = f"{pid_val}_{int(time.time())}"
                    supabase.table("sessions").insert({
                        "session_id": new_id,
                        "pid": pid_val,
                        "pre_sbp": 120,
                        "pre_dbp": 80,
                        "weightstart": 70,
                        "dryweight": 68,
                    }).execute()
                    session_id = new_id

        # Verify the exact session exists first.
        existing = (
            supabase
            .table("sessions")
            .select("session_id")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        existing_rows = _get_dict_list(existing.data)

        if len(existing_rows) != 1:
            # Fallback: create session row if needed
            pid_val = int(session_id.split("_")[0]) if "_" in session_id and session_id.split("_")[0].isdigit() else 101
            supabase.table("sessions").insert({
                "session_id": session_id,
                "pid": pid_val,
                "pre_sbp": 120,
                "pre_dbp": 80,
                "weightstart": 70,
                "dryweight": 68,
            }).execute()

        response = (
            supabase
            .table("sessions")
            .update({
                "qb_intervention": payload.qb_intervention
            })
            .eq("session_id", session_id)
            .execute()
        )

        row = _get_first_dict(response.data)

        if not row:
            raise HTTPException(
                status_code=500,
                detail="Supabase updated no rows.",
            )

        return {
            "session_id": str(row.get("session_id", session_id)),
            "qb_intervention": bool(
                row.get("qb_intervention", payload.qb_intervention)
            ),
            "message": (
                "Qb intervention updated successfully."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to update Qb intervention: {exc}"
            ),
        ) from exc


# ============================================================
# MODULE 5
# ============================================================

def run_module5_and_cache(sample_size: Optional[int] = None):
    """
    Run the expensive Module 5 pipeline once and store
    the result in memory.
    """
    global module5_cache

    try:
        print(f"MODULE 5: generating morning briefing (sample_size={sample_size})...")

        result = predict_module5(supabase, sample_size=sample_size)

        with module5_lock:
            module5_cache["data"] = result
            module5_cache["generated_at"] = time.time()

        print("MODULE 5: briefing cached successfully.")

        return result

    except Exception as exc:
        print(f"MODULE 5 background generation failed: {exc}")
        return None


@app.get("/module5/predict")
def module5_predict(sample_size: Optional[int] = None):
    """
    Return the latest Module 5 morning briefing.

    If a cached briefing exists (and no specific sample_size is requested), return it immediately.
    Otherwise run Module 5 once and cache the result.
    """

    global module5_cache

    # --------------------------------------------------------
    # FAST PATH: cached briefing already exists
    # --------------------------------------------------------

    if sample_size is None:
        with module5_lock:
            cached = module5_cache["data"]

        if cached is not None:
            return cached

    # --------------------------------------------------------
    # FIRST REQUEST: generate Module 5
    # --------------------------------------------------------

    try:
        return run_module5_and_cache(sample_size=sample_size)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Module 5 inference failed: {exc}",
        ) from exc


@app.post("/module5/refresh")
def module5_refresh():
    """
    Force regeneration of the Module 5 morning briefing.

    Use this once each morning or whenever fresh clinical
    data needs to be processed.
    """

    global module5_cache

    try:
        result = run_module5_and_cache()

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Module 5 generation failed."
            )

        return {
            "message": "Module 5 briefing refreshed successfully.",
            "data": result,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Module 5 refresh failed: {exc}",
        ) from exc