import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import Dict, Any, List
from dotenv import load_dotenv
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL and SUPABASE_SECRET_KEY required.")
    sys.exit(1)

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

DEMO_PATIENTS: List[Dict[str, Any]] = [
    {
        "pid": 100001,
        "name": "Robert Chen",
        "gender": "M",
        "birthday": 1958,
        "has_dm": True,
        "profile_type": "HIGH_FLUID_DRIFT",
        "pre_sbp": 94.0,
        "pre_dbp": 60.0,
        "dryweight": 70.0,
        "idwg": 3.8,
        "hypo_freq": 0.9,
        "ktv": 1.10,
    },
    {
        "pid": 100002,
        "name": "Eleanor Vance",
        "gender": "F",
        "birthday": 1964,
        "has_dm": True,
        "profile_type": "HIGH_BODY_DRIFT",
        "pre_sbp": 96.0,
        "pre_dbp": 62.0,
        "dryweight": 58.0,
        "idwg": 3.2,
        "hypo_freq": 0.8,
        "ktv": 1.15,
    },
    {
        "pid": 100003,
        "name": "Marcus Brody",
        "gender": "M",
        "birthday": 1952,
        "has_dm": False,
        "profile_type": "HIGH_SEVERE_HYPO",
        "pre_sbp": 90.0,
        "pre_dbp": 56.0,
        "dryweight": 82.0,
        "idwg": 4.1,
        "hypo_freq": 0.95,
        "ktv": 1.05,
    },
    {
        "pid": 100004,
        "name": "Sarah Jenkins",
        "gender": "F",
        "birthday": 1971,
        "has_dm": True,
        "profile_type": "MED_FLUID_DRIFT",
        "pre_sbp": 112.0,
        "pre_dbp": 70.0,
        "dryweight": 64.0,
        "idwg": 2.9,
        "hypo_freq": 0.5,
        "ktv": 1.25,
    },
    {
        "pid": 100005,
        "name": "David Miller",
        "gender": "M",
        "birthday": 1960,
        "has_dm": False,
        "profile_type": "MED_BODY_DRIFT",
        "pre_sbp": 118.0,
        "pre_dbp": 74.0,
        "dryweight": 76.0,
        "idwg": 2.7,
        "hypo_freq": 0.4,
        "ktv": 1.12,
    },
    {
        "pid": 100006,
        "name": "Maria Rodriguez",
        "gender": "F",
        "birthday": 1968,
        "has_dm": True,
        "profile_type": "MED_STABLE",
        "pre_sbp": 114.0,
        "pre_dbp": 72.0,
        "dryweight": 62.0,
        "idwg": 3.1,
        "hypo_freq": 0.4,
        "ktv": 1.30,
    },
    {
        "pid": 100007,
        "name": "James Wilson",
        "gender": "M",
        "birthday": 1955,
        "has_dm": False,
        "profile_type": "MED_STABLE",
        "pre_sbp": 110.0,
        "pre_dbp": 68.0,
        "dryweight": 78.0,
        "idwg": 2.8,
        "hypo_freq": 0.4,
        "ktv": 1.18,
    },
    {
        "pid": 100008,
        "name": "Patricia Taylor",
        "gender": "F",
        "birthday": 1975,
        "has_dm": False,
        "profile_type": "LOW_STABLE",
        "pre_sbp": 128.0,
        "pre_dbp": 78.0,
        "dryweight": 60.0,
        "idwg": 1.8,
        "hypo_freq": 0.0,
        "ktv": 1.45,
    },
    {
        "pid": 100009,
        "name": "Thomas Anderson",
        "gender": "M",
        "birthday": 1962,
        "has_dm": True,
        "profile_type": "LOW_STABLE",
        "pre_sbp": 132.0,
        "pre_dbp": 82.0,
        "dryweight": 85.0,
        "idwg": 2.0,
        "hypo_freq": 0.0,
        "ktv": 1.35,
    },
    {
        "pid": 100010,
        "name": "Elizabeth Scott",
        "gender": "F",
        "birthday": 1980,
        "has_dm": False,
        "profile_type": "LOW_STABLE",
        "pre_sbp": 125.0,
        "pre_dbp": 76.0,
        "dryweight": 55.0,
        "idwg": 1.5,
        "hypo_freq": 0.0,
        "ktv": 1.52,
    },
    {
        "pid": 100011,
        "name": "William Harris",
        "gender": "M",
        "birthday": 1950,
        "has_dm": False,
        "profile_type": "LOW_STABLE",
        "pre_sbp": 130.0,
        "pre_dbp": 80.0,
        "dryweight": 72.0,
        "idwg": 2.1,
        "hypo_freq": 0.0,
        "ktv": 1.40,
    },
    {
        "pid": 100012,
        "name": "Linda Martinez",
        "gender": "F",
        "birthday": 1973,
        "has_dm": True,
        "profile_type": "LOW_STABLE",
        "pre_sbp": 122.0,
        "pre_dbp": 75.0,
        "dryweight": 66.0,
        "idwg": 1.9,
        "hypo_freq": 0.0,
        "ktv": 1.38,
    },
    {
        "pid": 100013,
        "name": "Richard White",
        "gender": "M",
        "birthday": 1948,
        "has_dm": True,
        "profile_type": "HIGH_FLUID_DRIFT",
        "pre_sbp": 92.0,
        "pre_dbp": 58.0,
        "dryweight": 74.0,
        "idwg": 3.6,
        "hypo_freq": 0.9,
        "ktv": 1.08,
    },
    {
        "pid": 100014,
        "name": "Barbara Clark",
        "gender": "F",
        "birthday": 1966,
        "has_dm": False,
        "profile_type": "MED_BODY_DRIFT",
        "pre_sbp": 115.0,
        "pre_dbp": 72.0,
        "dryweight": 59.0,
        "idwg": 2.6,
        "hypo_freq": 0.4,
        "ktv": 1.22,
    },
    {
        "pid": 100015,
        "name": "Charles Lewis",
        "gender": "M",
        "birthday": 1957,
        "has_dm": False,
        "profile_type": "LOW_STABLE",
        "pre_sbp": 126.0,
        "pre_dbp": 76.0,
        "dryweight": 80.0,
        "idwg": 1.7,
        "hypo_freq": 0.0,
        "ktv": 1.48,
    },
]


def purge_old_dataset():
    print("Purging non-demo training dataset rows from Supabase...")
    demo_pids = set(int(p["pid"]) for p in DEMO_PATIENTS)

    all_pids: List[int] = []
    page_size = 1000
    start = 0
    while True:
        res = sb.table("patients").select("pid").range(start, start + page_size - 1).execute()
        page = res.data or []
        for row in page:
            p_val = int(row["pid"])
            if p_val not in demo_pids:
                all_pids.append(p_val)
        if len(page) < page_size:
            break
        start += page_size

    if not all_pids:
        print("No non-demo rows found to purge.")
        return

    print(f"Found {len(all_pids)} non-demo patients to purge.")
    chunk_size = 50
    total_chunks = (len(all_pids) + chunk_size - 1) // chunk_size

    for i in range(0, len(all_pids), chunk_size):
        chunk = all_pids[i : i + chunk_size]
        sb.table("sessions").delete().in_("pid", chunk).execute()
        sb.table("module4_d1_sessions").delete().in_("pid", chunk).execute()
        sb.table("patients").delete().in_("pid", chunk).execute()
        print(f"Purged chunk {i // chunk_size + 1}/{total_chunks}")

    print("Old training dataset purged completely.")


def seed_database():
    purge_old_dataset()
    print("Starting database seeding for 15 curated demonstration patients...")
    now = datetime.now()

    for p in DEMO_PATIENTS:
        pid = int(p["pid"])
        name = str(p["name"])
        gender = str(p["gender"])
        birthday = int(p["birthday"])
        has_dm = bool(p["has_dm"])
        profile_type = str(p["profile_type"])

        target_pre_sbp = float(p["pre_sbp"])
        pre_dbp = float(p["pre_dbp"])
        base_dw = float(p["dryweight"])
        p_idwg = float(p["idwg"])
        ktv = float(p["ktv"])

        print(f"Seeding PID {pid} - {name} ({profile_type})...")

        # 1. Upsert Patient
        sb.table("patients").upsert({
            "pid": pid,
            "name": name,
            "gender": gender,
            "birthday": birthday,
            "has_dm": has_dm,
        }).execute()

        # Delete old demo sessions for this PID to ensure clean data
        sb.table("sessions").delete().eq("pid", pid).execute()
        sb.table("module4_d1_sessions").delete().eq("pid", pid).execute()

        # 2. Build 30 historical sessions for Module 4 and Module 5
        sessions_to_insert = []
        m4_sessions_to_insert = []

        for i in range(30):
            sess_date = now - timedelta(days=(30 - i) * 2)
            date_str = sess_date.isoformat()
            sess_id = f"{pid}_sess_{i+1}"

            # Determine weights based on profile_type
            if "FLUID_DRIFT" in profile_type:
                idwg = p_idwg + (i / 30.0) * 1.5
                w_start = base_dw + idwg
                w_end = base_dw + 1.2 + random.uniform(0.1, 0.4)
            elif "BODY_DRIFT" in profile_type:
                idwg = p_idwg
                w_start = base_dw + idwg - (i / 30.0) * 2.0
                w_end = base_dw - (i / 30.0) * 1.8
            else:
                idwg = p_idwg + random.uniform(-0.2, 0.2)
                w_start = base_dw + max(idwg, 0.5)
                w_end = base_dw + random.uniform(-0.1, 0.1)

            # Hemodynamic trajectory design to drive XGBoost feature importances
            if "HIGH" in profile_type:
                # Baseline 142 mmHg dropping steeply to target_pre_sbp (90-96)
                sess_sbp = 142.0 - (i / 29.0) * (142.0 - target_pre_sbp) + random.uniform(-1, 1)
                drop_mag = random.uniform(34, 46) if i >= 15 else random.uniform(12, 22)
                is_hypo = 1 if i >= 15 else 0
                post_sbp = max(sess_sbp - drop_mag + random.uniform(1, 4), 78.0) # Unrecovered!
            elif "MED" in profile_type:
                sess_sbp = 136.0 - (i / 29.0) * (136.0 - target_pre_sbp) + random.uniform(-1.5, 1.5)
                drop_mag = random.uniform(22, 30) if i >= 20 else random.uniform(8, 14)
                is_hypo = 1 if i >= 20 else 0
                post_sbp = sess_sbp - 12
            else:
                sess_sbp = target_pre_sbp + random.uniform(-2, 2)
                drop_mag = random.uniform(5, 12)
                is_hypo = 0
                post_sbp = sess_sbp - 8

            during_sbp = max(sess_sbp - drop_mag, 55.0)

            # Create sessions row
            sessions_to_insert.append({
                "session_id": sess_id,
                "pid": pid,
                "session_start": date_str,
                "duration_min": 240,
                "pre_sbp": round(sess_sbp, 1),
                "pre_dbp": round(pre_dbp + random.uniform(-2, 2), 1),
                "during_sbp": round(during_sbp, 1),
                "during_dbp": round(max(during_sbp - 40, 40), 1),
                "post_sbp": round(post_sbp, 1),
                "post_dbp": round(pre_dbp - 5, 1),
                "avg_qb": 350.0 if ktv > 1.2 else 280.0,
                "max_uf": round((w_start - base_dw) / 4.0 * 1.2, 2),
                "avg_uf": round((w_start - base_dw) / 4.0, 2),
                "avg_conductivity": 14.0,
                "avg_dia_temp": 36.5,
                "weightstart": round(w_start, 2),
                "dryweight": base_dw,
                "weight_post": round(w_end, 2),
                "weight_assumed": 0,
                "ktv_proxy": ktv,
                "hypotension_event": is_hypo,
                "cramp_event": 1 if (is_hypo and random.random() > 0.5) else 0,
                "early_termination": 1 if (is_hypo and random.random() > 0.6) else 0,
                "ktv_below_target": 1 if ktv < 1.2 else 0,
            })

            # Create module4_d1_sessions row
            m4_sessions_to_insert.append({
                "pid": pid,
                "keyindate": date_str,
                "weightstart": round(w_start, 2),
                "weightend": round(w_end, 2),
                "dryweight": base_dw,
                "temperature": 36.5,
            })

        # Batch insert
        sb.table("sessions").insert(sessions_to_insert).execute()
        sb.table("module4_d1_sessions").insert(m4_sessions_to_insert).execute()

    print("\n[OK] Successfully seeded 15 curated demo patients with full hemodynamic decline trajectories!")


if __name__ == "__main__":
    seed_database()
