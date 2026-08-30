import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client


# ============================================================
# 1. LOAD SUPABASE
# ============================================================

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SECRET_KEY")

if not url or not key:
    raise ValueError("SUPABASE_URL or SUPABASE_SECRET_KEY is missing from .env")

supabase = create_client(url, key)


# ============================================================
# 2. LOCATE VIP.CSV
# ============================================================

vip_path = Path.home() / "Downloads" / "vip.csv"

if not vip_path.exists():
    raise FileNotFoundError(
        f"Could not find VIP file at:\n{vip_path}"
    )

print("Loading:", vip_path)


# ============================================================
# 3. LOAD VIP
# ============================================================

vip = pd.read_csv(vip_path)

print(f"VIP rows: {len(vip):,}")


# ============================================================
# 4. RECONSTRUCT SESSIONS
#    Same logic as the notebook:
#    gap > 120 minutes = new session
# ============================================================

vip["datatime"] = pd.to_datetime(vip["datatime"])

vip = (
    vip
    .sort_values(["pid", "datatime"])
    .reset_index(drop=True)
)

vip["time_diff"] = (
    vip.groupby("pid")["datatime"]
    .diff()
    .dt.total_seconds()
    / 60
)

vip["new_session"] = (
    vip["time_diff"].isna()
    | (vip["time_diff"] > 120)
)

vip["session_num"] = (
    vip.groupby("pid")["new_session"]
    .cumsum()
)


print(
    f"Reconstructed sessions: "
    f"{vip[['pid', 'session_num']].drop_duplicates().shape[0]:,}"
)


# ============================================================
# 5. CALCULATE QB_MODE
# ============================================================
#
# IMPORTANT:
# We do NOT create session_id for all 4.3M rows.
# That caused the memory error.
#
# We first keep only active blood-flow readings.
# Then calculate the mode per (pid, session_num).
# Only afterwards do we create session_id.
# ============================================================

active = vip.loc[
    vip["blood_flow"] > 0,
    ["pid", "session_num", "blood_flow"]
].copy()


print(
    f"Active Qb readings: {len(active):,}"
)


session_qb_mode = (
    active
    .groupby(["pid", "session_num"])["blood_flow"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)


# Now this dataframe contains only ~196K rows,
# so creating session_id is safe.

session_qb_mode["session_id"] = (
    session_qb_mode["pid"].astype(str)
    + "_"
    + session_qb_mode["session_num"].astype(str)
)


session_qb_mode = session_qb_mode[
    ["session_id", "blood_flow"]
].rename(
    columns={
        "blood_flow": "qb_mode"
    }
)


print(
    f"Calculated qb_mode for "
    f"{len(session_qb_mode):,} sessions"
)

print("\nSample:")
print(session_qb_mode.head(10))


# ============================================================
# 6. PREPARE RECORDS
# ============================================================

records = session_qb_mode.to_dict("records")

BATCH_SIZE = 1000

total_updated = 0


# ============================================================
# 7. UPDATE SUPABASE IN BATCHES
# ============================================================

print("\nUpdating Supabase...")


for start in range(0, len(records), BATCH_SIZE):

    batch = records[start:start + BATCH_SIZE]

    # Make sure values are normal Python floats
    # instead of NumPy values.
    for row in batch:
        row["qb_mode"] = float(row["qb_mode"])

    result = supabase.rpc(
        "update_qb_modes",
        {
            "data": batch
        }
    ).execute()

    count = result.data or 0

    total_updated += count

    processed = min(
        start + BATCH_SIZE,
        len(records)
    )

    print(
        f"Updated {processed:,} / "
        f"{len(records):,}"
    )


# ============================================================
# 8. FINAL RESULT
# ============================================================

print("\n========================================")
print("QB_MODE POPULATION COMPLETE")
print("========================================")
print(f"Sessions calculated: {len(records):,}")
print(f"Rows updated:       {total_updated:,}")
print("========================================")