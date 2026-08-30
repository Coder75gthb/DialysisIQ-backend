import os
import math
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

D1_PATH = r"C:\Users\Aayush\Desktop\DialysisIQ\backend\d1.csv"

TABLE_NAME = "module4_d1_sessions"

BATCH_SIZE = 500


# ============================================================
# CONNECT TO SUPABASE
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing from .env")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_SECRET_KEY is missing from .env")

print("Connecting to Supabase...")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print("Connected.")


# ============================================================
# LOAD d1.csv
# ============================================================

print("\nLoading d1.csv...")

d1 = pd.read_csv(D1_PATH)

print("Rows loaded:", len(d1))

required_columns = [
    "pid",
    "keyindate",
    "dialysisstart",
    "dialysisend",
    "weightstart",
    "weightend",
    "dryweight",
    "temperature"
]

missing_columns = [
    col for col in required_columns
    if col not in d1.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing columns: {missing_columns}"
    )

d1 = d1[required_columns].copy()


# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning data...")

# Convert timestamp
d1["keyindate"] = pd.to_datetime(
    d1["keyindate"],
    errors="coerce"
)

# Convert numeric columns
numeric_columns = [
    "pid",
    "weightstart",
    "weightend",
    "dryweight",
    "temperature"
]

for col in numeric_columns:

    d1[col] = pd.to_numeric(
        d1[col],
        errors="coerce"
    )


# ============================================================
# CONVERT EVERY VALUE TO JSON-SAFE VALUE
# ============================================================

def clean_value(value):

    # Missing pandas values
    if pd.isna(value):
        return None

    # Timestamp
    if isinstance(value, pd.Timestamp):

        if pd.isna(value):
            return None

        return value.isoformat()

    # Numpy numbers / floats
    if hasattr(value, "item"):

        value = value.item()

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None

        return value

    # Normal Python float
    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

    return value


# ============================================================
# CONVERT TO RECORDS
# ============================================================

print("Preparing records...")

records = []

for row in d1.to_dict(orient="records"):

    clean_row = {
        key: clean_value(value)
        for key, value in row.items()
    }

    records.append(clean_row)

print(
    "JSON-safe records prepared:",
    len(records)
)


# ============================================================
# DELETE EXISTING DATA
# ============================================================

print("\nClearing existing Module 4 table...")

supabase.table(
    TABLE_NAME
).delete().neq(
    "id",
    -1
).execute()

print("Existing data cleared.")


# ============================================================
# UPLOAD
# ============================================================

print("\nUploading data...")

total = len(records)

for start in range(
    0,
    total,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total
    )

    batch = records[start:end]

    supabase.table(
        TABLE_NAME
    ).insert(
        batch
    ).execute()

    print(
        f"Uploaded {end:,} / {total:,}"
    )


# ============================================================
# VERIFY
# ============================================================

print("\nVerifying upload...")

response = (
    supabase
    .table(TABLE_NAME)
    .select(
        "id",
        count="exact"
    )
    .limit(1)
    .execute()
)

print(
    "Rows in Supabase:",
    response.count
)


# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("UPLOAD COMPLETE")
print("========================================")