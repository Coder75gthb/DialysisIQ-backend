import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

D1_PATH = r"C:\Users\Aayush\Downloads\d1.csv"

PAGE_SIZE = 1000


# ============================================================
# 1. LOAD ORIGINAL d1.csv
# ============================================================

print("=" * 60)
print("MODULE 4 - d1 TEMPERATURE MAPPING CHECK")
print("=" * 60)

print("\nLoading d1.csv...")

d1 = pd.read_csv(D1_PATH)

d1["keyindate"] = pd.to_datetime(
    d1["keyindate"],
    errors="coerce"
)

# The notebook maps sessions using patient + calendar date
d1["session_date"] = d1["keyindate"].dt.date

print("D1 rows:", len(d1))

print(
    "D1 temperature available:",
    d1["temperature"].notna().sum()
)

print(
    "D1 temperature missing:",
    d1["temperature"].isna().sum()
)


# ============================================================
# 2. CONNECT TO SUPABASE
# ============================================================

print("\nConnecting to Supabase...")

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing from .env")

if not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_SECRET_KEY is missing from .env"
    )

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print("Supabase connection ready.")


# ============================================================
# 3. FETCH ALL SUPABASE SESSIONS
# ============================================================

print("\nFetching ALL Supabase sessions...")

all_rows = []

start = 0

while True:

    end = start + PAGE_SIZE - 1

    response = (
        supabase
        .table("sessions")
        .select(
            "session_id,pid,session_start"
        )
        .range(start, end)
        .execute()
    )

    rows = response.data or []

    if not rows:
        print(f"No rows returned for range {start}-{end}.")
        break

    all_rows.extend(rows)

    print(
        f"Fetched rows {start} -> "
        f"{start + len(rows) - 1}"
    )

    # If fewer than 1000 rows came back,
    # we reached the end.
    if len(rows) < PAGE_SIZE:
        break

    start += PAGE_SIZE


sessions = pd.DataFrame(all_rows)

print(
    "\nSupabase sessions fetched:",
    len(sessions)
)


# ============================================================
# 4. CHECK THAT WE ACTUALLY FETCHED DATA
# ============================================================

if sessions.empty:
    raise RuntimeError(
        "No sessions were fetched from Supabase."
    )


# ============================================================
# 5. PREPARE SUPABASE SESSION DATES
# ============================================================

sessions["session_start"] = pd.to_datetime(
    sessions["session_start"],
    errors="coerce"
)

sessions["session_date"] = (
    sessions["session_start"].dt.date
)


# ============================================================
# 6. CHECK DUPLICATES IN d1
# ============================================================

print("\nChecking duplicate d1 patient/date combinations...")

d1_map = d1[
    [
        "pid",
        "session_date",
        "temperature"
    ]
].copy()

duplicates = (
    d1_map
    .groupby(
        ["pid", "session_date"],
        dropna=False
    )
    .size()
    .reset_index(name="count")
)

duplicates = duplicates[
    duplicates["count"] > 1
]

print(
    "Duplicate d1 patient/date combinations:",
    len(duplicates)
)

if len(duplicates) > 0:

    print("\nDuplicate examples:")

    print(
        duplicates.head(20).to_string(
            index=False
        )
    )


# ============================================================
# 7. CHECK HOW MANY DUPLICATES HAVE DIFFERENT TEMPERATURES
# ============================================================

print(
    "\nChecking whether duplicate dates "
    "have conflicting temperatures..."
)

duplicate_keys = duplicates[
    ["pid", "session_date"]
]

if len(duplicate_keys) > 0:

    duplicate_rows = d1_map.merge(
        duplicate_keys,
        on=["pid", "session_date"],
        how="inner"
    )

    temperature_variation = (
        duplicate_rows
        .groupby(
            ["pid", "session_date"]
        )["temperature"]
        .nunique(dropna=True)
        .reset_index(
            name="unique_temperatures"
        )
    )

    conflicting = temperature_variation[
        temperature_variation["unique_temperatures"] > 1
    ]

    print(
        "Duplicate dates with different "
        "temperatures:",
        len(conflicting)
    )

    if len(conflicting) > 0:

        print("\nConflicting examples:")

        print(
            conflicting.head(20).to_string(
                index=False
            )
        )

else:

    print("No duplicate keys found.")


# ============================================================
# 8. CREATE UNIQUE d1 MAPPING
# ============================================================
#
# IMPORTANT:
# We are NOT modifying Supabase.
#
# This section only creates a mapping for the diagnostic.
#
# For duplicate pid/date combinations, we do NOT
# blindly choose a temperature.
#
# Such rows will remain unresolved for now.
# ============================================================

d1_counts = (
    d1_map
    .groupby(
        ["pid", "session_date"]
    )
    .size()
    .reset_index(
        name="d1_match_count"
    )
)

d1_unique = d1_map.merge(
    d1_counts,
    on=["pid", "session_date"],
    how="left"
)

# Keep only unique patient/date combinations
d1_unique = d1_unique[
    d1_unique["d1_match_count"] == 1
].copy()

d1_unique = d1_unique.drop(
    columns=["d1_match_count"]
)


# ============================================================
# 9. MATCH SUPABASE SESSIONS TO d1
# ============================================================

print(
    "\nMatching Supabase sessions "
    "to original d1 temperature..."
)

matched = sessions.merge(
    d1_unique,
    on=["pid", "session_date"],
    how="left"
)


# ============================================================
# 10. MAPPING RESULTS
# ============================================================

total = len(matched)

matched_temperature = (
    matched["temperature"].notna().sum()
)

missing_temperature = (
    matched["temperature"].isna().sum()
)

match_percentage = (
    matched_temperature / total * 100
    if total > 0
    else 0
)

print("\n")
print("=" * 60)
print("MAPPING RESULTS")
print("=" * 60)

print(
    "Total Supabase sessions:",
    total
)

print(
    "Matched temperature:",
    matched_temperature
)

print(
    "Missing temperature:",
    missing_temperature
)

print(
    "Match percentage:",
    round(match_percentage, 2),
    "%"
)


# ============================================================
# 11. SHOW UNMATCHED SESSIONS
# ============================================================

unmatched = matched[
    matched["temperature"].isna()
].copy()

print(
    "\nUnmatched sessions:",
    len(unmatched)
)

if len(unmatched) > 0:

    print("\nFirst 30 unmatched sessions:")

    print(
        unmatched[
            [
                "session_id",
                "pid",
                "session_start",
                "session_date"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# 12. SHOW SUCCESSFUL MATCHES
# ============================================================

successful = matched[
    matched["temperature"].notna()
].copy()

print(
    "\nSuccessful matches:",
    len(successful)
)

if len(successful) > 0:

    print(
        "\nFirst 20 successful matches:"
    )

    print(
        successful[
            [
                "session_id",
                "pid",
                "session_start",
                "temperature"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 13. TEMPERATURE SUMMARY
# ============================================================

if len(successful) > 0:

    print("\nTemperature summary:")

    print(
        successful["temperature"].describe()
    )


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 60)
print("CHECK COMPLETE")
print("=" * 60)

print(
    "\nIMPORTANT:"
    "\nNo Supabase data was modified."
    "\nNo model was modified."
    "\nNo temperature values were written."
)