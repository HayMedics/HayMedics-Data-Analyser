"""Generate a dirty healthcare export — the shape real hospital data arrives in.

Modelled on an actual messy export: mixed binary encodings, sex written eight
ways, currency as text, a header row pasted back in, and int64-min where nulls
were forced into an integer column.

Fully synthetic. No real patients, safe to commit.

    uv run python scripts/make_dirty_healthcare.py

Writes data/dirty_healthcare.csv
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

rng = random.Random(7)
np.random.seed(7)

N = 4_000
INT64_MIN = -9223372036854775808  # what a null becomes when forced to int

# The same answer, written every way a data-entry clerk might write it.
SEX = ["Male", "Female", "f", "m", "Man", "Woman", "1", "2", "U"]
BINARY = ["Yes", "No", "Y", "N", "TRUE", "FALSE", "1", "0", "?"]
WARDS = ["Maternity", "maternity ward", "ER", "A & E", "Med", "Medicine",
         "Internal Medicine", "I.C.U", "Intensive Care Unit", "paeds",
         "Paediatric", "oncology", "Surgery", "surgical"]
DIAGNOSES = ["Cardiac Failure", "hypertension", "HTN", "stroke", "CVA",
             "Pneumonia", "Pnemonia", "T2DM", "Type II Diabetes",
             "Type 2 Diabetes", "Sepsis", "Anemia", "MALARIA", "Gastro",
             "Gastroenteritis", "asthma", "Asthama", "Septicaemia"]
ARMS = ["Control", "New", "standard care", "New Protocol", "Standard",
        "A", "B", "Std", "intervention"]
OUTCOMES = ["recovered", "Died", "Discharged - Recovered", "Transferred",
            "referred"]
REGIONS = ["Western", "Volta", "Eastern", "Northern", "Ashanti", "Central",
           "Central Region", "Ashanti Region", "Greater Accra"]


def _bill() -> str:
    amount = rng.uniform(300, 4000)
    return rng.choice([
        f"{amount:.2f}", f"GHS {amount:.0f}", f"₵{amount:,.2f}",
        f"{amount:,.0f}", f"{amount:.2f}",
    ])


def _date(year: int) -> str:
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice([
        f"{year}-{month:02d}-{day:02d}",
        f"{day:02d}/{month:02d}/{year}",
        f"{day:02d}-{month:02d}-{year}",
        f"{year}/{month:02d}/{day:02d}",
    ])


def build() -> pd.DataFrame:
    rows = []
    for i in range(N):
        sex = rng.choice(SEX)
        # A real signal to find: the New Protocol arm genuinely does better.
        arm = rng.choice(ARMS)
        improved = arm in {"New", "New Protocol", "intervention"}
        died = rng.random() < (0.04 if improved else 0.11)

        rows.append({
            "EncounterID": f"ENC-{100000 + i}",
            "Patient Full Name": f"{rng.choice('abcdef0123456789')}{i:08x}",
            "Gender": sex,
            "Age": rng.choice([INT64_MIN] * 2 + [999] + list(range(1, 95))),
            "Admit Date": _date(rng.choice([2024, 2025])),
            "Discharge Date": _date(rng.choice([2024, 2025])),
            "Ward/Unit": rng.choice(WARDS),
            "Diagnosis": rng.choice(DIAGNOSES),
            "BP (mmHg)": rng.choice([
                f"{rng.randint(90, 160)}/{rng.randint(60, 100)}",
                f"{rng.randint(90, 160)}-{rng.randint(60, 100)}",
                f"{rng.randint(90, 160)} / {rng.randint(60, 100)}",
            ]),
            "Weight": rng.choice([None] * 2 + [round(rng.uniform(40, 110), 1)]),
            "Height": rng.choice([None] * 2 + [rng.randint(150, 190)]),
            "Temp": rng.choice([INT64_MIN] + [36, 37, 38] * 12),
            "Smoker?": rng.choice(BINARY),
            "Treatment Arm": arm,
            "Readmitted 30d": rng.choice(BINARY),
            "Satisfaction (1-10)": rng.choice([None] + list(range(1, 11))),
            "Outcome": "Died" if died else rng.choice(
                [o for o in OUTCOMES if o != "Died"]
            ),
            "Region": rng.choice(REGIONS + [None] * 2),
            "Bill (GHS)": rng.choice([_bill()] * 9 + [None]),
        })

    df = pd.DataFrame(rows)

    # A header line pasted back in — two exports concatenated.
    df = pd.concat(
        [df, pd.DataFrame([{c: c for c in df.columns}])], ignore_index=True
    )
    # Genuine duplicate submissions.
    df = pd.concat([df, df.sample(96, random_state=3)], ignore_index=True)
    return df.sample(frac=1, random_state=5).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "dirty_healthcare.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = build()
    frame.to_csv(out, index=False)
    print(f"Wrote {out} — {len(frame):,} rows x {len(frame.columns)} columns")
    print("\nPathologies baked in:")
    for item in [
        "Gender written 9 ways (Male/m/Man/1/f/Female/Woman/2/U)",
        "Smoker? and Readmitted 30d as Yes/Y/1/TRUE vs No/N/0/FALSE, plus '?'",
        "int64-min (-9223372036854775808) where nulls were forced to integer",
        "999 in Age",
        "one header row pasted back in as a record",
        "96 duplicate rows",
        "currency as text: 'GHS 716', '₵3,072.98', '1,314'",
        "four date formats across two columns",
        "PII: Patient Full Name, EncounterID",
        "a real signal: the New Protocol arm has lower mortality",
    ]:
        print(f"  - {item}")
