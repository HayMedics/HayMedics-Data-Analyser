"""Generate a synthetic messy survey — every pathology, no real respondents.

Run this before your Kaggle download lands, so you always have something to
test against. It's fully synthetic: no real people, nothing to de-identify for
real, safe to commit to a public repo.

    uv run python scripts/make_messy_survey.py

Writes data/messy_survey.csv
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

rng = random.Random(42)
np.random.seed(42)

N = 600

AGREEMENT = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
FREQUENCY = ["Never", "Rarely", "Sometimes", "Often", "Always"]
STATES = ["Kwara", "Lagos", "Oyo", "Kano", "Rivers", "FCT"]


def build() -> pd.DataFrame:
    rows = []
    for i in range(N):
        asked_followup = rng.random() < 0.35  # skip logic gate

        rows.append(
            {
                "respondent_id": f"R{i:04d}",
                "respondent_name": rng.choice(["A. Bello", "C. Okafor", "F. Musa", "T. Adeyemi"]),
                "email": f"resp{i:04d}@example.org",
                # 999 = "declined to state age"
                "age": rng.choice([999] * 3 + list(range(18, 76))),
                # inconsistent casing and padding, the classic
                "gender": rng.choice(["Male", "female", " Female ", "M", "F", "male", "Prefer not to say"]),
                "state": rng.choice(STATES + ["  Lagos", "lagos", "LAGOS"]),
                "Q1": rng.choice(AGREEMENT + ["Prefer not to say"]),
                "Q2": rng.choice(FREQUENCY),
                # multi-select exploded into parts; blank = not ticked
                "Q7_Part_1": "Radio" if rng.random() < 0.4 else None,
                "Q7_Part_2": "Television" if rng.random() < 0.6 else None,
                "Q7_Part_3": "Social media" if rng.random() < 0.7 else None,
                "Q7_Part_4": "Health worker" if rng.random() < 0.3 else None,
                "Q7_Part_5": "Newspaper" if rng.random() < 0.15 else None,
                # gated behind an earlier answer — mostly blank by design
                "Q9_followup": rng.choice(AGREEMENT) if asked_followup else None,
                "monthly_income": rng.choice([-99] * 2 + [None] + list(range(20000, 400000, 5000))),
                "survey_date": rng.choice(
                    ["2026-03-01", "01/03/2026", "2026-03-15", "15-03-2026", "2026/03/20"]
                ),
                "consent": "Yes",  # constant column
                "free_text_comment": rng.choice(
                    ["", "N/A", "-", "The clinic was far", "Nothing to add", "Don't know"]
                ),
            }
        )

    df = pd.DataFrame(rows)

    # sprinkle real blanks and exact duplicate rows
    for col in ["Q1", "Q2", "state"]:
        df.loc[df.sample(frac=0.08, random_state=1).index, col] = None
    df = pd.concat([df, df.sample(12, random_state=2)], ignore_index=True)

    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "messy_survey.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = build()
    frame.to_csv(out, index=False)
    print(f"Wrote {out} — {len(frame):,} rows x {len(frame.columns)} columns")
    print("\nPathologies baked in:")
    for item in [
        "999 sentinel in `age`, -99 in `monthly_income`",
        "'Prefer not to say' in Q1 and gender",
        "casing/whitespace chaos in `gender` and `state`",
        "Q7_Part_1..5 multi-select (blank = not ticked, NOT missing)",
        "Q9_followup gated by skip logic (~65% structurally blank)",
        "five different date formats in `survey_date`",
        "`consent` constant, 12 duplicate rows",
        "PII: respondent_name, email, respondent_id",
    ]:
        print(f"  - {item}")
