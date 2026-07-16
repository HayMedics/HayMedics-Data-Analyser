"""Every operation must render to code that does what the operation did.

This is the guard for the one bug that would quietly invalidate the entire
project. An op runs one way inside the app and renders as pandas source for
the exported script. If those two ever disagree, the app shows you clean data
while the script you hand your reviewer produces something else — and nothing
raises. It has already happened twice: `order_likert` skipped a normalisation
step, and `coerce_datetime` dropped `format='mixed'` and silently NaN'd every
date whose layout pandas didn't guess first.

Fixing each one as it surfaced was the wrong strategy, because the next op is
always one refactor away from the same mistake. So instead of trusting review,
this executes every registered op both ways and compares the frames.

CASES must cover the whole registry. That's asserted, so a new op cannot be
merged without proving its renderer honest.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hma.ops import REGISTRY, Stage  # noqa: E402


def _frame() -> pd.DataFrame:
    """One frame with something for every op to bite on."""
    return pd.DataFrame(
        {
            "age": [34, 999, 21, 44, 999, 18],
            "income": [50_000, -99, 75_000, None, 90_000, 12_000],
            "gender": ["Male", " female ", "MALE", "Female", "male", None],
            "state": ["Lagos", "lagos", " Lagos", "Kano", "KANO", "Oyo"],
            "Q1": ["Agree", "Strongly agree", "Neutral", "Disagree", "Agree", None],
            "start": ["2026-01-05", "05/01/2026", "2026-02-11", "11-02-2026", "2026-03-01", "2026-03-02"],
            "end": ["2026-01-06", "06/01/2026", "2026-02-10", "12-02-2026", "2026-03-05", "2026-03-01"],
            "name": ["A. Bello", "C. Okafor", "F. Musa", "T. Ade", "A. Bello", "K. Sani"],
            "consent": ["Yes"] * 6,
            "Q7_Part_1": ["Radio", None, "Radio", None, "Radio", None],
            "Q7_Part_2": [None, "TV", "TV", None, None, "TV"],
            "Q7_Part_3": ["Web", "Web", None, None, "Web", None],
        }
    )


def _numeric_frame() -> pd.DataFrame:
    """Ops in OUTLIERS/VALIDATE assume TYPES already ran, so hand them numbers."""
    df = _frame()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce", format="mixed")
    df["end"] = pd.to_datetime(df["end"], errors="coerce", format="mixed")
    return df


def _header_frame() -> pd.DataFrame:
    """A question-text row sitting where the data should be."""
    return pd.DataFrame(
        {
            "Q1": ["What is your age in completed years?", "34", "21"],
            "Q2": ["Which state do you currently reside in?", "Lagos", "Kano"],
        }
    )


def _header_residue_frame() -> pd.DataFrame:
    """A header line pasted back in as a record — two exports concatenated."""
    frame = _frame().head(4)
    return pd.concat(
        [frame, pd.DataFrame([{c: c for c in frame.columns}])], ignore_index=True
    )


# op name -> (frame, params). Every op in the registry needs an entry.
CASES: dict[str, tuple[pd.DataFrame, dict]] = {
    "promote_header_row": (_header_frame(), {"drop_rows": 1}),
    "drop_repeated_header_rows": (_header_residue_frame(), {}),
    "drop_duplicates": (pd.concat([_frame(), _frame().head(2)]), {}),
    "drop_empty_rows": (_frame(), {}),
    "drop_column": (_frame(), {"column": "consent"}),
    "drop_constant_columns": (_frame(), {"columns": ["consent"]}),
    "rename_column": (_frame(), {"column": "age", "new_name": "age_years"}),
    "recode_sentinel_missing": (_frame(), {"column": "age", "values": [999]}),
    "order_likert": (
        _frame(),
        {"column": "Q1", "order": ["Disagree", "Neutral", "Agree", "Strongly agree"]},
    ),
    "collapse_multiselect": (
        _frame(),
        {"stem": "Q7", "members": ["Q7_Part_1", "Q7_Part_2", "Q7_Part_3"]},
    ),
    "deidentify": (_frame(), {"column": "name", "method": "hash"}),
    "coerce_numeric": (_frame(), {"column": "age"}),
    "coerce_datetime": (_frame(), {"column": "start", "dayfirst": True}),
    "strip_whitespace": (_frame(), {"column": "gender"}),
    "map_categories": (_frame(), {"column": "state", "mapping": {"lagos": "Lagos", " Lagos": "Lagos"}}),
    "fill_missing": (_numeric_frame(), {"column": "income", "method": "median"}),
    "clip_outliers": (_numeric_frame(), {"column": "age", "lower": 0, "upper": 120}),
    "flag_out_of_range": (_numeric_frame(), {"column": "age", "lower": 18, "upper": 65}),
    "flag_date_order": (_numeric_frame(), {"start_column": "start", "end_column": "end"}),
}


def test_every_op_has_a_drift_case():
    """A new op without a case here is an op nobody has proved honest."""
    missing = set(REGISTRY) - set(CASES)
    assert not missing, (
        f"These ops have no renderer-vs-runner case: {sorted(missing)}. "
        f"Add one to CASES — an unproven renderer silently corrupts the "
        f"exported script."
    )
    assert not set(CASES) - set(REGISTRY), "CASES references ops that no longer exist"


@pytest.mark.parametrize("op_name", sorted(CASES))
def test_renderer_matches_runner(op_name: str):
    """Run the op. Run its rendered source. The frames must be identical."""
    frame, params = CASES[op_name]
    operation = REGISTRY[op_name]

    expected, _ = operation.apply(frame.copy(), **params)

    rendered = operation.code(**params)
    scope: dict = {"df": frame.copy(), "pd": pd, "hashlib": hashlib}
    exec(compile(rendered, f"<{op_name}.code>", "exec"), scope)  # noqa: S102
    actual = scope["df"]

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        obj=f"{op_name}: exported code does not reproduce the operation",
    )


@pytest.mark.parametrize("op_name", sorted(CASES))
def test_rendered_code_imports_nothing_exotic(op_name: str):
    """The script promises 'pandas and nothing else'. Keep it true.

    hashlib is permitted because it ships with Python; anything beyond the
    standard library would break that promise for whoever runs the script.
    """
    rendered = REGISTRY[op_name].code(**CASES[op_name][1])
    for banned in ("hma", "sklearn", "numpy as np", "import requests"):
        assert banned not in rendered, f"{op_name} renders a {banned!r} dependency"


def test_every_op_declares_a_stage():
    for name, operation in REGISTRY.items():
        assert isinstance(operation.stage, Stage), f"{name} has no pipeline stage"
