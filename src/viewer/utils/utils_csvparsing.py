from dataclasses import dataclass
from typing import Optional, Dict, Callable, List
import pandas as pd

@dataclass
class CsvFieldGate:
    ok: bool
    missing: list[str]
    present: list[str]
    extra: list[str]
    file_ok: bool
    note: str

@dataclass
class CSVIssue:
    row: int
    column: int
    value: object
    reason: str
    mrid: Optional[object] = None

COLUMN_VALIDATORS: Dict[str, Callable[[pd.DataFrame, str, str], List[CSVIssue]]] = {}

def register_csv_column(name: str):
    def deco(fn):
        COLUMN_VALIDATORS[name] = fn
    return deco

def issues_from_mask(df: pd.DataFrame, column: str, badmask: pd.Series, reason: str, mrid_col: str) -> List[Issue]:
    idx = df.index[badmask.fillna(False)]
    mrids = df.loc[idx, mrid_col] if mrid_col in df.columns else [None]*len(idx)
    values = df.loc[idx, column]
    return [CSVIssue(row=int(i), column=column, value=values.loc[i], reason=reason, mrid=(mrids.loc[i] if mrid_col in df.columns else None))
            for i in idx]

### Helpers

import re
import numpy as np

def v_int(df, col, *, ge=None, le=None):
    # not-integer if it fails regex for whole numbers (allows NaN as OK unless you add nonempty)
    s = df[col]
    bad = ~s.dropna().astype(str).str.fullmatch(r"-?\d+")
    issues = issues_from_mask(df, col, bad.reindex(df.index, fill_value=False), "not an integer", mrid_col="MRID")
    if ge is not None:
        issues += issues_from_mask(df, col, s < ge, f"< {ge}", mrid_col="MRID")
    if le is not None:
        issues += issues_from_mask(df, col, s > le, f"> {le}", mrid_col="MRID")
    return issues

def v_float(df, col, *, ge=None, le=None):
    s = df[col]
    try:
        x = pd.to_numeric(s, errors="coerce")
    except Exception:
        x = pd.Series(np.nan, index=s.index)
    issues = issues_from_mask(df, col, x.isna() & s.notna(), "not a float", mrid_col="MRID")
    if ge is not None:
        issues += issues_from_mask(df, col, x < ge, f"< {ge}", mrid_col="MRID")
    if le is not None:
        issues += issues_from_mask(df, col, x > le, f"> {le}", mrid_col="MRID")
    return issues

def v_enum(df, col, *, values: list[str]):
    s = df[col].astype(str)
    bad = ~s.isin(values)
    return issues_from_mask(df, col, bad, f"not in {values}", mrid_col="MRID")

def v_regex(df, col, *, pattern: str):
    pat = re.compile(pattern)
    s = df[col].astype(str)
    bad = ~s.str.match(pat)
    return issues_from_mask(df, col, bad, f"regex mismatch: {pattern}", mrid_col="MRID")

def v_nonempty(df, col):
    s = df[col]
    bad = s.isna() | (s.astype(str).str.len() == 0)
    return issues_from_mask(df, col, bad, "empty", mrid_col="MRID")

@register_csv_column("MRID")
def _validate_mrid(df: pd.DataFrame, col: str, mrid_col: str) -> List[Issue]:
    out = []
    out += v_nonempty(df, col)
    # duplicates
    dup_mask = df[col].duplicated(keep=False)
    out += issues_from_mask(df, col, dup_mask, "duplicate MRID", mrid_col)
    return out

@register_csv_column("Age")
def _validate_age(df: pd.DataFrame, col: str, mrid_col: str) -> List[CSVIssue]:
    out = []
    out += v_int(df, col, ge=0, le=120)   # tweak bounds
    return out

@register_csv_column("Sex")
def _validate_sex(df: pd.DataFrame, col: str, mrid_col: str) -> List[CSVIssue]:
    return v_enum(df, col, values=["M","F"])

from dataclasses import asdict

@dataclass
class CsvValidationReport:
    file_ok: bool
    columns_ok: bool
    missing_cols: List[str]
    present_cols: List[str]
    extra_cols: List[str]
    issues: List[CSVIssue]           # row-level details
    ok: bool                      # columns_ok AND no issues
    rows: int

def validate_csv(csv_path: str, required_cols: list[str], mrid_col: str = "MRID") -> CsvValidationReport:
    import os
    if not (csv_path and os.path.isfile(csv_path)):
        return CsvValidationReport(False, False, required_cols, [], [], [], False, 0)

    # Load fully to allow row-level checks
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    missing = [c for c in required_cols if c not in cols]
    present = [c for c in required_cols if c in cols]
    extra   = [c for c in cols if c not in required_cols]

    columns_ok = (len(missing) == 0)
    issues: List[Issue] = []

    if columns_ok:
        # Run registered validators for *present* columns we have rules for
        for col in present:
            if col in COLUMN_VALIDATORS:
                issues.extend(COLUMN_VALIDATORS[col](df, col, mrid_col))
        # You can also validate optional columns if you want:
        for col in df.columns:
            if (col not in present) and (col in COLUMN_VALIDATORS):
                issues.extend(COLUMN_VALIDATORS[col](df, col, mrid_col))

    ok = columns_ok and (len(issues) == 0)
    return CsvValidationReport(True, columns_ok, missing, present, extra, issues, ok, rows=len(df))


