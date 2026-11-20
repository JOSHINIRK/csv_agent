"""CSV import with schema detection.

``load_csv`` takes a path or raw bytes and returns the parsed DataFrame plus a
``DatasetSchema`` that says, for every column, what kind of data it holds
(integer, float, date, category, free text, identifier, yes/no) and what role it
plays in analysis (measure, dimension, time, id). The agent reads that schema in
its instructions, so it knows the columns before the user asks anything.

Detection steps:
1. encoding: utf-8 (with or without BOM), then cp1252, then latin-1;
2. delimiter: csv.Sniffer over the first 64 KB, limited to , ; tab and |;
3. tidy-up: strip header whitespace, rename blank/duplicate headers, drop a
   leaked pandas index column ("Unnamed: 0" holding 0..n-1);
4. per column: numbers written as text ("$1,234.50", "45%") become numbers,
   date-looking text becomes datetimes, yes/no text becomes booleans;
5. classification of each column into a semantic type and a role.
"""

from __future__ import annotations

import csv
import io
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .models import ColumnSchema, DatasetSchema, ValueCount

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
DELIMITERS = ",;\t|"

_NUMERIC_TEXT = re.compile(r"^\s*[-+(]?\s*[$€£¥₹]?\s*[-+]?\d[\d,.\s]*\s*%?\s*\)?\s*$")
_DATE_TEXT = re.compile(
    r"^\s*("
    r"\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?([ T]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?"  # 2024-03-01[ 10:00]
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?"  # 03/01/2024
    r"|\d{1,2}[- ][A-Za-z]{3,9}[- ,]+\d{2,4}"  # 1 Mar 2024, 01-Mar-24
    r"|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}"  # March 1, 2024
    r"|[A-Za-z]{3,9}[- ]\d{4}"  # Mar 2024
    r")\s*$"
)
_BOOL_WORDS = {
    "true": True, "false": False, "yes": True, "no": False, "y": True, "n": False, "t": True, "f": False,
}  # fmt: skip
_ID_NAME = re.compile(r"(^|[^a-z])(id|uuid|guid|key|code|number|no|num)$|_id$|^id_", re.IGNORECASE)
_YEAR_NAME = re.compile(r"\byear\b|^yr$|_year$|^year_", re.IGNORECASE)


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    schema: DatasetSchema


class CSVImportError(ValueError):
    """The bytes could not be read as a table."""


# --------------------------------------------------------------------------- reading
def _decode(raw: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise CSVImportError("Could not decode the file as text.")  # latin-1 never fails; kept for safety


def _sniff_delimiter(text: str) -> str:
    head = text[:65536]
    try:
        return csv.Sniffer().sniff(head, delimiters=DELIMITERS).delimiter
    except csv.Error:
        first = head.splitlines()[0] if head else ""
        counts = {d: first.count(d) for d in DELIMITERS}
        best = max(counts, key=counts.get)
        return best if counts[best] else ","


def read_table(raw: bytes, settings: Settings) -> tuple[pd.DataFrame, str, str]:
    if not raw.strip():
        raise CSVImportError("The file is empty.")
    text, encoding = _decode(raw)
    delimiter = _sniff_delimiter(text)
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            na_values=settings.na_values,
            keep_default_na=False,  # use only our list: pandas would also blank out "None"
            skipinitialspace=True,
            low_memory=False,
            dtype=str,  # every column starts as text; types are inferred below
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CSVImportError(f"Could not parse the CSV: {exc}") from exc
    if df.shape[1] == 0:
        raise CSVImportError("No columns found.")
    # pandas renames duplicate headers to "a.1"; restore the file's own header so we can report and rename them.
    first_line = text.lstrip("﻿").splitlines()[0] if text.strip() else ""
    header = next(csv.reader([first_line], delimiter=delimiter, skipinitialspace=True), [])
    if len(header) == df.shape[1]:
        df.columns = [h.strip() for h in header]
    return df, delimiter, encoding


# --------------------------------------------------------------------------- tidy-up
def _clean_headers(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    names, seen = [], {}
    for i, col in enumerate(df.columns):
        name = re.sub(r"\s+", " ", str(col)).strip()
        if not name or name.startswith("Unnamed:"):
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            new = f"{name}_{seen[name]}"
            notes.append(f"Duplicate header {name!r} renamed to {new!r}.")
            name = new
        else:
            seen[name] = 1
        names.append(name)
    df.columns = names
    return df


def _drop_index_column(df: pd.DataFrame, original: list[str], notes: list[str]) -> pd.DataFrame:
    """Drop a leading 'Unnamed: 0' column that is just a saved pandas index (0..n-1 or 1..n)."""
    if not original or str(original[0]).strip() not in ("", "Unnamed: 0") or df.shape[1] < 2:
        return df
    col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    n = len(df)
    if col.notna().all() and (col.tolist() == list(range(n)) or col.tolist() == list(range(1, n + 1))):
        notes.append("Dropped a leading unnamed index column.")
        return df.iloc[:, 1:]
    return df


# --------------------------------------------------------------------------- type inference
def _sample(s: pd.Series, n: int = 2000) -> pd.Series:
    s = s.dropna()
    return s.sample(n, random_state=0) if len(s) > n else s


def _to_number(s: pd.Series) -> pd.Series:
    """'$1,234.50' -> 1234.5, '(20)' -> -20, '45%' -> 45. Unparseable -> NaN."""
    t = s.astype(str).str.strip()
    negative = t.str.startswith("(") & t.str.endswith(")")
    if _uses_decimal_comma(t):  # "1.234,56" / "0,505" (European exports)
        t = t.str.replace(r"[.\s]", "", regex=True).str.replace(",", ".", regex=False)
    t = t.str.replace(r"[$€£¥₹%,\s()+]", "", regex=True)
    out = pd.to_numeric(t, errors="coerce")
    return out.where(~negative, -out)


def _uses_decimal_comma(text: pd.Series) -> bool:
    """True when commas are decimal separators rather than thousands separators."""
    sample = _sample(text)
    if sample.empty or not sample.str.contains(",", regex=False).any():
        return False
    if sample.str.contains(r"\d\.\d{3},\d", regex=True).any():  # 1.234,56
        return True
    if sample.str.contains(r"\d,\d{3}\.\d", regex=True).any() or sample.str.contains(".", regex=False).any():
        return False  # 1,234.56 or any dot decimals: commas are thousands separators
    # No dots at all: a comma followed by anything but exactly 3 digits, or "0,xxx", is a decimal comma.
    return bool(
        sample.str.contains(r",\d{1,2}(?:\D|$)|,\d{4,}", regex=True).any()
        or sample.str.contains(r"^[-+(]?\s*[$€£¥₹]?\s*0,\d", regex=True).any()
    )


def _try_numeric(s: pd.Series) -> pd.Series | None:
    sample = _sample(s)
    if sample.empty or sample.astype(str).str.match(_NUMERIC_TEXT).mean() < 0.97:
        return None
    # Leading zeros ("00123") are codes, not numbers.
    if sample.astype(str).str.match(r"^0\d+$").mean() > 0.2:
        return None
    out = _to_number(s)
    return out if out.notna().sum() >= 0.97 * s.notna().sum() else None


def _try_datetime(s: pd.Series) -> pd.Series | None:
    sample = _sample(s)
    if sample.empty or sample.astype(str).str.match(_DATE_TEXT).mean() < 0.95:
        return None
    # Decide day-first vs month-first from values like 25/03/2024.
    dayfirst = False
    parts = sample.astype(str).str.extract(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}")
    if parts.notna().all(axis=1).any():
        first = pd.to_numeric(parts[0], errors="coerce")
        second = pd.to_numeric(parts[1], errors="coerce")
        dayfirst = bool((first > 12).any() and not (second > 12).any())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(None)
    return out if out.notna().sum() >= 0.95 * s.notna().sum() else None


def _try_bool(s: pd.Series) -> pd.Series | None:
    values = set(s.dropna().astype(str).str.strip().str.lower().unique())
    if not values or not values <= set(_BOOL_WORDS) or len(values) > 2:
        return None
    # A lone "y"/"n" column with other letters is unlikely; require a matched pair or true/false words.
    mapped = s.astype(str).str.strip().str.lower().map(_BOOL_WORDS)
    return mapped.where(s.notna()).astype("boolean")


def infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to numbers, dates and booleans where the values allow it."""
    out = {}
    for col in df.columns:
        s = df[col]
        if s.dtype == object:
            s = s.where(s.isna(), s.astype(str).str.strip())
            converted = _try_bool(s)
            if converted is None:
                converted = _try_numeric(s)
            if converted is None:
                converted = _try_datetime(s)
            if converted is not None:
                s = converted
            if pd.api.types.is_float_dtype(s) and s.notna().any() and (s.dropna() % 1 == 0).all():
                if s.dropna().abs().max() < 2**53:
                    s = s.astype("Int64")
        out[col] = s
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- classification
def _fmt(v: object) -> str:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d") if v == v.normalize() else v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (float, np.floating)):
        return f"{v:,.4g}" if abs(v) < 1e6 else f"{v:,.0f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    return str(v)


def classify_column(name: str, s: pd.Series, rows: int, settings: Settings) -> ColumnSchema:
    non_null = int(s.notna().sum())
    missing = rows - non_null
    unique = int(s.nunique(dropna=True))
    uniq_ratio = unique / non_null if non_null else 0.0
    notes: list[str] = []
    kw: dict = {}

    if pd.api.types.is_bool_dtype(s):
        semantic, role = "boolean", "dimension"
    elif pd.api.types.is_datetime64_any_dtype(s):
        semantic, role = "datetime", "time"
        if non_null:
            kw.update(min=_fmt(s.min()), max=_fmt(s.max()))
    elif pd.api.types.is_numeric_dtype(s):
        is_int = pd.api.types.is_integer_dtype(s)
        semantic = "integer" if is_int else "float"
        role = "measure"
        if non_null:
            kw.update(min=_fmt(s.min()), max=_fmt(s.max()), mean=float(s.mean()))
        if is_int and _ID_NAME.search(name) and uniq_ratio > 0.9:
            semantic, role = "identifier", "id"
        elif is_int and non_null and _YEAR_NAME.search(name) and s.min() >= 1800 and s.max() <= 2200:
            role = "time"
            notes.append("Whole-number year: group by it directly (no time_grain needed).")
        elif is_int and unique <= 12 and non_null and s.min() >= 0 and s.max() <= 12 and _looks_ordinal(name):
            role = "dimension"
    else:
        text = s.dropna().astype(str)
        avg_len = float(text.str.len().mean()) if non_null else 0.0
        if unique <= settings.categorical_max_unique or (uniq_ratio <= 0.05 and avg_len < 40 and not _ID_NAME.search(name)):
            semantic, role = "categorical", "dimension"
        elif _ID_NAME.search(name) or (avg_len < 24 and uniq_ratio > 0.9 and _looks_like_codes(text)):
            semantic, role = "identifier", "id"
        elif avg_len >= 40:
            semantic, role = "text", "text"
        else:
            semantic, role = "categorical", "dimension"
            notes.append(f"High-cardinality category ({unique:,} values): use find_values to locate values.")

    if role in ("dimension", "text", "id") or semantic == "boolean":
        counts = s.astype("string").value_counts(dropna=True).head(10)
        kw["top_values"] = [ValueCount(value=str(k), count=int(v)) for k, v in counts.items()]
    if missing and rows and missing / rows > 0.2:
        notes.append(f"{missing / rows:.0%} of values are missing.")

    sample = s.dropna().head(5)
    return ColumnSchema(
        name=name,
        dtype=str(s.dtype),
        semantic_type=semantic,
        role=role,
        non_null=non_null,
        missing=missing,
        missing_pct=round(100 * missing / rows, 2) if rows else 0.0,
        unique=unique,
        sample_values=[_fmt(v) for v in sample],
        notes=notes,
        **kw,
    )


def _looks_like_codes(text: pd.Series) -> bool:
    """'AFG', 'E10023', 'US-2025-1001' are codes; 'United Kingdom', 'Afghanistan' are names."""
    sample = _sample(text, 500)
    code_like = sample.str.fullmatch(r"[A-Z0-9][A-Z0-9_\-./:#]*") | sample.str.contains(r"\d", regex=True)
    return bool(code_like.mean() >= 0.8)


def _looks_ordinal(name: str) -> bool:
    return bool(re.search(r"rating|score|level|grade|tier|class|rank|stars|month|quarter|day", name, re.IGNORECASE))


# --------------------------------------------------------------------------- entry point
def load_csv(source: str | Path | bytes, name: str | None = None, settings: Settings | None = None) -> LoadedDataset:
    """Read a CSV from a path or raw bytes and detect its schema."""
    settings = settings or load_settings()
    if isinstance(source, (str, Path)):
        path = Path(source)
        raw = path.read_bytes()
        name = name or path.name
    else:
        raw = source
        name = name or "uploaded.csv"

    df, delimiter, encoding = read_table(raw, settings)
    notes: list[str] = []
    original = list(df.columns)
    df = _drop_index_column(df, original, notes)
    df = _clean_headers(df, notes)
    df = df.dropna(how="all")
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty and len(df):
        df = df.drop(columns=empty)
        notes.append(f"Dropped empty column(s): {', '.join(empty)}.")
    df = infer_types(df).reset_index(drop=True)

    rows = len(df)
    columns = [classify_column(col, df[col], rows, settings) for col in df.columns]
    duplicates = int(df.duplicated().sum()) if rows else 0
    if duplicates:
        notes.append(f"{duplicates:,} fully duplicated rows.")
    if encoding != "utf-8-sig":
        notes.append(f"File was not UTF-8; read as {encoding}.")

    schema = DatasetSchema(
        name=name,
        rows=rows,
        columns=columns,
        delimiter=delimiter,
        encoding=encoding,
        duplicate_rows=duplicates,
        warnings=notes,
    )
    return LoadedDataset(df=df, schema=schema)
