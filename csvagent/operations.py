"""Runs a ``QueryPlan`` against a DataFrame: filter -> group -> aggregate -> sort -> limit.

Everything here is deterministic pandas; the LLM only chooses the plan. Problems
(unknown column, wrong value type) raise ``PlanError`` with a message written for
the model, which the agent turns into a retry so the model can fix its plan.
"""

from __future__ import annotations

import difflib
import re
import warnings

import pandas as pd

from .models import Aggregation, Filter, QueryPlan

MISSING_LABEL = "(missing)"
_GRAIN_FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


class PlanError(ValueError):
    """The plan cannot run on this dataset. The message tells the model how to fix it."""


# --------------------------------------------------------------------------- column names
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_column(name: str, columns: list[str] | pd.Index) -> str:
    """Map a column name the model wrote to the real one (exact, case-insensitive, punctuation-insensitive)."""
    columns = [str(c) for c in columns]
    if name in columns:
        return name
    lowered = {c.lower(): c for c in columns}
    if str(name).lower() in lowered:
        return lowered[str(name).lower()]
    normed = {_norm(c): c for c in columns}
    if _norm(name) in normed:
        return normed[_norm(name)]
    close = difflib.get_close_matches(str(name), columns, n=3, cutoff=0.5)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    raise PlanError(f"Unknown column {name!r}.{hint} Available columns: {', '.join(columns)}")


# --------------------------------------------------------------------------- filters
def _coerce_value(series: pd.Series, value: object, column: str) -> object:
    if value is None:
        return None
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.Timestamp(str(value))
        if pd.api.types.is_bool_dtype(series):
            if isinstance(value, str):
                return value.strip().lower() in {"true", "yes", "y", "1", "t"}
            return bool(value)
        if pd.api.types.is_numeric_dtype(series):
            return float(str(value).replace(",", "").replace("$", "").replace("%", ""))
    except (ValueError, TypeError) as exc:
        raise PlanError(f"Filter value {value!r} does not fit column {column!r} ({series.dtype}).") from exc
    return str(value)


def _as_list(value: object) -> list:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str) and "," in value:
        return [v.strip() for v in value.split(",")]
    return [value]


def _text_equals(series: pd.Series, values: list) -> pd.Series:
    wanted = {str(v).strip().casefold() for v in values}
    return series.astype("string").str.strip().str.casefold().isin(wanted).fillna(False)


def apply_filter(df: pd.DataFrame, f: Filter) -> pd.Series:
    col = resolve_column(f.column, df.columns)
    s = df[col]
    is_text = not (
        pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s)
    )
    op = f.op

    if op == "is null":
        return s.isna()
    if op == "not null":
        return s.notna()
    if f.value is None:
        raise PlanError(f"Filter on {col!r} with op {op!r} needs a value.")

    if op in ("in", "not in"):
        values = _as_list(f.value)
        if is_text:
            mask = _text_equals(s, values)
        else:
            mask = s.isin([_coerce_value(s, v, col) for v in values]).fillna(False)
        return ~mask if op == "not in" else mask

    if op == "between":
        values = _as_list(f.value)
        if len(values) != 2:
            raise PlanError(f"'between' on {col!r} needs a [low, high] pair, got {f.value!r}.")
        low, high = (_coerce_value(s, v, col) for v in values)
        return (s >= low) & (s <= high)

    if op == "contains":
        return s.astype("string").str.contains(str(f.value), case=False, regex=False).fillna(False)

    value = _coerce_value(s, f.value, col)
    if is_text and op in ("==", "!="):
        mask = _text_equals(s, [value])
        if op == "==" and not mask.any():
            raise PlanError(
                f"No rows have {col!r} == {f.value!r}. Use find_values to look up the exact spelling, "
                f"or 'contains' for a partial match."
            )
        return ~mask if op == "!=" else mask
    try:
        result = {
            "==": s == value, "!=": s != value, ">": s > value, ">=": s >= value, "<": s < value, "<=": s <= value,
        }[op]  # fmt: skip
    except TypeError as exc:
        raise PlanError(f"Cannot compare column {col!r} ({s.dtype}) with {f.value!r} using {op!r}.") from exc
    return result.fillna(False)


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for f in filters:
        mask &= apply_filter(df, f).astype(bool)
    return df[mask]


# --------------------------------------------------------------------------- grouping
def auto_grain(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "month"
    days = (s.max() - s.min()).days
    if days <= 62:
        return "day"
    if days <= 366:
        return "week" if days <= 120 else "month"
    if days <= 5 * 366:
        return "month"
    return "year"


def bucket_dates(series: pd.Series, grain: str) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "converting to PeriodArray drops timezone"
        return series.dt.to_period(_GRAIN_FREQ[grain]).dt.start_time


def _agg_alias(a: Aggregation, column: str | None) -> str:
    if a.alias:
        return a.alias
    if a.func == "count" and column is None:
        return "Count"
    if a.func == "sum":
        return str(column)
    names = {"mean": "Average", "median": "Median", "min": "Min", "max": "Max", "count": "Count of",
             "nunique": "Distinct", "std": "Std dev of"}  # fmt: skip
    return f"{names[a.func]} {column}"


def _aggregate(frame: pd.DataFrame, keys: list[str], aggs: list[tuple[str, str | None, str]]) -> pd.DataFrame:
    """aggs: (alias, column or None, func)."""
    if keys:
        grouped = frame.groupby(keys, dropna=False, sort=False, observed=True)
        parts = []
        for alias, col, func in aggs:
            if col is None:
                part = grouped.size()
            else:
                part = grouped[col].agg(func)
            parts.append(part.rename(alias))
        return pd.concat(parts, axis=1).reset_index()
    row = {}
    for alias, col, func in aggs:
        row[alias] = len(frame) if col is None else frame[col].agg(func)
    return pd.DataFrame([row])


def execute_plan(df: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    """Run the plan and return the result table."""
    data = apply_filters(df, plan.filters)

    group_cols = [resolve_column(c, df.columns) for c in plan.group_by]
    agg_specs: list[tuple[str, str | None, str]] = []
    for a in plan.aggregations:
        col = resolve_column(a.column, df.columns) if a.column and a.column != "*" else None
        if col is None and a.func != "count":
            raise PlanError(f"Aggregation {a.func!r} needs a column.")
        if col is not None and a.func not in ("count", "nunique") and not pd.api.types.is_numeric_dtype(data[col]):
            raise PlanError(
                f"Cannot {a.func} the non-numeric column {col!r}. Use 'count' or 'nunique', or pick a numeric column."
            )
        if col is not None and pd.api.types.is_bool_dtype(data[col]) and a.func in ("sum", "mean"):
            data = data.assign(**{col: data[col].astype("Float64")})
        agg_specs.append((_agg_alias(a, col), col, a.func))
    if group_cols and not agg_specs:
        agg_specs = [("Count", None, "count")]
    _dedupe_aliases(agg_specs, group_cols)

    if not group_cols and not agg_specs:
        cols = [resolve_column(c, df.columns) for c in plan.columns] if plan.columns else list(df.columns)
        result = data[cols]
    else:
        frame = data.copy()
        time_keys = []
        for col in group_cols:
            if pd.api.types.is_datetime64_any_dtype(frame[col]):
                grain = auto_grain(frame[col]) if plan.time_grain == "auto" else plan.time_grain
                frame[col] = bucket_dates(frame[col], grain)
                time_keys.append(col)
            elif not pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(frame[col]):
                frame[col] = frame[col].astype("string").fillna(MISSING_LABEL)
        result = _aggregate(frame, group_cols, agg_specs)
        if time_keys and not plan.sort_by:
            result = result.sort_values(time_keys, kind="stable")
        elif not plan.sort_by and agg_specs:
            result = result.sort_values(agg_specs[0][0], ascending=False, kind="stable")

    if plan.sort_by:
        key = resolve_column(plan.sort_by, result.columns)
        result = result.sort_values(key, ascending=not plan.descending, kind="stable", na_position="last")
    if plan.limit:
        if plan.limit < 1:
            raise PlanError("limit must be a positive number.")
        result = result.head(plan.limit)
    return result.reset_index(drop=True)


def _dedupe_aliases(specs: list[tuple[str, str | None, str]], taken: list[str]) -> None:
    used = set(taken)
    for i, (alias, col, func) in enumerate(specs):
        new = alias
        n = 2
        while new in used:
            new = f"{alias} ({func})" if n == 2 else f"{alias} ({func} {n})"
            n += 1
        used.add(new)
        specs[i] = (new, col, func)


# --------------------------------------------------------------------------- formatting for the model
def frame_to_text(df: pd.DataFrame, max_rows: int) -> str:
    """Compact CSV-style text of a result for the model, with rounded numbers."""
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        s = shown[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            has_time = (s.dropna() != s.dropna().dt.normalize()).any()
            shown[col] = s.dt.strftime("%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d")
        elif pd.api.types.is_float_dtype(s):
            shown[col] = s.round(4 if s.abs().max() < 10 else 2)
    text = shown.to_csv(index=False).strip()
    if len(df) > max_rows:
        text += f"\n... ({len(df) - max_rows:,} more rows not shown; {len(df):,} rows in total)"
    return text
