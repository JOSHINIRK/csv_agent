"""Deterministic statistics behind "Summarise key trends".

The model should not eyeball thousands of rows. ``summarize_trends`` computes the
facts (totals, change over time, peaks, leaders and laggards per category, strong
correlations, data-quality issues) and returns them as short bullet lines. The
agent then writes the narrative from those numbers.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .models import DatasetSchema
from .operations import auto_grain, bucket_dates, resolve_column

_PRIORITY = re.compile(
    r"sales|revenue|amount|profit|income|price|cost|spend|value|total|population|salary|wage|units|quantity|count",
    re.IGNORECASE,
)


def _num(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:,.2f}M"
    if a >= 1e4:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:,.1f}"
    return f"{v:,.3g}"


def _pct(new: float, old: float) -> str:
    if not old:
        return "n/a"
    return f"{(new - old) / abs(old) * 100:+.1f}%"


def pick_measure(schema: DatasetSchema) -> str | None:
    measures = [c for c in schema.columns if c.role == "measure"]
    if not measures:
        return None
    ranked = sorted(measures, key=lambda c: (not _PRIORITY.search(c.name), c.semantic_type != "float"))
    return ranked[0].name


def pick_time(schema: DatasetSchema) -> str | None:
    times = [c for c in schema.columns if c.role == "time"]
    times.sort(key=lambda c: c.semantic_type != "datetime")
    return times[0].name if times else None


_DIM_PRIORITY = re.compile(
    r"region|categor|segment|country|state|product|type|department|channel|gender|sex|education|occupation|class",
    re.IGNORECASE,
)


def pick_dimensions(schema: DatasetSchema, n: int = 3) -> list[str]:
    dims = [c for c in schema.columns if c.role == "dimension" and 2 <= c.unique <= 30]
    dims.sort(key=lambda c: (c.missing_pct > 20, not _DIM_PRIORITY.search(c.name), c.unique))
    return [c.name for c in dims[:n]]


def _time_section(df: pd.DataFrame, time_col: str, measure: str | None, agg: str) -> list[str]:
    s = df[time_col]
    lines: list[str] = []
    if pd.api.types.is_datetime64_any_dtype(s):
        grain = auto_grain(s)
        keys = bucket_dates(s, grain)
        label = {"day": "%Y-%m-%d", "week": "week of %Y-%m-%d", "month": "%b %Y", "quarter": "%Y-%m", "year": "%Y"}[grain]
        fmt = (lambda t: f"Q{(t.month - 1) // 3 + 1} {t.year}") if grain == "quarter" else (lambda t: t.strftime(label))
    else:
        grain, keys, fmt = "year", s, (lambda t: str(t))
    series = (df.groupby(keys)[measure].agg(agg) if measure else df.groupby(keys).size()).sort_index()
    what = f"{agg} of {measure}" if measure else "row count"
    if len(series) < 2:
        return [f"Only one {grain} of data, so there is no trend over time."]

    lines.append(
        f"Over time ({what} per {grain}, {len(series)} periods from {fmt(series.index[0])} to {fmt(series.index[-1])}):"
    )
    first, last = series.iloc[0], series.iloc[-1]
    lines.append(f"  - first period {_num(first)}, last period {_num(last)} ({_pct(last, first)} overall).")
    x = np.arange(len(series))
    slope = np.polyfit(x, series.values.astype(float), 1)[0]
    mean = series.mean()
    span_change = slope * (len(series) - 1) / abs(mean) if mean else 0.0  # fitted change across the whole span
    direction = "upward" if span_change > 0.1 else "downward" if span_change < -0.1 else "roughly flat"
    lines.append(
        f"  - overall direction: {direction} (fitted trend {_num(slope)} per {grain}, "
        f"{span_change * 100:+.0f}% of the average across the whole period)."
    )
    lines.append(f"  - peak: {fmt(series.idxmax())} at {_num(series.max())}; low: {fmt(series.idxmin())} at {_num(series.min())}.")
    changes = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(changes) >= 2:
        up, down = changes.idxmax(), changes.idxmin()
        lines.append(
            f"  - biggest jump: {fmt(up)} ({changes[up] * 100:+.1f}% vs previous); "
            f"biggest drop: {fmt(down)} ({changes[down] * 100:+.1f}%)."
        )
    if pd.api.types.is_datetime64_any_dtype(s) and s.dt.year.nunique() >= 2 and grain != "year":
        yearly = df.groupby(s.dt.year)[measure].agg(agg) if measure else df.groupby(s.dt.year).size()
        parts = [f"{y} {_num(v)}" + (f" ({_pct(v, yearly.iloc[i - 1])})" if i else "") for i, (y, v) in enumerate(yearly.items())]
        lines.append(f"  - by year: {', '.join(parts)}.")
    if pd.api.types.is_datetime64_any_dtype(s) and (s.max() - s.min()).days >= 700 and grain in ("day", "week", "month"):
        by_month = (df.groupby(s.dt.month)[measure].agg(agg) if measure else df.groupby(s.dt.month).size())
        by_month = by_month / by_month.mean()
        top = by_month.sort_values(ascending=False).index[:2]
        low = by_month.idxmin()
        names = lambda m: pd.Timestamp(2000, int(m), 1).strftime("%B")  # noqa: E731
        lines.append(
            f"  - seasonality: {', '.join(names(m) for m in top)} are typically strongest "
            f"({(by_month[top[0]] - 1) * 100:+.0f}% vs an average month); {names(low)} is weakest."
        )
    if pd.api.types.is_datetime64_any_dtype(s) and grain in ("month", "quarter", "year"):
        last_day = s.max()
        period_end = (pd.Timestamp(series.index[-1]) + pd.tseries.frequencies.to_offset(
            {"month": "MS", "quarter": "QS", "year": "YS"}[grain])) - pd.Timedelta(1, unit="D")
        if last_day < period_end - pd.Timedelta(3, unit="D"):
            lines.append(f"  - note: the last {grain} looks incomplete (data ends {last_day:%Y-%m-%d}).")
    return lines


def _dimension_section(df: pd.DataFrame, dim: str, measure: str | None, agg: str) -> list[str]:
    keys = df[dim].astype("string").fillna("(missing)")
    series = df.groupby(keys)[measure].agg(agg) if measure else df.groupby(keys).size()
    series = series.sort_values(ascending=False)
    if len(series) < 2:
        return []
    what = f"{agg} of {measure}" if measure else "rows"
    total = series.sum()
    top, bottom = series.index[0], series.index[-1]
    share = f" ({series.iloc[0] / total:.0%} of total)" if agg == "sum" or not measure else ""
    line = (
        f"By {dim} ({what}): top is {top} with {_num(series.iloc[0])}{share}; "
        f"lowest is {bottom} with {_num(series.iloc[-1])}"
    )
    if len(series) > 2:
        line += f"; ranking: {', '.join(f'{k} {_num(v)}' for k, v in series.head(6).items())}"
    return [line + "."]


def _correlations(df: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    measures = [c.name for c in schema.columns if c.role == "measure"]
    if len(measures) < 2:
        return []
    corr = df[measures].astype(float).corr()
    pairs = []
    for i, a in enumerate(measures):
        for b in measures[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.5:
                pairs.append((abs(r), a, b, r))
    pairs.sort(reverse=True)
    return [
        f"Correlation: {a} and {b} move {'together' if r > 0 else 'in opposite directions'} (r = {r:.2f})."
        for _, a, b, r in pairs[:3]
    ]


def summarize_trends(
    df: pd.DataFrame,
    schema: DatasetSchema,
    measure: str | None = None,
    time_column: str | None = None,
    dimensions: list[str] | None = None,
    agg: str = "sum",
) -> str:
    """Return bullet lines of key facts. Missing arguments are chosen from the schema."""
    measure = resolve_column(measure, df.columns) if measure else pick_measure(schema)
    time_column = resolve_column(time_column, df.columns) if time_column else pick_time(schema)
    dims = [resolve_column(d, df.columns) for d in dimensions] if dimensions else pick_dimensions(schema)
    if measure and not pd.api.types.is_numeric_dtype(df[measure]):
        measure = None
    if agg not in ("sum", "mean", "median", "max", "min"):
        agg = "sum"

    lines = [f"Dataset: {len(df):,} rows x {df.shape[1]} columns."]
    if measure:
        m = df[measure].dropna()
        total = f"total {_num(m.sum())}, " if agg == "sum" else ""
        lines.append(
            f"{measure}: {total}mean {_num(m.mean())}, median {_num(m.median())}, "
            f"range {_num(m.min())} to {_num(m.max())}."
        )
        if len(m) > 20 and m.std() > 0:
            q1, q3 = m.quantile([0.25, 0.75])
            outliers = int(((m < q1 - 3 * (q3 - q1)) | (m > q3 + 3 * (q3 - q1))).sum())
            if outliers:
                lines.append(f"{measure} has {outliers:,} extreme values (beyond 3x the interquartile range).")
            skew = float(m.skew())
            if abs(skew) > 1:
                lines.append(f"{measure} is {'right' if skew > 0 else 'left'}-skewed (skew {skew:.1f}): medians are more typical than means.")
    if time_column:
        lines += _time_section(df, time_column, measure, agg)
    for dim in dims:
        lines += _dimension_section(df, dim, measure, agg)
    lines += _correlations(df, schema)
    gaps = [f"{c.name} ({c.missing_pct:.0f}%)" for c in schema.columns if c.missing_pct >= 5]
    if gaps:
        lines.append(f"Missing data: {', '.join(gaps)}.")
    if schema.duplicate_rows:
        lines.append(f"{schema.duplicate_rows:,} duplicated rows.")
    return "\n".join(lines)
