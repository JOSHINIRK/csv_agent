"""Builds Plotly figures from a query result and a ``ChartSpec``.

Visual rules (from the project's data-viz guidelines):
- categorical colours are assigned in a fixed, colour-blind-checked order and
  follow the category, never its rank; categories past ``max_color_series``
  fold into "Other" instead of generating new hues;
- scatter plots use at most 3 colour series (all-pairs separation);
- one y axis only; thin 2px lines; bars with 4px rounded ends; light gridlines;
- heatmaps: one blue ramp (light -> dark) for magnitudes, a red/grey/blue diverging
  scale centred on zero when values change sign (e.g. profit and loss);
- a legend appears only when there are 2+ series; text stays in neutral ink.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .models import ChartSpec
from .operations import PlanError, resolve_column

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_COLOR = "#a3a29c"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging (signed values such as profit/loss): red <- neutral grey at zero -> blue, equal arms.
DIVERGING = [[0.0, "#a8302f"], [0.25, "#e34948"], [0.5, "#f0efec"], [0.75, "#3987e5"], [1.0, "#184f95"]]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
OTHER = "Other"
SCATTER_MAX_SERIES = 3


def _layout(fig: go.Figure, spec: ChartSpec, show_legend: bool) -> go.Figure:
    fig.update_layout(
        title={"text": spec.title, "x": 0, "xanchor": "left", "font": {"size": 17, "color": TEXT_PRIMARY}},
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "Inter, Segoe UI, system-ui, sans-serif", "size": 13, "color": TEXT_SECONDARY},
        margin={"l": 8, "r": 16, "t": 84 if show_legend else 56, "b": 8},
        showlegend=show_legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "title": None},
        hoverlabel={"bgcolor": "white", "font": {"color": TEXT_PRIMARY}},
        bargap=0.25,
        bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", automargin=True, title_font={"color": TEXT_SECONDARY})
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, automargin=True)
    return fig


def fold_categories(df: pd.DataFrame, column: str, max_series: int, weight: str | None = None) -> pd.DataFrame:
    """Keep the largest ``max_series - 1`` categories and relabel the rest as "Other"."""
    values = df[column].astype("string")
    if values.nunique(dropna=True) <= max_series:
        return df
    ranking = df.groupby(values)[weight].sum() if weight and weight in df else values.value_counts()
    keep = set(ranking.sort_values(ascending=False).index[: max_series - 1])
    out = df.copy()
    out[column] = values.where(values.isin(keep), OTHER)
    return out


def color_map(categories: list) -> dict:
    """Fixed colour per category: assigned in first-seen order, with "Other" always grey."""
    mapping, i = {}, 0
    for cat in categories:
        key = str(cat)
        if key in mapping:
            continue
        if key == OTHER:
            mapping[key] = OTHER_COLOR
        else:
            mapping[key] = SERIES[i % len(SERIES)]
            i += 1
    return mapping


def _order(df: pd.DataFrame, column: str, by: str | None) -> list[str]:
    s = df[column].astype("string")
    if by and by in df and pd.api.types.is_numeric_dtype(df[by]):
        order = df.assign(_k=s).groupby("_k")[by].sum().sort_values(ascending=False).index.tolist()
    else:
        order = s.drop_duplicates().tolist()
    if OTHER in order:
        order.remove(OTHER)
        order.append(OTHER)
    return order


def _require(spec: ChartSpec, df: pd.DataFrame, field: str, numeric: bool = False) -> str:
    name = getattr(spec, field)
    if not name:
        raise PlanError(f"A {spec.chart_type} chart needs '{field}'. Result columns: {', '.join(map(str, df.columns))}")
    col = resolve_column(name, df.columns)
    if numeric and not pd.api.types.is_numeric_dtype(df[col]):
        raise PlanError(f"'{field}' column {col!r} must be numeric for a {spec.chart_type} chart.")
    return col


def _sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    return df.sample(limit, random_state=0) if len(df) > limit else df


def build_chart(df: pd.DataFrame, spec: ChartSpec, max_points: int = 20000, max_series: int = 8) -> go.Figure:
    """Draw ``df`` (already filtered/aggregated) as described by ``spec``."""
    if df.empty:
        raise PlanError("The query returned no rows, so there is nothing to chart. Loosen the filters.")
    kind = spec.chart_type
    data = df.copy()
    color = resolve_column(spec.color, data.columns) if spec.color else None
    y_name = resolve_column(spec.y, data.columns) if spec.y else None

    if color:
        limit = SCATTER_MAX_SERIES if kind == "scatter" else max_series
        data = fold_categories(data, color, limit, weight=y_name)
        data[color] = data[color].astype("string")

    if kind in ("bar", "horizontal_bar", "stacked_bar"):
        x = _require(spec, data, "x")
        y = _require(spec, data, "y", numeric=True)
        if not color:
            data = fold_categories(data, x, 40, weight=y) if data[x].nunique() > 40 else data
        horizontal = kind == "horizontal_bar"
        cats = _order(data, color, y) if color else None
        fig = px.bar(
            data,
            x=y if horizontal else x,
            y=x if horizontal else y,
            color=color,
            orientation="h" if horizontal else "v",
            barmode="stack" if kind == "stacked_bar" else "group",
            color_discrete_map=color_map(cats) if cats else None,
            category_orders={color: cats} if cats else None,
            color_discrete_sequence=[SERIES[0]],
        )
        fig.update_traces(marker_cornerradius=4, marker_line_width=0)
        if kind == "stacked_bar":
            fig.update_traces(marker_line_width=2, marker_line_color=SURFACE, marker_cornerradius=0)
        if horizontal:
            fig.update_yaxes(autorange="reversed", title=None)
        if not pd.api.types.is_datetime64_any_dtype(data[x]) and not pd.api.types.is_numeric_dtype(data[x]):
            fig.update_xaxes(type="category") if not horizontal else fig.update_yaxes(type="category")
        return _layout(fig, spec, show_legend=bool(color))

    if kind in ("line", "area"):
        x = _require(spec, data, "x")
        y = _require(spec, data, "y", numeric=True)
        data = data.sort_values(x)
        cats = _order(data, color, y) if color else None
        maker = px.line if kind == "line" else px.area
        fig = maker(
            data,
            x=x,
            y=y,
            color=color,
            color_discrete_map=color_map(cats) if cats else None,
            category_orders={color: cats} if cats else None,
            color_discrete_sequence=[SERIES[0]],
            markers=len(data) <= 40 if kind == "line" else False,
        )
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_layout(hovermode="x unified")
        return _layout(fig, spec, show_legend=bool(color))

    if kind == "scatter":
        x = _require(spec, data, "x", numeric=True)
        y = _require(spec, data, "y", numeric=True)
        data = _sample(data, max_points)
        cats = _order(data, color, None) if color else None
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            color_discrete_map=color_map(cats) if cats else None,
            category_orders={color: cats} if cats else None,
            color_discrete_sequence=[SERIES[0]],
            opacity=0.7 if len(data) > 500 else 0.9,
        )
        fig.update_traces(marker={"size": 8, "line": {"width": 1, "color": SURFACE}})
        return _layout(fig, spec, show_legend=bool(color))

    if kind == "histogram":
        x = _require(spec, data, "x", numeric=True) if spec.x else _require(spec, data, "y", numeric=True)
        data = _sample(data, max_points * 5)
        cats = _order(data, color, None) if color else None
        fig = px.histogram(
            data,
            x=x,
            color=color,
            nbins=spec.bins or 30,
            barmode="overlay" if color else "relative",
            color_discrete_map=color_map(cats) if cats else None,
            category_orders={color: cats} if cats else None,
            color_discrete_sequence=[SERIES[0]],
            opacity=0.75 if color else 1.0,
        )
        fig.update_traces(marker_line_width=1, marker_line_color=SURFACE)
        fig.update_yaxes(title="Rows")
        return _layout(fig, spec, show_legend=bool(color))

    if kind == "box":
        y = _require(spec, data, "y", numeric=True)
        x = resolve_column(spec.x, data.columns) if spec.x else None
        if x:
            data = fold_categories(data, x, 20)
        data = _sample(data, max_points)
        fig = px.box(data, x=x, y=y, color=color, color_discrete_sequence=SERIES, points=False)
        fig.update_traces(line_width=2, marker_size=6)
        return _layout(fig, spec, show_legend=bool(color))

    if kind == "pie":
        names = _require(spec, data, "x")
        values = _require(spec, data, "y", numeric=True)
        data = fold_categories(data, names, 6, weight=values)
        data[names] = data[names].astype("string")
        data = data.groupby(names, as_index=False)[values].sum()
        cats = _order(data, names, values)
        fig = px.pie(
            data,
            names=names,
            values=values,
            hole=0.55,
            color=names,
            color_discrete_map=color_map(cats),
            category_orders={names: cats},
        )
        fig.update_traces(
            marker={"line": {"color": SURFACE, "width": 2}}, textinfo="percent", sort=False, direction="clockwise"
        )
        return _layout(fig, spec, show_legend=True)

    if kind == "heatmap":
        x = _require(spec, data, "x")
        y = _require(spec, data, "y")
        value = _require(spec, data, "value", numeric=True)
        grid = data.pivot_table(index=y, columns=x, values=value, aggfunc="sum", observed=True)
        signed = grid.min().min() < 0 < grid.max().max()
        if signed:  # losses and gains: diverging scale centred on zero
            fig = px.imshow(grid, color_continuous_scale=DIVERGING, color_continuous_midpoint=0, aspect="auto", text_auto=".3s")
        else:
            fig = px.imshow(grid, color_continuous_scale=SEQUENTIAL_BLUE, aspect="auto", text_auto=".3s")
        fig.update_traces(xgap=2, ygap=2)
        fig.update_layout(coloraxis_colorbar={"title": value})
        fig = _layout(fig, spec, show_legend=False)
        fig.update_xaxes(type="category", title=x)
        fig.update_yaxes(type="category", title=y, gridcolor=SURFACE)
        return fig

    raise PlanError(f"Unsupported chart type {kind!r}.")
