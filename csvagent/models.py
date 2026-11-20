"""Typed data structures shared by the importer, the query engine, the charts and the agent.

The agent never writes pandas code. It fills in a ``QueryPlan`` (and a ``ChartSpec``)
and Pydantic AI validates them against these models before anything runs, so a
malformed request is sent back to the model as a retry instead of crashing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SemanticType = Literal["integer", "float", "boolean", "datetime", "categorical", "text", "identifier"]
Role = Literal["measure", "dimension", "time", "id", "text"]


# --------------------------------------------------------------------------- schema
class ValueCount(BaseModel):
    value: str
    count: int


class ColumnSchema(BaseModel):
    name: str
    dtype: str = Field(description="pandas dtype after parsing")
    semantic_type: SemanticType
    role: Role = Field(description="measure = numbers to aggregate; dimension = groups; time = dates")
    non_null: int
    missing: int
    missing_pct: float
    unique: int
    sample_values: list[str] = []
    min: str | None = None
    max: str | None = None
    mean: float | None = None
    top_values: list[ValueCount] = []
    notes: list[str] = []

    def describe(self) -> str:
        """One compact line for the model's prompt."""
        bits = [f"{self.name!r}: {self.semantic_type} ({self.role})"]
        if self.min is not None and self.max is not None:
            bits.append(f"range {self.min} .. {self.max}")
        if self.mean is not None:
            bits.append(f"mean {self.mean:,.4g}")
        if self.role == "text":
            bits.append(f"{self.unique} distinct free-text values, e.g. {self.sample_values[0][:80]!r}" if self.sample_values else "free text")
        elif self.top_values and self.role in ("dimension", "id"):
            shown = ", ".join(v.value for v in self.top_values[:8])
            more = f" (+{self.unique - 8} more)" if self.unique > 8 else ""
            bits.append(f"{self.unique} values: {shown}{more}")
        elif self.role != "dimension":
            bits.append(f"{self.unique} distinct")
        if self.missing:
            bits.append(f"{self.missing_pct:.1f}% missing")
        return "; ".join(bits)


class DatasetSchema(BaseModel):
    name: str
    rows: int
    columns: list[ColumnSchema]
    delimiter: str = ","
    encoding: str = "utf-8"
    duplicate_rows: int = 0
    warnings: list[str] = []

    def column(self, name: str) -> ColumnSchema | None:
        return next((c for c in self.columns if c.name == name), None)

    def names(self, role: Role | None = None) -> list[str]:
        return [c.name for c in self.columns if role is None or c.role == role]

    def to_prompt(self) -> str:
        lines = [f"Dataset {self.name!r}: {self.rows:,} rows, {len(self.columns)} columns."]
        lines += [f"- {c.describe()}" for c in self.columns]
        if self.warnings:
            lines.append("Data notes: " + " ".join(self.warnings))
        return "\n".join(lines)


# --------------------------------------------------------------------------- queries
FilterOp = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not in", "contains", "between", "is null", "not null"]
AggFunc = Literal["sum", "mean", "median", "min", "max", "count", "nunique", "std"]
TimeGrain = Literal["auto", "day", "week", "month", "quarter", "year"]


class Filter(BaseModel):
    """Keep only rows where <column> <op> <value>."""

    column: str
    op: FilterOp = "=="
    value: Any = Field(
        default=None,
        description="Scalar for comparisons; list for 'in'/'not in'; [low, high] for 'between'; "
        "omit for 'is null'/'not null'. Dates as 'YYYY-MM-DD'.",
    )


class Aggregation(BaseModel):
    func: AggFunc
    column: str | None = Field(default=None, description="Column to aggregate. Omit with func='count' to count rows.")
    alias: str | None = Field(default=None, description="Name of the result column (optional)")


class QueryPlan(BaseModel):
    """A safe, declarative description of filtering + grouping + aggregation.

    Leave ``aggregations`` empty and ``group_by`` empty to get raw rows (e.g. for a
    scatter plot or histogram). With ``group_by`` but no aggregations, rows are counted.
    """

    filters: list[Filter] = []
    group_by: list[str] = Field(default=[], description="Columns to group by (dimensions or date columns)")
    time_grain: TimeGrain = Field(
        default="auto", description="How date columns in group_by are bucketed; 'auto' picks from the date span"
    )
    aggregations: list[Aggregation] = []
    columns: list[str] = Field(default=[], description="Columns to return when there is no grouping/aggregation")
    sort_by: str | None = Field(default=None, description="Result column to sort by (an alias or a group column)")
    descending: bool = True
    limit: int | None = Field(default=None, description="Keep only the first N result rows (e.g. top 10)")


# --------------------------------------------------------------------------- charts
ChartType = Literal["bar", "horizontal_bar", "stacked_bar", "line", "area", "scatter", "histogram", "box", "pie", "heatmap"]


class ChartSpec(BaseModel):
    """How to draw the result of a QueryPlan. x / y / color / value name result columns."""

    chart_type: ChartType
    title: str
    x: str | None = Field(default=None, description="Category, date or numeric column on the x axis")
    y: str | None = Field(default=None, description="Numeric column on the y axis (bar height, line value, ...)")
    color: str | None = Field(default=None, description="Optional column that splits the data into coloured series")
    value: str | None = Field(default=None, description="heatmap only: numeric column shown as cell colour")
    bins: int | None = Field(default=None, description="histogram only: number of bins")


# --------------------------------------------------------------------------- agent output
class AgentAnswer(BaseModel):
    """What the agent returns to the user at the end of a turn."""

    answer: str = Field(description="Markdown answer that quotes the key numbers from tool results")
    chart_ids: list[int] = Field(default=[], description="IDs of charts (from create_chart) to show, in order")
    table_ids: list[int] = Field(default=[], description="IDs of tables (from run_query) worth showing")
    follow_ups: list[str] = Field(default=[], description="Up to 3 short follow-up questions the user could ask")
