"""The Pydantic AI agent that answers questions about a loaded CSV.

How a question is handled:
1. The instructions carry the detected schema (column names, types, roles,
   example values), so the model knows the data before calling any tool.
2. The model calls tools with *typed* arguments. ``run_query`` and
   ``create_chart`` take a ``QueryPlan`` / ``ChartSpec`` (Pydantic models), so
   the filtering and aggregation it asks for is validated and executed by our own
   pandas code - the model never writes or runs code.
3. If a plan is wrong (unknown column, misspelt value), the tool raises
   ``ModelRetry`` with a precise hint and the model corrects itself.
4. The run ends with a structured ``AgentAnswer``: markdown text, the chart and
   table IDs to display, and suggested follow-up questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from .charts import build_chart
from .config import Settings, load_settings
from .ingest import LoadedDataset
from .insights import summarize_trends as compute_trends
from .lookup import ColumnIndex, ValueIndex, openai_embedder
from .models import AgentAnswer, ChartSpec, DatasetSchema, QueryPlan
from .operations import PlanError, execute_plan, frame_to_text, resolve_column


# --------------------------------------------------------------------------- run state
@dataclass
class ChartArtifact:
    id: int
    spec: ChartSpec
    plan: QueryPlan
    data: pd.DataFrame
    figure: go.Figure


@dataclass
class TableArtifact:
    id: int
    title: str
    plan: QueryPlan
    data: pd.DataFrame


@dataclass
class Step:
    """One tool call, kept so the UI can show how an answer was produced."""

    tool: str
    args: dict[str, Any]
    ok: bool
    summary: str


@dataclass
class DataContext:
    """Dependencies handed to every tool: the data plus what this turn produced."""

    df: pd.DataFrame
    schema: DatasetSchema
    settings: Settings
    values: ValueIndex
    columns: ColumnIndex
    charts: list[ChartArtifact] = field(default_factory=list)
    tables: list[TableArtifact] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def from_dataset(cls, dataset: LoadedDataset, settings: Settings | None = None) -> DataContext:
        settings = settings or load_settings()
        embed = openai_embedder(settings.embedding_model) if settings.semantic_search else None
        return cls(
            df=dataset.df,
            schema=dataset.schema,
            settings=settings,
            values=ValueIndex.build(dataset.df, dataset.schema),
            columns=ColumnIndex(dataset.schema, embed=embed),
        )

    def new_turn(self) -> None:
        self.charts, self.tables, self.steps = [], [], []


# --------------------------------------------------------------------------- instructions
INSTRUCTIONS = """\
You are a careful data analyst working on ONE tabular dataset that the user uploaded.
You answer questions about it with numbers, charts and short explanations.

Rules:
- Never invent numbers. Every figure you state must come from a tool result in this conversation.
- Use the exact column names from the schema below. If the user names a value (a region, product,
  city, ...) and you are not sure of its exact spelling or which column holds it, call find_values first.
  If they use a word that is not a column name ("revenue", "turnover"), call find_columns.
- To compute anything (totals, averages, counts, top-N, filters, values over time) call run_query.
- When the user asks for a chart/plot/graph/visual - or when a chart clearly helps (a trend over time,
  a comparison across many categories) - call create_chart. Pick the form from the data's job:
  * comparison across categories -> bar (horizontal_bar when labels are long or there are > 8 bars);
  * change over time -> line (group_by the date column; time_grain 'auto' unless the user names one);
  * part-of-whole with <= 6 parts -> pie; otherwise bar;
  * composition across categories -> stacked_bar with color;
  * distribution of one numeric column -> histogram (raw rows: no group_by, no aggregations);
  * relationship between two numeric columns -> scatter (raw rows);
  * spread by category -> box (raw rows); two categorical dimensions -> heatmap (value = the measure).
  Chart x / y / color / value must be column names of the query result: group columns keep their names,
  aggregation results are named by their alias (set a short alias such as "Total Sales").
- For "summarise", "key trends", "insights", "overview" questions call summarize_trends first, then
  add one or two charts that show the most important findings (usually a trend line and a top-categories bar).
- If the data cannot answer the question, say so plainly and suggest what it can answer.

Answer format (the final AgentAnswer):
- answer: concise markdown. Lead with the direct answer, then 2-5 bullets with the key numbers.
  Format large numbers readably (1.2M, 45,300). Mention filters or assumptions you applied.
  Do not paste whole tables; the UI shows charts and tables next to your text.
- chart_ids / table_ids: the IDs returned by create_chart / run_query that should be shown.
- follow_ups: up to 3 short, specific questions this dataset can answer next.
"""


def _schema_instructions(ctx: RunContext[DataContext]) -> str:
    return "Detected schema of the loaded dataset:\n" + ctx.deps.schema.to_prompt()


# --------------------------------------------------------------------------- agent
def build_agent(model: Model | str | None = None, settings: Settings | None = None) -> Agent[DataContext, AgentAnswer]:
    settings = settings or load_settings()
    model_settings = ModelSettings(temperature=settings.temperature) if settings.temperature is not None else None
    agent: Agent[DataContext, AgentAnswer] = Agent(
        model or settings.model,
        deps_type=DataContext,
        output_type=AgentAnswer,
        instructions=[INSTRUCTIONS, _schema_instructions],
        model_settings=model_settings,
        retries=3,
        defer_model_check=True,
        name="csv_data_explorer",
    )

    def log(ctx: RunContext[DataContext], tool: str, args: dict, ok: bool, summary: str) -> None:
        ctx.deps.steps.append(Step(tool=tool, args=args, ok=ok, summary=summary))

    @agent.tool
    def column_profile(ctx: RunContext[DataContext], column: str) -> str:
        """Detailed statistics for one column: type, missing values, range, quartiles and most common values."""
        try:
            col = resolve_column(column, ctx.deps.df.columns)
        except PlanError as exc:
            log(ctx, "column_profile", {"column": column}, False, str(exc))
            raise ModelRetry(str(exc)) from exc
        s = ctx.deps.df[col]
        info = ctx.deps.schema.column(col)
        lines = [info.describe() if info else col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            lines.append("quantiles 5/25/50/75/95%: " + ", ".join(f"{v:,.4g}" for v in q.values))
            lines.append(f"sum {s.sum():,.4g}; std {s.std():,.4g}")
        else:
            top = s.astype("string").value_counts(dropna=False).head(25)
            lines.append("most common values: " + "; ".join(f"{k} ({v:,})" for k, v in top.items()))
        log(ctx, "column_profile", {"column": col}, True, f"Profiled {col!r}")
        return "\n".join(lines)

    @agent.tool
    def find_values(ctx: RunContext[DataContext], text: str, column: str | None = None) -> str:
        """Find which column contains a value the user mentioned, and its exact spelling (fuzzy match).

        Args:
            text: the word(s) the user used, e.g. "california" or "tech".
            column: optional column to search in; omit to search every text column.
        """
        if column:
            try:
                column = resolve_column(column, ctx.deps.df.columns)
            except PlanError as exc:
                raise ModelRetry(str(exc)) from exc
        matches = ctx.deps.values.search(text, column=column)
        log(ctx, "find_values", {"text": text, "column": column}, True, f"{len(matches)} match(es) for {text!r}")
        if not matches:
            return f"No value similar to {text!r} found in the text columns."
        return "\n".join(f"column {m.column!r}: value {m.value!r} ({m.rows:,} rows, score {m.score})" for m in matches)

    @agent.tool
    def find_columns(ctx: RunContext[DataContext], phrase: str) -> str:
        """Rank columns by how well they match a phrase the user used (e.g. 'revenue', 'customer age')."""
        ranked = ctx.deps.columns.search(phrase)
        log(ctx, "find_columns", {"phrase": phrase}, True, f"Best match {ranked[0][0]!r}" if ranked else "none")
        return "\n".join(f"{name!r} (similarity {score})" for name, score in ranked)

    @agent.tool
    def run_query(ctx: RunContext[DataContext], plan: QueryPlan, title: str) -> str:
        """Filter, group and aggregate the dataset and return the result table.

        Args:
            plan: what to compute. Examples - total sales by region:
                {"group_by": ["Region"], "aggregations": [{"func": "sum", "column": "Sales", "alias": "Total Sales"}]}
                monthly order count in 2024: {"filters": [{"column": "Order Date", "op": "between",
                "value": ["2024-01-01", "2024-12-31"]}], "group_by": ["Order Date"], "time_grain": "month",
                "aggregations": [{"func": "count"}]}
            title: short description of the table, e.g. "Total sales by region".
        """
        args = {"plan": plan.model_dump(exclude_defaults=True), "title": title}
        try:
            result = execute_plan(ctx.deps.df, plan)
        except (PlanError, KeyError, TypeError, ValueError) as exc:
            log(ctx, "run_query", args, False, str(exc))
            raise ModelRetry(str(exc)) from exc
        table_id = len(ctx.deps.tables) + 1
        ctx.deps.tables.append(TableArtifact(table_id, title, plan, result))
        log(ctx, "run_query", args, True, f"Table {table_id}: {len(result):,} rows")
        body = frame_to_text(result, ctx.deps.settings.table_rows_to_model)
        return f"table_id={table_id} ({len(result):,} rows)\n{body}"

    @agent.tool
    def create_chart(ctx: RunContext[DataContext], plan: QueryPlan, chart: ChartSpec) -> str:
        """Compute data with a QueryPlan and draw it. Returns a chart_id plus the plotted numbers.

        Args:
            plan: the data to plot (same format as run_query). Use raw rows (no group_by/aggregations)
                for histogram, scatter and box charts.
            chart: how to draw it; x / y / color / value must be columns of the plan's result.
        """
        args = {"plan": plan.model_dump(exclude_defaults=True), "chart": chart.model_dump(exclude_none=True)}
        try:
            data = execute_plan(ctx.deps.df, plan)
            figure = build_chart(
                data, chart, max_points=ctx.deps.settings.max_chart_points, max_series=ctx.deps.settings.max_color_series
            )
        except (PlanError, KeyError, TypeError, ValueError) as exc:
            log(ctx, "create_chart", args, False, str(exc))
            raise ModelRetry(str(exc)) from exc
        chart_id = len(ctx.deps.charts) + 1
        ctx.deps.charts.append(ChartArtifact(chart_id, chart, plan, data, figure))
        log(ctx, "create_chart", args, True, f"Chart {chart_id}: {chart.chart_type} '{chart.title}' ({len(data):,} rows)")
        raw = not plan.group_by and not plan.aggregations
        preview = (
            f"{len(data):,} raw rows plotted (use run_query or summarize_trends for statistics)."
            if raw
            else frame_to_text(data, min(ctx.deps.settings.table_rows_to_model, 25))
        )
        return f"chart_id={chart_id} created: {chart.chart_type} '{chart.title}'.\nPlotted data:\n{preview}"

    @agent.tool
    def summarize_trends(
        ctx: RunContext[DataContext],
        measure: str | None = None,
        time_column: str | None = None,
        dimensions: list[str] | None = None,
        agg: str = "sum",
    ) -> str:
        """Key facts about the dataset: totals, change over time, peaks, seasonality, top/bottom categories,
        strong correlations and data-quality issues. Omitted arguments are chosen automatically.

        Args:
            measure: numeric column to analyse (e.g. 'Sales').
            time_column: date or year column for the trend.
            dimensions: up to 3 category columns to break the measure down by.
            agg: 'sum' (default) for additive values, 'mean' for rates/prices/ages.
        """
        args = {"measure": measure, "time_column": time_column, "dimensions": dimensions, "agg": agg}
        try:
            text = compute_trends(ctx.deps.df, ctx.deps.schema, measure, time_column, dimensions, agg)
        except (PlanError, KeyError, TypeError, ValueError) as exc:
            log(ctx, "summarize_trends", args, False, str(exc))
            raise ModelRetry(str(exc)) from exc
        log(ctx, "summarize_trends", args, True, "Computed key facts")
        return text

    return agent


# --------------------------------------------------------------------------- one conversational turn
@dataclass
class Turn:
    question: str
    answer: AgentAnswer
    charts: list[ChartArtifact]
    tables: list[TableArtifact]
    steps: list[Step]
    requests: int = 0
    tokens: int = 0


def history_messages(turns: list[Turn], keep: int) -> list[ModelMessage]:
    """Earlier turns as compact question/answer messages (tool chatter is not resent)."""
    messages: list[ModelMessage] = []
    for t in turns[-keep:] if keep else []:
        messages.append(ModelRequest(parts=[UserPromptPart(content=t.question)]))
        messages.append(ModelResponse(parts=[TextPart(content=t.answer.answer)]))
    return messages


def ask(
    agent: Agent[DataContext, AgentAnswer],
    ctx: DataContext,
    question: str,
    previous: list[Turn] | None = None,
    model: Model | str | None = None,
) -> Turn:
    """Run one question through the agent and collect what it produced."""
    ctx.new_turn()
    result = agent.run_sync(
        question,
        deps=ctx,
        message_history=history_messages(previous or [], ctx.settings.history_turns) or None,
        usage_limits=UsageLimits(request_limit=ctx.settings.agent_max_requests),
        model=model,
    )
    answer: AgentAnswer = result.output
    charts = {c.id: c for c in ctx.charts}
    tables = {t.id: t for t in ctx.tables}
    shown_charts = [charts[i] for i in dict.fromkeys(answer.chart_ids) if i in charts]
    if not shown_charts and ctx.charts:
        shown_charts = list(ctx.charts)  # the model drew charts but forgot to list them
    shown_tables = [tables[i] for i in dict.fromkeys(answer.table_ids) if i in tables]
    usage = result.usage
    return Turn(
        question=question,
        answer=answer,
        charts=shown_charts,
        tables=shown_tables,
        steps=list(ctx.steps),
        requests=usage.requests,
        tokens=usage.total_tokens or 0,
    )
