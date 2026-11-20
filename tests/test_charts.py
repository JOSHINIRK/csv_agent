import pandas as pd
import pytest

from csvagent.charts import OTHER, SERIES, build_chart, color_map, fold_categories
from csvagent.models import Aggregation, ChartSpec, QueryPlan
from csvagent.operations import PlanError, execute_plan

BY_REGION = QueryPlan(group_by=["Region"], aggregations=[Aggregation(func="sum", column="Sales")])
CASES = [
    (BY_REGION, ChartSpec(chart_type="bar", title="t", x="Region", y="Sales")),
    (BY_REGION, ChartSpec(chart_type="horizontal_bar", title="t", x="Region", y="Sales")),
    (BY_REGION, ChartSpec(chart_type="pie", title="t", x="Region", y="Sales")),
    (
        QueryPlan(group_by=["Order Date", "Category"], time_grain="quarter", aggregations=[Aggregation(func="sum", column="Sales")]),
        ChartSpec(chart_type="line", title="t", x="Order Date", y="Sales", color="Category"),
    ),
    (
        QueryPlan(group_by=["Region", "Category"], aggregations=[Aggregation(func="sum", column="Sales")]),
        ChartSpec(chart_type="stacked_bar", title="t", x="Region", y="Sales", color="Category"),
    ),
    (
        QueryPlan(group_by=["Region", "Category"], aggregations=[Aggregation(func="sum", column="Profit")]),
        ChartSpec(chart_type="heatmap", title="t", x="Region", y="Category", value="Profit"),
    ),
    (QueryPlan(), ChartSpec(chart_type="histogram", title="t", x="Discount")),
    (QueryPlan(), ChartSpec(chart_type="scatter", title="t", x="Sales", y="Profit", color="Category")),
    (QueryPlan(), ChartSpec(chart_type="box", title="t", x="Region", y="Profit")),
]


@pytest.mark.parametrize("plan,spec", CASES, ids=[c[1].chart_type for c in CASES])
def test_every_chart_type_builds(sales, plan, spec):
    fig = build_chart(execute_plan(sales.df, plan), spec)
    assert fig.layout.title.text == "t"
    assert len(fig.data) >= 1


def test_single_series_uses_first_palette_colour(sales):
    fig = build_chart(execute_plan(sales.df, BY_REGION), ChartSpec(chart_type="bar", title="t", x="Region", y="Sales"))
    assert fig.data[0].marker.color == SERIES[0]
    assert fig.layout.showlegend is False


def test_fold_into_other_and_fixed_colours():
    df = pd.DataFrame({"k": list("abcdefghij"), "v": range(10, 0, -1)})
    folded = fold_categories(df, "k", 4, weight="v")
    assert set(folded["k"]) == {"a", "b", "c", OTHER}
    mapping = color_map(["a", OTHER, "b"])
    assert mapping["a"] == SERIES[0] and mapping["b"] == SERIES[1] and mapping[OTHER] not in SERIES


def test_bad_chart_column_raises_plan_error(sales):
    plan = QueryPlan(group_by=["Region"], aggregations=[Aggregation(func="sum", column="Sales", alias="Total")])
    data = execute_plan(sales.df, plan)
    with pytest.raises(PlanError, match="Did you mean 'Total'"):
        build_chart(data, ChartSpec(chart_type="bar", title="t", x="Region", y="Totl"))
    with pytest.raises(PlanError, match="no rows"):
        build_chart(data.iloc[0:0], ChartSpec(chart_type="bar", title="t", x="Region", y="Total"))
