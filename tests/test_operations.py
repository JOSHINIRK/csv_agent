import pytest

from csvagent.models import Aggregation, Filter, QueryPlan
from csvagent.operations import PlanError, execute_plan, frame_to_text, resolve_column


def test_group_sum_sorted_desc(small_df):
    plan = QueryPlan(group_by=["region"], aggregations=[Aggregation(func="sum", column="sales")])
    out = execute_plan(small_df, plan)
    assert list(out.columns) == ["Region", "Sales"]
    assert out.iloc[0].tolist() == ["West", 350.0]
    assert "(missing)" in set(out["Region"])


def test_filters_case_insensitive_and_between(small_df):
    plan = QueryPlan(
        filters=[
            Filter(column="Region", op="in", value=["west", "EAST"]),
            Filter(column="Date", op="between", value=["2024-02-01", "2024-02-28"]),
        ],
        aggregations=[Aggregation(func="count")],
    )
    assert execute_plan(small_df, plan)["Count"].iloc[0] == 2


def test_numeric_comparison_and_raw_rows(small_df):
    plan = QueryPlan(filters=[Filter(column="Sales", op=">=", value="100")], columns=["Product", "Sales"], sort_by="Sales")
    out = execute_plan(small_df, plan)
    assert out["Sales"].tolist() == [300.0, 250.0, 100.0]
    assert list(out.columns) == ["Product", "Sales"]


def test_monthly_time_grain(small_df):
    plan = QueryPlan(
        group_by=["Date"], time_grain="month", aggregations=[Aggregation(func="sum", column="Units", alias="Units sold")]
    )
    out = execute_plan(small_df, plan)
    assert out["Units sold"].tolist() == [1, 3, 8]
    assert out["Date"].dt.day.tolist() == [1, 1, 1]


def test_count_by_default_and_limit(small_df):
    out = execute_plan(small_df, QueryPlan(group_by=["Product"], limit=1))
    assert len(out) == 1 and out["Count"].iloc[0] == 2


def test_multiple_aggs_get_unique_names(small_df):
    plan = QueryPlan(
        group_by=["Product"],
        aggregations=[Aggregation(func="sum", column="Sales"), Aggregation(func="mean", column="Sales")],
    )
    assert list(execute_plan(small_df, plan).columns) == ["Product", "Sales", "Average Sales"]


def test_errors_help_the_model(small_df):
    with pytest.raises(PlanError, match="Did you mean 'Sales'"):
        execute_plan(small_df, QueryPlan(aggregations=[Aggregation(func="sum", column="Saless")]))
    with pytest.raises(PlanError, match="non-numeric"):
        execute_plan(small_df, QueryPlan(aggregations=[Aggregation(func="mean", column="Region")]))
    with pytest.raises(PlanError, match="find_values"):
        execute_plan(small_df, QueryPlan(filters=[Filter(column="Region", value="Westt")]))


def test_resolve_column_variants():
    cols = ["Sub-Category", "Order Date"]
    assert resolve_column("sub category", cols) == "Sub-Category"
    assert resolve_column("ORDER_DATE", cols) == "Order Date"


def test_frame_to_text_truncates(small_df):
    text = frame_to_text(small_df, 2)
    assert "4 more rows" in text and "2024-01-05" in text
