from csvagent.insights import pick_dimensions, pick_measure, pick_time, summarize_trends
from csvagent.lookup import ColumnIndex, ValueIndex
from csvagent.suggest import starter_questions


def test_sales_trends_find_the_built_in_patterns(sales):
    text = summarize_trends(sales.df, sales.schema)
    assert "Sales: total" in text
    assert "direction: upward" in text
    assert "December" in text or "November" in text  # seasonal peak
    assert "By Region (sum of Sales): top is West" in text
    assert "By Category (sum of Sales): top is Technology" in text


def test_mean_aggregation_and_defaults(census):
    text = summarize_trends(census.df, census.schema, measure="age", agg="mean")
    assert "total" not in text.split("\n")[1]
    assert "Missing data: workclass" in text
    assert pick_time(census.schema) is None


def test_pickers(sales):
    assert pick_measure(sales.schema) == "Sales"
    assert pick_time(sales.schema) == "Order Date"
    assert "Region" in pick_dimensions(sales.schema)


def test_value_index_exact_substring_fuzzy(sales):
    index = ValueIndex.build(sales.df, sales.schema)
    assert index.search("california")[0].value == "California"
    assert index.search("tech")[0].value == "Technology"
    assert index.search("Califronia")[0].value == "California"
    assert index.search("zzzz") == []


def test_column_index_fuzzy_and_embeddings(sales):
    assert ColumnIndex(sales.schema).search("sub category")[0][0] == "Sub-Category"

    def fake_embed(texts):  # "revenue" lands next to the Sales column
        return [[1.0, 0.0] if t.startswith(("Sales", "revenue")) else [0.0, 1.0] for t in texts]

    assert ColumnIndex(sales.schema, embed=fake_embed).search("revenue")[0][0] == "Sales"


def test_starter_questions(sales):
    qs = starter_questions(sales.schema)
    assert qs[0].startswith("Summarise")
    assert any("bar chart of Sales by" in q for q in qs)
