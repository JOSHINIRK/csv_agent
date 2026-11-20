"""Agent tests with a scripted FunctionModel: real tools, real data, no network."""

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from csvagent import DataContext, ask, build_agent
from csvagent.agent import history_messages


def scripted(*steps):
    """A model that plays back tool calls in order; a dict for 'final' becomes the AgentAnswer."""
    seen = []

    def fn(messages, info: AgentInfo):
        seen.append(messages)
        name, args = steps[len(seen) - 1]
        if name == "final":
            name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(name, args)])

    return FunctionModel(fn), seen


def tool_returns(messages):
    return [p.content for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]


def test_bar_chart_question_end_to_end(sales, settings):
    model, seen = scripted(
        (
            "create_chart",
            {
                "plan": {"group_by": ["Region"], "aggregations": [{"func": "sum", "column": "Sales", "alias": "Total Sales"}]},
                "chart": {"chart_type": "bar", "title": "Sales by region", "x": "Region", "y": "Total Sales"},
            },
        ),
        ("final", {"answer": "West has the highest sales.", "chart_ids": [1], "follow_ups": ["Which state leads?"]}),
    )
    ctx = DataContext.from_dataset(sales, settings)
    turn = ask(build_agent(model, settings), ctx, "Show me a bar chart of sales by region")
    assert turn.answer.answer.startswith("West")
    assert len(turn.charts) == 1 and turn.charts[0].data.iloc[0]["Region"] == "West"
    assert [s.tool for s in turn.steps] == ["create_chart"]
    # the model saw the plotted numbers, so it can quote them
    assert "West," in tool_returns(seen[-1])[-1]
    # the detected schema is part of the instructions
    request = next(m for m in seen[0] if isinstance(m, ModelRequest))
    assert "Detected schema" in (request.instructions or "")


def test_bad_plan_is_retried_with_a_hint(sales, settings):
    model, seen = scripted(
        ("run_query", {"plan": {"group_by": ["Regoin"], "aggregations": [{"func": "sum", "column": "Sales"}]}, "title": "x"}),
        ("run_query", {"plan": {"group_by": ["Region"], "aggregations": [{"func": "sum", "column": "Sales"}]}, "title": "Sales by region"}),
        ("final", {"answer": "done", "table_ids": [1]}),
    )
    ctx = DataContext.from_dataset(sales, settings)
    turn = ask(build_agent(model, settings), ctx, "sales by region")
    assert [s.ok for s in turn.steps] == [False, True]
    retry_text = str(seen[1][-1].parts[0])
    assert "Did you mean 'Region'" in retry_text
    assert len(turn.tables) == 1 and turn.tables[0].title == "Sales by region"


def test_trend_summary_and_lookup_tools(sales, settings):
    model, seen = scripted(
        ("find_values", {"text": "califronia"}),
        ("summarize_trends", {}),
        ("final", {"answer": "Sales grew."}),
    )
    ctx = DataContext.from_dataset(sales, settings)
    turn = ask(build_agent(model, settings), ctx, "Summarise key trends in California")
    returns = tool_returns(seen[-1])
    assert "column 'State': value 'California'" in returns[0]
    assert "direction: upward" in returns[1]
    assert turn.charts == [] and turn.tables == []


def test_unlisted_charts_are_still_shown(sales, settings):
    model, _ = scripted(
        ("create_chart", {"plan": {}, "chart": {"chart_type": "histogram", "title": "Discounts", "x": "Discount"}}),
        ("final", {"answer": "Most orders have no discount."}),
    )
    turn = ask(build_agent(model, settings), DataContext.from_dataset(sales, settings), "distribution of discount")
    assert [c.spec.title for c in turn.charts] == ["Discounts"]


def test_history_is_compact_question_answer_pairs(sales, settings):
    model, _ = scripted(("final", {"answer": "first answer"}))
    ctx = DataContext.from_dataset(sales, settings)
    first = ask(build_agent(model, settings), ctx, "first question")
    messages = history_messages([first], keep=6)
    assert isinstance(messages[0].parts[0], UserPromptPart) and messages[0].parts[0].content == "first question"
    assert isinstance(messages[1].parts[0], TextPart) and messages[1].parts[0].content == "first answer"

    model2, seen = scripted(("final", {"answer": "second"}))
    ask(build_agent(model2, settings), ctx, "follow-up", previous=[first])
    texts = [getattr(p, "content", "") for m in seen[0] for p in m.parts]
    assert "first question" in texts and "follow-up" in texts
