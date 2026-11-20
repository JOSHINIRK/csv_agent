# CSV Data Explorer (AI agent)

Upload a CSV, then ask questions about it in plain English, for example:

- "Show me a bar chart of sales by region"
- "Summarise the key trends"
- "Top 5 states by profit in 2025"

The answer comes back as a short written summary with interactive charts and tables. It is built with **Pydantic AI**, **pandas**, **Plotly** and **Streamlit**.

```
CSV file ──► Import + schema detection ──► Pydantic AI agent ──► answer + charts + tables
            (encoding, delimiter, types,    (tools: run_query,     (Streamlit chat UI,
             roles: measure/dimension/time)  create_chart, ...)      CLI, HTML demo report)
```

## What it covers

| Deliverable | Where |
|---|---|
| CSV import module with schema detection | [csvagent/ingest.py](csvagent/ingest.py) |
| Agent that parses questions and applies filtering/aggregation | [csvagent/agent.py](csvagent/agent.py), [csvagent/operations.py](csvagent/operations.py) |
| Visualisations generated from prompts | [csvagent/charts.py](csvagent/charts.py) |
| Trend summaries | [csvagent/insights.py](csvagent/insights.py) |
| Fuzzy and embedding lookup for large datasets | [csvagent/lookup.py](csvagent/lookup.py) |
| UI for uploading and conversational questions | [app.py](app.py) |
| Demonstration on sample datasets | [samples/](samples/), [scripts/demo.py](scripts/demo.py) |

## Setup (Windows, PowerShell)

```powershell
cd E:\Joshini_tcs\TCS_Projects\csv_data_explorer
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then paste your OpenAI API key into .env
```

## Run

```powershell
.venv\Scripts\streamlit run app.py                                                   # web app
.venv\Scripts\python scripts\ask.py samples\retail_sales.csv "Sales by region as a bar chart"   # one question
.venv\Scripts\python scripts\ask.py samples\census_income.csv                         # interactive terminal chat
.venv\Scripts\python scripts\demo.py                                                  # full demo -> docs\demo_output\report.html
.venv\Scripts\python -m pytest                                                        # 45 offline tests, no API key needed
```

In the app, choose a sample or upload your own CSV in the sidebar. The **Schema** tab shows what was detected and the **Data** tab previews the rows. Ask questions in the **Ask** tab: you can click a starter question or type your own. Each answer has two expanders:
- **Data behind the chart**: the numbers in table form.
- **How this was computed**: every tool call the agent made, including the exact query plan.

## How it works

### 1. Import and schema detection ([ingest.py](csvagent/ingest.py))
- **Encoding.** Tries UTF-8 (with or without BOM), then cp1252, then latin-1.
- **Delimiter.** Sniffs `,` `;` tab and `|`.
- **Missing values.** `?`, `N/A`, `-`, `null` and similar markers are read as missing. The list is in `config.yaml`.
- **Header clean-up.** Trims headers, renames blank or duplicate ones, and drops a leaked `Unnamed: 0` index column.
- **Types.** Numbers written as text (`$1,234.50`, `45%`, `(20)`) become numbers. Dates become datetimes, and day-first versus month-first is detected. Yes/No columns become booleans. Codes with leading zeros, such as ZIP codes, stay text.
- **Roles.** Each column gets a type and a role:
  - **measure**: a number to aggregate;
  - **dimension**: a category to group by;
  - **time**: a date or year;
  - **id**: an identifier;
  - **text**: free text.

  The importer also records ranges, top values and missing %. This summary goes into the agent's instructions, so the agent knows the data before the first question.

### 2. The agent ([agent.py](csvagent/agent.py))
A Pydantic AI `Agent[DataContext, AgentAnswer]`. Its dependencies are the DataFrame, the schema, and lists that collect the charts and tables it produces. It has six tools:

| Tool | Purpose |
|---|---|
| `run_query(plan, title)` | Filter → group → aggregate → sort → limit. Returns the table. |
| `create_chart(plan, chart)` | Same plan, then draws it: bar, horizontal bar, stacked bar, line, area, scatter, histogram, box, pie, heatmap. |
| `summarize_trends(...)` | Computed facts: totals, change over time, year-over-year, peaks, seasonality, top/bottom categories, correlations, data quality. |
| `find_values(text)` | Finds which column holds "california" and its exact spelling (fuzzy). |
| `find_columns(phrase)` | Maps words like "revenue" to a column (fuzzy, plus optional embeddings). |
| `column_profile(column)` | Quantiles and value counts for one column. |

**The model never writes code.** `QueryPlan` and `ChartSpec` are Pydantic models ([models.py](csvagent/models.py)), so every request is validated, then executed by our own pandas code.

**Mistakes are retried with a hint.** If the model uses a wrong column name or a misspelt value, the tool raises `ModelRetry` with a hint such as *"Unknown column 'Regoin'. Did you mean 'Region'?"*, and the model fixes its plan.

**The final output is typed.** `AgentAnswer` holds the markdown answer, the chart and table IDs to show, and suggested follow-up questions.

**Numbers come from tools.** The instructions require every figure to come from a tool result. For "summarise" questions, the trends tool computes the statistics and the LLM only narrates them.

**Follow-up questions.** Earlier turns are resent as compact question/answer pairs, so questions like "now only for 2025" work.

### 3. Charts ([charts.py](csvagent/charts.py))
Plotly charts follow these colour and layout rules:
- **Colour.** A fixed, colour-blind-checked order of categorical colours. Each category keeps the same colour across charts. Categories beyond 8 fold into a grey "Other" (beyond 3 for scatter plots).
- **Heatmaps.** A single-hue ramp for magnitudes, and a red–grey–blue diverging scale centred on zero for signed values such as profit/loss.
- **Layout.** Thin lines, rounded bar ends, light gridlines, one y-axis, and a legend only when there are several series.

### 4. Large datasets
The LLM never sees raw rows. It sees the schema, aggregated results (capped at `table_rows_to_model` rows) and computed facts, so token cost does not grow with file size. Scatter, box and histogram charts are down-sampled to `max_chart_points`.

High-cardinality text columns (thousands of cities or products) are searched locally with `find_values` instead of being listed in the prompt.

Set `semantic_search: true` to also embed the column descriptions with OpenAI embeddings. This is useful for wide files whose column names don't match the user's words.

## Sample datasets

`scripts/make_samples.py` generates two seeded, synthetic files with the same shape as popular Kaggle datasets. The numbers are made up, but they contain real patterns to discover.

| File | Shaped like | Rows | Patterns to find |
|---|---|---|---|
| `retail_sales.csv` | Kaggle "Superstore Sales" | 6,484 | ~12%/yr growth, Sep/Nov/Dec peaks, West biggest region, Technology biggest category, **Tables lose money**, profit falls as discount rises |
| `census_income.csv` | Kaggle/UCI "Adult Census Income" | 6,000 | income bracket vs education/occupation/hours/sex, `?` for missing values (as in the original) |
| `hr_attrition.csv` | Kaggle "IBM HR Attrition" | 1,200 | attrition highest in Customer Support and Sales and with overtime; salaries stored as `"$85,000"` text, day-first dates, Yes/No columns |
| `website_traffic.csv` | European Excel export | 2,190 | `;` separator and decimal commas (`0,505`); traffic grows through the year, Black Friday spike, Email converts best |
| `house_prices.csv` | Kaggle "House Prices" | 1,500 | price vs living area (scatter), neighbourhood premiums, missing lot sizes, `None` as a real Garage value, free-text descriptions |

**Using the real Kaggle files:** download e.g. *Superstore Sales* or *Adult Census Income* from Kaggle and upload the CSV in the app (or pass it to `scripts/ask.py`). Nothing is dataset-specific. The importer adapts to any column names, and the starter questions are generated from the detected schema.

## Configuration ([config.yaml](config.yaml))

| Key | Default | Meaning |
|---|---|---|
| `model` | `openai:gpt-4.1-mini` | Any Pydantic AI model string with tool calling, e.g. `openai:gpt-4.1`, `anthropic:claude-sonnet-5-5` |
| `temperature` | `0.1` | Set to `null` for reasoning models (o-series, gpt-5) |
| `agent_max_requests` | `12` | Cap on model calls per question |
| `history_turns` | `6` | Earlier Q/A pairs sent with a follow-up |
| `categorical_max_unique` | `60` | Text columns with at most this many values are treated as categories |
| `max_chart_points` / `max_color_series` | `20000` / `8` | Chart down-sampling and "Other" folding |
| `semantic_search` | `false` | Embedding-based column matching |

Any key can be overridden with an environment variable, for example `CSVX_MODEL=openai:gpt-4.1`.

## Project layout

```
csv_data_explorer/
├── app.py                  Streamlit UI (upload, schema, data preview, chat)
├── config.yaml             settings
├── csvagent/
│   ├── ingest.py           CSV import + schema detection
│   ├── models.py           Pydantic models: schema, QueryPlan, ChartSpec, AgentAnswer
│   ├── operations.py       safe execution of QueryPlan with pandas
│   ├── charts.py           Plotly chart builder
│   ├── insights.py         trend / key-facts engine
│   ├── lookup.py           fuzzy value search, optional embedding column search
│   ├── agent.py            Pydantic AI agent, tools, conversation turns
│   ├── suggest.py          starter questions from the schema
│   └── config.py           settings loader
├── samples/                demo datasets (regenerate: python scripts/make_samples.py)
├── scripts/                ask.py (CLI), demo.py (HTML demo report), make_samples.py
└── tests/                  offline tests (scripted FunctionModel, Streamlit AppTest)
```

## Tests
`pytest` runs 45 tests with no network access. They cover:
- the importer: types, delimiters, encodings, day-first dates, missing markers, duplicate headers;
- query execution and its error hints;
- every chart type;
- the trend engine on the sample data's built-in patterns;
- value and column lookup;
- the agent loop, using Pydantic AI's `FunctionModel` to script tool calls, including the retry-on-bad-column path;
- the Streamlit page, using `AppTest`.

## Limitations
- One CSV at a time (no joins across files).
- The query engine covers filter / group / aggregate / sort / top-N. Calculated columns such as "profit margin = profit / sales" are not supported: the agent can only compute each part separately.
- Files are loaded fully into memory (fine up to a few hundred MB).
