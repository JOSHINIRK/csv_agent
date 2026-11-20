"""Streamlit front end: upload a CSV (or pick a sample), inspect its schema, and ask questions.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _use_project_venv() -> None:
    """Make `streamlit run app.py` work even when the global `streamlit` command starts it.

    The project's libraries (plotly, pydantic-ai, ...) live in .venv. When the app is
    launched by another Python, put .venv's site-packages first on the import path.
    Only done when that .venv was built from the same Python version (binary-compatible).
    """
    venv = Path(__file__).resolve().parent / ".venv"
    if Path(sys.prefix).resolve() == venv.resolve() or not venv.exists():
        return
    cfg = venv / "pyvenv.cfg"
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if cfg.exists() and f"version = {version}" not in cfg.read_text(encoding="utf-8", errors="ignore").replace("version_info", "version"):
        return
    for site in (venv / "Lib" / "site-packages", venv / "lib" / f"python{version}" / "site-packages"):
        if site.is_dir() and str(site) not in sys.path:
            sys.path.insert(0, str(site))


_use_project_venv()

import hashlib  # noqa: E402
import json  # noqa: E402

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from csvagent import CSVImportError, DataContext, LoadedDataset, Turn, ask, build_agent, load_csv, load_settings
from csvagent.config import SAMPLES_DIR, missing_api_key
from csvagent.suggest import starter_questions

load_dotenv()
st.set_page_config(page_title="CSV Data Explorer", page_icon=":material/query_stats:", layout="wide")

settings = load_settings()

SAMPLE_LABELS = {
    "retail_sales.csv": "Retail sales 2022-2025 (Superstore-style)",
    "census_income.csv": "Census income demographics (Adult-style)",
    "hr_attrition.csv": "HR employee attrition",
    "website_traffic.csv": "Website traffic 2025 (semicolon, decimal commas)",
    "house_prices.csv": "House prices (numeric relationships)",
}
ROLE_LABELS = {"measure": "Measure", "dimension": "Dimension", "time": "Time", "id": "Identifier", "text": "Free text"}


# --------------------------------------------------------------------------- cached resources
@st.cache_data(show_spinner="Reading the CSV and detecting its schema...", max_entries=8)
def load_dataset(raw: bytes, name: str) -> LoadedDataset:
    return load_csv(raw, name=name, settings=settings)


@st.cache_resource(show_spinner=False)
def get_agent(model: str):
    return build_agent(model, settings)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    source = st.segmented_control("Source", ["Sample dataset", "Upload a CSV"], default="Sample dataset", required=True)
    raw, name = None, None
    if source == "Upload a CSV":
        upload = st.file_uploader("CSV file", type=["csv", "tsv", "txt"], max_upload_size=settings.max_upload_mb)
        if upload is not None:
            raw, name = upload.getvalue(), upload.name
    else:
        samples = sorted(p.name for p in SAMPLES_DIR.glob("*.csv"))
        if samples:
            choice = st.selectbox("Sample", samples, format_func=lambda n: SAMPLE_LABELS.get(n, n))
            raw, name = (SAMPLES_DIR / choice).read_bytes(), choice
        else:
            st.info("No samples found. Run `python scripts/make_samples.py`.")

    st.header("Model")
    model = st.text_input("Pydantic AI model", value=settings.model, help="provider:model, e.g. openai:gpt-4.1")
    key_missing = missing_api_key(model)
    if key_missing:
        st.error(f"Set **{key_missing}** in a `.env` file next to app.py (see `.env.example`), then restart.")
    else:
        st.caption(f"Ready: `{model}`")

    if st.button("Clear conversation", icon=":material/delete:", width="stretch"):
        st.session_state.turns = []
        st.rerun()

# --------------------------------------------------------------------------- dataset
st.title("CSV Data Explorer")
if raw is None:
    st.info("Upload a CSV or choose a sample dataset in the sidebar to begin.")
    st.stop()

try:
    dataset = load_dataset(raw, name)
except CSVImportError as exc:
    st.error(f"Could not read **{name}**: {exc}")
    st.stop()

dataset_key = hashlib.sha1(raw).hexdigest() + name
if st.session_state.get("dataset_key") != dataset_key:
    st.session_state.dataset_key = dataset_key
    st.session_state.turns = []
    st.session_state.ctx = DataContext.from_dataset(dataset, settings)
st.session_state.setdefault("turns", [])
ctx: DataContext = st.session_state.ctx
schema = dataset.schema

st.caption(
    f"**{schema.name}** - {schema.rows:,} rows x {len(schema.columns)} columns "
    f"- {len(schema.names('measure'))} measures, {len(schema.names('dimension'))} dimensions, "
    f"{len(schema.names('time'))} time columns"
)


# --------------------------------------------------------------------------- rendering helpers
def render_turn(turn: Turn, index: int) -> None:
    st.markdown(turn.answer.answer)
    for chart in turn.charts:
        st.plotly_chart(chart.figure, theme=None, key=f"chart-{index}-{chart.id}", config={"displaylogo": False})
        with st.expander(f"Data behind \"{chart.spec.title}\""):
            st.dataframe(chart.data.head(5000), hide_index=True)
    for table in turn.tables:
        st.markdown(f"**{table.title}**")
        st.dataframe(table.data, hide_index=True)
        st.download_button(
            "Download CSV",
            table.data.to_csv(index=False).encode(),
            file_name=f"{table.title[:40].strip() or 'result'}.csv",
            key=f"dl-{index}-{table.id}",
            icon=":material/download:",
        )
    with st.expander(f"How this was computed ({len(turn.steps)} tool calls, {turn.requests} model requests)"):
        if not turn.steps:
            st.caption("Answered without tools.")
        for step in turn.steps:
            mark = ":material/check_circle:" if step.ok else ":material/replay: retried -"
            st.markdown(f"{mark} `{step.tool}` - {step.summary}")
            st.code(json.dumps(step.args, indent=2, default=str), language="json")
    if turn.answer.follow_ups and index == len(st.session_state.turns) - 1:
        cols = st.columns(len(turn.answer.follow_ups[:3]))
        for col, q in zip(cols, turn.answer.follow_ups[:3]):
            if col.button(q, key=f"follow-{index}-{q}", icon=":material/arrow_forward:"):
                st.session_state.pending = q
                st.rerun()


def run_question(question: str) -> None:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analysing..."):
                turn = ask(get_agent(model), ctx, question, previous=st.session_state.turns)
        except Exception as exc:  # noqa: BLE001 - show any model/provider error in the UI
            st.error(f"The agent could not answer: {type(exc).__name__}: {exc}")
            return
        st.session_state.turns.append(turn)
        render_turn(turn, len(st.session_state.turns) - 1)


# --------------------------------------------------------------------------- tabs
ask_tab, schema_tab, data_tab = st.tabs(["Ask", "Schema", "Data"])

with schema_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{schema.rows:,}")
    c2.metric("Columns", len(schema.columns))
    c3.metric("Delimiter", {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe"}.get(schema.delimiter, schema.delimiter))
    c4.metric("Duplicate rows", f"{schema.duplicate_rows:,}")
    table = pd.DataFrame(
        [
            {
                "Column": c.name,
                "Type": c.semantic_type,
                "Role": ROLE_LABELS[c.role],
                "Missing %": c.missing_pct,
                "Distinct": c.unique,
                "Range / top values": f"{c.min} .. {c.max}" if c.min is not None else ", ".join(v.value for v in c.top_values[:4]),
                "Examples": ", ".join(c.sample_values[:3]),
                "Notes": " ".join(c.notes),
            }
            for c in schema.columns
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        column_config={"Missing %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)},
    )
    for w in schema.warnings:
        st.warning(w, icon=":material/info:")

with data_tab:
    st.dataframe(dataset.df.head(2000), hide_index=True)
    if len(dataset.df) > 2000:
        st.caption(f"Showing the first 2,000 of {len(dataset.df):,} rows.")
    numeric = dataset.df.select_dtypes("number")
    if not numeric.empty:
        st.subheader("Numeric summary")
        st.dataframe(numeric.describe().T.round(3))

with ask_tab:
    for i, turn in enumerate(st.session_state.turns):
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            render_turn(turn, i)

    if not st.session_state.turns:
        st.markdown("Ask anything about the data, or start with one of these:")
        cols = st.columns(2)
        for i, q in enumerate(starter_questions(schema)):
            if cols[i % 2].button(q, key=f"starter-{i}", width="stretch"):
                st.session_state.pending = q
                st.rerun()

    question = st.chat_input("Ask about your data, e.g. \"Show me a bar chart of sales by region\"", disabled=bool(key_missing))
    pending = st.session_state.pop("pending", None)
    if pending and key_missing:
        st.error(f"Add {key_missing} to .env to ask questions.")
    elif question or pending:
        run_question(question or pending)
