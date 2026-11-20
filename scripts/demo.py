"""End-to-end demonstration on both sample datasets.

Runs a fixed list of questions through the real agent and writes
docs/demo_output/report.html: every question, the agent's answer, its charts
(interactive) and the tool calls it made. Needs an API key (see .env.example).

Usage:  python scripts/demo.py
"""

from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import markdown_it  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from csvagent import DataContext, ask, build_agent, load_csv, load_settings  # noqa: E402
from csvagent.config import PROJECT_ROOT, SAMPLES_DIR, missing_api_key  # noqa: E402

DEMO = {
    "retail_sales.csv": [
        "Show me a bar chart of sales by region",
        "Summarise the key trends",
        "How have monthly sales changed over time for each category?",
        "Which sub-categories lose money, and is discount the reason?",
        "Top 5 states by profit in 2025",
    ],
    "census_income.csv": [
        "What share of people earn more than 50K, by education level?",
        "Show the age distribution of the dataset",
        "Do men and women work different hours per week?",
        "Summarise the key trends",
    ],
}


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = load_settings()
    if var := missing_api_key(settings.model):
        print(f"Set {var} in .env first (see .env.example).")
        return 1
    agent = build_agent(settings=settings)
    md = markdown_it.MarkdownIt()
    sections, total_tokens, first_chart = [], 0, True
    for file, questions in DEMO.items():
        dataset = load_csv(SAMPLES_DIR / file, settings=settings)
        ctx = DataContext.from_dataset(dataset, settings)
        sections.append(f"<h2>{html.escape(file)}</h2><pre class='schema'>{html.escape(dataset.schema.to_prompt())}</pre>")
        turns = []
        for q in questions:
            start = time.perf_counter()
            print(f"[{file}] {q}")
            turn = ask(agent, ctx, q, previous=turns)
            turns.append(turn)
            total_tokens += turn.tokens
            charts = ""
            for c in turn.charts:  # the first chart loads the matching plotly.js from its CDN
                charts += c.figure.to_html(full_html=False, include_plotlyjs="cdn" if first_chart else False)
                first_chart = False
            steps = "".join(
                f"<li><code>{s.tool}</code> {'' if s.ok else '(retried) '}{html.escape(s.summary)}"
                f"<pre>{html.escape(json.dumps(s.args, indent=1, default=str))}</pre></li>"
                for s in turn.steps
            )
            sections.append(
                f"<section><h3>Q: {html.escape(q)}</h3>{md.render(turn.answer.answer)}{charts}"
                f"<details><summary>{len(turn.steps)} tool calls, {turn.requests} model requests, "
                f"{turn.tokens:,} tokens, {time.perf_counter() - start:.1f}s</summary><ul>{steps}</ul></details></section>"
            )
    out = PROJECT_ROOT / "docs" / "demo_output" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>CSV Data Explorer demo</title>"
        "<style>body{font-family:Segoe UI,system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 16px;"
        "background:#fcfcfb;color:#0b0b0b}section{border-top:1px solid #e6e5e0;padding:1rem 0}"
        "pre{background:#f0efec;padding:.6rem;overflow-x:auto;font-size:12px}.schema{max-height:260px}</style></head><body>"
        f"<h1>CSV Data Explorer - demo run</h1><p>Model: <code>{html.escape(settings.model)}</code> - "
        f"{total_tokens:,} tokens in total.</p>{''.join(sections)}</body></html>",
        encoding="utf-8",
    )
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
