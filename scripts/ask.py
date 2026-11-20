"""Ask questions about a CSV from the terminal.

Usage:
    python scripts/ask.py samples/retail_sales.csv "Show me a bar chart of sales by region"
    python scripts/ask.py samples/census_income.csv            # interactive: type questions, empty line quits
    python scripts/ask.py my.csv --schema                      # only print the detected schema

Charts are saved as interactive HTML files in docs/demo_output/.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from csvagent import DataContext, ask, build_agent, load_csv, load_settings  # noqa: E402
from csvagent.config import PROJECT_ROOT, missing_api_key  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "demo_output"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "chart"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--schema", action="store_true", help="print the detected schema and exit")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    settings = load_settings()
    dataset = load_csv(args.csv, settings=settings)
    print(dataset.schema.to_prompt(), "\n")
    if args.schema:
        return 0
    if var := missing_api_key(settings.model):
        print(f"Set {var} in .env first (see .env.example).")
        return 1

    agent = build_agent(settings=settings)
    ctx = DataContext.from_dataset(dataset, settings)
    turns = []
    questions = [args.question] if args.question else iter(lambda: input("Question> ").strip(), "")
    OUT.mkdir(parents=True, exist_ok=True)
    for q in questions:
        turn = ask(agent, ctx, q, previous=turns)
        turns.append(turn)
        print("\n" + turn.answer.answer)
        for chart in turn.charts:
            path = OUT / f"{slug(chart.spec.title)}.html"
            chart.figure.write_html(path, include_plotlyjs="cdn")
            print(f"  [chart] {chart.spec.title} -> {path}")
        for step in turn.steps:
            print(f"  [{'ok' if step.ok else 'retry'}] {step.tool}: {step.summary}")
        if turn.answer.follow_ups:
            print("  Try next: " + " | ".join(turn.answer.follow_ups))
        print(f"  ({turn.requests} model requests, {turn.tokens:,} tokens)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
