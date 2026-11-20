"""Starter questions built from the detected schema (shown before the first question)."""

from __future__ import annotations

from .insights import pick_dimensions, pick_measure, pick_time
from .models import DatasetSchema


def starter_questions(schema: DatasetSchema, limit: int = 4) -> list[str]:
    measure = pick_measure(schema)
    time = pick_time(schema)
    dims = pick_dimensions(schema, n=2)
    out: list[str] = ["Summarise the key trends in this dataset"]
    if measure and dims:
        out.append(f"Show me a bar chart of {measure} by {dims[0]}")
    if measure and time:
        out.append(f"How has {measure} changed over time?")
    if measure and len(dims) > 1:
        out.append(f"Compare {measure} across {dims[0]} and {dims[1]}")
    elif dims:
        out.append(f"What share of rows falls in each {dims[0]}?")
    if measure:
        out.append(f"What does the distribution of {measure} look like?")
    measures = schema.names("measure")
    if len(measures) >= 2:
        out.append(f"Is {measures[0]} related to {measures[1]}?")
    return out[:limit]
