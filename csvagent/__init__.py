"""CSV Data Explorer: a Pydantic AI agent that turns questions about a CSV into tables, charts and summaries."""

import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # keep the console clean

# ruff: noqa: E402

from .agent import DataContext, Turn, ask, build_agent
from .config import Settings, load_settings
from .ingest import CSVImportError, LoadedDataset, load_csv

__all__ = [
    "CSVImportError", "DataContext", "LoadedDataset", "Settings", "Turn", "ask", "build_agent", "load_csv",
    "load_settings",
]
