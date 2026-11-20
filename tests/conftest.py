import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from csvagent import load_csv, load_settings  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def sales(settings):
    return load_csv(ROOT / "samples" / "retail_sales.csv", settings=settings)


@pytest.fixture(scope="session")
def census(settings):
    return load_csv(ROOT / "samples" / "census_income.csv", settings=settings)


@pytest.fixture()
def small_df():
    return pd.DataFrame(
        {
            "Region": ["West", "West", "East", "East", "South", None],
            "Product": ["Chair", "Desk", "Chair", "Lamp", "Desk", "Lamp"],
            "Date": pd.to_datetime(["2024-01-05", "2024-02-10", "2024-02-11", "2024-03-01", "2024-03-15", "2024-03-20"]),
            "Sales": [100.0, 250.0, 80.0, 40.0, 300.0, 10.0],
            "Units": [1, 2, 1, 4, 3, 1],
        }
    )
