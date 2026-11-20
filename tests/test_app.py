"""Smoke test of the Streamlit page (no model calls)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_loads_sample_and_shows_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "CSV Data Explorer"
    assert any("rows x" in c.value for c in at.caption)
    labels = [b.label for b in at.button]
    assert any(label.startswith("Summarise") for label in labels)
    assert len(at.dataframe) >= 2  # schema table + data preview


def test_app_blocks_questions_without_api_key(monkeypatch):
    monkeypatch.setenv("CSVX_MODEL", "openai:gpt-4.1-mini")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("OPENAI_API_KEY" in e.value for e in at.error)
    assert at.chat_input[0].disabled
