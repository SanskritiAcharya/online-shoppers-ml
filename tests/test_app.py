"""Smoke test for the Streamlit app: it runs, has three tabs, and the presets change the prediction."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "ui.py")


def test_app_runs_with_three_tabs():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Score a session", "Test sessions", "How the model works"]


def test_bounce_preset_gets_no_popup():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.session_state["preset"] = "Bounce"
    for key, value in {"ProductRelated": 2, "product_min": 0.5, "exit_pct": 15.0, "bounce_pct": 12.0,
                       "Administrative": 0, "admin_min": 0.0, "Month": "Mar"}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    page = " ".join(str(h.proto) for h in at.get("html"))
    assert "No pop-up" in page
