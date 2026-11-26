import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from src import config, data, evaluate, features, predict
from src.predict import EXAMPLE

client = TestClient(app)


# ---------- data and features ----------


def test_raw_file_is_the_frozen_copy():
    assert data.file_md5(config.RAW_FILE) == config.RAW_MD5


def test_cleaning_drops_duplicates_and_fixes_june():
    df = data.clean(data.load_raw())
    assert len(df) == 12205
    assert not df.duplicated().any()
    assert "June" not in set(df["Month"]) and "Jun" in set(df["Month"])
    assert set(df[config.TARGET]) == {0, 1}


def test_split_is_stratified_and_disjoint():
    df = data.load_clean()
    train, test = data.split(df)
    assert len(train) + len(test) == len(df)
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.005
    # no session appears on both sides (duplicates were removed before the split)
    assert len(train.merge(test, how="inner")) == 0


def test_engineered_features():
    row = {**EXAMPLE, "Administrative": 2, "Informational": 0, "ProductRelated": 8,
           "Administrative_Duration": 10, "Informational_Duration": 0, "ProductRelated_Duration": 400}
    out = features.add_engineered_features(pd.DataFrame([row]))
    assert out.loc[0, "total_pages"] == 10
    assert out.loc[0, "total_duration"] == 410
    assert out.loc[0, "product_page_share"] == pytest.approx(0.8)
    assert out.loc[0, "seconds_per_product_page"] == pytest.approx(50)


def test_engineered_features_with_zero_pages_do_not_divide_by_zero():
    row = {**EXAMPLE, "Administrative": 0, "Informational": 0, "ProductRelated": 0}
    out = features.add_engineered_features(pd.DataFrame([row]))
    assert out.loc[0, "product_page_share"] == 0
    assert out.loc[0, "seconds_per_product_page"] == 0


def test_pipeline_never_sees_pagevalues():
    model, meta = predict.load()
    assert config.LEAKY_SUSPECT not in meta["input_features"]
    assert config.LEAKY_SUSPECT not in model.feature_names_in_


def test_to_frame_accepts_june_and_bool_weekend():
    df = features.to_frame({**EXAMPLE, "Month": "June", "Weekend": True})
    assert df.loc[0, "Month"] == "Jun" and df.loc[0, "Weekend"] == 1


# ---------- evaluation helpers ----------


def test_top_share_flags_exactly_ten_percent():
    flags = evaluate.top_share_flags(np.arange(200, dtype=float))
    assert flags.sum() == 20
    assert flags[-20:].all()  # the 20 highest scores


def test_constant_score_gets_base_rate_precision_not_row_order_luck():
    y = np.array([1] * 100 + [0] * 900)  # all buyers at the top of the file
    flags = evaluate.top_share_flags(np.zeros(1000))
    precision = evaluate.precision_recall_of_flags(y, flags)["precision"]
    assert precision < 0.3  # row order would have given 1.0


def test_psi_is_zero_for_same_data_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=5000))
    assert evaluate.psi(a, a) < 0.01
    assert evaluate.psi(a, a + 1.5) > 0.25


# ---------- saved model ----------


def test_round_trip_matches_metadata():
    model, meta = predict.load()
    df = data.load_clean()
    _, test = data.split(df)
    scores = model.predict_proba(test[config.RAW_INPUT_FEATURES])[:, 1]
    m = evaluate.all_metrics(test[config.TARGET], scores, meta["threshold"])
    assert m["pr_auc"] == pytest.approx(meta["test_metrics"]["pr_auc"], abs=1e-4)


# ---------- API ----------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid():
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["purchase_probability"] <= 1
    assert body["show_popup"] == (body["purchase_probability"] >= body["threshold"])


def test_predict_unseen_month_still_works():
    response = client.post("/predict", json={**EXAMPLE, "Month": "Jan"})
    assert response.status_code == 200


@pytest.mark.parametrize("bad", [
    {"ProductRelated": -3},
    {"ExitRates": 1.7},
    {"Month": "Smarch"},
    {"VisitorType": "Robot"},
    {"Region": 0},
    {"PageValues": 12.5},  # the excluded column is refused, not silently used
])
def test_predict_rejects_bad_input(bad):
    assert client.post("/predict", json={**EXAMPLE, **bad}).status_code == 422


def test_predict_rejects_missing_field():
    body = {k: v for k, v in EXAMPLE.items() if k != "ExitRates"}
    assert client.post("/predict", json=body).status_code == 422
