"""Load the saved pipeline once and turn a session into a purchase probability and a pop-up decision.

    python -m src.predict      (prints a demo prediction)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from src import config
from src.features import to_frame

_model = None
_meta = None


def load():
    """Load models/model.joblib and model.json once, then reuse them."""
    global _model, _meta
    if _model is None:
        _model = joblib.load(config.MODEL_PATH)
        _meta = json.loads(config.META_PATH.read_text())
    return _model, _meta


def recommend(p_buy, threshold):
    """The marketing team's two actions, from the charter."""
    if p_buy >= threshold:
        return True, "Show the discount pop-up. This visitor looks close to buying and is about to leave."
    return False, "Let the visitor leave without a pop-up. Save the discount for sessions closer to buying."


def predict(records):
    """records: one dict or a list of dicts with the fields in config.RAW_INPUT_FEATURES."""
    model, meta = load()
    probs = model.predict_proba(to_frame(records))[:, 1]
    out = []
    for p in probs:
        show, action = recommend(float(p), meta["threshold"])
        out.append({
            "purchase_probability": round(float(p), 4),
            "show_popup": show,
            "recommended_action": action,
            "threshold": meta["threshold"],
            "model_version": meta["model_version"],
        })
    return out


def drivers(record):
    """What moved this prediction: reset one input at a time to the typical session and see how far
    the probability moves. Easy to explain, and it works the same way for any model."""
    model, meta = load()
    typical = meta["typical_session"]
    base = float(model.predict_proba(to_frame(record))[:, 1][0])
    swapped = [{**record, col: typical[col]} for col in config.RAW_INPUT_FEATURES]
    swapped_p = model.predict_proba(to_frame(swapped))[:, 1]
    # positive = this input pushes the probability up compared with a typical session
    return {col: base - float(p) for col, p in zip(config.RAW_INPUT_FEATURES, swapped_p)}


# A returning visitor in November who has looked at many product pages and rarely hits exit pages.
EXAMPLE = {
    "Administrative": 4, "Administrative_Duration": 120.0,
    "Informational": 1, "Informational_Duration": 30.0,
    "ProductRelated": 45, "ProductRelated_Duration": 1800.0,
    "BounceRates": 0.005, "ExitRates": 0.015, "SpecialDay": 0.0,
    "Month": "Nov", "VisitorType": "Returning_Visitor",
    "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2, "Weekend": False,
}

if __name__ == "__main__":
    print(predict(EXAMPLE))
    print(sorted(drivers(EXAMPLE).items(), key=lambda kv: -abs(kv[1]))[:5])
