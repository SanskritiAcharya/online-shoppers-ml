"""Paths, seeds, feature lists and the operating point. Every other file imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "online_shoppers_intention.csv"
RAW_MD5 = "cc6ec1db03b4f10f8de52c56ff48b085"  # frozen copy, downloaded 2026-11-26
CLEAN_FILE = PROCESSED_DIR / "sessions_clean.csv"

MODEL_PATH = MODELS_DIR / "model.joblib"
META_PATH = MODELS_DIR / "model.json"

MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "online-shoppers-popup"
REGISTERED_MODEL_NAME = "shoppers-popup-purchase"

SEED = 42
TEST_SHARE = 0.20
CV_FOLDS = 5

# The marketing manager's limit: pop-ups go to at most this share of sessions (from the charter).
POPUP_CAPACITY = 0.10

TARGET = "Revenue"

# The suspect column. Only the "with_pagevalues" runs use it.
LEAKY_SUSPECT = "PageValues"

COUNT_FEATURES = ["Administrative", "Informational", "ProductRelated"]
DURATION_FEATURES = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
RATE_FEATURES = ["BounceRates", "ExitRates", "SpecialDay"]
# Integer codes in the raw file, but they are labels, so they get one-hot encoded.
CODE_FEATURES = ["OperatingSystems", "Browser", "Region", "TrafficType"]
CATEGORICAL_FEATURES = ["Month", "VisitorType"] + CODE_FEATURES
FLAG_FEATURES = ["Weekend"]

# Built inside the pipeline by features.add_engineered_features
ENGINEERED_FEATURES = ["total_pages", "total_duration", "product_page_share", "seconds_per_product_page"]

# What the API and the app ask for (the deployed model, so no PageValues).
RAW_INPUT_FEATURES = COUNT_FEATURES + DURATION_FEATURES + RATE_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES
RAW_INPUT_WITH_PV = RAW_INPUT_FEATURES + [LEAKY_SUSPECT]

# Simple baseline model: four obvious columns, nothing else.
SIMPLE_FEATURES = ["ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates"]

# Rule B from the charter (no PageValues). Written down before modelling.
RULE_B_MIN_PRODUCT_PAGES = 20
RULE_B_MAX_EXIT_RATE = 0.025

MONTHS = ["Feb", "Mar", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]
HOLIDAY_MONTHS = ["Nov", "Dec"]
