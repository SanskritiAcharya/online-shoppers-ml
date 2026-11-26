"""Feature pipeline. Training, the notebooks, the API and the Streamlit app all use this code,
because the whole pipeline is saved inside models/model.joblib."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def add_engineered_features(df):
    """Columns built from the session's own counters, so they exist whenever the counters do."""
    df = df.copy()
    df["total_pages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["total_duration"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                            + df["ProductRelated_Duration"])
    # A session with zero pages gets 0 instead of a divide-by-zero.
    pages = df["total_pages"].replace(0, np.nan)
    df["product_page_share"] = (df["ProductRelated"] / pages).fillna(0.0)
    product_pages = df["ProductRelated"].replace(0, np.nan)
    df["seconds_per_product_page"] = (df["ProductRelated_Duration"] / product_pages).fillna(0.0)
    return df


def log1p_clipped(x):
    # The data has no negative values, but log1p of a negative number breaks, so clip to 0 as a guard.
    return np.log1p(np.clip(x, 0, None))


def build_preprocessor(use_page_values=False):
    # Counts and durations have long right tails (one session spent 17 hours on product pages),
    # so they are log-transformed. That matters for logistic regression, and trees do not mind.
    skewed = config.COUNT_FEATURES + config.DURATION_FEATURES + ["total_pages", "total_duration",
                                                                  "seconds_per_product_page"]
    if use_page_values:
        skewed = skewed + [config.LEAKY_SUSPECT]
    skewed_steps = Pipeline([
        ("log", FunctionTransformer(log1p_clipped, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    # Rare codes (for example browser 9, one session) are grouped into one "infrequent" column,
    # and a code never seen in training is ignored instead of crashing the API.
    one_hot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False)
    plain_numeric = config.RATE_FEATURES + ["product_page_share"] + config.FLAG_FEATURES
    return ColumnTransformer([
        ("skewed", skewed_steps, skewed),
        ("numeric", StandardScaler(), plain_numeric),
        ("categorical", one_hot, config.CATEGORICAL_FEATURES),
    ])


def engineered_names(transformer, input_names):
    # Lets sklearn list the column names after the engineer step (used for coefficient tables).
    return list(input_names) + config.ENGINEERED_FEATURES


def make_pipeline(model, use_page_values=False):
    return Pipeline([
        ("engineer", FunctionTransformer(add_engineered_features, feature_names_out=engineered_names)),
        ("prep", build_preprocessor(use_page_values)),
        ("model", model),
    ])


def make_simple_pipeline(model):
    """Baseline 'simple model': four obvious columns, log + scale, nothing else."""
    prep = ColumnTransformer([
        ("simple", Pipeline([("log", FunctionTransformer(log1p_clipped)), ("scale", StandardScaler())]),
         config.SIMPLE_FEATURES),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def to_frame(records, columns=None):
    """Turn API/app input (one dict or a list of dicts) into the DataFrame the pipeline expects."""
    columns = columns or config.RAW_INPUT_FEATURES
    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    df["Month"] = df["Month"].replace({"June": "Jun"})
    df["Weekend"] = df["Weekend"].astype(int)
    return df[columns]
