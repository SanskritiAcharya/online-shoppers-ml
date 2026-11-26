"""Metrics, the baseline rules, slice analysis and drift (PSI).

train.py calls these, and the notebooks use them too.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import config

# ---------- baseline rules ----------


def score_constant(df):
    """'Nobody buys'. Every session gets the same score, so it cannot rank anyone."""
    return np.zeros(len(df))


def score_rule_a(df):
    """Rule A: the session has a page value above zero. Uses the suspect column."""
    return (df["PageValues"] > 0).astype(float).values


def score_rule_b(df):
    """Rule B, no page value: many product pages and a low exit rate (thresholds from the charter)."""
    many_products = df["ProductRelated"] >= config.RULE_B_MIN_PRODUCT_PAGES
    low_exit = df["ExitRates"] < config.RULE_B_MAX_EXIT_RATE
    return (many_products & low_exit).astype(float).values


# ---------- metrics ----------


def top_share_flags(scores, share=config.POPUP_CAPACITY):
    """Flag exactly `share` of the sessions, highest scores first.

    A rule gives many sessions the same score. Breaking those ties by row order could quietly
    help or hurt, so ties are broken by a fixed random order instead. For the constant baseline
    that means a random 10%, which is what 'no model' would really give the marketing team."""
    scores = np.asarray(scores, dtype=float)
    n_flag = int(round(share * len(scores)))
    tie_breaker = np.random.default_rng(config.SEED).random(len(scores))
    order = np.lexsort((tie_breaker, -scores))  # sort by score (high first), then by the tie breaker
    flags = np.zeros(len(scores), dtype=bool)
    flags[order[:n_flag]] = True
    return flags


def precision_recall_of_flags(y_true, flags):
    y_true = np.asarray(y_true)
    flags = np.asarray(flags, dtype=bool)
    hits = int((flags & (y_true == 1)).sum())
    return {
        "precision": float(hits / flags.sum()) if flags.sum() else 0.0,
        "recall": float(hits / y_true.sum()) if y_true.sum() else 0.0,
    }


def all_metrics(y_true, scores, threshold=None):
    """Every number we report for one set of scores. Never accuracy as the headline:
    84% of sessions do not buy, so 'nobody buys' is already 84% accurate."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    one_value = len(np.unique(scores)) == 1
    at_capacity = precision_recall_of_flags(y_true, top_share_flags(scores))
    out = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": 0.5 if one_value else float(roc_auc_score(y_true, scores)),
        "precision_at_10pct": at_capacity["precision"],
        "recall_at_10pct": at_capacity["recall"],
        # Only logged to show why it is useless here. "Nobody buys" scores about 0.84 on it.
        "accuracy_at_0_5": float(((scores >= 0.5).astype(int) == y_true).mean()),
        "prevalence": float(y_true.mean()),
    }
    if threshold is not None:
        # The live system cannot rank a whole day of sessions in advance. It compares each
        # session with a fixed threshold, so we also check how many that would really flag.
        flags = scores >= threshold
        at_threshold = precision_recall_of_flags(y_true, flags)
        out["flag_rate_at_threshold"] = float(flags.mean())
        out["precision_at_threshold"] = at_threshold["precision"]
        out["recall_at_threshold"] = at_threshold["recall"]
        out["brier"] = float(brier_score_loss(y_true, np.clip(scores, 0, 1)))
    return out


def capacity_threshold(scores, share=config.POPUP_CAPACITY):
    """The score above which `share` of sessions sit. Picked on out-of-fold training scores,
    never on the test set."""
    return float(np.quantile(np.asarray(scores), 1 - share))


# ---------- slices ----------


def traffic_group(code):
    # The top six traffic sources cover about 88% of sessions. The rest are pooled so each slice
    # has enough rows to say something.
    return f"Source {code}" if code in (1, 2, 3, 4, 13, 10) else "Other sources"


def slice_table(df, scores, threshold):
    """Precision, recall and PR-AUC for each group, at the deployed threshold."""
    tmp = df.copy()
    tmp["score"] = np.asarray(scores, dtype=float)
    tmp["flag"] = tmp["score"] >= threshold
    tmp["Weekend_label"] = tmp["Weekend"].map({0: "Weekday", 1: "Weekend"})
    tmp["Traffic_label"] = tmp["TrafficType"].map(traffic_group)
    tmp["Season"] = np.where(tmp["Month"].isin(config.HOLIDAY_MONTHS), "Nov-Dec", "Feb-Oct")
    groups = {"VisitorType": "VisitorType", "Month": "Month", "Weekend": "Weekend_label",
              "Traffic source": "Traffic_label", "Season": "Season"}
    rows = []
    for slice_name, col in groups.items():
        for value, part in tmp.groupby(col):
            y = part[config.TARGET].values
            both_classes = 0 < y.sum() < len(y)
            pr = precision_recall_of_flags(y, part["flag"].values)
            rows.append({
                "slice": slice_name, "group": value, "n": len(part), "buy_rate": y.mean(),
                "flag_rate": part["flag"].mean(),
                "precision": pr["precision"] if part["flag"].sum() else np.nan,
                "recall": pr["recall"] if y.sum() else np.nan,
                "pr_auc": average_precision_score(y, part["score"]) if both_classes else np.nan,
            })
    return pd.DataFrame(rows)


# ---------- drift ----------


def psi(reference, current, bins=10):
    """Population Stability Index. Under 0.1 stable, 0.1 to 0.25 watch, over 0.25 act.

    Numbers are cut into bins at the reference deciles. Text columns use their categories."""
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()
    if not pd.api.types.is_numeric_dtype(reference):
        cats = sorted(set(reference) | set(current))
        ref_share = reference.value_counts(normalize=True).reindex(cats, fill_value=0).values
        cur_share = current.value_counts(normalize=True).reindex(cats, fill_value=0).values
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            # Mostly one value (SpecialDay is 0 for 90% of sessions), so compare as categories.
            return psi(reference.astype(str), current.astype(str))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_share = np.histogram(reference, edges)[0] / len(reference)
        cur_share = np.histogram(current, edges)[0] / len(current)
    eps = 1e-4  # an empty bin would make the log blow up
    ref_share = np.clip(ref_share, eps, None)
    cur_share = np.clip(cur_share, eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_table(reference_df, current_df, columns):
    rows = []
    for c in columns:
        ref, cur = reference_df[c], current_df[c]
        if c in config.CATEGORICAL_FEATURES:
            ref, cur = ref.astype(str), cur.astype(str)  # codes are labels, so compare them as categories
        rows.append({"feature": c, "psi": psi(ref, cur)})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "watch", "act"]).astype(str)
    return out
