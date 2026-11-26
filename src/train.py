"""Train everything end to end and log it to MLflow.

    python src/train.py        (or: python -m src.train)

Steps:
 1. load the cleaned sessions and make the stratified 80/20 split
 2. log the baselines (constant, Rule A with page value, Rule B without, simple logistic regression)
 3. tune three model families with 5-fold CV, once WITHOUT PageValues and once WITH it
 4. log the PageValues leakage check
 5. pick the deployable family by CV PR-AUC (no PageValues), set the 10% threshold on out-of-fold scores
 6. touch the test set once: final model, plus the with-PageValues model as a comparison
 7. slices, permutation importance, figures
 8. save models/model.joblib + model.json, check the round trip, register the model
 9. month check (train Feb-Oct, test Nov-Dec) and drift check (PSI)
"""
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `python src/train.py` also works
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from src import config, data, evaluate, features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ACCENT, GREY, LIGHT = "#1d5f8a", "#8a939f", "#c9ced6"


# ---------- small helpers ----------


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=config.ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def lineage_tags():
    return {
        "data_md5": data.file_md5(config.RAW_FILE),
        "git_commit": git_commit(),
        "split": f"stratified random, test_share={config.TEST_SHARE}, seed={config.SEED}, duplicates dropped first",
        "cv": f"StratifiedKFold({config.CV_FOLDS}, shuffle, seed={config.SEED}) on the training rows",
    }


def fold_metrics(y, scores, folds):
    """PR-AUC, ROC-AUC and precision at 10% on each validation fold, so we can show the spread."""
    y = np.asarray(y)
    out = {"pr_auc": [], "roc_auc": [], "precision_at_10pct": []}
    for _, val_idx in folds:
        m = evaluate.all_metrics(y[val_idx], scores[val_idx])
        for k in out:
            out[k].append(m[k])
    return {k: np.array(v) for k, v in out.items()}


def log_cv(cv):
    for name, values in cv.items():
        mlflow.log_metric(f"cv_{name}_mean", values.mean())
        mlflow.log_metric(f"cv_{name}_std", values.std())
    for i, value in enumerate(cv["pr_auc"]):
        mlflow.log_metric("cv_pr_auc_fold", value, step=i)


def log_test(metrics):
    mlflow.log_metrics({"test_" + k: v for k, v in metrics.items()})


def result_row(run, kind, feature_set, cv=None, test=None):
    row = {"run": run, "kind": kind, "feature_set": feature_set}
    if cv is not None:
        row.update({"cv_pr_auc_mean": cv["pr_auc"].mean(), "cv_pr_auc_std": cv["pr_auc"].std(),
                    "cv_roc_auc_mean": cv["roc_auc"].mean(),
                    "cv_precision_at_10pct_mean": cv["precision_at_10pct"].mean()})
    if test is not None:
        row.update({"test_" + k: v for k, v in test.items()})
    return row


def typical_session(train_df, columns):
    typical = {}
    for c in columns:
        if c in config.CATEGORICAL_FEATURES + config.FLAG_FEATURES:
            value = train_df[c].mode()[0]
            typical[c] = value.item() if hasattr(value, "item") else value  # numpy int to plain int, for JSON
        else:
            typical[c] = float(train_df[c].median())
    return typical


def save_fig(fig, name, log=True):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    if log:
        mlflow.log_artifact(str(path))
    return path


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(colors="#4a5361")


# ---------- the model families and their grids ----------

MODEL_GRIDS = {
    "logistic_regression": (
        LogisticRegression(max_iter=5000),
        {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier(random_state=config.SEED),
        {
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [3, 6],
            "model__max_iter": [150, 400],
            "model__l2_regularization": [0.0, 1.0],
        },
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=-1),
        {
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [5, 20],
            "model__max_features": ["sqrt", 0.3],
        },
    ),
}

FEATURE_SETS = {
    "no_pagevalues": (config.RAW_INPUT_FEATURES, False),
    "with_pagevalues": (config.RAW_INPUT_WITH_PV, True),
}


def tune_family(family, feature_set, train_df, folds, tags):
    """GridSearchCV inside the training rows only. Returns the refit best pipeline and its OOF scores."""
    estimator, grid = MODEL_GRIDS[family]
    columns, use_pv = FEATURE_SETS[feature_set]
    X, y = train_df[columns], train_df[config.TARGET]
    with mlflow.start_run(run_name=f"tuned_{family}_{feature_set}") as run:
        mlflow.set_tags({**tags, "kind": "candidate_model", "feature_set": feature_set})
        search = GridSearchCV(features.make_pipeline(clone(estimator), use_pv), grid, scoring="average_precision",
                              cv=folds, n_jobs=-1, refit=True)
        search.fit(X, y)
        best = search.best_estimator_
        oof = cross_val_predict(best, X, y, cv=folds, method="predict_proba")[:, 1]
        cv = fold_metrics(y, oof, folds)
        mlflow.log_params({"model": family, "feature_set": feature_set, "features": ",".join(columns),
                           "seed": config.SEED, "grid_size": len(search.cv_results_["params"])})
        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        log_cv(cv)
        grid_table = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        grid_path = config.REPORTS_DIR / f"grid_{family}_{feature_set}.csv"
        grid_table.sort_values("rank_test_score").to_csv(grid_path, index=False)
        mlflow.log_artifact(str(grid_path))
        print(f"  {family:24s} {feature_set:16s} CV PR-AUC {cv['pr_auc'].mean():.3f} ± {cv['pr_auc'].std():.3f}"
              f"  best {search.best_params_}")
    return {"model": best, "oof": oof, "cv": cv, "run_id": run.info.run_id, "params": search.best_params_}


def main():
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.EXPERIMENT_NAME, artifact_location=(config.ROOT / "mlruns").as_uri())
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    tags = lineage_tags()
    print("lineage:", tags)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    # 1. data and split
    df = data.load_clean()
    train_df, test_df = data.split(df)
    y_train, y_test = train_df[config.TARGET].values, test_df[config.TARGET].values
    cv_splitter = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    folds = list(cv_splitter.split(train_df, y_train))
    print(f"train {len(train_df)} rows (buy rate {y_train.mean():.3f}), test {len(test_df)} rows (buy rate {y_test.mean():.3f})")

    results = []  # one row per run, saved as reports/results_table.csv

    # 2. baselines. The rules have nothing to fit, so they are scored on the same folds and test rows.
    rules = {
        "baseline_constant_nobody_buys": (evaluate.score_constant, "none"),
        "baseline_rule_A_pagevalue_above_0": (evaluate.score_rule_a, "with_pagevalues"),
        "baseline_rule_B_products_and_low_exit": (evaluate.score_rule_b, "no_pagevalues"),
    }
    for name, (rule, feature_set) in rules.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({**tags, "kind": "baseline", "feature_set": feature_set})
            mlflow.log_params({"model": name, "capacity": config.POPUP_CAPACITY})
            if "rule_B" in name:
                mlflow.log_params({"min_product_pages": config.RULE_B_MIN_PRODUCT_PAGES,
                                   "max_exit_rate": config.RULE_B_MAX_EXIT_RATE})
            cv = fold_metrics(y_train, rule(train_df), folds)
            log_cv(cv)
            test_metrics = evaluate.all_metrics(y_test, rule(test_df))
            log_test(test_metrics)
            results.append(result_row(name, "baseline", feature_set, cv, test_metrics))
            print(f"{name:40s} test PR-AUC {test_metrics['pr_auc']:.3f}  precision@10% {test_metrics['precision_at_10pct']:.3f}"
                  f"  accuracy {test_metrics['accuracy_at_0_5']:.3f}")

    with mlflow.start_run(run_name="baseline_logreg_4_features"):
        mlflow.set_tags({**tags, "kind": "baseline", "feature_set": "simple_4"})
        simple = features.make_simple_pipeline(LogisticRegression(max_iter=1000))
        mlflow.log_params({"model": "LogisticRegression", "features": ",".join(config.SIMPLE_FEATURES), "C": 1.0})
        oof = cross_val_predict(simple, train_df[config.SIMPLE_FEATURES], y_train, cv=folds, method="predict_proba")[:, 1]
        cv = fold_metrics(y_train, oof, folds)
        log_cv(cv)
        simple.fit(train_df[config.SIMPLE_FEATURES], y_train)
        test_metrics = evaluate.all_metrics(y_test, simple.predict_proba(test_df[config.SIMPLE_FEATURES])[:, 1])
        log_test(test_metrics)
        results.append(result_row("baseline_logreg_4_features", "baseline", "simple_4", cv, test_metrics))
        print(f"{'baseline_logreg_4_features':40s} test PR-AUC {test_metrics['pr_auc']:.3f}  precision@10% {test_metrics['precision_at_10pct']:.3f}")

    # 3. tune each family twice: without PageValues (deployable) and with it (comparison only)
    tuned = {}
    for feature_set in FEATURE_SETS:
        print(f"tuning, feature set {feature_set}:")
        for family in MODEL_GRIDS:
            tuned[(family, feature_set)] = tune_family(family, feature_set, train_df, folds, tags)
            t = tuned[(family, feature_set)]
            results.append(result_row(f"tuned_{family}_{feature_set}", "candidate_model", feature_set, t["cv"]))

    # Choose by CV score only. The test rows have not been looked at yet.
    def best_family(feature_set):
        return max(MODEL_GRIDS, key=lambda f: tuned[(f, feature_set)]["cv"]["pr_auc"].mean())

    final_family = best_family("no_pagevalues")
    pv_family = best_family("with_pagevalues")
    chosen = tuned[(final_family, "no_pagevalues")]
    with_pv = tuned[(pv_family, "with_pagevalues")]
    final_model = chosen["model"]
    threshold = evaluate.capacity_threshold(chosen["oof"])
    pv_threshold = evaluate.capacity_threshold(with_pv["oof"])
    print(f"deployable family: {final_family}, 10% threshold {threshold:.3f} | with-PageValues family: {pv_family}")

    # 4. leakage check for PageValues, using the training rows only
    with mlflow.start_run(run_name="leakage_check_pagevalues"):
        mlflow.set_tags({**tags, "kind": "leakage_check"})
        pv_pos = train_df["PageValues"] > 0
        buyers = train_df[config.TARGET] == 1
        gap = with_pv["cv"]["pr_auc"].mean() - chosen["cv"]["pr_auc"].mean()
        leak = {
            "buy_rate_pagevalue_above_0": float(train_df.loc[pv_pos, config.TARGET].mean()),
            "buy_rate_pagevalue_0": float(train_df.loc[~pv_pos, config.TARGET].mean()),
            "share_of_buyers_with_pagevalue": float(pv_pos[buyers].mean()),
            "share_of_nonbuyers_with_pagevalue": float(pv_pos[~buyers].mean()),
            "pagevalues_alone_roc_auc": float(roc_auc_score(y_train, train_df["PageValues"])),
            "pagevalues_alone_pr_auc": float(average_precision_score(y_train, train_df["PageValues"])),
            "best_cv_pr_auc_without_pagevalues": float(chosen["cv"]["pr_auc"].mean()),
            "best_cv_pr_auc_with_pagevalues": float(with_pv["cv"]["pr_auc"].mean()),
            "cv_pr_auc_gain_from_pagevalues": float(gap),
        }
        mlflow.log_metrics(leak)
        mlflow.log_params({"decision": "deploy without PageValues",
                           "reason": "GA page value is credited by the session's own transaction, not known at exit intent"})
        pd.Series(leak).to_csv(config.REPORTS_DIR / "leakage_check_pagevalues.csv", header=["value"])
        print("leakage check:", {k: round(v, 3) for k, v in leak.items()})

    # 5 + 6. the one look at the test rows
    columns = config.RAW_INPUT_FEATURES
    with mlflow.start_run(run_name=f"FINAL_{final_family}_no_pagevalues") as final_run:
        mlflow.set_tags({**tags, "kind": "final", "stage": "final", "feature_set": "no_pagevalues",
                         "chosen_from_run": chosen["run_id"]})
        mlflow.log_params({"model": final_family, "feature_set": "no_pagevalues", "features": ",".join(columns),
                           "seed": config.SEED, "threshold": round(threshold, 4), "capacity": config.POPUP_CAPACITY,
                           "threshold_rule": "90th percentile of out-of-fold training scores"})
        mlflow.log_params({k.replace("model__", ""): v for k, v in chosen["params"].items()})
        log_cv(chosen["cv"])
        test_scores = final_model.predict_proba(test_df[columns])[:, 1]
        test_metrics = evaluate.all_metrics(y_test, test_scores, threshold)
        log_test(test_metrics)
        results.append(result_row(f"FINAL_{final_family}_no_pagevalues", "final", "no_pagevalues",
                                  chosen["cv"], test_metrics))
        print("final test:", {k: round(v, 3) for k, v in test_metrics.items()})

        # Every session gets a score from a model that never saw it: out-of-fold for the training
        # rows, the final model for the test rows. Slices then have five times more rows.
        heldout = pd.concat([train_df.assign(split="train", p_buy=chosen["oof"]),
                             test_df.assign(split="test", p_buy=test_scores)], ignore_index=True)
        heldout["flagged"] = (heldout["p_buy"] >= threshold).astype(int)
        heldout.to_csv(config.REPORTS_DIR / "heldout_predictions.csv", index=False)

        slices = evaluate.slice_table(heldout, heldout["p_buy"].values, threshold)
        slices.to_csv(config.REPORTS_DIR / "slices.csv", index=False)
        mlflow.log_artifact(str(config.REPORTS_DIR / "slices.csv"))
        new_visitors = slices[(slices["slice"] == "VisitorType") & (slices["group"] == "New_Visitor")].iloc[0]
        overall_precision = heldout.loc[heldout["flagged"] == 1, config.TARGET].mean()
        mlflow.log_metrics({"heldout_precision_at_threshold": overall_precision,
                            "heldout_new_visitor_precision": new_visitors["precision"],
                            "heldout_new_visitor_flag_rate": new_visitors["flag_rate"]})

        # Permutation importance on the test rows: how much PR-AUC drops when one input is shuffled.
        perm = permutation_importance(final_model, test_df[columns], y_test, scoring="average_precision",
                                      n_repeats=10, random_state=config.SEED, n_jobs=-1)
        importance = pd.DataFrame({"feature": columns, "importance_mean": perm.importances_mean,
                                   "importance_std": perm.importances_std}).sort_values("importance_mean")
        importance.to_csv(config.REPORTS_DIR / "permutation_importance.csv", index=False)
        mlflow.log_artifact(str(config.REPORTS_DIR / "permutation_importance.csv"))
        fig, ax = plt.subplots(figsize=(6.2, 5))
        ax.barh(importance["feature"], importance["importance_mean"], xerr=importance["importance_std"],
                color=ACCENT, ecolor=GREY)
        ax.set_xlabel("drop in test PR-AUC when the column is shuffled")
        ax.set_title("Permutation importance, final model (no PageValues)")
        style_axes(ax)
        save_fig(fig, "permutation_importance.png")

        # Confusion matrix at the 10% threshold
        fig, ax = plt.subplots(figsize=(4.4, 3.9))
        ConfusionMatrixDisplay.from_predictions(y_test, (test_scores >= threshold).astype(int),
                                                display_labels=["no purchase", "purchase"], ax=ax,
                                                colorbar=False, cmap="Blues")
        ax.set_xlabel("pop-up shown (predicted)")
        ax.set_ylabel("actual")
        ax.set_title(f"Test sessions at threshold {threshold:.2f}")
        save_fig(fig, "confusion_matrix_test.png")

        # Calibration: do the probabilities mean what they say?
        frac, mean_pred = calibration_curve(y_test, test_scores, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot([0, 1], [0, 1], "--", color=LIGHT, label="perfect")
        ax.plot(mean_pred, frac, "o-", color=ACCENT, label=final_family.replace("_", " "))
        ax.set_xlabel("predicted purchase probability")
        ax.set_ylabel("share that actually bought")
        ax.set_title("Calibration on test sessions")
        ax.legend(frameon=False)
        style_axes(ax)
        save_fig(fig, "calibration_test.png")

        # 8. save the artifact + metadata, check the round trip, register the model
        config.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(final_model, config.MODEL_PATH)
        reloaded = joblib.load(config.MODEL_PATH)
        same = np.allclose(reloaded.predict_proba(test_df[columns])[:, 1], test_scores)
        assert same, "round trip failed: the reloaded model gives different predictions"
        mlflow.log_metric("round_trip_ok", int(same))

        metadata = {
            "model_name": config.REGISTERED_MODEL_NAME,
            "model_version": "1.0.0",
            "model_family": final_family,
            "description": "Probability that this browsing session ends in a purchase, scored when the visitor shows "
                           "exit intent. The top 10% of sessions get the discount pop-up.",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "training_data": "UCI Online Shoppers Purchasing Intention (id 468), duplicates dropped, stratified 80% split",
            "data_md5": tags["data_md5"],
            "git_commit": tags["git_commit"],
            "mlflow_run_id": final_run.info.run_id,
            "input_features": columns,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "engineered_features": config.ENGINEERED_FEATURES,
            "excluded_features": {config.LEAKY_SUSPECT: "credited by the session's own purchase, not known at exit intent"},
            "best_params": {k.replace("model__", ""): v for k, v in chosen["params"].items()},
            "threshold": round(threshold, 4),
            "popup_capacity": config.POPUP_CAPACITY,
            "cv_metrics": {f"{k}_mean": round(float(v.mean()), 4) for k, v in chosen["cv"].items()}
                          | {f"{k}_std": round(float(v.std()), 4) for k, v in chosen["cv"].items()},
            "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "allowed_values": {"Month": config.MONTHS, "VisitorType": config.VISITOR_TYPES},
            # The "typical session" the app compares against when it explains a prediction:
            # the median of each number and the most common value of each category, training rows only.
            "typical_session": typical_session(train_df, columns),
            "library_versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                                 "pandas": pd.__version__, "numpy": np.__version__, "mlflow": mlflow.__version__},
        }
        config.META_PATH.write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(config.META_PATH))
        # cloudpickle instead of MLflow's default skops format: skops refuses the functions from
        # src/features.py unless every type is whitelisted by hand.
        mlflow.sklearn.log_model(final_model, name="model", input_example=test_df[columns].head(3),
                                 serialization_format="cloudpickle",
                                 registered_model_name=config.REGISTERED_MODEL_NAME)
        client = MlflowClient()
        latest = max(client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'"), key=lambda v: int(v.version))
        client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", latest.version)
        final_run_id = final_run.info.run_id

    # The with-PageValues model on the same test rows, as a comparison. Never deployed.
    pv_columns = config.RAW_INPUT_WITH_PV
    with mlflow.start_run(run_name=f"COMPARISON_{pv_family}_with_pagevalues_do_not_deploy"):
        mlflow.set_tags({**tags, "kind": "comparison", "feature_set": "with_pagevalues", "do_not_deploy": "true",
                         "chosen_from_run": with_pv["run_id"]})
        mlflow.log_params({"model": pv_family, "feature_set": "with_pagevalues", "threshold": round(pv_threshold, 4)})
        mlflow.log_params({k.replace("model__", ""): v for k, v in with_pv["params"].items()})
        log_cv(with_pv["cv"])
        pv_scores = with_pv["model"].predict_proba(test_df[pv_columns])[:, 1]
        pv_metrics = evaluate.all_metrics(y_test, pv_scores, pv_threshold)
        log_test(pv_metrics)
        results.append(result_row(f"COMPARISON_{pv_family}_with_pagevalues", "comparison", "with_pagevalues",
                                  with_pv["cv"], pv_metrics))
        perm_pv = permutation_importance(with_pv["model"], test_df[pv_columns], y_test, scoring="average_precision",
                                         n_repeats=10, random_state=config.SEED, n_jobs=-1)
        imp_pv = pd.DataFrame({"feature": pv_columns, "importance_mean": perm_pv.importances_mean,
                               "importance_std": perm_pv.importances_std}).sort_values("importance_mean")
        imp_pv.to_csv(config.REPORTS_DIR / "permutation_importance_with_pagevalues.csv", index=False)
        mlflow.log_artifact(str(config.REPORTS_DIR / "permutation_importance_with_pagevalues.csv"))
        print("with-PageValues test:", {k: round(v, 3) for k, v in pv_metrics.items()})

    # Figures that compare everything on the test rows
    simple_scores = simple.predict_proba(test_df[config.SIMPLE_FEATURES])[:, 1]
    curves = [
        (f"Final: {final_family.replace('_', ' ')}, no PageValues", test_scores, ACCENT, "-"),
        ("Logistic regression, 4 features", simple_scores, GREY, "-"),
        ("Rule B: products + low exit", evaluate.score_rule_b(test_df), GREY, ":"),
        ("With PageValues (not deployable)", pv_scores, "#b9c3cf", "--"),
    ]
    with mlflow.start_run(run_id=final_run_id):
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for label, scores, color, style in curves:
            p, r, _ = precision_recall_curve(y_test, scores)
            ax.plot(r, p, style, color=color, lw=2 if "Final" in label else 1.4, label=label)
        ax.axhline(y_test.mean(), color=LIGHT, lw=1)
        ax.text(0.01, y_test.mean() - 0.045, "base rate 15.6%", ha="left", color=GREY, fontsize=8)
        ax.set_xlabel("recall (share of buyers who get the pop-up)")
        ax.set_ylabel("precision (share of pop-ups that go to buyers)")
        ax.set_title("Precision-recall on the test sessions")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        style_axes(ax)
        save_fig(fig, "pr_curve_test.png")

        # Capacity curve: precision as the pop-up budget grows. The charter's limit is 10%.
        shares = np.linspace(0.02, 0.5, 49)
        fig, ax = plt.subplots(figsize=(6.2, 4))
        for label, scores, color, style in curves[:1] + curves[3:]:
            prec = [evaluate.precision_recall_of_flags(y_test, evaluate.top_share_flags(scores, s))["precision"] for s in shares]
            ax.plot(shares * 100, prec, style, color=color, lw=2 if "Final" in label else 1.4, label=label)
        ax.axvline(config.POPUP_CAPACITY * 100, color=GREY, lw=1, ls=":")
        ax.axhline(y_test.mean(), color=LIGHT, lw=1)
        ax.set_xlabel("% of sessions that get the pop-up")
        ax.set_ylabel("precision")
        ax.set_title("Precision as the pop-up budget grows")
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
        save_fig(fig, "capacity_curve_test.png")

        # Fold spread for every tuned family, both feature sets
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        labels, values = [], []
        for (family, feature_set), t in tuned.items():
            short = {"logistic_regression": "log. reg.", "hist_gradient_boosting": "grad. boost.",
                     "random_forest": "rand. forest"}[family]
            labels.append(f"{short}\n{'with PV' if feature_set == 'with_pagevalues' else 'no PV'}")
            values.append(t["cv"]["pr_auc"])
        box = ax.boxplot(values, patch_artist=True, widths=0.5,
                         whiskerprops={"color": GREY}, capprops={"color": GREY},
                         flierprops={"markeredgecolor": GREY})
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(ACCENT if "no PV" in label else LIGHT)
            patch.set_edgecolor(GREY)
        for median in box["medians"]:
            median.set_color("white")
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8.5)
        ax.set_ylabel("PR-AUC on each CV fold")
        ax.set_title("5-fold CV spread, tuned models")
        style_axes(ax)
        save_fig(fig, "cv_fold_spread.png")

    # 9a. month check: does a model trained on Feb-Oct still rank Nov-Dec sessions well?
    with mlflow.start_run(run_name="month_check_train_feb_oct_test_nov_dec"):
        mlflow.set_tags({**tags, "kind": "robustness_check", "feature_set": "no_pagevalues"})
        early = df[~df["Month"].isin(config.HOLIDAY_MONTHS)]
        late = df[df["Month"].isin(config.HOLIDAY_MONTHS)]
        month_model = clone(final_model).fit(early[columns], early[config.TARGET])
        late_scores = month_model.predict_proba(late[columns])[:, 1]
        month_metrics = evaluate.all_metrics(late[config.TARGET], late_scores)
        rule_b_late = evaluate.all_metrics(late[config.TARGET], evaluate.score_rule_b(late))
        mlflow.log_params({"model": final_family, "train_months": "Feb-Oct", "test_months": "Nov-Dec",
                           "train_rows": len(early), "test_rows": len(late)})
        log_test(month_metrics)
        mlflow.log_metrics({"rule_b_test_pr_auc": rule_b_late["pr_auc"],
                            "rule_b_test_precision_at_10pct": rule_b_late["precision_at_10pct"]})
        results.append(result_row("month_check_train_feb_oct_test_nov_dec", "robustness_check", "no_pagevalues",
                                  None, month_metrics))
        print("month check (Nov-Dec):", {k: round(v, 3) for k, v in month_metrics.items()},
              "rule B PR-AUC", round(rule_b_late["pr_auc"], 3))

    # 9b. drift check. Reference = training rows. Two "new batches": the random test rows (should be
    # stable) and a simulated holiday shift (Nov-Dec sessions against Feb-Oct training sessions).
    drift_cols = config.RAW_INPUT_FEATURES + ["p_buy"]
    reference = heldout[heldout["split"] == "train"]
    current = heldout[heldout["split"] == "test"]
    with mlflow.start_run(run_name="drift_check_psi"):
        mlflow.set_tags({**tags, "kind": "monitoring", "model_run_id": final_run_id})
        psi_test = evaluate.psi_table(reference, current, drift_cols)
        psi_test.to_csv(config.REPORTS_DIR / "psi_train_vs_test.csv", index=False)
        ref_season = reference[~reference["Month"].isin(config.HOLIDAY_MONTHS)]
        holiday = heldout[heldout["Month"].isin(config.HOLIDAY_MONTHS)]
        psi_holiday = evaluate.psi_table(ref_season, holiday, [c for c in drift_cols if c != "Month"])
        psi_holiday.to_csv(config.REPORTS_DIR / "psi_holiday_shift.csv", index=False)
        for path in ("psi_train_vs_test.csv", "psi_holiday_shift.csv"):
            mlflow.log_artifact(str(config.REPORTS_DIR / path))
        mlflow.log_metrics({"max_psi_train_vs_test": psi_test["psi"].max(),
                            "max_psi_holiday_shift": psi_holiday["psi"].max(),
                            "features_over_0_25_holiday": int((psi_holiday["psi"] > 0.25).sum()),
                            "flag_rate_test": float((current["p_buy"] >= threshold).mean()),
                            "flag_rate_holiday": float((holiday["p_buy"] >= threshold).mean()),
                            "flag_rate_feb_oct": float((ref_season["p_buy"] >= threshold).mean())})
        print("PSI train vs test (top 3):\n", psi_test.head(3).to_string(index=False))
        print("PSI holiday shift (top 5):\n", psi_holiday.head(5).to_string(index=False))

    pd.DataFrame(results).to_csv(config.REPORTS_DIR / "results_table.csv", index=False)
    (config.REPORTS_DIR / "final_run_id.txt").write_text(final_run_id + "\n")
    print("final run id:", final_run_id)


if __name__ == "__main__":
    main()
