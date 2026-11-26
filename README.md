# Who should see the discount pop-up? Purchase intent for online shoppers

**Recommendation.** Run the model as a limited pilot that decides who gets the exit-intent discount pop-up, with
10% of flagged visitors held back as a control group. Do not use the version built on Google Analytics page value. On
2,441 held-out sessions, the model sends 45% of its pop-ups to visitors who really buy, against 16% for a random 10% and 24%
for the best simple rule. That is nearly three times the buyers for the same pop-up budget, and it reaches 29% of all
buyers while showing pop-ups to only 10% of sessions. A model that also uses page value scores much higher (PR-AUC 0.75
against 0.37), but page value is partly made from the purchase itself and does not exist yet when the visitor is about to
leave, so that number cannot be trusted. Two things must happen before a full rollout. The pop-up line has to be re-set
every month, because one fixed line would flag about 21% of holiday sessions and only 4% in spring. The control group
then has to show that pop-ups add sales, not just discounts.

* **Final model:** histogram gradient boosting without `PageValues`, **MLflow run ID `a3bb379198604ce29b251049a71c2de3`**,
  registered as `shoppers-popup-purchase` version 1 with alias `champion`. `mlflow.db` is git-ignored, so on a fresh clone
  `python src/train.py` rebuilds the tracker (same scores, new run IDs). `models/model.json` records the run it came from.
* **Live demo:** (add URL). Deploy your own copy on Streamlit Community Cloud with
  [this one-click link](https://share.streamlit.io/deploy?repository=SanskritiAcharya/online-shoppers-ml&branch=main&mainModule=app/ui.py)
  and pick **Python 3.12** under Advanced settings.
* Charter: [CHARTER.md](CHARTER.md) · Monitoring: [MONITORING.md](MONITORING.md) · Data: [data/README.md](data/README.md)
* Report: [reports/report.md](reports/report.md) ([PDF](reports/report.pdf)) · Slides: [reports/slides.md](reports/slides.md) ([PDF](reports/slides.pdf))

## Problem

When a visitor shows exit intent, the store's marketing system decides whether to show a discount pop-up. The owner is
the e-commerce marketing manager, and the budget is pop-ups for at most 10% of sessions. A wasted pop-up annoys a visitor
or gives away margin. A missed one lets a close-to-buying visitor walk away. The model gives each session a purchase
probability, and the top 10% get the pop-up. The success and kill criteria were written in [CHARTER.md](CHARTER.md)
before any modelling.

## Data

UCI Online Shoppers Purchasing Intention Dataset (id 468), Sakar et al. (2019), *Neural Computing and Applications*,
<https://doi.org/10.1007/s00521-018-3523-0>. Licence CC BY 4.0. Downloaded 2026-11-26, MD5
`cc6ec1db03b4f10f8de52c56ff48b085`, 12,330 sessions and 18 columns, committed in `data/raw/`. After dropping 125 exact
duplicates there are 12,205 sessions and 15.6% end in a purchase. Split: stratified random 80/20 (no dates exist), with a
second check that trains on February to October and tests on November and December. Details in
[data/README.md](data/README.md).

## Results

Test = 2,441 held-out sessions, scored once. Precision and recall are at a 10% pop-up budget. CV = 5 stratified folds on
the training sessions (mean ± standard deviation across folds). Every row is an MLflow run.

| Approach | Uses page value? | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Precision at 10% | Recall at 10% |
|---|---|---|---|---|---|---|
| Constant "nobody buys" (84% accurate) | no | 0.156 ± 0.000 | 0.156 | 0.500 | 16% | 10% |
| Rule B: 20+ product pages and exit rate < 2.5% | no | 0.217 ± 0.011 | 0.217 | 0.633 | 24% | 15% |
| Logistic regression, 4 columns | no | 0.298 ± 0.024 | 0.301 | 0.728 | 35% | 22% |
| Logistic regression, tuned | no | 0.342 ± 0.028 | CV only | | | |
| Random forest, tuned | no | 0.383 ± 0.027 | CV only | | | |
| **Gradient boosting, tuned (final)** | **no** | **0.384 ± 0.019** | **0.372** | **0.790** | **45%** | **29%** |
| Rule A: page value > 0 | yes | 0.483 ± 0.015 | 0.493 | 0.849 | 55% | 35% |
| Random forest, tuned, with page value (not deployable) | yes | 0.754 ± 0.021 | 0.753 | 0.931 | 80% | 51% |

Every charter criterion is met without page value: PR-AUC 0.372 (target 0.30), precision 0.451 at 10% (target 0.35),
and +0.155 PR-AUC over Rule B (target +0.05). The model is well calibrated, and new visitors get precision 0.45 at
the threshold, so the fairness guardrail holds. The month check (train Feb to Oct, test Nov to Dec) gives PR-AUC 0.335
against 0.286 for Rule B.

**The PageValues decision.** Sessions with a page value above zero buy 56% of the time, against 4% for the rest. About 80% of
buyers have one, but only 12% of non-buyers. Google Analytics credits a page with the revenue of purchases that come
after it *in the same session*, so a buying session's own purchase feeds its own page value. Adding the column lifts CV
PR-AUC from 0.38 to 0.75, and shuffling it alone drops the comparison model's PR-AUC by 0.53. Both versions are tracked.
Only the one without it is deployable. See notebook 01, notebook 02 and section 3 of the report.

## How to run (about 5 minutes)

Python **3.12**.

```bash
git clone https://github.com/SanskritiAcharya/online-shoppers-ml.git && cd online-shoppers-ml
python3.12 -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.12 .venv
pip install -r requirements.txt

python src/train.py          # about 2 minutes: baselines, tuning, final model, registry, drift check
pytest -q                    # 23 tests: features, evaluation helpers, saved model, API, Streamlit app
```

GitHub Actions (`.github/workflows/ci.yml`) runs the tests and builds the Docker image on every push.

**Track.** `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open <http://127.0.0.1:5000>. The experiment is
`online-shoppers-popup`. Every run carries the tags `data_md5` and `git_commit`. Reload the final model with
`mlflow.sklearn.load_model("models:/shoppers-popup-purchase@champion")`.

**Serve.**

```bash
uvicorn app.api:app --reload       # http://127.0.0.1:8000/docs   POST /predict, GET /health
streamlit run app/ui.py            # http://localhost:8501
docker compose up --build          # both: API on 8000, Streamlit on 8501
```

Example request:

```bash
curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{"Administrative": 4,
  "Administrative_Duration": 120, "Informational": 1, "Informational_Duration": 30, "ProductRelated": 45,
  "ProductRelated_Duration": 1800, "BounceRates": 0.005, "ExitRates": 0.015, "SpecialDay": 0, "Month": "Nov",
  "VisitorType": "Returning_Visitor", "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2, "Weekend": false}'
# {"purchase_probability":0.3872,"show_popup":true,"recommended_action":"Show the discount pop-up. ...","threshold":0.3423,"model_version":"1.0.0"}
```

Invalid input (a negative page count, an unknown month, or sending `PageValues`) gets a 422 with the reason.

**Notebooks.** `jupyter notebook notebooks/` after `python src/train.py`. Each one runs top to bottom after a kernel
restart.

## Repository layout

Own layout, following the course template:

```
CHARTER.md  MONITORING.md  requirements.txt  Dockerfile  docker-compose.yml
data/raw/            frozen UCI file (committed, CC BY 4.0)
data/processed/      cleaned sessions (python -m src.data)
notebooks/           01 data audit and EDA, 02 baselines and modelling, 03 evaluation and error analysis
src/                 config, data, features, train, evaluate, predict
models/              model.joblib (full pipeline) and model.json (metadata)
app/                 api.py (FastAPI) and ui.py (Streamlit, loads the model directly, no API needed)
tests/               pytest for features, evaluation, the saved model, the API and the app
reports/             report and slides (md + PDF), figures, screenshots, result tables written by train.py
```
