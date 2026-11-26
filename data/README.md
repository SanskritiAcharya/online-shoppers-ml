# Data

## Source

| File | Rows | Columns | Size | MD5 |
|---|---|---|---|---|
| `raw/online_shoppers_intention.csv` | 12,330 sessions | 18 | 1.07 MB | `cc6ec1db03b4f10f8de52c56ff48b085` |

* **Dataset:** Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository, id 468.
  <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>
* **Download URL:** <https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip>
  (the CSV is the only file inside the zip).
* **Accessed:** 2026-11-26. Downloaded once and frozen. Every MLflow run logs the MD5 above as the `data_md5` tag,
  and `src/data.py` refuses to load a file with a different hash.
* **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0). Redistribution is allowed with credit, so the file is
  committed here.
* **Citation:** Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of online
  shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.
  *Neural Computing and Applications, 31*, 6893-6908. <https://doi.org/10.1007/s00521-018-3523-0>

Check the hash with `md5 data/raw/online_shoppers_intention.csv` (macOS) or `md5sum` (Linux).

The sessions come from one online store over one year. Each session belongs to a different user, according to the
dataset authors, and there is no user ID and no date, only the month.

## Processed

`python -m src.data` rebuilds `processed/sessions_clean.csv` from `raw/`.

Cleaning steps (all in `src/data.py`):

* Drop the 125 exact duplicate rows (12,330 to 12,205). They are most likely bounced one-page sessions that happen to
  match, but a duplicate that lands in both train and test would still inflate the test score.
* Rename the month `June` to `Jun`, so every month uses the same three-letter style. January and April do not appear.
* Turn `Weekend` and `Revenue` from True/False into 1/0.
* Keep `OperatingSystems`, `Browser`, `Region` and `TrafficType` as codes, but the pipeline treats them as categories,
  not numbers (browser 8 is not "bigger" than browser 2).
* There are no missing values in this version of the file, so nothing is filled in.

## Columns and when they are known

The decision point is the moment a visitor shows exit intent. Every counter in this file covers the **whole session**,
so in real use it would be the count up to that moment. That is a limitation of the data, discussed in the report.

| Column | Meaning | Known at exit intent? |
|---|---|---|
| `Administrative`, `Informational`, `ProductRelated` | Number of pages of each type viewed in the session | yes (so far) |
| `*_Duration` | Seconds spent on each page type | yes (so far) |
| `BounceRates` | Google Analytics bounce rate of the pages visited (average) | yes, from past traffic |
| `ExitRates` | Google Analytics exit rate of the pages visited (average) | yes, from past traffic |
| `PageValues` | Average GA page value of the pages visited. A page's value is the revenue of transactions that came after it | **doubtful**, see below |
| `SpecialDay` | Closeness to a holiday such as Valentine's Day (0 to 1) | yes |
| `Month` | Month of the session | yes |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | Integer codes | yes |
| `VisitorType` | Returning, new, or other | yes |
| `Weekend` | Session on a weekend | yes |
| `Revenue` | **Target**: the session ended in a purchase | no, it is the label |

**Why `PageValues` is a suspect.** Google Analytics gives a page value to pages viewed *before* a transaction in the same
session. So when a session buys something, its own purchase can push up the value of the pages it just viewed.
Sessions with a page value above zero buy 56% of the time, against 4% for the rest. Notebook 01 and the report look at
this in detail. Both models (with and without it) are tracked in MLflow, and the deployed model leaves it out.
