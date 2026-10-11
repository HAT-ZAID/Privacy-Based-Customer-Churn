# Privacy-Based Customer Churn

Predicting churn usually means shipping raw customer records to one server. This builds the same
predictor two ways: a conventional XGBoost pipeline, and a federated Flower simulation where raw
rows never leave the branch they came from.

7,043 Telco customers · 52 features · XGBoost + FastAPI + Streamlit, and Flower/FedXgbBagging

```
Python 3.11 · xgboost 3.2 · scikit-learn 1.9 · flwr 1.31 · fastapi · streamlit
```

---

## Results

| | |
|---|---|
| Tuned XGBoost, held-out test (n=1,409) | accuracy 0.969, churn-F1 0.940 |
| Best model by 5-fold CV F1 | XGBoost 0.9307 ± 0.0107 |
| Majority-class baseline | accuracy 0.735 |

The 0.969 reads a little high — one feature carries more of the label than it should. I'm
re-running without it; see Open issues at the bottom.

---

## The dataset

IBM's Telco Customer Churn sample — a fictional California telco, one fiscal quarter, 7,043
customer records. **1,869 churned, 5,174 retained.**

The widely-used Kaggle version is 21 columns. This one is the wider Cognos Analytics variant, which
adds several fields the public version omits:

| Group | Columns |
|---|---|
| Geography | `Zip Code`, `Latitude`, `Longitude`, `Population` |
| Financials | `Total Refunds`, `Total Extra Data Charges`, `Total Long Distance Charges`, `Total Revenue` |
| Derived scores | `Satisfaction Score`, `Customer Status`, `Churn Score`, `CLTV` |
| Exit detail | `Churn Category`, `Churn Reason` |

50 raw columns in, 52 after one-hot encoding `Offer`, `Internet Type` and `Payment Method`. The
remaining derived fields are what make this dataset interesting *and* what makes it awkward — see
Open issues.

One structural quirk: churned rows are heavily front-loaded. Every one of the 1,869 sits in the
first 2,535 records, so **the last 60% of the file contains no churners at all.** Churn density by
tenth of the file runs 0.68, 0.38, 1.00, 0.60, then zero for the rest. Worth knowing before you
read a correlation matrix or split this by row.

Source: [`data/telco_original.csv`](data/telco_original.csv), generated IDs rather than real
customers.

---

## The notebook

`notebooks/Privacy_churn_v2.ipynb` — 67 cells, outputs recorded. This is where the numbers above
come from.

**Data.** 7,043 rows from the public [IBM/Kaggle Telco sample](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— generated IDs, not real customers. Dropped `Customer ID`, `Country`, `State`, `City`, `Quarter`,
`Customer Status` and `Churn Score`, then `Churn Category` and `Churn Reason` after looking at the
correlation matrix. `Offer`, `Internet Type` and `Payment Method` are one-hot encoded; the rest are
binary Yes/No maps. `Gender` → `Male=0, Female=1`. The test split is taken before any fitting.

**Model comparison** — five models, 5-fold CV on F1:

| Model | Mean F1 | Std |
|---|---:|---:|
| XGBoost | 0.9307 | 0.0107 |
| LightGBM | 0.9272 | 0.0111 |
| Random Forest | 0.9013 | 0.0211 |
| Decision Tree | 0.9006 | 0.0125 |
| AdaBoost | 0.8715 | 0.0161 |

**Tuning.** XGBoost and LightGBM were tuned separately — 200 trials each, seeded, with early
stopping on poor trials. Grid search ran first as a baseline. Final XGBoost settings:

```json
{ "max_depth": 19, "learning_rate": 0.031, "n_estimators": 909,
  "subsample": 0.949, "colsample_bytree": 0.510, "gamma": 0.971,
  "scale_pos_weight": 1.965, "reg_lambda": 0.323 }
```

`scale_pos_weight` ≈ 1.96 matches the 1:2.77 class imbalance (1,869 churned of 7,043), so the model
was pushed toward the minority class instead of ignoring it. On the held-out set that lands at
recall 0.922 / precision 0.958 on the churn class — the error profile you want for a retention
campaign, where a missed churner costs more than an unnecessary offer.

**Artifacts** (`models/`): `Telco_xgb.joblib`, `telco_encoder.joblib`, `feature_names.txt` (the 52
columns in order), `best_params.json`, `classification_report.txt`.

---

## Apps

`frontend/streamlit_app.py` renders all 52 features as a form and POSTs to `API/app.py`
(`/predict`, `/Health`, Pydantic-validated). `quickstart-xgboost/final_model.json` is the federated
model, served the same way.

```powershell
uvicorn API.app:app --reload
streamlit run frontend/streamlit_app.py
```

---

## Federated comparison

Same data, same algorithm, different constraint: no raw row crosses the network.

```
CENTRALIZED                                  FEDERATED
──────────                                   ─────────
notebooks/Privacy_churn_v2.ipynb             quickstart-xgboost/
  clean, split, fit, tune                     ├── preprocess.py   clean in place
  ↓                                            └── task.py        load → encode → shuffle
models/Telco_xgb.joblib                                         → partition → DMatrix
  + telco_encoder.joblib                           ↓
  ↓                                     ┌─────────────────────────────┐
API/app.py  FastAPI /predict              │  Client 0   rows stay here │
  ↓                                       │  Client 1   rows stay here │
frontend/streamlit_app.py                  └──────────┬──────────────────┘
                                                     │  ArrayRecord(model bytes)
                                                     │  MetricRecord(num-examples, auc)
                                                     ▼
                                              server_app.py  FedXgbBagging
                                              concatenates trees
                                                     ↓
                                              final_model.json  (6 trees)
```

Each round: the server sends the global trees → each branch boosts one new tree on local rows only →
the server concatenates them (`num_trees` + `tree_info` + updated tree IDs) → branches score the
result and return `auc` and `num-examples`.

Trees can't be weight-averaged the way neural-net weights can — averaging breaks the structure the
booster depends on — which is why Flower concatenates rather than averaging.

The two sides share no Python. What they share is the 52-column `FEATURE_NAMES` ordering, currently
duplicated in three files rather than imported from one config module.

### Federated results

| Model | Metric | Protocol |
|---|---|---|
| Centralized XGBoost | accuracy 0.969, churn-F1 0.940 | held-out test set, n=1,409, tuned |
| Federated round 1 | AUC 0.991 | 2 clients, per-client validation split |
| Federated round 2 | AUC 0.992 | |
| Federated round 3 | AUC 0.993 | 6-tree ensemble → `final_model.json` |

The federated AUCs come from each client's own 20% validation split — data that client just trained
on — so they're optimistic. The centralized row is a test set the model never saw. A server-side
global test set would make them comparable; that isn't built yet.

The federated run reproduces across invocations: `0.9909828030774182 → 0.9926511264596904`.

### Run it

```powershell
cd quickstart-xgboost
$env:PYTHONUTF8 = "1"                              # flwr prints emoji; cp1252 chokes
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"
flwr run . --run-config "num-server-rounds=3 local-epochs=1 save-model=true" --federation-config "num-supernodes=2" --stream
```

---

## Setup

```bash
python -m venv churnenv
churnenv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

Smaller cards should work by lowering `per_device_train_batch_size`; untested.

---

## Open issues

- **`Satisfaction Score` carries too much of the label.** It correlates −0.76 with churn and
  perfectly determines 62% of rows — scores 1–2 always churn, 4–5 never do. A single threshold on
  it alone reaches 0.939 accuracy, so it's carrying most of the headline number. `Customer Status`
  and `Churn Score` are dropped for the same reason; this one looked legitimate. Re-running without
  it is the first fix.
- **Hardcoded `D:\` paths** in `task.py` — a fresh clone raises `FileNotFoundError`. Written up in
  `docs/fed-guide.md` §7b.
- **`Gender` encoding differs between notebook passes.** v1 maps `Male→1`; everything shipped maps
  `Male→0`. Retraining from v1 gives an inverted model.
- **sklearn version skew** — encoder pickled by 1.6.1, env runs 1.9.0, so loading warns.
- **`df.reindex(columns=FEATURE_NAMES)` is dead code** in `task.py` — `inputs` binds before it and
  the result is never read.
- **A proper federated test set**, so the two result tables can be compared.
- **`FEATURE_NAMES` lives in three files** instead of one shared config.

---

## Docs

[`docs/fed-guide.md`](docs/fed-guide.md) — learn-by-doing guide: building the project, deployment
notes, practice exercises.

---

## Notes

- **AI use:** the centralized path is my own work; the federated side was adapted from Flower's
  XGBoost quickstart and was AI-assisted, used to learn Flower and XGBoost mechanics rather than to
  generate code.
- **Scope:** Flower simulation runtime only. The production path (TLS, SuperNode auth) is written up,
  not executed.
- MIT licensed.