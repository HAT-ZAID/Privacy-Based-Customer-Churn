# Privacy-Based Customer Churn

Predicting churn usually means shipping raw customer records to one server. This builds the same
predictor two ways: a conventional XGBoost pipeline, and a federated Flower simulation where raw
rows never leave the branch they came from.

7,043 Telco customers · 52 features · XGBoost + FastAPI + Streamlit, and Flower/FedXgbBagging

```
Python 3.11 · xgboost 3.2 · scikit-learn 1.9 · flwr 1.31 · fastapi · streamlit
```

---

## Headline, and the caveat attached to it

| | |
|---|---|
| Tuned XGBoost, held-out test (n=1,409) | accuracy **0.969**, churn-F1 **0.940** |
| Best model by 5-fold CV F1 | XGBoost **0.9307** ± 0.0107 |
| Majority-class baseline | accuracy 0.735 |

**The 0.969 is inflated, and I know why.** One feature — `Satisfaction Score` — is very close to a
copy of the label. Drop it and the number will change; that ablation hasn't been run yet. Everything
below is written with that in mind.

---

## The `Satisfaction Score` problem

Cross-tabulating `Satisfaction Score` against `Churn Label` across all 7,043 rows:

| Satisfaction Score | n | churned | churn rate |
|---|---:|---:|---:|
| 1 | 922 | 922 | **100%** |
| 2 | 518 | 518 | **100%** |
| 3 | 2,665 | 429 | 16.1% |
| 4 | 1,789 | 0 | **0%** |
| 5 | 1,149 | 0 | **0%** |

**62.2% of the dataset (4,378 rows) is perfectly determined by one integer.** A single rule —
*predict churn if `Satisfaction Score ≤ 2`* — reaches **0.939 accuracy using nothing else**, against
a 0.735 majority baseline.

That is why it dominates feature importance:

| Feature | Importance |
|---|---:|
| `Satisfaction Score` | **0.4544** |
| `Contract_Month-to-Month` | 0.0470 |
| `Tenure in Months` | 0.0392 |
| `Number of Referrals` | 0.0347 |

9.7× the next feature. The model is reading the answer off a column, not learning customer
behaviour.

**Why it survived preprocessing.** `preprocess_telco` drops `Customer Status` and `Churn Score` —
both derived by IBM's scoring engine and both near-copies of the label (`Customer Status` is a
one-to-one mapping; `Churn Score` agrees with the label 80.3% of the time). `Satisfaction Score`
comes from the same block of IBM-derived fields and is the most useful of the three. The obvious
leakers were removed; this one is subtle enough to look legitimate.

I can't prove from the data alone whether it's measured before or after the churn decision — the
dataset ships no timestamps for it. What the crosstab does establish is that it carries almost all
of the label's information. **The ablation is the test, and it's the first thing I'd fix.**

---

## The notebook

`notebooks/Privacy_churn_v2.ipynb` — 67 cells, outputs recorded. This is where the numbers above
come from.

**Data.** 7,043 rows from the public [IBM/Kaggle Telco sample](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— generated IDs, not real customers. Dropped: `Customer ID`, `Country`, `State`, `City`, `Quarter`,
`Customer Status`, `Churn Score`, then `Churn Category` and `Churn Reason` after looking at the
correlation matrix. `Offer`, `Internet Type` and `Payment Method` are one-hot encoded; the rest are
binary Yes/No maps. `Gender` → `Male=0, Female=1`. Test split is taken before any fitting.

**Model comparison** — five models, 5-fold CV on F1:

| Model | Mean F1 | Std |
|---|---:|---:|
| XGBoost | **0.9307** | 0.0107 |
| LightGBM | 0.9272 | 0.0111 |
| Random Forest | 0.9013 | 0.0211 |
| Decision Tree | 0.9006 | 0.0125 |
| AdaBoost | 0.8715 | 0.0161 |

**Tuning.** Optuna, 200 trials each for XGBoost and LightGBM — `TPESampler(seed=42)` with a
`MedianPruner`, so results reproduce. GridSearchCV was run first as a baseline. Best XGBoost
params:

```json
{ "max_depth": 19, "learning_rate": 0.031, "n_estimators": 909,
  "subsample": 0.949, "colsample_bytree": 0.510, "gamma": 0.971,
  "scale_pos_weight": 1.965, "reg_lambda": 0.323 }
```

`scale_pos_weight` ≈ 1.96 matches the 1:2.77 class imbalance (1,869 churned of 7,043), so the model
was pushed toward the minority class rather than ignoring it. On the held-out test set that lands at
**recall 0.922 / precision 0.958** on the churn class — the error type you'd want in a retention
campaign, since a missed churner costs more than an unnecessary offer.

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

Trees can't be weight-averaged the way neural-net weights can — the structure has to be preserved
or the model breaks — which is why Flower concatenates rather than averaging.

The two sides share no Python. What they share is the 52-column `FEATURE_NAMES` ordering, currently
duplicated in three files rather than imported from one config module. That's the main seam here.

### Results

| Model | Metric | Protocol |
|---|---|---|
| Centralized XGBoost | accuracy **0.969**, churn-F1 **0.940** | held-out test set, n=1,409, tuned |
| Federated round 1 | AUC **0.991** | 2 clients, per-client validation split |
| Federated round 2 | AUC **0.992** | |
| Federated round 3 | AUC **0.993** | 6-tree ensemble → `final_model.json` |

**These rows aren't comparable.** Federated AUC is scored on each client's own 20% validation split —
from the partition that client just trained on — so it's an optimistic upper bound. The centralized
row is a test set the model never saw. Comparing them properly needs a server-side global test set,
which isn't built.

The federated run reproduces across invocations: `0.9909828030774182 → 0.9926511264596904`.

### The `IidPartitioner` trap

`IidPartitioner` never shuffles — it calls `dataset.shard(contiguous=True)`. Telco is label-ordered
(all 1,869 churned rows sit inside the first ~3,500), so a contiguous split can hand one branch a
partition with **zero churned customers**. Its validation set goes single-class, ROC-AUC is
undefined, and `eval_set` returns `nan` — silently, with training otherwise looking fine.

```python
dataset = dataset.shuffle(seed=42)   # one line, before partitioning
```

IID partitioning is only IID if your input isn't ordered by the label.

### Run it

```powershell
cd quickstart-xgboost
$env:PYTHONUTF8 = "1"                              # flwr prints emoji; cp1252 chokes
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"
flwr run . --run-config "num-server-rounds=3 local-epochs=1 save-model=true" --federation-config "num-supernodes=2" --stream
```

---

## Open issues

- **Run the `Satisfaction Score` ablation.** The headline number depends on it.
- **Hardcoded `D:\` paths** in `task.py` — a fresh clone raises `FileNotFoundError`. Fix written up
  in [`docs/fed-guide.md`](docs/fed-guide.md) §7b.
- **`Gender` encoding differs between notebook passes.** v1 maps `Male→1`; everything shipped maps
  `Male→0`. Retraining from v1 gives an inverted model.
- **sklearn version skew** — encoder pickled by 1.6.1, env runs 1.9.0, so loading warns.
- **`df.reindex(columns=FEATURE_NAMES)` is dead code** in `task.py` — `inputs` binds before it and
  the result is never read. Order happens to match, so nothing breaks.
- **A proper federated test set** — without one the two result tables can't be compared.
- **`FEATURE_NAMES` lives in three files** instead of one shared config.

---

## Docs

| File | What it is |
|---|---|
| [`docs/fed-guide.md`](docs/fed-guide.md) | Learn-by-doing guide — building the project, deployment notes, practice exercises. |
| [`docs/UA_ONBOARDING.md`](docs/UA_ONBOARDING.md) | Architecture map — 20 files, 6 layers, 14-step tour. |
| [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md) | Deep dives on the five load-bearing components. |

---

## Setup

```bash
python -m venv churnenv
churnenv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

Smaller cards should work by lowering `per_device_train_batch_size`; untested.

## Notes

- **Data:** `telco_original.csv` is the public IBM/Kaggle sample set — generated IDs, not real
  customers.
- **AI use:** the centralized path is my own work; the federated side was adapted from Flower's
  XGBoost quickstart and was AI-assisted, used to learn Flower and XGBoost mechanics rather than to
  generate code.
- **Scope:** Flower simulation runtime only. The production path (TLS, SuperNode auth) is written up,
  not executed.
- MIT licensed.