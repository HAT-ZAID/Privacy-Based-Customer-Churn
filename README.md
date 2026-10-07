# Privacy-Based Customer Churn — Centralized & Federated

Predicting customer churn usually means shipping raw customer records to one server. This project builds the same predictor two ways, and the second one never lets a row leave the branch it came from.

- **Centralized path** — Jupyter → XGBoost → FastAPI → Streamlit
- **Federated path** — [Flower](https://flower.ai) `FedXgbBagging` across simulated branches, where only model trees and scalar metrics cross the network

Dataset: 7,043 Telco customers, 52 engineered features.

```python
Python 3.11 · flwr 1.31 · xgboost 3.2 · scikit-learn 1.9 · fastapi · streamlit
```

---

## Why federated

In the centralized setup every customer row is copied to one place, so the sensitive part of the system is the data lake. Federated learning inverts that: each branch trains on its own rows, and only the resulting *trees* are sent to an aggregator.

The privacy claim here is narrow and worth stating precisely — **raw rows never leave the client partition.** It is not differential privacy. Tree structures can still leak information, and I don't claim otherwise.

---

## Architecture

```
CENTRALIZED                                  FEDERATED
──────────                                   ─────────
notebooks/Privacy_churn_v2.ipynb             quickstart-xgboost/
  clean, split, fit, tune                     ├── preprocess.py   clean in place
  ↓                                            └── task.py        load → encode → shuffle
models/Telco_xgb.joblib                                         → partition → DMatrix
  + telco_encoder.joblib                           ↓
  ↓                                     ┌─────────────────────────────┐
API/app.py  FastAPI /predict              │  Client 0   rows stay here │   ClientApp.train
  holds model in app.state                │  Client 1   rows stay here │   → new trees only
  ↓                                       └──────────┬──────────────────┘
frontend/streamlit_app.py                            │  ArrayRecord(model bytes)
  52-field form ──POST──► /predict                   │  MetricRecord(num-examples, auc)
                                                   ▼
                                            server_app.py
                                            FedXgbBagging
                                            concatenates trees
                                                   ↓
                                            final_model.json  (6 trees)
                                                   ↓
                                            frontend/federated_demo.py
```

The federated side shares no Python with the centralized side. The one thing they do share is the 52-column `FEATURE_NAMES` ordering — and that is duplicated in three files rather than imported from one config module, which is the main architectural seam in this repo.

---

## Quickstart

```bash
python -m venv churnenv
churnenv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

**Federated run** — 3 rounds, 2 simulated branches:
```powershell
cd quickstart-xgboost
$env:PYTHONUTF8 = "1"                              # flwr prints emoji; cp1252 chokes
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"
flwr run . --run-config "num-server-rounds=3 local-epochs=1 save-model=true" --federation-config "num-supernodes=2" --stream
```

**Streamlit demo**
```powershell
streamlit run frontend/federated_demo.py          # federated: 4 tabs
streamlit run frontend/streamlit_app.py           # centralized: 52-field form
```

**Centralized API**
```powershell
uvicorn API.app:app --reload
```

> ⚠️ **A fresh clone does not run yet.** `quickstart_xgboost/task.py` still hardcodes absolute `D:\` paths, so `load_data` raises `FileNotFoundError`. The two-line fix is written up in [`docs/fed-guide.md`](docs/fed-guide.md) §7b. Everything else in this README is verified working.

<!-- SCREENSHOT: streamlit run frontend/federated_demo.py — capture the 4-tab view -->

---

## Results

| Model | Metric | Protocol |
|---|---|---|
| Centralized XGBoost | accuracy **0.969**, churn-F1 **0.940** | held-out test set, n=1409, 400 trees, tuned |
| Federated round 1 | AUC **0.991** | 2 clients, per-client validation split |
| Federated round 2 | AUC **0.992** | |
| Federated round 3 | AUC **0.993** | 6-tree ensemble → `final_model.json` |

**These rows are not comparable.** Federated AUC is scored on each client's own 20% validation split — drawn from the partition that client just trained on — so it is an optimistic upper bound. The centralized row is a held-out test set the model never saw. Comparing them properly needs a server-side global test set, which is the first thing I'd build next.

I verified the federated run reproduces bit-identically across separate invocations: `0.9909828030774182 → 0.9926511264596904`.

---

## The bug worth reading about

My first federated run completed all three rounds without error and reported **`auc: nan`** every time.

`IidPartitioner` is named for IID sampling, but its implementation calls `dataset.shard(num_shards=N, index=i, contiguous=True)` — **it never shuffles.** The Telco dataset is label-ordered: all 1,869 churned rows sit within the first ~3,500. So a contiguous 2-way split gave branch 1 a partition with **zero churned customers**. Its validation set was single-class, and ROC-AUC is undefined on one class.

The nasty part is that it failed *silently*. Training looked fine. `eval_set` just returned `nan`, and `float("nan")` propagated into the aggregated metric with no exception anywhere.

```python
dataset = dataset.shuffle(seed=42)   # one line, before partitioning
```

After: both partitions hold both classes, and per-client AUCs came back 0.988 / 0.995.

The general lesson I took from it: **IID partitioning is only IID if your input isn't ordered by the label.** Check your data's ordering before trusting the partitioner.

---

## How one federated round works

1. Server starts with an **empty** model (`b""`) and sends it to every branch.
2. Round 1: `client_app.py` sees `server-round == 1` and calls `xgb.train()` from scratch — there's nothing to load. (Rounds 2+ do `bst.load_model()` then boost more trees. This branch is load-bearing: `bst.load_model(b"")` crashes the process rather than raising.)
3. Each client trains **1 tree** on its local rows and slices out only that tree, so the server isn't re-receiving the whole model.
4. Server concatenates: `aggregate_bagging` parses both JSON models, bumps `num_trees`, appends to `trees` and `tree_info`, and renumbers the new tree IDs.
5. Clients evaluate the new global model and return only `auc` + `num-examples`.
6. Repeat. After 3 rounds: `3 rounds × 2 clients × 1 tree = 6 trees`.

### Why bagging and not FedAvg

FedAvg averages weights, which works because neural-net weights are continuous numbers in an aligned space. Trees aren't — a split is either `Age > 30` or it isn't, and different clients produce trees with different shapes. Averaging `Age > 30` and `Age > 45` gives you `Age > 37.5`, a threshold nobody learned and that may align with nothing.

So `FedXgbBagging` concatenates instead. The ensemble grows by `num_clients × local_epochs` per round, and every client's trees get an equal vote regardless of how many rows that client held.

---

## What I got wrong along the way

The first notebook pass had real problems, and they're still visible in `notebooks/Privacy_churn_v1.ipynb`:

- **Label leakage into feature selection.** `Customer Status` is the label restated (`Churned` ≡ label 1) and `Churn Score` is a vendor-side churn *prediction* shipped as an input. Both were fed into the importance analysis, which made the ranking meaningless.
- **A real pandas bug.** `df = df.drop(..., inplace=True)` assigns `None`, because pandas' `inplace` methods return `None`. That notebook's final export cell crashed with `AttributeError: 'NoneType' object has no attribute 'to_csv'`.
- **Encoder fitted before the split**, so test-set categories leaked into training.

`Privacy_churn_v2.ipynb` fixes the ordering. `preprocess.py` is that notebook's cleaning logic, extracted so the centralized and federated paths run literally the same code.

---

## Known open issues

Listed rather than hidden:

| # | Issue | Where |
|---|---|---|
| 1 | **Hardcoded `D:\` paths** break a fresh clone | `quickstart_xgboost/task.py:77-78` |
| 2 | **Gender encoding differs between notebook passes.** v1 maps `Male`→1; `preprocess.py`, v2, and `streamlit_app.py` all map `Male`→0. The shipped path is consistent because v2 produced the artifacts, but retraining from v1 gives an inverted model. | `Privacy_churn_v1.ipynb` |
| 3 | **`df.reindex(columns=FEATURE_NAMES)` is dead code.** `inputs` binds before the reindex and the result is never read. The order happens to match today (verified: 0 of 52 differ), so nothing breaks — but the line enforces nothing. | `task.py:119-120` |
| 4 | **`Satisfaction Score` is unresolved.** Correlation −0.76 with the label. If the survey is collected after the churn decision it's leakage and my headline numbers are partly explaining the answer. Needs an ablation. | `FEATURE_NAMES` |
| 5 | **sklearn version skew.** `telco_encoder.joblib` was pickled by 1.6.1, this env runs 1.9.0 → `InconsistentVersionWarning` on every load. Works, but noisy. | `models/` |
| 6 | **The two AUC protocols aren't comparable.** | both paths |

---

## Documentation

| File | What it is |
|---|---|
| [`docs/fed-guide.md`](docs/fed-guide.md) | Learn-by-doing guide. Part I builds the MVP (§0–§9), Part II covers post-upload work, interview prep, and 23 practice exercises. |
| [`docs/UA_ONBOARDING.md`](docs/UA_ONBOARDING.md) | Architecture map — all 20 files, 6 layers, 14-step guided tour. Every claim tagged verified vs read-from-source. |
| [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md) | Deep dives on the five load-bearing components. |

---

## Notes

- **Data:** `data/telco_original.csv` is the public [Kaggle Telco churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM). Customer IDs are generated, all rows are California — this is sample data, not real customers.
- **How much of this is AI-assisted:**
  - **The centralized path is my own work** — `notebooks/Privacy_churn_v1.ipynb`, `notebooks/Privacy_churn_v2.ipynb`, `API/app.py`, and `frontend/streamlit_app.py` were written without AI assistance.
  - **The federated lab is AI-assisted**, and I used AI as a learning tool rather than a code generator: I asked for explanations of Flower and XGBoost mechanics, read the answers, then implemented the Telco migration myself. The `auc: nan` bug was found by me reading the logs — `dataset.shuffle(seed=42)` is my fix.
- **Scope:** Flower simulation runtime only — no live multi-machine deployment. The production path (TLS, SuperNode auth) is written up, not executed.
- MIT licensed.
---

