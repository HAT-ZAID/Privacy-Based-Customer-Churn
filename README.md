# Privacy-Based Customer Churn — Centralized & Federated

Predicting customer churn usually means shipping raw customer records to one server. This project builds the same predictor two ways, and the second one never lets a row leave the branch it came from.

- **Centralized** — Jupyter → XGBoost → FastAPI → Streamlit
- **Federated** — [Flower](https://flower.ai) `FedXgbBagging` across simulated branches, where only model trees and scalar metrics cross the network

7,043 Telco customers · 52 engineered features

```python
Python 3.11 · flwr 1.31 · xgboost 3.2 · scikit-learn 1.9 · fastapi · streamlit
```

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
  ↓                                       │  Client 1   rows stay here │   → new trees only
frontend/streamlit_app.py                  └──────────┬──────────────────┘
  52-field form ──POST──► /predict                     │  ArrayRecord(model bytes)
                                                     │  MetricRecord(num-examples, auc)
                                                     ▼
                                              server_app.py  FedXgbBagging
                                              concatenates trees
                                                     ↓
                                              final_model.json  (6 trees)
                                                     ↓
                                              frontend/federated_demo.py
```

The two sides share no Python. What they do share is the 52-column `FEATURE_NAMES` ordering — currently duplicated in three files rather than imported from one config module, which is the main seam in this repo.

Each federated round: server sends the global trees → each branch boosts one new tree on local rows only → server concatenates the trees (`num_trees` + `tree_info` + tree IDs updated) → branches score the new model and return `auc` and `num-examples`. Trees can't be weight-averaged the way neural-net weights can, which is why Flower concatenates rather than averaging.

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

**Apps**
```powershell
streamlit run frontend/federated_demo.py          # federated: 4 tabs
streamlit run frontend/streamlit_app.py           # centralized: 52-field form
uvicorn API.app:app --reload                      # FastAPI /predict
```

<!-- SCREENSHOT: streamlit run frontend/federated_demo.py -->

---

## Results

| Model | Metric | Protocol |
|---|---|---|
| Centralized XGBoost | accuracy **0.969**, churn-F1 **0.940** | held-out test set, n=1409, 400 trees, tuned |
| Federated round 1 | AUC **0.991** | 2 clients, per-client validation split |
| Federated round 2 | AUC **0.992** | |
| Federated round 3 | AUC **0.993** | 6-tree ensemble → `final_model.json` |

**These rows aren't comparable.** Federated AUC is scored on each client's own 20% validation split — from the partition that client just trained on — so it's an optimistic upper bound. The centralized row is a held-out test set the model never saw. Comparing them properly needs a server-side global test set.

The federated run reproduces across invocations: `0.9909828030774182 → 0.9926511264596904`.

---

## One gotcha worth knowing

`IidPartitioner` never shuffles — it calls `dataset.shard(contiguous=True)`. On a label-ordered dataset like Telco (all 1,869 churned rows inside the first ~3,500), a contiguous split hands one branch a partition with **zero churned customers**. Its validation set is single-class, ROC-AUC is undefined, and `eval_set` returns `nan` — silently, with training otherwise looking fine.

```python
dataset = dataset.shuffle(seed=42)   # one line, before partitioning
```

IID partitioning is only IID if your input isn't ordered by the label.

---

## Known open issues

| # | Issue | Where |
|---|---|---|
| 1 | **Hardcoded `D:\` paths** — a fresh clone raises `FileNotFoundError`. Two-line fix written up in [`docs/fed-guide.md`](docs/fed-guide.md) §7b. | `quickstart_xgboost/task.py:77-78` |
| 2 | **Gender encoding differs between notebook passes** — v1 maps `Male`→1, everything shipped maps `Male`→0. Consistent today, but retraining from v1 gives an inverted model. | `Privacy_churn_v1.ipynb` |
| 3 | **`df.reindex(columns=FEATURE_NAMES)` is dead code** — `inputs` binds before it and the result is never read. Order matches today, so nothing breaks, but the line enforces nothing. | `task.py:119-120` |
| 4 | **`Satisfaction Score` is unresolved** — correlation −0.76 with the label. If measured after the churn decision it's leakage; needs an ablation. | `FEATURE_NAMES` |
| 5 | **sklearn version skew** — encoder pickled by 1.6.1, env runs 1.9.0 → `InconsistentVersionWarning` on load. | `models/` |

---

## Documentation

| File | What it is |
|---|---|
| [`docs/fed-guide.md`](docs/fed-guide.md) | Learn-by-doing guide — building the project, deployment notes, practice exercises. |
| [`docs/UA_ONBOARDING.md`](docs/UA_ONBOARDING.md) | Architecture map — 20 files, 6 layers, 14-step guided tour. |
| [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md) | Deep dives on the five load-bearing components. |

---

## Notes

- **Data:** [`telco_original.csv`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) is the public IBM/Kaggle sample set — generated IDs, not real customers.
- **AI use:** the centralized path is my own work; the federated lab was AI-assisted, used to learn Flower and XGBoost mechanics rather than to generate code.
- **Scope:** Flower simulation runtime only. The production path (TLS, SuperNode auth) is written up, not executed.
- MIT licensed.
