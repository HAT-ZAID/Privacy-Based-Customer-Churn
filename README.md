# Privacy-Based Customer Churn Prediction (Centralized + Federated)

A churn prediction system built on the Telco Customer Churn dataset (7,043 rows, 52 features), with both a centralized serving pipeline and a federated learning simulation using Flower (flwr) + XGBoost. Raw customer rows never leave their partition in the federated path — only model trees and metrics cross the network.

## What's inside

```
data/                  Telco churn dataset (CSV + Parquet)
notebooks/             EDA + model training (Privacy_churn_v1.ipynb)
models/                Saved XGBoost model + OneHotEncoder + params
API/                   FastAPI serving (/predict endpoint)
frontend/
  streamlit_app.py     Centralized predictor UI (full 52-field form)
  federated_demo.py    Federated demo UI (4 tabs: Learn / Run / Compare / Predict)
  fl_rounds_sample.csv Per-round federated AUC results
quickstart-xgboost/    Flower federated simulation app
  quickstart_xgboost/
    task.py            Telco data loader (shuffle → IID partition → DMatrix)
    preprocess.py      preprocess_telco() — shared cleaning logic
    client_app.py      @app.train / @app.evaluate (per-client XGBoost)
    server_app.py      FedXgbBagging server loop
docs/
  fed-guide.md         Learn-by-doing guide — Part I builds the MVP, Part II is post-upload
  UA_ONBOARDING.md     Architecture map: 20 files, 6 layers, 14-step guided tour
  CODE_WALKTHROUGH.md  Deep dives on the 5 load-bearing components
```

## Federated architecture

- **Framework:** Flower (flwr) 1.31.0, Simulation Runtime (Ray backend)
- **Algorithm:** FedXgbBagging — server concatenates client XGBoost trees (bagging), not weight averaging
- **Partitioning:** 2 clients, IID split after `dataset.shuffle(seed=42)` (shuffle required — source data is label-sorted)
- **Privacy boundary:** raw rows stay on the client; only tree bytes (`ArrayRecord`) + metrics (`num-examples`, `auc`) are transmitted. No differential privacy claimed — tree gradients can still leak information.

## How to run

### Environment setup

```powershell
.\churnenv\Scripts\Activate.ps1
python -m pip check   # expect: No broken requirements found
```

### Federated simulation (3 rounds, 2 clients)

```powershell
cd quickstart-xgboost
$env:PYTHONUTF8 = "1"
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"
flwr run . --run-config "num-server-rounds=3 local-epochs=1 save-model=true" --federation-config "num-supernodes=2" --stream
```

Produces `final_model.json` (6 trees, 52 features).

### Streamlit demo

```powershell
streamlit run frontend/federated_demo.py
```

### Centralized API

```powershell
uvicorn API.app:app --reload
```

## Results

| Model | Metric | Protocol |
|-------|--------|----------|
| Centralized XGBoost (notebook) | accuracy **0.969**, churn-F1 **0.940** | held-out test set, n=1409, 400 trees, tuned |
| Federated round 1 | AUC **0.991** | 2 clients, per-client validation split |
| Federated round 2 | AUC **0.993** | |
| Federated round 3 | AUC **0.993** | 6-tree ensemble → `final_model.json` |

> **The two rows are not directly comparable.** Federated AUC is measured on each client's own 20% validation split — drawn from the partition that client just trained on — so it is an optimistic upper bound. The centralized row is a held-out test set the model never saw. Comparing them honestly requires a server-side global test set (see `docs/fed-guide.md` §11, Tier 1 item 5).

## Key implementation notes

- **Shuffle before partitioning.** `IidPartitioner` does not shuffle — it calls `dataset.shard(contiguous=True)`. The Telco dataset is label-ordered, so all 1,869 churned rows landed inside the first partition. Client 2 received zero churned rows, its validation set became single-class, and `eval_set` returned `auc: nan` on **every** round without raising. Fixed with `dataset.shuffle(seed=42)` in `task.py`.
- **Round 1 is special.** The server initialises the global model as empty bytes `b""`; clients branch on `server-round == 1` and train from scratch. Rounds 2+ `load_model` then boost more trees. `bst.load_model(b"")` crashes the process rather than raising, so the branch is load-bearing.
- **Bagging, not averaging.** Trees are discrete split structures with no element-wise correspondence, so `FedXgbBagging` concatenates client trees instead of averaging them. The ensemble grows by `num_clients × local_epochs` per round — 6 trees after 3 rounds here.
- **`_local_boost` slices the last N trees** so the server receives only this round's new trees, keeping `aggregate_bagging`'s tree indexing valid.

## Learning guide

`docs/fed-guide.md` is split into two parts:

- **Part I — Build the MVP** (§0–§9): environment, ML/FL primers, three labs (HIGGS warm-up, Telco migration, portability), the Streamlit capstone, and an upload gate with a clean-clone test.
- **Part II — After upload** (§10–§13): deployment notes, a tiered future-work roadmap, an interview script, and 23 practice questions/exercises.

Two generated companions sit alongside it: `docs/UA_ONBOARDING.md` (architecture map) and `docs/CODE_WALKTHROUGH.md` (component deep dives).

## Known open issues

These are real and documented rather than hidden:

1. **Hardcoded absolute paths** in `quickstart-xgboost/quickstart_xgboost/task.py` — `D:\Privacy Based Customer Churn\...`. A fresh clone fails with `FileNotFoundError`. Fix documented in `docs/fed-guide.md` §7b.
2. **Gender encoding inversion** — `notebooks/Privacy_churn_v1.ipynb` maps `Male`→1 while `preprocess.py`, `Privacy_churn_v2.ipynb` and `frontend/streamlit_app.py` map `Male`→0. The shipped path is self-consistent because v2 produced the artifacts; retraining from the v1 pipeline would produce an inverted model.
3. **`df.reindex(columns=FEATURE_NAMES)` is dead code** in `task.py` — `inputs` is bound before the reindex and the reindexed frame is never read. The resulting order happens to match `FEATURE_NAMES` today, but the line enforces nothing.
4. **`Satisfaction Score` is unresolved** — correlation −0.76 with the label; not yet established whether it is leakage.
5. **sklearn version skew** — `models/telco_encoder.joblib` was pickled by 1.6.1, `churnenv` runs 1.9.0, so every load prints `InconsistentVersionWarning`.

## Honest scope

- Built with AI assistance for guidance; I wrote the data loader, ran the experiments, found and fixed the `auc: nan` bug, and built the Streamlit tabs.
- Simulation runtime only — no live multi-machine deployment. The deployment path (TLS, SuperNode auth) is written up, not executed.
- No differential privacy. The privacy claim is limited to "raw customer rows never leave the partition"; tree bytes can still leak.
- Most of this project is not yet committed — only 3 commits exist, and the federated app, models, frontend and docs are untracked. See `docs/fed-guide.md` §9 for the commit sequence.
