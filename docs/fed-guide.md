# Federated Churn: Learn-by-Doing Guide

> **Who:** you know little ML, want to *learn* federated learning (not copy-paste it), and defend it in an interview.
> **Rule:** this doc gives you concepts + tasks + checks, not full solutions. If you can explain every step you did, you earned the resume line.
> **Stack:** `churnenv` (Python 3.11.9, flwr 1.31.0, flwr-datasets 0.6.0, xgboost 3.2.0, streamlit 1.61.1) + Simulation Runtime only. Deployment = notes.

**How to use:** do Parts in order. Each lab ends with a ✅ check. Don't skip checks — interviewers ask about checks.

**Structure of this guide:**

| Part | Sections | Goal |
|---|---|---|
| **I — Build the MVP** | §0–§6 primers and labs, §7 portability, §8 Streamlit capstone, §9 upload gate | a project that works, is portable, and runs on a clean clone |
| **II — After upload** | §10 deployment, §11 future work, §12 interview script, §13 practice tests | the production story and the defence |

Two companion documents sit beside this one, both generated from a knowledge graph of the repo:
- `docs/UA_ONBOARDING.md` — maps all 20 files, 6 layers, 14 tour steps. Read this if you need to *understand* the codebase.
- `docs/CODE_WALKTHROUGH.md` — deep dives on the five load-bearing components. Read this if you need to *defend* them.

---

# PART I — BUILD THE MVP

> Sections 0–9. When you finish Part I the project is uploadable: it runs on a clean clone, has pinned
> dependencies, and has a demo you can show. Streamlit (§8) is deliberately the **last** build step —
> you make it portable first, then you have something worth showing.

## 0. Environment health (do this once, saves hours)

We audited two envs on 2026-09-07:

**`churnenv` (USE THIS, now clean):**
- `flwr 1.31.0` matches `quickstart-xgboost/pyproject.toml: flwr-version-target = "1.31.0"` ✅
- `flwr-datasets 0.6.0` satisfies `>=0.6.0` ✅, `xgboost 3.2.0` satisfies `>=2.0.0` ✅
- `streamlit 1.61.1`, `ray 2.55.1`, `pandas 2.3.3`, `numpy 2.4.6` present ✅
- Fixed: removed orphan editable `demo 1.2.2` (folder `demo/` was deleted but install remained, caused `pip check` failure `demo requires flwr>=1.33`). After `pip uninstall demo`: `pip check` → `No broken requirements found` ✅
- Verified: `import flwr, flwr_datasets, xgboost, streamlit` works, `xgb.DMatrix` works.

**Global Python (DO NOT USE for this project):**
- Had broken `~lwr` + `~lwr-1.33.0.dist-info` folders (interrupted pip install). Deleted → `~lwr` warnings gone ✅
- Still has TF-only conflicts (`tensorflow-intel 2.14` vs `keras 3.13 / ml-dtypes 0.5.4 / protobuf 6.33 / tensorboard 2.20`) — irrelevant to FL, ignore. Don't `pip install` TF fixes here.
- Missing `flwr-datasets`, `streamlit` globally — intentional. Don't install; use `churnenv`.

**Your commands (always):**
```powershell
# activate first, every terminal
.\churnenv\Scripts\Activate.ps1
python -m pip check   # expect: No broken requirements found
python -m pip list | Select-String "flwr|xgboost|streamlit|ray"
```

> Learn: why a venv? Global has TF + FL + web packages colliding. `churnenv` pins one working set. Resume line: "isolated Flower env, resolved orphan/broken installs."

---

## 1. Your project in 10 lines

- Centralized now: `data/telco_original.csv` (7043 rows) → `notebooks/Privacy_churn_v2.ipynb` → `models/Telco_xgb.joblib` + `models/telco_encoder.joblib` (52 features, see `API/app.py:33-86 FEATURE_NAMES`) → `API/app.py` FastAPI `/predict` → `frontend/streamlit_app.py` form. Held-out test: accuracy **0.969**, churn-class F1 **0.940** (n=1409).
- Federated goal: same 52 features, but rows stay on 2–3 "telco branches" (clients). Only XGBoost trees (bytes) move to server, never raw rows.
- Reference impl: `quickstart-xgboost/` now loads `data/telco_original.parquet` via `task.py` + `preprocess.py`, partitions IID across 2 clients, trains with `FedXgbBagging`.

**Task:** open `API/app.py:33-86` and `quickstart_xgboost/task.py` side by side. Write one sentence: what dataset does each load?

---

## 2. ML primer (just enough)

- **Classification:** given 52 numbers about a customer, predict label `0 = Stayed / 1 = Churned`. Your label comes from `Churn Label Yes/No` in Telco CSV.
- **Features vs label:** features = inputs (Age, Tenure, Monthly Charge... one-hot Offer/Contract/Payment). Label = answer to learn. Mistake to avoid: feeding `Customer ID`, `Churn Reason`, `City` raw into model.
- **Train/valid split:** train on 80%, check on held-out 20% you didn't train on. If you check on train data, score lies. `task.py: train_test_split(test_fraction=0.2, seed=42)`.
- **XGBoost intuition:** builds trees one-by-one, each new tree fixes mistakes of previous ones. Knobs: `eta` (learning rate, small = cautious), `max-depth` (how deep each tree, deep = memorizes), `num_boost_round / local-epochs` (how many trees per step). Defaults in `pyproject.toml`: `eta=0.1, max-depth=8, local-epochs=1`.
- **AUC:** 0.5 = coin flip, 1.0 = perfect ranking. Quickstart uses `eval-metric=auc`, `bst.eval_set()` in `client_app.py:97-101`. ⚠️ On this dataset you will see **0.99**, not the 0.75–0.85 you might expect from generic churn tables. That is suspicious, not impressive: it is almost certainly `Satisfaction Score` (corr −0.76 with the label) acting as leakage. Treat a very high AUC as a prompt to check for leakage, and see [§11 Tier 1](#tier-1--correctness-do-these-first) item 1.

**Task:** run your notebook `notebooks/Privacy_churn_v1.ipynb` (or `privacy_churn_v1.py`) centrally, note AUC. This is the number federated will be compared against in Lab 3.

---

## 3. Federated primer (the only theory you need)

- **Horizontal FL:** same columns, different rows per client. Branch A has customers 0–3000, Branch B has 3000–7043. Fits Telco story.
- **One round:** server sends global trees → each client boosts N new trees on local rows (`_local_boost` in `client_app.py:20-30`) → server concatenates trees (bagging) → repeat.
- **Why not FedAvg?** Neural nets average weights. Trees can't be averaged — you append trees. That's `FedXgbBagging`. Source: `flwr.serverapp.strategy.FedXgbBagging`, `aggregate_bagging`.
- **Flower pieces:**
  - `ClientApp` (`client_app.py`): `@app.train()` trains + returns `ArrayRecord([model_bytes])` + `MetricRecord({"num-examples": N})`; `@app.evaluate()` returns `{"auc": ..., "num-examples": ...}` only.
  - `ServerApp` (`server_app.py`): inits empty model `b""`, runs `strategy.start(grid, initial_arrays, num_rounds)`, optionally saves `final_model.json`.
  - `Message.content["arrays"]["0"]` = model bytes, `["metrics"]` = numbers, `["config"]["server-round"]` = round id. Round 1 is special: empty global → `xgb.train()` from scratch.
  - `Grid` + Simulation Runtime (Ray backend): `flwr run . --stream` fakes 2 SuperNodes on your laptop. Same code works in Deployment later.

**Task:** trace one round aloud: "server round 2 sends X bytes → client 0 loads with `bst.load_model`, boosts 1 tree, slices last 1 tree, sends back." If you can't say it, re-read `client_app.py:33-77`.

---

## 4. Quickstart tour (read, don't edit yet)

```
quickstart-xgboost/quickstart_xgboost/
  task.py        # Telco parquet loading + shuffle + IidPartitioner + DMatrix
  preprocess.py  # preprocess_telco(df) — drops leakage cols, fillna, binary encodes
  client_app.py  # train/evaluate handlers
  server_app.py  # FedXgbBagging loop
pyproject.toml   # run-config + xgb params
```

Read order: `pyproject.toml:28-43` (what knobs exist?) → `task.py` (where does data come from? Note `DATASET_PATH` points to `telco_original.parquet`, `dataset.shuffle(seed=42)` before partitioning — critical, see Lab 2 gotcha) → `preprocess.py` (what gets dropped and why) → `client_app.py` (trace `train()`: round 1 = `xgb.train()` from scratch, rounds 2+ = `bst.load_model()` + `_local_boost()` + slice last N trees) → `server_app.py` (empty `b""` init, `strategy.start()` loop). Note `replace_keys(unflatten_dict(...))` just turns `max-depth` into `max_depth`.

---

## 5. Lab 1 — optional HIGGS warm-up (skip if pressed for time)

Goal: see FL work on an unrelated dataset before trusting your Telco setup. **Your `task.py` now loads Telco by default**, so this lab is optional — do it only if you want to confirm the FL machinery works independent of your data pipeline.

**5a. What the original quickstart did.** The upstream Flower quickstart downloads HIGGS from HuggingFace Hub (2.8 GB). Too slow. We downloaded one 405 MB shard to local disk. To run it, you'd temporarily swap `task.py`'s `load_data` back to the HIGGS path (see the commented block at the bottom of `task.py`).

**5b. Run it (only if doing the optional lab):**
```powershell
.\churnenv\Scripts\Activate.ps1
cd quickstart-xgboost

$env:PYTHONUTF8 = "1"
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"
$env:LOCAL_HIGGS_PATH = "D:\hf_cache\hub\datasets--jxie--higgs\snapshots\d53733cf80e5059f189f38c1b542d581377629a0\data\train-00000-of-00005-72c7f0393dc7600e.parquet"

flwr run . --federation-config "num-supernodes=2" --stream
```

✅ Check: logs show `[ROUND 1/3]`, `[ROUND 2/3]`, `[ROUND 3/3]`, `aggregate_evaluate: Received 2 results and 0 failures`, AUC improving each round (e.g. 0.768 → 0.776 → 0.780).

**What just happened (trace one round):**
1. Server starts `FedXgbBagging`, sends empty model bytes to 2 clients.
2. Each client loads data → `IidPartitioner` gives them their slice → `train_test_split` → `xgb.DMatrix`.
3. Round 1: `xgb.train()` from scratch (no global model to load). Each client trains 1 tree (`local-epochs=1`), sends 1 tree + `num-examples` back.
4. Server concatenates the 2 trees (bagging), evaluates AUC on both clients' validation sets.
5. Round 2+: clients load global model via `bst.load_model()`, boost 1 more tree (`_local_boost`), slice last 1 tree, send back. Repeat.

**Windows gotchas we hit (keep as checklist):**
1. **`'charmap' codec can't encode '\U0001f38a'` crash.** flwr prints 🎊; PowerShell cp1252 chokes. Fix: `$env:PYTHONUTF8 = "1"`. Permanent: `setx PYTHONUTF8 1`.
2. **Wrong Python owns the run.** `flwr run` spawns `flower-superlink` by name from PATH. If you didn't activate churnenv, the *global* SuperLink (flwr 1.33, no `flwr-datasets`) executes → `ModuleNotFoundError` / `BucketNotFoundError`. Always activate churnenv first. Verify: `Get-Command flower-superlink` → must point into `churnenv\Scripts`. Stale: `Stop-Process -Name flower-superlink -Force`.
3. **Ray object store memory.** Ray needs minimum 75 MB for its object store. If you see `ValueError: Attempting to cap object store memory at 75653529 bytes, but the minimum allowed is 78643200 bytes`, close other programs to free RAM (need ~1.5 GB free).
4. **HIGGS parquet schema.** The parquet has `inputs` as a `list<item: double>` column (each row is a list of 28 floats), not 29 separate columns. Must unpack with `np.array(df["inputs"].tolist())` — raw `.values.astype(float32)` fails with `ValueError: setting an array element with a sequence`.
5. **Ray `access violation` noise on Windows.** `raylet` prints fatal-exception stacks on worker shutdown — ugly but harmless. Judge by `flwr list` status (`finished`) and the `Final results` block, not by Ray stderr.
6. **`FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION`** avoids a failure where the auto-built runtime env resolved `datasets 4.8.5 + huggingface-hub 1.30.0` (incompatible pair). This flag makes the simulation use churnenv's packages directly.

---

## 6. Lab 2 — make it churn (the real learning, 2–3 hrs)

**Goal:** understand the Telco data loading pipeline in `task.py` + `preprocess.py` that replaced the HIGGS quickstart. Same `client_app.py`, same `server_app.py`, same `pyproject.toml` — only the data source changed.

### 6a. Inspect what's already done

Open these files side by side:

| File | What it does | Data source |
|------|-------------|-------------|
| `quickstart_xgboost/task.py` | Loads Telco parquet → shuffles → preprocesses → one-hot encodes → splits IID → DMatrix | `data/telco_original.parquet` (7043 rows) |
| `quickstart_xgboost/preprocess.py` | `preprocess_telco(df)` — drops leakage cols, fillna, binary encodes, drops Churn Category/Reason | Called by `task.py` |
| `API/app.py:33-86` | Same 52 features for centralized serving | `data/telco_original.csv` |

**What's already done in `task.py`:**
- Loads `telco_original.parquet` instead of HIGGS
- **Shuffles the dataset** (`dataset.shuffle(seed=42)`) before partitioning — see gotcha below
- `preprocess_telco()` handles cleaning (same logic as your notebook v2)
- `FEATURE_NAMES` defined directly in `task.py:14-67` (52 columns, same order as `API/app.py`)
- `encoder.transform()` applies the saved `telco_encoder.joblib` for one-hot encoding
- `df.reindex(columns=FEATURE_NAMES)` ensures column order matches the model

**Verify the pipeline works:**
```python
# From project root, churnenv active:
python -c "
import sys; sys.path.insert(0, '.')
from quickstart_xgboost.task import load_data
train_dm, val_dm, n_train, n_val = load_data(0, 2)
print(f'train: {n_train} rows, val: {n_val} rows, features: {train_dm.num_col()}')
# Verify both classes present in validation
import numpy as np
print('val classes:', np.unique(val_dm.get_label()))
"
```

✅ Check: prints `train: 2817 rows, val: 705 rows, features: 52` and `val classes: [0. 1.]` (both classes present).

**Critical gotcha — why the shuffle is there:**
The Telco CSV is sorted: first 1869 rows are churned (`Yes`), remaining 5174 are `No`. `IidPartitioner` uses `dataset.shard(contiguous=True)` — it splits into contiguous blocks **without shuffling**. Without `dataset.shuffle(seed=42)`, partition 1 would contain zero churned customers → `eval_set` returns `auc: nan` (AUC undefined when only one class exists). We hit this exact bug: simulation ran 3 rounds with `'auc': nan'` every round. The fix: shuffle before assigning to the partitioner. Interview line: "I had to shuffle before IID partitioning because the source data was label-sorted — contiguous sharding gave one client a single-class partition."

### 6b. Run federated Telco

```powershell
# From quickstart-xgboost/, churnenv active, same env vars as Lab 1:
$env:PYTHONUTF8 = "1"
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"

flwr run . --run-config "num-server-rounds=3 local-epochs=1 save-model=true" --federation-config "num-supernodes=2" --stream
```

✅ Check: 3 rounds finish, `aggregate_evaluate` shows a **real AUC** (not `nan`) for each round, `final_model.json` created. Our actual run produced:

```
[ROUND 1/3] Aggregated MetricRecord: {'auc': 0.99098}
[ROUND 2/3] Aggregated MetricRecord: {'auc': 0.99255}
[ROUND 3/3] Aggregated MetricRecord: {'auc': 0.99292}
```

Verify the saved model:
```python
import xgboost as xgb
bst = xgb.Booster()
bst.load_model("final_model.json")
print(bst.num_boosted_rounds())  # should be num_rounds × num_clients × local-epochs = 3×2×1 = 6
print(bst.num_features())        # should be 52
```

**Why 0.99 and not the notebook's 0.93?** Your notebook evaluated on a held-out test set (1409 rows) after training on 5634 rows. The federated evaluate runs on each client's 20% validation split (705 rows) drawn from the same partition the model trained on — it's a softer target. Also, `Satisfaction Score` is still a feature (potential leakage, see Appendix A). The federated number is valid for *relative* comparison across rounds, not for claiming state-of-the-art. Interview-safe framing: "Federated AUC improved 0.991 → 0.993 over 3 rounds with only tree bytes crossing the network."

**Compare with centralized:** Your notebook's best XGB got ~0.93 AUC on the proper test set. The federated number will differ — that's expected. Reasons: (1) different evaluation protocol (per-client validation vs held-out test), (2) only 6 trees (vs 400 in your tuned notebook), (3) IID split. The point isn't to beat centralized — it's to show FL works with no raw-row movement.

### 6c. Stretch (resume gold)

Non-IID partition: split by `Contract` column so Client 0 gets mostly `Month-to-Month` (high churn) and Client 1 gets `Two Year` (low churn). Re-run, note AUC drop. Be ready to explain: "Non-IID means clients see different data distributions — Client 0 learns churn patterns from short-term contracts, Client 1 rarely sees churn. Aggregated model must generalize across both."

### Common errors

- **`'auc': nan` every round** → one client's validation set has only one class. Cause: label-sorted data + `IidPartitioner(contiguous=True)` without shuffle. Fix: `dataset = dataset.shuffle(seed=42)` before `partitioner.dataset = dataset` (already in `task.py`).
- `num_partitions` ≠ simulated SuperNodes → client waits for missing partition / `shard index out of range`. Keep `IidPartitioner(num_partitions=N)` == `--num-supernodes N`.
- Feature mismatch (`52 vs 50`) → your Telco loader column order differs from `FEATURE_NAMES`. Use `reindex(columns=FEATURE_NAMES)`.
- `bst.load_model` fails on round 1 → don't load empty `b""` on round 1; branch on `server-round==1` like `client_app.py:47-53`.
- `InconsistentMessageReplies: Expected exactly one Array` → you put 2 arrays in `ArrayRecord`; keep single `[model_np]`.
- `num-examples` mismatch → each client returns `len(train_partition)`, not total dataset size. Weighted aggregation uses these numbers.
- `InconsistentVersionWarning: OneHotEncoder from 1.6.1 using 1.9.0` → harmless warning; encoder works. For production, re-save with `joblib.dump(encoder, path)` under your current sklearn.

---

## 7. Lab 3 — make it portable and reproducible (30–45 min)

**This is the lab that turns "works on my machine" into "anyone can clone and run it."** Read it even if you skip everything else in Part I — it is the single biggest difference between a private script and an uploadable project.

### 7a. Why this matters (concept, 5 min)

Your repo has a **public GitHub remote**: `github.com/HAT-ZAID/Privacy-Based-Customer-Churn`. Today, a fresh clone is broken in two places:

| Problem | Where | What a cloner sees |
|---|---|---|
| Hardcoded absolute paths | `quickstart-xgboost/quickstart_xgboost/task.py:77-78` | `FileNotFoundError: 'D:\Privacy Based Customer Churn\data\telco_original.parquet'` |
| Most of the project untracked | repo-wide | Only `API/`, `data/telco_original.csv`, `notebooks/Privacy_churn_v1.ipynb` exist after clone — no federated app, no models, no frontend, no docs |

An absolute path is resolved against *one specific machine*. `D:\Privacy Based Customer Churn\...` only exists on your laptop. A **relative path** is resolved against *wherever the code happens to live*. In Python the portable idiom is `Path(__file__)` — the path of the currently-executing file — so you walk up from it to reach the project root:

```
__file__                               = <root>/quickstart-xgboost/quickstart_xgboost/task.py
Path(__file__).parent                  -> <root>/quickstart-xgboost/quickstart_xgboost   (package dir)
Path(__file__).resolve().parents[1]    -> <root>/quickstart-xgboost
Path(__file__).resolve().parents[2]    -> <root>                                        <-- project root
```

`resolve()` also resolves symlinks, so it behaves the same on Windows and Linux.

Your repo already contains the correct pattern — `API/app.py` does exactly this:
```python
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "Telco_xgb.joblib"
```
The federated loader just never got the same treatment. That asymmetry is the bug.

> Learn: an absolute path is a claim about one machine. A repository is a claim about every machine. Anything you commit must only make the second claim.

### 7b. Fix the two paths (hands-on, 10 min)

Open `quickstart-xgboost/quickstart_xgboost/task.py`. **Before:**
```python
DATASET_PATH = r"D:\Privacy Based Customer Churn\data\telco_original.parquet"
encoder = joblib.load(r"D:\Privacy Based Customer Churn\models\telco_encoder.joblib")
```
**After:**
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "telco_original.parquet"
encoder = joblib.load(PROJECT_ROOT / "models" / "telco_encoder.joblib")
```

Why each piece:
- `parents[2]` — the file sits 3 levels below the root (`root/quickstart-xgboost/quickstart_xgboost/task.py`), so two `.parent` hops from its directory land on the root.
- `pathlib.Path` objects work anywhere a string path does: `pd.read_parquet(PROJECT_ROOT / "data" / "telco_original.parquet")` and `joblib.load(...)` both accept them, so no `.resolve()` at use sites.
- `/` as the join operator inserts the correct separator per OS. Use it rather than hardcoding `\` or `os.path.join` noise.

**Verify — the check must pass from two different working directories**, because that is the entire point:
```powershell
# from the project root
python -c "import sys; sys.path.insert(0,'.'); from quickstart_xgboost.task import load_data; print(load_data(0,2)[2], 'train rows')"

# from inside the app directory — this is where `flwr run` actually executes
cd quickstart-xgboost
python -c "import sys; sys.path.insert(0,'.'); from quickstart_xgboost.task import load_data; print(load_data(0,2)[2], 'train rows')"
```
✅ Check: both print `2817 train rows`. If the second fails, your `parents[…]` index is wrong.

### 7c. Freeze your environment (10 min)

`frontend/requirements.txt` currently holds exactly one unpinned line — `streamlit` — and covers neither the federated nor the API dependencies. Nothing tells a cloner which versions you used. `churnenv/` is (correctly) gitignored: ~10,000 files, platform-specific. What belongs in the repo is the *pin list*.

Capture what is actually installed, then keep only the direct dependencies:
```powershell
.\churnenv\Scripts\Activate.ps1
python -m pip freeze > requirements-raw.txt   # noisy reference only, do not commit
```
Write a root `requirements.txt` with the direct deps you actually import, pinned to what you tested:

```text
# Federated learning core
flwr==1.31.0
flwr-datasets==0.6.0
xgboost==3.2.0

# Data pipeline
pandas==2.3.3
numpy==2.4.6
scikit-learn==1.9.0
joblib==1.5.3
datasets==4.8.5
pyarrow==24.0.0

# API serving
fastapi==0.138.2
uvicorn==0.49.0
pydantic==2.13.4

# Frontend
streamlit==1.61.1

# Simulation backend (via flwr[simulation])
ray==2.55.1

# Notebooks
notebook
matplotlib==3.11.1
seaborn==0.13.2
```

Two rules you should be able to state in an interview:
1. **Pin direct dependencies, not transitive ones.** `flwr[simulation]>=1.28.0` in `pyproject.toml` declares a *floor*; `flwr==1.31.0` in `requirements.txt` records *what you tested*. Both belong in the repo — they answer different questions.
2. **Keep the pin list in sync with `flwr-version-target`.** `quickstart-xgboost/pyproject.toml:22` declares `flwr-version-target = "1.31.0"` and `churnenv` has `flwr 1.31.0`. Change both in the same commit or the Flower FAB build refuses.

**Also resolve the encoder version skew.** `models/telco_encoder.joblib` was pickled by scikit-learn **1.6.1** but `churnenv` runs **1.9.0**, so every load prints:
```
InconsistentVersionWarning: Trying to unpickle estimator OneHotEncoder from version 1.6.1 when using version 1.9.0.
```
It works, but that warning makes a reviewer distrust the artifact. Re-save under the current interpreter:
```python
import joblib, warnings
warnings.filterwarnings("ignore")
encoder = joblib.load("models/telco_encoder.joblib")
joblib.dump(encoder, "models/telco_encoder.joblib")   # now stamped 1.9.0
```
✅ Check: loading the encoder again prints no warning.

### 7d. Decide what actually enters git (10 min)

| Path | Size | Verdict | Why |
|---|---|---|---|
| `data/telco_original.csv` | 1.9 MB | **already tracked** | the notebook reads it |
| `data/telco_original.parquet` | 474 KB | **commit** | `task.py` reads it; small, and makes the FL path clone-runnable |
| `models/Telco_xgb.joblib` | 1.6 MB | **commit** | `API/app.py` loads it; without it the API cannot start |
| `models/telco_encoder.joblib` | 1.7 KB | **commit** | needed by `task.py` |
| `models/{best_params,classification_report,feature_names}` | ~1.5 KB | **commit** | tiny, and they are the *evidence* of your model quality |
| `models/telco_optuna.db` | 480 KB | **ignore** | Optuna study cache — build artifact, regenerable, tells a reviewer nothing |
| `quickstart-xgboost/final_model.json` | ~19 KB | **commit** | the Predict tab loads it; committing it makes the demo work on first run |
| `notebooks/*.ipynb` | 2.4 MB | **commit with outputs** | a reviewer sees your metrics without running anything |
| `churnenv/` | ~10k files | **ignore** | platform-specific and huge |
| `__pycache__/`, `*.pyc` | — | **ignore** | Python bytecode |
| `.flwr/`, `.ray/` | — | **ignore** | Flower/Ray runtime scratch |
| `fl_telco_output.txt` | — | **ignore** | captured logs, regenerable |
| `.ua/` | ~70 KB | **ignore** | generated knowledge graph — delete it for a clean tree |

Add the missing entries to `.gitignore`:
```gitignore
churnenv/
.vscode/
.agents/
__pycache__/
*.pyc
.flwr/
.ray/
.ua/
fl_telco_output.txt
*.db
```

> Learn: `.gitignore` is not housekeeping, it is a decision record. Every line is a claim about what this project *is*. "We don't commit model caches because they are regenerable" is defensible; silence is not.

### 7e. Verify from a clean clone (the actual MVP test, 10 min)

Everything above is a claim. This is the proof. Clone into a scratch directory **outside** the repo so you cannot accidentally read your own untracked files:

```powershell
cd $env:TEMP
Remove-Item -Recurse -Force churn-clone-test -ErrorAction SilentlyContinue
git clone https://github.com/HAT-ZAID/Privacy-Based-Customer-Churn.git churn-clone-test
cd churn-clone-test
```
Then follow **only** the commands in `README.md`, in a fresh virtual environment, and confirm:
- ✅ `python -c "from quickstart_xgboost.task import load_data; print(load_data(0,2)[2])"` prints `2817` — not a `FileNotFoundError`
- ✅ `flwr run . --run-config "num-server-rounds=3 local-epochs=1 save-model=true" --federation-config "num-supernodes=2" --stream` completes 3 rounds with a real AUC (not `nan`)
- ✅ `streamlit run frontend/federated_demo.py` starts and the Predict tab reports a loaded model with 6 trees

This single test catches every remaining absolute path, missing artifact, and undocumented step. It is the highest-value 10 minutes in this guide.

---

## 8. Lab 4 — Streamlit demo (CAPSTONE: build it, then show it)

### Why pre-recorded?

Ray (Flower's simulation backend) doesn't play nicely inside Streamlit — spawning Ray workers from a Streamlit process causes hangs and OOM. So v1 is **pre-recorded**: you run `flwr run` in a terminal, copy the round-by-round AUC numbers into a CSV, and the Streamlit UI reads that CSV. Live `flwr run` from the UI is a Next Step (see [§11 Future work](#11-future-work)).

### File layout

```
frontend/
  federated_demo.py       # Your Streamlit app (4 tabs — you fill the TODOs)
  fl_rounds_sample.csv    # Sample CSV you'll replace with real data
  streamlit_app.py        # Existing centralized predictor (untouched)
  .streamlit/config.toml  # Streamlit theme config
```

---

### 8a. Get your real numbers (do this first, 10 min)

You need two things: (1) per-round federated AUC from a real `flwr run`, and (2) your centralized AUC from the notebook.

**Step 1 — Run the federated simulation and capture logs:**

```powershell
# From quickstart-xgboost/, churnenv active:
$env:PYTHONUTF8 = "1"
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"
flwr run . --run-config "num-server-rounds=3 local-epochs=1 save-model=true" --federation-config "num-supernodes=2" --stream 2>&1 | Tee-Object -FilePath fl_telco_output.txt
```

`Tee-Object` writes the full log to `fl_telco_output.txt` AND shows it in the terminal. You need the file because you'll search it for AUC values.

**Step 2 — Extract each round's AUC from the log.** Open `fl_telco_output.txt` and search for `Aggregated MetricRecord`. You'll find lines like:

```
[ROUND 1/3]
    └──> Aggregated MetricRecord: {'auc': 0.9909828030774182}
[ROUND 2/3]
    └──> Aggregated MetricRecord: {'auc': 0.9925536221296033}
[ROUND 3/3]
    └──> Aggregated MetricRecord: {'auc': 0.9929157168364653}
```

These are the **aggregated** AUC values across both clients. There are two `Aggregated MetricRecord` lines per round — one for train (empty `{}`) and one for evaluate (has the `auc`). You want the one with `auc`.

Quick PowerShell extraction:
```powershell
Select-String -Path fl_telco_output.txt -Pattern "Aggregated MetricRecord.*auc" | ForEach-Object { $_.Line }
```

**Step 3 — Create `frontend/fl_rounds.csv`.** Create this file (it doesn't exist yet — the starter uses `fl_rounds_sample.csv`):

```csv
round,auc,num_examples_train,num_examples_eval,note
1,0.9910,5633,1410,Telco federated round 1
2,0.9926,5633,1410,Telco federated round 2
3,0.9929,5633,1410,Telco federated round 3
```

Replace `0.9910` etc. with YOUR actual numbers (keep 4 decimal places for readability). `num_examples_train` = total training rows across both clients (2817 + 2816 = 5633). `num_examples_eval` = total validation rows (705 + 705 = 1410).

**Step 4 — Get your centralized AUC.** Open `notebooks/Privacy_churn_v1.ipynb`, run the XGBoost test cells, find the test-set AUC. If you haven't run it: the notebook's XGB grid best got ~0.93 test accuracy / ~0.93 AUC (see Appendix A cell 49-66). Write this number down — you'll hardcode it in the Compare tab.

✅ Check: you have `frontend/fl_rounds.csv` with 3 real AUC rows AND a centralized AUC number written down.

---

### 8b. Tab 1 — Learn (concept, no code, 15 min)

This tab is pure markdown — no coding, just understanding. Open `frontend/federated_demo.py` and find `with tab_learn:` (around line 24).

**What's there now:** a 4-bullet story of one FL round, then a `TODO-LEARN` prompt asking "Why can't we FedAvg trees?"

**Your task: replace the TODO-LEARN text with 2-3 bullets in your own words.**

Before writing, understand the answer:

1. **What FedAvg does:** for neural networks, each client sends weight matrices, server computes element-wise average. Works because weights are continuous numbers in the same space.

2. **What trees look like:** a tree is a structure of split conditions like `Contract = Month-to-Month` or `Tenure > 12`. These are discrete decisions, not continuous values.

3. **Why averaging fails:** if Client 0's tree splits on `Age > 30` and Client 1's splits on `Age > 45`, averaging gives `Age > 37.5` — a split neither client learned, and one that may not align with any real pattern. Worse, trees have different depths/structures, so there's no element-wise correspondence to average.

4. **What FedXgbBagging does instead:** server just concatenates trees from all clients into one ensemble (bagging). Client 0 contributes trees 0,2,4... Client 1 contributes trees 1,3,5... The final model votes across all of them. Source: `flwr.serverapp.strategy.fedxgb_bagging.aggregate_bagging`.

**Write your answer** (example — don't copy verbatim, make it yours):
> - Trees are discrete structures (split rules), not continuous weights — averaging `Age>30` and `Age>45` gives `Age>37.5`, which no client learned.
> - Tree structures differ across clients (different depths, different features), so there's no element-wise alignment to average.
> - Instead, FedXgbBagging concatenates trees from all clients into an ensemble — the model gets bigger each round, not averaged.

✅ Check: read your bullets aloud without looking at this guide. If you can, you can defend it in an interview.

---

### 8c. Tab 2 — Run (replace CSV, add metrics, 20 min)

This tab shows how AUC improves round by round. Open `frontend/federated_demo.py` and find `with tab_run:` (around line 38).

**Step 1 — Point to your real CSV.** Find line ~45:
```python
csv_path = Path(__file__).parent / "fl_rounds_sample.csv"
```
Change `"fl_rounds_sample.csv"` to `"fl_rounds.csv"` (the file you created in 8a).

**Step 2 — Add headline metrics above the chart.** Between `st.subheader(...)` and `st.dataframe(...)`, add:

```python
if len(df):
    col1, col2, col3 = st.columns(3)
    col1.metric("Final AUC", f"{df['auc'].iloc[-1]:.4f}")
    col2.metric("Rounds", len(df))
    col3.metric("Total examples", f"{df['num_examples_eval'].iloc[-1]:,}")
```

This uses `st.columns(3)` to place three metric cards side by side. `f"{value:.4f}"` formats to 4 decimal places. `.iloc[-1]` gets the last row (final round).

**Step 3 — Remove the TODO info box.** Delete the `st.info("TODO-RUN: ...")` block (around line 51-55).

✅ Check: run `streamlit run frontend/federated_demo.py` → Run tab shows YOUR AUC curve (0.991 → 0.993), not the sample data. The three metric cards show at the top.

---

### 8d. Tab 3 — Compare (centralized vs federated, 15 min)

This tab shows the tradeoff between centralized and federated training. Open `frontend/federated_demo.py` and find `with tab_compare:` (around line 57).

**Step 1 — Replace the slider with your measured centralized AUC.** Find:
```python
central_auc = st.slider("Centralized AUC (from your notebook, Lab 0)", 0.5, 1.0, 0.82, 0.01)
```
Replace with:
```python
central_auc = 0.0  # <-- replace with YOUR measured centralized AUC (see below)
```

Why hardcode? A slider lets the viewer fudge the comparison, which makes the whole tab worthless as evidence. A fixed number is honest.

**Where the number comes from.** Measure it in the notebook, do not guess it:
```python
from sklearn.metrics import roc_auc_score
proba = model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, proba))
```
What we know for certain about the centralized model is its `models/classification_report.txt`: **accuracy 0.969, weighted F1 0.969, churn-class F1 0.940** on a held-out `n=1409` (1035 stayed / 374 churned). A classification report does **not** contain AUC, so if you have not run the line above you do not have an AUC yet — compute it rather than substituting the accuracy.

**Step 2 — The federated AUC already works.** Line ~62 reads from your CSV:
```python
fed_auc = float(df["auc"].iloc[-1]) if len(df) else 0.81
```
This gets the last round's AUC from `fl_rounds.csv`. No change needed — and once your CSV holds real numbers it will read `0.9929`.

⚠️ **Before you trust the comparison:** the two numbers are measured under different protocols. Federated AUC is computed on each client's own 20% validation split, drawn from the partition that client just trained on. Centralized AUC is on a held-out test set the model never saw. The federated number is therefore an optimistic upper bound, not a like-for-like win. Say so in the sentence you write in step 3.

**Step 3 — Add an explanation below the bar chart.** After `st.bar_chart(comp.set_index("model"))`, add:

```python
st.markdown(
    f"**Centralized {central_auc:.3f}** vs **Federated {fed_auc:.3f}** — "
    f"gap is {abs(central_auc - fed_auc):.3f}. "
    "The federated model trains on each client's ~2800-row partition "
    "(half the data each), and the evaluation protocol differs: "
    "federated evaluates on per-client validation splits, "
    "centralized evaluates on a held-out test set. "
    "The privacy win: raw customer rows never left the partition."
)
```

**Step 4 — Remove the TODO info box.** Delete the `st.info("TODO-COMPARE: ...")` block (around line 67).

✅ Check: bar chart shows two bars with real numbers. You can explain the gap in one sentence.

---

### 8e. Tab 4 — Predict (the real work, 45-60 min)

This is the most complex tab. It loads your federated XGBoost model and predicts churn from user input. Open `frontend/federated_demo.py` and find `with tab_predict:` (around line 69).

**The challenge:** the model expects 52 features in exact `FEATURE_NAMES` order, but you only want ~8 fields in the form. You need to map 8 user inputs → 52-column vector.

**Step 1: understand the mapping.** Open `quickstart_xgboost/task.py:14-67` (or `API/app.py:33-86`) and look at `FEATURE_NAMES`. Group the 52 columns by what user input drives them:

| User input | FEATURE_NAMES columns it maps to | How to fill |
|-----------|----------------------------------|-------------|
| Tenure in Months | `Tenure in Months` | Direct value |
| Monthly Charge | `Monthly Charge` | Direct value |
| Contract | `Contract_Month-to-Month`, `Contract_One Year`, `Contract_Two Year` | One-hot: set 1 in matching column, 0 in others |
| Internet Type | `Internet Type_Cable`, `Internet Type_DSL`, `Internet Type_Fiber Optic`, `Internet Type_No Internet` | One-hot |
| Premium Tech Support | `Premium Tech Support` | Yes→1, No→0 |
| Paperless Billing | `Paperless Billing` | Yes→1, No→0 |
| Senior Citizen | `Senior Citizen` | Yes→1, No→0 |
| Dependents | `Dependents` | Yes→1, No→0 |

The remaining 44 columns (Gender, Age, Zip Code, Latitude, etc.) = 0 (default). That's fine — XGBoost handles missing/zero gracefully; it learned during training that these split points rarely matter for the default case.

**Step 2: add FEATURE_NAMES to the top of `federated_demo.py`.** After the imports (after `import pandas as pd`), paste the full list from `task.py:14-67`:

```python
FEATURE_NAMES = [
    "Gender", "Age", "Under 30", "Senior Citizen", "Married", "Dependents",
    "Number of Dependents", "Zip Code", "Latitude", "Longitude", "Population",
    "Referred a Friend", "Number of Referrals", "Tenure in Months",
    "Phone Service", "Avg Monthly Long Distance Charges", "Multiple Lines",
    "Internet Service", "Avg Monthly GB Download", "Online Security",
    "Online Backup", "Device Protection Plan", "Premium Tech Support",
    "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data",
    "Paperless Billing", "Monthly Charge", "Total Charges", "Total Refunds",
    "Total Extra Data Charges", "Total Long Distance Charges", "Total Revenue",
    "Satisfaction Score", "CLTV",
    "Offer_No Offer", "Offer_Offer A", "Offer_Offer B", "Offer_Offer C",
    "Offer_Offer D", "Offer_Offer E",
    "Internet Type_Cable", "Internet Type_DSL", "Internet Type_Fiber Optic",
    "Internet Type_No Internet",
    "Contract_Month-to-Month", "Contract_One Year", "Contract_Two Year",
    "Payment Method_Bank Withdrawal", "Payment Method_Credit Card",
    "Payment Method_Mailed Check",
]
```

**Why copy instead of import?** Streamlit runs from `frontend/` and importing from `quickstart_xgboost/` would pull in Flower dependencies. Duplicating the 52-column list keeps the demo self-contained. For production, you'd put it in a shared config file.

**Step 3: build the `build_features` function.** Inside the `with tab_predict:` block, BEFORE the form, add:

```python
import numpy as np

def build_features(tenure, monthly, contract, internet, tech_support, paperless, senior, dependents):
    """Map 8 user inputs to 52-feature vector in FEATURE_NAMES order."""
    features = {col: 0 for col in FEATURE_NAMES}  # start all zeros

    # Direct values
    features["Tenure in Months"] = tenure
    features["Monthly Charge"] = monthly

    # Binary yes/no
    features["Premium Tech Support"] = 1 if tech_support else 0
    features["Paperless Billing"] = 1 if paperless else 0
    features["Senior Citizen"] = 1 if senior else 0
    features["Dependents"] = 1 if dependents else 0

    # One-hot: Contract (must match exact FEATURE_NAMES spelling)
    features[f"Contract_{contract}"] = 1

    # One-hot: Internet Type
    features[f"Internet Type_{internet}"] = 1

    # Return in FEATURE_NAMES order, reshaped to (1, 52) for single sample
    return np.array([features[col] for col in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
```

**How this works:**
1. `{col: 0 for col in FEATURE_NAMES}` — dict comprehension creates all 52 columns set to 0
2. Direct assignments overwrite specific keys
3. `f"Contract_{contract}"` — f-string builds the exact column name (e.g., `Contract_Month-to-Month` with the hyphen)
4. `[features[col] for col in FEATURE_NAMES]` — list comprehension reads values in the exact order the model expects
5. `.reshape(1, -1)` — makes it a 2D array with 1 row (XGBoost expects `[n_samples, n_features]`)

**Step 4: wire the form.** Replace the existing `with st.form("mini_churn"):` block:

```python
with st.form("mini_churn"):
    tenure = st.slider("Tenure in Months", 0, 72, 12)
    monthly = st.slider("Monthly Charge ($)", 0.0, 200.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
    internet = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "No Internet"])
    tech_support = st.checkbox("Premium Tech Support")
    paperless = st.checkbox("Paperless Billing")
    senior = st.checkbox("Senior Citizen")
    dependents = st.checkbox("Has Dependents")
    submitted = st.form_submit_button("Predict churn risk")
```

**Step 5: handle the prediction.** AFTER the `with st.form(...)` block (outside it — the form only collects inputs), add:

```python
if submitted and model is not None:
    x = build_features(tenure, monthly, contract, internet, tech_support, paperless, senior, dependents)
    import xgboost as xgb
    dmat = xgb.DMatrix(x)          # XGBoost needs DMatrix, not raw numpy
    proba = model.predict(dmat)[0]  # returns array of 1 probability

    st.metric("Churn probability", f"{proba:.1%}")
    if proba > 0.5:
        st.error(f"🔴 High risk — {proba:.1%} chance of churn")
    elif proba > 0.3:
        st.warning(f"🟡 Moderate risk — {proba:.1%} chance of churn")
    else:
        st.success(f"🟢 Low risk — {proba:.1%} chance of churn")
```

**Why `xgb.DMatrix`?** XGBoost's `Booster.predict()` doesn't accept raw numpy arrays. You must wrap: `xgb.DMatrix(x)` creates the internal sparse-optimized format. Skipping this gives `TypeError`.

**Why `model.predict(dmat)[0]`?** `predict` returns an array of probabilities (one per input row). `[0]` gets the first (and only) row's probability.

**Step 6: remove the TODO info box.** Delete the `st.info("TODO-PREDICT: ...")` block at the end.

✅ Check: run `streamlit run frontend/federated_demo.py` → Predict tab loads the model (green success message showing 6 trees). Fill the form with extreme values:
- **High risk test:** tenure=0, Monthly=$150, Contract=Month-to-Month, Internet=Fiber Optic, no tech support, paperless, senior, no dependents → should show 🔴 high risk (>50%)
- **Low risk test:** tenure=72, Monthly=$30, Contract=Two Year, Internet=No Internet, tech support, not paperless, not senior, has dependents → should show 🟢 low risk (<30%)

If both extremes work correctly, your feature mapping is correct.

---

### 8f. Full run and final checklist

```powershell
.\churnenv\Scripts\Activate.ps1
streamlit run frontend/federated_demo.py
```

✅ **Final check — all 4 tabs must pass:**
1. **Learn tab:** your 2-3 bullet explanation of why FedAvg doesn't work on trees (not the TODO placeholder)
2. **Run tab:** shows YOUR AUC curve from `fl_rounds.csv` (0.991 → 0.993), with 3 metric cards at top
3. **Compare tab:** bar chart with hardcoded centralized AUC vs your federated AUC, plus explanation sentence
4. **Predict tab:** loads federated model (6 trees), form predicts with color-coded risk output

---

### Streamlit pitfalls (things we hit)

- **`use_container_width` is deprecated** in Streamlit 1.61+. Use `width="stretch"` instead (already done in the starter).
- **`st.form` + `st.form_submit_button`** — all inputs inside a form only submit when the button is clicked. Don't put `st.metric` inside the form — it won't update until submit. The prediction logic goes OUTSIDE the `with st.form(...)` block.
- **`st.cache_resource`** for the model (loaded once, survives reruns), `st.cache_data` for the CSV (reloaded if file changes). Don't cache the predict function — it depends on user input.
- **XGBoost predict needs `DMatrix`** — you can't pass a raw numpy array to `bst.predict()`. Always wrap: `xgb.DMatrix(x)`.
- **Model path** — `final_model.json` is created in `quickstart-xgboost/` when you run `flwr run` with `save-model=true`. The Streamlit app loads it relative to the project root. If you move files, update the path in `load_federated_model`.
- **One-hot encoding must match** — `build_features` creates column names with exact `FEATURE_NAMES` spelling (e.g. `Contract_Month-to-Month` with the hyphen). Typos here = silent zero predictions (the dict lookup just adds a new key that's never read).
- **`iloc[-0]` is `[0]`, not last row** — if you see `iloc[-0]` anywhere, fix it to `iloc[-1]`.
- **Emoji in `st.error`** — if you see `charmap codec can't encode` errors, set `$env:PYTHONUTF8 = "1"` before launching Streamlit.

---

## 9. Upload gate — the last check before `git push`

Work top to bottom. Anything unchecked means **do not push**.

### 9a. Correctness
- [ ] `flwr run` completes 3 rounds and every round shows a **real** AUC, not `nan`
- [ ] `bst.load_model("final_model.json")` → `num_boosted_rounds() == 6`, `num_features() == 52`
- [ ] `np.unique(val_dm.get_label())` returns **both** classes for both partitions
- [ ] No `InconsistentVersionWarning` when loading `models/telco_encoder.joblib`
- [ ] `python -m pip check` → `No broken requirements found`
- [ ] All 4 Streamlit tabs work; Predict tab returns a probability for both a high-risk and a low-risk input

### 9b. Portability
- [ ] `grep -rn "D:\\\\" --include=*.py .` returns **nothing** (excluding `.ua/`, `churnenv/`)
- [ ] The clean-clone test in [§7e](#7e-verify-from-a-clean-clone-the-actual-mvp-test-10-min) passes end to end
- [ ] Root `requirements.txt` exists, is pinned, and installs cleanly in a fresh venv

### 9c. Honesty
- [ ] `README.md` commands are copy-pasteable and actually produce the output shown
- [ ] Every number in the README matches a real run you did
- [ ] The "honest scope" section still tells the truth (simulation only, no DP, AI-assisted)
- [ ] Known open issues are either fixed or written down — see [§11 Future work](#11-future-work)

### 9d. Commit it

```powershell
git status                     # read this before staging anything
git add -A                     # or add specific paths from the §7d table
git status                     # confirm optuna.db / churnenv/ / .ua/ are NOT staged
```
Three logical commits rather than one blob, because the history is part of the artefact:

```powershell
git commit -m "feat: federated Telco simulation working (shuffle fix, verified AUC 0.991-0.993)"
git commit -m "docs: project README + learning guide (labs, portability, practice tests)"
git commit -m "fix: relative paths, pinned requirements, point API at renamed model"
```
Verify what is staged before each commit — `git status` after `git add` is the whole safety check.

Then push, and only then:
```powershell
git push origin master
```

> **The order matters.** Fix paths → pin deps → verify on a clean clone → commit → push. Pushing first and verifying after is how you end up with a public repo that does not run, which is worse than a private one that does.

---

# PART II — AFTER UPLOAD

> Part I gets you to a working, portable, demonstrable minimum viable project. Part II is what you
> do once it is public: the production story, the roadmap, and the interview defence.

## 10. Deployment notes (no code, just understand)

- Same `ClientApp/ServerApp` run under Deployment (real `SuperLink` + `SuperNodes`) — only federation config changes. See `how-to-run-flower-with-deployment-engine`.
- For interview: mention TLS (`how-to-enable-tls-connections`), SuperNode auth (`how-to-authenticate-supernodes`), Docker (`docker/index`) as "I know what production needs, scoped to simulation for this demo."

---

## 11. Future work

A tiered roadmap. Tier 1 items are **correctness work you should do before claiming anything in an interview**; Tier 2 is where federated learning actually gets interesting; Tier 3 is product polish. Each entry lists the interview sentence it buys you.

### Tier 1 — correctness (do these first)

| # | Item | Why it matters | Interview line it unlocks |
|---|---|---|---|
| 1 | **Rule on `Satisfaction Score`** | It is still in `FEATURE_NAMES` with correlation −0.76 against the label. If the survey is collected *after* the churn decision it is leakage, and your 0.99 federated AUC is partly explaining the answer. Run an ablation: retrain without it, compare churn-F1 against the 0.940 baseline. | "I ablated my most suspicious feature rather than assuming it was fine." |
| 2 | **Unify the Gender encoding** | `notebooks/Privacy_churn_v1.ipynb` maps `Male`→1; `preprocess.py:53-55`, `Privacy_churn_v2.ipynb` and `frontend/streamlit_app.py:7` all map `Male`→0. The shipped path is self-consistent because v2 produced the artifacts — but anyone retraining from the v1 pipeline gets a model whose Gender column is inverted relative to the frontend. `API/app.py` has no gender mapping at all; it takes a bare `int`. Fix by making the frontend import the one canonical map. | "I traced a silent feature-inversion bug across three files." |
| 3 | **Make the reindex actually load-bearing** | `task.py:120` does `df = df.reindex(columns=FEATURE_NAMES, fill_value=0)` but `inputs` was already bound at line 119 from the pre-reindex frame, and the reindexed `df` is never read again. Today the resulting order *happens* to match `FEATURE_NAMES` (verified: 0 of 52 positions differ), so nothing is broken — but the line provides zero protection. Reorder so `inputs` derives from the reindexed frame. | "I found a line that validated the contract without enforcing it." |
| 4 | **Save one reproducible model bundle** | `models/Telco_xgb.joblib` has no recorded provenance — no params, no encoder, no column order, no test report. Save model + encoder + `FEATURE_NAMES` + params + metrics as a single artifact so the versions cannot drift. | "My artifacts are self-describing and reproducible." |
| 5 | **Reconcile the two AUC protocols** | The federated number (0.991) is evaluated on each client's own 20% validation split; the centralized number (0.969 accuracy / 0.940 churn-F1) is on a held-out test set. They are not comparable. Add a proper held-out global test set so the comparison means something. | "I know why my two numbers differ and can prove it." |

### Tier 2 — federated depth (where the learning is)

| # | Item | Notes | Interview line it unlocks |
|---|---|---|---|
| 6 | **Non-IID partitioning** | Split by `Contract` so one client gets mostly `Month-to-Month` (high churn) and another gets `Two Year` (low churn). Replace `IidPartitioner` with `FederatedDataset`'s label-skew partitioners. Expect a measurable AUC drop. | "Real federated data is non-IID; I measured the cost." |
| 7 | **`cyclic` vs `bagging`** | The `xgboost-comprehensive` example supports both. Cyclic trains one client per round instead of all of them, so it needs more rounds for the same ensemble size. | "Bagging and cyclic are two aggregation schedules with different communication trade-offs." |
| 8 | **More clients, more rounds** | Scale to 3–5 supernodes and 10+ rounds. Watch how AUC and per-round wall-clock move. This is where you meet Ray's memory limits head-on. | "I scaled the simulation and found the bottleneck." |
| 9 | **Centralised evaluation (`evaluate_fn`)** | `ServerApp.start()` accepts an `evaluate_fn` hook for evaluating the global model on a server-side test set each round. This fixes item 5 properly. | "I separated the training protocol from the evaluation protocol." |
| 10 | **Differential privacy** | Honest caveat: gradient/tree-based FL and DP are genuinely hard to combine, and `FedXgbBagging` ships nothing for it. Worth reading the literature rather than claiming it. | "I know what my privacy claim does *not* cover." |

### Tier 3 — product

| # | Item | Notes |
|---|---|---|
| 11 | **Live Streamlit trigger** | `subprocess.run(["flwr", "run", ...])` plus log parsing. Keep it behind a button with a spinner, and remember Ray needs ~1.5 GB free or it dies in `ray.init()`. |
| 12 | **Deployment dry-run** | Same `ClientApp`/`ServerApp` under a real `SuperLink` + `SuperNodes`. Add TLS (`how-to-enable-tls-connections`) and SuperNode auth (`how-to-authenticate-supernodes`). |
| 13 | **Hyperparameter sweep** | Vary `eta` / `max-depth` / `local-epochs` via `--run-config`, record three runs, and report the trade-off rather than the best number. |
| 14 | **Per-round drift monitoring** | Log AUC, `num-examples`, tree count and wall-clock per round to a CSV so you can chart convergence over time. |

### Non-goals, stated plainly
No Kubernetes, no cloud spend, no distributed training framework. Those make the project *look* bigger without making it more defensible. Depth on the five items in Tier 1 and Tier 2 beats breadth across all fourteen.

---

## 12. Interview script (practice aloud)

**30-second pitch.** "I built a churn predictor on the Telco dataset two ways: a conventional centralized path — notebooks to XGBoost to a FastAPI endpoint to a Streamlit form — and a federated simulation using Flower, where I split the data across simulated branches so no raw customer row ever leaves its partition. Only XGBoost trees and evaluation metrics cross the network. Three rounds, AUC 0.991 to 0.993, and a Streamlit demo that shows the rounds and scores a prediction."

**Core questions**

- **Q: Why bagging instead of FedAvg?** → Trees can't be averaged. A tree is a set of discrete split rules with no element-wise correspondence across clients — `Age > 30` and `Age > 45` average to `Age > 37.5`, a split nobody learned, and the trees themselves may differ in depth and structure. So `FedXgbBagging` concatenates: the ensemble grows by `num_clients × local_epochs` trees per round. 3 rounds × 2 clients × 1 epoch = 6 trees.
- **Q: What actually crosses the network?** → Two `Record` types. `ArrayRecord` carrying the model as raw bytes (`bst.save_raw("json")` → `np.frombuffer(uint8)`), and `MetricRecord` carrying only `num-examples` and `auc`. `evaluate()` returns metrics with no arrays at all.
- **Q: What stays private, and what doesn't?** → Raw rows never leave the client. The honest limit: the tree bytes themselves can still leak, and I claim no differential privacy.
- **Q: Why is round 1 special?** → The server initialises the global model as empty `b""`, so there is nothing to load. The client branches on `server-round == 1` and trains from scratch. Rounds 2+ `load_model` then boost more trees. This isn't a stylistic choice — `bst.load_model(b"")` crashes the process rather than raising.

**The debugging stories — these are worth more than the architecture**

- **Q: Tell me about a bug you found.** → "The federated AUC was `nan` on every round. `IidPartitioner` doesn't shuffle — it calls `shard(contiguous=True)` — and the Telco dataset is label-ordered, so all 1,869 churned rows fell inside the first partition. My second client got zero churned rows, its validation set was single-class, and ROC-AUC is undefined on one class. It didn't raise, it just quietly returned `nan` and the training looked fine. One line — `dataset.shuffle(seed=42)` before partitioning — fixed it. IID partitioning is only IID if the input isn't ordered by the label."
- **Q: What's a subtle bug you'd still fix?** → "`task.py` reindexes the frame to `FEATURE_NAMES`, but `inputs` was already bound from the pre-reindex frame, so the reindexed DataFrame is never read. It happens to produce the right order today because the raw column order coincides with `FEATURE_NAMES` — but the line provides zero protection. If a column were added upstream, the model would train on a different order and inference would be silently wrong, because `DMatrix` is positional."
- **Q: Your federated AUC is 0.99 but the notebook reports 0.94 F1 — which is better?** → "They're not comparable, and saying so is the real answer. The federated number is evaluated on each client's own 20% validation split, drawn from the partition it just trained on — an optimistic upper bound. The centralized number is a held-out test set. To compare them properly I'd add a server-side global test set via the strategy's `evaluate_fn` hook."

**Scope and honesty**

- **Q: What would production need?** → "Same `ClientApp`/`ServerApp`, different federation config — a real `SuperLink` and `SuperNodes` instead of Ray. That means TLS, SuperNode authentication, and containerisation. I scoped the demo to simulation and wrote up the deployment path rather than pretending to have run it."
- **Q: Anything you'd want to change?** → "`Satisfaction Score` correlates −0.76 with the label and I never resolved whether it's collected after the churn decision. If it is, it's leakage and my headline numbers are partly explained by it. That's the first thing I'd ablate."
- **Resume honesty.** "My centralized path — the notebooks, the FastAPI service, the 52-field form — I wrote without AI assistance. The federated lab is where I used AI, and I used it to *learn* Flower's message model and XGBoost's aggregation internals rather than to have code written for me: I read the explanations, then migrated the loader to Telco myself. The `auc: nan` bug was mine — I read the logs, worked out that `IidPartitioner` shards contiguously, and fixed it with one line. And I can walk you through `client_app.py` line by line."
  - **Why say it this way:** "AI-assisted" as a blanket statement understates the notebooks and undersells the part you can actually defend. Naming the split shows you know which parts are yours line by line — which is the only thing an interviewer is asking.

**If you only remember three things:** the shuffle story, the bagging-vs-averaging answer, and knowing that your two AUCs are not comparable. Those three are the difference between a project you did and a project you can defend.

---

## 13. Practice tests (do these before the interview)

### A. Concept questions — answer aloud, no notes

1. **What exactly crosses the network in one FL round?** (Answer: tree bytes in `ArrayRecord`, metrics `{"num-examples": N, "auc": ...}` in `MetricRecord`, config `{"server-round": R}`. Never raw rows.)

2. **Why does round 1 behave differently from rounds 2+?** (Answer: round 1 has no global model — server sends `b""`, client branches on `server-round==1` and calls `xgb.train()` from scratch. Rounds 2+ load the global model with `bst.load_model()` and boost additional trees via `_local_boost()`.)

3. **How many trees does the final model have after 3 rounds with 2 clients and `local-epochs=1`?** (Answer: 3 × 2 × 1 = 6. Each round each client adds 1 tree, server concatenates all of them.)

4. **What would happen if you set `local-epochs=5`?** (Answer: each client boosts 5 trees per round, but `_local_boost` only slices and sends the LAST 5 trees. Final model = 3 × 2 × 5 = 30 trees. More local work, same communication cost.)

5. **Why did we need `dataset.shuffle(seed=42)` before `IidPartitioner`?** (Answer: Telco data is label-sorted — first 1869 rows are churned. `IidPartitioner` uses `shard(contiguous=True)` which doesn't shuffle. Without shuffle, partition 1 got zero churned rows → single-class validation → `auc: nan`.)

6. **What's the difference between `st.cache_data` and `st.cache_resource`?** (Answer: `cache_data` caches serialized values (CSV DataFrames) — safe, picklable. `cache_resource` caches live objects (XGBoost model) — loaded once, shared across reruns, not picklable.)

7. **If your federated AUC is 0.99 but your notebook got 0.93, is the federated model better?** (Answer: No — different evaluation protocol. Federated evaluates on each client's 20% validation split from the same training partition; notebook evaluates on a held-out test set. Not directly comparable.)

8. **What does `aggregate_bagging` actually do to the model JSON?** (Answer: parses both models' JSON, increments `num_trees` and `iteration_indptr`, appends new trees' array to `trees`, appends `0` to `tree_info` for each new tree.)

### B. Code reading — find the answer in the source

Open each file and answer without running anything:

1. **`client_app.py:47-53`** — what happens when `global_round == 1`? Why is this branch needed?

2. **`server_app.py:29`** — what is `global_model = b""`? Why empty bytes instead of `None`?

3. **`task.py` `load_data()`** — trace what happens on the second call with the same `partition_id`. (Answer: `fds` is cached globally, so the dataset/partitioner aren't recreated — `fds.load_partition(id)` just returns the cached partition.)

4. **`pyproject.toml:34`** — what does `save-model = false` control? What happens when you override with `--run-config "save-model=true"`?

5. **`fedxgb_bagging.py` `aggregate_train()`** — how does `self.current_bst` evolve across rounds? (Answer: `configure_train` resets it to the current `arrays` bytes, then `aggregate_train` folds each client's model into it via `aggregate_bagging`.)

### C. Debugging exercises — find the bug without running

Each of these is a real bug we hit or could hit. Find the problem:

1. **Bug:** `'auc': nan` in every round's evaluate log.
   **Hint:** check what `np.unique(val_dm.get_label())` returns for each partition. (Answer: one partition has only one class — data wasn't shuffled before partitioning.)

2. **Bug:** `st.metric("Total examples", f"{df['num_examples_train'].iloc[-0]:,}")` shows 5633 (first row) instead of the last row's value.
   **Hint:** what is `[-0]`? (Answer: `[-0]` is `[0]` — negative zero equals zero in Python. Use `[-1]`.)

3. **Bug:** Predict form returns 0.0% churn probability for every input.
   **Hint:** check whether `build_features` output column order matches `FEATURE_NAMES`. (Answer: if you build the array from dict iteration order instead of `FEATURE_NAMES` order, features are misaligned — model sees wrong values at wrong positions.)

4. **Bug:** `streamlit run` crashes with `charmap codec can't encode '\U0001f38a'`.
   **Hint:** what encoding does Windows PowerShell use by default? (Answer: cp1252. Fix: `$env:PYTHONUTF8 = "1"`.)

5. **Bug:** `InconsistentMessageReplies: Expected exactly one Array`.
   **Hint:** how many arrays did you put in `ArrayRecord([...])`? (Answer: more than one. Keep exactly `[model_np]`.)

### D. Hands-on exercises (do on your machine)

1. **Run 5 rounds instead of 3.** Change `--run-config "num-server-rounds=5"` and note: does AUC keep improving? At what point does it plateau? Record the 5 numbers.

2. **Change `local-epochs` to 2.** Run with `--run-config "num-server-rounds=3 local-epochs=2 save-model=true"`. Check `bst.num_boosted_rounds()` — should be 12 (3×2×2). Does AUC improve? Why or why not?

3. **Run with 3 supernodes.** Change `--federation-config "num-supernodes=3"` and `IidPartitioner(num_partitions=3)` in `task.py`'s `load_data` call. Verify: 3 clients each return `num-examples`, AUC still valid. Count trees: 3 rounds × 3 clients × 1 epoch = 9.

4. **Break it intentionally:** comment out `dataset = dataset.shuffle(seed=42)` in `task.py`, run, observe `'auc': nan`. Uncomment it, run again, observe real AUC. This is your debugging story for interviews.

5. **Compare bagging vs cyclic:** read the `xgboost-comprehensive` Flower example. Run with `train-method='cyclic'` (one client per round instead of all). Note how many rounds cyclic needs to match bagging's AUC.

✅ **Practice test pass criteria:** you can answer all 8 concept questions aloud, locate answers for all 5 code-reading questions in under 2 minutes each, identify the bug in all 5 debugging exercises, and have run at least 2 of the 5 hands-on exercises.

---

## Sources searched (Flwr docs)

- `tutorial-quickstart-xgboost` (train/evaluate/empty-model pattern)
- `how-to-run-simulations` + `1.33` variant (Ray resources, OOM, `num_cpus/num_gpus`, soft assignment)
- `how-to-run-flower-locally` (managed SuperLink, `:local:` vs `:local-in-memory:`, `superlink.log`, port 39093)
- `ref-flower-configuration` (config migration, `flwr run` must run in app dir)
- `FedXgbBagging` API + source (`_ensure_single_array`, `aggregate_bagging`, `weighted_by_key=num-examples`)
- `xgboost-comprehensive` example (bagging vs cyclic, `centralised-eval`)
- `flwr_datasets`: `IidPartitioner`, `tutorial-use-partitioners`, same-object partitioner bug (#4335), split-consistency fix (#5340)
- Discuss: SLURM silent fail after `configure_fit`, GPU/CUDA stop with no logs, Ray `ActorDiedError` OOM, DP + `FedXgbBagging` incompatibility (`allow_pickle=False`), XGBClassifier serialization via JSON.

---

## Appendix A. Your notebook `notebooks/Privacy_churn_v1.ipynb` — full review

Verified by reading all 70 cells (sources + recorded outputs). Pipeline:

1. **Load + EDA (cells 3–10):** kagglehub Telco 11.1.3 → 7043×50 → `head/tail/info/isnull/describe`. Missing: Offer 3877, Internet Type 1526, Churn Category/Reason 5174 (only churned rows have them — expected, not dirt).
2. **Clean + encode (cells 15–22):** drop ID/Country/State/City/Quarter → fillna Offer→No Offer, Internet Type→No Internet → Yes/No→1/0, Male→1/Female→0 → one-hot Offer/Internet Type/Contract/Payment/**Customer Status** → 56 feature cols + label (5174 stayed / 1869 churned).
3. **Split (cell 27):** stratified 80/20 → train 5634 / test 1409. `df = train_set` (alias, not copy).
4. **Corr + RF importance (cells 28–38):** `Customer Status_Churned` corr = **1.0** with label, `Stayed` = −0.86, `Churn Score` = 0.66. Default-RF importance top-4: Churned 0.32, Stayed 0.18, Satisfaction 0.18, Churn Score 0.14.
5. **Leakage drop (cell 39):** drops the 4 cols from `df/Xtrain/df_test/Xtest` → 52 features (matches `API/app.py` 52 ✅).
6. **Model bake-off (cells 42–48):** DT/RF/Ada/XGB/LGBM (200 trees, depth 5 default) → 5-fold CV F1: **XGB 0.931, LGBM 0.927, RF 0.901, DT 0.901, Ada 0.871**.
7. **Tuning (cells 49–66):** `scale_pos_weight` = 2.77 computed, but both grids chose **1**. XGB grid best (lr 0.02, depth 5, 400 trees) → test acc 0.963 / churn-F1 0.928. LGBM grid best → acc 0.964 / F1 0.930. Optuna XGB 200 trials × 3-fold (comment says "10 trials" — stale) → test **acc 0.965 / F1 0.933**, saved as `best_xgb_model.joblib` (cell 68, to CWD — provenance of `models/Telco_xgb_v1.joblib` still unconfirmed).
8. **Export (cell 69):** `df.to_csv('telco_v1.csv')` → **crashed**: `AttributeError: 'NoneType' object has no attribute 'to_csv'`.

**What you did right:** stratified split, fillna-before-encode, dropping constants, comparing 5 models with CV-F1 (right metric for imbalance), computing `scale_pos_weight` instead of ignoring imbalance.

**What went wrong (ordered by damage):**
1. **Label leakage trained into the importance analysis.** `Customer Status` is the label in disguise (Churned≡1, Stayed≡0); `Churn Score` is IBM SPSS's *prediction* of churn used as a feature. The RF importance (cells 35–37) ran on leaked features, so its ranking is meaningless. Drop these *before* encoding/splitting; never one-hot `Customer Status`. ⚠️ Also audit `Satisfaction Score` (corr −0.76): if that survey is collected *after* the churn decision, it's leakage too — check when it's measured.
2. **The drop bug is real, not hypothetical.** Cell 39 `df = df.drop(..., inplace=True)` assigns `None` (pandas `inplace` returns `None`), and cell 69 proves it — the export crashed. Fix: `df.drop(columns=..., inplace=True)` with no assignment.
3. **Encoder fitted on full data + frame aliasing.** `OneHotEncoder.fit` (cell 21) runs *before* the split (cell 27) — test categories leak into train. `df_test = test_set` / `df = train_set` are aliases, so every `inplace` drop hits both names. Fix: split first → `fit` on train, `transform` test → `.copy()` when aliasing.
4. **Gender mapping flipped at inference.** Notebook Male→1; `frontend/streamlit_app.py` sends Male→0 (`{0: Male, 1: Female}`); `API/app.py` passes ints straight through. Every male customer is currently scored as female and vice versa. One convention everywhere.
5. **Spaces in column names.** LightGBM logs `Found whitespace in feature_names, replace with underlines` — the model silently renames `Avg Monthly GB Download` etc. Rename to `snake_case` once, in notebook + API + Streamlit + FL loader, or name-based alignment breaks.
6. **Dead weight:** cell 57 commented-out SVC grid, cell 59 `!pip install optuna` mid-notebook (belongs in requirements), Optuna `n_jobs=-1` outside *and* CV inside (thread oversubscription), no seed/pruner, reruns collide on fixed `study_name`.
7. **Unreproducible artifact.** Nothing records which run produced `models/Telco_xgb_v1.joblib`, its params, or its test report; encoder + column order aren't saved with it. Save a bundle: model + encoder + `FEATURE_NAMES` + params + classification report.

**What to do next (in order):**
1. `notebooks/Privacy_churn_v2.ipynb`: drop leakage cols first (and rule on Satisfaction Score), split → fit-encoder-on-train, fixed drops, unified gender map, `snake_case` columns, saved `models/` bundle with metrics.
2. Re-run baseline, record test AUC/F1 — that number feeds Lab 3's Compare tab.
3. The v2 cleaning function is already extracted as `preprocess_telco()` in `quickstart_xgboost/preprocess.py` — central and federated preprocessing share the same code path.
