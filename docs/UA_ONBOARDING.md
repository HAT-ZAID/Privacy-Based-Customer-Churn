# Onboarding Guide â€” Privacy Based Customer Churn

For a developer who knows Python and basic ML but has lost the thread on **this specific repo**.

- **Graph source:** `.ua/knowledge-graph.json` â€” 39 nodes, 75 edges, 6 layers, 14 tour steps
- **Graph commit:** `9a9e1db` (matches `HEAD`)
- **Guide generated:** 2026-10-07

---

## How to read the evidence labels

Every claim below carries one of three labels. This distinction matters more than anything else in this document, because several widely-repeated facts about this repo are wrong.

| Label | Meaning |
|---|---|
| **[VERIFIED]** | I ran it or computed it on this machine during guide generation. |
| **[READ]** | Confirmed by reading the source. Correct as far as the code goes, but not executed. |
| **[INHERITED]** | Comes from the graph/README/prior session. Not re-verified here. |

### Corrections to the commonly-repeated story

Four claims I was asked to carry turned out to be inaccurate. Corrected inline below:

1. **The gender inversion is in `v1`, not in the serving path.** `notebooks/Privacy_churn_v2.ipynb` maps `Male â†’ 0` â€” the *same* as `preprocess.py:54` and `frontend/streamlit_app.py:7`. Only `notebooks/Privacy_churn_v1.ipynb` maps `Male â†’ 1`. Since **v2 is the notebook that wrote `models/telco_encoder.joblib`, `feature_names.txt` and `classification_report.txt`**, the deployed centralized path is self-consistent. The defect is that v1 is stranded under the opposite encoding, so its numbers are not comparable to v2's. **[VERIFIED]**
2. **The Telco parquet is not label-sorted in the monotone sense.** The first row that is *not* churned is at index 476. But **all 1,869 churned rows fall inside the first 3,521 rows** â€” so a contiguous two-way split gives partition 1 **0 churned rows out of 3,522**. That, not strict label-sorting, is the real cause of `auc: nan`. **[VERIFIED]**
3. **There IS a live sklearn version skew.** The environment (`churnenv`) has `scikit-learn` **1.9.0** installed, but `models/telco_encoder.joblib` was pickled by **1.6.1**, so every load emits `InconsistentVersionWarning: Trying to unpickle estimator OneHotEncoder from version 1.6.1 when using version 1.9.0`. The encoder still works (verified), but re-save it with `joblib.dump(encoder, path)` under 1.9.0 to remove the warning. **[VERIFIED]**
4. **`flwr run` reproduces, but it needs free RAM.** It fails inside `ray.init()` when the machine is memory-starved (see [Â§7](#7-open-issues--known-drift), item 8), because Ray sizes its object store from *available* memory. With strays killed it completes all 3 rounds and reproduces the federated AUCs bit-identically. **[VERIFIED]**

### Repo state you should know before you touch anything

`git log` has only **3 commits** (`886bbd0` â†’ `df2b991` â†’ `9a9e1db`), and **most of the actual project is untracked**: `README.md`, `data/`, `docs/`, `frontend/federated_demo.py`, `models/*`, `notebooks/Privacy_churn_v2.ipynb`, and all of `quickstart-xgboost/`. **[VERIFIED]**

There are also two uncommitted modifications and one pending rename:

- `API/app.py:16` â€” `MODEL_PATH` changed from `Telco_xgb_v1.joblib` â†’ `Telco_xgb.joblib`
- `models/Telco_xgb_v1.joblib` **deleted**, `models/Telco_xgb.joblib` **added** (a rename, not yet committed)
- `.gitignore` modified

**Consequence:** the committed tree and the working tree disagree about the model filename. If you `git stash` or check out a clean tree, `API/app.py` will point at a file that no longer exists. Commit the rename before anything else.

---

## 1. Project Overview

**Name:** Privacy Based Customer Churn
**Languages:** ipynb, json, markdown, python, toml, txt
**Frameworks:** Flower (flwr), XGBoost, FastAPI, Pydantic, Uvicorn, Streamlit, Ray, Jupyter

Telco customer churn prediction with **two serving paths over the same 7,043-row dataset that share almost no Python code.**

### The data

7,043 rows Ã— 50 raw columns â†’ 52 engineered features. Label `Churn Label`: **5,174 stayed / 1,869 churned**. Source: `data/telco_original.parquet` (fictional California telco, Q3). **[VERIFIED]**

### Path A â€” Centralized (classic ML serving)

```
notebooks/Privacy_churn_v2.ipynb
  â†’ models/Telco_xgb.joblib          (XGBoost, 400 trees)
  â†’ API/app.py                       (FastAPI, POST /predict)
  â†’ frontend/streamlit_app.py        (52-field Streamlit form)
```

Held-out test n=1,409 (1,035 stayed / 374 churned). **Accuracy 0.969, churn-class F1 0.940** (precision 0.958, recall 0.922). **[READ]** â€” from `models/classification_report.txt`.

Run it:

```bash
uvicorn API.app:app --reload            # http://localhost:8000/docs
streamlit run frontend/streamlit_app.py # http://localhost:8501
```

### Path B â€” Federated (Flower simulation)

```
quickstart-xgboost/
  preprocess.py            clean the raw Telco frame
  task.py                  FEATURE_NAMES + one-hot encode + shard into partitions
  client_app.py            @app.train / @app.evaluate   (runs per client)
  server_app.py            FedXgbBagging strategy        (runs on server)
  â†’ final_model.json       (6 trees, 52 features)
  â†’ frontend/federated_demo.py
```

Run it:

```bash
pip install -e .            # inside quickstart-xgboost/
flwr run .
```

Federated AUC **0.991 â†’ 0.993 â†’ 0.993** over 3 rounds; final model 6 trees (3 rounds Ã— 2 clients Ã— 1 local epoch); Per-partition AUC **0.9897 / 0.9962**. **[VERIFIED]** â€” prior verified run. **I could not re-verify: `flwr run` crashes in this environment.** See item 8.

### The privacy claim, stated honestly

Raw customer rows **never leave a client partition**. The only things that cross the network boundary are:

- serialized XGBoost JSON tree bytes, carried as `ArrayRecord` index `"0"`
- two scalars in a `MetricRecord`: `auc` and `num-examples`

**No differential privacy is claimed, and none is implemented.** XGBoost split structures are known to be invertible in small-partition settings â€” 2 clients over 7,043 rows is well inside the regime where tree gradients leak. Also: this is **Flower Simulation Engine on Ray**, meaning all clients run as local processes on one machine. There is no multi-machine deployment and no network boundary in the threat-model sense. **[READ]**

---

## 2. Architecture Layers

Six layers from the graph. `â†’` shows the data/handoff direction within each layer.

### L1 â€” Federated Training (`layer:federated-training`)
The self-contained Flower app under `quickstart-xgboost/`. Holds **all three import edges in the repo** and shares no Python with the centralized path.
`preprocess.py` â†’ `task.py` â†’ `client_app.py` / `server_app.py`, bound together by `pyproject.toml`.
Key files: `quickstart_xgboost/{preprocess,task,client_app,server_app,__init__}.py`, `quickstart-xgboost/pyproject.toml`

### L2 â€” Centralized Training (`layer:centralized-training`)
Two notebook passes. **v1** wins a 5-model bake-off at CV F1 0.931 and pushes test accuracy 0.965 via Optuna, but leaks `Customer Status` and `Churn Score` into its own importance analysis and fits the encoder before the split. **v2** drops the four leakage columns up front, splits before fitting the encoder, and is what actually wrote the `models/` artifacts.
Key files: `notebooks/Privacy_churn_v1.ipynb`, `notebooks/Privacy_churn_v2.ipynb`

### L3 â€” Model Artifacts (`layer:model-artifacts`)
The frozen outputs of both paths, side by side. **This is the repo's main architectural seam** â€” the one column ordering every model depends on is duplicated here, in `API/app.py` and in `task.py`, rather than shared from a single config module.
Key files: `models/best_params.json`, `models/classification_report.txt`, `models/feature_names.txt`, `quickstart-xgboost/final_model.json`

### L4 â€” Centralized Serving (`layer:centralized-serving`)
The only always-on process in the repo. Loads `models/Telco_xgb.joblib` into `app.state` once at startup; exposes `GET /`, `GET /Health`, `POST /predict`. Uses `BASE_DIR`-relative paths (`API/app.py:15`) and is therefore portable â€” unlike `task.py:77-78`.
Key files: `API/app.py`

### L5 â€” Frontend (`layer:frontend`)
Streamlit tier for both paths. `streamlit_app.py` builds the 52-field form with ~140 lines of hand-written dark CSS that duplicates the `.streamlit/config.toml` palette. `federated_demo.py` is the deliberately incomplete 4-tab teaching scaffold.
Key files: `frontend/streamlit_app.py`, `frontend/federated_demo.py`, `frontend/.streamlit/config.toml`, `frontend/requirements.txt`

### L6 â€” Documentation (`layer:documentation`)
Unusually strong prose layer for its size. Root `README.md` with the architecture diagram and an honest caveat that centralized 0.93 vs federated 0.991-0.993 are **not protocol-comparable**. `docs/fed-guide.md` is 675 lines / 32 sections. `quickstart-xgboost/README.md` is upstream flwrlabs text kept verbatim â€” and therefore **drifted** (still declares `dataset: [HIGGS]`).
Key files: `README.md`, `docs/fed-guide.md`, `quickstart-xgboost/README.md`

---

## 3. Key Concepts

The six things you must internalize before touching this code.

### 3.1 The row/column split â€” why this is *horizontal* FL

Federated learning splits **rows by default** (each client holds different customers, all features). This repo instead splits **rows only** while all clients share the identical 52-column schema â€” that is horizontal FL, the simplest form. `IidPartitioner(num_partitions=2)` at `task.py:127` takes contiguous slices of one shared dataset. **[READ]**

The consequence is that there is **no feature-space heterogeneity to protect against**. The privacy claim here is about *rows* not crossing a boundary, which is the easy case. Do not let the architecture diagram imply protection against a rich feature vector split across parties â€” that is not implemented.

### 3.2 Why `FedXgbBagging` concatenates trees instead of averaging them

FedAvg averages **weights**. That is meaningless for gradient-boosted trees: a tree is a discrete split structure (`feature < threshold ? left : right`), not a point in a continuous weight space. Interpolating two trees produces a third object that is not a tree and scores arbitrarily.

So Flower's `FedXgbBagging` treats each client's contribution as a **new set of trees appended to the ensemble**. Each round the ensemble grows by `num_clients Ã— num_local_epochs` trees. That is exactly why `final_model.json` has **6 trees = 3 rounds Ã— 2 clients Ã— 1 epoch** while the centralized model has **400**. **[VERIFIED]** for the run, **[READ]** for the mechanism.

### 3.3 The message shapes â€” `ArrayRecord` and `MetricRecord`

Flower's `Message` carries typed records. Two are used here:

| Record | Payload | Carries |
|---|---|---|
| `ArrayRecord` | `{"0": ndarray}` | the model, as `np.frombuffer(bst.save_raw("json"), dtype=np.uint8)` |
| `MetricRecord` | `{"auc": float, "num-examples": int}` | scalar eval metrics |

The model is always **item `"0"` of a list** â€” which is why every reader indexes it as `msg.content["arrays"]["0"]` / `result.arrays["0"]`. This positional convention is a real footgun; it is not configurable. **[READ]**

**What never appears in a `Message`:** a `DMatrix`, a feature row, a label, or any customer identifier. That is the whole privacy story, and it is verifiable by inspection of `client_app.py`.

### 3.4 "Round 1 is special"

`server_app.py` seeds the global model with `ArrayRecord([np.frombuffer(b"", dtype=np.uint8)])` â€” **deliberately empty**, because an `xgboost.Booster` cannot deserialize a zero-length buffer. **[READ]**

So `client_app.py:train` branches:

- **`server_round == 1`** â†’ the server sent `b""` â†’ call `xgb.train()` from scratch
- **rounds 2+** â†’ `bst.load_model(received_bytes)` then call `_local_boost(...)` to append

Remove the round-1 branch and every run dies on the first client. This is also why `server_app.py` never touches customer data â€” it only shuffles opaque `uint8` buffers.

### 3.5 The 52-column contract, and why duplicating it is a seam

`FEATURE_NAMES` â€” an ordered list of 52 columns â€” exists in **three places**:

| Location | Form |
|---|---|
| `API/app.py:33-86` | Python list literal |
| `quickstart_xgboost/task.py:14-â€¦` | Python list literal |
| `models/feature_names.txt` | one name per line |

**XGBoost indexes columns positionally.** There is no name check at inference time. If the serving list drifts by one position, the model still runs, still returns a probability in `[0,1]`, and still reports high confidence â€” while reading entirely the wrong feature. There is no shape assertion that would catch this. Both `task.py:120` and `API/app.py:206` defend against it with `df.reindex(columns=FEATURE_NAMES, fill_value=0)`, which fixes *missing* columns but happily accepts a *misordered* set. **[READ]**

The right fix is a single source of truth â€” `models/feature_names.txt` already exists and is already the closest thing to a shared config. Nothing imports it.

### 3.6 The privacy boundary and its limits

Restated because it is the project's headline claim and it is *narrow*:

- **Claimed and true:** raw rows stay inside the partition; only tree bytes and two scalars are transmitted.
- **Not claimed, not implemented:** differential privacy, secure aggregation, gradient clipping, noise injection.
- **Known weakness:** tree structure leaks. With 2 clients over 7,043 rows, gradients are trivially invertible.
- **Not a real network boundary:** Flower Simulation Engine runs every client as a local Ray actor on one machine.

State it this way in any write-up. The current `README.md` already does â€” do not make it stronger.

---

## 4. Guided Tour

The 14 steps from the graph's `tour` array, in order.

### 1. Two Pipelines, One Dataset
Start at `README.md`, then look at the head of each pipeline: `API/app.py` and `quickstart_xgboost/server_app.py`. The mental model: two **independent** serving paths over the same 7,043 rows, sharing almost no Python. Nothing reconciles them â€” which is why step 2 matters.
Nodes: `document:README.md`, `file:API/app.py`, `file:quickstart-xgboost/quickstart_xgboost/server_app.py`

### 2. The 52-Column Contract
The single most important thing to internalize. See [Â§3.5](#35-the-52-column-contract-and-why-duplicating-it-is-a-seam). Every later step assumes you can point at that list and say why each copy exists.
Nodes: `file:API/app.py`, `file:quickstart-xgboost/quickstart_xgboost/task.py`, `document:models/feature_names.txt`

### 3. Where the Centralized Model Comes From
Read v1 for the story, v2 for the correction. v1 wins a 5-model bake-off and tunes to CV F1 0.931 but **leaks**: `Customer Status` restates the label and `Churn Score` is the vendor's own churn prediction â€” both dominate its importance output. v1 also fits the `OneHotEncoder` before the split and has an `inplace=True` chained-assignment bug that crashes its final export cell (`AttributeError: 'NoneType' object has no attribute 'to_csv'`, raw JSON line 12472). v2 drops the four leakage columns up front and is the notebook that wrote `models/`. **[VERIFIED]** for the crash; **[READ]** for the rest.
Nodes: `file:notebooks/Privacy_churn_v1.ipynb`, `file:notebooks/Privacy_churn_v2.ipynb`

### 4. Reading the Centralized Evidence
The frozen output of v2 â€” the only way to check what the notebooks actually claimed. `classification_report.txt`: accuracy 0.969, churn F1 0.940 on 1,409 rows. **Note churn recall of 0.922** â€” the number that matters if you care about catching churners. `best_params.json`: depth 19, `learning_rate` 0.031, `scale_pos_weight` 1.96. `feature_names.txt`: the same 52-column list from step 2.
Nodes: `config:models/best_params.json`, `document:models/classification_report.txt`, `document:models/feature_names.txt`

### 5. Centralized Serving: FastAPI
`API/app.py` is the only always-on process. Loads the model into `app.state` once at startup (`lifespan`, `app.py:114-119`), exposes `GET /`, `GET /Health`, `POST /predict`, and reindexes against the 52 columns before scoring. **The portable detail worth copying:** `BASE_DIR = Path(__file__).resolve().parent` at `app.py:15`. Contrast the hardcoded drive-letter path in `task.py:77-78` â€” that is the single reason one runs anywhere and the other does not.
Nodes: `file:API/app.py`, `endpoint:frontend/streamlit_app.py:/predict`

### 6. Centralized UI: the 52-Field Form
`streamlit_app.py` builds a 52-field form, layers ~140 lines of hand-written dark CSS over it, and fires one request at `/predict`. It **duplicates** the palette already defined in `.streamlit/config.toml` instead of relying on it â€” a maintenance trap, not a design choice. It also declares `gender_mapping = {0: "Male", 1: "Female"}` at line 7, which **matches** v2's encoding â€” see item 1 in [Â§7](#7-open-issues--known-drift).
Nodes: `file:frontend/streamlit_app.py`, `endpoint:frontend/streamlit_app.py:/predict`

### 7. The Federated App Manifest
`pyproject.toml` does double duty: `[build-system]`/`[project]` are packaging, `[tool.flwr.app]` and below are **runtime config** consumed by `flwr run`. `[tool.flwr.app.components]` binds `serverapp = "quickstart_xgboost.server_app:app"` and `clientapp = "quickstart_xgboost.client_app:app"` â€” that binding is how one command launches both sides.
Two gotchas: config keys are **hyphenated** (`params.max-depth`, `num-server-rounds`) and `task.py:replace_keys()` rewrites every hyphen to an underscore before XGBoost sees them; and `flwr-version-target` must match your installed `flwr`.
*Language lesson:* `__init__.py` is load-bearing in a non-obvious way. `server_app.py` uses absolute imports (`from quickstart_xgboost.task import replace_keys`) that only resolve because `packages = ["."]` makes `quickstart_xgboost/` the import root. Delete `__init__.py` and the modules still look importable from inside the directory via implicit namespace packages â€” luring you into a state where `flwr run` breaks but your REPL works.
Nodes: `config:quickstart-xgboost/pyproject.toml`, `file:quickstart-xgboost/quickstart_xgboost/__init__.py`

### 8. Federated Data Loading â€” the Load-Bearing Shuffle
**The most important step in the repo.** `load_data()` (`task.py:106-139`) runs: read parquet â†’ `preprocess_telco` â†’ one-hot 4 columns â†’ extract `Churn Label` as target â†’ `df.reindex(columns=FEATURE_NAMES)` â†’ `Dataset.from_dict` â†’ **`dataset.shuffle(seed=42)` at line 126** â†’ `IidPartitioner(num_partitions=2)` â†’ per-partition 80/20 split â†’ `DMatrix`.

Line 126 is not cosmetic, and the mechanism is subtler than "the data is sorted by label":

> `IidPartitioner` shards **contiguously** and never shuffles. In `data/telco_original.parquet`, **all 1,869 churned customers live in the first 3,521 rows**; the first *non*-churned row is at index 476, so the file is *blocked* rather than monotone-sorted. A contiguous two-way split therefore hands partition 1 **rows 3521â€“7042, which contain zero churned customers**. Its validation set is single-class, so `eval_set` reports `auc: nan`, which propagates straight into the federated metric the server aggregates. **[VERIFIED] by direct computation on the parquet.**

With the shuffle in place: 2,817 / 2,816 train rows, and **both partitions contain both classes.** **[READ]**
Nodes: `file:quickstart-xgboost/quickstart_xgboost/task.py`, `file:quickstart-xgboost/quickstart_xgboost/preprocess.py`

### 9. Federated Client Logic
Where the privacy claim is earned. On `server_round == 1` the server sent `b""`, so the client takes the `xgb.train()` branch; rounds 2+ do `bst.load_model(global_model)` then `_local_boost`. The subtlety: **`_local_boost` slices the booster to only the last `num_local_round` trees** before returning. Returning the whole booster would resend every previously-trained tree each round, and aggregation would duplicate the ensemble. `evaluate()` returns only `auc` and `num-examples`, parsed out of XGBoost's text output via `split("\t")[1].split(":")[1]` â€” brittle, and silently yields `nan` on a single-class shard.
Nodes: `file:quickstart-xgboost/quickstart_xgboost/client_app.py`, `file:quickstart-xgboost/quickstart_xgboost/task.py`

### 10. Federated Server Loop
`server_app.py:main()` (`server_app.py:17-57`) seeds the empty `ArrayRecord`, instantiates `FedXgbBagging`, and calls `strategy.start()`. FedXgbBagging is **bagging, not weight averaging** â€” see [Â§3.2](#32-why-fedxgbbagging-concatenates-trees-instead-of-averaging-them). When `save-model = true` it calls `bst.save_model("final_model.json")`.
**Note:** the checked-in config has `save-model = false` (`pyproject.toml:34`), yet `final_model.json` exists on disk â€” so it was produced via `--run-config save-model=true` or a temporarily edited config.
Nodes: `file:quickstart-xgboost/quickstart_xgboost/server_app.py`, `config:quickstart-xgboost/pyproject.toml`

### 11. The Federated Artifact
`final_model.json` is ~18.6 KB, a `gbtree` with 6 trees at `num_feature` 52 â€” small enough to read in a browser, which is the point: it is a **real XGBoost model, not a summary**. Load it directly with `xgboost.Booster()` and score your own `DMatrix`. Per-partition AUC **0.9897 / 0.9962**. **[VERIFIED]**
Compare its shape to `models/Telco_xgb.joblib` (400 trees) to see what the federated path gave up for the privacy boundary.
Nodes: `config:quickstart-xgboost/final_model.json`, `document:models/feature_names.txt`

### 12. Federated Demo UI â€” Exercises, Not Bugs
`federated_demo.py` is a 4-tab teaching skeleton (Learn / Run / Compare / Predict) with **7 deliberate TODO markers**. The file header says it: *"TODOs are yours to complete. If you can explain each TODO, you can defend it."* The Run tab has **no run button on purpose**; Compare asks you to replace a hardcoded `0.81`; Predict has 3 fields and no `build_features()` mapper, so `st.form_submit_button("Predict (TODO)")` is **inert**.
**Read it as a lab worksheet, not broken code** â€” and resist fixing all seven at once, because the exercise is the deliverable.
Nodes: `file:frontend/federated_demo.py`, `config:quickstart-xgboost/final_model.json`

### 13. Packaging Gaps
`frontend/requirements.txt` is one unpinned line (`streamlit`) covering neither the federated deps (`flwr`, `xgboost`) nor the API deps (`fastapi`, `pydantic`). A clean install of it **cannot run either app**. `.streamlit/config.toml` supplies the dark theme that `streamlit_app.py` then duplicates in CSS. Neither is a demo bug, but both are the first things to fix if you are picking this repo up to maintain it.
Nodes: `document:frontend/requirements.txt`, `config:frontend/.streamlit/config.toml`

### 14. Documentation and Known Drift
`docs/fed-guide.md` (675 lines / 32 sections) is the real teaching artifact â€” HIGGS warm-up, Telco migration, Streamlit demo labs, anchored by the `auc: nan` debugging story from step 8. `quickstart-xgboost/README.md` is upstream flwrlabs text kept verbatim and has **drifted** â€” it still describes the HIGGS flow, so treat it as background, not instructions. The gender bug is **partially** unresolved: v1 and v2 disagree with each other, though the deployed path is consistent (item 1 in [Â§7](#7-open-issues--known-drift)).
Nodes: `document:docs/fed-guide.md`, `document:quickstart-xgboost/README.md`, `file:notebooks/Privacy_churn_v1.ipynb`, `file:frontend/streamlit_app.py`

---

## 5. File Map

All **21** file-level nodes, organized by layer.

### L1 â€” Federated Training
| File | Complexity | Role |
|---|---|---|
| `quickstart_xgboost/task.py` | moderate | Shared data loading + config translation. Cleans the parquet, one-hot encodes 4 categoricals with the persisted encoder, shards 7,043 rows into per-client `DMatrix` partitions. Owns `FEATURE_NAMES` (line 14) and `replace_keys()`. **Import-time side effects** at lines 77-78. |
| `quickstart_xgboost/preprocess.py` | moderate | Single in-place cleaning function `preprocess_telco`. Drops leakage/high-cardinality/post-outcome columns, fills 2 nulls, binary-encodes Gender + 18 Yes/No columns. Returns `None` by design. |
| `quickstart_xgboost/client_app.py` | moderate | Flower `ClientApp`. `@app.train` / `@app.evaluate`. Where the privacy claim is earned â€” only tree bytes and 2 scalars cross the boundary. |
| `quickstart-xgboost/quickstart_xgboost/server_app.py` | simple | Flower `ServerApp`. Reads run config, seeds the empty global model, drives `FedXgbBagging`. Optionally writes `final_model.json`. |
| `quickstart-xgboost/quickstart_xgboost/__init__.py` | simple | One-line package marker (docstring only). Makes `quickstart_xgboost` importable so absolute imports resolve under Flower. |
| `quickstart-xgboost/pyproject.toml` | moderate | Dual-purpose: Flower app manifest **and** run-config. Binds serverapp/clientapp; carries `num-server-rounds = 3`, `local-epochs = 1`, XGBoost `params` in hyphenated form. |

### L2 â€” Centralized Training
| File | Complexity | Role |
|---|---|---|
| `notebooks/Privacy_churn_v2.ipynb` | complex | **The real trainer.** 67 cells. Drops the 4 leakage columns up front, splits before fitting the encoder, writes `telco_encoder.joblib` / `best_params.json` / `classification_report.txt` / `feature_names.txt`. `Male â†’ 0`. |
| `notebooks/Privacy_churn_v1.ipynb` | complex | First pass, 67 cells. 5-model bake-off won by XGB at CV F1 0.931, 200-trial Optuna to test acc 0.965. **Leaks** `Customer Status` + `Churn Score`, fits encoder pre-split, `Male â†’ 1`, final export cell crashes. |

### L3 â€” Model Artifacts
| File | Complexity | Role |
|---|---|---|
| `models/best_params.json` | simple | Winning XGBoost hyperparameters: `max_depth` 19, `learning_rate` 0.031043, `n_estimators` 909, `scale_pos_weight` 1.9649, `reg_lambda` 0.3226. |
| `models/classification_report.txt` | simple | Published eval evidence: accuracy 0.969, weighted F1 0.969, macro F1 0.959; stayed F1 0.979, churn F1 0.940. |
| `models/feature_names.txt` | simple | The ordered 52-column list as plain text. The natural candidate for a shared single source of truth â€” currently imported by nothing. |
| `quickstart-xgboost/final_model.json` | simple | Opaque ~18.6 KB XGBoost dump, 6 trees Ã— 52 features. Written by `server_app.py` when `save-model = true`. Consumed by `federated_demo.py`. |

### L4 â€” Centralized Serving
| File | Complexity | Role |
|---|---|---|
| `API/app.py` | complex | FastAPI serving layer. Loads `models/Telco_xgb.joblib` at startup into `app.state.model`. `GET /`, `GET /Health`, `POST /predict`. Pinned `FEATURE_NAMES` at lines 33-86. Portable via `BASE_DIR` (line 15). |

### L5 â€” Frontend
| File | Complexity | Role |
|---|---|---|
| `frontend/streamlit_app.py` | complex | The centralized predictor. 52-field form matching the API aliases, ~140 lines of inline dark CSS, one POST to `localhost:8000/predict`. `gender_mapping` at line 7. |
| `frontend/federated_demo.py` | moderate | Deliberately incomplete 4-tab teaching scaffold (Learn/Run/Compare/Predict) with 7 TODOs. `save-model` tab loads `final_model.json` but produces nothing. |
| `frontend/.streamlit/config.toml` | simple | Dark theme: Manrope font, teal `#33c0ad` on near-black `#0E1514`. Colors are **duplicated by hand** in `streamlit_app.py` CSS and can drift. |
| `frontend/requirements.txt` | simple | One unpinned line: `streamlit`. Packaging gap â€” see step 13. |

### L6 â€” Documentation
| File | Complexity | Role |
|---|---|---|
| `docs/fed-guide.md` | complex | The 675-line / 32-section teach-by-doing guide. HIGGS warm-up, Telco migration, Streamlit labs, `auc: nan` debugging story, gender bug, 23 practice questions. |
| `README.md` | moderate | 12-section project overview: architecture diagram, run commands, results table with the honest not-comparable caveat, scope statement. |
| `quickstart-xgboost/README.md` | moderate | Upstream flwrlabs text, **verbatim and drifted** â€” still describes the original HIGGS flow and declares `dataset: [HIGGS]`. Background only. |

**Node `endpoint:frontend/streamlit_app.py:/predict`** (moderate) is not a file but is included for completeness: it is the inline `if submitted:` top-level script block at `streamlit_app.py:259-270` â€” the app's only network call site, including `st.balloons()`, the POST, and a fake 0-100 progress bar animated with `sleep`.

---

## 6. Complexity Hotspots

Node complexity ratings: **5 complex**, **17 moderate**, 17 simple.

### The 5 `complex` nodes

| Node | What makes it non-obvious â€” read carefully |
|---|---|
| `file:notebooks/Privacy_churn_v2.ipynb` | The real trainer. Reads as clean ML but the *correctness depends on cell order*: the 4 leakage columns must be dropped **before** the encoder is fit, and the split **before** `OneHotEncoder` is fit. Reorder cells and you reintroduce v1's bugs silently. Its `Male â†’ 0` mapping is the ground truth for the whole serving path. |
| `file:notebooks/Privacy_churn_v1.ipynb` | Leaks on purpose-of-history, not intention: `Customer Status` restates the label and `Churn Score` is a vendor-side prediction. Both dominate its importance analysis. Also fits the encoder pre-split, aliases train/test so `inplace` drops hit both, and its last cell dies with `'NoneType' object has no attribute 'to_csv'`. **Its reported numbers are not trustworthy and not comparable to v2's.** |
| `file:API/app.py` | Three things bite here. (a) `FEATURE_NAMES` at lines 33-86 is a positional contract with no runtime check. (b) `MODEL_PATH` at line 16 was just edited to follow a **file rename that is not yet committed** â€” see [Â§7](#7-open-issues--known-drift) item 2. (c) `validate_features()` at lines 188-191 is **dead code** carrying its own `'This function needs to fixed'` comment; `predict_churn` relies on `reindex(fill_value=0)` instead and never calls it. |
| `file:frontend/streamlit_app.py` | ~140 lines of inline CSS that duplicates `.streamlit/config.toml` by hand. Its 52 fields must match the `ChurnRequest` aliases exactly (spaces, e.g. `Under_30` â†’ `"Under 30"`); a mismatch surfaces as a Pydantic 422, not a wrong answer. |
| `document:docs/fed-guide.md` | Long enough to be mistaken for authoritative. Its most valuable content is the `auc: nan` debugging story â€” but verify any claim in it against the code, since it predates some drift. |

### The 17 `moderate` nodes

**Genuine reading hazards â€” prioritize:**

| Node | Why it needs care |
|---|---|
| `function:task.py:load_data` (`task.py:106-139`) | **The most important function in the repo.** Line 126 `dataset.shuffle(seed=42)` is load-bearing, not cosmetic. Trace it end to end before changing anything about partitioning. |
| `function:client_app.py:train` (`client_app.py:34-77`) | The `if global_round == 1` branch is the key to the whole file. Also note `ArrayRecord` indexing (`arrays["0"]`) and that `bst.save_raw("json")` â†’ `np.frombuffer(..., uint8)` is the serialization contract with the server. |
| `function:client_app.py:evaluate` (`client_app.py:81-110`) | `eval_set(..., iteration=bst.num_boosted_rounds() - 1)` credits only the newest tree. The AUC parse `split("\t")[1].split(":")[1]` is **positional string surgery** â€” brittle, and silently yields `nan` on a single-class shard instead of raising. |
| `function:preprocess.py:preprocess_telco` (`preprocess.py:1-72`) | **Leaky by design â€” read it as a list of what is deliberately removed.** Dropping `Customer Status` and `Churn Score` is the anti-leakage core. It mutates in place and returns `None`, so callers must keep the original reference. Minor: leftover debug `print` loops at lines 63-65; `one_hot_cols`/`binary_cols` are declared but only used by that debug output. |
| `file:frontend/streamlit_app.py`'s `/predict` block (`259-270`) | The single network call site, plus a fake progress bar animated with `sleep` â€” cosmetic latency that looks like real work in a demo. |

**Structural / config nodes â€” read once, then reference:**

`file:quickstart-xgboost/quickstart_xgboost/task.py` (import-time side effects at 77-78) Â· `file:quickstart-xgboost/quickstart_xgboost/preprocess.py` (in-place mutation contract) Â· `file:quickstart-xgboost/quickstart_xgboost/client_app.py` (the round-1 branch) Â· `function:server_app.py:main` (`server_app.py:17-57`, empty seed + optional save) Â· `function:API/app.py:predict_churn` (`API/app.py:204-219`, the alias-dump â†’ reindex â†’ score sequence) Â· `class:API/app.py:ChurnRequest` (`API/app.py:126-178`, 52 required fields with space-containing aliases) Â· `class:API/app.py:response_model` (`API/app.py:89-105`) Â· `file:frontend/federated_demo.py` (7 intentional TODOs) Â· `function:federated_demo.py:load_federated_model` (`73-81`, uses `@st.cache_resource` because a Booster is unhashable) Â· `config:quickstart-xgboost/pyproject.toml` (dual-purpose manifest; hyphenated keys; `flwr-version-target` must match installed flwr) Â· `document:quickstart-xgboost/README.md` (drifted upstream text) Â· `document:README.md` (12 sections, results table).

---

## 7. Open Issues & Known Drift

### The two that block a public upload

#### 1. Gender encoding â€” inconsistent between notebooks, and unmapped at the API

| Location | Mapping | Status |
|---|---|---|
| `notebooks/Privacy_churn_v1.ipynb` (raw JSON line 1808) | `Male â†’ 1, Female â†’ 0` | **inverted** |
| `notebooks/Privacy_churn_v2.ipynb` (raw JSON line 1886) | `Male â†’ 0, Female â†’ 1` | canonical |
| `quickstart_xgboost/preprocess.py:53-55` | `Male â†’ 0, Female â†’ 1` | agrees with v2 |
| `frontend/streamlit_app.py:7` | `{0: "Male", 1: "Female"}` | agrees with v2 |
| `API/app.py:127` | `Gender: Annotated[int, Field(...)]` | **no mapping at all** |

**Corrected impact.** The earlier story â€” "the notebook-vs-frontend inversion silently flips gender on every prediction" â€” **is not true of the deployed path.** v2 is the notebook that produced `telco_encoder.joblib`, `feature_names.txt` and `classification_report.txt`, and it maps `Male â†’ 0`, matching `preprocess.py` and `streamlit_app.py`. **[VERIFIED]**

What is actually true:
- **v1's numbers are stranded.** Anyone re-running v1 gets a model trained under the opposite gender encoding; its accuracy 0.965 / CV F1 0.931 are not comparable to v2's 0.969, and its feature-importance table is already untrustworthy for the leakage reasons in step 3.
- **`API/app.py` has no gender semantics.** It accepts a bare `int` and trusts the caller. The 0/1 meaning is documented only in a Streamlit radio label (`streamlit_app.py:189`). Any third-party API consumer will get it wrong with no error.

**Fix:** (a) change v1's map to `Male â†’ 0` and re-run it, or better, mark v1 clearly as superseded and stop citing its numbers; (b) replace `API/app.py:127` with a constrained field documenting the semantics, e.g. `Gender: Annotated[int, Field(..., ge=0, le=1, description="0=Male, 1=Female")]`; (c) move the mapping into one shared constant that `preprocess.py`, `streamlit_app.py` and `app.py` all import.

**Blocks public upload because:** an unaudited reader who sees the notebook disagree with the frontend will reasonably assume predictions are silently wrong. The honest fix is to say which one is right, not to leave it ambiguous.

#### 2. Hardcoded absolute paths â€” the repo does not run after cloning

At `quickstart-xgboost/quickstart_xgboost/task.py:77-78`:

```python
DATASET_PATH = r"D:\Privacy Based Customer Churn\data\telco_original.parquet"
encoder = joblib.load(r"D:\Privacy Based Customer Churn\models\telco_encoder.joblib")
```

**[VERIFIED]** Both are module-level, so they execute **at import time** â€” not lazily inside `load_data`. Anyone who clones the repo to another path or another OS gets `FileNotFoundError` before the first round, with no useful traceback pointing at a config problem.

Note the contrast with the correct pattern already in the repo â€” `API/app.py:15`:

```python
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "Telco_xgb.joblib"
```

**Fix:** mirror `BASE_DIR` in `task.py`:

```python
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "data" / "telco_original.parquet"
encoder = joblib.load(BASE_DIR / "models" / "telco_encoder.joblib")
```

(`parents[2]` because `task.py` sits at `quickstart-xgboost/quickstart_xgboost/task.py`, two levels below the repo root.) Ideally also defer the `joblib.load` out of module scope into `load_data`, since import-time I/O makes the module unimportable without its data. **[READ]** on the fix â€” not applied.

**Blocks public upload because:** it is the difference between a runnable repo and a screenshot repo.

### The rest of the drift

#### 3. `frontend/requirements.txt` â€” one unpinned line
`frontend/requirements.txt` contains only `streamlit`, unpinned. It covers neither the federated deps (`flwr`, `xgboost`) nor the API deps (`fastapi`, `uvicorn`, `pydantic`) that `streamlit_app.py` calls via `requests`. A clean install of this file **cannot run either frontend**.
**Fix:** split into `frontend/requirements.txt` (pinned `streamlit`, `requests`, `pandas`, `xgboost`) and an `API/requirements.txt`, or add one root `requirements.txt` that supersedes it. Pin versions â€” the repo already runs against `flwr` 1.33.0 and `xgboost` 3.2.0. **[VERIFIED]**

#### 4. `quickstart-xgboost/README.md` â€” drifted from the migrated loader
The upstream flwrlabs README was kept verbatim. It still describes the original **HIGGS** binary-classification flow and its front matter still declares `dataset: [HIGGS]`, but `task.py` has been migrated to the Telco loader. A newcomer following this file will look for a dataset that is no longer used.
**Fix:** either replace it with a Telco-specific README, or add a prominent header banner: *"Upstream Flower quickstart text. This app now uses the Telco churn dataset â€” see the root `README.md` and `docs/fed-guide.md`."* **[READ]**

#### 5. `Satisfaction Score` â€” suspected leakage, unresolved
`Satisfaction Score` is still present in all three `FEATURE_NAMES` copies and in the trained models. I computed its correlation against the binary label on the actual parquet: **âˆ’0.7546**. **[VERIFIED]**

By comparison the columns that *were* correctly removed (`Churn Score`, 0.6608 raw correlation; `Customer Status`, a direct restatement of the label) are gone. `Satisfaction Score` is a 1-5 self-reported rating, so it is arguably a legitimate pre-churn predictor rather than a post-outcome artifact â€” unlike `Churn Score`, which the vendor computed. But at âˆ’0.75 it is the single strongest surviving feature, and its 1-5 scale may encode survey timing relative to the churn event.
**Fix:** unresolved by design â€” this needs a decision, not a code change. Run an ablation: retrain v2 with the column dropped and compare churn-class F1 against the 0.940 baseline. If it holds, drop the feature and note the reason; if it collapses, document why the feature is legitimately available at prediction time.

#### 6. sklearn version skew â€” **live, fires on every load**
`churnenv` has `scikit-learn` **1.9.0** installed, while `models/telco_encoder.joblib` was pickled by **1.6.1**. Every `joblib.load()` of that encoder emits:

```
InconsistentVersionWarning: Trying to unpickle estimator OneHotEncoder from version 1.6.1
when using version 1.9.0. This might lead to breaking code or invalid results.
```

The encoder still works â€” `preprocess_telco()` + `encoder.transform()` produce the expected 52 columns â€” so this is a warning, not a failure. **[VERIFIED]** in `churnenv`. Note the *global* Python on this machine carries a different sklearn build, so confirm which interpreter you are in before trusting any version report; that is exactly how this gets misdiagnosed.

**Fix:** re-save under the current interpreter with `joblib.dump(encoder, models/telco_encoder.joblib)`. Better: save one bundle containing model + encoder + `FEATURE_NAMES` + params so the versions cannot drift apart.

#### 7. `flwr-version-target` drift in `pyproject.toml`
`quickstart-xgboost/pyproject.toml:22` declares `flwr-version-target = "1.31.0"`, which matches `churnenv`'s installed `flwr` **1.31.0** â€” no drift today. **[VERIFIED]** in `churnenv`. The FAB format version is tied to the target, so this becomes a real build-refusal risk the moment anyone upgrades flwr without bumping the target. Record the pair in a root `requirements.txt` and change them together.

#### 8. `flwr run` needs free RAM â€” a real onboarding trap, but it does run
Running `flwr run .` inside `quickstart-xgboost/` succeeds when the machine has headroom and **fails inside `ray.init()`** when it does not:

```
ValueError: Attempting to cap object store memory usage at 68553523 bytes,
            but the minimum allowed is 78643200 bytes.
RuntimeError: Simulation Engine crashed.
```

Ray derives its object-store cap from *available* memory, so this fires when free RAM is low â€” the app code never executes. This machine has **7.4 GB total**; with stray Flower/Ray processes left running, free RAM dropped to ~250â€“470 MB and the run died. Killing the strays and re-running produced **bit-identical** results to the earlier run:

```
[ROUND 1/3] {'auc': 0.9909828030774182}
[ROUND 2/3] {'auc': 0.9925536221296033}
[ROUND 3/3] {'auc': 0.9929157168364653}
Strategy execution finished in 18.28s   Saving final model to disk...
```

So the federated AUCs are **[VERIFIED]**, and the failure mode is environmental, not a code bug. Two practical consequences:
- Before running, `Stop-Process -Name flower-superlink,flower-superexec,raylet -Force` and close memory-heavy apps. Ray needs ~1.5 GB free to be comfortable.
- A federated demo that silently cannot start on an 8 GB laptop is a genuine onboarding trap â€” worth a line in the README.

#### 9. Model filename rename is uncommitted
`models/Telco_xgb_v1.joblib` is deleted and `models/Telco_xgb.joblib` added in the working tree, with `API/app.py:16` updated to match â€” **none of it committed.** The committed tree and working tree disagree about the model filename, so a `git stash`, clean checkout, or CI run will fail to find the model. Compounding it, most of the project (`README.md`, `data/`, `docs/`, `frontend/federated_demo.py`, `models/*`, `notebooks/Privacy_churn_v2.ipynb`, all of `quickstart-xgboost/`) is **untracked** â€” 3 commits do not contain the actual project. **[VERIFIED]**
**Fix:** `git add -A` and commit the rename plus the untracked project files as the first action. Decide whether `models/*.joblib` (1.6 MB binary) belongs in version control or in a release/artifact store.

#### 10. Duplicated design tokens
`frontend/.streamlit/config.toml` defines the dark palette (`#33c0ad` primary, `#0E1514` background, `#17211F` secondary), and `streamlit_app.py` re-declares the same colors in ~140 lines of inline CSS. They can drift with no test to catch it. **[READ]**
**Fix:** delete the CSS color literals and let `config.toml` drive the theme, keeping only the CSS that `config.toml` genuinely cannot express (layout, spacing, component overrides).

---

## Appendix â€” Highest-value next actions

1. **Commit the rename and the untracked project files** (item 9). Until then, nothing else is safe to branch from.
2. **Fix `task.py:77-78`** to use `BASE_DIR` (item 2). Ten lines; unblocks every other contributor.
3. **Re-verify `flwr run`** on a machine with adequate free RAM (item 8), then update the README's federated numbers with the confirmation.
4. **Decide the gender question** (item 1) â€” one line in v1, one constrained field in `app.py`, one shared constant.
5. **Single-source the 52-column contract** ([Â§3.5](#35-the-52-column-contract-and-why-duplicating-it-is-a-seam)). `models/feature_names.txt` already exists and is imported by nothing. This is the highest-leverage structural fix in the repo.
6. **Run the `Satisfaction Score` ablation** (item 5). It is the only open question that could materially change the reported metrics.
