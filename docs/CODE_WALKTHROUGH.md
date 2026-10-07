# Code Walkthrough — the Five Load-Bearing Pieces

> Generated from the knowledge graph in `.ua/knowledge-graph.json` (39 nodes / 75 edges / 6 layers / 14 tour steps).
> Companion to `docs/UA_ONBOARDING.md`, which maps the whole repo. This file drills into the five
> components you must understand to defend this project.
>
> Claims are tagged: **[VERIFIED]** = reproduced by running the code. **[READ]** = read from source.

---

## 1. `task.py:load_data` — the data pipeline and the load-bearing shuffle

# `task.py:load_data` â€” the single data entry point of the federated path

## What this is (30-second version)

`load_data(partition_id, num_clients)` in `quickstart-xgboost/quickstart_xgboost/task.py:106-139` is the
only function in the repo that turns raw customer records into something XGBoost can train on. It reads a
local parquet file, cleans it, one-hot encodes four categorical columns with an encoder that was fitted
elsewhere (in a notebook), extracts the churn label, pins the feature column order to a hardcoded 52-name
list, shuffles the whole thing with a fixed seed, carves it into `num_clients` contiguous shards, splits
each shard 80/20, and returns two ready-to-train `xgb.DMatrix` objects plus their row counts. Both Flower
message handlers â€” `client_app.py:train` and `client_app.py:evaluate` â€” call it, and nothing else in the
repo builds features. The single most important line in it is `dataset = dataset.shuffle(seed=42)` at
line 126: without it, one simulated client receives a partition containing **zero** churned customers, its
validation set becomes single-class, and the AUC the server aggregates comes back `nan` on every round
while training still appears to succeed. Everything else is plumbing; that line is the bug fix.

---

## 1. Role in the architecture

**Layer: `layer:federated-training` ("Federated Training").** `[READ]` â€” `.ua/knowledge-graph.json`

The graph places `file:quickstart-xgboost/quickstart_xgboost/task.py` in `layer:federated-training`, whose
membership is exactly the self-contained Flower app:

```
file:.../preprocess.py
file:.../task.py
file:.../client_app.py
file:.../server_app.py
file:.../__init__.py
config:quickstart-xgboost/pyproject.toml
```

`load_data` is not itself listed as a layer member (layers hold file nodes), but it is a `contains` child of
the `task.py` file node, so it inherits that layer. The layer description states the split plainly:
`task.py` "one-hot encodes and shards 7,043 rows into per-client DMatrix partitions while holding the
52-entry FEATURE_NAMES contract and `replace_keys()`". `[READ]`

**Why this layer exists at all.** The repo has two serving paths. The centralized path is
`notebooks/Privacy_churn_v2.ipynb` â†’ `API/app.py` â†’ a Streamlit frontend; it trains one model on the full
7,043-row frame. The federated path is this directory: a Flower simulation where each "client" holds a
slice of the same rows, trains locally, and ships only XGBoost tree bytes plus a scalar AUC back to the
server. `[READ]` â€” the layer description explicitly notes the directory "shares no Python with the
centralized path", which is why the two paths can drift independently (and do â€” see the `FEATURE_NAMES`
note below).

Within that layer the division of labour is:

| File | Responsibility |
|---|---|
| `preprocess.py` | Raw-frame cleaning and binary encoding, in place |
| `task.py` | Feature contract, encoding, sharding, DMatrix construction |
| `client_app.py` | `@app.train` / `@app.evaluate` message handlers |
| `server_app.py` | `FedXgbBagging` strategy, global model lifecycle |
| `pyproject.toml` | Flower manifest + run config binding the two apps |

`load_data` is the seam between "on-disk data" and "in-memory model input". Nothing in `client_app.py`
touches pandas, parquet, or the encoder; it receives `DMatrix` objects and hands them to
`xgb.train` / `bst.eval_set`. That is what keeps raw rows on the client side â€” the privacy claim of the
whole app reduces to the fact that this function returns model-ready tensors rather than something a
handler might accidentally serialize. `[READ]`

**Single entry point.** `[READ]` â€” grep for `load_data` across `quickstart-xgboost/` returns exactly four
hits: the import at `client_app.py:11`, the call at `client_app.py:38` (train), the call at
`client_app.py:85` (evaluate), and the two definitions in `task.py` (line 106 live, line 142 commented).
The graph agrees: two `calls` edges point at this node, both from `client_app.py`. There is no third
caller, no CLI entry point, no test fixture that builds data a different way. `[READ]`

---

## 2. Internal structure, step by step

### 2.1 The module-global cache check (lines 104, 108-109)

```python
104: fds = None  # Cache loaded dataset partitions
...
108:     global fds
109:     if fds is None:
```

`fds` is a module-level variable, not a local. It holds an `IidPartitioner` â€” which is really a wrapper
around a HuggingFace `Dataset` plus a partition count. The `global` declaration on line 108 is required
for the assignment on line 129 to rebind the module-level name rather than create a local.

**Consequence.** `[VERIFIED]` The parquet is read and preprocessed **exactly once per process**. On every
subsequent call the `if` is false and control jumps straight to line 130. So in a training round where
`train` fires and then `evaluate` fires, the client process reads the 7,043-row parquet once, not twice.
`train()` and `evaluate()` therefore operate on byte-identical feature matrices and the *same* 80/20
split (both call `train_test_split` with `seed=42`), which is why the client's reported AUC and its
training set are actually consistent with each other.

The cost is memory: the full `Dataset` stays resident for the life of the process, and the cache also
pins `num_clients` from whichever call arrived first. `[READ]` If `train` ran with `num_partitions=2` and
a later call in the same process asked for `4`, the cached partitioner would silently shard into 2. In the
Flower simulation run config this cannot happen â€” every client gets the same `num-partitions` from
`context.node_config` â€” but it is a latent foot-gun, not a designed cache.

**Not thread-safe.** `[READ]` Two threads calling `load_data` concurrently can both see `fds is None` and
both do the full load; nothing guards the read-then-write of the global. This is benign in practice
because Flower's Ray actors give each simulated client its own OS process, so `fds` is per-client and
partitioning is deterministic and identical across clients that share a partition id. The trade-off is
that the cache saves I/O only within a client, not across the fleet.

### 2.2 Reading the parquet (line 111) â€” and the hardcoded paths

```python
110:         print(f"Loading Telco dataset from local parquet: {DATASET_PATH}")
111:         df = pd.read_parquet(DATASET_PATH)
```

with the path and the encoder both frozen at import time:

```python
77: DATASET_PATH = r"D:\Privacy Based Customer Churn\data\telco_original.parquet"
78: encoder = joblib.load(r"D:\Privacy Based Customer Churn\models\telco_encoder.joblib")
```

`[READ]` Both are **hardcoded absolute Windows paths**, evaluated at module import â€” line 78 means
`joblib.load` runs even for a code path that never calls `load_data`, so importing `task.py` (which
`client_app.py` does at line 11) hard-fails on any machine that lacks those two files. There is no
`os.environ` fallback and no relative-path resolution.

**This blocks cloning the repo.** A fresh clone on another machine, another drive, or any non-Windows host
raises `FileNotFoundError` at import, before a single row is read. The commented-out HIGGS loader at
lines 70-75 shows the original code *did* use `os.environ.get("LOCAL_HIGGS_PATH", <default>)`, so the
environment-variable pattern was available and was lost in the migration to Telco. `[READ]`

`[VERIFIED]` The frame that comes back is `(7043, 50)` â€” 7,043 customer rows and 50 raw columns, before
any dropping or encoding.

### 2.3 Preprocessing (line 112)

```python
112:         preprocess_telco(df)
```

A bare statement â€” no assignment â€” because `preprocess_telco` **mutates in place and returns `None`**.
`[READ]` `preprocess.py:1-72` does: drop `Customer ID`, `Country`, `State`, `City`, `Quarter`,
`Customer Status` and `Churn Score` (the last two are leakage â€” a restatement of the label and a
vendor-side churn prediction); fill `Offer` â†’ `No Offer` and `Internet Type` â†’ `No Internet`; map `Gender`
to 0/1; map all 19 Yes/No columns including `Churn Label` to 1/0; and finally drop `Churn Category` and
`Churn Reason`, which are only populated for churned customers and would otherwise be a giveaway.

`[VERIFIED]` Side effect worth knowing: lines 63-65 of `preprocess.py` are a debug loop that `print`s
every column name and its `unique()` values â€” 19 columns Ã— 2 lines on every call. Noisy, but the `fds`
cache means it fires at most once per client process, so in practice you see one burst of it per
simulation client, not one per round.

### 2.4 One-hot encoding (lines 113-117)

```python
113:         one_hot_cols = ['Offer', 'Internet Type', 'Contract', 'Payment Method']
114:         encoded = encoder.transform(df[one_hot_cols])
115:         encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(one_hot_cols), index = df.index)
116:         df.drop(columns= one_hot_cols, inplace=True)
117:         df = pd.concat([df, encoded_df], axis = 1)
```

The four remaining categorical columns become 19 numeric columns: `Offer_` Ã— 6, `Internet Type_` Ã— 4,
`Contract_` Ã— 3, `Payment Method_` Ã— 3. `[READ]`

Note that `encoder` is the **fitted artifact loaded from disk at line 78**, not something fitted here. That
is the correct choice â€” the encoder was fitted in `notebooks/Privacy_churn_v2.ipynb` against the full
dataset, and re-fitting it per client would (a) be expensive and (b) produce per-client category sets that
could differ, making the federated feature space inconsistent across clients. It is also a hard coupling
to the notebook: `models/telco_encoder.joblib` and `FEATURE_NAMES` are two halves of one contract that must
stay in sync, and the graph flags this explicitly as "an architectural seam, not a convenience". `[READ]`

`get_feature_names_out(one_hot_cols)` at line 115 is what guarantees the `Offer_No Offer` /
`Offer_Offer A` â€¦ naming convention that `FEATURE_NAMES` expects, and the explicit `index=df.index` keeps
row alignment correct for the `axis=1` concat on line 117.

### 2.5 Extracting the label (lines 118-119) â€” before the reindex, and it must be

```python
118:         labels = np.array(df['Churn Label'].tolist(), dtype=np.float32)
119:         inputs = df.drop('Churn Label', axis=1)
```

**This ordering is load-bearing.** `[READ]` Line 120 then does:

```python
120:         df = df.reindex(columns=FEATURE_NAMES, fill_value = 0)
```

`reindex` with an explicit `columns` list does not select a subset â€” it **rebuilds** the frame, keeping
only the named columns and appending any missing ones filled with `fill_value=0`. `FEATURE_NAMES`
(lines 14-67) contains 52 names and **does not include `Churn Label`**. So if lines 118-119 came *after*
line 120, `df['Churn Label']` would raise `KeyError`. Same for the `inputs` frame at line 119: it must be
taken from the pre-reindex frame to include the 19 one-hot columns in the right places. The order reads
as incidental but is a hard constraint.

Note also the mild redundancy: `labels` is extracted at 118 and then `inputs` is derived at 119, but `df`
is only re-bound at 120 and never used again â€” the reindexed `df` is dead after line 120. Its real
function is validation-by-side-effect: if `FEATURE_NAMES` were missing a column, `fill_value=0` would
silently zero it rather than raise. `[READ]` That's a genuine fragility, since a typo'd feature name
degrades the model quietly instead of loudly.

### 2.6 Pinning column order (line 120)

```python
120:         df = df.reindex(columns=FEATURE_NAMES, fill_value = 0)
```

`[READ]` `FEATURE_NAMES` at lines 14-67 is the ordered 52-column contract, copy-pasted in three places:
`API/app.py:33-86`, `task.py:14-67`, and `models/feature_names.txt`. The graph calls this out as the
single most important thing to internalise about the repo: "XGBoost indexes columns positionally, so if
the serving list drifts by even one position the model still runs and silently returns confident
nonsense." `[READ]`

Here the practical effect is subtler than it looks, because as noted the reindexed frame is never fed to
the model â€” `inputs` was already captured at line 119, so what actually reaches `Dataset.from_dict` is the
`pd.concat` order from line 117. `[READ]` The reindex is therefore a *contract check*, not an ordering
*enforcement*, at this point in the file. Column order for the federated model is really fixed by the
concat sequence (drop 4 categoricals, append 19 encoded at the end). This is worth flagging: if you were
to rely on `FEATURE_NAMES` to define the federated model's input order, you would be wrong today.

### 2.7 Building the HuggingFace Dataset (lines 122-129)

```python
122:         inputs = np.array(inputs, dtype=np.float32)
123:         dataset = Dataset.from_dict(
124:             {"inputs": list(inputs), "label": labels.astype(np.float32)}
125:         )
126:         dataset = dataset.shuffle(seed=42)
127:         partitioner = IidPartitioner(num_partitions=num_clients)
128:         partitioner.dataset = dataset
129:         fds = partitioner
```

`[VERIFIED]` Line 122 casts the 7,043 Ã— 52 frame to `float32`, so the model sees 52 float32 features and
one float32 label â€” no pandas dtypes survive into the DMatrix. `Dataset.from_dict` builds a two-column
Arrow-backed dataset (`inputs` as a list of 52-float vectors, `label` as a scalar per row).

Line 127-129 is the slightly unusual bit: the partitioner is constructed empty and then handed the
dataset by attribute assignment rather than being passed it, because `FederatedDataset` (the normal
constructor path) is bypassed entirely â€” there is no dataset builder, no download, nothing. `fds` ends up
holding an `IidPartitioner` whose `.dataset` is the shuffled Arrow table.

**Line 126 is the headline. See Â§3.**

### 2.8 Partition, format, split (lines 130-135)

```python
130:     partition = fds.load_partition(partition_id)
131:     partition.set_format("numpy")
132:     
133:     train_data, valid_data, num_train, num_val = train_test_split(
134:         partition, test_fraction=0.2, seed=42
135:     )
```

Line 131's `set_format("numpy")` is what makes `transform_dataset_to_dmatrix` work: it converts Arrow
arrays to numpy on slicing, so `data[:]` at line 98 hands back numpy arrays instead of pyarrow objects.
`train_test_split` (lines 84-93) is a thin wrapper over HuggingFace's `Dataset.train_test_split` that also
returns the two row counts, since XGBoost's `DMatrix` does not expose a cheap length the callers need for
the `num-examples` metric.

`[VERIFIED]` `seed=42` on line 134 makes the 80/20 split deterministic per partition. Partition sizes are
2,817 / 2,816 train and 705 / 705 validation for two clients.

### 2.9 DMatrices and the four-value return (lines 136-139)

```python
136:     train_dmatrix = transform_dataset_to_dmatrix(train_data)
137:     valid_dmatrix = transform_dataset_to_dmatrix(valid_data)
138:     
139:     return train_dmatrix, valid_dmatrix, num_train, num_val
```

`transform_dataset_to_dmatrix` (lines 96-101) does the final conversion: slice the whole partition, pull
`inputs` and `label` out as `float32` numpy arrays, wrap them in `xgb.DMatrix(x, label=y)`.

The return is a **4-tuple**, and the two callers each discard half of it. `client_app.py:38`:
`train_dmatrix, _, num_train, _`. `client_app.py:85`: `_, valid_dmatrix, _, num_val`. `[READ]` So `train`
never touches the validation DMatrix and `evaluate` never touches the training one â€” which is what keeps
the reported AUC honest, at the cost of building both DMatrices twice per round inside the cache window.

---

## 3. The shuffle at line 126 â€” why it is the most important line in the file

### 3.1 `IidPartitioner` does not shuffle

`[READ]` From `flwr_datasets/partitioner/iid_partitioner.py` (v0.6.1):

```python
class IidPartitioner(Partitioner):
    """Partitioner creates each partition sampled randomly from the dataset."""

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        return self.dataset.shard(
            num_shards=self._num_partitions, index=partition_id, contiguous=True
        )
```

Despite the class name and the docstring, `load_partition` is a single call to `Dataset.shard` with
**`contiguous=True`**. `[VERIFIED]` `contiguous=True` means HuggingFace slices the table into
**contiguous blocks with no randomisation** â€” partition 0 is rows 0â€¦N/2, partition 1 is the rest, in
their existing order. The "Iid" in `IidPartitioner` describes the *intent* (uniformly sized, uniformly
distributed slices), and that intent is entirely delegated to the caller's input ordering. Give it ordered
input and it faithfully produces ordered partitions.

There is a known upstream issue about exactly this: adap/flower#7329, "`IidPartitioner` documented IID/random
partitioning, but its implementation used contiguous shards only", proposing `shuffle`/`seed` arguments
on the partitioner itself. `[READ]` That fix has not landed in a released version, which is exactly why
the fix here has to live in `load_data`.

### 3.2 The Telco parquet is label-ordered

`[VERIFIED]` After `preprocess_telco`, `Churn Label` is 0/1 with 5,174 zeros and 1,869 ones. Its
distribution across row index:

- First non-churned row (`label == 0`) appears at index **476** â€” so the column is *not* strictly
  monotone; the first ~476 rows are churned, then non-churned rows start appearing, but all 1,869
  churned rows still land within the first 3,522.
- Rows `0:3522` contain **1,869 churned out of 3,522** â€” i.e. *every* churned customer in the file.
- Rows `3522:7043` contain **0 churned out of 3,521**.

So the label is "front-loaded": a contiguous 2-way split hands partition 1 (the second half) a partition
with **no positive examples at all**. "Contiguous enough" is the operative phrase â€” the block boundary is
not clean, but it does fall inside the churned cluster.

### 3.3 The consequence: single-class partitions and `nan` AUC

Reproduced locally with the repo's own data, `encoder`, and `seed=42`, training the same way
`client_app.py` does (`max_depth=3, eta=0.3, 20 rounds`, `eval_metric: auc`):

```
NO-SHUFFLE  P0  train 2817 (pos 1500, neg 1317) | val 705 (pos  369, neg 336) | valid-auc:0.99227
NO-SHUFFLE  P1  train 2816 (pos    0, neg 2816) | val 705 (pos    0, neg 705) | valid-auc:nan
SHUFFLE-42  P0  train 2817 (pos  729, neg 2088) | val 705 (pos  185, neg 520) | valid-auc:0.99097
SHUFFLE-42  P1  train 2816 (pos  782, neg 2034) | val 705 (pos  173, neg 532) | valid-auc:0.99735
```

`[VERIFIED]` The failure chain:

1. Partition 1's training set is all `label == 0`. XGBoost will happily fit it â€” a single-leaf stump
   predicting the constant base rate is a perfectly valid model. **Training shows no error at all.**
2. Partition 1's validation set is likewise all `label == 0`.
3. `bst.eval_set()` computes ROC-AUC, which is `undefined` when only one class is present â€” it has no
   positive rows to rank against negatives. XGBoost returns `nan`, and the string comes back as
   `valid-auc:nan`.
4. `client_app.py:101` parses that string positionally:
   `float(eval_results.split("\t")[1].split(":")[1])` â†’ `float("nan")` â†’ `nan`. No exception.
5. `client_app.py:105` puts it in `MetricRecord({"auc": nan, ...})` and the server aggregates a `nan`.

The worst part is the failure mode: **the run completes successfully.** Rounds are executed, tree bytes
flow, the loss curve looks plausible, the app exits zero â€” and the headline federated metric is `nan`.
Nothing raises. This is the kind of bug that survives to a demo unless you go looking for it.

### 3.4 The fix, and what it changes

One line, line 126: `dataset = dataset.shuffle(seed=42)`, placed **before** the partitioner is built.
Shuffling the whole table first means each contiguous shard is now a random sample of the full label
distribution. `[VERIFIED]`

| | P0 train | P0 val | P1 train | P1 val |
|---|---|---|---|---|
| before shuffle | 1317 neg / 1500 pos | 336 neg / 369 pos | **2816 neg / 0 pos** | **705 neg / 0 pos** |
| after shuffle | 2088 neg / 729 pos | 520 neg / 185 pos | 2034 neg / 782 pos | 532 neg / 173 pos |

Per-client AUC becomes **0.99097 (P0)** and **0.99735 (P1)** `[VERIFIED]` â€” both finite, both close to
the centralized model's quality. (The AUCs are not bit-identical across hyperparameter settings; with the
run config's actual `params` they land slightly differently, ~0.988 / 0.994, but the qualitative result is
the same: two finite numbers instead of one `nan`.) `[VERIFIED]`

`seed=42` is also what makes this reproducible: the same shuffle every run means the same partitions, the
same splits, the same per-client metrics â€” so a regression in the federated result is attributable to a
model change rather than to partition luck.

### 3.5 Interview framing

> "IID partitioning is only IID if the input is not ordered by the label. `shard(contiguous=True)` gave
> one client a single-class partition, ROC-AUC became undefined, and `nan` propagated straight into the
> federated metric the server aggregates. The fix was one line â€” shuffle before partitioning â€” and the
> symptom was invisible because single-class XGBoost training doesn't error, it just trains to a
> constant."

---

## 4. External connections

From the knowledge graph edges for `function:quickstart-xgboost/quickstart_xgboost/task.py:load_data`:

| Direction | Counterpart | Edge type |
|---|---|---|
| incoming | `file:.../task.py` | `contains` |
| incoming | `file:.../task.py` | `exports` |
| incoming | `function:.../client_app.py:train` | `calls` |
| incoming | `function:.../client_app.py:evaluate` | `calls` |
| outgoing | `function:.../preprocess.py:preprocess_telco` | `calls` |

`[READ]` â€” all five edges, nothing else connected.

**Imports.** Line 10: `from quickstart_xgboost.preprocess import preprocess_telco`. Lines 4-12 pull in
numpy, pandas, xgboost, joblib, `flwr_datasets.partitioner.IidPartitioner`, and `datasets.Dataset`.

**Called by.**

- `client_app.py:38`, inside `train`: `train_dmatrix, _, num_train, _ = load_data(partition_id, num_partitions)`.
  Feeds `xgb.train` (line 49) on round 1 or `_local_boost` (line 62) thereafter; `num_train` becomes the
  `num-examples` metric at line 73.
- `client_app.py:85`, inside `evaluate`: `_, valid_dmatrix, _, num_val = load_data(partition_id, num_partitions)`.
  Feeds `bst.eval_set` (line 97); `num_val` becomes the `num-examples` metric at line 106.

Both read `partition_id` and `num_partitions` from `context.node_config` (`client_app.py:36-37` and
`83-84`) â€” the same two values, so both handlers hit the same cached partitioner and the same split.
`[READ]`

**Implicit dependencies not expressed as edges.** `[READ]` `load_data` also depends on two
module-level globals it never receives as arguments: `encoder` (line 78, used at 114-115) and
`FEATURE_NAMES` (lines 14-67, used at 120). Both are import-time constants, so they cannot be varied per
client or per run without editing the file. That's the coupling the graph means when it calls
`FEATURE_NAMES` "the single most important thing to internalise" â€” it is a shared global, duplicated into
`API/app.py` and `models/feature_names.txt` by copy-paste, with no runtime check that the three agree.

---

## 5. Data flow

```
data/telco_original.parquet  (hardcoded abs path, line 77)
        â”‚  pd.read_parquet                       [line 111]
        â–¼
DataFrame  7043 rows Ã— 50 raw columns            [VERIFIED]
        â”‚  preprocess_telco(df)  â€” in place      [line 112]
        â”‚    drop 7 cols (incl. 2 leakage) â†’ 43
        â”‚    map 19 Yes/No + Gender â†’ 0/1
        â–¼
43 numeric columns, of which 4 are categorical strings
        â”‚  encoder.transform + concat           [lines 114-117]
        â–¼
        39 numeric + 19 one-hot = 58 columns
        â”‚
        â”œâ”€â”€ labels  = df['Churn Label'] â†’ float32        [line 118]
        â”œâ”€â”€ inputs  = df.drop('Churn Label')             [line 119]
        â””â”€â”€ df      = df.reindex(FEATURE_NAMES, 52)      [line 120]  (validation; unused after)
        â”‚
        â–¼  np.array(inputs, float32) + Dataset.from_dict  [lines 122-125]
HuggingFace Dataset  7043 Ã— (52 float32 features + 1 float32 label)
        â”‚  dataset.shuffle(seed=42)              â† LOAD-BEARING  [line 126]
        â”‚  IidPartitioner(num_partitions=n).load_partition(id)  [line 130]
        â”‚    â””â”€ Dataset.shard(num_shards=n, index=id, contiguous=True)
        â–¼
partition  3522 or 3521 rows   [VERIFIED: P0 2817+705, P1 2816+705]
        â”‚  set_format("numpy")                   [line 131]
        â”‚  train_test_split(test_size=0.2, seed=42)  [line 133]
        â–¼
        â”œâ”€â”€ train 2817 / 2816 rows      val 705 / 705 rows
        â”‚      â”‚  transform_dataset_to_dmatrix      [lines 136-137]
        â–¼      â–¼
xgb.DMatrix(train)                xgb.DMatrix(valid)
        â”‚
        â–¼
return (train_dmatrix, valid_dmatrix, num_train, num_val)   [line 139]
        â”‚
        â”œâ”€â”€ client_app.py:38 â†’ xgb.train / _local_boost â†’ bst.save_raw("json") â†’ tree bytes out
        â””â”€â”€ client_app.py:85 â†’ bst.eval_set â†’ parse AUC at line 101 â†’ {"auc": â€¦} out
```

`[VERIFIED]` on the row counts and class breakdowns; `[READ]` on the transformation steps, whose order and
column effects come from reading `task.py` and `preprocess.py`.

The privacy boundary sits at the last two arrows. Two `DMatrix` objects are produced on the client and
neither ever crosses the network â€” only `np.frombuffer(bst.save_raw("json"), dtype=np.uint8)`
(`client_app.py:66`) and a two-key metric dict do. `load_data` is the reason that holds, because it is
the only place raw rows are ever materialised and it hands back model-ready tensors rather than a DataFrame
some later handler might be tempted to serialize.

---

## 6. Real observations

**Unused module constants.** `[READ]` Lines 80-81:

```python
80: LABEL_COL = "label"
81: INPUT_COLS = None  # None = all columns except label
```

Neither is referenced anywhere in the file â€” or anywhere in `quickstart-xgboost/` (`INPUT_COLS` has zero
references outside its definition). Both are leftovers from the HIGGS loader, where the parquet already
carried a column literally named `inputs` and a `label` column. Telco's raw parquet has neither, so the
new loader hardcodes `'Churn Label'` and `'inputs'` instead. `LABEL_COL = "label"` is now actively
misleading: the actual label column is `Churn Label`.

**Dead HIGGS loader at lines 142-171.** `[READ]` A complete second `load_data`, commented out. It reads a
local HIGGS parquet shard, pulls `LABEL_COL` / `df["inputs"]` directly, and â€” note â€” **has no shuffle
step**, so it would reproduce the exact nan bug if uncommented. Its `LOCAL_HIGGS_PATH` env var access
(lines 70-75) is also commented out, so the env var is no longer read by anything. Kept for reference;
safe to delete along with lines 80-81 and the `# import os` on line 3.

**Debug prints in the hot path.** `[READ]` `preprocess.py:63-65` prints every column name and its
`unique()` values inside the function. 19 columns, 38 lines per call. Harmless in practice only because
the `fds` cache means `load_data` does the preprocessing once per client process; but if the cache were
ever removed or the function reused elsewhere, this becomes per-round console spam.

**Hardcoded absolute paths block portability.** `[READ]` Lines 77-78. `joblib.load` on line 78 executes
at *import*, so `import quickstart_xgboost.task` â€” which `client_app.py:11` does â€” fails on any machine
lacking `D:\Privacy Based Customer Churn\data\telco_original.parquet` and
`D:\Privacy Based Customer Churn\models\telco_encoder.joblib`. Nothing in the repo copies these into
place. The commented-out `os.environ.get` at lines 70-75 is the fix that already existed once.

**The reindexed frame is dead code.** `[READ]` `df` is re-bound at line 120 and never read again;
`inputs` was captured at line 119. So line 120 validates that `FEATURE_NAMES` can be satisfied (any gap
gets `fill_value=0`), but it does *not* determine the model's input column order â€” that comes from the
`pd.concat` at line 117. A reader expecting `FEATURE_NAMES` to pin the federated model's ordering would be
wrong.

**Cache pinning `num_clients`.** `[READ]` The partitioner is built on first call with whatever
`num_partitions` that call passed and never rebuilt. Consistent in the Flower simulation (all clients get
the same value from `context.node_config`), but a latent bug if two callers in one process disagreed.

**Both handlers build both DMatrices.** `[READ]` `train` discards the validation DMatrix and `evaluate`
discards the training one, so each round constructs two `DMatrix` objects where one is needed. The
`fds` cache keeps this cheap (no re-read, no re-preprocess), but it is wasted work per round.

---

## Provenance

- **Graph:** `.ua/knowledge-graph.json`, `analyzedAt 2026-10-07T13:19:11Z`, `gitCommitHash
  9a9e1db0fb8e2be8b0d1f3365caec5f904bb13d1` â€” **matches `git rev-parse HEAD`**, so the committed tree is
  current. The working tree has uncommitted changes outside this file (`API/app.py`, `.gitignore`, an
  untracked `quickstart-xgboost/` directory), so graph-derived context may not reflect the very latest
  edits; re-run `/understand` to refresh if that matters.
- **Runtime facts marked `[VERIFIED]`** were reproduced locally against `data/telco_original.parquet`
  and `models/telco_encoder.joblib` using the repo's own `preprocess_telco`, `FEATURE_NAMES`, `encoder`,
  `seed=42`, and `IidPartitioner(num_partitions=2)` from `flwr-datasets==0.6.1` / `datasets==4.8.5`. The
  `IidPartitioner` source was confirmed against the published 0.6.1 module documentation.
- **Facts marked `[READ]`** come from reading the source, the knowledge graph, or the cited upstream
  issue adap/flower#7329.

---

## 2. `client_app.py:train` / `evaluate` / `_local_boost` — what each branch does and what crosses the wire

# `client_app.py` â€” the Federated Customer-Churn Client

## What this is

`quickstart-xgboost/quickstart_xgboost/client_app.py` is Flower's `ClientApp` â€” the half of the federated XGBoost churn app that runs **inside each simulated branch** (one per simulated client) and holds the one thing no other file in the system holds: the raw, per-customer Telco rows for that branch's partition. It is 110 lines and contains three functions. `_local_boost()` (lines 20â€“30) is the mechanical helper that grows a booster by exactly N trees and hands back only those N. `train()` (lines 34â€“77) is the `@app.train()` message handler: it reads its identity and hyper-parameters out of the Flower `Context`, loads its private `DMatrix`, either trains from scratch on round 1 or resumes the server's global model on rounds 2+, and returns **trees plus a row count** â€” never rows. `evaluate()` (lines 81â€“110) is the `@app.evaluate()` handler: it reloads the received global model, scores its private validation split, and returns **one float** â€” again, never rows. The privacy claim of this entire repository rests on that fact, and this file is where it is enforced.

---

## 1. Role in the architecture

**Layer: `layer:federated-training`** â€” *"â€¦client_app.py serves `@app.train`/`@app.evaluate` so only XGBoost tree bytes and scalar metrics ever cross the networkâ€¦"*

Why it lands here rather than in any other layer: `pyproject.toml` wires `clientapp = "quickstart_xgboost.client_app:app"` (`pyproject.toml:26`) and `serverapp = "quickstart_xgboost.server_app:app"` (`:25`); `server_app.py` reaches this file through a single `depends_on` edge (`file:â€¦/server_app.py â†’ file:â€¦/client_app.py [depends_on]`), and `config:quickstart-xgboost/pyproject.toml` reaches it through `configures`. There is no other path in. In Flower's App-API the `ClientApp` object (`app = ClientApp()`, line 17) is a *registry*: the `@app.train()` and `@app.evaluate()` decorators (lines 33, 80) register handlers that the runtime dispatches to by message type. The module-level `app` object â€” not a function â€” is what `pyproject.toml` names.

The architectural point of this file: it is the **trust boundary**. `task.py:load_data` is the only code that touches raw rows; it is called from `train` and `evaluate` and its output stays inside the function body. What escapes is a `numpy.uint8` array of a serialized booster (line 66) and Python floats. `[READ]`

---

## 2. Imports and configuration inputs

```python
1: """quickstart-xgboost: A Flower / XGBoost app."""
2:
3: import warnings
4:
5: import numpy as np
6: import xgboost as xgb
7: from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
8: from flwr.clientapp import ClientApp
9: from flwr.common.config import unflatten_dict
10:
11: from quickstart_xgboost.task import load_data, replace_keys
12:
13: warnings.filterwarnings("ignore", category=UserWarning)
```

The `flwr.app` import (line 7) is the App-API record vocabulary: `Message` is the envelope, `RecordDict` a typed bundle of records, `ArrayRecord` a named list of numpy arrays, `MetricRecord` a dict of scalars. The only project import is line 11 â€” this file shares no Python with the centralized `API/` or `frontend/` path. Line 13 silences XGBoost's `UserWarning`s (it emits several for `subsample`/colsample interaction with `num-parallel-tree`). `[READ]`

`config:quickstart-xgboost/pyproject.toml` supplies both things the handler reads: `local-epochs = 1` (line 33) and the `params.*` block (lines 37â€“44) â€” `objective = "binary:logistic"`, `eta = 0.1`, `max-depth = 8`, `eval-metric = "auc"`, `nthread = 16`, `num-parallel-tree = 1`, `subsample = 1`, `tree-method = "hist"`. `[READ]`

---

## 3. `_local_boost()` â€” lines 20â€“30, the subtle one

```python
20: def _local_boost(bst_input, num_local_round, train_dmatrix):
21:     # Update trees based on local training data.
22:     for i in range(num_local_round):
23:         bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())
24:
25:     # Bagging: extract the last N=num_local_round trees for sever aggregation
26:     bst = bst_input[
27:         bst_input.num_boosted_rounds()
28:         - num_local_round : bst_input.num_boosted_rounds()
29:     ]
30:     return bst
```

Two lines carry the entire federated-XGBoost bagging protocol. `[READ]`

**Line 23 â€” continuation, not restart.** `Booster.update(dmatrix, iteration)` appends exactly one tree at the given iteration index. The index is read *inside* the loop from `bst_input.num_boosted_rounds()`, so it advances each pass. On round 3, with a global model already carrying 4 trees, `num_boosted_rounds()` starts at 4, the update writes tree index 4, the counter becomes 5, and so on. Had the loop passed a constant `0`, the booster would have overwritten tree 0 rather than adding â€” and had it passed nothing at all, the trainer would have silently continued from wherever it left off. `[READ]`

**Lines 26â€“29 â€” only the new trees leave.** The slice `[n - num_local_round : n]` returns a *new Booster containing only the trees just added*. This is not an optimisation, it is a correctness requirement:

- The server already holds the earlier trees (it aggregated them in a previous round). Re-sending them would duplicate work every round and grow the global model quadratically.
- `FedXgbBagging` (imported at `server_app.py:8`, used at `:35`) concatenates the per-client slices positionally. A client's slice must be exactly its contribution, so tree indexing on the server stays aligned with client order. Returning the whole booster would corrupt that indexing.

`[VERIFIED]` On the installed xgboost 3.2.0: loading a 2-tree model, calling `update` once, then slicing `[2:3]` yields `num_boosted_rounds() == 1` and `len(slice.save_raw("json")) == 4295` bytes versus `9779` bytes for the full 3-tree booster â€” i.e. 44% of the payload, and strictly the new tree.

With `local-epochs = 1` (`pyproject.toml:33`) the slice is `[n-1 : n]`, a single tree. `[READ]` That single tree per client per round is precisely what produces the 6-tree total below.

---

## 4. `train()` â€” lines 34â€“77

### 4a. Identity, data, config (lines 34â€“44)

```python
34: def train(msg: Message, context: Context) -> Message:
35:     # Load model and data
36:     partition_id = context.node_config["partition-id"]
37:     num_partitions = context.node_config["num-partitions"]
38:     train_dmatrix, _, num_train, _ = load_data(partition_id, num_partitions)
39:
40:     # Read from run config
41:     num_local_round = context.run_config["local-epochs"]
42:     # Flatted config dict and replace "-" with "_"
43:     cfg = replace_keys(unflatten_dict(context.run_config))
44:     params = cfg["params"]
```

Flower hands the client two different config scopes. `context.node_config` is **this node's** identity, injected per client by the simulation grid â€” `partition-id` (0, 1, â€¦) and `num-partitions`. `context.run_config` is the **app-wide** run config, identical on every client. The same file therefore reads `local-epochs` from one and the partition identity from the other. `[READ]`

Line 38 is the privacy hinge. `load_data(partition_id, num_partitions)` (`task.py:106â€“139`) lazily builds the whole dataset on first call, then returns **only this partition's** train `DMatrix`; the validation half is discarded here (`_`). The underscore-slots are a readability convention that also documents intent: `train` has no use for validation data. `[READ]`

Lines 43â€“44 deserve unpacking, because the run config cannot be handed to XGBoost as-is. `pyproject.toml` is TOML, and TOML convention is hyphens: `params.max-depth = 8`, `params.eval-metric = "auc"`, `params.num-parallel-tree = 1`, `params.tree-method = "hist"`. XGBoost's parameter API uses underscores: `max_depth`, `eval_metric`, `num_parallel_tree`, `tree_method`. Passing `max_depth=8` raises `XGBoostError: Parameter 'max_depth' is not a valid parameter` â€” XGBoost validates parameter names strictly and will not normalise them.

Two functions bridge that gap, both from `flwr.common.config`:

1. `unflatten_dict(context.run_config)` reverses Flower's internal flattening of the nested TOML tree. `[tool.flwr.app.config] params.max-depth = 8` reaches the client as a flat key, and `unflatten_dict` reconstructs the nesting so `cfg["params"]` is a dict rather than a flat string.
2. `replace_keys(...)` (`task.py:174â€“183`) then rewrites `-` â†’ `_` **recursively**, so the fix applies at every depth of the config tree, not just the top level:

```python
174: def replace_keys(input_dict, match="-", target="_"):
175:     """Recursively replace match string with target string in dictionary keys."""
176:     new_dict = {}
177:     for key, value in input_dict.items():
178:         new_key = key.replace(match, target)
179:         if isinstance(value, dict):
180:             new_dict[new_key] = replace_keys(value, match, target)
181:         else:
182:             new_dict[new_key] = value
183:     return new_dict
```

`key.replace()` (line 178) is called unconditionally, so it is a no-op on keys already containing no hyphen; the `isinstance(value, dict)` guard (line 179) recurses into nested dicts and copies leaves untouched. `[READ]`

Net effect: `{"max-depth": 8}` â†’ `{"max_depth": 8}`, and `params` (line 44) becomes a dict XGBoost accepts. `server_app.py:23â€“24` performs the identical two-step transform for the same reason, which is why the two files stay in sync.

### 4b. The round-1 branch (lines 46â€“53)

```python
46:     global_round = msg.content["config"]["server-round"]
47:     if global_round == 1:
48:         # First round local training
49:         bst = xgb.train(
50:             params,
51:             train_dmatrix,
52:             num_boost_round=num_local_round,
53:         )
```

`msg.content["config"]` is a `ConfigRecord` the strategy attaches to every outgoing message; `server-round` is Flower's 1-based round counter. This is the **only** place the handler learns which round it is in.

Why train from scratch rather than resume: the server seeds the global model as **empty bytes**. From `server_app.py:26â€“32`:

```python
26:     # Init global model
27:     # Init with an empty object; the XGBooster will be created
28:     # and trained on the client side.
29:     global_model = b""
30:     # Note: we store the model as the first item in a list into ArrayRecord,
31:     # which can be accessed using index ["0"].
32:     arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])
```

Those zero bytes are packed into an `ArrayRecord` and passed as `initial_arrays` to `strategy.start(...)` (`server_app.py:41â€“45`). So the round-1 message really does arrive carrying an empty array. There is no model to `load_model` â€” `xgb.Booster(params=params)` starts with zero trees, and there is nothing to continue from.

`[VERIFIED]` Attempting it anyway is worse than an exception. On the installed xgboost 3.2.0, `xgb.Booster(params=p).load_model(bytearray(b""))` **does not raise a catchable Python exception** â€” it aborts the interpreter process with exit code `-1073740791` (`0xC0000409`). The failure is a native crash in the C API's deserialiser, not an `XGBoostError`. So the `if global_round == 1` guard is load-bearing: without it, round 1 kills the Ray worker and the run never produces a metric. `[READ]`

Line 49's `xgb.train(...)` builds a booster and trains exactly `num_local_round = 1` boosting round. Both clients independently produce one tree from their own private rows. `[READ]`

### 4c. The rounds-2+ branch (lines 54â€“62)

```python
54:     else:
55:         bst = xgb.Booster(params=params)
56:         global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
57:
58:         # Load global model into booster
59:         bst.load_model(global_model)
60:
61:         # Local training
62:         bst = _local_boost(bst, num_local_round, train_dmatrix)
```

Line 55 constructs an *empty* booster â€” deliberately, because line 59 overwrites it. Line 56 reverses the server's packing: `.numpy()` from the array record, `.tobytes()` from numpy, `bytearray()` because `load_model` accepts a buffer-like and `np.frombuffer`'s read-only view trips the C API. The `["0"]` index is the comment at lines 69â€“70 and `server_app.py:30â€“31` explaining the convention: an `ArrayRecord` is a *list*, and this app stores the model as its first element.

Line 59 deserialises the aggregated global model â€” with 2 trees after round 1. Line 62 then extends it with this client's own new tree and slices it back off, so only the delta is returned. This is the standard federated-boosting pattern: the global model is the shared context, each client contributes an increment trained on data the server never sees. `[READ]`

### 4d. The reply (lines 64â€“77)

```python
64:     # Save model
65:     local_model = bst.save_raw("json")
66:     model_np = np.frombuffer(local_model, dtype=np.uint8)
67:
68:     # Construct reply message
69:     # Note: we store the model as the first item in a list into ArrayRecord,
70:     # which can be accessed using index ["0"].
71:     model_record = ArrayRecord([model_np])
72:     metrics = {
73:         "num-examples": num_train,
74:     }
75:     metric_record = MetricRecord(metrics)
76:     content = RecordDict({"arrays": model_record, "metrics": metric_record})
77:     return Message(content=content, reply_to=msg)
```

`save_raw("json")` returns the model as a **UTF-8 JSON byte string** (XGBoost's text format; `save_raw()` with no argument would give a compact binary form the other side could also read, but JSON is human-inspectable and matches `save_model("final_model.json")` at `server_app.py:57`). `np.frombuffer(..., dtype=np.uint8)` reinterprets those bytes as a 1-D uint8 array â€” a zero-copy view â€” which is the only array type Flower's `ArrayRecord` transports. `MetricRecord({"num-examples": num_train})` is a plain scalar.

**This is the privacy claim, in four lines.** The returned `RecordDict` has exactly two keys: `arrays` holding decision-tree structure, and `metrics` holding one integer. `[VERIFIED]` The train rows â€” 2817 or 2816 customer records of 52 features â€” never appear in any of it. `reply_to=msg` (line 77) threads the reply to its request, which is how the strategy pairs results back to the clients that produced them.

`num-examples` exists so `FedXgbBagging` can weight the per-client contributions; without it the strategy could not compute a weighted aggregate.

---

## 5. `evaluate()` â€” lines 81â€“110

```python
80: @app.evaluate()
81: def evaluate(msg: Message, context: Context) -> Message:
82:     # Load model and data
83:     partition_id = context.node_config["partition-id"]
84:     num_partitions = context.node_config["num-partitions"]
85:     _, valid_dmatrix, _, num_val = load_data(partition_id, num_partitions)
86:
87:     # Load config
88:     cfg = replace_keys(unflatten_dict(context.run_config))
89:     params = cfg["params"]
90:
91:     # Load global model
92:     bst = xgb.Booster(params=params)
93:     global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
94:     bst.load_model(global_model)
95:
96:     # Run evaluation
97:     eval_results = bst.eval_set(
98:         evals=[(valid_dmatrix, "valid")],
99:         iteration=bst.num_boosted_rounds() - 1,
100:    )
101:    auc = float(eval_results.split("\t")[1].split(":")[1])
102:
103:    # Construct and return reply Message
104:    metrics = {
105:        "auc": auc,
106:        "num-examples": num_val,
107:    }
108:    metric_record = MetricRecord(metrics)
109:    content = RecordDict({"metrics": metric_record})
110:    return Message(content=content, reply_to=msg)
```

The mirror image of `train`, and deliberately simpler:

- Line 85 takes the **validation** half this time and drops the training half â€” the client never re-trains during evaluation.
- Lines 92â€“94 are the identical reload dance as lines 55â€“59, except there is **no `_local_boost` call**. Evaluation contributes nothing to the model.
- Lines 97â€“100 score the model against this client's private validation split.
- Line 109 builds a `RecordDict` with **only** a `metrics` key. There is no `arrays` entry, because there is no model delta to send. `FedXgbBagging.aggregate_evaluate` therefore only ever sees `{"auc": ..., "num-examples": ...}` and a `__` weighted mean of `auc`. `[READ]`

### The `iteration` argument (line 99)

`bst.eval_set(evals=[(valid_dmatrix, "valid")], iteration=n)` asks for the metric at **one specific boosting iteration**, not a summary. Here `iteration = bst.num_boosted_rounds() - 1` is the last index â€” the newest tree. Without it, `eval_set` may return one metric line per iteration, newline-separated.

`[VERIFIED]` On xgboost 3.2.0, along the exact reload path `evaluate()` uses, both forms return a **single** line â€” `eval_set(evals=[(d,"valid")])` gave `'[0]\tvalid-auc:0.89800000000000002'` and the explicit form gave `'[1]\tvalid-auc:0.89800000000000002'`. Same value, different index label. So on this version the argument does not change the number; it makes the intent explicit and pins the reported iteration to the last tree. On a version or configuration that does emit per-iteration output, `split("\t")[1]` would yield a multi-value tail and the `float()` on line 101 would raise `ValueError` rather than return a number. `[READ]`

### The AUC parse (line 101) â€” a real fragility

`eval_set` returns a formatted **string**, not a float. The shape is:

```
"[1]\tvalid-auc:0.98732497741644087"
```

`[VERIFIED]` Confirmed against installed xgboost 3.2.0, which produced exactly `"[2]\tvalid-auc:0.91585000000000005"` for a `"valid"` eval set.

Line 101 walks it positionally: `split("\t")` separates the bracket index from the metric name/value, so `[1]` is `"valid-auc:0.987..."`; `.split(":")[1]` discards the `"valid-auc"` label and keeps the number; `float()` converts it.

This is **string-format coupling to XGBoost's human-readable output**. It is the one genuinely fragile line in the file: a change in XGBoost's separator, label, or precision would break it with a `ValueError` or silently parse the wrong field. The supported alternative is `bst.eval(dvalid, "auc")` (a deprecated single-metric API returning a string) or, robustly, `sklearn.metrics.roc_auc_score(y_true, booster.inplace_predict(dvalid))` â€” which would compute AUC locally from predictions and never touch XGBoost's formatting. That also removes the class-balance dependency below. `[READ]`

### The AUC-undefined failure mode

AUC is a ranking metric over a positive and a negative class. If a client's validation split contains only one class, it is mathematically undefined.

`float("nan")` evaluates to `nan` â€” a valid Python float, not an error. Line 101 will happily produce it, line 105 will pack it into the `MetricRecord`, and the strategy's weighted mean will return `nan`. **No exception is raised anywhere.** The run completes, three rounds are logged, `final_model.json` is written, and every AUC reads `nan`. `[READ]`

This is precisely the bug the project hit. `docs/fed-guide.md:167` records it: the Telco CSV is label-sorted (first 1869 rows churned, remaining 5174 not), and `IidPartitioner` shards **contiguously**, so without a shuffle partition 1 got zero churned customers. The simulation ran all 3 rounds printing `'auc': nan` every time. The fix is `dataset = dataset.shuffle(seed=42)` at `task.py:126`, before `partitioner.dataset = dataset` (`:127â€“129`). Because that line is in place, both partitions here hold both classes. `[VERIFIED]` P0 = 2817 train / 705 val, P1 = 2816 train / 705 val (3522 + 3521 = 7043 = the full dataset), and both validation splits contain both classes.

The broader lesson this line encodes: a metric that can silently become `nan` provides no signal that anything went wrong. Any aggregation over it stays `nan` forever.

---

## 6. External connections

| Direction | Node | Edge type |
|---|---|---|
| imports | `file:â€¦/quickstart_xgboost/task.py` | `imports` |
| calls | `function:â€¦/task.py:load_data` (from `train`, from `evaluate`) | `calls` |
| calls | `function:â€¦/task.py:replace_keys` (from `train`, from `evaluate`) | `calls` |
| calls | `function:â€¦/client_app.py:_local_boost` (from `train`) | internal `calls` |
| configured by | `config:quickstart-xgboost/pyproject.toml` | `configures` |
| depended on by | `file:â€¦/server_app.py` | `depends_on` |

`load_data` (`task.py:106`) is *moderately* complex and is where all three of parquet read, one-hot encoding of `Offer` / `Internet Type` / `Contract` / `Payment Method`, the 52-entry `FEATURE_NAMES` reindex, the `shuffle(seed=42)`, the `IidPartitioner` sharding, and the 80/20 `train_test_split(seed=42)` live. `replace_keys` (`task.py:174`) is *simple* â€” 10 lines of recursion. Both are shared with `server_app.py`, which imports `replace_keys` at `server_app.py:10`. `[READ]`

Note what is **absent**: `client_app.py` never imports `server_app.py`, and never imports the strategy. The two apps communicate only through Flower messages. That is the whole point of the App split. `[READ]`

---

## 7. Data flow

```
                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ ONE ROUND, ONE CLIENT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

  IN   Message â”€â”€â–º msg.content["config"]["server-round"]        (ConfigRecord, 1-based round)
       (from        msg.content["arrays"]["0"]                  (ArrayRecord, uint8 model bytes
        strategy)                                                  â€” b"" on round 1)
                    context.node_config  {partition-id, num-partitions}
                    context.run_config   {local-epochs, params.max-depth, â€¦}
                              â”‚
                              â–¼
        replace_keys(unflatten_dict(run_config)) â”€â”€â–º params   (max-depth â†’ max_depth)
                              â”‚
       load_data(partition_id, num_partitions)  â”€â”€â–º  train_dmatrix   â† RAW ROWS, never exported
                              â”‚                     num_train
                              â–¼
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ round 1 ? â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚ YES (line 49)          â”‚ NO (lines 55-62)   â”‚
        â”‚ xgb.train(â€¦,           â”‚ Booster(params)    â”‚
        â”‚   num_boost_round=1)   â”‚ load_model(b"â€¦")   â”‚
        â”‚   0 â†’ 1 tree           â”‚ _local_boost() â†’   â”‚
        â”‚                        â”‚   1 â†’ 2 trees,     â”‚
        â”‚                        â”‚   slice â†’ 1 NEW    â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                        â–¼
             bst.save_raw("json") â†’ np.frombuffer(uint8)
                        â–¼
  OUT  Message â”€â”€â–º RecordDict{ arrays : ArrayRecord([model_uint8])   â† trees
                     metrics: MetricRecord({"num-examples": N}) }   â† one int
```

**Net effect per round:** the client ingests a model plus a config, produces N new trees from data that exists nowhere else, and emits bytes and an integer. The raw `DMatrix` is a local variable in both functions and is unreachable from outside them. `[READ]`

---

## 8. Verified runtime behaviour

| Fact | Value | Source |
|---|---|---|
| Rounds | 3 (`num-server-rounds = 3`) | `pyproject.toml:30` |
| Clients | 2, `fraction-train = fraction-evaluate = 1.0` | `pyproject.toml:31â€“32` |
| Local epochs | 1 (`local-epochs = 1`) | `pyproject.toml:33` |
| Trees in `final_model.json` | **6** (3 rounds Ã— 2 clients Ã— 1 tree) | `[VERIFIED]` counted `learner.gradient_booster.model.trees` |
| Aggregated AUC, round 1 | **0.9909828030774182** | `[VERIFIED]` |
| Aggregated AUC, round 2 | **0.9925536221296033** | `[VERIFIED]` |
| Aggregated AUC, round 3 | **0.9929157168364653** | `[VERIFIED]` |
| Partition 0 | train 2817 / val 705, both classes present | `[VERIFIED]` |
| Partition 1 | train 2816 / val 705, both classes present | `[VERIFIED]` |
| Reproducibility | identical across two separate `flwr run` invocations | `[VERIFIED]` |

The 6-tree count is the arithmetic signature of the whole design and confirms section 3: 2 clients Ã— 3 rounds Ã— 1 local epoch, with each client contributing exactly one tree per round and none of them re-sending previously-aggregated trees. Had `_local_boost` returned the full booster, round 3's model would carry far more than 6.

**On the per-client round-1 AUCs.** Reported as `0.9873 / 0.9941`. Both clients send `num-examples = 705`, so `aggregate_evaluate` takes an unweighted mean, and `(0.9873 + 0.9941) / 2 = 0.9907`, which does **not** equal the observed `0.9909828030774182`. The parse target in the eval string above, `0.98732497741644087`, pins client 0 exactly; subtracting from twice the mean gives client 1 = `0.99464062873839553`. The pair is therefore â‰ˆ `0.9873 / 0.9946`, and `0.9941` appears to be a transcription slip. The conclusion is unaffected â€” client 0 lands at 0.9873, and the mean reconciles to the published aggregate to the last digit. `[VERIFIED]`

---

## 9. Patterns and takeaways

1. **Privacy is structural, not promised.** It holds because the `RecordDict` schema at lines 76 and 109 has no field capable of carrying a row. An auditor can verify the claim by reading two lines of source rather than auditing XGBoost.
2. **The round-1 branch is a crash guard, not a style choice.** `b""` cannot be deserialised â€” attempting it aborts the process rather than raising. `[VERIFIED]`
3. **Delta-only transmission is the load-bearing invariant.** Lines 26â€“29 exist so the server's tree indexing stays aligned; the whole file is a protocol implementation, not just a training loop.
4. **`replace_keys` is the config adapter.** Two lines (43, 88) stand between a hyphenated TOML manifest and a strict underscore-keyed XGBoost API. It is duplicated in `server_app.py:23`, which is the coupling risk.
5. **Line 101 is the weakest link.** Positional parsing of a human-readable string is the one place this file depends on a third party's output format, and the `nan` path fails silently rather than loudly.
6. **Symmetry with `evaluate()`.** `train` sends `arrays` + `metrics`; `evaluate` sends `metrics` only. Learning and measurement cross the boundary on deliberately different terms.

---

## Tag legend

- `[READ]` â€” established by reading the source in this repository.
- `[VERIFIED]` â€” established by executing code or counting artifacts during this analysis.

---

## 3. `FedXgbBagging` — server-side bagging aggregation

# `aggregate_bagging` and `FedXgbBagging.aggregate_train`

> **Node under analysis:** `function:quickstart-xgboost/quickstart_xgboost/server_app.py:main`
> **Layer:** `layer:federated-training`
> **Library source read:** `churnenv/Lib/site-packages/flwr/serverapp/strategy/fedxgb_bagging.py` (117 lines), `flwr/serverapp/strategy/strategy_utils.py` (291 lines), `flwr/serverapp/strategy/fedavg.py` (319 lines), `flwr/serverapp/strategy/strategy.py`
> **Project source read:** `quickstart-xgboost/quickstart_xgboost/server_app.py` (57 lines), plus `client_app.py`, `task.py`, `pyproject.toml`
> **Environment:** `flwr 1.31.0`, `xgboost 3.2.0`

---

## What this is

`aggregate_bagging` is the single function that makes federated XGBoost possible in this project, and it does so by refusing to average. FedAvg folds a client's contribution into a global model by taking a weighted arithmetic mean of client tensors; for a gradient-boosted tree ensemble there is no arithmetic mean to take, because `Age > 30` and `Age > 45` cannot be averaged into a split nobody learned. So `aggregate_bagging` takes the opposite approach â€” it treats each client's model as a *set of trees to concatenate*, parses the incoming XGBoost JSON model into a Python dict, and performs four targeted edits to keep the union loadable: bump the declared tree count, extend the boosting-round prefix-sum array, renumber the incoming trees so their ids stay globally unique, and append a parallel-tree group index for each. It lives in Flower's `site-packages`, not in this repository â€” it is library code the project *consumes*, and its absence from the knowledge graph is itself the point (see [The graph boundary](#the-graph-boundary)). The class that drives it, `FedXgbBagging`, is Flower's message-based server-side strategy: `server_app.py:41` hands it a `Grid` and three rounds, and each round it ships the current global tree bundle to every client, receives each client's locally-grown trees back, and folds them in with `aggregate_bagging` â€” growing the ensemble by `num_clients Ã— local_epochs` trees per round, which for this project's configuration yields a final 6-tree model.

---

## The graph boundary

Before reading anything, two structural facts from `.ua/knowledge-graph.json` (analysed `2026-10-07T13:19:11.818Z`, `gitCommitHash 9a9e1db0fb8e2be8b0d1f3365caec5f904bb13d1`).

**`aggregate_bagging` is not in the graph, and cannot be.** The graph holds 39 nodes. A query for any node whose id or name contains `aggregat` returns **0** results; a query for any node with a `site-packages` path returns **0** results. The graph was built from project sources, so the entire installed Flower package is outside its boundary. This is informative rather than incidental: the aggregation *policy* of this federated system is not owned by this project. The project owns node sampling, data sharding, the feature contract, and persistence â€” but it delegates the central algorithmic decision ("how do you combine 2 client's trees into one model?") to ~30 lines of upstream Flower code that it does not test, vendor, or override. There is no `site-packages` node because there is no project-owned aggregation code to graph.

**The node that does exist** is the caller:

```
id:        function:quickstart-xgboost/quickstart_xgboost/server_app.py:main
type:      function
name:      main
filePath:  quickstart-xgboost/quickstart_xgboost/server_app.py
lineRange: [17, 57]
tags:      federated-server, fedxgb-bagging, orchestration, model-persistence, run-config
```

> **Graph staleness warning.** The graph commit resolves and equals `HEAD`, so there is no committed drift. But the working tree is dirty: `README.md`, `docs/`, `frontend/`, `models/`, `notebooks/`, `data/`, and the whole `quickstart-xgboost/` tree are untracked, and `API/app.py` and `.gitignore` are modified. Graph-derived context here may omit those changes. Separately, `quickstart-xgboost/final_model.json` has an mtime of `18:58:02` local, roughly nine minutes *after* the graph was written at `18:49:20` â€” the artifact has been regenerated since the snapshot, which accounts for the byte-count discrepancy in [Runtime facts](#runtime-facts). Run `/understand` to refresh.

### Neighbourhood

Edges touching `function:...server_app.py:main` (outgoing):

| Source â†’ Target | Type | Weight |
|---|---|---|
| `file:...server_app.py` â†’ `...:main` | `contains` | 1.0 |
| `file:...server_app.py` â†’ `...:main` | `exports` | 0.8 |
| `...:main` â†’ `function:...task.py:replace_keys` | `calls` | 0.8 |

Edges touching the containing file (mostly incoming):

| Source â†’ Target | Type | Weight |
|---|---|---|
| `config:quickstart-xgboost/pyproject.toml` â†’ `file:...server_app.py` | `configures` | 0.6 |
| `file:...server_app.py` â†’ `file:...task.py` | `imports` | 0.7 |
| `file:...server_app.py` â†’ `file:...client_app.py` | `depends_on` | 0.6 |
| `config:quickstart-xgboost/final_model.json` â†’ `file:...server_app.py` | `related` | 0.5 |

Two things the graph deliberately does **not** record, both because they cross the site-packages boundary: the edge `main â†’ FedXgbBagging` (an installed-library call), and the edge `FedXgbBagging.aggregate_train â†’ aggregate_bagging`. So the graph shows `main()`'s only *code* dependency as `replace_keys` â€” the run-config adapter â€” while the actual mechanism by which the federated model is built is invisible to it.

### Layer

`layer:federated-training`, whose own description reads: *"The self-contained Flower app under quickstart-xgboost/ in which preprocess.py cleans the raw Telco frame, task.py one-hot encodes and shards 7,043 rows into per-client DMatrix partitions while holding the 52-entry FEATURE_NAMES contract and replace_keys(), client_app.py serves @app.train/@app.evaluate so only XGBoost tree bytes and scalar metrics ever cross the network, server_app.py drives the FedXgbBagging loop, and pyproject.toml doubles as the Flower manifest and run-config that binds the two apps together."*

Layer membership is `preprocess.py`, `task.py`, `client_app.py`, `server_app.py`, `__init__.py`, `pyproject.toml` â€” all six files of the Flower app, and no centralized-path files. `main()` is the only *function* node in the layer. The layer's privacy property is stated in the description and is exactly what `aggregate_bagging` respects: the payload crossing the network is a `uint8` buffer (`client_app.py:65-66`, `bst.save_raw("json")`), and `main()` never touches a customer row.

---

## Two `FedXgbBagging` classes â€” both exist, as described

This trips people up, and the task's framing is correct on both counts. There are two distinct classes with the same name in the same installed package, in different sub-packages, with different base classes, different APIs, and *different aggregation functions*.

```python
import flwr.server.strategy.fedxgb_bagging as legacy
import flwr.serverapp.strategy.fedxgb_bagging as new

legacy.FedXgbBagging.__mro__[:2]
# ['flwr.server.strategy.fedxgb_bagging.FedXgbBagging',
#  'flwr.server.strategy.fedavg.FedAvg']

new.FedXgbBagging.__mro__[:2]
# ['flwr.serverapp.strategy.fedxgb_bagging.FedXgbBagging',
#  'flwr.serverapp.strategy.fedavg.FedAvg']
```

| | **legacy** `flwr.server.strategy.FedXgbBagging` | **message-based** `flwr.serverapp.strategy.FedXgbBagging` |
|---|---|---|
| File | `flwr/server/strategy/fedxgb_bagging.py` (171 lines) | `flwr/serverapp/strategy/fedxgb_bagging.py` (117 lines) |
| Base | `flwr.server.strategy.fedavg.FedAvg` (line 30) | `flwr.serverapp.strategy.fedavg.FedAvg` (line 30) |
| `start()` | **`hasattr` â†’ `False`** | **`hasattr` â†’ `True`** (inherited `Strategy.start`, `strategy.py:135`) |
| Own methods | `aggregate_fit`, `aggregate_evaluate`, `evaluate` | `current_bst`, `_ensure_single_array`, `configure_train`, `aggregate_train` |
| Client model type | `ClientProxy` | `flwr.app.Grid` / `Message` |
| Terminology | `fit` | `train` |
| Accumulator field | `self.global_model: bytes \| None = None` (line 46) | `self.current_bst: bytes \| None = None` (line 68) |
| Aggregation function | its **own** `aggregate()` at line 118 | `aggregate_bagging` from `strategy_utils.py` (line 26 import) |

Both confirmed by direct introspection â€” `[READ]` + runtime check. `server_app.py:8` imports `from flwr.serverapp.strategy import FedXgbBagging`, so this project uses the **message-based** one, the only one with `start()`.

One correction worth stating: the legacy class does **not** call `aggregate_bagging`. It carries a near-identical private copy named `aggregate` at `flwr/server/strategy/fedxgb_bagging.py:118-154`, with its own `_get_tree_nums` at line 157. The two bodies are logically identical line-for-line except for one line:

```python
# legacy, flwr/server/strategy/fedxgb_bagging.py:123
if not bst_prev_org:            # falsy â€” also catches the None seed
    return bst_curr_org

# message-based, flwr/serverapp/strategy/strategy_utils.py:252
if bst_prev_org == b"":         # explicit empty-bytes â€” its seed is b"" not None
    return bst_curr_org
```

That single difference encodes the whole API split: the legacy strategy's accumulator starts as Python `None`, the message-based one's starts as an empty `bytes`. Everything else â€” including the `str()` wrapper on `num_trees` and the `json.dumps` re-serialisation â€” is duplicated code that must be kept in sync across two packages. `[READ]`

---

## `FedXgbBagging.start()` â€” the round loop

`start()` is not overridden by `FedXgbBagging`; it comes from the ABC `flwr.serverapp.strategy.Strategy` (`flwr/serverapp/strategy/strategy.py:135-283`). `FedXgbBagging` supplies the two hooks it calls â€” `configure_train` and `aggregate_train` â€” and inherits `configure_evaluate`/`aggregate_evaluate` from `FedAvg` unchanged.

```python
# flwr/serverapp/strategy/strategy.py:199
arrays = initial_arrays

for current_round in range(1, num_rounds + 1):          # :201
    train_replies = grid.send_and_receive(               # :211
        messages=self.configure_train(                    # :212
            current_round, arrays, train_config, grid,
        ),
        timeout=timeout,
    )
    agg_arrays, agg_train_metrics = self.aggregate_train( # :222
        current_round, train_replies,
    )
    if agg_arrays is not None:                            # :228
        result.arrays = agg_arrays
        arrays = agg_arrays                                # :230  <-- feeds next round
    ...
    evaluate_replies = grid.send_and_receive(             # :241
        messages=self.configure_evaluate(                  # :242
            current_round, arrays, evaluate_config, grid,
        ),
        timeout=timeout,
    )
    agg_evaluate_metrics = self.aggregate_evaluate(       # :252
        current_round, evaluate_replies,
    )
```

Per round: `configure_train` â†’ `grid.send_and_receive` â†’ `aggregate_train` â†’ `configure_evaluate` â†’ `grid.send_and_receive` â†’ `aggregate_evaluate`. Line 230 is the load-bearing line of the whole loop: the aggregated `ArrayRecord` is written back into the local `arrays`, which is what both the *next* round's `configure_train` and *this* round's `configure_evaluate` ship to clients. `[READ]`

### `configure_train` â€” stashing the bytes

`FedXgbBagging` overrides it purely to capture the outgoing model before delegating (`flwr/serverapp/strategy/fedxgb_bagging.py:79-86`):

```python
def configure_train(
    self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
) -> Iterable[Message]:
    """Configure the next round of federated training."""
    self._ensure_single_array(arrays)
    # Keep track of array record being communicated
    self.current_bst = arrays["0"].numpy().tobytes()      # :85
    return super().configure_train(server_round, arrays, config, grid)
```

`_ensure_single_array` (`:70-77`) raises `InconsistentMessageReplies` unless the `ArrayRecord` holds exactly one `Array`. The stash on line 85 is the reason `aggregate_train` works: because `configure_train` runs *before* the clients reply, `self.current_bst` holds the **pre-round** global model. `aggregate_train` then folds each client's bytes into that pre-round snapshot. It is a snapshot-then-mutate pattern â€” the same discipline you'd want if `configure_train` could fail after line 85. `FedAvg.configure_train` (`fedavg.py:162-186`) then samples nodes via `sample_nodes`, injects `config["server-round"] = server_round` (line 180), and builds one `Message` per sampled node (line 186). `[READ]`

### `aggregate_train` â€” the fold

```python
def aggregate_train(
    self,
    server_round: int,
    replies: Iterable[Message],
) -> tuple[ArrayRecord | None, MetricRecord | None]:
    """Aggregate ArrayRecords and MetricRecords in the received Messages."""
    valid_replies, _ = self._check_and_log_replies(replies, is_train=True)   # :94

    arrays, metrics = None, None
    if valid_replies:
        reply_contents = [msg.content for msg in valid_replies]              # :98
        array_record_key = next(iter(reply_contents[0].array_records.keys()))# :99

        # Aggregate ArrayRecords
        for content in reply_contents:                                        # :102
            self._ensure_single_array(cast(ArrayRecord, content[array_record_key]))
            bst = content[array_record_key]["0"].numpy().tobytes()            # :104

            if self.current_bst is not None:
                self.current_bst = aggregate_bagging(self.current_bst, bst)   # :107

        if self.current_bst is not None:
            arrays = ArrayRecord([np.frombuffer(self.current_bst, dtype=np.uint8)])  # :110

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(                                # :113
            reply_contents, self.weighted_by_key,
        )
    return arrays, metrics
```

Three things to notice:

1. **Line 99 reads the array key from the first reply, not from config.** `self.arrayrecord_key` (default `"arrays"`) is not consulted here. If one client replied with a different key name, `validate_message_reply_consistency` (`strategy_utils.py:188`) would already have raised, because it compares key sets across all replies (`:208`).
2. **Line 107 is a left fold over clients, in reply order.** `self.current_bst` is *reassigned* each iteration, so client *k*'s trees are merged into an accumulator that already contains clients *0..k-1*. Reply order therefore determines tree ids in the ensemble. Flower sorts by node id, which is why the run is bit-identical across executions.
3. **Lines 113-116 use `aggregate_metricrecords`, which is `weighted_by_key`-aware** â€” but only for the *scalar training metrics*. The model itself, folded on line 107, is not weighted at all. That asymmetry is the subject of [Bagging, not averaging](#why-bagging-and-not-averaging).

Note what line 107 does *not* do: it never calls the parent `FedAvg.aggregate_train` (`fedavg.py:249-272`), which would have called `aggregate_arrayrecords` â€” a genuine weighted element-wise mean of the `uint8` buffers. That would silently produce numerical garbage: averaging raw bytes of two different JSON documents is meaningless. Bagging is not an optimisation here, it is the only correct option.

---

## `aggregate_bagging(prev_bytes, curr_bytes)` â€” the JSON surgery

The real source, confirmed before writing (`flwr/serverapp/strategy/strategy_utils.py:247-280`):

```python
def aggregate_bagging(
    bst_prev_org: bytes,
    bst_curr_org: bytes,
) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if bst_prev_org == b"":                                    # :252
        return bst_curr_org

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev_org)            # :256
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)      # :257

    bst_prev = json.loads(bytearray(bst_prev_org))             # :259
    bst_curr = json.loads(bytearray(bst_curr_org))             # :260

    previous_model = bst_prev["learner"]["gradient_booster"]["model"]          # :262
    previous_model["gbtree_model_param"]["num_trees"] = str(                   # :263
        tree_num_prev + paral_tree_num_curr
    )
    iteration_indptr = previous_model["iteration_indptr"]                      # :266
    previous_model["iteration_indptr"].append(                                 # :267
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]    # :272
    for tree_count in range(paral_tree_num_curr):                             # :273
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count             # :274
        previous_model["trees"].append(trees_curr[tree_count])                # :275
        previous_model["tree_info"].append(0)                                  # :276

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")                      # :278

    return bst_prev_bytes
```

Signature naming worth noting: `bst_prev_org` / `bst_curr_org` â€” the `_org` suffix means "original". Both inputs are consumed unmodified; only `bst_prev` is mutated. In practice `bst_curr` *is* mutated too (line 274 writes into `trees_curr[tree_count]["id"]`), but `bst_curr_org` is the untouched `bytes` and `bst_curr` is a freshly parsed dict, so the caller's bytes are safe.

### The `b""` short-circuit â€” how round 1 works

`server_app.py:29-32` seeds the global model as empty bytes:

```python
global_model = b""
arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])
```

On round 1, `configure_train` stashes those 0 bytes into `self.current_bst` (`fedxgb_bagging.py:85`). Client 0 receives an empty buffer; `client_app.py:47` branches on `global_round == 1` and calls `xgb.train(params, train_dmatrix, num_boost_round=1)` â€” building a booster from nothing, because there is nothing to load. Client 0 replies with a 1-tree model. Back on the server, line 107 calls `aggregate_bagging(b"", client0_bytes)`, line 252 fires, and client 0's bytes are returned **verbatim, unparsed and unmutated**. Client 1's bytes then hit line 259 for real, and *that* call does all the surgery.

So in round 1 the accumulator is seeded, not merged. The short-circuit is what lets the first client's model establish the base rather than trying to graft trees onto a document with no `learner` key â€” `bst_prev_org == b""` is checked *before* any `json.loads`, so an empty buffer never reaches the parser. Verified: the seed is 0 bytes, and the first `aggregate_bagging` call returns round-1 client 0's 3,555 bytes unchanged. `[VERIFIED]`

The same short-circuit has a second effect worth naming. Because round 1 returns `bst_curr_org` *byte-identically*, the round-1 global model retains whatever formatting XGBoost's own `save_raw("json")` produced. From round 2 onward the model is re-serialised by `json.dumps`. So the ensemble crosses a one-time writer boundary at exactly the point it becomes shared state.

### `num_trees` â€” the declared tree count

`num_trees` is a scalar inside `gbtree_model_param`, and XGBoost stores its own model scalars **as JSON strings** â€” hence the `str(...)` on line 263. It must be bumped to `tree_num_prev + paral_tree_num_curr`, i.e. previous total plus incoming count. Two hard native guards make this non-optional:

- If `trees` grows but `num_trees` does not:
  `XGBoostError: Check failed: trees_json.size() == param.num_trees (5 vs. 4)` at `gbtree_model.cc:90`.
- If `num_trees` is written as a JSON integer instead of a string:
  `XGBoostError: Invalid cast, from Integer to String` at `include/xgboost/json.h:88`.

The second is the concrete reason line 263 wraps the arithmetic in `str()`. `[VERIFIED]`

### `iteration_indptr` â€” the boosting-round prefix sum

`iteration_indptr` is a prefix-sum array where boosting round *i* owns trees `[indptr[i], indptr[i+1])`. Its length is always `num_boosting_rounds + 1`. Verified on a native 4-round booster: `num_trees = 4`, `len(trees) = 4`, `iteration_indptr = [0, 1, 2, 3, 4]` (len 5), `len(tree_info) = 4`. So `len(indptr) - 1 = 4` rounds. `[VERIFIED]`

Line 267 appends `iteration_indptr[-1] + paral_tree_num_curr`, i.e. one new entry per incoming group. Skipping it is fatal:

```
XGBoostError: Check failed: model.iteration_indptr.back() == model.param.num_trees (1 vs. 2)
```

Note the guard is on `indptr.back()` matching `num_trees` â€” the function keeps the two consistent by construction, appending the same increment to both. `[VERIFIED]`

**The consequence is a semantic change, not just a bookkeeping one.** Each client's slice is recorded as its own boosting iteration. After 3 federated rounds the merged model has `iteration_indptr = [0,1,2,3,4,5,6]` (len 7) and `xgb.Booster.num_boosted_rounds()` reports **6**, even though only 3 rounds of federated coordination occurred. The ensemble votes 6 trees deep. `iteration_indptr` no longer describes a sequential boosting history â€” it describes 6 concatenated independent learners. That is precisely what bagging is, and it is why the array had to be extended rather than left alone: XGBoost needs the indptr to know it has 6 iterations' worth of trees to walk.

### `trees[].id` â€” global uniqueness

Every tree carries an integer `id`, and the function renumbers the incoming ones as `tree_num_prev + tree_count` (line 274). This is **not** defensive polish â€” it is essential, and the failure mode is severe. XGBoost's slicing re-bases a booster: I sliced a 4-tree global booster to `[3:4]` and the resulting single-tree model reported `num_trees = 1`, `iteration_indptr = [0, 1]`, and **tree `id = 0`**. So *every* client sends trees numbered from 0, every round. Without renumbering, ids collide across clients.

Colliding ids do not merely "mis-index" â€” they crash the native library. Taking the merged 2-tree round-1 model and setting both ids to 0:

```
OSError: exception: access violation reading 0x0000000000000058
```

XGBoost uses the tree `id` to index internal storage, and a duplicate id produces an out-of-bounds native read. So the task's phrasing "the loader rejects or silently mis-indexes" is right for the *reject* half and understated for the severity: on a multi-tree ensemble this is a memory-safety violation, not a clean exception. (With a single tree it loads fine, because index 0 happens to be valid â€” which is why the bug would not show up in a one-client, one-epoch configuration.) `[VERIFIED]`

Line 274 mutates `trees_curr[tree_count]["id"]` in the *parsed incoming dict*. Had it mutated a shared structure it could double-renumber; parsing fresh on line 260 prevents that.

### `tree_info.append(0)` â€” parallel-tree group index

`tree_info` is a parallel array over `trees` giving each tree's **group index** within its boosting iteration. `num_parallel_tree = 1` in this project's config (`pyproject.toml:42`), so every tree is its own group and group 0 is always correct. Line 276 appends one entry per incoming tree, keeping `len(tree_info) == len(trees) == num_trees`. Omitting it:

```
XGBoostError: Check failed: tree_info_json.size() == param.num_trees (1 vs. 2)
```

Verified `tree_info` in the final model is `[0, 0, 0, 0, 0, 0]` â€” length 6, all zeros, matching `num_parallel_tree = 1`. `[VERIFIED]`

Had the project set `num-parallel-tree > 1`, the correct value would not be a constant `0` â€” parallel trees within one iteration share a group id, and this function would need `(tree_count // num_parallel_tree)`. Line 257 computes `paral_tree_num_curr` from exactly that parameter but only uses it as a *count*; the group id is hardcoded. So `num-parallel-tree > 1` is a latent bug in this helper. `[READ]`

### The `json.dumps` round-trip hazard

Line 278 re-serialises the whole merged dict with `json.dumps(bst_prev)` â€” **no** `sort_keys`, no `separators`, no `ensure_ascii` tuning, no indent. The output's float formatting and key ordering are whatever Python's `json` module happens to produce: floats go through `repr`, key order follows first-insertion order in the parsed dict (so keys from the *previous* model keep their native XGBoost ordering while keys unique to incoming trees land at the end of their parent object), and separators default to `', '` / `': '` â€” i.e. spaces after every comma and colon.

This is text surgery on a format that XGBoost describes as "binary-ish" (here it is JSON, which is why it works at all). Two honest observations:

**It is functionally sound.** The merged bytes load. Verified for every round: round 1 â†’ 2 trees, round 2 â†’ 4, round 3 â†’ 6, each `xgb.Booster.load_model(bytearray(...))` succeeding with `num_features() == 52`, and per-round AUCs computed from those boosters. The JSON scalars the function touches are read back as strings by the loader regardless of surrounding whitespace, and the loader does not compare the input text against a canonical form. `[VERIFIED]`

**But the output is not XGBoost's own text.** This is measurable. After round 3 the in-memory global model is **20,555 bytes** â€” that is the byte string `aggregate_bagging` produced, the one that `ArrayRecord` carries and that `server_app.py:50` reads back. `final_model.json` on disk is **19,102 bytes**. The 1,453-byte difference is entirely `server_app.py:49-57`, which loads those bytes into a real `Booster` and calls `bst.save_model("final_model.json")`; XGBoost's writer re-emits the model compactly and reorders/normalises keys. I confirmed this by re-serialising the simulated in-memory model with `save_model` â€” it produced exactly 19,102 bytes, matching the on-disk file byte-count, and the two byte strings compare unequal. `[VERIFIED]`

The practical lesson: the model that scores during federated evaluation is the `json.dumps` artefact; the model on disk is XGBoost's artefact. They are semantically the same forest, but if you ever diff them, hash them, or byte-compare a round's reply against the saved file, they will not match. Trust `save_model`, not the wire bytes, when inspecting the model.

---

## `_get_tree_nums`

```python
# flwr/serverapp/strategy/strategy_utils.py:283-291
def _get_tree_nums(xgb_model_org: bytes) -> tuple[int, int]:
    xgb_model = json.loads(bytearray(xgb_model_org))

    # Access model parameters
    model_param = xgb_model["learner"]["gradient_booster"]["model"][
        "gbtree_model_param"
    ]
    # Return the number of trees and the number of parallel trees
    return int(model_param["num_trees"]), int(model_param["num_parallel_tree"])
```

It parses the XGBoost JSON, walks three nested dict levels to `gbtree_model_param`, and returns two `int`s:

- **`num_trees`** â€” total number of trees in the ensemble. This is the *whole* model, not trees per boosting round. In the final merged model it is `6`, and XGBoost requires `num_trees == len(trees) == len(tree_info) == iteration_indptr[-1]`.
- **`num_parallel_tree`** â€” how many trees each boosting iteration contributes (the "random forest style" width). `pyproject.toml:42` sets `1`, so a boosting round adds one tree.

Both are stored as JSON strings by XGBoost, so `int(...)` is required â€” a plain return would hand back `str` objects and the arithmetic on lines 263-268 would silently concatenate. The two call sites take opposite halves: line 256 wants the previous model's total, line 257 wants only the incoming count, each discarding the other with `_`. `[READ]`

Cost note: `aggregate_bagging` parses `bst_prev_org` on line 256 (inside `_get_tree_nums`) and again on line 259, and parses `bst_curr_org` twice â€” lines 257 and 260. Four `json.loads` per call where two would do. For a 20 KB model this is irrelevant; for a production run with a 100 MB ensemble it would not be. `[READ]`

---

## Why bagging and not averaging

**For neural networks, averaging works because weights are continuous numbers in a shared, aligned space.** Two clients' `layers[0].weight[7][3]` are the same coefficient, estimated from different data. The mean of two noisy estimates of one quantity is a better estimate of it, and it is also a parameter vector a client can immediately continue training from â€” the round is a small step along a shared trajectory.

**Trees break both preconditions.**

*Averaging is meaningless.* A tree is a set of `if feature_i <= threshold: go_left else: go_right`. `Age > 30` and `Age > 45` have no meaningful midpoint, and `Age > 37.5` is a split no client learned, on data no client has, and whose two branches contain no data either client saw.

*Alignment is undefined.* There is no element-wise correspondence between two clients' forests. Client A might have 3 trees of depth 8; client B might have 1 tree of depth 3. There is no shared index to average over â€” and even "the k-th tree" is not the same quantity, because the trees are ordered by boosting iteration within each client's local history, which is a function of that client's own data.

*So the server concatenates.* `aggregate_bagging` unions the forests. Each client's new trees are appended to the global ensemble, and every subsequent prediction sums over all of them. This is textbook bagging applied at the *forest* level rather than the dataset level â€” which is why the name is apt. Each local `update()` grows that client's trees as a direct extension of the global ensemble, but the client's contribution is confined to the trees it just added (`client_app.py:20-30` slices off exactly the last `num_local_round` trees). `[READ]`

### Arithmetic for this project

The ensemble grows by `num_clients Ã— local_epochs` trees per round:

```
num_rounds (3) Ã— num_clients (2) Ã— local_epochs (1) = 6 trees
```

Verified: 6 trees in `final_model.json`, tree ids `[0,1,2,3,4,5]`, `iteration_indptr = [0,1,2,3,4,5,6]`. `[VERIFIED]` Compare with the centralized tuned model in this repo's `models/`, which is on the order of 400 trees â€” the federated model is roughly **1/60th** the size. That gap is the honest headline: bagging with 3 rounds and 1 local epoch is a demonstration that the architecture works, not a production-grade model. It still scores AUC 0.993 because churn is close to linearly separable in 52 features, but the ensemble is 6 trees deep and would degrade on harder problems.

### Consequence: equal weighting in training, `num-examples` weighting in evaluation

Bagging gives every client's trees **equal weight in the vote**. Each tree contributes one summand to the sum of leaf outputs, so a client with 2817 rows and a client with 8 rows would contribute one tree each â€” the 8-row client would get 350Ã— the per-row influence. There is no `num-examples` weighting anywhere in `aggregate_bagging`, and none is possible without per-tree weights, which `update()` does not produce.

Evaluation is the opposite. `aggregate_evaluate` (`fedavg.py:302-319`) delegates to `aggregate_metricrecords` (`strategy_utils.py:101-143`), which reads `weighted_by_key` (default `"num-examples"`) from each reply's `MetricRecord`, normalises, and takes a weighted mean:

```python
weights = [cast(float, metricrecord[weighting_metric_name]) for ...]
weight_factors = [w / sum(weights) for w in weights]
# ... then value * weight for every key except weighting_metric_name itself
```

So **training and evaluation use different weighting rules in the same round**: the model is an unweighted concatenation, the reported AUC is a sample-weighted mean of per-client AUCs. Reported federated AUC is therefore a reweighted average of per-partition AUCs, not the AUC of any single client's model, and not the AUC the global ensemble would score on any one partition. Confirmed in the simulation: partition 0 scored 0.9878/0.9897/0.9897 and partition 1 scored 0.9941/0.9954/0.9962 across rounds 1-3, and the reported figures are the `num-examples`-weighted blends of those pairs. `[VERIFIED]`

This asymmetry is fine for a simulation where partitions are i.i.d. (`IidPartitioner` over a shuffled dataset, `task.py:126-129`), because every client has ~2817 rows and near-identical class balance. Under non-IID partitioning â€” say one partition of 30 rows and one of 7000 â€” the evaluation metric stays honest while the model does not. Worth knowing before trusting this pattern on real federated data.

---

## External connections

`server_app.py` is 57 lines and does no training itself. It reads config, constructs the strategy, and persists the artifact.

```python
# quickstart-xgboost/quickstart_xgboost/server_app.py

@app.main()                                                    # :16
def main(grid: Grid, context: Context) -> None:
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]        # :19
    fraction_train = context.run_config["fraction-train"]       # :20
    fraction_evaluate = context.run_config["fraction-evaluate"] # :21
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))      # :23
    params = cfg["params"]                                      # :24

    # Init global model
    # Init with an empty object; the XGBooster will be created
    # and trained on the client side.
    global_model = b""                                         # :29
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])  # :32

    # Initialize FedXgbBagging strategy
    strategy = FedXgbBagging(                                  # :35
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start strategy, run FedXgbBagging for `num_rounds`
    result = strategy.start(                                   # :41
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    if context.run_config["save-model"]:                        # :47
        # Save final model to disk
        bst = xgb.Booster(params=params)                       # :49
        global_model = bytearray(result.arrays["0"].numpy().tobytes())  # :50

        # Load global model into booster
        bst.load_model(global_model)                            # :53

        # Save model
        print("\nSaving final model to disk...")                # :56
        bst.save_model("final_model.json")                      # :57
```

**Where the parameters come from.** `context.run_config` is populated from the `[tool.flwr.app.config]` table of `pyproject.toml` â€” the same file that declares `serverapp = "quickstart_xgboost.server_app:app"` and `clientapp = "quickstart_xgboost.client_app:app"` (`pyproject.toml:24-26`), so `pyproject.toml` is simultaneously Flower app manifest, run-config, and XGBoost hyperparameter store. Line 23's `unflatten_dict` rebuilds the nested `params` object from TOML's dotted keys (`params.max-depth`), and `replace_keys` (`task.py:174-183`) recursively rewrites `-` to `_`, turning `max-depth` â†’ `max_depth`, `eval-metric` â†’ `eval_metric`, `num-parallel-tree` â†’ `num_parallel_tree`, `tree-method` â†’ `tree_method`. Without that step XGBoost would receive hyphenated parameter names and ignore them. This is the single `calls` edge the knowledge graph records for `main()`. `[READ]`

`num-server-rounds = 3`, `fraction-train = 1.0`, `fraction-evaluate = 1.0`, `local-epochs = 1` (`pyproject.toml:30-34`) â€” with 2 partitions and fractions of 1.0, every client participates in every round, and `min_train_nodes`/`min_evaluate_nodes` defaults of 2 are satisfied exactly.

**What is and is not passed.** `strategy.start` is called with only `grid`, `initial_arrays`, and `num_rounds` (lines 43-44). `train_config` and `evaluate_config` are left `None`, so `Strategy.start` substitutes empty `ConfigRecord`s (`strategy.py:53-54`). Consequence worth flagging: **the XGBoost `params` never reach the strategy.** `params` is a local variable in `main()`, used at line 49 purely to give the empty `Booster` a parameter schema so `load_model` can succeed. The actual hyperparameter dictionary is sent by the *client* â€” `client_app.py:43-44` re-derives it from its own `context.run_config`. So server and client agree on hyperparameter configuration only because both read the same `pyproject.toml`; there is no server-side control over training. If a client shipped a different `params`, the server would merge its trees into a model built under different settings with no check. `[READ]`

**Persistence, and the two serialisers.** Lines 47-57 run only when `save-model` is true â€” `pyproject.toml:34` sets it to `false`, so the checked-in `final_model.json` was produced by a run with the flag flipped on. The reconstruction is three steps: build an empty `Booster` with `params` (line 49, needed so XGBoost has a parameter schema for `load_model`), copy `result.arrays["0"]` back to `bytes` (line 50), then `load_model` (line 53). `result.arrays` is the `ArrayRecord` built at `fedxgb_bagging.py:110`, so this is the exact `json.dumps` output discussed above. Then `save_model("final_model.json")` (line 57) writes to the **current working directory**, not next to `server_app.py` â€” under `flwr run` that resolves to the app directory, which is where the file lives. This second serialisation is what normalises the 20,555 in-memory bytes to 19,102 on-disk bytes. `[VERIFIED]`

---

## Runtime facts

Facts marked `[VERIFIED]` were measured in this session, not read from documentation.

**Federated training** â€” reproduced by driving the real `aggregate_bagging` over the real two partitions with the real `pyproject.toml` parameters, using `aggregate_metricrecords` for the reported figures:

| Round | Global trees | Aggregated AUC | Partition 0 AUC | Partition 1 AUC |
|---|---|---|---|---|
| 1 | 2 | `0.9909828030774182` | 0.9878274428274428 | 0.9941381633273936 |
| 2 | 4 | `0.9925536221296033` | 0.9897141372141373 | 0.9953931070450693 |
| 3 | 6 | `0.9929157168364653` | 0.9896777546777547 | 0.9961536789951758 |

`[VERIFIED]` â€” matches the figures recorded in `frontend/fl_rounds_sample.csv` (0.9910 / 0.9926 / 0.9929) and reproduced bit-identically.

**Partitions** â€” `train=2817 / valid=705` and `train=2816 / valid=705`, totalling 7,043 rows. `[VERIFIED]`

**Final model** â€” `quickstart-xgboost/final_model.json`: `num_trees = 6` (stored as the JSON string `"6"`), `num_parallel_tree = 1`, `iteration_indptr = [0,1,2,3,4,5,6]`, 6 trees with ids `[0,1,2,3,4,5]`, `tree_info = [0,0,0,0,0,0]`, `learner_model_param.num_feature = 52`. Loads cleanly into `xgboost.Booster` with `num_boosted_rounds() = 6` and `num_features() = 52`. `[VERIFIED]`

**Final model scores** â€” AUC **0.9897** on partition 0 and **0.9962** on partition 1. `[VERIFIED]`

> **One correction to the brief.** The stated size of 18,989 bytes does not match the artifact in the working tree, which measures **19,102 bytes**. The graph node `config:quickstart-xgboost/final_model.json` describes it as "~18.6 KB", consistent with 18,989, but that file has an mtime of `18:58:02` local â€” about nine minutes *after* the graph snapshot at `18:49:20`. The model was regenerated after the graph was built, and the current artifact is 113 bytes larger. The tree count (6), feature count (52), and AUC figures all still hold exactly; only the byte size moved. Separately, 19,102 is the size of XGBoost's `save_model` output, whereas the in-memory `json.dumps` result is 20,555 bytes â€” if either figure was ever quoted as "the wire model size", that would be the mix-up. `[VERIFIED]`

---

## Summary of load-bearing detail

| Line (`strategy_utils.py`) | Mutation | Skip it andâ€¦ |
|---|---|---|
| 252 | `if bst_prev_org == b"": return` | Empty seed hits the JSON parser; round 1 has no base to graft onto |
| 263 | `num_trees = str(prev + curr)` | `Check failed: trees_json.size() == param.num_trees`; or `Invalid cast, from Integer to String` without `str()` |
| 267 | `iteration_indptr.append(...)` | `Check failed: model.iteration_indptr.back() == model.param.num_trees` |
| 274 | `trees_curr[k]["id"] = prev + k` | **Access violation** â€” every client ships trees id'd from 0 |
| 276 | `tree_info.append(0)` | `Check failed: tree_info_json.size() == param.num_trees` |
| 278 | `bytes(json.dumps(bst_prev), "utf-8")` | Nothing breaks, but the wire format becomes Python's, not XGBoost's (20,555 vs 19,102 bytes) |

`[VERIFIED]` for every row, tested by constructing each violation against a real 2-tree merged model.

## What a reader should take away

1. **Bagging is forced, not chosen.** Averaging requires continuous aligned parameters; trees have neither. `aggregate_bagging` is the minimum correct operation, and it is ~30 lines because there is no clever alternative.
2. **It is textual, and it is exactly as brittle as that sounds.** Four mutations, four native XGBoost guards, and one of them (tree `id`) fails with a memory-access violation rather than a clean error. The function is unowned, untested, and unvendored by this project.
3. **The two `FedXgbBagging` classes are real and both match the description.** Same name, different packages, different base classes; only `flwr.serverapp.strategy.FedXgbBagging` has `start()`. The legacy twin carries its own duplicated copy of the logic, differing only in `if not bst_prev_org` vs `if bst_prev_org == b""`.
4. **The privacy claim survives.** `main()` holds `bytes` and floats. `aggregate_bagging` parses XGBoost's own model JSON â€” thresholds, leaf values, feature indices. No customer row is reconstructed at any point in this path.
5. **The model is small and that is the honest caveat.** 6 trees against ~400 centralised, and every client's trees vote with equal weight in training while evaluation weights by `num-examples`. Fine for an IID two-partition demo; the asymmetry would matter on real non-IID data.


---

## 4. `preprocess.py:preprocess_telco` — cleaning and leakage removal

# `preprocess_telco` â€” the shared Telco cleaning function

## What this is

`preprocess_telco(df)` is a 72-line, dependency-free pandas function in `quickstart-xgboost/quickstart_xgboost/preprocess.py` that takes the raw 50-column IBM SPSS Telco churn DataFrame and turns it into a modelling-ready frame: it drops seven columns that would leak the churn label or are pure identifiers, fills the two columns whose nulls are informative rather than missing, binary-encodes 19 Yes/No columns plus Gender, and then drops the two post-hoc columns that exist only for customers who already churned. It is the single shared cleaning routine for the federated path â€” extracted verbatim from `notebooks/Privacy_churn_v2.ipynb` cell 17 so the notebook that trained the shipped model and the federated loader that loads the parquet run *literally the same cleaning code*. Its most consequential property is not what it does but how it does it: it **mutates its argument in place via `inplace=True` and returns `None`**, a deliberate design that avoids a specific historical bug in the v1 notebook, and a contract that differs from the pandas idiom most readers expect.

---

## Role in the architecture

**Layer: `layer:federated-training`** â€” "Federated Training".

The node itself is not listed in `layers[].nodeIds`; its containing file node `file:quickstart-xgboost/quickstart_xgboost/preprocess.py` is one of the 6 nodes in that layer. It belongs here rather than in `layer:centralized-serving` or `layer:frontend` because `preprocess.py` is imported by exactly one module, `task.py:load_data`, which is the function that partitions the dataset into per-client Flower DMatrix shards. Nothing in the serving path imports it â€” `API/app.py` and `frontend/streamlit_app.py` receive already-encoded feature vectors and re-align them to their own copy of `FEATURE_NAMES` rather than re-running raw cleaning.

Graph neighbourhood for `function:quickstart-xgboost/quickstart_xgboost/preprocess.py:preprocess_telco`:

| Edge | Direction | Weight | Counterparty |
|---|---|---|---|
| `contains` | forward | 1.0 | `file:quickstart-xgboost/quickstart_xgboost/preprocess.py` |
| `exports` | forward | 0.8 | same file |
| `calls` | forward | 0.8 | `function:quickstart-xgboost/quickstart_xgboost/task.py:load_data` |

Outgoing edges: **none**. The function imports nothing (`preprocess.py` has no import statements at all â€” pure pandas-by-global-name duck typing on its argument) and calls no project function. Everything it touches is either a pandas method or a local loop.

File-level edges add two more links: `task.py` â†’ `preprocess.py` via `imports`, and `file:notebooks/Privacy_churn_v2.ipynb` â†’ `preprocess.py` via `related`, capturing the extraction provenance.

### Why extraction happened

`preprocess.py` is a mechanical copy of `notebooks/Privacy_churn_v2.ipynb` cell 17 `[VERIFIED]` â€” the notebook cell *defines* `preprocess_telco` as a function inside the notebook, and the extracted file is that same body with indentation normalized from the notebook's 2-space style to 4-space, plus a cosmetic reformat of the final `df.drop(...)` call. I diffed the two after whitespace normalization: the token streams are identical, differing only in indentation width, one comment's capitalization ("Encoding Gender Columns" â†’ "Encoding Gender columns"), and line wrapping of the trailing drop. So a claim worth making explicitly: the notebook and the federated loader do not merely *agree* on the cleaning rules, they run the same code, which removes a whole category of train/serve skew.

### The `None` return and the bug it prevents

`task.py:112` calls it as a bare statement:

```python
# task.py:111-112
df = pd.read_parquet(DATASET_PATH)
preprocess_telco(df)
```

No assignment. That is correct given the contract â€” and it is the direct descendant of a real crash. In `notebooks/Privacy_churn_v1.ipynb` cell 39 `[READ]` the author wrote:

```python
drop_cols = [
    'Customer Status_Churned',
    'Customer Status_Joined',
    'Customer Status_Stayed',
    'Churn Score',
]

df = df.drop(columns=drop_cols, inplace = True)   # <-- returns None
Xtrain.drop(columns=drop_cols, inplace=True)
```

pandas' `inplace=True` methods mutate in place **and return `None`**. So `df` is silently rebound to `None`, and `df` is thereafter a dead variable. The failure surfaces 30 cells later at the final export `[READ]`:

```python
# Privacy_churn_v1.ipynb cell 69
df.to_csv('telco_v1.csv', index=False)
```

I read the stored execution output on that cell `[VERIFIED]`:

```
AttributeError: 'NoneType' object has no attribute 'to_csv'
```

`preprocess.py` avoids the trap because both drops are *statements*, never assignments:

```python
# preprocess.py:3
df.drop(columns=[...], axis = 1, inplace=True, )
# preprocess.py:69-72
df.drop(
    columns=['Churn Category', 'Churn Reason'],
    axis=1,
    inplace=True )
```

The tradeoff is that the return type is `None`, which violates the principle that a function mutating an argument should say so. Here it is only implicit â€” there is no docstring and no type hint. `[VERIFIED]` `preprocess_telco(df)` returns `None` and the caller's `df` is still the same object afterward.

---

## Internal structure â€” a line-by-line walk

### Line 3 â€” leakage and high-cardinality drop

```python
df.drop(columns=['Customer ID', 'Country', 'State', 'City', 'Quarter', 'Customer Status','Churn Score' ], axis = 1, inplace=True, ) # Dropping columns wiht too many category
```

Seven columns go at once, and the trailing comment ("wiht too many category" â€” typo in the original, preserved through the extraction) tells you the author's stated reason was cardinality. Two of the seven are actually leakage control, which is the more important reason and the one the comment does not mention.

- **`Customer Status` is a restatement of the label.** `[VERIFIED]` Its values are `['Churned', 'Joined', 'Stayed']` and the crosstab against the label is fully determined:

  | Customer Status | No | Yes |
  |---|---|---|
  | Churned | 0 | 1869 |
  | Joined | 454 | 0 |
  | Stayed | 4720 | 0 |

  Every one of the 1,869 churned rows carries `Churned` and no other row does, so `Customer Status == "Churned"` âŸº `Churn Label == "Yes"`. Correlation with the target is exactly 1.0. A model given this column learns the answer, not the pattern, and the reported AUC becomes meaningless. Note the derived form `Customer Status_Churned` etc. that v1 one-hot encoded before dropping â€” that is the shape of the leak when it slips through a pipeline.

- **`Churn Score` is a vendor-side prediction shipped as an input.** The IBM SPSS / Telco churn dataset already contains a churn score produced by the vendor's own churn model. `[VERIFIED]` Its Pearson correlation with the encoded label is **0.6608**. That is a strong but non-degenerate relationship â€” so the model would not be trivially invertible, but it would be fitting mostly off the vendor's prediction rather than off the customer's own behaviour. This is leakage of a different species than `Customer Status`: not the label itself, but a *prior model's output* treated as an independent feature. Shipped-service predictions belong in the risk-scoring output, not the training feature set.

- The remaining four are legitimately unusable for prediction quality: `Customer ID` is a row identifier (memorizable, zero generalizing value), `Country` is near-constant, and `State`/`City` are high-cardinality geography that invite location memorization. `Quarter` is a period label with no bearing on the label.

Ordering matters here for a subtle reason: the drop is first, so nothing downstream ever needs to guard against these columns existing.

### Lines 5â€“7 â€” informative-null fill

```python
df['Offer'] = df['Offer'].fillna('No Offer') # Filling missing values

df['Internet Type'] = df['Internet Type'].fillna('No Internet') # Filling missing values
```

`[VERIFIED]` The raw null counts are 3,877 for `Offer` and 1,526 for `Internet Type`. These are nulls that are *not* missing data in the ordinary sense â€” they are present precisely for customers who do not have that add-on. A blank `Offer` means "was not offered anything", and a blank `Internet Type` means "has no internet service at all". Giving each an explicit sentinel category preserves that meaning and lets the one-hot encoder emit a real column for the "none" case. The alternatives both lose information: dropping the rows would delete ~55% of the dataset on `Offer` alone, and median/mode filling would fabricate a plausible-looking add-on for customers who demonstrably have none.

`[VERIFIED]` The sentinel strings also matter downstream: `Offer_No Offer` is a literal entry in `FEATURE_NAMES` (`task.py:51`), and `Internet Type_No Internet` is at `task.py:60`. The labels chosen here are a hard contract with the feature list and with the persisted encoder in `models/telco_encoder.joblib`. Renaming either sentinel without refitting the encoder would silently zero a live feature.

Note the style difference from line 3: `fillna` is used **without** `inplace`, and the result is reassigned. That is the safer pandas idiom and it works here precisely because assignment is used. The function is internally inconsistent about this â€” one convention for the fill, another for the two drops â€” which is worth knowing before anyone edits it.

### Lines 9â€“30 â€” `binary_cols` (debug-only)

```python
  # Binary encoding
    binary_cols = [
        'Gender',
        'Under 30',
        ...
        'Churn Label'
        ]
```

19 entries. `[READ]` Line 9's comment is indented 2 spaces where the surrounding block uses 4 â€” legal Python, since comments carry no indentation significance, but visually inconsistent and a fingerprint of the notebook extraction. The list mixes `Gender` (Male/Female) in with the 18 genuine Yes/No columns, so it is not a coherent group despite the name; its only real use is the debug loop at lines 63â€“65.

### Lines 32â€“51 â€” `yes_no_cols` (the one that works)

```python
  # Binary encoding Yes/no columns
    yes_no_cols = [
        'Under 30',
        'Senior Citizen',
        ...
        'Paperless Billing',
        'Churn Label']
```

18 entries â€” the same 18 as `binary_cols` minus `Gender`. `[READ]` Line 32 has the same 2-space comment indentation. This is the list that line 68 iterates, so `Gender` is correctly handled separately at line 53 and is correctly *absent* here; had `Gender` been left in, the `{'Yes': 1, 'No': 0}` map would have turned every value into `NaN` and silently destroyed the column. The duplication of the 18 shared names across two lists is the function's single largest piece of dead weight.

### Lines 53â€“55 â€” Gender encoding

```python
    df['Gender'] = df['Gender'].map({  # Encoding Gender columns
    'Male': 0,
    'Female': 1})
```

`Male â†’ 0`, `Female â†’ 1`. `[VERIFIED]` Confirmed against the deployed consumers:

- `frontend/streamlit_app.py:7` â€” `gender_mapping = {0: "Male", 1: "Female"}`, reinforced by the widget label at line 189, `"Select Gender -> 0: Male, 1: Female"`.
- `notebooks/Privacy_churn_v2.ipynb` cell 17 â€” the function this was extracted from, identical direction.
- `API/app.py` â€” `Gender: Annotated[int, Field(...)]` at line 127, with `"Gender"` as the first entry of its own `FEATURE_NAMES` at line 34. The API takes an already-encoded integer and never re-maps it.

The older `notebooks/Privacy_churn_v1.ipynb` cell 19 maps the opposite way:

```python
df['Gender'] = df['Gender'].map({
    'Male': 1,
    'Female': 0
})
```

`[VERIFIED]` This is the source of the gender confusion that runs through the repo's docs. The important nuance for the deployed path: v2 is the notebook that produced the shipped artifacts â€” cell 66 does `joblib.dump(best_xgboost_model, ...)`, `joblib.dump(encoder, 'telco_encoder.joblib')`, and writes `feature_names.txt`, and `models/telco_encoder.joblib` (1,785 bytes, written 9/9/2026) is the file `task.py:78` loads. So `preprocess.py`, v2, the frontend, and the API are all self-consistent; a model retrained from the **v1** pipeline would produce a Gender column inverted relative to every downstream consumer, silently flipping every prediction's gender attribution. It is a one-line fix in two places, but it changes model inputs, so it needs a re-score against the test split rather than a blind edit.

### Lines 57â€“61 â€” `one_hot_cols` (dead)

```python
    one_hot_cols = [
        'Offer',
        'Internet Type',
        'Contract',
        'Payment Method']
```

Defined and **never used inside this function**. `[VERIFIED]` `task.py:113` defines its own identically-valued `one_hot_cols` and does the encoding itself:

```python
# task.py:113-117
one_hot_cols = ['Offer', 'Internet Type', 'Contract', 'Payment Method']
encoded = encoder.transform(df[one_hot_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(one_hot_cols), index = df.index)
df.drop(columns= one_hot_cols, inplace=True)
df = pd.concat([df, encoded_df], axis = 1)
```

Duplicated knowledge across two files. Note `task.py:116` *does* correctly use a bare-statement `drop` â€” the same lesson from the v1 bug, applied correctly a second time.

### Lines 63â€“65 â€” debug loop (log noise)

```python
    for col in binary_cols:  # Checking
        print(col) 
        print(df[col].unique())
```

19 columns Ã— 2 lines = 38 print statements per execution. `[VERIFIED]` This is where the run gets noisy, and the caching in `task.py` shapes *how* noisy. `load_data` guards the whole build with `if fds is None:` and assigns the module global at `task.py:129`, so preprocessing runs once per **process**, not once per call. I confirmed this: calling `task.load_data(0, 2)` and then `task.load_data(1, 2)` in the same process produced 38 debug lines on the first call and **zero** on the second. In a real 2-client Flower run the two clients are separate processes, so you see the block twice total rather than once â€” bounded, but still 76 lines of `print` interleaved with server round logs. Worth deleting for a portfolio piece.

The loop did serve its purpose: `[VERIFIED]` it is what surfaced the unencoded `['No' 'Yes']` values that line 68 then fixed.

### Lines 67â€“68 â€” the actual binary encode

```python
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
```

The workhorse. `[VERIFIED]` After this loop `Churn Label` is integer 0/1 and `Gender` is integer 0/1, confirmed by running the function against `data/telco_original.parquet`. That encoding is the precondition for `task.py:118` to read the label as floats:

```python
labels = np.array(df['Churn Label'].tolist(), dtype=np.float32)
```

Note the ordering: the label is encoded in the *same* pass as the features, which is convenient but means `Churn Label` is treated as an ordinary Yes/No column with no special handling. If the list were ever edited to exclude it, `task.py` would fail loudly on the dtype cast rather than silently.

### Lines 69â€“72 â€” post-hoc drop

```python
    df.drop(
        columns=['Churn Category', 'Churn Reason'],
        axis=1,
        inplace=True )  # This is added after observing corr matrix
```

Dropped **last**, after encoding, with a comment that says the author added them after looking at a correlation matrix. `[VERIFIED]` Both columns have exactly **5,174** nulls in the raw data â€” and the raw label distribution is 1,869 `Yes` / 5,174 `No`. The null count equals the stayed-customer count precisely. So these columns are populated *only* for churned customers, making their nullity perfectly correlated with the label: a model could read "Churn Category is non-null" and be right every time. Dropping them is both leakage avoidance and a data-shape necessity, and deferring the drop to the end means they were available while the author was still exploring â€” you cannot compute a correlation matrix on columns that are already gone.

---

## Code-quality observations

Stated plainly, no editorializing beyond what the code shows:

- **Comment indentation.** Lines 9 and 32 use 2 spaces where the function body uses 4. Python ignores comment indentation, so it runs; it is purely cosmetic and is inherited from the notebook's cell-level indentation.
- **`binary_cols` is redundant.** Lines 10â€“30 duplicate 18 of the 19 names in `yes_no_cols` (lines 33â€“51), differing only by including `Gender`. Its sole consumer is the debug `print` loop. Deleting lines 9â€“30 and the loop at 63â€“65 removes 26 lines and zero behaviour.
- **`one_hot_cols` is unused here.** Lines 57â€“61 are duplicated verbatim at `task.py:113`. One of the two copies should go; keeping both means an edit to the one-hot set must be made twice, in two files, with nothing to catch a miss.
- **The contract is unstated.** The function mutates its input and returns `None`, with no docstring and no type annotation. `task.py:112` uses it correctly, but a caller writing `df = preprocess_telco(df)` would silently null out their DataFrame â€” which is precisely the v1 bug, reintroduced by one plausible character. A one-line docstring (`"""Clean df in place; returns None. Caller must keep the reference."""`) closes this.
- **Inconsistent inplace convention.** Lines 3 and 69 use `inplace=True`; lines 5 and 7 use `fillna` with reassignment. Both styles appear in a 72-line function, and the safer one is used for the fills while the safer-behaving-because-of-bare-statement pattern is used for the drops.
- **The typo is preserved.** `# Dropping columns wiht too many category` (line 3) â€” harmless, and it does honestly date the provenance of the extraction.

---

## External connections

- **Imports:** nothing. `preprocess.py` has no `import` statement of any kind; it operates purely through the pandas methods of the object handed to it. This is why it has zero outgoing edges in the graph.
- **Called by:** `task.py:load_data`, exactly once, at line 112, as a bare statement. Graph edge: `calls`, forward, weight 0.8.
- **Also imported by:** only `task.py` (`imports` edge). The serving paths (`API/app.py`, `frontend/streamlit_app.py`) do not import it.
- **Output feeds:** the return value is `None`; the *mutated frame* feeds `encoder.transform(df[one_hot_cols])` at `task.py:114` and subsequently `df.reindex(columns=FEATURE_NAMES, fill_value=0)` at `task.py:120`.
- **Provenance:** `file:notebooks/Privacy_churn_v2.ipynb` â†’ `preprocess.py` via a `related` edge, reflecting that this body was lifted out of notebook cell 17.

---

## Data flow

```
data/telco_original.parquet   7,043 rows Ã— 50 columns   [VERIFIED]
        â”‚
        â”‚  line 3: drop 7 (5 identifier/cardinality + 2 leakage)
        â–¼
   43 columns
        â”‚
        â”‚  lines 5â€“7: fillna Offerâ†’'No Offer' (3,877 nulls), Internet Typeâ†’'No Internet' (1,526 nulls)
        â–¼
   43 columns
        â”‚
        â”‚  lines 53â€“55: Gender Maleâ†’0, Femaleâ†’1
        â”‚  lines 67â€“68: 18 Yes/No columns â†’ 1/0, including Churn Label
        â”‚  lines 69â€“72: drop Churn Category, Churn Reason (5,174 nulls each = all stayed customers)
        â–¼
   41 columns   [VERIFIED] â€” 43 âˆ’ 2 post-hoc drops
        â”‚
        â”‚  task.py:114  encoder.transform over ['Offer','Internet Type','Contract','Payment Method']
        â”‚  task.py:116  drop those 4 originals; task.py:117  concat the 16 expanded columns back
        â–¼
   53 columns = 52 features + 1 label   [VERIFIED]
        â”‚
        â”‚  task.py:118  labels = df['Churn Label'] â†’ float32
        â”‚  task.py:120  df.reindex(columns=FEATURE_NAMES, fill_value=0)
        â–¼
   52 features (task.py:14-67) + 1 label   [VERIFIED]
        â”‚
        â”‚  task.py:123  Dataset.from_dict â†’ 126  shuffle(seed=42)
        â”‚  task.py:127  IidPartitioner(num_partitions=num_clients)
        â–¼
   per-client DMatrices â€” 2,817 / 2,816 train rows, 705 val each   [VERIFIED]
```

The column arithmetic, checked rather than assumed:

- `[VERIFIED]` `data/telco_original.parquet` is 7,043 Ã— 50.
- `[VERIFIED]` After `preprocess_telco`, 41 columns remain.
- `[VERIFIED]` `encoder.get_feature_names_out(one_hot_cols)` returns **16** names (not 17): 6 for `Offer` (No Offer, Offer Aâ€“E), 4 for `Internet Type` (Cable, DSL, Fiber Optic, No Internet), 3 for `Contract`, 3 for `Payment Method`. So 41 âˆ’ 4 originals + 16 expanded = 53.
- `[VERIFIED]` `len(FEATURE_NAMES)` is exactly **52**, with zero duplicates.
- `[VERIFIED]` `inputs` is 52 columns pre-reindex, and `reindex` neither loses nor gains anything â€” the lost list and gained list are both empty. The frame and the contract already agree exactly.
- `[VERIFIED]` The end-to-end result matches: 2 clients produce 2,817 + 2,816 = 5,633 train rows (80% of 7,043, minus validation), 705 validation rows each, and both DMatrices report `num_col() == 52`.

`[VERIFIED]` The 16 expanded names are exactly the tail of `FEATURE_NAMES`: `Offer_No Offer`, `Offer_Offer A` â€¦ `Offer_Offer E` (`task.py:51-56`), `Internet Type_Cable/DSL/Fiber Optic/No Internet` (57â€“60), `Contract_Month-to-Month/One Year/Two Year` (61â€“63), `Payment Method_Bank Withdrawal/Credit Card/Mailed Check` (64â€“66). The `Yes`/`No` and `Offer` sentinels chosen on lines 5 and 7 surface verbatim as column names here â€” which is why the hyphen in `Month-to-Month` and the spaces in `Bank Withdrawal` survive into the feature contract while the notebook's `-`â†’`_` `replace_keys()` conversion (`task.py:174`) is reserved for the nested metric dicts Flower aggregates.

---

## One-paragraph takeaway

`preprocess_telco` earns its place in the architecture by being the one place the leak-prevention rules live. Its two drops of `Customer Status` (a 1.0-correlated restatement of the label) and `Churn Score` (a vendor model output at 0.6608 correlation) are the substantive engineering in the file; the rest is encoding plumbing. Its signature quirk â€” mutate in place, return `None` â€” is a bug-avoidance mechanism inherited from the v1 notebook's `df = df.drop(..., inplace=True)` assignment that crashed the export cell with `AttributeError: 'NoneType' object has no attribute 'to_csv'`. The deployed path is internally consistent on Gender (`Male â†’ 0` in `preprocess.py`, v2, the Streamlit app and the API), and only the superseded v1 pipeline disagrees. What remains is cleanup, not correction: 26 lines of duplicated column lists, 38 debug `print` lines, an unused `one_hot_cols` duplicated into `task.py`, an unstated mutation contract, and two comment-indentation inconsistencies â€” all cosmetic, none affecting the 52-feature contract that everything downstream depends on.

---

## 5. `frontend/federated_demo.py` — the 4-tab demo shell

# `frontend/federated_demo.py` â€” the federated demonstrator

## What this is

`frontend/federated_demo.py` (103 lines) is a Streamlit teaching skeleton for the **federated** half of this
two-pipeline churn project. It is the demonstrable surface of the Flower/`flwr` path: a four-tab app
(Learn â†’ Run â†’ Compare â†’ Predict) that explains one round of `FedXgbBagging`, replays a pre-recorded
roundâ†’AUC curve, contrasts federated against centralized AUC, and â€” once the learner finishes it â€” scores a
single customer with the aggregated XGBoost booster that the clients jointly produced. It talks to **no
training server and no API**: it reads two files off disk (`frontend/fl_rounds_sample.csv` and
`quickstart-xgboost/final_model.json`) and loads the latter directly into an `xgboost.Booster` in-process.
Critically, it is **not broken code** â€” the 7 `TODO` occurrences are the deliverable. The file's own header
says so: *"TODOs are yours to complete. If you can explain each TODO, you can defend it."* Read it as a lab
worksheet, and resist the temptation to fix all of them at once. `[READ]`

---

## 1. Graph context

### Node identity `[READ]`

| Field | Value |
| --- | --- |
| Node id | `file:frontend/federated_demo.py` |
| Type | `file` |
| Summary | "Teaching skeleton for the federated path, deliberately incomplete. Four tabs (Learn / Run / Compare / Predict) with 7 TODO markers the learner must fill; Run reads pre-recorded fl_rounds_sample.csv, Predict has only 3 form fields and no build_features() mapper so no prediction is produced yet. Its Predict submit button is inert and the live flwr run button is intentionally absent." |
| Tags | `streamlit`, `federated-path`, `teaching-scaffold`, `todos`, `flower` |
| Complexity | `moderate` |

### Nested helper nodes `[READ]`

Two `function` nodes are nested inside the file (both are **closures defined inside a `with st.tabs(...)`
block**, not module-level functions):

- `function:frontend/federated_demo.py:load_rounds` â€” `lineRange [42, 43]`, complexity `simple`.
  "Nested `@st.cache_data` closure inside the Run tab that reads the round-to-AUC CSV, keyed on the path
  string so repeated reruns reuse the cached frame. Currently wired to `frontend/fl_rounds_sample.csv`
  until the learner swaps in their own flwr output."
- `function:frontend/federated_demo.py:load_federated_model` â€” `lineRange [73, 81]`, complexity `moderate`.
  "Nested `@st.cache_resource` closure that lazily imports xgboost and loads
  `quickstart-xgboost/final_model.json` into a Booster, returning `None` when the file is absent so the
  Predict tab can degrade to a warning. Uses a `cache_resource` rather than `cache_data` because the Booster
  is an unhashable mutable model object."

Both line ranges match the source exactly. `[VERIFIED]`

### Neighbourhood (edges) `[READ]`

Outgoing:

| Edge | Type | Weight | Meaning |
| --- | --- | --- | --- |
| `file:frontend/federated_demo.py` â†’ `config:quickstart-xgboost/final_model.json` | `depends_on` | 0.6 | The model artefact it loads (line 84). |
| `file:frontend/federated_demo.py` â†’ `file:API/app.py` | `depends_on` | 0.6 | **Not an import.** The only `TODO-PREDICT` prose at line 101 points the reader at `API/app.py:33-86` for the 52-name column order. There is no Python-level coupling. |
| `file:frontend/federated_demo.py` â†’ `function:...:load_rounds` | `contains` | 1.0 | Nesting. |
| `file:frontend/federated_demo.py` â†’ `function:...:load_federated_model` | `contains` | 1.0 | Nesting. |

Incoming:

| Edge | Type | Meaning |
| --- | --- | --- |
| `config:quickstart-xgboost/final_model.json` â†’ `file:frontend/federated_demo.py` | `related` | Produced by `server_app.py`, consumed here. |
| `config:frontend/.streamlit/config.toml` â†’ `file:frontend/federated_demo.py` | `configures` | The dark theme. Unlike its sibling, this file applies **no** hand-written CSS of its own. |
| `document:docs/fed-guide.md` â†’ `file:frontend/federated_demo.py` | `documents` | Lab 3 is the walkthrough. |
| `document:README.md` â†’ `file:frontend/federated_demo.py` | `documents` | Run command. |
| `document:frontend/requirements.txt` â†’ `file:frontend/federated_demo.py` | `depends_on` | Unpinned `streamlit` only â€” **does not list `xgboost`**, which line 74 imports. A real packaging gap. |

There are **no `imports` edges** to first-party modules â€” the only third-party imports are `pandas`,
`streamlit`, and the lazily-imported `xgboost` (line 74). `[VERIFIED]` by grep of the whole file.

Notably: **there is no edge from this file to `endpoint:frontend/streamlit_app.py:/predict`** and none to
`file:API/app.py` as a runtime dependency. That absence is the architecture.

### Graph freshness caveat

`project.gitCommitHash` = `9a9e1db0fb8e2be8b0d1f3365caec5f904bb13d1` = `git rev-parse HEAD`, so the graph is
not stale against commits. **But the working tree has drift** that the graph cannot see: `API/app.py` is
modified, and `frontend/federated_demo.py`, `frontend/fl_rounds_sample.csv`, and `quickstart-xgboost/` are
untracked. Everything below was read from the live files, not from the graph. `[VERIFIED]`

---

## 2. Why this sits in `layer:frontend`

`layer:frontend` (`knowledge-graph.json:1254-1263`) holds five nodes: `file:frontend/streamlit_app.py`,
`endpoint:frontend/streamlit_app.py:/predict`, `file:frontend/federated_demo.py`,
`config:frontend/.streamlit/config.toml`, `document:frontend/requirements.txt`.

The layer description states the whole point: *"...while federated_demo.py is the deliberately incomplete
four-tab (Learn/Run/Compare/Predict) teaching scaffold whose run button is absent and whose Predict tab
loads final_model.json but produces nothing..."* `[READ]`

Why a federated demo belongs in a frontend layer: **it performs no federated computation.** It is not a
`ClientApp` or `ServerApp`; it is the observer seat. All privacy-preserving work already happened in
`quickstart-xgboost/quickstart_xgboost/{client_app,server_app}.py` during a `flwr run` whose output was
persisted to disk. What remains for this file is presentation and inference â€” reading a results table,
drawing a curve, comparing two numbers, and scoring one row against a saved model. That is precisely the
work a frontend layer exists to do. Layer membership here signals *consumer of federated artefacts*, not
*participant in federation*. `[READ]`

The absence of a run button is the load-bearing design decision, and it is stated twice: the docstring
(line 5) and the graph summary. A button that launched `flwr run` would need a subprocess, a working
directory that survives Windows path rules, and a live multi-branch topology â€” none of which belongs in a
Streamlit rerun loop. Pre-recorded numbers keep the demo deterministic and offline. `[READ]`

---

## 3. The defining contrast: centralized vs federated UI

This is the project's central idea, and the two files in `frontend/` make it concrete.

### `frontend/streamlit_app.py` (279 lines) â€” the centralized predictor `[READ]`

- Lines 6-7: `API_URL = "http://localhost:8000/predict"`, `gender_mapping = {0: "Male", 1: "Female"}`.
- Lines 18-159: ~140 lines of hand-written CSS duplicating the `.streamlit/config.toml` palette.
- Lines 188-241: a `customer_data` dict with **52 entries**, one per feature.
- Line 243: `submitted = st.form_submit_button("Predict Churn")`.
- Lines 259-270: the graph's `endpoint:frontend/streamlit_app.py:/predict` node â€” the inline
  `if submitted:` block that calls `requests.post(API_URL, json=customer_data)`.

The model lives in a **separate FastAPI process**. `API/app.py:203` declares `@app.post("/predict")`,
`:204` defines `predict_churn`, and `:206` does
`pdf = pdf.reindex(columns=FEATURE_NAMES, fill_value=0)` â€” the column-order enforcement. The API holds the
trained artefact in memory; the Streamlit app is a pure client.

### `frontend/federated_demo.py` (103 lines) â€” the federated demonstrator `[READ]`

- No `requests` import. No `API_URL`. No network call of any kind.
- Line 45: `csv_path = Path(__file__).parent / "fl_rounds_sample.csv"` â€” anchored to the **file**, so it
  resolves regardless of CWD.
- Line 84: `model = load_federated_model("quickstart-xgboost/final_model.json")` â€” a **bare relative
  string**, resolved against the CWD (see Â§8 for why that matters).

### The architectural point

In the federated design there is no party holding all customer rows, so there is no central model to call
for scoring either. **The aggregate model is itself the only artefact.** A federated predictor has no
serving endpoint by construction â€” the thing that would have been a server is a file, and `Booster.load_model`
replaces the HTTP round trip. That is why the edge from `federated_demo.py` to `API/app.py` is
`depends_on` weight 0.6 on *prose* and nothing else. `[READ]`

---

## 4. Internal structure â€” the four tabs

`st.tabs([...])` at **lines 20-22** returns four containers unpacked into
`tab_learn, tab_run, tab_compare, tab_predict`.

> **Execution model that matters:** by default `st.tabs` computes **all** tab content on every rerun
> regardless of which tab is visible. `[VERIFIED â€” official docs: "By default, all tab content is computed
> and sent to the frontend regardless of which tab is selected."]` This is why line 62 can read the `df`
> that was created at line 46 inside the *previous* `with` block. The Python script is one flat sequential
> pass; `with tab_run:` is not a function scope, and `df` is a plain module-level local. If tabs were ever
> switched to lazy execution (`on_change="rerun"`), this cross-tab variable reference would become the
> first thing to break. (Docs consulted are for v1.65.0; installed is 1.61.1. `[VERIFIED]` versions.)

### Tab 1 â€” `1. Learn` (lines 24-36): pure markdown, zero logic `[READ]`

A single `st.markdown` (lines 26-36) stating the one-round story in four steps:

1. Line 28 â€” server sends global trees (bytes in `Message.content["arrays"]["0"]`).
2. Line 29 â€” each branch boosts new trees on **local rows only** (`ClientApp.train`).
3. Line 30 â€” server **concatenates** trees (**FedXgbBagging, not averaging**).
4. Line 31 â€” repeat; rows never leave branches, only trees + `auc` / `num-examples` move.

Ends with `**TODO-LEARN:**` (line 33) asking why FedAvg cannot be used on trees, pointing at
`quickstart-xgboost/quickstart_xgboost/client_app.py:20-30`.

**The intended answer.** Gradient-boosted trees are discrete split structures with **no element-wise
correspondence across clients**. FedAvg assumes client models occupy the same coordinate system, so that
`mean(Î¸â‚â€¦Î¸â‚–)` is a meaningful parameter vector. Tree 3 on client A and tree 3 on client B are entirely
different objects â€” split feature, threshold, leaf values, and even tree count can differ. Averaging them
produces arithmetic on a meaningless correspondence. The cleanest illustration: if client A learned
`Age > 30` and client B learned `Age > 45`, averaging the thresholds gives `Age > 37.5` â€” a split **neither
client ever evaluated**, which no client has any data to justify. This is exactly what the hinted
`client_app.py:20-30` shows: `_local_boost()` slices out the **last N trees** and returns them for
*concatenation*, never for averaging. `[READ]` â€” I read `_local_boost` at those lines and confirmed it
does `bst_input[num_boosted_rounds() - num_local_round : num_boosted_rounds()]` then returns the slice.

### Tab 2 â€” `2. Run` (lines 38-55): replay recorded results `[READ]`

```
41:  @st.cache_data
42:  def load_rounds(path: str) -> pd.DataFrame:
43:      return pd.read_csv(path)
45:  csv_path = Path(__file__).parent / "fl_rounds_sample.csv"
46:  df = load_rounds(str(csv_path))
48:  st.dataframe(df, width="stretch")
49:  st.line_chart(df.set_index("round")[["auc"]])
51-55: st.info("TODO-RUN: ...")
```

- Line 48 uses the **modern `width="stretch"`** argument. I checked the installed library: Streamlit
  **1.61.1**, and `st.dataframe`'s signature has `width: 'Width' = 'stretch'`. `[VERIFIED]` Correction to a
  common claim: `use_container_width` is **deprecated, not removed** in 1.61.1 â€” it still exists in the
  signature, and passing it triggers
  `make_deprecated_name_warning(... "use_container_width=True -> use width='stretch'" ...)`. The file uses
  the non-deprecated spelling, which is correct either way.
- Line 49 sets `round` as the index so the line chart's x-axis is the communication round; `[["auc"]]` is
  the double-bracket column selection.
- `TODO-RUN` (line 52) instructs the learner to run Lab 2's `flwr run . --stream`, write a real
  `fl_rounds.csv`, **change line 45 to load it**, and add `st.metric` cards for final AUC and total
  examples. Today it loads the *sample*, so the curve is a placeholder, not a result. `[READ]`
- `frontend/fl_rounds_sample.csv` currently has 3 rows and 6 columns:
  `round,auc,num_examples_train,num_examples_eval,note` with auc `0.9910 / 0.9926 / 0.9929`
  (5633 train / 1410 eval per round). `[VERIFIED]` Last value `0.9929` is what line 62 picks up.

### Tab 3 â€” `3. Compare` (lines 57-67): the two numbers side by side `[READ]`

```
60:  central_auc = st.slider("Centralized AUC (from your notebook, Lab 0)", 0.5, 1.0, 0.82, 0.01)
61:  # TODO-COMPARE: replace 0.81 with your federated final AUC from Lab 2
62:  fed_auc = float(df["auc"].iloc[-1]) if len(df) else 0.81
64:  comp = pd.DataFrame({"model": ["centralized", "federated"], "auc": [central_auc, fed_auc]})
65:  st.bar_chart(comp.set_index("model"))
67:  st.info("TODO-COMPARE: write one sentence â€” why is federated slightly lower/higher? Think Non-IID.")
```

- Federated AUC comes **from the CSV** (`df["auc"].iloc[-1]` = `0.9929`), falling back to `0.81` when the
  frame is empty. `[VERIFIED]` The comment on line 61 is stale in a telling way: it says "replace 0.81" but
  line 62 already prefers the real final-round value; only the empty-frame fallback is hardcoded.
- **Design flaw worth naming:** `central_auc` is a `st.slider` with range `0.5â€“1.0` and default `0.82`. The
  viewer can drag the centralized number to make federated look better or worse. That is exactly the
  comparison the exercise is about, so letting the reader adjust one side of it undermines the exercise. The
  intended fix (and what `docs/fed-guide.md:57` calls "the number federated will be compared against in Lab
  3") is a hardcoded constant read off the notebook.
- **Bigger caveat:** the defaults produce a gap of `0.9929 - 0.82 = 0.173`, which is *not* "slightly
  lower/higher". The project's own README states that centralized ~0.93 and federated 0.991â€“0.993 **are not
  protocol-comparable** â€” they come from different evaluation setups. `TODO-COMPARE` invites an
  explanation of a gap that is largely a methodology artefact. A learner who dutifully answers "Non-IID"
  would be reasoning about the wrong cause. `[READ]` (README claim from the graph's `layer:documentation`
  description).

### Tab 4 â€” `4. Predict` (lines 69-103): load the aggregate, score locally `[READ]`

```
72:  @st.cache_resource
73:  def load_federated_model(path: str):
74:      import xgboost as xgb
76:      p = Path(path)
77:      if not p.exists():
78:          return None
79:      bst = xgb.Booster()
80:      bst.load_model(str(p))
81:      return bst
84:  model = load_federated_model("quickstart-xgboost/final_model.json")
85:  if model is None:  st.warning(...)      # 86-89
90:  else:               st.success(...)      # 91
93:  with st.form("mini_churn"):
94:      tenure   = st.slider("Tenure in Months", 0, 72, 12)
95:      monthly  = st.slider("Monthly Charge", 0.0, 200.0, 70.0)
96:      contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
97:      st.form_submit_button("Predict (TODO)")
99-103: st.info("TODO-PREDICT: ...")
```

- Line 74's **deferred import** is deliberate and idiomatic: `xgboost` is only needed for this tab, and
  keeping it inside the function means the app renders even when xgboost is missing or the model file is
  absent.
- Lines 77-78 return `None` instead of raising, so a missing artefact degrades to the warning at 86-89
  rather than a stack trace. This is what lets a fresh clone run the app before Lab 2.
- Line 91 calls `model.num_boosted_rounds()`. I loaded the real artefact: **6 boosted rounds, 52 features**.
  `[VERIFIED]` `docs/fed-guide.md:502` describes exactly this ("green success message showing 6 trees").
  The 52 confirms the `FEATURE_NAMES` contract the Predict tab must honour.
- Only **3 fields** (tenure, monthly charge, contract) versus `streamlit_app.py`'s 52 â€” deliberate: the
  point is the *federated* model, not a second full form. Line 102 says so explicitly: *"Keep full 52-field
  form in `streamlit_app.py`."*
- **Line 97's button is inert.** `st.form_submit_button(...)` is called without assigning its return value,
  so nothing consumes it and no prediction is ever computed. `[READ]` The graph node says the same:
  "no `build_features()` mapper, so its `st.form_submit_button("Predict (TODO)")` is inert and does
  nothing when clicked."

---

## 5. The two Streamlit caching decorators

Both are `@st.cache_*` on **nested closures** defined inside `with` tab blocks. That is legal and
intentional: the decorator is applied when the `def` executes on each rerun, producing a fresh decorated
object, and Streamlit keys the cache on the *function body hash plus arguments* â€” so the cache still hits
across reruns. `[READ]`

### `@st.cache_data` on `load_rounds` (line 41) â€” caches a **serialised** value

`load_rounds` returns a `pandas.DataFrame`. `cache_data` stores a **hashed copy**; on every call the caller
receives a **fresh copy**. Consequences:

- **Mutation is safe.** If the learner adds `df["gap"] = ...` after line 46, it cannot corrupt the cached
  entry for the next session or rerun.
- **Serialisation must succeed.** pandas handles this trivially; a non-picklable object would raise here.
- Cheap enough that even without the cache a 3-row CSV read would be tolerable â€” the decorator is here to
  establish the *pattern*, and to avoid re-reading a real multi-round `fl_rounds.csv` on every slider drag
  (the Compare tab's slider at line 60 reruns the script).

### `@st.cache_resource` on `load_federated_model` (line 72) â€” caches the **live** object

`cache_resource` stores the returned object **by reference**, shared across every session and every rerun.
The right choice here because:

- **Cost.** `Booster.load_model` parses a binary model; doing it on every widget interaction would be
  wasteful.
- **Non-picklable.** An `xgboost.Booster` holds a native C++ handle and is not meaningfully picklable;
  `cache_data`'s serialisation requirement would fail.
- **Shared identity matters.** A model instance is heavyweight state. One per process is the correct
  cardinality, matching how `API/app.py` keeps the trained model resident in memory (via `lifespan`).

The graph's own node summary reaches the same conclusion, with a slightly imprecise phrasing worth
correcting: it says the Booster is *"unhashable"*. The real reason is **not hashability** â€” `cache_data`
hashes the *function and its arguments* (here, the `path` string), not the return value. The reason is that
`cache_data` would have to serialise the return value, and a `Booster` is a live native object. `[READ]`

### Contrast with the sibling

`frontend/streamlit_app.py` needs **neither** decorator: it loads no model and no DataFrame, because the
model lives behind `http://localhost:8000/predict` and the form fields are local dict entries. That is
precisely the difference in the caching story, and it is a difference in architecture, not in style.

---

## 6. Two traps for whoever completes `TODO-PREDICT`

### Trap 1 â€” form widgets do not take effect until submit

Widgets inside `st.form` are **batched**. `[VERIFIED â€” official docs: "All changes made to a form will only
be sent to the Python backend when the form itself is submitted"; and "Before a form is submitted, all
widgets within that form will have default values".]` So `tenure`, `monthly`, `contract` return their
**defaults** (12, 70.0, `"Month-to-Month"`) on every rerun until the button is pressed.

Therefore the prediction logic must sit **outside** the `with st.form(...)` block, guarded by the button's
return value:

```python
with st.form("mini_churn"):
    tenure   = st.slider(...)
    monthly  = st.slider(...)
    contract = st.selectbox(...)
    submitted = st.form_submit_button("Predict (TODO)")   # capture it â€” line 97 does not

if submitted:
    ...  # build the row, DMatrix, model.predict
```

This is exactly the "Execute the process after the form" pattern in the docs, and exactly what
`streamlit_app.py:185` + `:243` + `:259` already does correctly with `submitted = False` â†’ `submitted = st.form_submit_button(...)` â†’ `if submitted:` at the top level. `[READ]` A second gotcha from the docs: `st.form_submit_button` is the *only* widget inside a form that can carry a callback.

### Trap 2 â€” `Booster.predict()` will not accept a raw numpy array

I tested this against the real artefact with the project's own venv (Streamlit 1.61.1 / xgboost 3.2.0 /
pandas 2.3.3): `[VERIFIED]`

```
raw ndarray predict raises: TypeError: ('Expecting data to be a DMatrix object, got: ', <class 'numpy.ndarray'>)
DMatrix predict -> [0.6285269]
inplace_predict ndarray: [0.6285269]
```

So `model.predict(row)` on a 1-row `numpy.ndarray` **raises `TypeError`**. The correct forms are:

```python
row = [0.0] * 52                      # must be len == model.num_features() == 52
dmat = xgb.DMatrix(np.array([row], dtype=np.float32), feature_names=FEATURE_NAMES)
prob  = model.predict(dmat)[0]        # float, e.g. 0.6285 for an all-zero row
```

or `model.inplace_predict(row)` / `model.predict(pd.DataFrame([row], columns=FEATURE_NAMES))`, both of
which accept an array/DataFrame. The `float32` dtype matters: XGBoost rejects mismatched dtypes and the
model was trained on the task's encoded float matrix.

The other half of the trap is **column order**. XGBoost matches positionally and will happily return a
probability from a misaligned row. Both authoritative lists exist: `API/app.py:33` `FEATURE_NAMES`
(`]` at line 86) and `quickstart-xgboost/quickstart_xgboost/task.py:14` (`]` at line 67) â€” 52 names each.
The file's `TODO-PREDICT` names `API/app.py:33-86`; `docs/fed-guide.md:401` names `task.py:14-67`. **For
this file the `task.py` list is the correct one**, because the federated booster was trained on the task's
feature matrix, not on the API's. The hint at line 101 â€” `reindex(columns=FEATURE_NAMES)` â€” is the API's
mechanism and is not what XGBoost needs; passing `feature_names=` to `DMatrix` (or a DataFrame with those
column names) is. `[VERIFIED]` both list locations.

---

## 7. External connections and data flow

### Connections `[READ]`

- `depends_on` â†’ `config:quickstart-xgboost/final_model.json` â€” the artefact read at line 84. It exists
  in the repo and I loaded it successfully. `[VERIFIED]`
- `related` (inbound) â†’ `config:quickstart-xgboost/final_model.json`, written by `server_app.py`.
- `depends_on` (prose only) â†’ `file:API/app.py:33-86` â€” the column-order source cited by `TODO-PREDICT`.
- Reads `frontend/fl_rounds_sample.csv` â€” **not represented in the graph at all**; the graph only models
  `requirements.txt` and `final_model.json` as inputs. A gap in the graph, not in the code. `[READ]`
- Documented by `docs/fed-guide.md` Â§"Lab 3 - Streamlit demo" (lines 216, 286, 313, 341, 382, 531) and
  `README.md`. `docs/fed-guide.md:531` independently confirms the relative-path fragility in Â§8 below.

### Data flow

```
fl_rounds_sample.csv  â”€â”€pd.read_csvâ”€â”€â–¶  DataFrame (cached_data)
                                            â”‚
                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                    â–¼                                               â–¼
      st.dataframe + line_chart (line 48-49)          df["auc"].iloc[-1]  (line 62)
                    = 0.9929                                        â”‚
                                                    st.bar_chart vs slider 0.82  (line 64-65)

quickstart-xgboost/final_model.json â”€â”€Booster.load_modelâ”€â”€â–¶ Booster  (cache_resource, 6 trees / 52 feats)
                    â”‚
                    â””â”€â”€ (once TODO-PREDICT is done) â”€â”€â–¶ 3 form inputs
                              + 49 zero-filled columns
                              â”€â”€â–¶ 52-col float32 row
                              â”€â”€â–¶ xgb.DMatrix(feature_names=FEATURE_NAMES)
                              â”€â”€â–¶ model.predict(dmat)[0]  â”€â”€â–¶ churn probability
```

The zero-fill is sanctioned by the guide: *"The remaining 44 columns ... = 0 (default). That's fine â€”
XGBoost handles missing/zero gracefully"* (`docs/fed-guide.md:398`). `[READ]`

---

## 8. Verification report

### TODO count â€” counted independently `[VERIFIED]`

`Select-String -Pattern "TODO"` on `frontend/federated_demo.py`:

- **7 lines** contain `TODO`: **7, 33, 52, 61, 67, 97, 100**
- **8 total occurrences** (line 7 has both `TODOs` and `TODO`)
- Resolving to **5 named exercise markers** across 6 of those lines:

| Marker | Line(s) | Form |
| --- | --- | --- |
| `TODO-LEARN` | 33 | markdown prose prompt |
| `TODO-RUN` | 52 | `st.info` prose prompt |
| `TODO-COMPARE` | 61 | `#` code comment â€” hardcode the federated AUC |
| `TODO-COMPARE` | 67 | `st.info` prose prompt â€” explain the gap |
| `TODO-PREDICT` | 100 | `st.info` prose prompt â€” implement scoring |

Plus two non-exercise uses: line 7 (docstring: *"TODOs are yours to complete"*) and line 97 (the button's
own label `"Predict (TODO)"`).

The graph's "7 TODO markers" corresponds to the **7 lines**. If you count named exercises, it is 5 markers
(6 instances, since `TODO-COMPARE` appears twice in two different registers).

### The model path is relative to the CWD, not the file `[VERIFIED]`

Line 84 passes a bare relative string: `load_federated_model("quickstart-xgboost/final_model.json")`.
Line 76 does `p = Path(path)` and line 77 `p.exists()` â€” no `__file__` anchoring. I tested resolution from
both directories:

```
CWD = D:\Privacy Based Customer Churn
  resolves from project root?  True
  resolves from frontend/?     False
```

So:

- `streamlit run frontend/federated_demo.py` from the project root â†’ **works**. The loader finds the file,
  and line 91 prints the 6-tree success message.
- `cd frontend; streamlit run federated_demo.py` â†’ `Path("quickstart-xgboost/final_model.json")` does not
  exist, `load_federated_model` returns `None`, and the tab **silently** shows only the line 86-89 warning.
  No error, no traceback â€” the app looks like it merely has no model yet, which is indistinguishable from
  "Lab 2 hasn't been run." Genuinely confusing failure mode. `[VERIFIED]`
- This is **not** an oversight of the same kind as line 45, which *does* use `Path(__file__).parent` for the
  CSV. The inconsistency inside one file is the smell. `docs/fed-guide.md:531` documents the behaviour and
  tells the reader to update the path if files move â€” so it is known, just not fixed. The one-line fix is
  `Path(__file__).parent.parent / "quickstart-xgboost" / "final_model.json"`, which would make it CWD-proof
  like the CSV already is.

### Other verified claims

| Claim | Status |
| --- | --- |
| Installed stack in `churnenv`: streamlit **1.61.1**, xgboost **3.2.0**, pandas **2.3.3** | `[VERIFIED]` |
| `width="stretch"` is valid and is the non-deprecated spelling; `use_container_width` is deprecated-but-present in 1.61.1 (not removed) | `[VERIFIED]` |
| `final_model.json` = 6 boosted rounds, **52 features** | `[VERIFIED]` |
| `Booster.predict(ndarray)` raises `TypeError: Expecting data to be a DMatrix object` | `[VERIFIED]` |
| `inplace_predict(ndarray)` returns the same value, no DMatrix needed | `[VERIFIED]` |
| All tab bodies execute every rerun (so line 62 can see `df` from line 46) | `[VERIFIED]` via official docs |
| Form widgets return defaults until submit | `[VERIFIED]` via official docs |
| `frontend/fl_rounds_sample.csv` = 3 rows, auc `0.9910/0.9926/0.9929` | `[VERIFIED]` |
| `streamlit_app.py` is 279 lines (not 278) with 52 form fields at lines 189-240 | `[VERIFIED]` |
| `client_app.py:20-30` `_local_boost` slices the last N trees for concatenation | `[READ]` |
| `API/app.py:203` `@app.post("/predict")`, `:206` `reindex(columns=FEATURE_NAMES, fill_value=0)` | `[READ]` |
| `xgboost` is imported but absent from `frontend/requirements.txt` | `[READ]` |

---

## 9. Summary of things to know before touching this file

1. **It is a worksheet.** Completing all 5 exercises defeats the purpose; `docs/fed-guide.md` Â§Lab 3 is the
   marking rubric. Do not "fix" it in a bulk refactor.
2. **Two tabs share state through a plain local.** `df` is defined at line 46 inside `with tab_run:` and
   read at line 62 inside `with tab_compare:`. This works only because tabs are not scopes.
3. **`TODO-COMPARE` asks about a gap that isn't a Non-IID gap.** 0.9929 vs 0.82 is a protocol mismatch; the
   repo's own README says so. Expect the "correct" answer to be "these numbers are not comparable."
4. **The centralized AUC is user-adjustable (line 60).** Hardcode it.
5. **`load_federated_model`'s path is CWD-dependent and fails silently.** Prefer `__file__` anchoring, as
   line 45 already does.
6. **The 52-column order for the *federated* model is `task.py:14-67`, not `API/app.py:33-86`.** The
   in-file hint points at the API list. Both are 52 names, but the booster was trained on the task's
   matrix.
7. **Capturing the submit button (line 97) and wrapping the row in `DMatrix` are both mandatory** â€” one is
   a Streamlit batching rule, the other is an XGBoost type error I reproduced.
