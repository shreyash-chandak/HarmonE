# Original HarmonE: From Data Preprocessing to Streaming Adaptation

## 1. Overview

The original HarmonE implementation treats the PEMS traffic-flow dataset as a **univariate, ordered time series**. The overall workflow consists of four main stages:

```text
Raw PEMS Data
      ↓
Data Preprocessing & Sequential Split
      ↓
Initial Model Training
      ↓
Streaming Inference + Runtime Adaptation
```

The important distinction is that the initial model training and the subsequent streaming phase serve different purposes. A small portion of the data is used to establish the initial model versions, while the majority of the data is reserved for running and evaluating the adaptive system.

## 2. Data Preprocessing

The preprocessing script, `tools/store_pems.py`, reads all CSV files from `data/pems/raw/`. For each file, it extracts only the `Flow (Veh/5 Minutes)` column and concatenates the resulting data into a single ordered dataframe. The column is then renamed to `flow`.

Thus, the original pipeline intentionally reduces the PEMS records to:

```text
flow
317
306
305
299
290
...
```

The timestamp and other metadata are not retained by this preprocessing step. Temporal information is instead represented implicitly by the ordering of the observations.

The combined stream is then split sequentially rather than randomly:

```text
Entire PEMS stream
│
├──────────── 10% ────────────┬──────────────────── 90% ────────────────────┐
│                             │                                             │
│       Initial training      │             System simulation               │
│                             │                                             │
│   flow_data_train.csv       │             flow_data_test.csv               │
└─────────────────────────────┴─────────────────────────────────────────────┘
```

The default ratio is 10%/90%. The first portion is intended for training the initial model versions, while the remaining 90% is used to simulate the running system.

## 3. Initial Model Training

`train_models.py` loads `flow_data_train.csv` and extracts the `flow` values. The data is normalized using a `MinMaxScaler`.

The training data is then split again:

```text
flow_data_train
      │
      ├── 80% → model-training portion
      │
      └── 20% → local test portion
```

This means that, with the default preprocessing configuration, the models are ultimately trained using approximately **8% of the complete PEMS stream** (80% of the initial 10%).

### Temporal sequence construction

The models are trained as next-step predictors rather than as independent row classifiers/regressors.

A sequence of five consecutive flow observations is constructed:

```text
flow[t-5]  flow[t-4]  flow[t-3]  flow[t-2]  flow[t-1]
                                      │
                                      ▼
                                  predict flow[t]
```

The `create_sequences()` function constructs these input/output pairs, with `seq_length = 5`.

Three models are trained on these sequences:

* **LSTM**
* **Ridge/Linear Regression**
* **Linear-kernel SVR**

The LSTM receives the five-value sequence and predicts the next flow value. Ridge and SVR are trained using the same sequence representation.
The resulting models are saved both as the current models in `models/` and as versioned copies in `versionedMR/`.

## 4. Transition to Streaming

Once the initial models have been trained, the remaining 90% of the original data forms the system's simulation stream.

This is an important point: the 90% portion is not simply a conventional held-out test set. The repository describes it as the data used to **simulate the system**.

Conceptually:

```text
                INITIALIZATION
                     │
                     ▼
              First 10% of data
                     │
                     ▼
              Train initial
                model versions
                     │
                     ▼
══════════════════════════════════════════════
             STREAMING PHASE
══════════════════════════════════════════════
                     │
                     ▼
            Next flow observation
                     │
                     ▼
              Model inference
                     │
                     ▼
          Monitoring / adaptation
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Model switching         Retraining
          │                     │
          └──────────┬──────────┘
                     ▼
              Continue stream
```

The purpose of reserving 90% of the data is therefore to provide a sufficiently long period over which HarmonE's runtime adaptation mechanisms can operate.

## 5. Drift Injection

The streaming/test portion can optionally be modified to introduce controlled concept drift.

`tools/induce_drift.py` loads `flow_data_test.csv` and operates directly on its `flow` column.

The user specifies:

* Start index
* End index
* Scale factor
* Shift amount

The selected region is modified according to:

```text
new_flow = old_flow × scale + shift
```

The drift is therefore introduced into selected regions of the streaming data rather than during initial model training.

This produces a stream containing regions where the statistical behavior of the traffic-flow data has deliberately changed.

## 6. Runtime HarmonE Phase

The prepared models and streaming data are then used by the HarmonE runtime.

The repository defines several configurations, including:

* `harmone` — dynamic model switching plus drift detection
* `switch` — dynamic model switching without drift detection
* `switch+retrain` — model switching plus periodic retraining
* Single-model baselines, with or without retraining.

For the full HarmonE configuration, the inference system runs alongside the management system. The inference subsystem uses the models from the `models/` directory, while the management system launches the monitoring/adaptation threads.

The resulting experiment therefore evaluates not just whether an individual model predicts well, but how the **self-adaptive system behaves over a long-running stream**.

## 7. Complete Pipeline

Putting the pieces together:

```text
                    RAW PEMS DATA
                          │
                          ▼
              ┌──────────────────────┐
              │ tools/store_pems.py  │
              │                      │
              │ Extract FLOW only    │
              │ Merge files          │
              │ Sequential split     │
              └──────────┬───────────┘
                         │
              ┌──────────┴───────────┐
              │                      │
              ▼                      ▼
          First 10%                Remaining 90%
              │                      │
              ▼                      ▼
    flow_data_train.csv      flow_data_test.csv
              │                      │
              ▼                      │
     tools/train_models.py          │
              │                      │
       ┌──────┼──────┐               │
       ▼      ▼      ▼               │
      LSTM  Ridge   SVR              │
       │      │      │               │
       └──────┼──────┘               │
              ▼                      │
       Initial model versions        │
              │                      │
              └──────────┬───────────┘
                         │
                         ▼
                HARMONE RUNTIME
                         │
                  Streaming data
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Inference             Monitoring
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                    Model switch          Retraining
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                              Continue stream
```

## 8. Key Takeaway

The original HarmonE experiment is **not simply a conventional train/test ML experiment**.

The 10%/90% split establishes two different phases:

**Initialization:** use a small historical portion to train the initial model versions.

**Streaming evaluation:** use the remaining large portion as a long-running stream on which HarmonE performs inference, monitors the system, detects drift, switches models, and/or retrains depending on the selected approach.

Consequently, the unusual **10% training / 90% simulation** ratio is deliberate within the original HarmonE experimental design. It prioritizes having a long-running stream for evaluating self-adaptation rather than maximizing the amount of data available for initial model training.
