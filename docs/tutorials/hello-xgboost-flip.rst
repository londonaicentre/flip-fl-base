Adapting NVFlare Hello-XGBoost to FLIP Deployment
==================================================

Overview
--------

This tutorial shows how to take the `NVFlare Hello-XGBoost example
<https://nvflare.readthedocs.io/en/main/hello-world/hello-xgboost/index.html>`_
and adapt it to run on the **FLIP platform**.

**What you will build**: A federated binary classification job that trains an
XGBoost model on the HIGGS particle physics dataset across multiple sites,
coordinated by FLIP's server-side aggregation and deployed via the FLIP-API.

**Why this tutorial matters**: The previous tutorials (Hello-World, Spleen
Segmentation) use PyTorch neural networks — models whose parameters are
continuous tensors that can be averaged via FedAvg. XGBoost is fundamentally
different: it builds **decision trees**, not weight matrices. This tutorial
shows how FLIP accommodates non-neural-network machine learning, exposing the
platform's internal architecture and its extension points.

**What you will learn**:

- How FLIP handles tabular (non-imaging) data
- How to implement ``FLIP_TRAINER(ClientAlgo)`` for XGBoost
- How federated tree-based learning differs from federated deep learning
- How to extend FLIP's architecture when the default FedAvg aggregation
  doesn't directly apply


The Federated XGBoost Challenge
-------------------------------

In NVFlare's Hello-XGBoost example, three distinct federated strategies are
demonstrated:

1. **Histogram-based (horizontal)**: XGBoost's native federated plugin
   aggregates gradient histograms across sites via a gRPC server. Each site
   contributes its local gradient statistics, and the server builds trees from
   the combined information.

2. **Tree-based (bagging)**: Each site independently trains a few trees.
   The server collects and merges all trees into a single ensemble.

3. **Vertical**: Sites hold different features for the same samples, using
   Private Set Intersection (PSI) to align rows before training.

NVFlare provides specialised recipes (``XGBHorizontalRecipe``,
``XGBBaggingRecipe``, ``XGBVerticalRecipe``) and custom controllers for each.

**FLIP's architecture** centres on the **ClientAlgo interface** with
**ScatterAndGather** (FedAvg) aggregation — designed for neural networks where
model parameters are continuous arrays that can be averaged. XGBoost models are
collections of decision trees, which **cannot be numerically averaged**.

This means we need to choose an approach that works within or extends FLIP's
framework. This tutorial implements two complementary strategies:

- **Strategy A** — Cyclic (sequential) XGBoost training, where each site adds
  boosting rounds to a shared model passed sequentially between clients.
- **Strategy B** — Federated gradient-boosted MLP, using a neural network that
  can be federated via standard FedAvg on the same HIGGS dataset.

Both strategies use the same data loading, FLIP integration, and deployment
pipeline. Strategy A demonstrates FLIP's flexibility; Strategy B shows the
natural fit between neural networks and FedAvg.

.. table::
   :widths: 20 40 40

   ======================  ====================================  ====================================
   Aspect                  NVFlare Hello-XGBoost                  FLIP Deployment
   ======================  ====================================  ====================================
   Model                   XGBoost Booster (native)               XGBoost Booster / PyTorch MLP
   FL strategy             Histogram-based / Bagging              Cyclic model passing / FedAvg
   Data format             CSV + DMatrix                          FLIP ``get_dataframe()`` → DMatrix
   Job config              ``XGBHorizontalRecipe`` Python API     Declarative JSON configs
   Execution               ``SimEnv``                             REST submission to NVFlare server
   Aggregation             Gradient histogram aggregation         Model passthrough / weight averaging
   ======================  ====================================  ====================================


Before You Start
----------------

1. Review the NVFlare XGBoost examples:

   .. code-block:: bash

       git clone https://github.com/NVIDIA/NVFlare.git
       cd NVFlare/examples/advanced/xgboost/fedxgb

   Key files:

   - ``job.py`` — ``XGBHorizontalRecipe`` + ``SimEnv`` launcher
   - ``job_tree.py`` — ``XGBBaggingRecipe`` (bagging/cyclic modes)
   - ``higgs_data_loader.py`` — ``XGBDataLoader`` for HIGGS dataset

2. Install dependencies:

   .. code-block:: bash

       pip install flip-utils xgboost scikit-learn pandas torch


The HIGGS Dataset
-----------------

The HIGGS dataset is a benchmark for binary classification in particle physics:

- **Task**: Distinguish Higgs boson signal events from background processes
- **Size**: 11 million instances, 28 features, 1 binary label
- **Features**: Kinematic properties of particle detector measurements
- **Source**: UCI Machine Learning Repository

This is a **tabular classification** task — no images, no sequences, just rows
and columns. It demonstrates that FLIP is not limited to medical imaging.

In the federated setting, the dataset is split horizontally: each site has
different samples (rows) with the same features (columns). This mirrors a
real-world scenario where different institutions collect the same type of data
independently.


Dependencies
------------

Create ``requirements.txt``:

.. code-block:: text

    torch>=2.0.0
    xgboost>=2.0.0
    scikit-learn>=1.3.0
    pandas>=2.0.0
    numpy
    nvflare>=2.7.1
    flip-utils>=0.1.0


Strategy A: Cyclic XGBoost on FLIP
-----------------------------------

In cyclic federated XGBoost, clients take turns adding boosting rounds to a
shared model. The server acts as a relay, passing the model from one client to
the next:

.. code-block:: text

    Round 1: Server → Client A (trains 10 trees) → Server
    Round 2: Server → Client B (adds 10 trees)  → Server
    Round 3: Server → Client A (adds 10 trees)  → Server
    ...

This approach works with FLIP's ``ScatterAndGather`` controller because the
"weights" exchanged are the serialised XGBoost model bytes — the server simply
passes them through without averaging.

Job Structure
~~~~~~~~~~~~~

.. code-block:: text

    hello-xgboost-flip/
    ├── custom/
    │   ├── trainer.py           # FLIP_TRAINER with XGBoost
    │   ├── validator.py         # FLIP_VALIDATOR with XGBoost
    │   ├── models.py            # XGBoost model wrapper + get_model()
    │   ├── data_utils.py        # HIGGS data loading via FLIP
    │   ├── config.json          # Hyperparameters
    │   └── requirements.txt     # Dependencies
    └── config/
        ├── config_fed_server.json
        └── config_fed_client.json

Step 1: Create data_utils.py — HIGGS Data Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In NVFlare, the ``HIGGSDataLoader`` reads a pre-split JSON file that points to
CSV row ranges. In FLIP, data is accessed through the platform API. For tabular
data, ``flip.get_dataframe()`` returns the data directly as a pandas DataFrame.

.. code-block:: python

    """
    HIGGS data loading utilities for FLIP.

    In production, the HIGGS data (or equivalent tabular clinical data) is
    stored in the institution's database and accessed via FLIP's SQL query
    interface. For local development, FLIP reads from a CSV file configured
    in .env.development.
    """

    from __future__ import annotations

    import logging

    import numpy as np
    import pandas as pd
    import xgboost as xgb

    from flip import FLIP

    logger = logging.getLogger(__name__)


    def load_higgs_data(
        flip: FLIP,
        project_id: str,
        query: str,
        val_split: float = 0.2,
    ) -> tuple[xgb.DMatrix, xgb.DMatrix]:
        """Load HIGGS data via FLIP API and return XGBoost DMatrix objects.

        The dataframe returned by FLIP is expected to have:
          - Column 0: binary label (0 or 1)
          - Columns 1-28: kinematic features

        Args:
            flip: Initialised FLIP instance.
            project_id: FLIP project identifier.
            query: SQL query to retrieve the data.
            val_split: Fraction of data to reserve for validation.

        Returns:
            Tuple of (train DMatrix, validation DMatrix).
        """
        df = flip.get_dataframe(project_id, query)
        logger.info(f"Loaded {len(df)} rows from FLIP")

        # The HIGGS dataset has the label in the first column
        if "label" in df.columns:
            y = df["label"].values
            X = df.drop("label", axis=1).values
        else:
            # Raw HIGGS CSV format: first column is label
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values

        # Train/validation split
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        logger.info(
            f"Train: {dtrain.num_row()} rows, "
            f"Val: {dval.num_row()} rows, "
            f"Features: {dtrain.num_col()}"
        )
        return dtrain, dval


Step 2: Create models.py — XGBoost Model Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FLIP's ``PTFileModelPersistor`` expects a PyTorch ``nn.Module`` from
``get_model()``. For XGBoost, we create a thin wrapper that satisfies this
interface while internally managing the XGBoost Booster.

.. code-block:: python

    """
    XGBoost model wrapper for FLIP.

    FLIP's server-side persistor expects a PyTorch nn.Module returned by
    get_model(). For XGBoost, we wrap the booster state in a Module that
    stores model bytes as a buffer — this integrates with the standard
    FLIP weight serialisation pipeline while keeping XGBoost as the actual
    training engine.
    """

    import json
    from pathlib import Path

    import numpy as np
    import torch
    from torch import nn


    def load_config():
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, "r") as f:
            return json.load(f)


    class XGBoostModelWrapper(nn.Module):
        """Wrapper that stores XGBoost model bytes as a PyTorch parameter.

        The model state_dict contains a single key: "model_bytes" — a 1D
        float tensor encoding the raw XGBoost booster bytes. This allows
        the standard NVFlare weight exchange pipeline to serialise and
        deserialise the XGBoost model without modification.

        The wrapper also stores default XGBoost parameters for initialisation.
        """

        def __init__(self):
            super().__init__()
            config = load_config()
            xgb_params = config.get("xgb_params", {})

            # Store default XGBoost parameters as JSON bytes
            params_bytes = json.dumps(xgb_params).encode("utf-8")
            params_array = np.frombuffer(params_bytes, dtype=np.uint8).astype(np.float32)
            self.register_buffer(
                "xgb_params",
                torch.from_numpy(params_array),
            )

            # Placeholder for model bytes — empty until first training round
            self.register_buffer(
                "model_bytes",
                torch.zeros(1, dtype=torch.float32),
            )

        def forward(self, x):
            """Not used — XGBoost training is handled in the trainer."""
            raise NotImplementedError(
                "XGBoostModelWrapper.forward() should not be called. "
                "Use the XGBoost booster directly for prediction."
            )

        def get_xgb_params(self) -> dict:
            """Decode the stored XGBoost parameters."""
            params_array = self.xgb_params.numpy().astype(np.uint8)
            return json.loads(params_array.tobytes().decode("utf-8"))


    _model = XGBoostModelWrapper()


    def get_model() -> nn.Module:
        """Factory function — required by FLIP's PTFileModelPersistor."""
        return _model

**How the wrapper works:**

The ``model_bytes`` buffer starts as a single zero (empty model). After the
first training round, the trainer serialises the XGBoost booster into bytes,
converts them to a float32 tensor, and stores them in this buffer. The NVFlare
weight exchange pipeline (``ExchangeObject``, ``Shareable``, ``DXO``) handles
the tensor as if it were neural network weights.

This is the key insight: **FLIP's weight exchange is type-agnostic** — it
transmits numpy arrays keyed by name. Any model that can be serialised to
numpy arrays can participate in the federated pipeline.


Step 3: Create trainer.py — XGBoost Training Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    """
    XGBoost Trainer — MONAI FL ClientAlgo interface for FLIP.

    Implements federated XGBoost using cyclic model passing: the server
    sends the current model, each client adds boosting rounds on local
    data, and returns the updated model.

    Key difference from neural network trainers: weights are serialised
    XGBoost booster bytes, not PyTorch state dicts.
    """

    from __future__ import annotations

    import json
    import logging
    from pathlib import Path

    import numpy as np
    import xgboost as xgb
    from data_utils import load_higgs_data
    from monai.fl.client.client_algo import ClientAlgo
    from monai.fl.utils.constants import WeightType
    from monai.fl.utils.exchange_object import ExchangeObject
    from nvflare.app_common.app_constant import AppConstants

    from flip import FLIP
    from flip.nvflare.metrics import send_metrics_value


    class FLIP_TRAINER(ClientAlgo):
        """XGBoost trainer using the MONAI FL ClientAlgo interface.

        Lifecycle:
          1. initialize(extra) — load config, FLIP data → DMatrix (once)
          2. train(data, extra) — receive model, add boosting rounds (per round)
          3. get_weights(extra) — serialise booster and return (per round)
        """

        def __init__(
            self,
            project_id: str = "",
            query: str = "",
            train_task_name: str = AppConstants.TASK_TRAIN,
        ) -> None:
            self._project_id = project_id
            self._query = query
            self._train_task_name = train_task_name
            self.logger = logging.getLogger(self.__class__.__name__)

        def initialize(self, extra=None):
            """One-time setup: load config and HIGGS data via FLIP."""
            working_dir = Path(__file__).parent.resolve()
            with open(str(working_dir / "config.json")) as f:
                config = json.load(f)

            self._local_rounds = config["LOCAL_ROUNDS"]
            self._xgb_params = config["xgb_params"]
            self._val_split = config.get("VAL_SPLIT", 0.2)

            # Load data through FLIP API
            self.flip = FLIP()
            self.dtrain, self.dval = load_higgs_data(
                flip=self.flip,
                project_id=self._project_id,
                query=self._query,
                val_split=self._val_split,
            )
            self.logger.info(
                f"Loaded {self.dtrain.num_row()} train, "
                f"{self.dval.num_row()} val samples"
            )

            self._booster = None
            self._n_iterations = 0

        def train(self, data, extra=None):
            """Receive global model, add LOCAL_ROUNDS boosting rounds.

            Args:
                data: ExchangeObject with serialised XGBoost model bytes.
                extra: Dict with fl_ctx, abort_signal, current_round.
            """
            fl_ctx = (extra or {}).get("fl_ctx")
            current_round = (extra or {}).get("current_round", 0)

            # Deserialise incoming model
            model_bytes = data.weights.get("model_bytes", np.zeros(1))
            model_bytes_raw = model_bytes.astype(np.uint8).tobytes()

            if len(model_bytes_raw) > 1:
                # Load existing model and continue training
                self._booster = xgb.Booster(params=self._xgb_params)
                self._booster.load_model(bytearray(model_bytes_raw))
                self.logger.info(
                    f"Round {current_round}: loaded model with "
                    f"{self._booster.num_boosted_rounds()} trees, "
                    f"adding {self._local_rounds} more"
                )
                # Continue training from the existing model
                self._booster = xgb.train(
                    params=self._xgb_params,
                    dtrain=self.dtrain,
                    num_boost_round=self._local_rounds,
                    evals=[(self.dval, "val")],
                    xgb_model=self._booster,
                    verbose_eval=False,
                )
            else:
                # First round: train from scratch
                self.logger.info(
                    f"Round {current_round}: training from scratch "
                    f"with {self._local_rounds} trees"
                )
                self._booster = xgb.train(
                    params=self._xgb_params,
                    dtrain=self.dtrain,
                    num_boost_round=self._local_rounds,
                    evals=[(self.dval, "val")],
                    verbose_eval=False,
                )

            self._n_iterations = self._booster.num_boosted_rounds()

            # Evaluate and report metrics
            preds = self._booster.predict(self.dval)
            accuracy = float(
                np.mean((preds > 0.5).astype(int) == self.dval.get_label())
            )
            self.logger.info(
                f"Round {current_round}: {self._n_iterations} total trees, "
                f"val accuracy: {accuracy:.4f}"
            )
            if fl_ctx is not None:
                send_metrics_value(
                    label="val_accuracy", value=accuracy,
                    fl_ctx=fl_ctx, flip=self.flip,
                )

        def get_weights(self, extra=None):
            """Serialise the XGBoost booster and return as ExchangeObject.

            The booster is converted to raw bytes → uint8 numpy array →
            float32 array (required by NVFlare's weight serialisation).
            """
            if self._booster is None:
                return ExchangeObject(
                    weights={"model_bytes": np.zeros(1, dtype=np.float32)},
                    weight_type=WeightType.WEIGHTS,
                    statistics={"num_steps": 0},
                )

            raw_bytes = self._booster.save_raw()
            model_array = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)

            return ExchangeObject(
                weights={"model_bytes": model_array},
                weight_type=WeightType.WEIGHTS,
                statistics={"num_steps": self._n_iterations},
            )

        def finalize(self, extra=None):
            pass

**Key design decisions:**

- **Model serialisation**: ``booster.save_raw()`` returns bytes. We convert to
  ``float32`` numpy array because NVFlare's DXO pipeline expects numeric arrays.
  The uint8 → float32 conversion is lossless (all byte values 0-255 are exactly
  representable as float32).

- **Continuation training**: ``xgb.train(..., xgb_model=self._booster)`` adds
  new trees to the existing model rather than replacing it. This is the "cyclic"
  training pattern — each client contributes additional trees.

- **No weight averaging**: The full model is returned as ``WeightType.WEIGHTS``
  (not ``WEIGHT_DIFF``). For a single-client setup, the server passes the model
  through unchanged. For multi-client setups, see the aggregation section below.


Step 4: Create validator.py — XGBoost Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    """
    XGBoost Validator — MONAI FL ClientAlgo interface for FLIP.

    Evaluates the federated XGBoost model on the local validation set.
    """

    from __future__ import annotations

    import json
    import logging
    from pathlib import Path

    import numpy as np
    import xgboost as xgb
    from data_utils import load_higgs_data
    from monai.fl.client.client_algo import ClientAlgo
    from monai.fl.utils.exchange_object import ExchangeObject
    from nvflare.app_common.app_constant import AppConstants
    from sklearn.metrics import accuracy_score, roc_auc_score

    from flip import FLIP
    from flip.nvflare.metrics import send_metrics_value


    class FLIP_VALIDATOR(ClientAlgo):
        """XGBoost validator — evaluates global model on local data.

        Lifecycle:
          1. initialize(extra) — load data via FLIP
          2. evaluate(data, extra) — deserialise model, compute metrics
        """

        def __init__(
            self,
            project_id: str = "",
            query: str = "",
            validate_task_name: str = AppConstants.TASK_VALIDATION,
        ) -> None:
            self._project_id = project_id
            self._query = query
            self._validate_task_name = validate_task_name
            self.logger = logging.getLogger(self.__class__.__name__)

        def initialize(self, extra=None):
            """Load validation data via FLIP."""
            working_dir = Path(__file__).parent.resolve()
            with open(str(working_dir / "config.json")) as f:
                config = json.load(f)

            self.flip = FLIP()
            _, self.dval = load_higgs_data(
                flip=self.flip,
                project_id=self._project_id,
                query=self._query,
                val_split=config.get("VAL_SPLIT", 0.2),
            )
            self._xgb_params = config["xgb_params"]

        def evaluate(self, data, extra=None):
            """Deserialise XGBoost model and compute metrics.

            Args:
                data: ExchangeObject with serialised model bytes.
                extra: Dict with fl_ctx and abort_signal.

            Returns:
                ExchangeObject with metrics dict.
            """
            fl_ctx = (extra or {}).get("fl_ctx")

            model_bytes = data.weights.get("model_bytes", np.zeros(1))
            model_bytes_raw = model_bytes.astype(np.uint8).tobytes()

            if len(model_bytes_raw) <= 1:
                self.logger.warning("No model received for validation")
                return ExchangeObject(metrics={"val_accuracy": 0.0, "val_auc": 0.0})

            booster = xgb.Booster(params=self._xgb_params)
            booster.load_model(bytearray(model_bytes_raw))

            preds = booster.predict(self.dval)
            labels = self.dval.get_label()

            accuracy = accuracy_score(labels, (preds > 0.5).astype(int))
            auc = roc_auc_score(labels, preds)

            self.logger.info(
                f"Validation — Accuracy: {accuracy:.4f}, AUC: {auc:.4f}"
            )

            if fl_ctx is not None:
                send_metrics_value(
                    label="val_accuracy", value=accuracy,
                    fl_ctx=fl_ctx, flip=self.flip,
                )
                send_metrics_value(
                    label="val_auc", value=auc,
                    fl_ctx=fl_ctx, flip=self.flip,
                )

            return ExchangeObject(
                metrics={"val_accuracy": accuracy, "val_auc": auc}
            )


Step 5: Create config.json
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
      "job_type": "standard",
      "LOCAL_ROUNDS": 10,
      "GLOBAL_ROUNDS": 5,
      "VAL_SPLIT": 0.2,
      "xgb_params": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 8,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "tree_method": "hist"
      }
    }

- ``LOCAL_ROUNDS: 10`` — each client adds 10 trees per federated round
- ``GLOBAL_ROUNDS: 5`` — 5 rounds of cyclic training = 50 total trees
- ``xgb_params.objective: "binary:logistic"`` — binary classification with
  logistic loss (matches the HIGGS task)
- ``xgb_params.max_depth: 8`` — tree depth (NVFlare example uses 8)
- ``xgb_params.eta: 0.1`` — learning rate (shrinkage applied to each tree)
- ``xgb_params.tree_method: "hist"`` — histogram-based tree construction
  (fastest for large datasets)


Step 6: Create NVFlare Config Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Server config** (``config/config_fed_server.json``):

.. code-block:: json

    {
      "format_version": 2,
      "model_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "python_path": [".", "custom"],
      "global_rounds": 5,
      "min_clients": 1,
      "server": {
        "heart_beat_timeout": 600
      },
      "task_data_filters": [],
      "task_result_filters": [],
      "components": [
        {
          "id": "persistor",
          "path": "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor",
          "args": {
            "model": {
              "path": "models.get_model"
            }
          }
        },
        {
          "id": "shareable_generator",
          "path": "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator",
          "args": {}
        },
        {
          "id": "aggregator",
          "name": "InTimeAccumulateWeightedAggregator",
          "args": {
            "expected_data_kind": "WEIGHTS"
          }
        },
        {
          "id": "model_locator",
          "path": "flip.nvflare.components.PTModelLocator",
          "args": {}
        },
        {
          "id": "json_generator",
          "path": "flip.nvflare.components.ValidationJsonGenerator",
          "args": {}
        },
        {
          "id": "flip_server_event_handler",
          "path": "flip.nvflare.components.ServerEventHandler",
          "args": {
            "model_id": "{model_id}"
          }
        },
        {
          "id": "persist_and_cleanup",
          "path": "flip.nvflare.components.PersistToS3AndCleanup",
          "args": {
            "model_id": "{model_id}",
            "persistor_id": "persistor"
          }
        }
      ],
      "workflows": [
        {
          "id": "init_training",
          "path": "flip.nvflare.controllers.InitTraining",
          "args": {
            "model_id": "{model_id}",
            "min_clients": "{min_clients}"
          }
        },
        {
          "id": "scatter_and_gather",
          "path": "flip.nvflare.controllers.ScatterAndGather",
          "args": {
            "min_clients": "{min_clients}",
            "num_rounds": "{global_rounds}",
            "start_round": 0,
            "wait_time_after_min_received": 10,
            "aggregator_id": "aggregator",
            "persistor_id": "persistor",
            "shareable_generator_id": "shareable_generator",
            "train_task_name": "train",
            "train_timeout": 0,
            "model_id": "{model_id}",
            "ignore_result_error": false
          }
        },
        {
          "id": "cross_site_validate",
          "name": "CrossSiteModelEval",
          "path": "flip.nvflare.controllers.CrossSiteModelEval",
          "args": {
            "model_locator_id": "model_locator",
            "participating_clients": "{participating_clients}",
            "validation_timeout": 12000,
            "model_id": "{model_id}"
          }
        }
      ]
    }

**Client config** (``config/config_fed_client.json``):

.. code-block:: json

    {
      "format_version": 2,
      "project_id": "",
      "query": "SELECT * FROM higgs_data;",
      "local_rounds": 10,
      "executors": [
        {
          "tasks": ["init_training", "post_validation"],
          "executor": {
            "path": "flip.nvflare.components.CleanupImages",
            "args": {}
          }
        },
        {
          "tasks": ["train", "submit_model"],
          "executor": {
            "path": "flip.nvflare.executors.RUN_MONAI_FL_TRAINER",
            "args": {
              "project_id": "{project_id}",
              "query": "{query}"
            }
          }
        },
        {
          "tasks": ["validate"],
          "executor": {
            "path": "flip.nvflare.executors.RUN_MONAI_FL_VALIDATOR",
            "args": {
              "project_id": "{project_id}",
              "query": "{query}"
            }
          }
        }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
        {
          "id": "flip_client_event_handler",
          "path": "flip.nvflare.components.ClientEventHandler",
          "args": {}
        }
      ]
    }

**Note**: The ``task_result_filters`` section omits ``PercentilePrivacy`` because
that filter clips extreme weight values — meaningful for neural network weight
diffs, but destructive for serialised XGBoost model bytes. Privacy for XGBoost
is handled at the data level (each site trains only on local data) rather than
the weight level.


Data Preparation
----------------

**Option 1: Download the HIGGS dataset for local development**

.. code-block:: bash

    # Download HIGGS dataset (2.6 GB compressed)
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
    gunzip HIGGS.csv.gz

    # For quick testing, use a subset
    head -100000 HIGGS.csv > data/higgs_sample.csv

**Option 2: Use scikit-learn for a quick test**

For rapid prototyping, generate a synthetic binary classification dataset:

.. code-block:: python

    from sklearn.datasets import make_classification
    import pandas as pd

    X, y = make_classification(
        n_samples=10000, n_features=28, n_informative=20,
        n_redundant=5, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(28)])
    df.insert(0, "label", y)
    df.to_csv("data/higgs_sample.csv", index=False)

Configure local development in ``.env.development``:

.. code-block:: bash

    LOCAL_DEV=true
    DEV_DATAFRAME=../data/higgs_sample.csv
    JOB_TYPE=standard


Multi-Client Considerations: Aggregation for XGBoost
-----------------------------------------------------

With a **single client**, the cyclic approach works seamlessly — the server
receives the model and passes it back next round.

With **multiple clients**, the server's ``InTimeAccumulateWeightedAggregator``
will attempt to average the model byte arrays from different clients. Since
these are serialised tree structures (not continuous parameters), byte-level
averaging produces garbage.

There are three solutions:

**Solution 1: Use min_clients=1 with sequential rounds**

Set ``min_clients: 1`` in the server config. Each round, only one client trains.
Clients take turns across rounds, building the model collaboratively:

.. code-block:: json

    {
      "min_clients": 1,
      "num_rounds": 10
    }

This is the simplest approach and matches NVFlare's cyclic tree-based strategy.

**Solution 2: Implement a custom aggregator**

For true parallel multi-client XGBoost, you would implement a custom aggregator
that merges tree ensembles rather than averaging weights:

.. code-block:: python

    # Sketch of a custom XGBoost aggregator (extension point)
    from nvflare.app_common.aggregators import Aggregator

    class XGBoostTreeMergeAggregator(Aggregator):
        """Merge XGBoost trees from multiple clients into one ensemble."""

        def accept(self, shareable, fl_ctx):
            # Deserialise each client's booster, collect tree JSONs
            ...

        def aggregate(self, fl_ctx):
            # Concatenate all trees into a single booster
            # This produces a bagged ensemble
            ...

This is conceptually similar to NVFlare's ``XGBBaggingRecipe`` but implemented
as a FLIP-compatible aggregator component.

**Solution 3: Use Strategy B (neural network)**

If multi-site FedAvg is required, use a neural network on the same data.
See Strategy B below.


Strategy B: Federated MLP on HIGGS (FedAvg-Compatible)
-------------------------------------------------------

For scenarios requiring standard FedAvg aggregation across multiple sites,
replace XGBoost with a small multi-layer perceptron (MLP). This achieves
comparable accuracy on the HIGGS dataset while being fully compatible with
FLIP's weight averaging.

The job structure is identical — only ``models.py`` and ``trainer.py`` change.

Step 1: Create models.py — MLP for Binary Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import json
    from pathlib import Path

    import torch
    from torch import nn


    def load_config():
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, "r") as f:
            return json.load(f)


    class HIGGSClassifier(nn.Module):
        """Simple MLP for HIGGS binary classification.

        Architecture mirrors the discriminating power of XGBoost with
        max_depth=8: three hidden layers with batch normalisation and
        dropout for regularisation.

        Input:  28 kinematic features
        Output: 1 logit (binary classification)
        """

        def __init__(self):
            super().__init__()
            config = load_config()
            hidden_dim = config.get("hidden_dim", 256)

            self.net = nn.Sequential(
                nn.Linear(28, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            return self.net(x)


    _model = HIGGSClassifier()


    def get_model() -> nn.Module:
        return _model


Step 2: Create trainer.py — PyTorch Training Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    """
    MLP Trainer for HIGGS binary classification — standard FedAvg compatible.
    """

    from __future__ import annotations

    import json
    import logging
    from pathlib import Path

    import numpy as np
    import torch
    from models import get_model
    from monai.fl.client.client_algo import ClientAlgo
    from monai.fl.utils.constants import WeightType
    from monai.fl.utils.exchange_object import ExchangeObject
    from nvflare.app_common.app_constant import AppConstants

    from flip import FLIP
    from flip.nvflare.metrics import send_metrics_value
    from flip.utils import get_model_weights_diff


    class FLIP_TRAINER(ClientAlgo):
        """HIGGS MLP trainer — standard FedAvg pattern.

        Follows the same lifecycle as the spleen segmentation and X-ray
        classification trainers, showing that FLIP's ClientAlgo interface
        works identically for tabular and imaging tasks.
        """

        def __init__(
            self,
            project_id: str = "",
            query: str = "",
            train_task_name: str = AppConstants.TASK_TRAIN,
        ) -> None:
            self._project_id = project_id
            self._query = query
            self._train_task_name = train_task_name
            self.logger = logging.getLogger(self.__class__.__name__)

        def initialize(self, extra=None):
            working_dir = Path(__file__).parent.resolve()
            with open(str(working_dir / "config.json")) as f:
                config = json.load(f)

            self._epochs = config["LOCAL_ROUNDS"]
            self._lr = config.get("LEARNING_RATE", 0.001)
            self._batch_size = config.get("BATCH_SIZE", 1024)
            self._val_split = config.get("VAL_SPLIT", 0.2)

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = get_model().to(self.device)

            # Load data via FLIP
            self.flip = FLIP()
            df = self.flip.get_dataframe(self._project_id, self._query)

            if "label" in df.columns:
                y = df["label"].values
                X = df.drop("label", axis=1).values
            else:
                y = df.iloc[:, 0].values
                X = df.iloc[:, 1:].values

            split_idx = int(len(X) * (1 - self._val_split))
            self._X_train = torch.FloatTensor(X[:split_idx])
            self._y_train = torch.FloatTensor(y[:split_idx]).unsqueeze(1)
            self._X_val = torch.FloatTensor(X[split_idx:])
            self._y_val = torch.FloatTensor(y[split_idx:]).unsqueeze(1)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

            self._global_weights = None
            self._n_iterations = 0
            self.logger.info(
                f"Loaded {len(self._X_train)} train, {len(self._X_val)} val samples"
            )

        def train(self, data, extra=None):
            fl_ctx = (extra or {}).get("fl_ctx")
            abort_signal = (extra or {}).get("abort_signal")

            torch_weights = {k: torch.as_tensor(v) for k, v in data.weights.items()}
            self._global_weights = {k: v.clone() for k, v in torch_weights.items()}
            self.model.load_state_dict(torch_weights)

            self.model.train()
            self._n_iterations = 0
            n_samples = len(self._X_train)

            for epoch in range(self._epochs):
                # Shuffle data each epoch
                perm = torch.randperm(n_samples)
                epoch_loss = 0.0
                n_batches = 0

                for i in range(0, n_samples, self._batch_size):
                    if abort_signal is not None and abort_signal.triggered:
                        return

                    idx = perm[i:i + self._batch_size]
                    X_batch = self._X_train[idx].to(self.device)
                    y_batch = self._y_train[idx].to(self.device)

                    self.optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = self.loss_fn(output, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1
                    self._n_iterations += 1

                avg_loss = epoch_loss / max(1, n_batches)
                if fl_ctx is not None:
                    send_metrics_value(
                        label="train_loss", value=avg_loss,
                        fl_ctx=fl_ctx, flip=self.flip,
                    )

        def get_weights(self, extra=None):
            weight_type = (extra or {}).get("weight_type", WeightType.WEIGHT_DIFF)
            current_weights = self.model.state_dict()

            if weight_type == WeightType.WEIGHT_DIFF and self._global_weights is not None:
                dxo = get_model_weights_diff(
                    self._global_weights, current_weights, self._n_iterations
                )
                return ExchangeObject(
                    weights=dxo.data,
                    weight_type=WeightType.WEIGHT_DIFF,
                    statistics={"num_steps": self._n_iterations},
                )
            return ExchangeObject(
                weights={k: v.cpu().numpy() for k, v in current_weights.items()},
                weight_type=WeightType.WEIGHTS,
                statistics={"num_steps": self._n_iterations},
            )

        def finalize(self, extra=None):
            pass

**Config for Strategy B** — replace the ``config.json`` with:

.. code-block:: json

    {
      "job_type": "standard",
      "LOCAL_ROUNDS": 5,
      "GLOBAL_ROUNDS": 10,
      "LEARNING_RATE": 0.001,
      "BATCH_SIZE": 1024,
      "VAL_SPLIT": 0.2,
      "hidden_dim": 256
    }


Deployment
----------

The deployment process is identical for both strategies:

.. code-block:: bash

    # Package the job
    zip -r hello-xgboost-flip.zip hello-xgboost-flip/

    # Submit via FLIP-API
    curl -X POST \
      -F "job_folder=@hello-xgboost-flip.zip" \
      http://flip-api:8000/submit_job/hello-xgboost-flip

    # Monitor
    curl http://flip-api:8000/list_jobs?detailed=true

    # Download results
    curl -X GET http://flip-api:8000/download_job/{job_id} -o results.zip


Choosing the Right Strategy
---------------------------

.. table::
   :widths: 20 40 40

   ========================  ====================================  ====================================
   Consideration             Strategy A (Cyclic XGBoost)            Strategy B (Federated MLP)
   ========================  ====================================  ====================================
   Multi-site                Sequential (1 client per round)        Parallel (all clients per round)
   Aggregation               Model passthrough                      FedAvg (weight averaging)
   Model type                XGBoost (decision trees)               Neural network (MLP)
   Feature importance        Built-in (``booster.get_score()``)     Requires separate analysis
   Interpretability          High (tree structure inspectable)       Lower (black-box weights)
   Large tabular data        Excellent (XGBoost strength)           Good (with proper architecture)
   Privacy filters           Not applicable (byte model)            PercentilePrivacy supported
   GPU acceleration          Optional (``tree_method: "gpu_hist"``) Standard PyTorch CUDA
   ========================  ====================================  ====================================

**Recommendation**:

- Use **Strategy A** for single-site or sequential multi-site deployments
  where XGBoost's interpretability and tabular performance are priorities.
- Use **Strategy B** for multi-site parallel deployments where standard FedAvg
  aggregation and privacy filtering are required.


How It Fits Into FLIP's Architecture
-------------------------------------

This tutorial demonstrates two important aspects of FLIP's design:

**1. The ClientAlgo interface is model-agnostic**

Both XGBoost and PyTorch models implement the same four methods:
``initialize()``, ``train()``, ``get_weights()``, ``finalize()``. The
``RUN_MONAI_FL_TRAINER`` executor doesn't care what happens inside these
methods — it only orchestrates the lifecycle and handles weight
serialisation/deserialisation.

.. code-block:: text

    FLIP_TRAINER (any ML framework)
    ├── initialize()     → Set up model, data, optimizer
    ├── train(data)      → data.weights → train → updated model
    ├── get_weights()    → model → ExchangeObject(weights={...})
    └── finalize()       → Cleanup

The "weights" dict in ``ExchangeObject`` is just a ``Dict[str, np.ndarray]``.
Any data that can be represented as named numpy arrays — neural network
parameters, serialised tree bytes, embedding matrices, or gradient statistics
— can flow through this pipeline.

**2. FLIP's aggregation is configurable**

The ``InTimeAccumulateWeightedAggregator`` is the default, but FLIP's config
system allows dropping in any NVFlare-compatible aggregator. This is how
the platform supports diverse ML paradigms without modifying core code.


Verification Checklist
----------------------

- |box| ``data_utils.py``: ``load_higgs_data()`` returns ``(DMatrix, DMatrix)`` from FLIP data
- |box| ``models.py``: ``get_model()`` returns a model (wrapper for XGBoost / MLP) with no arguments
- |box| ``trainer.py``: ``FLIP_TRAINER`` extends ``ClientAlgo`` with ``initialize``, ``train``, ``get_weights``
- |box| ``validator.py``: ``FLIP_VALIDATOR`` extends ``ClientAlgo`` with ``initialize``, ``evaluate``
- |box| ``config.json``: Contains ``LOCAL_ROUNDS``, ``GLOBAL_ROUNDS``, ``VAL_SPLIT``, and either ``xgb_params`` (A) or ``hidden_dim`` (B)
- |box| Server config: ``min_clients: 1`` for Strategy A
- |box| Client config: No ``PercentilePrivacy`` filter for Strategy A (XGBoost)
- |box| Data: HIGGS CSV accessible via FLIP's ``get_dataframe()``

.. |box| unicode:: U+2610


Common Pitfalls
---------------

**Pitfall 1: Applying PercentilePrivacy to XGBoost model bytes**

The privacy filter clips extreme values in weight arrays. For neural networks,
this limits individual weight updates. For serialised XGBoost bytes, it
corrupts the model:

.. code-block:: json

    // Wrong for XGBoost
    "task_result_filters": [{"filters": [{"path": "flip.nvflare.components.PercentilePrivacy"}]}]

    // Correct for XGBoost — no weight-level privacy filter
    "task_result_filters": []

**Pitfall 2: Using WEIGHT_DIFF with XGBoost**

Weight diffs compute ``current_weights - global_weights``. For neural networks,
this is a meaningful gradient-like quantity. For serialised bytes, it's
meaningless:

.. code-block:: python

    # Wrong for XGBoost
    return ExchangeObject(weight_type=WeightType.WEIGHT_DIFF, ...)

    # Correct — always return full model
    return ExchangeObject(weight_type=WeightType.WEIGHTS, ...)

**Pitfall 3: Expecting FedAvg to work with multiple XGBoost clients**

With ``min_clients > 1``, the aggregator averages byte arrays from different
clients, producing a corrupt model:

.. code-block:: json

    // Wrong for parallel XGBoost
    "min_clients": 2

    // Correct for cyclic XGBoost
    "min_clients": 1

**Pitfall 4: Forgetting to convert model bytes to float32**

NVFlare's DXO pipeline expects float arrays. Raw bytes (uint8) must be
converted:

.. code-block:: python

    # Wrong — uint8 may be truncated or mishandled
    model_array = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Correct — lossless conversion to float32
    model_array = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)


Next Steps
----------

After completing this tutorial:

1. **Try the MSD HIGGS full dataset**: Scale from 100K samples to the full
   11M for production-grade results
2. **Implement a custom aggregator**: Build ``XGBoostTreeMergeAggregator`` for
   true parallel multi-client XGBoost
3. **Add clinical tabular data**: Replace HIGGS features with clinical
   variables (lab results, vitals, demographics) for healthcare FL
4. **Explore vertical FL**: Adapt the vertical XGBoost pattern for scenarios
   where different sites hold different features for the same patients


References
----------

- `NVFlare Hello-XGBoost <https://nvflare.readthedocs.io/en/main/hello-world/hello-xgboost/index.html>`_
- `NVFlare XGBoost Examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost>`_
- `XGBoost Documentation <https://xgboost.readthedocs.io/>`_
- `HIGGS Dataset (UCI) <https://archive.ics.uci.edu/ml/datasets/HIGGS>`_
- `FLIP API Reference <../reference/api/flip/index.html>`_
