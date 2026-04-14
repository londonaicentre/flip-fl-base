Adapting NVFlare Hello-PyTorch to FLIP Deployment
===================================================

Overview
--------

This tutorial shows how to take the `NVFlare Hello-PyTorch example <https://nvflare.readthedocs.io/en/main/hello-world/hello-pt/index.html>`_ 
and adapt it to run on the **FLIP platform** via the FLIP-API.

**Key distinction**: The official NVFlare example uses the **Client API** (``flare.init()`` + ``while`` loop) 
and runs locally via ``SimEnv``. FLIP-compatible jobs use the **Executor pattern** (``ClientAlgo`` interface) 
and are submitted via REST to an NVFlare server.

This transition reflects how federated learning moves from **teaching environments** (Recipe API + SimEnv) 
to **production deployments** (Executor configs + centralized server).

Why Two Patterns?
~~~~~~~~~~~~~~~~~

.. table::
   :widths: 25 40 35

   ================================  ================================  ================================
   Aspect                            Client API (NVFlare Tutorial)      Executor (FLIP-compatible)
   ================================  ================================  ================================
   Entry point                       ``client.py`` script              ``trainer.py`` module
   Lifecycle control                 User code (``while`` loop)        NVFlare framework
   Job submission                    Local ``SimEnv``                  REST API → NVFlare server
   Data loading                      Per-round (inefficient)           Once in ``initialize()``
   Isolation/monitoring              None                              Full (container, cgroup, audit)
   Deployment target                 Research/testing                  Hospital/production
   ================================  ================================  ================================


Before You Start
----------------

1. Clone the `NVFlare hello-pt example <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-pt>`_:

   .. code-block:: bash

       git clone https://github.com/NVIDIA/NVFlare.git
       cd NVFlare/examples/hello-world/hello-pt
       git switch v2.7.2  # or your preferred release

2. Ensure you have FLIP installed:

   .. code-block:: bash

       pip install flip-utils

3. Familiarize yourself with the NVFlare source structure:

   - ``client.py`` — uses ``flare.init()``, ``flare.receive()``, ``flare.send()``
   - ``model.py`` — ``SimpleNetwork()`` model definition
   - ``job.py`` — ``FedAvgRecipe`` setup (we'll replace this)
   - ``requirements.txt`` — dependencies


Architectural Changes: 5 Key Transitions
-----------------------------------------

This section maps the five core changes from NVFlare Client API to FLIP Executor pattern.

Change 1: From Script Entry Point to Module Import
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NVFlare (Client API):**

.. code-block:: python

    # client.py is executed as a standalone script
    flare.init()
    while flare.is_running():
        input_model = flare.receive()
        # train...
        flare.send(output_model)

    if __name__ == "__main__":
        main()

**FLIP (Executor):**

.. code-block:: python

    # trainer.py is imported as a module by NVFlare
    from monai.fl.client.client_algo import ClientAlgo

    class FLIP_TRAINER(ClientAlgo):
        def __init__(self, project_id="", query=""):
            # Constructor called by executor
            pass

        def initialize(self, extra=None):
            # Called once when executor starts
            pass

        def train(self, data, extra=None):
            # Called per round by executor
            pass

        def get_weights(self, extra=None):
            # Called after train() to serialize weights
            pass

**Why?** NVFlare executors are class-based for state management and lifecycle control.
The executor framework (server-side) decides *when* to call each method, not your code.


Change 2: From Per-Round Data Loading to Cached Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NVFlare (inefficient for medical data):**

.. code-block:: python

    def main():
        trainloader = load_data(batch_size)  # ← Loaded ONCE at startup

        flare.init()
        while flare.is_running():
            # But in production (FLIP deployment):
            # Round 1: Download CIFAR-10 again? No—pre-loaded
            # BUT if using FLIP API to fetch DICOM: Round 1-10 fetch same patients repeatedly!
            ...

**FLIP (efficient):**

.. code-block:: python

    def initialize(self, extra=None):
        # Load data setup ONCE
        self.flip = FLIP()
        self.dataframe = self.flip.get_dataframe(self._project_id, self._query)
        # Dataframe cached across all training rounds

        self.training_dataloader = DataLoader(
            Dataset(..., transform=self._train_transforms),
            batch_size=self._batch_size,
            shuffle=True  # Shuffle per epoch, not per round
        )
        self._n_iterations = 0

    def train(self, data, extra=None):
        # Just run training on cached loader
        for epoch in range(self._epochs):
            for batch in self.training_dataloader:
                # train...
                self._n_iterations += 1

**Why?** Medical DICOM data is expensive to fetch. Loading once at ``initialize()`` 
and caching is critical for performance and resource efficiency.


Change 3: From Direct Model Exchange to ExchangeObject
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NVFlare (Client API):**

.. code-block:: python

    input_model = flare.receive()  # Returns FLModel
    params = input_model.params     # Dict of model weights

    # ... training ...

    new_params = net.state_dict()

    output_model = flare.FLModel(
        params=new_params,
        meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: num_steps}
    )
    flare.send(output_model)  # Sends FLModel back


**FLIP (Executor + MONAI):**

.. code-block:: python

    from monai.fl.utils.exchange_object import ExchangeObject
    from monai.fl.utils.constants import WeightType

    def train(self, data, extra=None):
        # data is already an ExchangeObject with .weights
        torch_weights = {k: torch.as_tensor(v) for k, v in data.weights.items()}
        self.model.load_state_dict(torch_weights)
        # ... training ...
        self._global_weights = {k: v.clone() for k, v in torch_weights.items()}

    def get_weights(self, extra=None):
        weight_type = (extra or {}).get("weight_type", WeightType.WEIGHTS)
        current_weights = self.model.state_dict()

        if weight_type == WeightType.WEIGHT_DIFF:
            # Return weight DIFFERENCES (efficient for large models)
            dxo = get_model_weights_diff(self._global_weights, current_weights, self._n_iterations)
            return ExchangeObject(
                weights=dxo.data,
                weight_type=WeightType.WEIGHT_DIFF,
                statistics={"num_steps": self._n_iterations}
            )

        # Or return full weights
        return ExchangeObject(
            weights={k: v.cpu().numpy() for k, v in current_weights.items()},
            weight_type=WeightType.WEIGHTS
        )

**Why?** ``ExchangeObject`` is MONAI's abstraction. Splitting ``train()`` and ``get_weights()`` 
allows the framework to handle weight marshaling and differential privacy separately.


Change 4: From Job Recipe to Config Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NVFlare (Recipe API in job.py):**

.. code-block:: python

    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.recipe import SimEnv

    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=n_clients,
        num_rounds=num_rounds,
        model=SimpleNetwork(),
        train_script="client.py",
        train_args=f"--batch_size {batch_size}",
    )
    recipe.execute(env=SimEnv(num_clients=n_clients))

**FLIP (Declarative configs):**

Server config (``config_fed_server.json``):

.. code-block:: text

    {
      "format_version": 2,
      "global_rounds": 3,
      "min_clients": 1,
      "workflows": [
        {
          "id": "scatter_and_gather",
          "path": "flip.nvflare.controllers.ScatterAndGather",
          "args": {
            "min_clients": 1,
            "num_rounds": 3,
            ...
          }
        }
      ]
    }

Client config (``config_fed_client.json``):

.. code-block:: text

    {
      "executors": [
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
      ]
    }

**Why?** Configs are declarative and environment-agnostic. They define *what* to run, 
not *how* to orchestrate it. This enables reproducibility and auditing.


Change 5: From Local SimEnv to REST-based Submission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NVFlare (Local):**

.. code-block:: bash

    python job.py
    # Runs entirely in-process with 2 simulated clients
    # Output: /tmp/nvflare/simulation/hello-pt/...

**FLIP (Centralized Deployment):**

.. code-block:: bash

    # 1. Package your job
    mkdir -p hello-world-flip/custom
    cp trainer.py validator.py model.py config.json hello-world-flip/custom/

    # 2. Submit via FLIP-API
    curl -X POST http://flip-api:8000/submit_job/hello-world-flip

    # 3. Monitor via FLIP-API
    curl http://flip-api:8000/list_jobs

    # 4. Download results
    curl -X GET http://flip-api:8000/download_job/{job_id}

**Why?** FLIP-API is the http boundary. Jobs are submitted as folders, validated, 
and dispatched to NVFlare clients at hospital sites.


Step-by-Step Implementation
----------------------------

Step 1: Create the Module Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by creating the directory structure for your FLIP job:

.. code-block:: bash

    hello-world-flip/
    ├── model.py           # Copy from NVFlare, unchanged
    ├── trainer.py         # Refactored from client.py
    ├── validator.py       # New: validation-only logic
    ├── config.json        # Hyperparameters for trainer/validator
    └── requirements.txt   # Dependencies

**Key difference from NVFlare:**
- No ``client.py`` (we use ``trainer.py`` as a module)
- No ``job.py`` (we use JSON config files)
- New ``validator.py`` (separate validation logic)


Step 2: Copy and Rename model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is almost unchanged. Just ensure it's importable:

.. code-block:: python

    # model.py — from NVFlare, unchanged
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SimpleNetwork(nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def get_model():
        """Factory function for model creation."""
        return SimpleNetwork()

**Why the factory?** NVFlare configs reference models via string paths like ``"models.get_model"``.


Step 3: Create trainer.py (Refactor from client.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the core refactoring. Replace the Client API loop with ``ClientAlgo`` methods.

.. code-block:: python

    """
    Hello World Trainer — MONAI FL ClientAlgo style for FLIP deployment.
    
    Refactored from NVFlare's client.py to work with FLIP-API and executors.
    """

    import argparse
    import logging
    from collections import OrderedDict
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from monai.fl.client.client_algo import ClientAlgo
    from monai.fl.utils.constants import WeightType
    from monai.fl.utils.exchange_object import ExchangeObject

    from model import get_model
    from flip.utils import get_model_weights_diff

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "/tmp/data/cifar10"


    class FLIP_TRAINER(ClientAlgo):
        """
        Hello World federated trainer using MONAI FL ClientAlgo interface.
        
        This class implements the three-phase lifecycle:
          1. initialize(extra) — Set up model, data, optimizer once
          2. train(data, extra) — Run local training for current round
          3. get_weights(extra) — Return trained weights to server
        """

        def __init__(
            self,
            project_id: str = "",
            query: str = "",
            train_task_name: str = "train",
        ) -> None:
            """
            Initialize the trainer.
            
            Args:
                project_id: FLIP project ID (placeholder for consistency with image_classification)
                query: SQL query (placeholder for consistency)
                train_task_name: NVFlare task name (for compatibility)
            """
            super().__init__()
            self._project_id = project_id
            self._query = query
            self._train_task_name = train_task_name
            self.logger = logging.getLogger(self.__class__.__name__)
            
            # State management
            self._model = None
            self._device = None
            self._optimizer = None
            self._trainloader = None
            self._epochs = 1
            self._global_weights = None
            self._n_iterations = 0
            self._batch_size = 32

        def initialize(self, extra=None):
            """
            One-time initialization: load data, create model, set up optimizer.
            
            This method is called once when the executor starts, NOT per round.
            This is where you'd fetch FLIP data in production.
            
            Args:
                extra: Dict with optional fl_ctx, abort_signal, etc.
            """
            self.logger.info("[FLIP_TRAINER] Initializing trainer...")
            
            # Parse config (in production, load from config.json)
            self._epochs = 1
            self._batch_size = 32

            # Device setup
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self._device}")

            # Model setup
            self._model = get_model().to(self._device)

            # Data setup (in production, replace with FLIP.get_dataframe())
            self._trainloader = self._load_data(self._batch_size)
            self.logger.info(f"Loaded training data: {len(self._trainloader)} batches")

            # Optimizer setup
            self._optimizer = optim.SGD(self._model.parameters(), lr=0.001, momentum=0.9)
            self.logger.info("Optimizer initialized")

        def train(self, data, extra=None):
            """
            Run local training for one round.
            
            The executor calls this method per round, passing:
              - data: ExchangeObject with global weights
              - extra: Dict with fl_ctx, abort_signal, current_round
            
            Args:
                data: ExchangeObject carrying model weights
                extra: Runtime context
            """
            fl_ctx = (extra or {}).get("fl_ctx")
            abort_signal = (extra or {}).get("abort_signal")
            current_round = (extra or {}).get("current_round", 0)
            
            self.logger.info(f"[FLIP_TRAINER] Round {current_round} starting")

            # Receive and load global weights
            torch_weights = {k: torch.as_tensor(v) for k, v in data.weights.items()}
            self._global_weights = {k: v.clone() for k, v in torch_weights.items()}
            self._model.load_state_dict(torch_weights)

            # Run local training
            self._local_train(fl_ctx, abort_signal, current_round)

        def get_weights(self, extra=None):
            """
            Serialize trained weights and return to server.
            
            The executor calls this after train() completes.
            Returns weight diff (efficient) or full weights depending on weight_type.
            
            Args:
                extra: Dict potentially containing weight_type preference
                
            Returns:
                ExchangeObject with weights and metadata
            """
            weight_type = (extra or {}).get("weight_type", WeightType.WEIGHTS)
            current_weights = self.model.state_dict()

            if weight_type == WeightType.WEIGHT_DIFF and self._global_weights is not None:
                # Return only the differences (smaller transmission)
                dxo = get_model_weights_diff(self._global_weights, current_weights, self._n_iterations)
                return ExchangeObject(
                    weights=dxo.data,
                    weight_type=WeightType.WEIGHT_DIFF,
                    statistics={"num_steps": self._n_iterations},
                )

            # Return full weights
            self.logger.info(f"[FLIP_TRAINER] Returning {len(current_weights)} weight tensors")
            return ExchangeObject(
                weights={k: v.cpu().numpy() for k, v in current_weights.items()},
                weight_type=WeightType.WEIGHTS,
                statistics={"num_steps": self._n_iterations},
            )

        def finalize(self, extra=None):
            """
            Clean up resources after training completes.
            Optional, but good practice.
            """
            self.logger.info("[FLIP_TRAINER] Finalizing")
            # In production: close database connections, clean up temp files, etc.

        # ── Private methods (training logic) ────────────────────────────────────

        def _load_data(self, batch_size: int):
            """
            Download (first run) and return CIFAR-10 DataLoader.
            
            In production FLIP jobs, replace this with:
                from flip import FLIP
                flip = FLIP()
                df = flip.get_dataframe(project_id, sql_query)
            """
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            trainset = torchvision.datasets.CIFAR10(
                root=DATA_ROOT,
                train=True,
                download=True,
                transform=transform
            )
            return torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )

        def _local_train(self, fl_ctx, abort_signal, global_round):
            """
            Run epochs of training on local data.
            
            This is the training loop from NVFlare client.py, 
            migrated to a private method.
            """
            self._model.train()
            criterion = nn.CrossEntropyLoss()
            
            self._n_iterations = 0
            
            for epoch in range(self._epochs):
                running_loss = 0.0
                
                for inputs, labels in self._trainloader:
                    # Check for abort signal
                    if abort_signal is not None and abort_signal.triggered:
                        self.logger.warning("Training aborted by signal")
                        return

                    # Training step
                    inputs, labels = inputs.to(self._device), labels.to(self._device)
                    self._optimizer.zero_grad()
                    
                    outputs = self._model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    self._optimizer.step()
                    
                    running_loss += loss.item()
                    self._n_iterations += 1

                avg_loss = running_loss / len(self._trainloader)
                self.logger.info(
                    f"[FLIP_TRAINER] Round {global_round}, "
                    f"Epoch {epoch + 1}/{self._epochs}, "
                    f"loss={avg_loss:.4f}"
                )


Step 4: Create validator.py (Validation Logic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validation is separated into its own executor task.

.. code-block:: python

    """
    Hello World Validator — MONAI FL ClientAlgo style for FLIP deployment.
    
    Runs validation/test on a separate dataset and returns metrics.
    """

    import logging
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from monai.fl.client.client_algo import ClientAlgo
    from monai.fl.utils.exchange_object import ExchangeObject

    from model import get_model

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "/tmp/data/cifar10"


    class FLIP_VALIDATOR(ClientAlgo):
        """
        Hello World federated validator using MONAI FL ClientAlgo interface.
        
        Implements evaluate(data, extra) to compute validation metrics.
        """

        def __init__(
            self,
            project_id: str = "",
            query: str = "",
            validate_task_name: str = "validate",
        ) -> None:
            super().__init__()
            self._project_id = project_id
            self._query = query
            self._validate_task_name = validate_task_name
            self.logger = logging.getLogger(self.__class__.__name__)

            self._model = None
            self._device = None
            self._testloader = None
            self._batch_size = 32

        def initialize(self, extra=None):
            """Set up model and test data."""
            self.logger.info("[FLIP_VALIDATOR] Initializing validator...")

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = get_model().to(self._device)
            
            self._batch_size = 32
            self._testloader = self._load_data(self._batch_size)
            
            self.logger.info(f"Loaded test data: {len(self._testloader)} batches")

        def evaluate(self, data, extra=None):
            """
            Run validation and return metrics.
            
            Args:
                data: ExchangeObject with model weights from server
                extra: Runtime context with fl_ctx, abort_signal
                
            Returns:
                ExchangeObject with metrics dict
            """
            fl_ctx = (extra or {}).get("fl_ctx")
            abort_signal = (extra or {}).get("abort_signal")
            
            self.logger.info("[FLIP_VALIDATOR] Starting evaluation")

            # Load weights and run validation
            torch_weights = {k: torch.as_tensor(v) for k, v in data.weights.items()}
            self._model.load_state_dict(torch_weights)

            metrics = self._do_validation(fl_ctx, abort_signal)

            return ExchangeObject(metrics=metrics)

        def finalize(self, extra=None):
            """Clean up."""
            self.logger.info("[FLIP_VALIDATOR] Finalizing")

        # ── Private methods ────────────────────────────────────────────────

        def _load_data(self, batch_size: int):
            """Load CIFAR-10 test set."""
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            testset = torchvision.datasets.CIFAR10(
                root=DATA_ROOT,
                train=False,
                download=True,
                transform=transform
            )
            return torch.utils.data.DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )

        def _do_validation(self, fl_ctx, abort_signal):
            """Compute loss and accuracy on test set."""
            self._model.eval()
            criterion = nn.CrossEntropyLoss()
            
            correct = 0
            total = 0
            total_loss = 0.0
            
            with torch.no_grad():
                for inputs, labels in self._testloader:
                    if abort_signal is not None and abort_signal.triggered:
                        self.logger.warning("Validation aborted")
                        break

                    inputs, labels = inputs.to(self._device), labels.to(self._device)
                    outputs = self._model(inputs)
                    loss = criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            avg_loss = total_loss / len(self._testloader)
            
            self.logger.info(f"[FLIP_VALIDATOR] Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")

            return {
                "accuracy": accuracy,
                "loss": avg_loss,
            }


Step 5: Create config.json
~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperparameters for trainer and validator:

.. code-block:: text

    {
      "job_type": "standard",
      "LOCAL_ROUNDS": 1,
      "BATCH_SIZE": 32,
      "LR": 0.001,
      "MOMENTUM": 0.9,
      "GLOBAL_ROUNDS": 2
    }

**For Hello-World, this is minimal.** In production (image_classification), 
you'd include extensive parameters like ``LESIONS``, ``VALIDATE_EVERY``, etc.


Step 6: Create requirements.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    torch>=2.0.0
    torchvision>=0.15.0
    monai>=1.2.0
    nvflare>=2.7.2
    flip-utils>=0.1.0


Step 7: Create/Copy NVFlare Config Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NVFlare configs go in a parallel folder structure. Create:

.. code-block:: bash

    hello-world-flip/
    ├── custom/
    │   ├── trainer.py
    │   ├── validator.py
    │   ├── model.py
    │   ├── config.json
    │   └── requirements.txt
    └── config/
        ├── config_fed_server.json
        └── config_fed_client.json

**server config** (``config/config_fed_server.json``):

.. code-block:: text

    {
      "format_version": 2,
      "model_id": "hello-world-cifar10",
      "python_path": [".", "custom"],
      "global_rounds": 2,
      "min_clients": 1,
      "server": {
        "heart_beat_timeout": 600
      },
      "workflows": [
        {
          "id": "scatter_and_gather",
          "path": "flip.nvflare.controllers.ScatterAndGather",
          "args": {
            "min_clients": 1,
            "num_rounds": 2,
            "wait_time_after_min_received": 10,
            "aggregator_id": "aggregator",
            "persistor_id": "persistor",
            "shareable_generator_id": "shareable_generator",
            "train_task_name": "train",
            "train_timeout": 3600,
            "submit_model_task_name": "submit_model"
          }
        }
      ],
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
        }
      ]
    }

**client config** (``config/config_fed_client.json``):

.. code-block:: text

    {
      "format_version": 2,
      "project_id": "hello-world-cifar10",
      "query": "SELECT * FROM cifar10;",
      "local_rounds": 1,
      "python_path": [".", "custom"],
      "executors": [
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
      ]
    }

**Key lines:**

- ``"path": "flip.nvflare.executors.RUN_MONAI_FL_TRAINER"`` — Executor that imports training module
- ``"python_path": [".", "custom"]`` — NVFlare will look in ``custom/`` for your trainer.py
- ``{project_id}`` and ``{query}`` — Replaced at job submission time


Deployment & Verification
--------------------------

Local Testing (Before Deployment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before submitting to FLIP-API, test locally in a simulation:

.. code-block:: bash

    # Create a test script
    cat > test_job.py << 'EOF'
    from nvflare.recipe import SimEnv
    from flip.app_organiser import prepare_job_folder

    # Prepare the job folder
    job_folder = prepare_job_folder(
        job_type="standard",
        path_to_app="./hello-world-flip/",
    )

    # Run in simulation
    env = SimEnv(num_clients=1, num_threads=1)
    env.run_job(job_folder)
    EOF

    python test_job.py

Expected output:

.. code-block:: text

    [NVFlare] Starting training...
    [FLIP_TRAINER] Round 0 starting
    [FLIP_TRAINER] Round 0, Epoch 1/1, loss=2.31
    [FLIP_TRAINER] Returning weights...
    [FLIP_VALIDATOR] Starting evaluation
    [FLIP_VALIDATOR] Accuracy: 15.32%, Loss: 2.28
    ...

Deployment via FLIP-API
~~~~~~~~~~~~~~~~~~~~~~~

Once verified, submit to FLIP-API:

.. code-block:: bash

    # 1. Package the job
    zip -r hello-world-flip.zip hello-world-flip/

    # 2. Submit via FLIP-API
    curl -X POST \
      -F "job_folder=@hello-world-flip.zip" \
      http://localhost:8000/submit_job/hello-world-cifar10

    # Response: {"job_id": "abc123"}

    # 3. Monitor progress
    curl http://localhost:8000/list_jobs?detailed=true

    # 4. Download results
    curl http://localhost:8000/download_job/abc123 \
      -o results-abc123.zip


Verification Checklist
~~~~~~~~~~~~~~~~~~~~~~

Before you implement, verify your understanding by checking these:

□ **Module structure**: Can you import ``trainer.py`` as ``from trainer import FLIP_TRAINER``?

□ **Executor interface**: Does ``FLIP_TRAINER`` inherit from ``ClientAlgo`` and implement:

  - ``__init__()`` 
  - ``initialize(extra)``
  - ``train(data, extra)``
  - ``get_weights(extra)``
  - ``finalize(extra)``

□ **Data loading**: Can you load CIFAR-10 *once* in ``initialize()`` and cache it?

□ **Weight handling**: Does ``train()`` accept ``ExchangeObject`` and return it from ``get_weights()``?

□ **Config files**: Do both ``config_fed_server.json`` and ``config_fed_client.json`` exist?

□ **Python path**: Does ``"python_path": [".", "custom"]`` point to where your trainer.py lives?

□ **Executor paths**: Do configs reference ``flip.nvflare.executors.RUN_MONAI_FL_TRAINER``?

□ **Local test**: Does simulation run without errors on your machine?

□ **REST submission**: Can you submit and list jobs via FLIP-API?


Common Pitfalls
---------------

**Pitfall 1: Keeping Client API in trainer.py**

❌ Wrong:

.. code-block:: python

    # trainer.py — DO NOT DO THIS
    import nvflare.client as flare
    
    def train():
        flare.init()
        while flare.is_running():
            ...

✓ Correct:

.. code-block:: python

    # trainer.py
    from monai.fl.client.client_algo import ClientAlgo
    
    class FLIP_TRAINER(ClientAlgo):
        def train(self, data, extra):
            ...


**Pitfall 2: Loading data per round**

❌ Wrong:

.. code-block:: python

    def train(self, data, extra):
        trainloader = load_data()  # ← Every round!
        for batch in trainloader:
            ...

✓ Correct:

.. code-block:: python

    def initialize(self, extra):
        self.trainloader = load_data()  # ← Once

    def train(self, data, extra):
        for batch in self.trainloader:
            ...


**Pitfall 3: Returning FLModel instead of ExchangeObject**

❌ Wrong:

.. code-block:: python

    return flare.FLModel(params=weights)

✓ Correct:

.. code-block:: python

    return ExchangeObject(weights=weights, ...)


**Pitfall 4: Missing validate task in config**

❌ Wrong: Only trainer executor, no validator

.. code-block:: text

    "executors": [
      {"tasks": ["train"], "executor": {...}}
    ]

✓ Correct: Both trainer and validator

.. code-block:: text

    "executors": [
      {"tasks": ["train"], "executor": {...}},
      {"tasks": ["validate"], "executor": {...}}
    ]


Next Steps
----------

Once you've implemented and verified this tutorial:

1. **Add FLIP data integration**: Replace ``load_data()`` with ``FLIP().get_dataframe()``
2. **Add config persistence**: Load hyperparameters from ``config.json``
3. **Add metrics reporting**: Use ``flip.nvflare.metrics.send_metrics_value()``
4. **Add differential privacy**: Insert privacy filters in config
5. **Extend to 3D**: Adapt for volumetric (NIFTI) data for segmentation tasks

Refer to [tutorials/image_classification/](./image_classification/README.md) for a full production example.


References
----------

- `NVFlare Hello-PyTorch <https://nvflare.readthedocs.io/en/main/hello-world/hello-pt/index.html>`_
- `MONAI FL ClientAlgo <https://docs.monai.io/en/latest/fl.html>`_
- `FLIP API Reference <../reference/api/flip/index.html>`_
- `NVFlare Job Config <https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html>`_
