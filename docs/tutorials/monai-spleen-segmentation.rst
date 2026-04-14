Adapting NVFlare MONAI Spleen CT Segmentation to FLIP
=====================================================

Overview
--------

This tutorial shows how to take the `NVFlare MONAI Spleen CT Segmentation example
<https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/monai/spleen_ct_segmentation>`_
and adapt it to run on the **FLIP platform**.

**What you will build**: A federated 3D spleen segmentation job that trains a
volumetric UNet across multiple hospital sites using NIfTI CT scans, coordinated
by FLIP's server-side aggregation and deployed via the FLIP-API.

**Clinical context**: Spleen segmentation from abdominal CT scans is a common
medical imaging task used in surgical planning, trauma assessment, and radiation
therapy. Federated learning enables hospitals to collaboratively train a
segmentation model without sharing patient data — each site trains on its own
CT scans and only shares model weight updates.

**What changes from the NVFlare example**:

.. table::
   :widths: 25 40 35

   ================================  ====================================  ====================================
   Aspect                            NVFlare Example                        FLIP Deployment
   ================================  ====================================  ====================================
   Entry point                       ``client.py`` script with             ``trainer.py`` module with
                                     ``flare.init()`` loop                 ``FLIP_TRAINER(ClientAlgo)``
   Data loading                      Local ``MonaiAlgo`` bundle            ``FLIPDataset`` + FLIP API
   Model definition                  ``FLUNet`` in ``model.py``            ``SegmentationNetwork`` wrapping
                                                                           MONAI ``UNet``
   Job configuration                 ``FedAvgRecipe`` Python API           Declarative JSON config files
   Execution                         ``SimEnv`` (local simulation)         REST submission to NVFlare server
   Data access                       Direct filesystem (MSD download)      ``flip.get_by_accession_number()``
   ================================  ====================================  ====================================


Before You Start
----------------

1. Familiarise yourself with the NVFlare source:

   .. code-block:: bash

       git clone https://github.com/NVIDIA/NVFlare.git
       cd NVFlare/examples/advanced/monai/spleen_ct_segmentation

   Key files to review:

   - ``job_fedavg/client.py`` — ``MonaiAlgo`` + ``flare.receive()`` / ``flare.send()`` loop
   - ``job_fedavg/model.py`` — ``FLUNet`` (MONAI UNet wrapper)
   - ``job_fedavg/job.py`` — ``FedAvgRecipe`` + ``SimEnv`` launcher

2. Ensure FLIP is installed:

   .. code-block:: bash

       pip install flip-utils

3. Install the medical imaging dependencies:

   .. code-block:: bash

       pip install monai>=1.5.1 nibabel torch torchvision


Understanding the Architecture
------------------------------

The NVFlare example uses the **Client API** pattern: a standalone ``client.py``
script calls ``flare.init()``, enters a ``while flare.is_running()`` loop, and
explicitly receives and sends models. The ``MonaiAlgo`` class wraps a MONAI
Bundle to handle training.

FLIP uses the **Executor pattern**: your code lives in a ``FLIP_TRAINER`` class
that extends MONAI's ``ClientAlgo`` interface. The NVFlare framework (not your
code) controls *when* each method is called:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────┐
    │  NVFlare Server (FLIP ScatterAndGather controller)      │
    │                                                         │
    │  1. Load global model via persistor                     │
    │  2. Scatter "train" task to all clients                 │
    │  3. Wait for clients to return weight diffs             │
    │  4. Aggregate via FedAvg (weighted average)             │
    │  5. Repeat for num_rounds                               │
    └───────────────────────┬─────────────────────────────────┘
                            │
    ┌───────────────────────▼─────────────────────────────────┐
    │  NVFlare Client (RUN_MONAI_FL_TRAINER executor)         │
    │                                                         │
    │  First task arrival:                                    │
    │    → import trainer; FLIP_TRAINER()                     │
    │    → FLIP_TRAINER.initialize(extra)     [once]          │
    │                                                         │
    │  Each "train" task:                                     │
    │    → Shareable → ExchangeObject (adapter converts)      │
    │    → FLIP_TRAINER.train(data, extra)    [per round]     │
    │    → FLIP_TRAINER.get_weights(extra)    [per round]     │
    │    → ExchangeObject → Shareable (adapter converts back) │
    │                                                         │
    │  Each "validate" task:                                  │
    │    → FLIP_VALIDATOR.evaluate(data, extra)               │
    └─────────────────────────────────────────────────────────┘

This is the same lifecycle used for image classification (X-ray) and the
Hello-World tutorial — only the model, transforms, and dataset change.


Dependencies
------------

Create ``requirements.txt``:

.. code-block:: text

    torch>=2.0.0
    monai>=1.5.1
    nibabel>=5.3.2
    nvflare>=2.7.1
    flip-utils>=0.1.0
    numpy
    pandas

These cover:

- **torch** — deep learning backend
- **monai** — medical imaging transforms, UNet, losses, metrics, inferers
- **nibabel** — NIfTI file I/O (reading 3D CT volumes)
- **nvflare** — federated learning framework
- **flip-utils** — FLIP platform integration (data access, metrics)


Step-by-Step Implementation
---------------------------

Your final FLIP job will have this structure:

.. code-block:: text

    spleen-segmentation/
    ├── custom/
    │   ├── trainer.py           # FLIP_TRAINER(ClientAlgo) — training lifecycle
    │   ├── validator.py         # FLIP_VALIDATOR(ClientAlgo) — validation lifecycle
    │   ├── models.py            # SegmentationNetwork (3D UNet) + get_model()
    │   ├── transforms.py        # MONAI transform pipelines
    │   ├── flip_datasets.py     # FLIPDataset subclasses for spleen data
    │   ├── config.json          # Hyperparameters
    │   └── requirements.txt     # Dependencies
    └── config/
        ├── config_fed_server.json
        └── config_fed_client.json


Step 1: Create models.py — The 3D UNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NVFlare example defines ``FLUNet``, a thin wrapper around MONAI's ``UNet``.
For FLIP, we use the same architecture but follow FLIP's model pattern: a module-level
instance with a ``get_model()`` factory function.

**NVFlare (model.py):**

.. code-block:: python

    # NVFlare creates the model inside job.py and passes it to FedAvgRecipe
    from monai.networks.nets import UNet

    class FLUNet(UNet):
        def __init__(self, ...):
            super().__init__(spatial_dims=3, in_channels=1, out_channels=2,
                             channels=[16, 32, 64, 128, 256], strides=[2, 2, 2, 2],
                             num_res_units=2, norm="batch")

**FLIP (models.py):**

.. code-block:: python

    import json
    from pathlib import Path

    import torch
    from monai.networks.nets.unet import UNet
    from torch import nn


    def load_net_config():
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        net_config = config.get("net_config", {})
        return net_config


    class SegmentationNetwork(nn.Module):
        """3D UNet for spleen segmentation.

        Wraps MONAI UNet and returns raw logits (required for sliding-window
        inference during validation and DiceCELoss during training).
        """

        def __init__(self, num_classes: int = 1):
            super().__init__()
            net_config = load_net_config()

            self.net = UNet(
                spatial_dims=net_config["spatial_dims"],
                in_channels=1,
                out_channels=num_classes + 1,  # background + spleen
                num_res_units=2,
                norm="batch",
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
            )

        def forward(self, x: torch.Tensor):
            return self.net(x)


    _net = SegmentationNetwork()


    def get_model() -> nn.Module:
        """Factory function — no input arguments allowed.

        NVFlare's PTFileModelPersistor calls this via the string path
        ``"models.get_model"`` defined in config_fed_server.json.
        """
        return _net

**Key differences:**

- The model is instantiated at **module load time** (``_net = SegmentationNetwork()``).
  This is required because NVFlare's persistor calls ``get_model()`` to obtain the
  initial model architecture, and it must return the same instance each time.
- Configuration (e.g. ``spatial_dims``) is loaded from ``config.json`` rather than
  hard-coded, making the model portable across different imaging modalities.
- The ``get_model()`` factory takes **no arguments** — this is a FLIP convention
  enforced by the NVFlare config system.


Step 2: Create transforms.py — MONAI Preprocessing Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3D medical image segmentation requires careful preprocessing. The NVFlare example
uses MONAI Bundle configs (``configs/train.json``) to declare transforms. In FLIP,
we define them in Python for explicit control.

.. code-block:: python

    import numpy as np
    import torch
    from monai.inferers.inferer import SlidingWindowInferer
    from monai.transforms.compose import Compose
    from monai.transforms.croppad.dictionary import CropForegroundd, RandCropByPosNegLabeld
    from monai.transforms.intensity.dictionary import ScaleIntensityRanged
    from monai.transforms.io.dictionary import LoadImaged
    from monai.transforms.spatial.dictionary import Orientationd, RandAffined, Spacingd
    from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped


    def get_train_transforms():
        """Training transforms with augmentation.

        Pipeline steps:
        1. LoadImaged         — Read NIfTI files into numpy arrays
        2. EnsureChannelFirstd — Add channel dim: (D,H,W) → (1,D,H,W)
        3. Orientationd        — Reorient to RAS (Right-Anterior-Superior)
        4. ScaleIntensityRanged — Clip CT Hounsfield units [-57, 250] → [0, 1]
        5. CropForegroundd     — Remove empty background slices
        6. Spacingd            — Resample to uniform 1.5×1.5×2.0 mm voxels
        7. RandCropByPosNegLabeld — Extract random 96³ patches (balanced pos/neg)
        8. EnsureTyped         — Convert to PyTorch tensors
        9. RandAffined         — Random rotation/scaling augmentation
        """
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            EnsureTyped(keys=["image", "label"]),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1),
            ),
        ])


    def get_val_transforms():
        """Validation transforms — same preprocessing, no augmentation."""
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                allow_smaller=True,
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"]),
        ])


    def get_sliding_window_inferer(sw_device: torch.device):
        """Sliding-window inferer for volumetric validation.

        Full CT volumes are too large for a single forward pass.
        SlidingWindowInferer tiles the volume into overlapping 96³ patches,
        runs inference on each, and stitches the results.
        """
        return SlidingWindowInferer(
            sw_batch_size=1,
            roi_size=(96, 96, 96),
            sw_device=sw_device,
            progress=False,
        )

**Why these specific values?**

- ``a_min=-57, a_max=250``: Hounsfield unit range covering soft tissue and the
  spleen. Values outside this range are clipped.
- ``pixdim=(1.5, 1.5, 2.0)``: Uniform voxel spacing for consistent model input
  regardless of scanner resolution.
- ``spatial_size=(96, 96, 96)``: Patch size balancing GPU memory and spatial context.
  The NVFlare example uses the same values via the MONAI Bundle config.


Step 3: Create flip_datasets.py — FLIP Data Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the most significant change from the NVFlare example. Instead of downloading
the Medical Segmentation Decathlon (MSD) dataset to a local directory, FLIP
resolves imaging data through its API:

.. code-block:: text

    NVFlare:  download_spleen_dataset.py → local /data/Task09_Spleen/
    FLIP:     flip.get_dataframe(project_id, query) → DataFrame of accession IDs
              flip.get_by_accession_number(project_id, acc_id) → Path to NIfTI folder

FLIP provides the ``FLIPDataset`` abstract base class (in ``flip.utils``) that
handles the ``query`` → ``dataframe`` → ``datalist`` lifecycle. You only implement
``_build_datalist(dataframe)`` to define how your domain maps accession IDs
to sample dicts.

.. code-block:: python

    from __future__ import annotations

    from pathlib import Path

    import nibabel as nib

    from flip.constants import ResourceType
    from flip.utils import FLIPDataset


    def _build_spleen_datalist(flip, dataframe, project_id: str) -> list[dict]:
        """Build QC-validated list of {image, label} dicts from FLIP data.

        For each accession ID in the dataframe:
          1. Resolve the NIfTI folder via FLIP API
          2. Match input_*.nii.gz images to label_*.nii.gz segmentations
          3. Validate 3D shape consistency
          4. Skip invalid pairs with a warning
        """
        datalist = []
        for accession_id in dataframe["accession_id"]:
            try:
                folder = flip.get_by_accession_number(
                    project_id,
                    accession_id,
                    resource_type=[ResourceType.NIFTI],
                )
            except Exception as err:
                print(f"Could not get data for accession {accession_id}: {err}")
                continue

            matched = 0
            for img in folder.rglob("input_*.nii.gz"):
                seg = str(img).replace("/input_", "/label_")
                if not Path(seg).exists():
                    print(f"No matching segmentation for {img}.")
                    continue

                try:
                    ih = nib.load(str(img))
                    sh = nib.load(seg)
                except nib.filebasedimages.ImageFileError as err:
                    print(f"Could not load image/label pair ({img}): {err}")
                    continue

                if len(ih.shape) != 3:
                    print(f"Image {img} is not 3-D (has {len(ih.shape)} dims).")
                    continue

                if any(a != b for a, b in zip(ih.shape, sh.shape)):
                    print(f"Shape mismatch for {img}: image {ih.shape} vs seg {sh.shape}.")
                    continue

                datalist.append({"image": str(img), "label": seg})
                matched += 1

            print(f"Added {matched} image-label pairs for {accession_id}.")

        print(f"Built {len(datalist)} total spleen image-label pairs.")
        return datalist


    class SpleenTrainFLIPDataset(FLIPDataset):
        """Training split — first (1 - val_split) fraction of matched pairs."""

        def __init__(self, flip, project_id, query, val_split=0.1, transform=None):
            self._val_split = val_split
            super().__init__(flip=flip, project_id=project_id, query=query, transform=transform)

        def _build_datalist(self, dataframe):
            all_data = _build_spleen_datalist(self.flip, dataframe, self.project_id)
            if not all_data:
                return all_data
            split_idx = int((1 - self._val_split) * len(all_data))
            train_data = all_data[:split_idx]
            print(f"SpleenTrainFLIPDataset: {len(train_data)} training samples.")
            return train_data


    class SpleenValFLIPDataset(FLIPDataset):
        """Validation split — last val_split fraction of matched pairs."""

        def __init__(self, flip, project_id, query, val_split=0.1, transform=None):
            self._val_split = val_split
            super().__init__(flip=flip, project_id=project_id, query=query, transform=transform)

        def _build_datalist(self, dataframe):
            all_data = _build_spleen_datalist(self.flip, dataframe, self.project_id)
            if not all_data:
                return all_data
            split_idx = int((1 - self._val_split) * len(all_data))
            val_data = all_data[split_idx:]
            print(f"SpleenValFLIPDataset: {len(val_data)} validation samples.")
            return val_data

**How FLIPDataset works internally:**

``FLIPDataset`` inherits from ``monai.data.Dataset``. Its ``__init__`` method:

1. Calls ``flip.get_dataframe(project_id, query)`` to get the cohort DataFrame
2. Calls your ``_build_datalist(dataframe)`` to build sample dicts
3. Passes the resulting list to ``monai.data.Dataset.__init__``

This means data resolution happens **once** during construction. The resulting
dataset object is a standard MONAI Dataset that can be passed directly to
``DataLoader``.

**QC logic**: The ``_build_spleen_datalist`` function validates each NIfTI pair
before including it. In a hospital environment, data quality varies — some scans
may be corrupt, have mismatched dimensions, or lack segmentation masks. Robust
QC at data loading time prevents cryptic training errors later.

**Naming convention**: Input images must be named ``input_*.nii.gz`` and
segmentation masks ``label_*.nii.gz`` in the same directory. This convention
is enforced by the FLIP data ingestion pipeline.


Step 4: Create trainer.py — The Training Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the core refactoring from NVFlare's Client API to FLIP's Executor pattern.

**NVFlare (client.py) — imperative loop:**

.. code-block:: python

    import nvflare.client as flare
    from monai.fl.client import MonaiAlgo

    flare.init()
    algo = MonaiAlgo(bundle_root=args.bundle_root, ...)
    algo.initialize(extra={"CLIENT_NAME": flare.get_site_name()})

    while flare.is_running():
        input_model = flare.receive()
        global_weights = ExchangeObject(weights=input_model.params)
        algo.evaluate(data=global_weights)
        algo.train(data=global_weights)
        updated_weights = algo.get_weights()
        output_model = flare.FLModel(params=updated_weights.weights, ...)
        flare.send(output_model)

**FLIP (trainer.py) — framework-driven lifecycle:**

.. code-block:: python

    from __future__ import annotations

    import json
    import logging
    from pathlib import Path

    import numpy as np
    import torch
    from flip_datasets import SpleenTrainFLIPDataset
    from models import get_model
    from monai.data import DataLoader
    from monai.fl.client.client_algo import ClientAlgo
    from monai.fl.utils.constants import WeightType
    from monai.fl.utils.exchange_object import ExchangeObject
    from monai.inferers import SimpleInferer
    from monai.losses import DiceCELoss
    from nvflare.app_common.app_constant import AppConstants
    from transforms import get_train_transforms

    from flip import FLIP
    from flip.nvflare.metrics import send_metrics_value
    from flip.utils import get_model_weights_diff


    class FLIP_TRAINER(ClientAlgo):
        """3D Spleen Segmentation trainer using the MONAI FL ClientAlgo interface.

        Lifecycle:
          1. initialize(extra) — build model, optimizer, loss, and FLIP dataset (once)
          2. train(data, extra) — load global weights, train LOCAL_ROUNDS epochs (per round)
          3. get_weights(extra) — return weight diff to server (per round)
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
            """One-time setup: model, optimizer, loss, data.

            This is called once by the executor when the first task arrives.
            All expensive operations (FLIP API calls, dataset construction)
            happen here and are cached for the lifetime of the job.
            """
            working_dir = Path(__file__).parent.resolve()
            with open(str(working_dir / "config.json")) as f:
                config = json.load(f)

            self._local_rounds = config["LOCAL_ROUNDS"]
            self._learning_rate = config["LEARNING_RATE"]
            self._val_split = config.get("VAL_SPLIT", 0.1)

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = get_model()
            self.model.to(self.device)

            self.loss_fn = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                squared_pred=False,
                batch=True,
                lambda_ce=0.2,
                lambda_dice=0.8,
            )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self._learning_rate
            )
            self.inferer = SimpleInferer()

            # FLIP data integration — replaces NVFlare's local MSD dataset download
            self.flip = FLIP()
            self._train_dataset = SpleenTrainFLIPDataset(
                flip=self.flip,
                project_id=self._project_id,
                query=self._query,
                val_split=self._val_split,
                transform=get_train_transforms(),
            )
            self._n_iterations = 0
            self._global_weights = None

        def train(self, data, extra=None):
            """Load global weights and run LOCAL_ROUNDS training epochs.

            Args:
                data: ExchangeObject carrying global model weights from the server.
                extra: Dict with fl_ctx, abort_signal, current_round.
            """
            fl_ctx = (extra or {}).get("fl_ctx")
            abort_signal = (extra or {}).get("abort_signal")

            # Store global weights for weight-diff computation in get_weights()
            self._global_weights = {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in data.weights.items()
            }
            self.model.load_state_dict({
                k: torch.as_tensor(v, device=self.device)
                for k, v in data.weights.items()
            })

            # Fresh DataLoader per round so shuffle state resets
            train_loader = DataLoader(
                self._train_dataset, batch_size=3, shuffle=True, num_workers=1
            )
            self._n_iterations = 0

            self.model.train()
            for epoch in range(self._local_rounds):
                epoch_loss = 0.0
                for batch in train_loader:
                    if abort_signal is not None and abort_signal.triggered:
                        return
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.inferer(inputs=images, network=self.model)
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    self._n_iterations += 1

                avg_loss = epoch_loss / max(1, len(train_loader))
                if fl_ctx is not None:
                    send_metrics_value(
                        label="train_loss", value=avg_loss,
                        fl_ctx=fl_ctx, flip=self.flip,
                    )
                self.logger.info(
                    f"Epoch {epoch + 1}/{self._local_rounds} loss: {avg_loss:.4f}"
                )

        def get_weights(self, extra=None):
            """Return weight diff (default) or full weights (for submit_model).

            Weight diffs are more efficient for large 3D models — only the
            change from the global model is transmitted, reducing network I/O.
            """
            weight_type = (extra or {}).get("weight_type", WeightType.WEIGHT_DIFF)

            if weight_type == WeightType.WEIGHT_DIFF and self._global_weights is not None:
                dxo = get_model_weights_diff(
                    self._global_weights, self.model.state_dict(), self._n_iterations
                )
                return ExchangeObject(
                    weights=dxo.data,
                    weight_type=WeightType.WEIGHT_DIFF,
                    statistics={"num_steps": self._n_iterations},
                )
            else:
                return ExchangeObject(
                    weights={k: v.cpu().numpy() for k, v in self.model.state_dict().items()},
                    weight_type=WeightType.WEIGHTS,
                    statistics={"num_steps": self._n_iterations},
                )

        def finalize(self, extra=None):
            pass


Step 5: Create validator.py — Sliding-Window Dice Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The validator runs after each training round to measure segmentation quality
using the Dice coefficient. For 3D volumes, we use **sliding-window inference**
since full volumes don't fit in GPU memory at once.

.. code-block:: python

    from __future__ import annotations

    import json
    import logging
    from pathlib import Path

    import numpy as np
    import torch
    from flip_datasets import SpleenValFLIPDataset
    from models import get_model
    from monai.data import DataLoader, decollate_batch
    from monai.fl.client.client_algo import ClientAlgo
    from monai.fl.utils.exchange_object import ExchangeObject
    from monai.metrics import DiceMetric
    from monai.transforms import AsDiscrete
    from nvflare.app_common.app_constant import AppConstants
    from transforms import get_sliding_window_inferer, get_val_transforms

    from flip import FLIP
    from flip.nvflare.metrics import send_metrics_value


    class FLIP_VALIDATOR(ClientAlgo):
        """3D Spleen Segmentation validator — sliding-window Dice evaluation.

        Lifecycle:
          1. initialize(extra) — build model, transforms, validation dataset
          2. evaluate(data, extra) — load weights, compute Dice, return metrics
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
            """Set up model, transforms, and FLIP validation dataset."""
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = get_model()
            self.model.to(self.device)

            with open(str(Path(__file__).parent.resolve() / "config.json")) as f:
                config = json.load(f)
            val_split = config.get("VAL_SPLIT", 0.1)

            self.flip = FLIP()
            self._val_dataset = SpleenValFLIPDataset(
                flip=self.flip,
                project_id=self._project_id,
                query=self._query,
                val_split=val_split,
                transform=get_val_transforms(),
            )

            self.inferer = get_sliding_window_inferer(self.device)
            self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
            self.post_pred_gt = AsDiscrete(to_onehot=2)
            self.dice_acc = DiceMetric(include_background=False, reduction="mean")

        def evaluate(self, data, extra=None):
            """Run sliding-window Dice validation.

            Args:
                data: ExchangeObject with global model weights.
                extra: Dict with fl_ctx and abort_signal.

            Returns:
                ExchangeObject with metrics={"val_acc": mean_dice}.
            """
            fl_ctx = (extra or {}).get("fl_ctx")
            abort_signal = (extra or {}).get("abort_signal")

            weights = {k: torch.as_tensor(v) for k, v in data.weights.items()}
            val_accuracy = self._do_validation(weights, fl_ctx, abort_signal)

            return ExchangeObject(metrics={"val_acc": val_accuracy})

        def _do_validation(self, weights, fl_ctx, abort_signal):
            self.model.load_state_dict(weights)
            self.model.eval()

            test_loader = DataLoader(self._val_dataset, batch_size=1, shuffle=False)
            val_dice = []

            with torch.no_grad():
                for batch in test_loader:
                    if abort_signal is not None and abort_signal.triggered:
                        return 0.0

                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)

                    predictions = self.inferer(images, self.model)

                    self.dice_acc.reset()
                    predictions = decollate_batch(predictions)
                    predictions = torch.stack(
                        [self.post_pred(p) for p in predictions], 0
                    )
                    labels = torch.stack(
                        [self.post_pred_gt(lb) for lb in labels], 0
                    )
                    self.dice_acc(y_pred=predictions, y=labels)
                    val_dice.append(self.dice_acc.aggregate().item())

            metric = float(np.mean(val_dice)) if val_dice else 0.0
            self.logger.info(f"Validation Dice: {metric:.4f}")
            if fl_ctx is not None:
                send_metrics_value(
                    label="TEST_DICE", value=metric,
                    fl_ctx=fl_ctx, flip=self.flip,
                )
            return metric


Step 6: Create config.json
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
      "job_type": "standard",
      "LOCAL_ROUNDS": 4,
      "GLOBAL_ROUNDS": 2,
      "LEARNING_RATE": 5e-5,
      "VAL_SPLIT": 0.1,
      "net_config": {
        "spatial_dims": 3
      }
    }

- ``LOCAL_ROUNDS: 4`` — each client trains 4 epochs per federated round (NVFlare
  example defaults to 1; more local epochs improve convergence with fewer
  communication rounds)
- ``GLOBAL_ROUNDS: 2`` — 2 rounds of federated aggregation (increase for
  production)
- ``LEARNING_RATE: 5e-5`` — conservative for 3D medical segmentation; larger
  values cause oscillation with FedAvg
- ``VAL_SPLIT: 0.1`` — reserve 10% of data for validation
- ``net_config.spatial_dims: 3`` — volumetric 3D convolutions (vs. 2D for X-rays)


Step 7: Create NVFlare Config Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These declarative JSON files replace NVFlare's ``FedAvgRecipe`` Python API.

**Server config** (``config/config_fed_server.json``):

.. code-block:: json

    {
      "format_version": 2,
      "model_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "python_path": [".", "custom"],
      "global_rounds": 2,
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
      "query": "SELECT * FROM spleen_scans;",
      "local_rounds": 4,
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
      "task_result_filters": [
        {
          "tasks": ["train"],
          "filters": [
            {
              "id": "percentile_privacy",
              "path": "flip.nvflare.components.PercentilePrivacy",
              "args": {
                "gamma": 2.0,
                "percentile": 95,
                "off": false
              }
            }
          ]
        }
      ],
      "task_data_filters": [],
      "components": [
        {
          "id": "flip_client_event_handler",
          "path": "flip.nvflare.components.ClientEventHandler",
          "args": {}
        }
      ]
    }

**Understanding the config** (what each section does):

*Server-side:*

- ``workflows`` run sequentially: ``InitTraining`` (set status) →
  ``ScatterAndGather`` (multi-round FedAvg) → ``CrossSiteModelEval``
  (final validation)
- ``aggregator`` uses ``InTimeAccumulateWeightedAggregator`` — weights from each
  client are averaged proportional to ``num_steps`` (the number of training
  iterations that client performed)
- ``persistor`` uses ``PTFileModelPersistor`` — saves/loads PyTorch state dicts
- ``model_locator`` and ``json_generator`` support cross-site validation

*Client-side:*

- ``executors`` map NVFLARE tasks to FLIP executors:
  ``train`` → ``RUN_MONAI_FL_TRAINER`` → imports ``FLIP_TRAINER`` from ``trainer.py``
  ``validate`` → ``RUN_MONAI_FL_VALIDATOR`` → imports ``FLIP_VALIDATOR`` from ``validator.py``
- ``task_result_filters`` apply ``PercentilePrivacy`` — clips extreme weight updates
  (percentile 95, gamma 2.0) to prevent model memorisation of individual patients
- ``{project_id}`` and ``{query}`` are replaced at job submission time


Data Preparation
----------------

For local development with FLIP, you need NIfTI data organised by accession ID:

.. code-block:: text

    data/
    └── accession-resources/
        ├── spleen_001/
        │   ├── input_spleen_001.nii.gz    # CT volume
        │   └── label_spleen_001.nii.gz    # Segmentation mask
        ├── spleen_002/
        │   ├── input_spleen_002.nii.gz
        │   └── label_spleen_002.nii.gz
        └── ...

And a CSV file mapping accession IDs:

.. code-block:: text

    accession_id
    spleen_001
    spleen_002
    ...

**Using the MSD Spleen Dataset for testing**: You can download the Medical
Segmentation Decathlon Task09_Spleen dataset and reorganise it into the
FLIP-expected format:

.. code-block:: bash

    # Download the MSD spleen dataset
    python -c "
    from monai.apps import download_and_extract
    download_and_extract(
        url='https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar',
        filepath='/tmp/Task09_Spleen.tar',
        output_dir='/tmp/msd_data'
    )
    "

    # Reorganise into FLIP format
    mkdir -p data/accession-resources
    for f in /tmp/msd_data/Task09_Spleen/imagesTr/spleen_*.nii.gz; do
        base=$(basename "$f" .nii.gz)
        mkdir -p "data/accession-resources/${base}"
        cp "$f" "data/accession-resources/${base}/input_${base}.nii.gz"
        label="/tmp/msd_data/Task09_Spleen/labelsTr/${base}.nii.gz"
        if [ -f "$label" ]; then
            cp "$label" "data/accession-resources/${base}/label_${base}.nii.gz"
        fi
    done

    # Generate the CSV
    ls data/accession-resources/ | awk 'BEGIN{print "accession_id"}{print}' > data/dataframe.csv

Configure local development in ``.env.development``:

.. code-block:: bash

    LOCAL_DEV=true
    DEV_IMAGES_DIR=../data/accession-resources
    DEV_DATAFRAME=../data/dataframe.csv
    JOB_TYPE=standard


Deployment
----------

**Package and submit the job:**

.. code-block:: bash

    # Package the job
    zip -r spleen-segmentation.zip spleen-segmentation/

    # Submit via FLIP-API
    curl -X POST \
      -F "job_folder=@spleen-segmentation.zip" \
      http://flip-api:8000/submit_job/spleen-segmentation

    # Monitor progress
    curl http://flip-api:8000/list_jobs?detailed=true

    # Download results
    curl -X GET http://flip-api:8000/download_job/{job_id} -o results.zip


Verification Checklist
----------------------

Before deploying, verify:

- |box| ``models.py``: ``get_model()`` returns a ``SegmentationNetwork`` with no arguments
- |box| ``transforms.py``: Both ``get_train_transforms()`` and ``get_val_transforms()`` return ``Compose`` pipelines
- |box| ``flip_datasets.py``: Both dataset classes extend ``FLIPDataset`` and implement ``_build_datalist``
- |box| ``trainer.py``: ``FLIP_TRAINER`` extends ``ClientAlgo`` with ``initialize``, ``train``, ``get_weights``, ``finalize``
- |box| ``validator.py``: ``FLIP_VALIDATOR`` extends ``ClientAlgo`` with ``initialize``, ``evaluate``
- |box| ``config.json``: Contains ``LOCAL_ROUNDS``, ``LEARNING_RATE``, ``VAL_SPLIT``, ``net_config``
- |box| Server config: ``persistor`` references ``"models.get_model"``
- |box| Client config: Executors reference ``flip.nvflare.executors.RUN_MONAI_FL_TRAINER`` and ``RUN_MONAI_FL_VALIDATOR``
- |box| NIfTI files are named ``input_*.nii.gz`` / ``label_*.nii.gz`` and pass 3D shape validation

.. |box| unicode:: U+2610


Common Pitfalls
---------------

**Pitfall 1: Not using sliding-window inference for validation**

3D volumes are too large for a single forward pass. Without ``SlidingWindowInferer``,
you'll get out-of-memory errors:

.. code-block:: python

    # Wrong — will OOM on full CT volumes
    predictions = self.model(images)

    # Correct — tile into overlapping 96³ patches
    predictions = self.inferer(images, self.model)

**Pitfall 2: Missing shape validation in data loading**

Hospital NIfTI data is not guaranteed to be clean. Without QC, a 4D fMRI file
or a shape-mismatched segmentation will crash training:

.. code-block:: python

    # Wrong — trust all files blindly
    datalist.append({"image": str(img), "label": seg})

    # Correct — validate before including
    if len(ih.shape) != 3:
        continue
    if ih.shape != sh.shape:
        continue
    datalist.append({"image": str(img), "label": seg})

**Pitfall 3: Loading data per round instead of caching**

The FLIP API call ``get_by_accession_number()`` downloads imaging data from the
hospital PACS. Calling it every training round wastes bandwidth and time:

.. code-block:: python

    # Wrong — in train()
    def train(self, data, extra=None):
        dataset = SpleenTrainFLIPDataset(...)  # Downloads every round!

    # Correct — in initialize()
    def initialize(self, extra=None):
        self._train_dataset = SpleenTrainFLIPDataset(...)  # Downloaded once

**Pitfall 4: Returning full weights instead of weight diffs**

For a 3D UNet with ~2M parameters, transmitting full weights each round doubles
network I/O. Weight diffs are smaller and equally effective:

.. code-block:: python

    # Inefficient — full weights every round
    return ExchangeObject(
        weights={k: v.cpu().numpy() for k, v in self.model.state_dict().items()},
        weight_type=WeightType.WEIGHTS,
    )

    # Efficient — only the change from global weights
    dxo = get_model_weights_diff(self._global_weights, self.model.state_dict(), ...)
    return ExchangeObject(
        weights=dxo.data,
        weight_type=WeightType.WEIGHT_DIFF,
    )


How It All Fits Together
------------------------

.. code-block:: text

    ┌───────────────────────────────────────────────────────────────────┐
    │                        FLIP Server                                │
    │                                                                   │
    │  PTFileModelPersistor → loads SegmentationNetwork                 │
    │         │                                                         │
    │         ▼                                                         │
    │  ScatterAndGather controller                                      │
    │    Round 1:                                                       │
    │      ├─ Send global weights → Client A (Hospital 1)               │
    │      └─ Send global weights → Client B (Hospital 2)               │
    │                                                                   │
    │    Wait for weight diffs from both clients                        │
    │      ├─ Client A: trained on 30 spleen CTs, 120 iterations        │
    │      └─ Client B: trained on 50 spleen CTs, 200 iterations        │
    │                                                                   │
    │    InTimeAccumulateWeightedAggregator:                            │
    │      global_weights += (120/(120+200)) * diff_A                   │
    │                      + (200/(120+200)) * diff_B                   │
    │                                                                   │
    │    Round 2: repeat with updated global weights                    │
    │                                                                   │
    │  CrossSiteModelEval → final Dice scores from each client          │
    └───────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────────────┐
    │                   Client A (Hospital 1)                           │
    │                                                                   │
    │  RUN_MONAI_FL_TRAINER executor:                                   │
    │    1. import trainer → FLIP_TRAINER()                             │
    │    2. FLIP_TRAINER.initialize():                                  │
    │       - flip.get_dataframe() → 30 accession IDs                   │
    │       - flip.get_by_accession_number() × 30 → NIfTI paths         │
    │       - Build SpleenTrainFLIPDataset (27 train / 3 val)           │
    │    3. FLIP_TRAINER.train(global_weights):                         │
    │       - Load weights into UNet                                    │
    │       - Train 4 epochs × 9 batches = 36 iterations                │
    │       - DiceCELoss → Adam optimizer                               │
    │    4. FLIP_TRAINER.get_weights():                                 │
    │       - Return weight diff (local - global)                       │
    │                                                                   │
    │  RUN_MONAI_FL_VALIDATOR executor:                                 │
    │    1. FLIP_VALIDATOR.evaluate(global_weights):                    │
    │       - Sliding-window inference on 3 validation volumes          │
    │       - Return mean Dice score                                    │
    └───────────────────────────────────────────────────────────────────┘


Next Steps
----------

After completing this tutorial:

1. **Scale up**: Increase ``GLOBAL_ROUNDS`` to 10-20 and ``LOCAL_ROUNDS`` to 2-4
   for production-quality segmentation
2. **Add more metrics**: Report per-class Dice, Hausdorff distance, and surface
   Dice via ``send_metrics_value()``
3. **Tune privacy**: Adjust ``PercentilePrivacy`` parameters (``gamma``, ``percentile``)
   based on your institution's data governance requirements
4. **Adapt to other organs**: Change the UNet ``out_channels`` and modify
   ``_build_spleen_datalist`` to match your segmentation task


References
----------

- `NVFlare MONAI Spleen CT Segmentation <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/monai/spleen_ct_segmentation>`_
- `Medical Segmentation Decathlon — Task09 Spleen <http://medicaldecathlon.com/>`_
- `MONAI UNet <https://docs.monai.io/en/latest/networks.html#unet>`_
- `MONAI SlidingWindowInferer <https://docs.monai.io/en/latest/inferers.html#slidingwindowinferer>`_
- `FLIP API Reference <../reference/api/flip/index.html>`_
