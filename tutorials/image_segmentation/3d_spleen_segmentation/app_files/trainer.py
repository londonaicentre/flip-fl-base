# Copyright (c) 2026 Guy's and St Thomas' NHS Foundation Trust & King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
3D Spleen Segmentation Trainer — MONAI FL Option B (ClientAlgo).

``FLIP_TRAINER`` extends ``ClientAlgo`` and implements the standard training
lifecycle.  Training hyperparameters and network architecture mirror the
original ``configs/train.json`` Bundle config:

- Network: from ``models.get_model()``
- Loss: ``DiceCELoss(to_onehot_y, softmax, lambda_ce=0.2, lambda_dice=0.8)``
- Optimizer: ``Adam(lr=LEARNING_RATE)``
- DataLoader: ``batch_size=3``, ``shuffle=True``, ``num_workers=1``
- Epochs per round: ``LOCAL_ROUNDS``

Driven by ``flip.nvflare.executors.RUN_MONAI_FL_TRAINER``.
"""

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
    """3D Spleen Segmentation trainer using the MONAI FL ``ClientAlgo`` interface.

    The dataset is built once at ``initialize()`` time (FLIP API call) and
    reused across all training rounds.  Each call to ``train()`` creates a
    fresh ``DataLoader`` so that shuffle state resets each round.

    Args:
        project_id: FLIP project identifier.
        query: SQL cohort query passed to ``flip.get_dataframe()``.
        train_task_name: NVFLARE task name for training (informational only).
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
        """Set up device, model, optimizer, loss, transforms, and FLIP dataset."""
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._learning_rate)
        self.inferer = SimpleInferer()

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
        """Load global weights and run ``LOCAL_ROUNDS`` training epochs.

        Args:
            data: ``ExchangeObject`` carrying global model weights.
            extra: Dict with ``fl_ctx``, ``abort_signal``, ``current_round``.
        """
        fl_ctx = (extra or {}).get("fl_ctx")
        abort_signal = (extra or {}).get("abort_signal")

        # Store global weights for weight-diff computation in get_weights()
        self._global_weights = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.weights.items()}
        self.model.load_state_dict({k: torch.as_tensor(v, device=self.device) for k, v in data.weights.items()})

        train_loader = DataLoader(self._train_dataset, batch_size=3, shuffle=True, num_workers=1)
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

            if fl_ctx is not None:
                send_metrics_value(label="train_loss", value=epoch_loss / max(1, len(train_loader)), fl_ctx=fl_ctx, flip=self.flip)
            self.logger.info(f"Epoch {epoch + 1}/{self._local_rounds} loss: {epoch_loss / max(1, len(train_loader)):.4f}")

    def get_weights(self, extra=None):
        """Return weight diff (or full weights for submit_model).

        Args:
            extra: Dict with ``weight_type`` key.

        Returns:
            ``ExchangeObject`` with weights and statistics.
        """
        weight_type = (extra or {}).get("weight_type", WeightType.WEIGHT_DIFF)

        if weight_type == WeightType.WEIGHT_DIFF and self._global_weights is not None:
            dxo = get_model_weights_diff(self._global_weights, self.model.state_dict(), self._n_iterations)
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

