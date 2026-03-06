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
3D Spleen Segmentation Validator — MONAI FL ClientAlgo interface.

``FLIP_VALIDATOR`` extends ``ClientAlgo`` and implements the ``evaluate()``
lifecycle method.  It receives global model weights from the server via an
``ExchangeObject``, runs sliding-window Dice validation on the FLIP cohort's
validation split, and returns an ``ExchangeObject`` carrying the mean Dice
score as a metric.

This class is driven by ``flip.nvflare.executors.RUN_MONAI_FL_VALIDATOR``:
  - ``initialize(extra)`` — set up model, transforms, and FLIP val dataset.
  - ``evaluate(data, extra)`` — load weights, run validation, return metrics.
"""

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
    """3D Spleen Segmentation validator using the MONAI FL ``ClientAlgo`` interface.

    Datasets are built once at ``initialize()`` time (FLIP API call) and reused
    across validation rounds.  Each call to ``evaluate()`` creates a fresh
    ``DataLoader`` from the cached dataset so that no state leaks between rounds.

    Args:
        project_id: FLIP project identifier.
        query: SQL cohort query passed to ``flip.get_dataframe()``.
        validate_task_name: NVFLARE task name for validation (informational only;
            the adapter routes the call to ``evaluate()`` directly).
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
        """Set up device, model, transforms, and FLIP validation dataset."""
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
        """Run sliding-window Dice validation and return metrics.

        Args:
            data: ``ExchangeObject`` carrying the global model weights.
            extra: Dict containing ``fl_ctx`` and ``abort_signal`` from the adapter.

        Returns:
            ``ExchangeObject`` with ``metrics={"val_acc": mean_dice}``.
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
                predictions = torch.stack([self.post_pred(p) for p in predictions], 0)
                labels = torch.stack([self.post_pred_gt(lb) for lb in labels], 0)
                self.dice_acc(y_pred=predictions, y=labels)
                val_dice.append(self.dice_acc.aggregate().item())

        metric = float(np.mean(val_dice)) if val_dice else 0.0
        self.logger.info(f"Validation Dice: {metric:.4f}")
        if fl_ctx is not None:
            send_metrics_value(label="TEST_DICE", value=metric, fl_ctx=fl_ctx, flip=self.flip)

        return metric
