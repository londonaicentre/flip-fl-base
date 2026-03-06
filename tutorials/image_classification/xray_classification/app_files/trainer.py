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
X-Ray Classification Trainer — MONAI FL Option B (ClientAlgo).

``FLIP_TRAINER`` extends ``ClientAlgo``.  The existing training loop, loss
functions, metrics, and data loading logic are preserved; only the interface
is changed from ``nvflare.apis.executor.Executor`` to the platform-agnostic
``monai.fl.client.client_algo.ClientAlgo``.

Key differences from the legacy executor:
- ``__init__`` no longer receives a Shareable; heavy setup moves to
  ``initialize(extra)``.
- ``execute()`` is replaced by ``train(data, extra)`` and
  ``get_weights(extra)``.
- Model weights are exchanged via ``ExchangeObject`` instead of DXO.
- ``fl_ctx`` and ``abort_signal`` come from ``extra`` dict.
- Data is loaded once at ``initialize()`` and cached across rounds.

Driven by ``flip.nvflare.executors.RUN_MONAI_FL_TRAINER``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pydicom
import torch
from data_utils import Lesion, LesionDict, get_labels_from_radiology_row, get_lesion_label, get_xray_transforms
from loss_and_metrics import compute_precision_recall_f1, get_bce_loss
from models import get_model
from monai.data import DataLoader, Dataset
from monai.fl.client.client_algo import ClientAlgo
from monai.fl.utils.constants import WeightType
from monai.fl.utils.exchange_object import ExchangeObject
from nvflare.app_common.app_constant import AppConstants

from flip import FLIP
from flip.constants import ResourceType
from flip.nvflare.metrics import send_metrics_value
from flip.utils import get_model_weights_diff


class FLIP_TRAINER(ClientAlgo):
    """X-ray multi-label classification trainer using the MONAI FL ``ClientAlgo`` interface.

    Args:
        project_id: FLIP project identifier.
        query: SQL cohort query.
        train_task_name: NVFLARE task name (informational only; used to keep the
            constructor signature compatible with the adapter).
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

    # ------------------------------------------------------------------
    # ClientAlgo lifecycle
    # ------------------------------------------------------------------

    def initialize(self, extra=None):
        """Set up model, optimizer, transforms, and FLIP datasets."""
        working_dir = Path(__file__).parent.resolve()

        with open(str(working_dir / "config.json")) as f:
            config = json.load(f)
        self._epochs = config["LOCAL_ROUNDS"]
        self._lr_start = config["LR_START"]
        self._lr_end = config["LR_END"]
        self._val_split = config["VAL_SPLIT"]
        self._test_split = config["TEST_SPLIT"]
        lesions_raw = config["LESIONS"]
        self._value_to_numerical = {int(k): v for k, v in config["value_to_numerical"].items()}
        if 0 not in self._value_to_numerical or 1 not in self._value_to_numerical:
            raise ValueError("value_to_numerical must contain mappings for 0 and 1.")
        if "-1" in lesions_raw:
            self._normal_key = lesions_raw.pop("-1")
        else:
            self._normal_key = "Normal"
        self._batch_size = config["BATCH_SIZE"]
        self.validate_every = config.get("VALIDATE_EVERY", 1)

        self._lesions = LesionDict(items=[Lesion(id=int(k), lesion=v) for k, v in lesions_raw.items()])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = get_model()
        self.model.to(self.device)

        self._train_transforms = get_xray_transforms()
        self._val_transforms = get_xray_transforms(is_validation=True)

        self.flip = FLIP()
        self.dataframe = self.flip.get_dataframe(self._project_id, self._query)
        if "accession_id" not in self.dataframe.columns:
            raise ValueError("Dataframe must contain 'accession_id' column.")
        train_dict, val_dict = self._build_datalist()

        self.training_dataset = Dataset(train_dict, transform=self._train_transforms)
        self.training_dataloader = DataLoader(self.training_dataset, batch_size=self._batch_size, shuffle=True)
        self.validation_dataset = Dataset(val_dict, transform=self._val_transforms)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=self._batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr_start)
        gamma_lr = (self._lr_end / self._lr_start) ** (1 / self._epochs)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma_lr)

        self._global_weights = None  # populated in train(), used in get_weights()
        self._n_iterations = 0

    def train(self, data, extra=None):
        """Load global weights and run local training epochs."""
        fl_ctx = (extra or {}).get("fl_ctx")
        abort_signal = (extra or {}).get("abort_signal")
        global_round = (extra or {}).get("current_round", 0)

        torch_weights = {k: torch.as_tensor(v) for k, v in data.weights.items()}
        self._global_weights = {k: v.clone() for k, v in torch_weights.items()}
        self._local_train(fl_ctx, torch_weights, abort_signal, global_round)

    def get_weights(self, extra=None):
        """Return weight diff or full weights depending on ``weight_type`` in extra."""
        weight_type = (extra or {}).get("weight_type", WeightType.WEIGHTS)
        current_weights = self.model.state_dict()

        if weight_type == WeightType.WEIGHT_DIFF and self._global_weights is not None:
            dxo = get_model_weights_diff(self._global_weights, current_weights, self._n_iterations)
            return ExchangeObject(
                weights=dxo.data,
                weight_type=WeightType.WEIGHT_DIFF,
                statistics={"num_steps": self._n_iterations},
            )

        return ExchangeObject(
            weights={k: v.cpu().numpy() for k, v in current_weights.items()},
            weight_type=WeightType.WEIGHTS,
        )

    def finalize(self, extra=None):
        pass

    # ------------------------------------------------------------------
    # Training loop (unchanged from legacy executor)
    # ------------------------------------------------------------------

    def _local_train(self, fl_ctx, weights, abort_signal, global_round):
        self.model.load_state_dict(state_dict=weights)
        self.model.train()

        training_metrics = {"loss": {"train": [], "val": []}, "f1-score": {}, "precision": {}, "recall": {}}
        for lesion_name in self._lesions.get_lesion_list():
            training_metrics["f1-score"][lesion_name] = {"train": [], "val": []}
            training_metrics["precision"][lesion_name] = {"train": [], "val": []}
            training_metrics["recall"][lesion_name] = {"train": [], "val": []}

        self._n_iterations = 0
        for epoch in range(self._epochs):
            training_metrics_ = {"loss": {"train": [], "val": []}, "f1-score": {}, "precision": {}, "recall": {}}
            for lesion_name in self._lesions.get_lesion_list():
                training_metrics_["f1-score"][lesion_name] = {"train": [], "val": []}
                training_metrics_["precision"][lesion_name] = {"train": [], "val": []}
                training_metrics_["recall"][lesion_name] = {"train": [], "val": []}

            for _, batch in enumerate(self.training_dataloader):
                if abort_signal is not None and abort_signal.triggered:
                    return

                images = batch["image"].to(self.device)
                labels = get_lesion_label(batch, self._lesions).to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = get_bce_loss(output, labels)
                loss.backward()
                self.optimizer.step()
                training_metrics_["loss"]["train"].append(loss.item())
                output = torch.sigmoid(output)
                for pathology in self._lesions.get_lesion_list():
                    precision, recall, f1_score = compute_precision_recall_f1(
                        output, labels, pathology, lesions=self._lesions
                    )
                    training_metrics_["precision"][pathology]["train"].append(precision)
                    training_metrics_["recall"][pathology]["train"].append(recall)
                    training_metrics_["f1-score"][pathology]["train"].append(f1_score)
                self._n_iterations += 1

            if epoch % self.validate_every == 0:
                self.model.eval()
                for _, batch in enumerate(self.validation_dataloader):
                    if abort_signal is not None and abort_signal.triggered:
                        return

                    images = batch["image"].to(self.device)
                    labels = get_lesion_label(batch, self._lesions).to(self.device)
                    output = self.model(images)
                    loss = get_bce_loss(output, labels).item()
                    training_metrics_["loss"]["val"].append(loss)
                    output = torch.sigmoid(output)
                    for pathology in self._lesions.get_lesion_list():
                        precision, recall, f1_score = compute_precision_recall_f1(
                            output, labels, pathology, lesions=self._lesions
                        )
                        training_metrics_["precision"][pathology]["val"].append(precision)
                        training_metrics_["recall"][pathology]["val"].append(recall)
                        training_metrics_["f1-score"][pathology]["val"].append(f1_score)
                self.model.train()

            self.scheduler.step()

            for metric, metric_dump in training_metrics_.items():
                if metric == "loss":
                    training_metrics["loss"]["train"].append(np.mean(metric_dump["train"]))
                    if epoch % self.validate_every == 0:
                        training_metrics["loss"]["val"].append(np.mean(metric_dump["val"]))
                    else:
                        training_metrics["loss"]["val"].append(
                            training_metrics["loss"]["val"][-1] if training_metrics["loss"]["val"] else 0
                        )
                else:
                    for lesion_name in self._lesions.get_lesion_list():
                        training_metrics[metric][lesion_name]["train"].append(
                            np.mean(metric_dump[lesion_name]["train"])
                        )
                        if epoch % self.validate_every == 0:
                            training_metrics[metric][lesion_name]["val"].append(
                                np.mean(metric_dump[lesion_name]["val"])
                            )
                        else:
                            prev = training_metrics[metric][lesion_name]["val"]
                            training_metrics[metric][lesion_name]["val"].append(prev[-1] if prev else 0)

            if fl_ctx is not None:
                round_num = global_round * self._epochs + epoch + 1
                send_metrics_value(
                    label="TRAIN_LOSS",
                    round=round_num,
                    value=training_metrics["loss"]["train"][-1],
                    fl_ctx=fl_ctx,
                    flip=self.flip,
                )
                send_metrics_value(
                    label="VAL_LOSS",
                    round=round_num,
                    value=training_metrics["loss"]["val"][-1],
                    fl_ctx=fl_ctx,
                    flip=self.flip,
                )
                for metric in ["f1-score", "precision", "recall"]:
                    for lesion_name in self._lesions.get_lesion_list():
                        send_metrics_value(
                            label=f"TRAIN-{metric.upper()}",
                            round=round_num,
                            value=training_metrics[metric][lesion_name]["train"][-1],
                            fl_ctx=fl_ctx,
                            flip=self.flip,
                        )
                        send_metrics_value(
                            label=f"VAL-{metric.upper()}",
                            round=round_num,
                            value=training_metrics[metric][lesion_name]["val"][-1],
                            fl_ctx=fl_ctx,
                            flip=self.flip,
                        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _build_datalist(self):
        """Build train and val datalists from the FLIP cohort dataframe."""
        datalist = []
        for _, row in self.dataframe.iterrows():
            accession_id = row["accession_id"]
            pathology_dict = get_labels_from_radiology_row(
                row, self._lesions, self._value_to_numerical, self._normal_key
            )
            try:
                folder = self.flip.get_by_accession_number(
                    self._project_id, accession_id, resource_type=[ResourceType.DICOM]
                )
            except Exception as err:
                print(f"Could not get data for {accession_id}: {err}")
                continue

            matched = 0
            for img in folder.rglob("*.dcm"):
                try:
                    pydicom.dcmread(str(img))
                except Exception as e:
                    print(f"Could not read DICOM {img}: {e}")
                    continue
                item = {"image": str(img)}
                item.update(pathology_dict)
                datalist.append(item)
                matched += 1
            print(f"Added {matched} DICOM images for {accession_id}.")

        print(f"Found {len(datalist)} total DICOM images.")
        train_end = int(len(datalist) * (1 - self._val_split - self._test_split))
        val_end = int(len(datalist) * (1 - self._test_split))
        return datalist[:train_end], datalist[train_end:val_end]
