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
X-Ray Classification Validator — MONAI FL Option B (ClientAlgo).

``FLIP_VALIDATOR`` extends ``ClientAlgo`` and implements ``evaluate()``.
The existing multi-label classification validation loop is preserved;
weights are received via ``ExchangeObject`` instead of DXO.

Driven by ``flip.nvflare.executors.RUN_MONAI_FL_VALIDATOR``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pydicom
import torch
from data_utils import Lesion, LesionDict, get_labels_from_radiology_row, get_lesion_label, get_xray_transforms
from loss_and_metrics import compute_precision_recall_f1, get_bce_loss
from models import get_model
from monai.data import DataLoader, Dataset
from monai.fl.client.client_algo import ClientAlgo
from monai.fl.utils.exchange_object import ExchangeObject
from nvflare.app_common.app_constant import AppConstants

from flip import FLIP
from flip.constants import ResourceType
from flip.nvflare.metrics import send_metrics_value


class FLIP_VALIDATOR(ClientAlgo):
    """X-ray multi-label classification validator using the MONAI FL ``ClientAlgo`` interface.

    Args:
        project_id: FLIP project identifier.
        query: SQL cohort query.
        validate_task_name: NVFLARE task name (informational only).
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
        """Set up model, transforms, and FLIP test dataset."""
        working_dir = Path(__file__).parent.resolve()

        with open(str(working_dir / "config.json")) as f:
            config = json.load(f)
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

        self._lesions = LesionDict(items=[Lesion(id=int(k), lesion=v) for k, v in lesions_raw.items()])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = get_model()
        self.model.to(self.device)

        self._test_transforms = get_xray_transforms(is_validation=True)

        self.flip = FLIP()
        self.dataframe = self.flip.get_dataframe(self._project_id, self._query)
        test_dict = self._build_test_datalist()

        test_dataset = Dataset(test_dict, transform=self._test_transforms)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False)

    def evaluate(self, data, extra=None):
        """Run multi-label classification validation and return metrics.

        Args:
            data: ``ExchangeObject`` carrying the global model weights.
            extra: Dict containing ``fl_ctx`` and ``abort_signal``.

        Returns:
            ``ExchangeObject`` with ``metrics`` dict.
        """
        fl_ctx = (extra or {}).get("fl_ctx")
        abort_signal = (extra or {}).get("abort_signal")

        weights = {k: torch.as_tensor(v, device=self.device) for k, v in data.weights.items()}
        metrics = self._do_validation(fl_ctx, weights, abort_signal)

        return ExchangeObject(metrics=metrics)

    def finalize(self, extra=None):
        pass

    def _do_validation(self, fl_ctx, weights, abort_signal):
        self.model.load_state_dict(weights)
        self.model.eval()

        metrics = {"loss": [], "f1-score": {}, "precision": {}, "recall": {}}
        for lesion_name in self._lesions.get_lesion_list():
            metrics["f1-score"][lesion_name] = []
            metrics["precision"][lesion_name] = []
            metrics["recall"][lesion_name] = []

        with torch.no_grad():
            for _, batch in enumerate(self.test_dataloader):
                if abort_signal is not None and abort_signal.triggered:
                    return metrics

                images = batch["image"].to(self.device)
                labels = get_lesion_label(batch, self._lesions).to(self.device)
                output = self.model(images)
                loss = get_bce_loss(output, labels).item()
                metrics["loss"].append(loss)
                output = torch.sigmoid(output)
                for pathology in self._lesions.get_lesion_list():
                    precision, recall, f1_score = compute_precision_recall_f1(
                        output, labels, pathology, lesions=self._lesions
                    )
                    metrics["precision"][pathology].append(precision)
                    metrics["recall"][pathology].append(recall)
                    metrics["f1-score"][pathology].append(f1_score)

        if fl_ctx is not None:
            send_metrics_value(label="TEST_LOSS", value=metrics["loss"][-1], fl_ctx=fl_ctx, round=0, flip=self.flip)
            for metric in ["f1-score", "precision", "recall"]:
                for lesion_name in self._lesions.get_lesion_list():
                    send_metrics_value(
                        label=f"TEST-{metric.upper()}",
                        value=metrics[metric][lesion_name][-1],
                        fl_ctx=fl_ctx,
                        round=0,
                        flip=self.flip,
                    )

        return metrics

    def _build_test_datalist(self):
        """Build the test-split datalist (third split after train/val)."""
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

        print(f"Found {len(datalist)} total DICOM test images.")
        train_end = int(len(datalist) * (1 - self._val_split - self._test_split))
        val_end = int(len(datalist) * (1 - self._test_split))
        return datalist[val_end:]
