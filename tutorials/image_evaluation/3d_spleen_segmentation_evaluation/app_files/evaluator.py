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
3D Spleen Segmentation Multi-Model Evaluator — MONAI FL Option B (ClientAlgo).

``FLIP_EVALUATOR`` extends ``ClientAlgo`` and implements ``evaluate()``.
Each model's weights arrive inside ``data.weights`` as a dict keyed by model
name (each value is a DXO object produced by the server's ``ModelLocator``).
Validation returns per-model Dice scores wrapped in an ``ExchangeObject``.

Driven by ``flip.nvflare.executors.RUN_MONAI_FL_EVALUATOR``.
"""

import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from models import model_paths
from monai.data import DataLoader, Dataset, decollate_batch
from monai.fl.client.client_algo import ClientAlgo
from monai.fl.utils.exchange_object import ExchangeObject
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from monai.transforms import AsDiscrete
from nvflare.apis.dxo import DataKind
from transforms import get_eval_transforms, get_sliding_window_inferer

from flip import FLIP
from flip.constants import PTConstants, ResourceType


class FLIP_EVALUATOR(ClientAlgo):
    def __init__(self, evaluate_task_name=PTConstants.EvalTaskName, project_id="", query=""):
        """Multi-model spleen evaluation via the MONAI FL ``ClientAlgo`` interface."""
        self._evaluate_task_name = evaluate_task_name
        self._project_id = project_id
        self._query = query
        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize(self, extra=None):
        """Set up models, transforms, and FLIP dataset."""
        working_dir = Path(__file__).parent.resolve()
        self.config = {}
        with open(str(working_dir / "config.json")) as file:
            self.config = json.load(file)
        self._test_split = self.config["TEST_SPLIT"]
        self.num_classes = self.config["num_classes"]

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.models = {}
        for model_name, model_values in self.config["models"].items():
            self.models[model_name] = model_paths[model_values["path"]]

        self.val_transforms = get_eval_transforms()
        self.flip = FLIP()
        self.dataframe = self.flip.get_dataframe(self._project_id, self._query)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.swi = get_sliding_window_inferer(sw_device=self.device)

    def evaluate(self, data, extra=None):
        """Run multi-model Dice evaluation and return metrics.

        ``data.weights`` carries a dict mapping model name to a DXO object
        (produced by the server-side ``ModelLocator``).  Each DXO's ``.data``
        holds the flat weight tensors for that model.

        Args:
            data: ``ExchangeObject`` whose ``weights`` field is
                  ``{model_name: DXO}``.
            extra: Dict containing ``fl_ctx`` and ``abort_signal``.

        Returns:
            ``ExchangeObject`` with ``metrics`` = JSON results dict.
        """
        fl_ctx = (extra or {}).get("fl_ctx")
        abort_signal = (extra or {}).get("abort_signal")

        test_dict = self.get_image_and_label_list(self.dataframe)
        self._test_dataset = Dataset(test_dict, transform=self.val_transforms)
        self.test_loader = DataLoader(self._test_dataset, batch_size=1, shuffle=False)

        if fl_ctx is not None:
            self.logger.info(f"Received task name: {self._evaluate_task_name}")

        for model_name in self.models:
            if fl_ctx is not None:
                self.logger.info(f"Loading model {model_name}...")
            dxo_model = data.weights.get(model_name)
            if dxo_model.data_kind != DataKind.WEIGHTS:
                self.logger.error(f"DXO for model {model_name} is of type {dxo_model.data_kind} but expected WEIGHTS.")
                raise ValueError(f"Unexpected data kind {dxo_model.data_kind} for model {model_name}")
            weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo_model.data.items()}
            self.models[model_name].load_state_dict(weights)

        json_results = self.do_validation(fl_ctx, abort_signal)
        return ExchangeObject(metrics=json_results)

    def finalize(self, extra=None):
        pass

    def get_image_and_label_list(self, dataframe):
        """Returns a list of dicts, each dict containing the path to an image and its corresponding label."""

        datalist = []
        # loop over each accession id in the train set
        for accession_id in dataframe["accession_id"]:
            try:
                accession_folder_path = self.flip.get_by_accession_number(
                    self._project_id,
                    accession_id,
                    resource_type=[
                        ResourceType.NIFTI,
                        # ResourceType.SEGMENTATION,
                    ],
                )
            except Exception as err:
                print(f"Could not get image data folder path for {accession_id}: {err}")
                continue
            # accession_folder_path = Path(f"/app/data/images/net-1/{accession_id}")

            print(accession_folder_path)

            all_images = list(accession_folder_path.rglob("input_*.nii.gz"))
            print(all_images)

            this_accession_matches = 0
            print(f"Total base count found for accession_id {accession_id}: {len(all_images)}")
            for img in all_images:
                # for each image, find the corresponding segmentation mask
                seg = str(img).replace("/input_", "/label_")

                if not Path(seg).exists():
                    print(f"No matching segmentation mask for {img}.")
                    continue

                try:
                    img_header = nib.load(str(img))
                except nib.filebasedimages.ImageFileError as err:
                    print(f"Problem loading header of base image {str(img)}.")
                    print(f"{err=}")
                    print(f"{type(err)=}")
                    print(f"{err.args=}")
                    continue

                try:
                    seg_header = nib.load(seg)
                except nib.filebasedimages.ImageFileError as err:
                    print(f"Problem loading header of segmentation {str(seg)}.")
                    print(f"{err=}")
                    print(f"{type(err)=}")
                    print(f"{err.args=}")
                    continue

                # check is 3D and at least 128x128x128 in size and seg is the same
                if len(img_header.shape) != 3:
                    print(f"Image has other than 3 dimensions (it has {len(img_header.shape)}.)")
                    continue
                elif any([img_dim != seg_dim for img_dim, seg_dim in zip(img_header.shape, seg_header.shape)]):
                    print(
                        f"Image dimensions ({img_header.shape}) do not match "
                        f"segmentation dimensions ({seg_header.shape})."
                    )
                    continue
                else:
                    # defines keys for image and segmentation
                    datalist.append({"image": str(img), "label": seg})
                    print("Matching base image and segmentation added.")
                    this_accession_matches += 1

            print(f"Added {this_accession_matches} matched image + segmentation pairs for {accession_id}.")

        print(f"Found {len(datalist)} files in total.")

        # split into the training and testing data
        _, test_datalist = np.split(datalist, [int((1 - self._test_split) * len(datalist))])

        print(f"Found {len(test_datalist)} files in testing.")

        return test_datalist

    def _get_json_results_from_numpy(self, metric_results):
        output_results = {}
        for model_name, dice_scores in metric_results.items():
            output_results[model_name] = {
                "spleen": {
                    "mean_dice": np.mean(dice_scores[:, 1]).item(),
                    "raw_dice": [float(i) for i in dice_scores[:, 1]],
                }
            }

        return output_results

    def do_validation(self, fl_ctx, abort_signal):
        metric_results = {}
        for model_name, _ in self.models.items():
            self.models[model_name].eval()
            metric_results[model_name] = DiceMetric(reduction="none")  # Create DiceMetric instance

        num_images = 0
        self.logger.info(len(self.test_loader))

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = (
                    batch["image"].to(self.device),
                    batch["label"].to(self.device),
                )

                for model_name, model in self.models.items():
                    self.models[model_name].to(self.device)
                    # perform sliding window inference to get a prediction for the whole volume.
                    output = self.swi(inputs=images, network=self.models[model_name])

                    # Softmax
                    output = torch.softmax(output, dim=1)
                    # Ensure labels are one-hot encoded
                    output = decollate_batch(output)
                    output = torch.stack([self.post_pred(i) for i in output], 0)
                    labels_one_hot = one_hot(labels, num_classes=self.num_classes)

                    # Compute Dice metric using DiceMetric
                    metric_results[model_name](output, labels_one_hot)  # Accumulate Dice scores
                    self.models[model_name].cpu()

                batch_size = images.shape[0]
                num_images += batch_size
                self.logger.info(f"Validator Iteration: {i}, Num Images: {num_images}")

            # Compute final Dice score
            for model_name, dice_metric in metric_results.items():
                metric_results[model_name] = dice_metric.aggregate().cpu().numpy()

            dice_metric.reset()  # Reset metric for next validation phase

            json_results = self._get_json_results_from_numpy(metric_results)
            message = ""
            for model_name in self.models.keys():
                message += f"{json_results[model_name]['spleen']['mean_dice']:.4f} ({model_name})"
            self.logger.info(f"Validator finished on the client side: {message}")

        return json_results
