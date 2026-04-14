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
Latent Diffusion Model Validator — MONAI FL Option B (ClientAlgo).

``FLIP_VALIDATOR`` extends ``ClientAlgo`` and implements ``evaluate()``.
The AE and DM validation loops are preserved; the ``validate_task_name``
constructor parameter selects which loop to run.

Bug fixed: same dict-comprehension bug as in the trainer is corrected here.

Driven by ``flip.nvflare.executors.RUN_MONAI_FL_VALIDATOR``.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from models import get_model
from monai.data import DataLoader, Dataset
from monai.fl.client.client_algo import ClientAlgo
from monai.fl.utils.exchange_object import ExchangeObject
from monai.inferers import LatentDiffusionInferer
from monai.metrics import compute_ssim_and_cs
from monai.networks.schedulers import DDPMScheduler
from nvflare.app_common.app_constant import AppConstants
from torch.amp import autocast
from transforms import get_val_transforms

from flip import FLIP
from flip.constants import ResourceType
from flip.nvflare.metrics import send_metrics_value


class FLIP_VALIDATOR(ClientAlgo):
    """Latent Diffusion Model validator using the MONAI FL ``ClientAlgo`` interface.

    Selects the validation loop via ``validate_task_name``:
    - ``"validate_ae"``: AE reconstruction quality (L1 + SSIM).
    - ``"validate_dm"``: Diffusion model noise-prediction loss.

    Args:
        project_id: FLIP project identifier.
        query: SQL cohort query.
        validate_task_name: Determines which validation phase to execute.
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

    def initialize(self, extra=None):
        """Set up model, losses, transforms, and FLIP val dataset."""
        working_dir = Path(__file__).parent.resolve()
        self.working_dir = working_dir

        with open(str(working_dir / "config.json")) as f:
            self.config = json.load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = get_model()
        self.model.to(self.device)

        self.losses_ae = {"reconstruction_loss": torch.nn.L1Loss()}
        self.losses_dm = {"loss": torch.nn.functional.mse_loss}
        self.optimizers_dm = {
            "scheduler": DDPMScheduler(
                num_train_timesteps=1000,
                schedule="scaled_linear_beta",
                clip_sample=False,
                prediction_type="epsilon",
                beta_start=0.0015,
                beta_end=0.0195,
            ),
        }

        torch.hub.set_dir("/code/torch_hub")
        self._transforms = get_val_transforms(self.config["spatial_shape"])

        self.flip = FLIP()
        self.axial_anisotropy = None
        self.dataframe = self.flip.get_dataframe(project_id=self._project_id, query=self._query)
        _, val_dict = self._build_datalist(self.dataframe)
        self._test_dataset = Dataset(val_dict, transform=self._transforms)

    def evaluate(self, data, extra=None):
        """Run the appropriate validation loop and return metrics.

        Args:
            data: ``ExchangeObject`` carrying the global model weights.
            extra: Dict containing ``fl_ctx`` and ``abort_signal``.

        Returns:
            ``ExchangeObject`` with ``metrics`` dict.
        """
        fl_ctx = (extra or {}).get("fl_ctx")
        abort_signal = (extra or {}).get("abort_signal")

        weights = {k: torch.as_tensor(v, device=self.device) for k, v in data.weights.items()}
        batch_size = (
            self.config["BATCH_SIZE_AE"] if self._validate_task_name == "validate_ae" else self.config["BATCH_SIZE_DM"]
        )
        self._test_loader = DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        if self._validate_task_name == "validate_ae":
            val_loss, val_ssim = self.do_validation_ae(fl_ctx, weights, abort_signal)
            metrics = {f"metrics_{self._validate_task_name}": {"val_loss": val_loss, "val_ssim": val_ssim}}
        elif self._validate_task_name == "validate_dm":
            val_loss = self.do_validation_dm(fl_ctx, weights, abort_signal)
            metrics = {f"metrics_{self._validate_task_name}": {"val_loss": val_loss}}
        else:
            metrics = {}

        return ExchangeObject(metrics=metrics)

    def finalize(self, extra=None):
        pass

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _build_datalist(self, dataframe):
        datalist = []
        for accession_id in dataframe["accession_id"]:
            try:
                folder = self.flip.get_by_accession_number(
                    self._project_id, accession_id, resource_type=[ResourceType.NIFTI]
                )
            except Exception as err:
                print(f"Could not get data for {accession_id}: {err}")
                continue

            all_images = list(folder.rglob("input_*.nii.gz"))
            if self.axial_anisotropy is None and all_images:
                try:
                    nib_image = np.asarray(nib.load(str(all_images[0])).dataobj)
                    ip_2_op = nib_image.shape[0] / nib_image.shape[-1]
                    self.axial_anisotropy = ip_2_op > 1.5
                except Exception as e:
                    print(f"Error loading NIfTI for {accession_id}: {e}")

            for img in all_images:
                datalist.append({"image": str(img)})

        print(f"Found {len(datalist)} NIfTI val files.")
        val_end = int(self.config["VAL_SPLIT"] * len(datalist))
        return datalist[val_end:], datalist[:val_end]

    def derive_new_latent_shape(self, input_shape: list, num_downsamplings: int) -> list:
        output_shape = []
        for s in input_shape:
            while s % (2**num_downsamplings) != 0:
                s += 1
            output_shape.append(int(s))
        return output_shape

    # ------------------------------------------------------------------
    # AE validation
    # ------------------------------------------------------------------

    def do_validation_ae(self, fl_ctx, weights, abort_signal):
        self.model.load_state_dict(weights)
        self.model.eval()

        l1_test_loss = 0.0
        ssim_test = 0.0

        for _, batch in enumerate(self._test_loader):
            if abort_signal is not None and abort_signal.triggered:
                return 0.0, 0.0
            images = batch["image"].to(self.device)
            with torch.no_grad():
                reconstruction, _, _ = self.model.autoencoder(images)
            reconstruction = reconstruction.detach().cpu()
            images = images.detach().cpu()
            reconstruction_norm = (reconstruction - reconstruction.min()) / (
                reconstruction.max() - reconstruction.min() + 1e-8
            )
            _, ssim_metric = compute_ssim_and_cs(
                reconstruction_norm,
                images,
                spatial_dims=len(reconstruction.shape[2:]),
                data_range=1,
                kernel_size=[11] * len(reconstruction.shape[2:]),
                kernel_sigma=[1.5] * len(reconstruction.shape[2:]),
            )
            l1_test_loss += self.losses_ae["reconstruction_loss"](reconstruction.float(), images.float()).item()
            ssim_test += ssim_metric.mean().item()

        n = max(1, len(self._test_loader))
        l1_test_loss /= n
        ssim_test /= n

        if fl_ctx is not None:
            send_metrics_value(label="val_l1_loss", value=l1_test_loss, fl_ctx=fl_ctx, flip=self.flip)
            send_metrics_value(label="val_ssim", value=ssim_test, fl_ctx=fl_ctx, flip=self.flip)

        return l1_test_loss, ssim_test

    # ------------------------------------------------------------------
    # DM validation
    # ------------------------------------------------------------------

    def do_validation_dm(self, fl_ctx, weights, abort_signal):
        self.model.load_state_dict(state_dict=weights, strict=False)
        self.model.diffusion_model.to(device=self.device)
        self.model.autoencoder.to(device=self.device)

        autoencoder_latent_shape = [
            i / (2 ** (len(self.model.autoencoder.decoder.channels) - 1)) for i in self.config["spatial_shape"]
        ]
        ldm_latent_shape = self.derive_new_latent_shape(
            autoencoder_latent_shape, len(self.model.diffusion_model.block_out_channels) - 1
        )

        with torch.no_grad():
            with autocast(enabled=True, device_type=self.device.type):
                sample_z = self.model.autoencoder.encode_stage_2_inputs(
                    next(iter(self._test_loader))["image"].to(self.device)
                )
                scale_factor = 1 / torch.std(sample_z)
                del sample_z

        inferer = LatentDiffusionInferer(
            scheduler=self.optimizers_dm["scheduler"],
            ldm_latent_shape=ldm_latent_shape,
            autoencoder_latent_shape=autoencoder_latent_shape,
            scale_factor=scale_factor.item(),
        )

        self.model.diffusion_model.eval()
        self.model.autoencoder.eval()
        val_loss = []

        for _, batch in enumerate(self._test_loader):
            if abort_signal is not None and abort_signal.triggered:
                return 0.0
            images = batch["image"].to(self.device)
            with autocast(enabled=False, device_type=self.device.type):
                with torch.no_grad():
                    noise = torch.randn(
                        [images.shape[0]] + [self.model.autoencoder.encoder.blocks[-1].out_channels] + ldm_latent_shape
                    ).to(self.device)
                    timesteps = torch.randint(
                        0,
                        self.optimizers_dm["scheduler"].num_train_timesteps,
                        (images.shape[0],),
                        device=self.device,
                    ).long()
                    noise_pred = inferer(
                        inputs=images,
                        diffusion_model=self.model.diffusion_model,
                        autoencoder_model=self.model.autoencoder,
                        noise=noise,
                        timesteps=timesteps,
                        condition=None,
                        mode="crossattn",
                    )
                loss = self.losses_dm["loss"](noise.float(), noise_pred.float()).item()
                val_loss.append(loss)

        result = float(np.mean(val_loss)) if val_loss else 0.0
        if fl_ctx is not None:
            send_metrics_value(label="Total loss DM (val)", value=result, fl_ctx=fl_ctx, flip=self.flip)

        return result
