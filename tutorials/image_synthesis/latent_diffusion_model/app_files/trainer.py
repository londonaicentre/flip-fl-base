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
Latent Diffusion Model Trainer — MONAI FL Option B (ClientAlgo).

``FLIP_TRAINER`` extends ``ClientAlgo``.  The two-phase adversarial training
loop (AE + discriminator, then diffusion model) is preserved unchanged from
the legacy executor; only the interface boundary is changed.

The ``train_task_name`` constructor parameter tells the trainer which phase
to execute:

- ``"train_ae"``  → ``local_train_ae()``
- ``"train_dm"``  → ``local_train_dm()``

Two separate ``RUN_MONAI_FL_TRAINER`` executor entries in
``config_fed_client.json`` (one per phase, each with the appropriate
``train_task_name``) instantiate two independent ``FLIP_TRAINER`` objects.

Bug fixed: the legacy ``datalist.append({"image": str(i) for i in all_images})``
was a dict comprehension that collapsed all images into one dict with a single
``"image"`` key (the last path).  This is corrected to append one dict per
image file.

Driven by ``flip.nvflare.executors.RUN_MONAI_FL_TRAINER``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import einops
import nibabel as nib
import numpy as np
import torch
from models import get_model
from monai.data import DataLoader, Dataset
from monai.fl.client.client_algo import ClientAlgo
from monai.fl.utils.constants import WeightType
from monai.fl.utils.exchange_object import ExchangeObject
from monai.inferers import LatentDiffusionInferer
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.schedulers import DDPMScheduler
from nvflare.app_common.app_constant import AppConstants
from torch.amp import GradScaler, autocast
from transforms import get_train_transforms, get_val_transforms

from flip import FLIP
from flip.constants import FlipConstants, ResourceType
from flip.nvflare.metrics import send_metrics_value
from flip.utils import get_model_weights_diff


class KLDivergenceLoss:
    """KL-divergence loss for VAE regularisation."""

    def __call__(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        z_mu_m = z_mu.flatten(start_dim=2)
        z_sigma_m = z_sigma.flatten(start_dim=2)
        kl_loss = 0.5 * torch.sum(z_mu_m.pow(2) + z_sigma_m.pow(2) - torch.log(z_sigma_m.pow(2)) - 1, dim=[-1])
        return torch.sum(kl_loss) / kl_loss.shape[0]


class FLIP_TRAINER(ClientAlgo):
    """Latent Diffusion Model trainer using the MONAI FL ``ClientAlgo`` interface.

    Implements two training phases selected by ``train_task_name``:
    - ``"train_ae"``: autoencoder + discriminator adversarial training.
    - ``"train_dm"``: diffusion model training.

    Args:
        project_id: FLIP project identifier.
        query: SQL cohort query.
        train_task_name: Determines which training phase to execute.
        epochs_ae: Fallback AE epoch count if not in config.json.
        epochs_dm: Fallback DM epoch count if not in config.json.
    """

    def __init__(
        self,
        project_id: str = "",
        query: str = "",
        train_task_name: str = AppConstants.TASK_TRAIN,
        epochs_ae: int = 5,
        epochs_dm: int = 5,
    ) -> None:
        self._project_id = project_id
        self._query = query
        self._train_task_name = train_task_name
        self._epochs_ae_default = epochs_ae
        self._epochs_dm_default = epochs_dm
        self._global_weights = None
        self._n_iterations = 0

    # ------------------------------------------------------------------
    # ClientAlgo lifecycle
    # ------------------------------------------------------------------

    def initialize(self, extra=None):
        """Set up models, losses, optimizers, schedulers, and FLIP datasets."""
        working_dir = Path(__file__).parent.resolve()
        self.working_dir = working_dir

        with open(str(working_dir / "config.json")) as f:
            self.config = json.load(f)

        epochs_ae = self.config.get("LOCAL_ROUNDS_AE", self._epochs_ae_default)
        epochs_dm = self.config.get("LOCAL_ROUNDS_DM", self._epochs_dm_default)
        self.params_autoencoder = {"lr_g": self.config["LR_G"], "lr_d": self.config["LR_D"], "epochs": epochs_ae}
        self.params_diffusion = {"lr": self.config["LR_DM"], "epochs": epochs_dm}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = get_model()

        torch.hub.set_dir(f"{working_dir}/torch_hub")
        self.losses_ae = {
            "reconstruction_loss": torch.nn.L1Loss(),
            "kld_loss": KLDivergenceLoss(),
            "gan_loss": PatchAdversarialLoss(criterion="least_squares"),
            "perceptual_loss": PerceptualLoss(2, network_type="alex"),
        }
        self.perceptual_slices = 12
        self.losses_dm = {"loss": torch.nn.functional.mse_loss}
        self.weights_ae = {
            "w_reconstruction_loss": self.config["w_reconstruction_loss"],
            "w_perceptual_loss": self.config["w_perceptual_loss"],
            "w_kl_loss": self.config["w_kl_loss"],
            "w_gan_loss": self.config["w_gan_loss"],
        }

        self.optimizers_ae = {
            "optimizer_g": torch.optim.Adam(
                self.model.autoencoder.parameters(), lr=self.params_autoencoder["lr_g"], weight_decay=1e-6, amsgrad=True
            ),
            "optimizer_d": torch.optim.Adam(
                self.model.discriminator.parameters(),
                lr=self.params_autoencoder["lr_d"],
                weight_decay=1e-6,
                amsgrad=True,
            ),
        }
        self.optimizers_dm = {
            "optimizer": torch.optim.Adam(
                self.model.diffusion_model.parameters(), lr=self.params_diffusion["lr"], weight_decay=1e-6, amsgrad=True
            ),
            "scheduler": DDPMScheduler(
                num_train_timesteps=1000,
                schedule="scaled_linear_beta",
                clip_sample=False,
                prediction_type="epsilon",
                beta_start=0.0015,
                beta_end=0.0195,
            ),
        }

        self.flip = FLIP()
        self.axial_anisotropy = None
        self.dataframe = self.flip.get_dataframe(project_id=self._project_id, query=self._query)
        self.train_dict, self.val_dict = self._build_datalist(self.dataframe)

        self._transforms = get_train_transforms(self.config["spatial_shape"])
        self._val_transforms = get_val_transforms(self.config["spatial_shape"])

    def train(self, data, extra=None):
        """Load global weights and run the appropriate training phase."""
        fl_ctx = (extra or {}).get("fl_ctx")
        abort_signal = (extra or {}).get("abort_signal")
        global_round = (extra or {}).get("current_round", 0)

        torch_weights = {k: torch.as_tensor(v) for k, v in data.weights.items()}
        self._global_weights = {k: v.clone() for k, v in torch_weights.items()}

        fl_ctx_job_id = fl_ctx.get_job_id() if fl_ctx is not None else "local"
        train_dict = self._site_split(train_dict=self.train_dict, fl_ctx=fl_ctx)

        train_dataset = Dataset(train_dict, transform=self._transforms)
        val_dataset = Dataset(self.val_dict, transform=self._val_transforms)

        if self._train_task_name == "train_ae":
            self._train_loader = DataLoader(
                train_dataset, batch_size=self.config["BATCH_SIZE_AE"], shuffle=True, num_workers=1
            )
            self._val_loader = DataLoader(
                val_dataset, batch_size=self.config["BATCH_SIZE_AE"], shuffle=True, num_workers=1
            )
            self._n_iterations = len(self._train_loader)
            self.local_train_ae(fl_ctx, torch_weights, abort_signal, global_round)
        elif self._train_task_name == "train_dm":
            self._train_loader = DataLoader(
                train_dataset, batch_size=self.config["BATCH_SIZE_DM"], shuffle=True, num_workers=1
            )
            self._val_loader = DataLoader(
                val_dataset, batch_size=self.config["BATCH_SIZE_DM"], shuffle=True, num_workers=1
            )
            self._n_iterations = len(self._train_loader)
            self.local_train_dm(fl_ctx, torch_weights, abort_signal, global_round)

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
    # Data loading
    # ------------------------------------------------------------------

    def _build_datalist(self, dataframe):
        """Build train / val datalists (NIfTI images only)."""
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
                    if ip_2_op > 1.5:
                        self.axial_anisotropy = True
                        print(f"Found axial anisotropy (in-plane/out-plane ratio {ip_2_op:.2f}).")
                        self.reset_perceptual_to_anisotropic()
                    else:
                        self.axial_anisotropy = False
                except Exception as e:
                    print(f"Error loading NIfTI for anisotropy check ({accession_id}): {e}")

            for img in all_images:
                datalist.append({"image": str(img)})

        print(f"Found {len(datalist)} NIfTI files in total.")
        val_end = int(self.config["VAL_SPLIT"] * len(datalist))
        return datalist[val_end:], datalist[:val_end]

    def _site_split(self, train_dict, fl_ctx):
        """Optionally split training data deterministically by site name."""
        if fl_ctx is None:
            return train_dict
        site_name = fl_ctx.get_prop("__client_name__", "") or ""
        if site_name == "site1":
            return train_dict[: len(train_dict) // 2]
        if site_name == "site2":
            return train_dict[len(train_dict) // 2 :]
        return train_dict

    def reset_perceptual_to_anisotropic(self):
        if self.axial_anisotropy and self.losses_ae["perceptual_loss"].spatial_dims == 3:
            self.losses_ae["perceptual_loss"] = PerceptualLoss(spatial_dims=2, network_type="radimagenet_resnet50")
            if self.weights_ae["w_perceptual_loss"] <= 1.0:
                self.weights_ae["w_perceptual_loss"] = 10

    def config_batch_accumulation(self, phase: str):
        if self.config[phase] < 8:
            self.batch_accumulation_step = 8 // self.config[phase]
        else:
            self.batch_accumulation_step = 1

    def derive_new_latent_shape(self, input_shape: list, num_downsamplings: int) -> list:
        output_shape = []
        for s in input_shape:
            while s % (2**num_downsamplings) != 0:
                s += 1
            output_shape.append(int(s))
        return output_shape

    # ------------------------------------------------------------------
    # AE training loop (unchanged logic from legacy executor)
    # ------------------------------------------------------------------

    def local_train_ae(self, fl_ctx, weights, abort_signal, global_round: int = 0):
        self.config_batch_accumulation(phase="BATCH_SIZE_AE")
        self.model.load_state_dict(state_dict=weights, strict=False)
        self.model.autoencoder.to(device=self.device)
        self.model.discriminator.to(device=self.device)
        self.losses_ae["perceptual_loss"].to(self.device)

        scaler_g = GradScaler(enabled=True)
        scaler_d = GradScaler(enabled=True)

        if FlipConstants.LOCAL_DEV and fl_ctx is not None:
            os.makedirs(os.path.join(os.path.join(os.getcwd(), fl_ctx.get_job_id()), "saved_images_ae"), exist_ok=True)

        self.model.autoencoder.train()
        self.model.discriminator.train()
        val_loss_total = []
        nan_signal = False

        for epoch in range(self.params_autoencoder["epochs"]):
            if nan_signal:
                if abort_signal is not None:
                    abort_signal.trigger()
                break

            train_g_loss = train_d_loss = 0.0
            train_g_l1loss = train_g_percloss = train_g_ganloss = train_g_klloss = 0.0
            batch_acc_counter = 0

            for ind, batch in enumerate(self._train_loader):
                if abort_signal is not None and abort_signal.triggered:
                    return

                images = batch["image"].to(self.device)
                perceptual_slices = None

                with autocast(enabled=True, device_type=self.device.type):
                    reconstruction, z_mu, z_sigma = self.model.autoencoder(images)
                    if torch.isnan(reconstruction).any():
                        abort_signal.trigger() if abort_signal else None
                        nan_signal = True
                        break

                    kl_loss = self.weights_ae["w_kl_loss"] * self.losses_ae["kld_loss"](z_mu, z_sigma)
                    l1_loss = self.weights_ae["w_reconstruction_loss"] * self.losses_ae["reconstruction_loss"](
                        reconstruction.float(), images.float()
                    )
                    if self.axial_anisotropy or self.config["net_config"]["stage_1"]["spatial_dims"] == 2:
                        perceptual_slices = np.random.randint(0, images.shape[-1], size=self.perceptual_slices)
                        p_loss = self.weights_ae["w_perceptual_loss"] * self.losses_ae["perceptual_loss"](
                            einops.rearrange(
                                reconstruction[..., perceptual_slices], "b c h w d -> (b d) c h w"
                            ).float(),
                            einops.rearrange(images[..., perceptual_slices], "b c h w d -> (b d) c h w").float(),
                        )
                    else:
                        p_loss = self.weights_ae["w_perceptual_loss"] * self.losses_ae["perceptual_loss"](
                            reconstruction.float(), images.float()
                        )

                    if (
                        self.config["net_config"]["stage_1"]["spatial_dims"] == 3
                        and self.config["net_config"]["discriminator"]["spatial_dims"] == 2
                    ):
                        if perceptual_slices is None:
                            perceptual_slices = np.random.randint(0, images.shape[-1], size=self.perceptual_slices)
                        logits_fake = self.model.discriminator(
                            einops.rearrange(reconstruction[..., perceptual_slices], "b c h w d -> (b d) c h w")
                            .contiguous()
                            .float()
                        )[-1]
                    else:
                        logits_fake = self.model.discriminator(reconstruction.contiguous().float())[-1]

                    gan_loss = self.weights_ae["w_gan_loss"] * self.losses_ae["gan_loss"](
                        logits_fake, target_is_real=True, for_discriminator=False
                    )
                    train_g_loss_ = l1_loss + kl_loss + p_loss + gan_loss

                if nan_signal:
                    break

                scaler_g.scale(train_g_loss_).backward()
                batch_acc_counter += 1
                if batch_acc_counter % self.batch_accumulation_step == 0 or ind == len(self._train_loader) - 1:
                    scaler_g.step(self.optimizers_ae["optimizer_g"])
                    scaler_g.update()
                    self.optimizers_ae["optimizer_g"].zero_grad(set_to_none=True)

                del z_mu, z_sigma, logits_fake

                self.optimizers_ae["optimizer_d"].zero_grad(set_to_none=True)
                if (
                    self.config["net_config"]["stage_1"]["spatial_dims"] == 3
                    and self.config["net_config"]["discriminator"]["spatial_dims"] == 2
                ):
                    if perceptual_slices is None:
                        perceptual_slices = np.random.randint(0, images.shape[-1], size=self.perceptual_slices)
                    logits_fake = self.model.discriminator(
                        einops.rearrange(reconstruction.detach()[..., perceptual_slices], "b c h w d -> (b d) c h w")
                        .contiguous()
                        .float()
                    )[-1]
                    logits_real = self.model.discriminator(
                        einops.rearrange(images[..., perceptual_slices], "b c h w d -> (b d) c h w")
                        .contiguous()
                        .float()
                    )[-1]
                else:
                    logits_real = self.model.discriminator(images.float())[-1]
                    logits_fake = self.model.discriminator(reconstruction.detach().contiguous().float())[-1]

                d_loss = (
                    self.weights_ae["w_gan_loss"]
                    * (
                        self.losses_ae["gan_loss"](logits_fake, target_is_real=False, for_discriminator=True)
                        + self.losses_ae["gan_loss"](logits_real, target_is_real=True, for_discriminator=True)
                    )
                    * 0.5
                )
                scaler_d.scale(d_loss).backward()
                if batch_acc_counter % self.batch_accumulation_step == 0 or ind == len(self._train_loader) - 1:
                    scaler_d.step(self.optimizers_ae["optimizer_d"])
                    scaler_d.update()

                del logits_real, logits_fake, reconstruction
                train_g_loss += train_g_loss_.item()
                train_g_ganloss += gan_loss.item()
                train_g_percloss += p_loss.item()
                train_g_l1loss += l1_loss.item()
                train_g_klloss += kl_loss.item()
                train_d_loss += d_loss.item()
                del train_g_loss_, gan_loss, p_loss, l1_loss, kl_loss, d_loss, images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            n = max(1, len(self._train_loader))
            train_g_loss /= n
            train_d_loss /= n

            val_loss = 0.0
            self.model.autoencoder.eval()
            for _, batch in enumerate(self._val_loader):
                if abort_signal is not None and abort_signal.triggered:
                    return
                images = batch["image"].to(self.device)
                with autocast(enabled=True, device_type=self.device.type):
                    with torch.no_grad():
                        reconstruction, _, _ = self.model.autoencoder(images)
                    if torch.isnan(reconstruction).any():
                        break
                    val_loss += (
                        self.weights_ae["w_reconstruction_loss"]
                        * self.losses_ae["reconstruction_loss"](reconstruction.float(), images.float()).item()
                    )
            val_loss /= max(1, len(self._val_loader))
            val_loss_total.append(val_loss)
            self.model.autoencoder.train()

            if fl_ctx is not None:
                send_metrics_value(label="Train loss (G)", value=train_g_loss, fl_ctx=fl_ctx, flip=self.flip)
                send_metrics_value(label="Train loss (D)", value=train_d_loss, fl_ctx=fl_ctx, flip=self.flip)
                send_metrics_value(label="Validation loss (L1)", value=val_loss, fl_ctx=fl_ctx, flip=self.flip)

        return float(np.mean(val_loss_total)) if val_loss_total else 0.0

    # ------------------------------------------------------------------
    # DM training loop (unchanged logic from legacy executor)
    # ------------------------------------------------------------------

    def local_train_dm(self, fl_ctx, weights, abort_signal, global_round: int = 0):
        self.config_batch_accumulation(phase="BATCH_SIZE_DM")
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
                    next(iter(self._train_loader))["image"].to(self.device)
                )
                scale_factor = 1 / torch.std(sample_z)
                del sample_z

        inferer = LatentDiffusionInferer(
            scheduler=self.optimizers_dm["scheduler"],
            ldm_latent_shape=ldm_latent_shape,
            autoencoder_latent_shape=autoencoder_latent_shape,
            scale_factor=scale_factor.item(),
        )
        scaler = GradScaler()
        nan_signal = False
        train_loss = []
        val_loss = []

        for epoch in range(self.params_diffusion["epochs"]):
            self.model.diffusion_model.train()
            if nan_signal:
                if abort_signal is not None:
                    abort_signal.trigger()
                break

            batch_acc_counter = 0
            train_loss_epoch = 0.0

            for ind, batch in enumerate(self._train_loader):
                if abort_signal is not None and abort_signal.triggered:
                    return

                images = batch["image"].to(self.device)
                with autocast(enabled=False, device_type=self.device.type):
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
                    loss = self.losses_dm["loss"](noise.float(), noise_pred.float())

                if torch.isnan(loss).any() or torch.isnan(noise_pred).any():
                    if abort_signal is not None:
                        abort_signal.trigger()
                    nan_signal = True
                    break

                scaler.scale(loss).backward()
                batch_acc_counter += 1
                if batch_acc_counter == self.batch_accumulation_step:
                    scaler.step(self.optimizers_dm["optimizer"])
                    scaler.update()
                    batch_acc_counter = 0
                    self.optimizers_dm["optimizer"].zero_grad(set_to_none=True)
                train_loss_epoch += loss.item()

            train_loss.append(train_loss_epoch / max(1, len(self._train_loader)))

            val_loss_epoch = 0.0
            for _, batch in enumerate(self._val_loader):
                if abort_signal is not None and abort_signal.triggered:
                    return
                images = batch["image"].to(self.device)
                self.model.diffusion_model.eval()
                with autocast(enabled=False, device_type=self.device.type):
                    with torch.no_grad():
                        noise = torch.randn(
                            [images.shape[0]]
                            + [self.model.autoencoder.encoder.blocks[-1].out_channels]
                            + ldm_latent_shape
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
                    val_loss_epoch += self.losses_dm["loss"](noise.float(), noise_pred.float()).item()
            val_loss.append(val_loss_epoch / max(1, len(self._val_loader)))

            if fl_ctx is not None:
                send_metrics_value(label="Total loss DM", value=train_loss[-1], fl_ctx=fl_ctx, flip=self.flip)
                send_metrics_value(label="Validation loss DM", value=val_loss[-1], fl_ctx=fl_ctx, flip=self.flip)

        return float(np.mean(val_loss)) if val_loss else 0.0
