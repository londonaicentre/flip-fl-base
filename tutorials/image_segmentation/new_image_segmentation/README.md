# New Image Segmentation (Migrated Client API + Recipe)

This tutorial is the migrated MONAI image segmentation pattern based on the NVFlare migration guide:
<https://github.com/NVIDIA/NVFlare/blob/main/integration/monai/MIGRATION.md>

It uses the modern NVFlare stack:

- `nvflare.client` Client API
- Bundle-native `FLIP_TRAINER` (`ClientAlgo`) for MONAI training
- `FedAvgRecipe` + `SimEnv` for local simulation

This tutorial expects local data env vars (`DEV_IMAGES_DIR`, `DEV_DATAFRAME`, `RUNS_DIR`).
The provided Make target sets these automatically.

## Bundle Source

By default, this tutorial reuses the existing spleen bundle assets from:
`tutorials/image_segmentation/3d_spleen_segmentation/app_files`

## Run Directly

From repository root:

```bash
.venv/bin/python tutorials/image_segmentation/new_image_segmentation/job.py \
  --bundle_root tutorials/image_segmentation/3d_spleen_segmentation/app_files \
  --num_rounds 1 --local_epochs 1 --n_clients 2
```

## Run via Make

From repository root:

```bash
make test-spleen-new-image-segmentation
```
