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
FLIPDataset subclasses for 3D spleen segmentation.

``SpleenTrainFLIPDataset`` and ``SpleenValFLIPDataset`` encapsulate all FLIP
data resolution and quality-control logic for the spleen NIfTI / segmentation
data, including:

- Calling ``flip.get_by_accession_number()`` per accession ID
- Matching ``input_*.nii.gz`` images to ``label_*.nii.gz`` segmentations
- Performing 3D shape-consistency QC on each pair
- Splitting into train / val subsets based on ``val_split``

These classes are consumed by the trainer and validator without either needing
to implement data-loading logic directly.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib

from flip.constants import ResourceType
from flip.utils import FLIPDataset


def _build_spleen_datalist(flip, dataframe, project_id: str) -> list[dict]:
    """Build the full QC-passed list of ``{image, label}`` dicts.

    Iterates over every accession ID in *dataframe*, resolves the NIfTI folder
    via the FLIP API, globs for ``input_*.nii.gz`` files and matches them to
    ``label_*.nii.gz`` segmentations.  Each pair is validated for:

    - 3-dimensionality
    - Shape consistency between image and segmentation

    Args:
        flip: An initialised FLIP instance.
        dataframe: DataFrame returned by ``flip.get_dataframe()``.
        project_id: FLIP project identifier.

    Returns:
        List of ``{"image": str, "label": str}`` dicts.
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
    """Training-split NIfTI + segmentation dataset for 3D spleen segmentation.

    Calls ``_build_spleen_datalist`` and returns only the first
    ``(1 - val_split)`` fraction of matched pairs as training samples.

    Args:
        flip: An initialised FLIP instance.
        project_id: FLIP project identifier.
        query: SQL query passed to ``flip.get_dataframe()``.
        val_split: Fraction of data reserved for validation (default 0.1).
        transform: Optional MONAI transform pipeline.
    """

    def __init__(
        self,
        flip,
        project_id: str,
        query: str,
        val_split: float = 0.1,
        transform=None,
    ) -> None:
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
    """Validation-split NIfTI + segmentation dataset for 3D spleen segmentation.

    Calls ``_build_spleen_datalist`` and returns only the last ``val_split``
    fraction of matched pairs as validation samples.

    Args:
        flip: An initialised FLIP instance.
        project_id: FLIP project identifier.
        query: SQL query passed to ``flip.get_dataframe()``.
        val_split: Fraction of data used for validation (default 0.1).
        transform: Optional MONAI transform pipeline.
    """

    def __init__(
        self,
        flip,
        project_id: str,
        query: str,
        val_split: float = 0.1,
        transform=None,
    ) -> None:
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
