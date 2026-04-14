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
#

"""
FLIPDataset — abstract MONAI Dataset adaptor for FLIP platform data.

Subclasses implement ``_build_datalist`` to produce the domain-specific list of
sample dicts from the FLIP project dataframe.  The resulting list is passed to
:class:`monai.data.Dataset`, making every FLIP-backed dataset a drop-in
replacement wherever a MONAI Dataset is expected.

FLIP data resolution happens once during ``__init__``: the FLIP query is issued,
accession paths are resolved, and the datalist is built before the first
training iteration.

Example usage (Option A — MONAI Bundle)::

    class SpleenFLIPDataset(FLIPDataset):
        def _build_datalist(self, dataframe):
            datalist = []
            for accession_id in dataframe["accession_id"]:
                folder = self.flip.get_by_accession_number(
                    self.project_id, accession_id, resource_type=[ResourceType.NIFTI]
                )
                for img in folder.rglob("input_*.nii.gz"):
                    seg = str(img).replace("/input_", "/label_")
                    if Path(seg).exists():
                        datalist.append({"image": str(img), "label": seg})
            return datalist

    dataset = SpleenFLIPDataset(
        flip=FLIP(),
        project_id="my-project",
        query="SELECT * FROM table",
        transform=get_train_transforms(),
    )

The class can also be declared in a MONAI Bundle ``configs/train.json`` via
``_target_`` so that data loading is fully declarative::

    "train_dataset": {
        "_target_": "custom.flip_dataset.SpleenFLIPDataset",
        "flip": "$FLIP()",
        "project_id": "@project_id",
        "query": "@query",
        "transform": "@train_transforms"
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import pandas as pd
from monai.data import Dataset


class FLIPDataset(Dataset, ABC):
    """Abstract base class for MONAI Datasets backed by the FLIP platform API.

    Subclasses must implement :meth:`_build_datalist` which receives the
    DataFrame returned by ``flip.get_dataframe()`` and returns a list of
    sample dicts (e.g. ``[{"image": "/path/img.nii.gz", "label": "/path/seg.nii.gz"}]``).

    The datalist is built once at construction time and stored in ``self.data``
    (the standard :class:`monai.data.Dataset` attribute).  Modifying
    ``self.data`` in-place after construction is safe — DataLoader iterators
    are created fresh each epoch and read ``len(self.data)`` at iterator
    creation time.

    Args:
        flip: An initialised FLIP instance (``FLIPStandardProd`` or
            ``FLIPStandardDev``) used for dataframe retrieval and accession
            path resolution.
        project_id: The FLIP project identifier.
        query: SQL query string passed to ``flip.get_dataframe()``.
        transform: Optional MONAI transform pipeline applied to each sample.
    """

    def __init__(
        self,
        flip,
        project_id: str,
        query: str,
        transform: Callable | None = None,
    ) -> None:
        self.flip = flip
        self.project_id = project_id
        self.query = query
        dataframe: pd.DataFrame = flip.get_dataframe(project_id, query)
        datalist: Sequence = self._build_datalist(dataframe)
        super().__init__(data=datalist, transform=transform)

    @abstractmethod
    def _build_datalist(self, dataframe: pd.DataFrame) -> Sequence:
        """Build the list of sample dicts from the FLIP project dataframe.

        Each entry must be a dict with at least an ``"image"`` key whose value
        is the absolute path to the image file.  Additional keys (e.g.
        ``"label"``) are domain-specific.

        Args:
            dataframe: DataFrame returned by ``flip.get_dataframe()``; contains
                at minimum an ``accession_id`` column.

        Returns:
            A sequence of sample dicts suitable for passing to a MONAI transform
            pipeline.
        """
        ...
