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

import pandas as pd
import pytest

from flip.utils.flip_dataset import FLIPDataset


class ConcreteFLIPDataset(FLIPDataset):
    """Minimal concrete subclass used for testing."""

    def _build_datalist(self, dataframe: pd.DataFrame):
        return [{"image": f"/data/{row['accession_id']}.nii.gz"} for _, row in dataframe.iterrows()]


class TestFLIPDataset:
    def _make_flip(self, rows=None):
        """Return a mock FLIP object whose get_dataframe returns the given rows."""
        from unittest.mock import MagicMock

        if rows is None:
            rows = [{"accession_id": "ACC001"}, {"accession_id": "ACC002"}]
        dataframe = pd.DataFrame(rows)
        flip = MagicMock()
        flip.get_dataframe.return_value = dataframe
        return flip

    def test_init_calls_get_dataframe(self):
        """FLIPDataset.__init__ calls flip.get_dataframe with project_id and query."""
        flip = self._make_flip()
        ConcreteFLIPDataset(flip=flip, project_id="proj", query="SELECT 1")
        flip.get_dataframe.assert_called_once_with("proj", "SELECT 1")

    def test_datalist_length_matches_rows(self):
        """Dataset length equals the number of accession rows returned."""
        flip = self._make_flip([{"accession_id": f"ACC{i:03d}"} for i in range(5)])
        ds = ConcreteFLIPDataset(flip=flip, project_id="proj", query="q")
        assert len(ds) == 5

    def test_item_keys_are_present(self):
        """Each sample dict contains an 'image' key built from the accession ID."""
        flip = self._make_flip([{"accession_id": "ACC001"}])
        ds = ConcreteFLIPDataset(flip=flip, project_id="proj", query="q")
        # Dataset without a transform returns the raw dict
        item = ds[0]
        assert "image" in item
        assert "ACC001" in item["image"]

    def test_in_place_extension_visible_to_dataset(self):
        """Extending data list in-place after construction is reflected in __len__."""
        flip = self._make_flip([{"accession_id": "ACC001"}])
        ds = ConcreteFLIPDataset(flip=flip, project_id="proj", query="q")
        assert len(ds) == 1
        # Simulate MonaiAlgo-style injection: extend the underlying list in-place
        ds.data.append({"image": "/data/ACC002.nii.gz"})
        assert len(ds) == 2

    def test_cannot_instantiate_abstract(self):
        """FLIPDataset cannot be instantiated directly (abstract method)."""
        flip = self._make_flip()
        with pytest.raises(TypeError):
            FLIPDataset(flip=flip, project_id="proj", query="q")

    def test_transform_applied_at_getitem(self):
        """Transform pipeline is applied when indexing into the dataset."""
        flip = self._make_flip([{"accession_id": "ACC001"}])

        def upper_transform(sample):
            return {k: v.upper() for k, v in sample.items()}

        ds = ConcreteFLIPDataset(flip=flip, project_id="proj", query="q", transform=upper_transform)
        item = ds[0]
        assert item["image"] == item["image"].upper()
