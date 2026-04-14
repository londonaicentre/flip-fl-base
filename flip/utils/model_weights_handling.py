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

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
from nvflare.apis.dxo import DXO, DataKind, MetaKey

if TYPE_CHECKING:
    pass


def _to_numpy_array(value):
    """Convert a tensor/array-like weight value to a detached numpy array."""

    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_model_weights_diff(original_weights: OrderedDict, new_weights: OrderedDict, iterations: int) -> DXO:
    """Compute the weights differences to send a weight update.

    Args:
        original_weights (OrderedDict): weights coming from the server (before training)
        new_weights (OrderedDict): weights coming out of the server (post training)
        iterations (int): number of iterations

    Returns:
        DXO: DXO containing the weight updates for the server.
    """

    weight_diff = {}
    for k in new_weights.keys():
        original = _to_numpy_array(original_weights[k])
        updated = _to_numpy_array(new_weights[k])
        weight_diff[k] = updated - original

    outgoing_dxo = DXO(
        data_kind=DataKind.WEIGHT_DIFF,
        data=weight_diff,
        meta={MetaKey.NUM_STEPS_CURRENT_ROUND: iterations},
    )

    return outgoing_dxo
