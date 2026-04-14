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

"""Client API training script for MONAI bundle based segmentation."""

import argparse
import os
import sys

import nvflare.client as flare
import torch
from monai.fl.utils.constants import WeightType
from monai.fl.utils.exchange_object import ExchangeObject


def main():
    """Run one NVFlare client process backed by the existing bundle trainer."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", type=str, required=True, help="Path to MONAI bundle")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local epochs per FL round")
    parser.add_argument("--send_weight_diff", action="store_true", help="Send weight diffs instead of full weights")
    args = parser.parse_args()

    bundle_root = os.path.abspath(args.bundle_root)
    if bundle_root not in sys.path:
        sys.path.insert(0, bundle_root)

    from trainer import FLIP_TRAINER

    flare.init()

    algo = FLIP_TRAINER()
    algo.initialize(extra={"CLIENT_NAME": flare.get_site_name()})
    if hasattr(algo, "_local_rounds"):
        algo._local_rounds = args.local_epochs

    while flare.is_running():
        input_model = flare.receive()
        global_weights = ExchangeObject(weights=input_model.params)

        algo.train(data=global_weights, extra={})
        expected_weight_type = WeightType.WEIGHT_DIFF if args.send_weight_diff else WeightType.WEIGHTS
        updated_weights = algo.get_weights(extra={"weight_type": expected_weight_type})

        statistics = updated_weights.statistics or {}
        executed_steps = statistics.get("num_steps", 0)

        # Ensure aggregation receives tensor values regardless of trainer return type.
        tensor_params = {k: torch.as_tensor(v) for k, v in updated_weights.weights.items()}

        output_model = flare.FLModel(
            params=tensor_params,
            metrics=statistics,
            meta={
                "weight_type": updated_weights.weight_type.value if updated_weights.weight_type else "WEIGHTS",
                "NUM_STEPS_CURRENT_ROUND": executed_steps,
            },
        )
        flare.send(output_model)

    algo.finalize()


if __name__ == "__main__":
    main()
