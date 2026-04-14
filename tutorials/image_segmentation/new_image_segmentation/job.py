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

"""MONAI image segmentation on NVFlare Client API + FedAvgRecipe."""

import argparse
import os
import sys

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def parse_args():
    """Parse command line options for the local simulation recipe."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle_root",
        type=str,
        default="../3d_spleen_segmentation/app_files",
        help="Path to MONAI bundle relative to this job directory",
    )
    parser.add_argument("--n_clients", type=int, default=2, help="Number of simulated clients")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of FL rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local epochs per FL round")
    parser.add_argument("--threads", type=int, default=2, help="Number of parallel threads")
    parser.add_argument("--workspace", type=str, default="/tmp/nvflare/simulation", help="Simulation workspace")
    parser.add_argument("--send_weight_diff", action="store_true", help="Send weight diffs instead of full weights")
    parser.add_argument(
        "--tracking",
        type=str,
        default="none",
        choices=["tensorboard", "mlflow", "both", "none"],
        help="Experiment tracking backend",
    )
    return parser.parse_args()


def main():
    """Create and execute the migrated FedAvg simulation job."""

    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    bundle_arg = args.bundle_root
    if os.path.isabs(bundle_arg):
        bundle_path = bundle_arg
    else:
        cwd_candidate = os.path.abspath(bundle_arg)
        script_candidate = os.path.abspath(os.path.join(here, bundle_arg))
        bundle_path = cwd_candidate if os.path.isdir(cwd_candidate) else script_candidate

    if not os.path.isdir(bundle_path):
        raise FileNotFoundError(f"Bundle root not found: {bundle_path}")

    if bundle_path not in sys.path:
        sys.path.insert(0, bundle_path)

    from models import get_model

    train_args = f"--bundle_root {bundle_path} --local_epochs {args.local_epochs}"
    if args.send_weight_diff:
        train_args += " --send_weight_diff"

    recipe = FedAvgRecipe(
        name="spleen_bundle_fedavg_migrated",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        initial_model=get_model(),
        train_script=os.path.join(here, "client.py"),
        train_args=train_args,
        aggregator_data_kind=DataKind.WEIGHT_DIFF if args.send_weight_diff else DataKind.WEIGHTS,
    )

    recipe.job.to_server(os.path.join(bundle_path, "config.json"))

    if args.tracking in ["tensorboard", "both"]:
        add_experiment_tracking(recipe, tracking_type="tensorboard")
    if args.tracking in ["mlflow", "both"]:
        add_experiment_tracking(recipe, tracking_type="mlflow")

    env = SimEnv(num_clients=args.n_clients, num_threads=args.threads, workspace_root=args.workspace)
    run = recipe.execute(env)

    print()
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
    print()


if __name__ == "__main__":
    main()
