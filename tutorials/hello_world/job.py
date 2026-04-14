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
Hello World job recipe — NVFlare 2.7+ Recipe API style.

Wires together the FedAvg server-side algorithm and the client training script
using the high-level Recipe API introduced in NVFlare 2.7 and made generally
available in 2.7.2.

Run from this directory:

    python job.py

The simulation runs entirely in-process via SimEnv (no Docker, no provisioning).
Results are written to /tmp/nvflare/simulation/hello-world/.
"""

import os
import sys

# Ensure model.py (in this directory) is importable when job.py is invoked
# from a different working directory (e.g. via `make test-hello`).
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import SimpleNetwork  # noqa: E402
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe  # noqa: E402
from nvflare.recipe import SimEnv  # noqa: E402

# ── Hyper-parameters ──────────────────────────────────────────────────────────
n_clients = 2
num_rounds = 2
batch_size = 32
local_epochs = 1

# ── Recipe ────────────────────────────────────────────────────────────────────
recipe = FedAvgRecipe(
    name="hello-world",
    min_clients=n_clients,
    num_rounds=num_rounds,
    # The initial global model — converted to config before job submission.
    # For large models prefer the dict-config form to avoid unnecessary
    # instantiation overhead:
    #   initial_model={"class_path": "model.SimpleNetwork", "args": {}}
    initial_model=SimpleNetwork(),
    train_script=os.path.join(HERE, "client.py"),
    train_args=f"--batch_size {batch_size} --epochs {local_epochs}",
)

# model.py must also be available on each simulated client so that client.py
# can run `from model import SimpleNetwork`.
recipe.job.to_clients(os.path.join(HERE, "model.py"))

# ── Execution environment ─────────────────────────────────────────────────────
env = SimEnv(
    num_clients=n_clients,
    num_threads=n_clients,  # run each client in its own thread
)

recipe.execute(env=env)
