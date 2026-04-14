# Hello World — Federated Learning with FLIP & NVFlare

This tutorial demonstrates federated averaging (FedAvg) on CIFAR-10 using the
NVFlare 2.7+ **Recipe API** (`FedAvgRecipe` + `SimEnv`). It is the simplest
possible end-to-end federated learning example and serves as a template for
building FLIP-compatible FL jobs.

## Code structure

```bash
hello_world/
├── model.py          # SimpleNetwork — a small CNN for CIFAR-10
├── client.py         # Client training script (NVFlare Client API)
├── job.py            # Job recipe — server orchestration (Recipe API)
└── requirements.txt  # Standalone pip dependencies
```

## Running the simulation

No Docker, no provisioning, and no external data required. From the repo root:

```bash
make test-hello
```

This runs `python tutorials/hello_world/job.py` inside the project virtual
environment. On the first run CIFAR-10 (~170 MB) is downloaded to
`/tmp/data/cifar10` automatically.

Expected output (2 rounds, 2 simulated clients):

```bash
Round 0 started.
[Client] Round 0 — starting local training
[Client] Epoch 1/1  loss=2.24
Round 0 finished.
Round 1 started.
[Client] Round 1 — starting local training
[Client] Epoch 1/1  loss=1.87
Round 1 finished.
Finished ScatterAndGather Training.
```

## Viewing training metrics in TensorBoard

NVFlare automatically writes TensorBoard event files to:

```bash
/tmp/nvflare/simulation/hello-world/server/simulate_job/tb_events/
```

After (or while) `make test-hello` runs, launch TensorBoard in a second
terminal:

```bash
source .venv/bin/activate.fish   # activate the project venv (fish shell)
tensorboard --logdir /tmp/nvflare/simulation/hello-world/server/simulate_job/tb_events
```

Or if you prefer a bash/zsh shell:

```bash
source .venv/bin/activate
tensorboard --logdir /tmp/nvflare/simulation/hello-world/server/simulate_job/tb_events
```

Then open <http://localhost:6006> in your browser. You will see the **global
model loss** curve logged by the server after each aggregation round.

> **Tip:** The logs from previous runs accumulate in the same directory. Delete
> `/tmp/nvflare/simulation/hello-world/` before re-running if you want a clean
> TensorBoard view:
>
> ```bash
> rm -rf /tmp/nvflare/simulation/hello-world/
> make test-hello
> ```

## Adapting to a real FLIP job

The `client.py` script contains a comment `# FLIP data` marking the
`load_data()` function. In a production FLIP job replace that function with
calls to the FLIP API:

```python
from flip import FLIP
from flip.constants import ResourceType

flip = FLIP()

# 1. Fetch the cohort dataframe from the trust OMOP database
df = flip.get_dataframe(project_id, sql_query)

# 2. Download imaging data for each accession
for accession_id in df["accession_id"]:
    path = flip.get_by_accession_number(
        project_id, accession_id, resource_type=ResourceType.DICOM
    )
    # build your dataset from `path` ...
```

Everything else — model definition, optimiser, loss, training loop, and all
NVFlare Client API calls — remains identical.
