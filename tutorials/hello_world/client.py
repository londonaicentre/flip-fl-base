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
Hello World client training script — FLIP / NVFlare 2.7+ Client API style.

This script is executed on each simulated FL client.  The only NVFlare-specific
additions compared to ordinary centralised PyTorch training are the five lines
marked with "#  NVFlare".

In a real FLIP deployment the data-loading section (marked "# FLIP data") would
be replaced with calls to the FLIP API:

    from flip import FLIP
    flip = FLIP()
    df   = flip.get_dataframe(project_id, sql_query)
    path = flip.get_by_accession_number(project_id, accession_id)

Everything else — model, optimiser, loss, training loop — stays the same.
"""

import argparse

import nvflare.client as flare  # NVFlare
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SimpleNetwork
from nvflare.apis.fl_constant import FLMetaKey  # NVFlare — metadata constants

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data/cifar10"


def load_data(batch_size: int):
    """Download (first run only) and return a CIFAR-10 DataLoader.

    In production FLIP jobs replace this with FLIP.get_dataframe() /
    FLIP.get_by_accession_number() to access trust-held imaging data.
    """  # FLIP data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


def train(net, trainloader, epochs: int):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        print(f"[Client] Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    return net.state_dict(), len(trainloader.dataset)


def main():
    parser = argparse.ArgumentParser(description="FLIP hello-world client")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    args = parser.parse_args()

    trainloader = load_data(args.batch_size)

    net = SimpleNetwork().to(DEVICE)

    flare.init()  # NVFlare — initialise Client API

    while flare.is_running():  # NVFlare — loop until server signals completion
        input_model = flare.receive()  # NVFlare — receive global model from server
        print(f"[Client] Round {input_model.current_round} — starting local training")

        net.load_state_dict(input_model.params)  # load global weights

        new_params, num_steps = train(net, trainloader, args.epochs)

        output_model = flare.FLModel(  # NVFlare — wrap updated weights
            params=new_params,
            # num_steps is passed so the server-side aggregator can weight each
            # client's contribution by its dataset size (weighted FedAvg).
            meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: num_steps},
        )
        flare.send(output_model)  # NVFlare — send updated model back to server


if __name__ == "__main__":
    main()
