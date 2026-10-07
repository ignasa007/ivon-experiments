import os

import torch


DATASTORE = f"{os.environ.get("HOME")}/ivon-experiments/datasets"
ASSETS = f"{os.environ.get("HOME")}/ivon-experiments/assets"
SUBSET_SIZE = 5000
CKPTS = 10
HUTCHINSON_SAMPLES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0