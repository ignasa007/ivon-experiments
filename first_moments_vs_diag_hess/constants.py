import os

import torch

DATASTORE = f"{os.path.dirname(__file__)}/../datasets"
ASSETS = f"{os.path.dirname(__file__)}/assets"
SUBSET_SIZE = 5000
CKPTS = 10
HUTCHINSON_SAMPLES = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0