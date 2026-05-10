import os
import random
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import yaml


@dataclass
class EarlyStopping:
    patience: int = 8
    min_delta: float = 0.0

    def __post_init__(self):
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
