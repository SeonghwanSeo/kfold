from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from kfold.data import model_input
from kfold.data.metadata import Metadata
from kfold.utils.boltz.process import parse_structure
from kfold.utils.boltz.structure import BoltzStructure


class BaseDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> model_input.FoldingInput: ...


class SafeIterDataset(BaseDataset):
    def __getitem__(self, index: int) -> model_input.FoldingInput:
        return self.get_item_safe(index)

    def get_item_safe(self, index: int) -> model_input.FoldingInput:
        try_indexes = []
        for _ in range(10):
            try:
                return self.get_item(index)
            except Exception as e:
                print(f"Error loading index {index}: {e}. Retrying...")
                index = np.random.randint(0, len(self))
                try_indexes.append(index)
        else:
            raise RuntimeError(
                f"Failed to load data after 10 attempts. Tried indexes: {try_indexes}"
            )

    @abstractmethod
    def get_item(self, index: int) -> model_input.FoldingInput: ...


class BoltzDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        boltz_processed_dir: str | Path,
        keys: list[str],
        metadatas: dict[str, Metadata],
    ):
        self.boltz_processed_dir: Path = Path(boltz_processed_dir)
        self.keys: list[str] = keys
        self.metadatas: dict[str, Metadata] = metadatas

    def __len__(self) -> int:
        return len(self.keys)

    def get_item(self, index: int) -> model_input.FoldingInput:
        name = self.keys[index]

        # Load BoltzStructure
        path = self.boltz_processed_dir / f"{name}.npz"
        boltz_structure = BoltzStructure.load(path)

        chains = boltz_structure.chains[boltz_structure.mask]
        folding_input = parse_structure(chains, boltz_structure)

        return folding_input
