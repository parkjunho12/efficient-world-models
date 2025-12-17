"""Base dataset class."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List

class BaseDataset(Dataset):
    """Base class for driving datasets."""
    
    def __init__(self, data_root, sequence_length=10, split='train'):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.split = split
        self.sequences = self._load_sequences()
    
    def _load_sequences(self) -> List[Dict]:
        """Override this in subclasses."""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Override this in subclasses."""
        raise NotImplementedError