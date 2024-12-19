"""Dataset for FuteFWI. Efficiency is limited for insufficient RAM."""

import os
import numpy as np
import torch
from typing import List, Tuple


class Dataset(torch.utils.data.Dataset):
    """
    Dataset for OpenFWI.
    """

    def __init__(self, root_dir: str, fid_list: List[int]):
        """
        Initialize dataset.

        Args:
            root_dir: root directory.
            fid_list: list of npy file id.
        """
        super().__init__()
        data_files = sorted(os.listdir(os.path.join(root_dir, "data")))
        model_files = sorted(os.listdir(os.path.join(root_dir, "model")))

        self.dataset = []
        for idx in fid_list:
            data = np.load(os.path.join(root_dir, "data", data_files[idx]))
            self.dataset.extend(data)

        self.labelset = np.concatenate([np.load(os.path.join(root_dir, "model", model_files[fid])) for fid in fid_list])
        self.labelset = self._normalize()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.dataset[idx], self.labelset[idx]

    def _normalize(self) -> np.ndarray:
        min_vals = np.min(self.labelset, axis=(1, 2, 3), keepdims=True)
        max_vals = np.max(self.labelset, axis=(1, 2, 3), keepdims=True)
        return ((self.labelset - min_vals) / (max_vals - min_vals) - 0.5) * 2


class LargeDataset(torch.utils.data.Dataset):
    """
    Dataset for OpenFWI with memory-mapped data loading.
    """

    def __init__(self, root_dir: str, fid_list: List[int]):
        """
        Initialize dataset.

        Args:
            root_dir: root directory.
            fid_list: list of npy file id.
        """
        super().__init__()
        data_files = sorted(os.listdir(os.path.join(root_dir, "data")))
        model_files = sorted(os.listdir(os.path.join(root_dir, "model")))

        self.dataset = [np.load(os.path.join(root_dir, "data", data_files[fid]), mmap_mode="r+") for fid in fid_list]
        self.labelset = np.concatenate([np.load(os.path.join(root_dir, "model", model_files[fid])) for fid in fid_list])

        # build index
        self.index = self._build_index()

        # Normalization
        self.labelset = self._normalize()

    def __len__(self) -> int:
        return len(self.labelset)

    def _normalize(self) -> np.ndarray:
        min_vals = np.min(self.labelset, axis=(1, 2, 3), keepdims=True)
        max_vals = np.max(self.labelset, axis=(1, 2, 3), keepdims=True)
        return ((self.labelset - min_vals) / (max_vals - min_vals) - 0.5) * 2

    def _build_index(self) -> List[tuple]:
        index = []
        for i, data in enumerate(self.dataset):
            for j in range(data.shape[0]):
                index.append((i, j))
        return index

    def __getitem__(self, idx: int):
        dataset_idx, sample_idx = self.index[idx]
        return self.dataset[dataset_idx][sample_idx], self.labelset[idx]


class TestDataset(torch.utils.data.Dataset):
    """
    Dataset used in test program.
    """

    def __init__(self, root_dir: str, fid_list: List[int]):
        """
        Initialize dataset.

        Args:
            root_dir: root directory.
            fid_list: list of npy file id.
            gaussian_noise: variance of Gaussian noise, default: None.
        """
        super().__init__()
        data_files = sorted(os.listdir(os.path.join(root_dir, "data")))
        model_files = sorted(os.listdir(os.path.join(root_dir, "model")))

        self.dataset = []
        for idx in fid_list:
            data = np.load(os.path.join(root_dir, "data", data_files[idx]))
            self.dataset.extend(data)

        self.labelset = np.concatenate([np.load(os.path.join(root_dir, "model", model_files[fid])) for fid in fid_list])
        self.normalized_labelset = self._normalize()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.dataset[idx], self.normalized_labelset[idx], self.labelset[idx]

    def _normalize(self) -> np.ndarray:
        min_vals = np.min(self.labelset, axis=(1, 2, 3), keepdims=True)
        max_vals = np.max(self.labelset, axis=(1, 2, 3), keepdims=True)
        return ((self.labelset - min_vals) / (max_vals - min_vals) - 0.5) * 2
