import os
from src.config import DATASET_CACHE_DIR
import numpy as np
from torch.utils.data import DataLoader

class DiskDataset:
    def __init__(self, id, cache_id, samples, labels):
        self.id = id                     # original WHARDatasetID
        self.cache_id = str(cache_id)    # e.g. "<id>_train"
        self.samples = samples
        self.labels = labels

    @classmethod
    def from_dataloader(cls, id, cache_id, dataloader: DataLoader):
        all_samples = []
        all_labels = []

        for labels, samples in dataloader:
            all_samples.append(samples.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        samples_array = np.concatenate(all_samples, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)
        return cls(id, cache_id, samples_array, labels_array)

    @classmethod
    def from_disk(cls, id, cache_id):
        cache_id = str(cache_id)
        path = os.path.join(DATASET_CACHE_DIR, f"{cache_id}.npz")

        if not os.path.exists(path):
            raise FileNotFoundError(f"No cached dataset at {path}")

        data = np.load(path)
        samples = data["samples"]
        labels = data["labels"]
        return cls(id, cache_id, samples, labels)

    def to_disk(self):
        os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
        path = os.path.join(DATASET_CACHE_DIR, f"{self.cache_id}.npz")
        np.savez_compressed(path, samples=self.samples, labels=self.labels)