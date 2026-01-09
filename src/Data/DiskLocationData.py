import os
from src.config import DATASET_LOCATION_CACHE_DIR
import numpy as np
from torch.utils.data import DataLoader

class DiskLocationData:
    def __init__(self, id, cache_id, samples, labels, locations):
        self.id = id                     # original WHARDatasetID
        self.cache_id = str(cache_id)    # e.g. "<id>_train"
        self.samples = samples
        self.labels = labels
        self.locations = locations


    @classmethod
    def from_disk(cls, id, cache_id):
        cache_id = str(cache_id)
        path = os.path.join(DATASET_LOCATION_CACHE_DIR, f"{cache_id}.npz")

        if not os.path.exists(path):
            raise FileNotFoundError(f"No cached dataset at {path}")

        data = np.load(path)
        samples = data["samples"]
        labels = data["labels"]
        locations= data["locations"]
        return cls(id, cache_id, samples, labels, locations)

    def to_disk(self):
        os.makedirs(DATASET_LOCATION_CACHE_DIR, exist_ok=True)
        path = os.path.join(DATASET_LOCATION_CACHE_DIR, f"{self.cache_id}.npz")
        np.savez_compressed(path, samples=self.samples, labels=self.labels, locations=self.locations)