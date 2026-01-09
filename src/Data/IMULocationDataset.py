from src.Data.IMUBaseDataset import IMUBaseDataset
from whar_datasets import WHARDatasetID
from src.Data.SensorTypes import get_imu_groups
from src.Data.DiskData import DiskDataset
from src.Data.DiskLocationData import DiskLocationData
import numpy as np


class IMULocationDataset:

    def __init__(self, id: WHARDatasetID):
        self.imu_groups = get_imu_groups(id)
        train_cache = f"{id}_train_location"
        val_cache = f"{id}_val_location"
        test_cache = f"{id}_test_location"
        legacy_train_cache = f"{id}_train"
        legacy_val_cache = f"{id}_val"
        legacy_test_cache = f"{id}_test"
        try:
            self.train = self._load_or_migrate_cache(id, train_cache, legacy_train_cache)
            self.val = self._load_or_migrate_cache(id, val_cache, legacy_val_cache)
            self.test = self._load_or_migrate_cache(id, test_cache, legacy_test_cache)
            print("Loaded IMU Location Dataset from cache.")
            return
        except FileNotFoundError:
            pass

        base_dataset = IMUBaseDataset(id)
        self.train = self._transform_samples(base_dataset.train, train_cache)
        self.val = self._transform_samples(base_dataset.val, val_cache)
        self.test = self._transform_samples(base_dataset.test, test_cache)

    def _load_or_migrate_cache(self, id: WHARDatasetID, cache_id: str, legacy_cache_id: str) -> DiskLocationData:
        try:
            return DiskLocationData.from_disk(id, cache_id)
        except FileNotFoundError:
            data = DiskLocationData.from_disk(id, legacy_cache_id)
            data.cache_id = str(cache_id)
            data.to_disk()
            return data

    
    def _transform_samples(self, disk_dataset: DiskDataset, cache_id: str):
        transformed_samples = []
        transformed_labels = []
        transformed_locations = []
        for group_name, group in self.imu_groups:
            location_id = group_name.value
            group_samples = disk_dataset.samples[:, :, group]
            transformed_samples.append(group_samples)
            transformed_labels.append(disk_dataset.labels.copy())
            transformed_locations.append(np.full(group_samples.shape[0], location_id, dtype=np.int32))

        transformed_samples = np.concatenate(transformed_samples, axis=0)
        transformed_labels = np.concatenate(transformed_labels, axis=0)
        transformed_locations = np.concatenate(transformed_locations, axis=0)
        assert transformed_samples.shape[0] == transformed_labels.shape[0] == transformed_locations.shape[0]

        data = DiskLocationData(disk_dataset.id, cache_id, transformed_samples, transformed_labels, transformed_locations)
        data.to_disk()
        return data
    

    @property
    def num_classes(self) -> int:
        return len(np.unique(self.train.labels))
