from src.Data.IMULocationDataset import IMULocationDataset
from src.Data.DiskLocationData import DiskLocationData
from src.Data.SensorTypes import SensorLocation
from src.Data.ClassNames import get_class_names, global_class_map

import numpy as np
from whar_datasets import WHARDatasetID


class MergedIMULocationDataset:
    def __init__(
        self,
        imu_location_datasets: list[IMULocationDataset],
        locations: list[SensorLocation] | None = None,
        ratio: float = 1.0,
        seed: int | None = 0,
        ratio_splits: tuple[str, ...] = ("train",),
    ):
        """
        Merge multiple IMU-location datasets into a single dataset with global label space.

        Args:
            imu_location_datasets: Source datasets (each already expanded by sensor location).
            locations: Optional filter of sensor locations to keep.
            ratio: If < 1.0, randomly subsample selected split(s) after merging.
            seed: RNG seed for subsampling (set to None for non-deterministic).
            ratio_splits: Split names to apply `ratio` to. Default ("train",) is useful
                for fine-tuning on a subset while keeping val/test intact.
        """
        self.imu_location_datasets = imu_location_datasets
        self.locations = locations

        self.train = self._merge_datasets([ds.train for ds in imu_location_datasets], cache_id="merged_train")
        self.val = self._merge_datasets([ds.val for ds in imu_location_datasets], cache_id="merged_val")
        self.test = self._merge_datasets([ds.test for ds in imu_location_datasets], cache_id="merged_test")

        self.ratio = float(ratio)
        self.ratio_splits = tuple(ratio_splits)
        self.seed = seed

        if not (0.0 < self.ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {self.ratio}.")

        if self.ratio < 1.0:
            rng = np.random.default_rng(self.seed)
            if "train" in self.ratio_splits:
                self.train = self._subsample(self.train, ratio=self.ratio, rng=rng, cache_id="merged_train_sub")
            if "val" in self.ratio_splits:
                self.val = self._subsample(self.val, ratio=self.ratio, rng=rng, cache_id="merged_val_sub")
            if "test" in self.ratio_splits:
                self.test = self._subsample(self.test, ratio=self.ratio, rng=rng, cache_id="merged_test_sub")

    def _map_labels_to_global_space(self, dataset: DiskLocationData) -> np.ndarray:
        dataset_id = dataset.id
        if isinstance(dataset_id, str):
            dataset_id = WHARDatasetID(dataset_id)

        local_names = get_class_names(dataset_id)
        local_to_global = np.asarray([global_class_map[name] for name in local_names], dtype=np.int64)

        labels = np.asarray(dataset.labels)
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        labels = labels.astype(np.int64, copy=False)
        return local_to_global[labels]

    def _merge_datasets(self, datasets: list[DiskLocationData], cache_id: str) -> DiskLocationData:
        if len(datasets) == 0:
            raise ValueError("Expected at least one dataset to merge.")

        selected_locations: np.ndarray | None = None
        if self.locations is not None:
            selected_locations = np.asarray([loc.value for loc in self.locations], dtype=np.int32)

        merged_samples: list[np.ndarray] = []
        merged_labels: list[np.ndarray] = []
        merged_locations: list[np.ndarray] = []

        expected_sample_tail_shape: tuple[int, ...] | None = None
        max_seq_len: int | None = None
        sample_dtype = None

        for dataset in datasets:
            samples = np.asarray(dataset.samples)
            locations = np.asarray(dataset.locations)
            labels = self._map_labels_to_global_space(dataset)

            if selected_locations is not None:
                mask = np.isin(locations, selected_locations)
                samples = samples[mask]
                labels = labels[mask]
                locations = locations[mask]

            if samples.shape[0] == 0:
                continue

            if samples.ndim < 2:
                raise ValueError(
                    f"Expected samples with shape (N,T,...) but got {samples.shape} for dataset {dataset.id}."
                )

            if sample_dtype is None:
                sample_dtype = samples.dtype

            sample_tail_shape = samples.shape[2:]
            if expected_sample_tail_shape is None:
                expected_sample_tail_shape = sample_tail_shape
            elif sample_tail_shape != expected_sample_tail_shape:
                raise ValueError(
                    "Cannot merge datasets with different feature/channel shapes; "
                    f"expected *x(T,{expected_sample_tail_shape}), got *x(T,{sample_tail_shape}) for dataset {dataset.id}."
                )

            seq_len = int(samples.shape[1])
            if max_seq_len is None or seq_len > max_seq_len:
                max_seq_len = seq_len

            merged_samples.append(samples)
            merged_labels.append(labels)
            merged_locations.append(locations)

        if expected_sample_tail_shape is None:
            expected_sample_tail_shape = ()
        if max_seq_len is None:
            max_seq_len = 0
        if sample_dtype is None:
            sample_dtype = np.float32

        if len(merged_samples) == 0:
            empty_samples = np.empty((0, max_seq_len, *expected_sample_tail_shape), dtype=sample_dtype)
            empty_labels = np.empty((0,), dtype=np.int64)
            empty_locations = np.empty((0,), dtype=np.int32)
            return DiskLocationData("merged", cache_id, empty_samples, empty_labels, empty_locations)

        padded_samples: list[np.ndarray] = []
        for samples in merged_samples:
            pad_len = max_seq_len - int(samples.shape[1])
            if pad_len <= 0:
                padded_samples.append(samples)
                continue
            pad_width = [(0, 0)] * samples.ndim
            pad_width[1] = (0, pad_len)
            padded_samples.append(np.pad(samples, pad_width=pad_width, mode="constant", constant_values=0))

        samples = np.concatenate(padded_samples, axis=0)
        labels = np.concatenate(merged_labels, axis=0)
        locations = np.concatenate(merged_locations, axis=0)
        assert samples.shape[0] == labels.shape[0] == locations.shape[0]

        return DiskLocationData("merged", cache_id, samples, labels, locations)

    def _subsample(
        self,
        dataset: DiskLocationData,
        ratio: float,
        rng: np.random.Generator,
        cache_id: str,
    ) -> DiskLocationData:
        n = int(np.asarray(dataset.samples).shape[0])
        if n == 0 or ratio >= 1.0:
            return dataset
        k = int(np.floor(n * ratio))
        k = max(1, min(n, k))
        idx = rng.choice(n, size=k, replace=False)
        return DiskLocationData(
            dataset.id,
            cache_id,
            np.asarray(dataset.samples)[idx],
            np.asarray(dataset.labels)[idx],
            np.asarray(dataset.locations)[idx],
        )

    @property
    def num_classes(self) -> int:
        return int(len(global_class_map))

    @property
    def present_class_ids(self) -> np.ndarray:
        return np.unique(self.train.labels)
