from src.Data.DiskData import DiskDataset
from src.config import WHAR_DATASETS_CACHE_DIR

from whar_datasets import (
    WHARDatasetID,
    get_dataset_cfg,
    PostProcessingPipeline,
    PreProcessingPipeline,
    RandomSplitter,
    TorchAdapter,
    Loader,
)
from whar_datasets.utils.types import NormType

class IMUBaseDataset:
    def __init__(self, id: WHARDatasetID):

        train_cache = f"{id}_train"
        val_cache   = f"{id}_val"
        test_cache  = f"{id}_test"

        # Try loading from disk
        try:
            self.train = DiskDataset.from_disk(id, train_cache)
            self.val   = DiskDataset.from_disk(id, val_cache)
            self.test  = DiskDataset.from_disk(id, test_cache)
            print("Loaded IMU Dataset from cache.")
            return
        except FileNotFoundError:
            pass

        # Build from scratch
        cfg = get_dataset_cfg(id, datasets_dir=WHAR_DATASETS_CACHE_DIR)
        cfg.normalization = NormType.STD_PER_SAMPLE

        pre_pipeline = PreProcessingPipeline(cfg)
        activity_df, session_df, window_df = pre_pipeline.run()

        splitter = RandomSplitter(cfg, train_percentage=0.75, val_percentage=0.05, test_percentage=0.20)
        splits = splitter.get_splits(session_df, window_df)
        split = splits[0]

        post_pipeline = PostProcessingPipeline(cfg, pre_pipeline, window_df, split.train_indices)
        samples = post_pipeline.run()

        loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
        adapter = TorchAdapter(cfg, loader, split)

        dataloaders = adapter.get_dataloaders(batch_size=64)
        # {"train": train_loader, "val": val_loader, "test": test_loader}
        train_loader = dataloaders["train"]
        val_loader   = dataloaders["val"]
        test_loader  = dataloaders["test"]

        # Convert loaders
        self.train = DiskDataset.from_dataloader(id, train_cache, train_loader)
        self.val   = DiskDataset.from_dataloader(id, val_cache, val_loader)
        self.test  = DiskDataset.from_dataloader(id, test_cache, test_loader)

        # Save to disk
        self.train.to_disk()
        self.val.to_disk()
        self.test.to_disk()