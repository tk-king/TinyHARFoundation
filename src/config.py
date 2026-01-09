from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_CACHE_DIR = PROJECT_ROOT / "cache" / "datasets"
DATASET_LOCATION_CACHE_DIR = PROJECT_ROOT / "cache" / "location_datasets"
WHAR_DATASETS_CACHE_DIR = PROJECT_ROOT / "cache" / "whar_datasets"
