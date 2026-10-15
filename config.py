from pathlib import Path

# -----------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------
# Default DATASET_ID (can be overridden per configuration)
DEFAULT_DATASET_ID = 100

# -----------------------------------------------------------------------
# Label Configurations for Different Applications/Datasets
# -----------------------------------------------------------------------
# Define multiple label configurations for different segmentation tasks
LABEL_CONFIGS = {
    "muscle_11class": {
        "dataset_id": 100,
        "name": "Thigh Muscle (11 classes)",
        "labels": {
            "background":        0,
            "adductormagnus":    1,
            "bicepslonghead":    2,
            "bicepsshorthead":   3,
            "gracilis":          4,
            "rectusfemoris":     5,
            "sartorius":         6,
            "semimembranosus":   7,
            "semitendinosus":    8,
            "vastusintermedius": 9,
            "vastuslateralis":   10,
            "vastusmedialis":    11,
        },
        "channel_names": {"0": "T1w"},
    },
    "vessel": {
        "dataset_id": 999,
        "name": "Periventricular Veins (Vessel Segmentation)",
        "labels": {
            "background":  0,
            "vein":  1,
        },
        "channel_names": {"0": "SWI"},
    },    
}

# Select the active label configuration
# Options: "muscle_11class", "vessel", etc.
ACTIVE_CONFIG = "muscle_11class"

# Retrieve active configuration
def get_active_config():
    """Get the active label configuration."""
    if ACTIVE_CONFIG not in LABEL_CONFIGS:
        raise ValueError(
            f"Invalid ACTIVE_CONFIG '{ACTIVE_CONFIG}'. "
            f"Available: {list(LABEL_CONFIGS.keys())}"
        )
    return LABEL_CONFIGS[ACTIVE_CONFIG]

def get_labels():
    """Get labels for the active configuration."""
    return get_active_config()["labels"]

def get_channel_names():
    """Get channel names for the active configuration."""
    return get_active_config()["channel_names"]

def get_dataset_id():
    """Get dataset ID for the active configuration."""
    config = get_active_config()
    return config.get("dataset_id", DEFAULT_DATASET_ID)

def get_dataset_id_for_config(config_name: str) -> int:
    """Get dataset ID for a specific configuration."""
    if config_name not in LABEL_CONFIGS:
        raise ValueError(
            f"Invalid config_name '{config_name}'. "
            f"Available: {list(LABEL_CONFIGS.keys())}"
        )
    config = LABEL_CONFIGS[config_name]
    return config.get("dataset_id", DEFAULT_DATASET_ID)

def get_dataset_name():
    """Get dataset name based on active configuration."""
    config_key = ACTIVE_CONFIG
    dataset_id = get_dataset_id()
    # Convert config_key to dataset name (e.g., "vessel" -> "Vessel", "muscle_11class" -> "Muscle11class")
    name_parts = config_key.split("_")
    name_suffix = "".join(word.capitalize() for word in name_parts)
    return f"Dataset{dataset_id:03d}_{name_suffix}"

def get_dataset_name_for_config(config_name: str) -> str:
    """Get dataset name for a specific configuration."""
    if config_name not in LABEL_CONFIGS:
        raise ValueError(
            f"Invalid config_name '{config_name}'. "
            f"Available: {list(LABEL_CONFIGS.keys())}"
        )
    dataset_id = get_dataset_id_for_config(config_name)
    name_parts = config_name.split("_")
    name_suffix = "".join(word.capitalize() for word in name_parts)
    return f"Dataset{dataset_id:03d}_{name_suffix}"

def get_dataset_paths(config_name: str | None = None):
    """
    Get dataset directory paths for a specific configuration.
    
    Parameters
    ----------
    config_name : configuration name; uses ACTIVE_CONFIG if None
    
    Returns
    -------
    dict with keys: dataset_raw_dir, images_tr, labels_tr, images_ts, dataset_name
    """
    if config_name is None:
        config_name = ACTIVE_CONFIG
    
    dataset_name = get_dataset_name_for_config(config_name)
    dataset_raw_dir = NNUNET_RAW / dataset_name
    
    return {
        "dataset_raw_dir": dataset_raw_dir,
        "images_tr": dataset_raw_dir / "imagesTr",
        "labels_tr": dataset_raw_dir / "labelsTr",
        "images_ts": dataset_raw_dir / "imagesTs",
        "dataset_name": dataset_name,
    }

# Backward compatibility: maintain LABELS, CHANNEL_NAMES, DATASET_NAME, and DATASET_ID
LABELS = get_labels()
CHANNEL_NAMES = get_channel_names()
DATASET_ID = get_dataset_id()
DATASET_NAME = get_dataset_name()
FILE_ENDING = ".nii.gz"

# -----------------------------------------------------------------------
# nnUNet training configuration
# -----------------------------------------------------------------------
NNUNET_CONFIG = "3d_fullres"   # "2d" | "3d_lowres" | "3d_fullres" | "3d_cascade_fullres"
TRAINER      = "nnUNetTrainer" # default trainer
FOLDS        = [0, 1, 2, 3, 4] # 5-fold cross-validation; use [0] for a quick single fold
NUM_EPOCHS   = 100             # number of epochs per fold (default nnUNetv2: 1000)

# -----------------------------------------------------------------------
# Directory layout
# -----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

# Point to external data directory at /home/fe2929/data
# (Change to BASE_DIR / "data" to use local project folder instead)
DATA_DIR = Path("/home/fe2929/data")

# nnUNet expects these three environment variables
NNUNET_RAW          = DATA_DIR / "nnunet_raw"
NNUNET_PREPROCESSED = DATA_DIR / "nnunet_preprocessed"
NNUNET_RESULTS      = DATA_DIR / "nnunet_results"

# Dataset sub-folders inside nnunet_raw
DATASET_RAW_DIR = NNUNET_RAW / DATASET_NAME
IMAGES_TR       = DATASET_RAW_DIR / "imagesTr"
LABELS_TR       = DATASET_RAW_DIR / "labelsTr"
IMAGES_TS       = DATASET_RAW_DIR / "imagesTs"

# Raw DICOM input folder (put patient sub-folders here)
DICOM_RAW_DIR = DATA_DIR / "dicom_raw"
