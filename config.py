from pathlib import Path

# -----------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------
DATASET_ID = 100                          # nnUNet dataset number (100 -> Dataset100_Muscle)
DATASET_NAME = f"Dataset{DATASET_ID:03d}_Muscle"

LABELS = {
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
}

CHANNEL_NAMES = {"0": "T1w"}
FILE_ENDING = ".nii.gz"

# -----------------------------------------------------------------------
# nnUNet training configuration
# -----------------------------------------------------------------------
NNUNET_CONFIG = "3d_fullres"   # "2d" | "3d_lowres" | "3d_fullres" | "3d_cascade_fullres"
TRAINER      = "nnUNetTrainer" # default trainer
FOLDS        = [0, 1, 2, 3, 4] # 5-fold cross-validation; use [0] for a quick single fold

# -----------------------------------------------------------------------
# Directory layout
# -----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

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
