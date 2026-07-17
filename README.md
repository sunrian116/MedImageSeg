# Muscle Segmentation with nnUNetv2

Automated segmentation pipeline for thigh muscle MRI using nnUNetv2. Includes preprocessing, training, and inference workflows.

## Overview

This project segments 11 thigh muscle classes from T1-weighted MRI:
- Adductor Magnus
- Biceps Long Head
- Biceps Short Head
- Gracilis
- Rectus Femoris
- Sartorius
- Semimembranosus
- Semitendinosus
- Vastus Intermedius
- Vastus Lateralis
- Vastus Medialis

## Project Structure

```
Segment/
├── config.py              # Configuration: dataset ID, labels, paths, training settings
├── Preprocessing.py       # Data preparation: DICOM→NIfTI, folder layout, dataset.json
├── training.py            # nnUNetv2 training: plan, preprocess, train folds
├── prediction.py          # Inference: prediction and post-processing
├── pipeline.py            # CLI entry point orchestrating the workflow
├── requirements.txt       # Python dependencies
├── data/                  # (created automatically)
│   ├── dicom_raw/         # Input: raw DICOM files (patient_id/*)
│   ├── nifti_converted/   # Intermediate: converted NIfTI files
│   ├── nnunet_raw/        # nnUNet dataset structure
│   ├── nnunet_preprocessed/ # nnUNet preprocessed data
│   ├── nnunet_results/    # Training outputs and checkpoints
│   └── predictions/       # Segmentation masks from inference
└── README.md
```

## Installation

### 1. Create Virtual Environment

```powershell
# Already created: .venv/
.venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
uv pip install -r requirements.txt
```

Or manually:
```powershell
uv pip install nnunetv2 dicom2nifti SimpleITK numpy
```

## Workflow

### Step 1: Prepare Data

Organize your DICOM files:

```
data/
└── dicom_raw/
    ├── patient_001/
    │   ├── image.dcm
    │   ├── image.dcm
    │   └── ...
    ├── patient_002/
    │   └── ...
    └── ...
```

Prepare segmentation masks (NIfTI format) in the nnUNet structure:

```
data/
└── nnunet_raw/
    └── Dataset100_Muscle/
        └── labelsTr/
            ├── patient_001.nii.gz
            ├── patient_002.nii.gz
            └── ...
```

Run preprocessing:

```powershell
python pipeline.py preprocess
```

This:
- Creates required folder structure
- Converts DICOM series → NIfTI (`.nii.gz`)
- Renames files to nnUNet convention (`<case_id>_0000.nii.gz` for images, `<case_id>.nii.gz` for labels)
- Generates `dataset.json`

### Step 2: Train

Plan, preprocess, and train:

```powershell
python pipeline.py train
```

Train specific folds:
```powershell
python pipeline.py train --folds 0 1 2
```

Skip preprocessing (if already done):
```powershell
python pipeline.py train --no-preprocess
```

### Step 3: Predict

Run inference on test images:

```powershell
python pipeline.py predict
```

With post-processing:
```powershell
python pipeline.py predict --postprocess
```

Custom input/output folders:
```powershell
python pipeline.py predict --input "D:/path/to/test/images" --output "D:/path/to/predictions"
```

### All-in-One

Run the complete pipeline end-to-end:

```powershell
python pipeline.py all
```

## Configuration

Edit [config.py](config.py) to customize:

| Setting | Description |
|---------|-------------|
| `DATASET_ID` | nnUNet dataset number (100 → Dataset100_Muscle) |
| `NNUNET_CONFIG` | `"3d_fullres"` (default) \| `"2d"` \| `"3d_lowres"` |
| `TRAINER` | `"nnUNetTrainer"` (default) |
| `FOLDS` | Cross-validation folds `[0, 1, 2, 3, 4]` |
| `LABELS` | Muscle class mapping (0=background, 1-11=muscles) |
| `BASE_DIR` | Project root directory |
| `DATA_DIR` | Data folder location |

## Input/Output Formats

### Images

- **Format:** NIfTI (`.nii.gz`)
- **Naming:** `<case_id>_0000.nii.gz` (channel 0 = T1w)
- **Required:** Single channel (grayscale MRI)

### Segmentation Masks

- **Format:** NIfTI (`.nii.gz`)
- **Naming:** `<case_id>.nii.gz`
- **Values:** 0 (background), 1-11 (muscle classes, per `config.LABELS`)

### Predictions

- **Location:** `data/predictions/`
- **Format:** `<case_id>.nii.gz` (label map)
- **Optional:** `<case_id>.npz` (softmax probabilities, if `--save-probabilities`)

## Environment Variables

nnUNetv2 requires three paths (automatically set by scripts):

```powershell
$env:nnUNet_raw          = "D:\work\code\wip\muscle\Segment\data\nnunet_raw"
$env:nnUNet_preprocessed = "D:\work\code\wip\muscle\Segment\data\nnunet_preprocessed"
$env:nnUNet_results      = "D:\work\code\wip\muscle\Segment\data\nnunet_results"
```

## Troubleshooting

### `dataset.json` missing
- Run `python pipeline.py preprocess` to generate it

### Training fails on image size
- Adjust `NNUNET_CONFIG` to `"3d_lowres"` for memory constraints
- Ensure images are in `data/nnunet_raw/Dataset100_Muscle/imagesTr/`

### Prediction outputs not found
- Check that test images are in `data/nnunet_raw/Dataset100_Muscle/imagesTs/`
- Verify checkpoint exists: `data/nnunet_results/Dataset100_Muscle/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/`

## References

- [nnUNetv2 Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [DICOM to NIfTI conversion](https://github.com/icometrix/dicom2nifti)

## License

[Specify your license here]
