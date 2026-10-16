# Medical Image Segmentation with nnUNetv2

Multi-configuration segmentation pipeline supporting different anatomical structures and imaging modalities using nnUNetv2. Includes preprocessing, training, and inference workflows.

## Overview

This project supports multiple segmentation tasks through configurable label sets and dataset configurations:

### Supported Configurations

1. **Muscle Segmentation (muscle_11class)** — Dataset 100
   - Modality: T1-weighted MRI
   - Target: 11 thigh muscle classes
   - Classes:
     - Adductor Magnus, Biceps Long Head, Biceps Short Head
     - Gracilis, Rectus Femoris, Sartorius
     - Semimembranosus, Semitendinosus
     - Vastus Intermedius, Vastus Lateralis, Vastus Medialis

2. **Vessel Segmentation (vessel)** — Dataset 999
   - Modality: SWI (Susceptibility Weighted Imaging)
   - Target: Periventricular veins
   - Classes: Background, Vein

## Project Structure

```
MuscleSeg/
├── config.py              # Multi-config hub: LABEL_CONFIGS, paths, training settings
├── Preprocessing.py       # Data prep: DICOM→NIfTI, nnUNet structure, dataset.json
├── training.py            # Training: nnUNetv2 plan, preprocess, 5-fold cross-validation
├── prediction.py          # Inference: predictions, post-processing
├── pipeline.py            # CLI orchestrator with multi-config support
├── check_training_results.py # Training status monitoring script
├── requirements.txt       # Python dependencies (pinned versions)
├── data/                  # (created automatically)
│   ├── nnunet_raw/        # Input: organized dataset structure
│   ├── nnunet_preprocessed/ # nnUNet plans and preprocessing
│   ├── nnunet_results/    # Training checkpoints and logs
│   └── predictions_*/     # Segmentation predictions by config
└── README.md
```

## Installation

### 1. Create Virtual Environment

```bash
# On Linux/Mac
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements** (pinned versions for reproducibility):
```txt
nnunetv2==2.8.1
dicom2nifti==2.6.2
SimpleITK==2.5.6
numpy==2.5.4
```

## Workflow

### Step 0: List Available Configurations

```bash
python pipeline.py list-configs
```

Output:
```
muscle_11class
  Name: Thigh Muscle (11 classes)
  Labels: 11

vessel (active)
  Name: Periventricular Veins (Vessel Segmentation)
  Labels: 2
```

### Step 1: Prepare Data

**Option A: From DICOM files**

Organize input DICOMs:
```
data/dicom_raw/
├── patient_001/
│   ├── series_001.dcm
│   └── ...
└── patient_002/
```

**Option B: Pre-converted NIfTI files**

Place in nnUNet structure:
```
data/nnunet_raw/
└── Dataset999_Vessel/           # or Dataset100_Muscle
    ├── imagesTr/                # Training images
    │   ├── HC_001_0000.nii.gz
    │   ├── HC_002_0000.nii.gz
    │   └── ...
    ├── labelsTr/                # Training labels
    │   ├── HC_001.nii.gz
    │   ├── HC_002.nii.gz
    │   └── ...
    └── imagesTs/                # Test images (optional)
        ├── HC_101_0000.nii.gz
        └── ...
```

**Run preprocessing:**

```bash
# Preprocess vessel dataset (skip DICOM conversion)
python pipeline.py preprocess --config vessel --no-dicom-convert

# Preprocess muscle dataset with DICOM conversion
python pipeline.py preprocess --config muscle_11class
```

This:
- Creates folder structure automatically
- Converts DICOM → NIfTI (if needed)
- Validates file naming (`*_0000.nii.gz` for images)
- Generates `dataset.json` with correct labels and channels
- Extracts preprocessing fingerprints

### Step 2: Train

**Train all 5 folds** (full cross-validation):

```bash
python pipeline.py train --config vessel
```

**Train specific folds:**

```bash
python pipeline.py train --config vessel --folds 0 1 2
```

**Skip preprocessing** (if already done):

```bash
python pipeline.py train --config vessel --no-preprocess
```

**Check training progress:**

```bash
python check_training_results.py --config vessel
```

### Step 3: Predict

Run inference on test images:

```bash
# Predict with all 5 folds (ensemble)
python pipeline.py predict --config vessel

# Use specific folds
python pipeline.py predict --config vessel --folds 0 2 4

# Save probability maps
python pipeline.py predict --config vessel --save-probabilities

# Custom input/output
python pipeline.py predict --config vessel \
  --input /path/to/test/images \
  --output /path/to/predictions
```

### All-in-One

Run complete pipeline (preprocess → train → predict):

```bash
python pipeline.py all --config vessel
```

## Configuration System

### Multiple Configurations (config.py)

```python
LABEL_CONFIGS = {
    "muscle_11class": {
        "dataset_id": 100,
        "labels": {...},        # Background + 11 muscle classes
        "channel_names": {"0": "T1w"},
    },
    "vessel": {
        "dataset_id": 999,
        "labels": {             # Background + Vein
            "background": 0,
            "vein": 1,
        },
        "channel_names": {"0": "SWI"},
    },
}

ACTIVE_CONFIG = "muscle_11class"  # Change to "vessel" for vessel segmentation
```

### Dynamic Dataset IDs

Each configuration automatically maps to its own dataset ID:
- `muscle_11class` → Dataset100_Muscle
- `vessel` → Dataset999_Vessel

### Configuration Switching

**In Python:**
```python
# config.py
ACTIVE_CONFIG = "vessel"  # Switch to vessel segmentation
```

**Via CLI:**
```bash
python pipeline.py train --config vessel
python pipeline.py preprocess --config muscle_11class
python pipeline.py predict --config vessel
```

## Input/Output Formats

### Images

- **Format:** NIfTI (`.nii.gz`, compressed)
- **Naming:** `<case_id>_0000.nii.gz` (single channel, 0-padded suffix)
- **Modality:** T1w (muscle) or SWI (vessel)
- **Requirement:** 3D volume (automatically handled by nnUNetv2)

### Segmentation Labels

- **Format:** NIfTI (`.nii.gz`)
- **Naming:** `<case_id>.nii.gz`
- **Values:** Integer labels per configuration
  - Muscle: 0 (background), 1–11 (muscle classes)
  - Vessel: 0 (background), 1 (vein)

### Predictions

- **Location:** `data/predictions_{config_name}/`
- **Format:** `<case_id>.nii.gz` (label map)
- **Optional:** `<case_id>.npz` (softmax probabilities with `--save-probabilities`)

## Training Details

### Cross-Validation

- **Folds:** 5 (0, 1, 2, 3, 4)
- **Strategy:** Stratified split across training/validation per fold
- **Output:** One best checkpoint per fold
- **Ensemble:** Predictions can average across all 5 folds

### Configuration

| Setting | Default | Options |
|---------|---------|---------|
| `NNUNET_CONFIG` | `3d_fullres` | `2d`, `3d_lowres`, `3d_fullres`, `3d_cascade_fullres` |
| `TRAINER` | `nnUNetTrainer` | Custom trainer classes |
| `NUM_EPOCHS` | `100` | Any positive integer |
| `FOLDS` | `[0,1,2,3,4]` | Subset for quick testing |

### Environment Variables

Automatically set by scripts (no manual setup needed):

```bash
export nnUNet_raw=/home/fe2929/data/nnunet_raw
export nnUNet_preprocessed=/home/fe2929/data/nnunet_preprocessed
export nnUNet_results=/home/fe2929/data/nnunet_results
```

## Monitoring Training

### Live Progress

```bash
# Monitor fold 0 training in real-time
tail -f data/nnunet_results/Dataset999_Vessel/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt

# Watch with updates every 30 seconds
watch -n 30 'tail -5 data/nnunet_results/Dataset999_Vessel/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt'
```

### Training Summary

```bash
python check_training_results.py --config vessel
```

Displays:
- Checkpoint sizes and update times per fold
- Pseudo Dice scores and validation loss
- Average performance across all folds
- Completion status

## Troubleshooting

### 1. Configuration Not Recognized

```
Error: Invalid ACTIVE_CONFIG 'vessel'. Available: ['muscle_11class']
```

**Solution:**
- Check spelling in `config.py`: `ACTIVE_CONFIG = "vessel"`
- Ensure configuration exists in `LABEL_CONFIGS` dictionary
- Use `python pipeline.py list-configs` to see available options

### 2. Dataset.json Missing

```
FileNotFoundError: data/nnunet_raw/Dataset999_Vessel/dataset.json
```

**Solution:**
```bash
python pipeline.py preprocess --config vessel --no-dicom-convert
```

### 3. File Naming Issues

```
Error: Expected *_0000.nii.gz for images
```

**Solution:**
- Images must have `_0000` suffix: `HC_001_0000.nii.gz`
- Labels use base name: `HC_001.nii.gz` (no suffix)
- Rename with: `mv old_name.nii.gz HC_001_0000.nii.gz`

### 4. Training Crashes Early

```
CUDA out of memory or image size error
```

**Solutions:**
- Reduce memory usage: change `NNUNET_CONFIG = "3d_lowres"` in `config.py`
- Reduce batch size (if available)
- Use fewer folds: `python pipeline.py train --config vessel --folds 0`

### 5. Prediction Fails: Dataset ID Mismatch

```
Error: dataset_id mismatch between training and prediction
```

**Solution:**
- Ensure training and prediction use the **same `--config`**:
  ```bash
  python pipeline.py train --config vessel
  python pipeline.py predict --config vessel  # Must match
  ```

### 6. Environment Variables Not Set

```
UnboundLocalError: nnUNet_raw not set
```

**Solution:**
- Scripts automatically set environment variables
- If running nnUNet commands directly, set manually:
  ```bash
  export nnUNet_raw=/path/to/nnunet_raw
  export nnUNet_preprocessed=/path/to/nnunet_preprocessed
  export nnUNet_results=/path/to/nnunet_results
  ```

### 7. Different Results on Different Runs

**Note:** This is expected behavior! nnUNetv2 uses:
- Random seed variations during training
- Augmentation randomness
- Floating-point precision differences

For exact reproducibility, pin:
- Random seeds in training code
- NumPy version (already pinned in `requirements.txt`)
- CUDA/cuDNN versions (if using GPU)

## References

- [nnUNetv2 Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [DICOM to NIfTI conversion](https://github.com/icometrix/dicom2nifti)

## License

[Specify your license here]
