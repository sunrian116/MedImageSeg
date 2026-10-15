# Label Configuration Guide

This project supports multiple label configurations, allowing you to work with different segmentation tasks and datasets while using the same codebase.

## Available Configurations

### 1. `muscle_11class` (Default)
**Name:** Thigh Muscle (11 classes)

Segments all 11 thigh muscles individually:
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

### 2. `muscle_quadriceps_hamstring`
**Name:** Quadriceps vs Hamstring

Simplified classification into two muscle groups:
- Quadriceps
- Hamstring

### 3. `muscle_simple`
**Name:** Simple Muscle Segmentation

Basic binary classification:
- Muscle
- Background

## How to Use

### Option 1: Change the Default Configuration

Edit `config.py` and change the `ACTIVE_CONFIG` variable:

```python
ACTIVE_CONFIG = "muscle_quadriceps_hamstring"  # Change this line
```

Then run preprocessing as usual:
```bash
python pipeline.py preprocess
python pipeline.py train
```

### Option 2: Specify Configuration at Runtime

Use the `--config` flag with preprocessing or the full pipeline:

```bash
# Preprocess with a specific configuration
python pipeline.py preprocess --config muscle_quadriceps_hamstring

# Run the full pipeline with a specific configuration
python pipeline.py all --config muscle_simple
```

### List Available Configurations

View all available configurations and their details:

```bash
python pipeline.py list-configs
```

## Adding a New Configuration

To add a new label configuration:

1. Open `config.py`
2. Add a new entry to the `LABEL_CONFIGS` dictionary:

```python
LABEL_CONFIGS = {
    # ... existing configs ...
    "my_custom_config": {
        "name": "My Custom Segmentation",
        "labels": {
            "background": 0,
            "label_1": 1,
            "label_2": 2,
            # ... add more labels as needed
        },
        "channel_names": {"0": "T1w"},  # or {"0": "T2w"}, etc.
    },
}
```

3. Either set it as the default:
```python
ACTIVE_CONFIG = "my_custom_config"
```

Or use it at runtime:
```bash
python pipeline.py preprocess --config my_custom_config
```

## How It Works

1. **Configuration Selection:**
   - The `ACTIVE_CONFIG` setting in `config.py` defines the default
   - The `--config` flag at runtime can override the default

2. **Dataset JSON Generation:**
   - When preprocessing runs, it generates `dataset.json` with the selected labels
   - This file tells nnUNet which classes to segment and their numeric IDs

3. **Backward Compatibility:**
   - Existing code that references `config.LABELS` and `config.CHANNEL_NAMES` still works
   - These are automatically populated from the active configuration

## Example Workflow

```bash
# List all configurations
python pipeline.py list-configs

# Preprocess with 11-class muscle segmentation (default)
python pipeline.py preprocess

# Switch to quadriceps vs hamstring for a different dataset
python pipeline.py preprocess --config muscle_quadriceps_hamstring

# Generate dataset.json for the new configuration
# (automatically done during preprocessing)

# Train with the new labels
python pipeline.py train

# Make predictions
python pipeline.py predict --input data/test_images --output data/predictions
```

## Notes

- Each configuration generates its own `dataset.json` in the dataset folder
- Ensure that your labeled data matches the selected label configuration
- When switching configurations, you may need to:
  - Re-label your data
  - Re-run preprocessing
  - Retrain the model
