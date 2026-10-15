# Quick Reference: Label Configurations

## Current Setup
✅ Three example configurations included:
- `muscle_11class` - 11 individual thigh muscles (default)
- `vessel` - 2 groups

## Quick Commands

```bash
# View all available configurations
python pipeline.py list-configs

# Preprocess with default config (muscle_11class)
python pipeline.py preprocess

# Preprocess with specific config
python pipeline.py preprocess --config vessel

# Full pipeline with specific config
python pipeline.py all --config vessel
```

## To Change Default Configuration

Edit `config.py`:
```python
ACTIVE_CONFIG = "vessel"  # Change to your preferred default
```

## To Add New Configuration

Add to `LABEL_CONFIGS` dictionary in `config.py`:
```python
"my_new_config": {
    "name": "My Custom Labels",
    "labels": {
        "background": 0,
        "class_1": 1,
        "class_2": 2,
    },
    "channel_names": {"0": "T1w"},
},
```

## What Gets Generated
When preprocessing runs:
- Creates `data/nnunet_raw/Dataset100_Muscle/dataset.json`
- JSON contains the labels for the selected configuration
- nnUNet uses this to know which classes to train on

## Important Notes
- Each configuration can have different number of classes
- Labels must match your annotated data
- Switching configurations requires re-preprocessing
- The `--config` flag overrides the `ACTIVE_CONFIG` setting in config.py

## Support for Multiple Datasets
You can now:
1. Define different label configurations in config.py
2. Process different datasets with different labels
3. Train separate models for each task
4. All from the same codebase!
