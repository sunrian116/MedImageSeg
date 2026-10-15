"""
Preprocessing.py
----------------
Handles all data preparation steps before nnUNet training:
  1. Create the required nnUNet folder structure.
  2. Convert DICOM series to NIfTI (.nii.gz).
  3. Rename files to nnUNet convention  (<case>_0000.nii.gz for images,
     <case>.nii.gz for labels).
  4. Generate dataset.json.
"""

import json
import os
import shutil
from pathlib import Path

import dicom2nifti
import dicom2nifti.settings as dicom2nifti_settings

import config


# ---------------------------------------------------------------------------
# 1. Folder structure
# ---------------------------------------------------------------------------

def create_nnunet_folders(config_name: str | None = None) -> None:
    """
    Create all directories required by nnUNet.
    
    Parameters
    ----------
    config_name : label configuration name; uses ACTIVE_CONFIG if None
    """
    paths = config.get_dataset_paths(config_name)
    
    for folder in [
        paths["images_tr"],
        paths["labels_tr"],
        paths["images_ts"],
        config.NNUNET_PREPROCESSED,
        config.NNUNET_RESULTS,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    
    dataset_name = paths["dataset_name"]
    print(f"[Preprocessing] nnUNet folder structure created for dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# 2. DICOM → NIfTI conversion
# ---------------------------------------------------------------------------

def convert_dicom_to_nifti(dicom_dir: Path, output_dir: Path) -> None:
    """
    Convert a single DICOM series folder to NIfTI and save in output_dir.

    Parameters
    ----------
    dicom_dir  : folder that contains the DICOM files for one series
    output_dir : destination folder for the converted .nii.gz file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dicom2nifti_settings.disable_validate_slice_increment()
    dicom2nifti.convert_directory(str(dicom_dir), str(output_dir), compression=True, reorient=True)


def convert_all_dicoms(
    dicom_root: Path | None = None,
    output_root: Path | None = None,
) -> None:
    """
    Batch-convert every patient sub-folder under dicom_root.

    Expected layout::

        dicom_root/
            patient_001/
                *.dcm
            patient_002/
                *.dcm

    Converted NIfTI files land in output_root/<patient_id>/.
    """
    dicom_root  = dicom_root  or config.DICOM_RAW_DIR
    output_root = output_root or (config.DATA_DIR / "nifti_converted")
    output_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([d for d in dicom_root.iterdir() if d.is_dir()])
    if not patient_dirs:
        print(f"[Preprocessing] No patient folders found in {dicom_root}")
        return

    for patient_dir in patient_dirs:
        out_dir = output_root / patient_dir.name
        print(f"[Preprocessing] Converting {patient_dir.name} ...")
        convert_dicom_to_nifti(patient_dir, out_dir)

    print(f"[Preprocessing] Conversion complete → {output_root}")


# ---------------------------------------------------------------------------
# 3. Copy / rename files into nnUNet layout
# ---------------------------------------------------------------------------

def copy_image_to_nnunet(src: Path, case_id: str, is_test: bool = False) -> None:
    """
    Copy a NIfTI image into the correct nnUNet imagesTr / imagesTs folder
    and rename it to <case_id>_0000.nii.gz (channel 0 = T1w).
    """
    dest_dir  = config.IMAGES_TS if is_test else config.IMAGES_TR
    dest_file = dest_dir / f"{case_id}_0000.nii.gz"
    shutil.copy2(src, dest_file)
    print(f"[Preprocessing] Image  → {dest_file.name}")


def copy_label_to_nnunet(src: Path, case_id: str) -> None:
    """
    Copy a NIfTI segmentation mask into labelsTr and rename it to
    <case_id>.nii.gz.
    """
    dest_file = config.LABELS_TR / f"{case_id}.nii.gz"
    shutil.copy2(src, dest_file)
    print(f"[Preprocessing] Label  → {dest_file.name}")


# ---------------------------------------------------------------------------
# 4. dataset.json
# ---------------------------------------------------------------------------

def generate_dataset_json(
    json_path: Path | None = None,
    num_training: int | None = None,
    config_name: str | None = None,
) -> None:
    """
    Write the nnUNet dataset.json file.

    Parameters
    ----------
    json_path     : destination path; defaults to <DATASET_RAW_DIR>/dataset.json
    num_training  : number of training cases; auto-counted if None
    config_name   : label configuration name; uses ACTIVE_CONFIG if None
    """
    # Get paths for the specified or active configuration
    paths = config.get_dataset_paths(config_name)
    default_json_path = paths["dataset_raw_dir"] / "dataset.json"
    json_path = json_path or default_json_path

    if num_training is None:
        num_training = len(list(paths["images_tr"].glob("*_0000.nii.gz")))

    # Get labels and channel names for the specified configuration
    if config_name is not None:
        if config_name not in config.LABEL_CONFIGS:
            raise ValueError(
                f"Invalid config_name '{config_name}'. "
                f"Available: {list(config.LABEL_CONFIGS.keys())}"
            )
        active_config = config.LABEL_CONFIGS[config_name]
        labels = active_config["labels"]
        channel_names = active_config["channel_names"]
        print(f"[Preprocessing] Using label configuration: '{config_name}' ({active_config['name']})")
    else:
        # Use the active configuration from config
        labels = config.get_labels()
        channel_names = config.get_channel_names()
        print(f"[Preprocessing] Using active configuration: '{config.ACTIVE_CONFIG}'")

    dataset = {
        "channel_names": channel_names,
        "labels":        labels,
        "numTraining":   num_training,
        "file_ending":   config.FILE_ENDING,
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(dataset, indent=4))
    print(f"[Preprocessing] dataset.json written → {json_path}")


# ---------------------------------------------------------------------------
# 5. Convenience: run full preprocessing pipeline
# ---------------------------------------------------------------------------

def run(convert_dicoms: bool = True, config_name: str | None = None) -> None:
    """
    End-to-end preprocessing:
      1. Create folders
      2. (Optionally) convert DICOMs to NIfTI
      3. Generate dataset.json

    Parameters
    ----------
    convert_dicoms : whether to convert DICOM files to NIfTI
    config_name    : label configuration name; uses ACTIVE_CONFIG if None
    """
    create_nnunet_folders(config_name=config_name)

    if convert_dicoms:
        convert_all_dicoms()

    generate_dataset_json(config_name=config_name)
    print("[Preprocessing] Done.")


if __name__ == "__main__":
    run()

