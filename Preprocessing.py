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

def create_nnunet_folders() -> None:
    """Create all directories required by nnUNet."""
    for folder in [
        config.IMAGES_TR,
        config.LABELS_TR,
        config.IMAGES_TS,
        config.NNUNET_PREPROCESSED,
        config.NNUNET_RESULTS,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    print("[Preprocessing] nnUNet folder structure created.")


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

def generate_dataset_json(json_path: Path | None = None, num_training: int | None = None) -> None:
    """
    Write the nnUNet dataset.json file.

    Parameters
    ----------
    json_path     : destination path; defaults to <DATASET_RAW_DIR>/dataset.json
    num_training  : number of training cases; auto-counted if None
    """
    json_path = json_path or (config.DATASET_RAW_DIR / "dataset.json")

    if num_training is None:
        num_training = len(list(config.IMAGES_TR.glob("*_0000.nii.gz")))

    dataset = {
        "channel_names": config.CHANNEL_NAMES,
        "labels":        config.LABELS,
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

def run(convert_dicoms: bool = True) -> None:
    """
    End-to-end preprocessing:
      1. Create folders
      2. (Optionally) convert DICOMs to NIfTI
      3. Generate dataset.json
    """
    create_nnunet_folders()

    if convert_dicoms:
        convert_all_dicoms()

    generate_dataset_json()
    print("[Preprocessing] Done.")


if __name__ == "__main__":
    run()

