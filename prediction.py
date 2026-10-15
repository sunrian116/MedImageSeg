"""
prediction.py
-------------
Wraps nnUNetv2 inference (prediction) commands:
  1. Set required environment variables.
  2. Run prediction on a folder of test images.
  3. Optionally apply post-processing.
"""

import os
import subprocess
from pathlib import Path

import config


# ---------------------------------------------------------------------------
# Environment setup (reuse from training.py logic)
# ---------------------------------------------------------------------------

def set_nnunet_env() -> None:
    os.environ["nnUNet_raw"]          = str(config.NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(config.NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"]      = str(config.NNUNET_RESULTS)


def _run(cmd: list[str]) -> None:
    print(f"[Prediction] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    folds: list[int] | None = None,
    checkpoint: str = "checkpoint_best.pth",
    save_probabilities: bool = False,
    config_name: str | None = None,
) -> None:
    """
    Run nnUNetv2 inference.

    Parameters
    ----------
    input_dir          : folder with test images (*_0000.nii.gz); defaults to imagesTs
    output_dir         : folder for predicted segmentation masks
    folds              : which fold checkpoints to use (default: all in config)
    checkpoint         : checkpoint filename to load ("checkpoint_best.pth" or
                         "checkpoint_final.pth")
    save_probabilities : also save softmax probability maps (.npz)
    config_name        : label configuration name; uses ACTIVE_CONFIG if None
    """
    set_nnunet_env()

    # Get dataset paths and ID for the specified configuration
    dataset_id = config.get_dataset_id_for_config(config_name) if config_name else config.DATASET_ID
    dataset_paths = config.get_dataset_paths(config_name)
    
    input_dir  = input_dir  or dataset_paths["images_ts"]
    output_dir = output_dir or (config.DATA_DIR / f"predictions_{config_name or config.ACTIVE_CONFIG}")
    folds      = folds      or config.FOLDS

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i",  str(input_dir),
        "-o",  str(output_dir),
        "-d",  str(dataset_id),
        "-c",  config.NNUNET_CONFIG,
        "-tr", config.TRAINER,
        "-chk", checkpoint,
        "-f",  *[str(f) for f in folds],
    ]

    if save_probabilities:
        cmd.append("--save_probabilities")

    _run(cmd)
    print(f"[Prediction] Predictions saved → {output_dir}")


# ---------------------------------------------------------------------------
# Post-processing (optional)
# ---------------------------------------------------------------------------

def apply_postprocessing(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """
    Apply the post-processing configuration determined during training
    (removes small connected components, etc.).

    Parameters
    ----------
    input_dir  : folder with raw predictions (output of predict())
    output_dir : folder for post-processed results
    """
    set_nnunet_env()

    input_dir  = input_dir  or (config.DATA_DIR / "predictions")
    output_dir = output_dir or (config.DATA_DIR / "predictions_postprocessed")

    # Locate the automatically generated postprocessing.json
    pp_json = (
        config.NNUNET_RESULTS
        / config.DATASET_NAME
        / config.TRAINER
        / f"__nnUNetPlans__{config.NNUNET_CONFIG}"
        / "crossval_results_folds_0_1_2_3_4"
        / "postprocessing.json"
    )

    if not pp_json.exists():
        print(f"[Prediction] postprocessing.json not found at {pp_json}. Skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_apply_postprocessing",
        "-i",  str(input_dir),
        "-o",  str(output_dir),
        "-pp_pkl_file", str(pp_json),
        "-np", "8",
        "-plans_json", str(pp_json.parent / "plans.json"),
    ]
    _run(cmd)
    print(f"[Prediction] Post-processed results → {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    folds: list[int] | None = None,
    postprocess: bool = False,
    config_name: str | None = None,
) -> None:
    """
    Full inference pipeline:
      1. Predict segmentation masks.
      2. (Optionally) apply post-processing.
    
    Parameters
    ----------
    input_dir   : folder with test images; defaults to dataset imagesTs
    output_dir  : folder for predictions
    folds       : which folds to use (default: all)
    postprocess : apply post-processing after prediction
    config_name : label configuration name; uses ACTIVE_CONFIG if None
    """
    predict(input_dir=input_dir, output_dir=output_dir, folds=folds, config_name=config_name)

    if postprocess:
        apply_postprocessing()

    print("[Prediction] Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nnUNet prediction wrapper")
    parser.add_argument("--input",  type=Path, default=None,
                        help="Input folder with test images (default: imagesTs)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output folder for predictions")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Folds to use for prediction")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply post-processing after prediction")
    parser.add_argument("--config", type=str, default=None,
                        help=f"Label configuration to use (default: {config.ACTIVE_CONFIG})")
    args = parser.parse_args()

    run(
        input_dir=args.input,
        output_dir=args.output,
        folds=args.folds,
        postprocess=args.postprocess,
        config_name=args.config,
    )
