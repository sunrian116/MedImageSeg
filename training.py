"""
training.py
-----------
Wraps nnUNetv2 training commands:
  1. Set required environment variables.
  2. Plan and preprocess the dataset.
  3. Train one or all folds.
"""

import os
import subprocess
import sys
from pathlib import Path

import config


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def set_nnunet_env() -> None:
    """Export the three paths that nnUNetv2 requires."""
    os.environ["nnUNet_raw"]          = str(config.NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(config.NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"]      = str(config.NNUNET_RESULTS)
    print("[Training] nnUNet environment variables set.")


def _run(cmd: list[str]) -> None:
    """Run a shell command and stream output; raise on failure."""
    print(f"[Training] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


# ---------------------------------------------------------------------------
# Plan & preprocess
# ---------------------------------------------------------------------------

def plan_and_preprocess(verify_integrity: bool = True) -> None:
    """
    Run nnUNetv2_plan_and_preprocess for the configured dataset.

    Parameters
    ----------
    verify_integrity : pass --verify_dataset_integrity to catch common errors
    """
    set_nnunet_env()
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(config.DATASET_ID),
        "-c", config.NNUNET_CONFIG,
    ]
    if verify_integrity:
        cmd.append("--verify_dataset_integrity")
    _run(cmd)
    print("[Training] Plan and preprocess complete.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fold(fold: int) -> None:
    """
    Train a single cross-validation fold.

    Parameters
    ----------
    fold : fold index (0-4)
    """
    set_nnunet_env()
    cmd = [
        "nnUNetv2_train",
        str(config.DATASET_ID),
        config.NNUNET_CONFIG,
        str(fold),
        "--trainer", config.TRAINER,
    ]
    _run(cmd)
    print(f"[Training] Fold {fold} complete.")


def train_all_folds() -> None:
    """Train all folds defined in config.FOLDS sequentially."""
    set_nnunet_env()
    for fold in config.FOLDS:
        print(f"\n[Training] ===== Starting fold {fold} =====")
        train_fold(fold)
    print("[Training] All folds complete.")


# ---------------------------------------------------------------------------
# Find best configuration (optional post-training step)
# ---------------------------------------------------------------------------

def find_best_config() -> None:
    """
    Run nnUNetv2_find_best_configuration to pick the best ensemble/config
    after all folds have been trained.
    """
    set_nnunet_env()
    cmd = [
        "nnUNetv2_find_best_configuration",
        str(config.DATASET_ID),
        "-c", config.NNUNET_CONFIG,
    ]
    _run(cmd)
    print("[Training] Best configuration identified.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(folds: list[int] | None = None, preprocess: bool = True) -> None:
    """
    Full training pipeline:
      1. Plan & preprocess (optional)
      2. Train specified folds (default: all folds in config)
    """
    if preprocess:
        plan_and_preprocess()

    folds = folds if folds is not None else config.FOLDS
    for fold in folds:
        train_fold(fold)

    print("[Training] Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nnUNet training wrapper")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Folds to train (default: all folds in config)")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip plan_and_preprocess step")
    args = parser.parse_args()

    run(folds=args.folds, preprocess=not args.no_preprocess)
