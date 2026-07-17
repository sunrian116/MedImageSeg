"""
pipeline.py
-----------
Main entry point that orchestrates the full muscle segmentation workflow:

    python pipeline.py preprocess
    python pipeline.py train [--folds 0 1 2] [--no-preprocess]
    python pipeline.py predict [--input <dir>] [--output <dir>] [--postprocess]
    python pipeline.py all
"""

import argparse
from pathlib import Path

import Preprocessing
import training
import prediction


def cmd_preprocess(args: argparse.Namespace) -> None:
    Preprocessing.run(convert_dicoms=not args.no_dicom_convert)


def cmd_train(args: argparse.Namespace) -> None:
    training.run(
        folds=args.folds,
        preprocess=not args.no_preprocess,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    prediction.run(
        input_dir=args.input,
        output_dir=args.output,
        folds=args.folds,
        postprocess=args.postprocess,
    )


def cmd_all(args: argparse.Namespace) -> None:
    """Run the complete pipeline end-to-end."""
    print("=" * 60)
    print("STEP 1/3 — Preprocessing")
    print("=" * 60)
    Preprocessing.run(convert_dicoms=True)

    print("\n" + "=" * 60)
    print("STEP 2/3 — Training")
    print("=" * 60)
    training.run()

    print("\n" + "=" * 60)
    print("STEP 3/3 — Prediction")
    print("=" * 60)
    prediction.run(postprocess=True)

    print("\nPipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Muscle segmentation pipeline using nnUNetv2"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- preprocess ---
    p_pre = sub.add_parser("preprocess", help="Prepare data for nnUNet")
    p_pre.add_argument("--no-dicom-convert", action="store_true",
                       help="Skip DICOM → NIfTI conversion")
    p_pre.set_defaults(func=cmd_preprocess)

    # --- train ---
    p_train = sub.add_parser("train", help="Plan, preprocess and train nnUNet")
    p_train.add_argument("--folds", nargs="+", type=int, default=None,
                         help="Folds to train (default: all)")
    p_train.add_argument("--no-preprocess", action="store_true",
                         help="Skip nnUNet plan_and_preprocess step")
    p_train.set_defaults(func=cmd_train)

    # --- predict ---
    p_pred = sub.add_parser("predict", help="Run inference on new images")
    p_pred.add_argument("--input",  type=Path, default=None,
                        help="Input folder with *_0000.nii.gz images")
    p_pred.add_argument("--output", type=Path, default=None,
                        help="Output folder for segmentation masks")
    p_pred.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Folds to use (default: all)")
    p_pred.add_argument("--postprocess", action="store_true",
                        help="Apply post-processing after prediction")
    p_pred.set_defaults(func=cmd_predict)

    # --- all ---
    p_all = sub.add_parser("all", help="Run the full pipeline end-to-end")
    p_all.set_defaults(func=cmd_all)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
