#!/usr/bin/env python3
"""
Script to check nnUNetv2 training results for all folds.
Usage: python check_training_results.py --config vessel
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import config


def get_latest_log_file(fold_dir):
    """Get the most recent training log file for a fold."""
    logs = list(fold_dir.glob("training_log_*.txt"))
    if not logs:
        return None
    return max(logs, key=os.path.getctime)


def get_final_metrics(log_file):
    """Extract final metrics from training log."""
    if not log_file or not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Get last 20 lines to find final epoch info
        final_lines = ''.join(lines[-20:])
        
        # Extract final epoch number
        epoch = None
        dice = None
        val_loss = None
        
        for line in lines[-20:]:
            if "Epoch " in line and "Current learning rate" in line:
                try:
                    epoch = int(line.split("Epoch ")[1].split()[0])
                except:
                    pass
            if "Pseudo dice" in line:
                try:
                    dice_str = line.split("[")[1].split("]")[0].replace("np.float32(", "").replace(")", "")
                    dice = float(dice_str)
                except:
                    pass
            if "val_loss" in line:
                try:
                    val_loss_str = line.split("val_loss ")[-1].strip()
                    val_loss = float(val_loss_str)
                except:
                    pass
        
        return {"epoch": epoch, "dice": dice, "val_loss": val_loss}
    except Exception as e:
        print(f"  Error reading log: {e}")
        return None


def check_results(config_name="muscle_11class"):
    """Check training results for a configuration."""
    
    # Get dataset info
    dataset_name = config.get_dataset_name_for_config(config_name)
    dataset_id = config.get_dataset_id_for_config(config_name)
    
    results_dir = Path("/home/fe2929/data/nnunet_results") / dataset_name / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    
    print(f"\n{'='*70}")
    print(f"Training Results for: {config_name} (Dataset {dataset_id})")
    print(f"Dataset Name: {dataset_name}")
    print(f"Results Directory: {results_dir}")
    print(f"{'='*70}\n")
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"❌ Results directory does not exist: {results_dir}")
        return
    
    # Check each fold
    all_complete = True
    fold_results = {}
    
    for fold_num in range(5):
        fold_dir = results_dir / f"fold_{fold_num}"
        print(f"\n📁 Fold {fold_num}:")
        
        if not fold_dir.exists():
            print(f"  ❌ Fold {fold_num} not found")
            all_complete = False
            continue
        
        # Check checkpoint files
        checkpoint_best = fold_dir / "checkpoint_best.pth"
        checkpoint_latest = fold_dir / "checkpoint_latest.pth"
        
        if checkpoint_best.exists():
            size_mb = checkpoint_best.stat().st_size / (1024**2)
            print(f"  ✅ checkpoint_best.pth ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ checkpoint_best.pth not found")
        
        if checkpoint_latest.exists():
            size_mb = checkpoint_latest.stat().st_size / (1024**2)
            mtime = datetime.fromtimestamp(checkpoint_latest.stat().st_mtime)
            print(f"  ✅ checkpoint_latest.pth ({size_mb:.1f} MB, updated: {mtime.strftime('%H:%M:%S')})")
        else:
            print(f"  ❌ checkpoint_latest.pth not found")
            all_complete = False
        
        # Get training log
        log_file = get_latest_log_file(fold_dir)
        if log_file:
            print(f"  📋 Log: {log_file.name}")
            metrics = get_final_metrics(log_file)
            if metrics:
                fold_results[fold_num] = metrics
                if metrics["epoch"]:
                    print(f"  📊 Final Epoch: {metrics['epoch']}")
                if metrics["dice"] is not None:
                    print(f"  🎯 Pseudo Dice: {metrics['dice']:.4f}")
                if metrics["val_loss"] is not None:
                    print(f"  📉 Val Loss: {metrics['val_loss']:.4f}")
        else:
            print(f"  ⚠️  No training log found")
            all_complete = False
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"{'='*70}")
    
    completed_folds = len(fold_results)
    print(f"✅ Completed Folds: {completed_folds}/5")
    
    if fold_results:
        avg_dice = sum(r["dice"] for r in fold_results.values() if r["dice"]) / len([r for r in fold_results.values() if r["dice"]])
        print(f"📊 Average Dice Score: {avg_dice:.4f}")
        
        print("\n📋 Per-Fold Results:")
        for fold_num in sorted(fold_results.keys()):
            r = fold_results[fold_num]
            print(f"  Fold {fold_num}: Epoch {r['epoch']}, Dice={r['dice']:.4f}, ValLoss={r['val_loss']:.4f}")
    
    if all_complete and completed_folds == 5:
        print(f"\n✅ Training COMPLETE! All 5 folds finished.")
        print(f"\n🚀 Next steps:")
        print(f"  1. Run predictions: python pipeline.py predict --config {config_name} --input <image> --output <dir>")
        print(f"  2. Or ensemble predictions from all folds for better results")
    elif completed_folds == 5:
        print(f"\n⚠️  All 5 folds exist but some may still be training.")
    else:
        print(f"\n⏳ Training in progress... ({completed_folds}/5 folds)")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check nnUNet training results")
    parser.add_argument("--config", default="vessel", help="Configuration name (default: vessel)")
    args = parser.parse_args()
    
    check_results(args.config)
