#!/usr/bin/env python3
"""
Quick visualization script for TartanAir P008 sequence.
Shows ground truth vs prediction side by side.

Usage:
    python visualize_comparison.py --pred_file predictions/000000.npy
    python visualize_comparison.py --pred_root ./predictions --all
"""

import os
import sys
import argparse

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_tartanair import (
    visualize_side_by_side,
    visualize_sequence_comparison,
    draw_voxels,
    get_fov_mask_tartanair,
    load_voxel_file,
)
import numpy as np

# =============================================================================
# Configuration - Update these paths for your setup
# =============================================================================

# Ground truth root (containing P008/voxel_label_lcam_front/)
GT_ROOT = "data/tartanair/CarWelding/Data_easy"

# Sequence and camera
SEQUENCE = "P008"
CAMERA = "lcam_front"

# Full GT directory path
GT_DIR = os.path.join(GT_ROOT, SEQUENCE, f"voxel_label_{CAMERA}")


def find_gt_file(frame_id):
    """Find the GT file for a given frame ID."""
    candidates = [
        os.path.join(GT_DIR, f"{frame_id}_voxel_label.npy"),
        os.path.join(GT_DIR, f"{frame_id}_voxel.npy"),
        os.path.join(GT_DIR, f"{frame_id}.npy"),
        os.path.join(GT_DIR, f"{frame_id}_voxel_label.label"),
        os.path.join(GT_DIR, f"{frame_id}_voxel.label"),
        os.path.join(GT_DIR, f"{frame_id}.label"),
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # Try with different zero-padding
    frame_num = int(frame_id) if frame_id.isdigit() else 0
    for padding in [6, 5, 4, 3]:
        padded_id = str(frame_num).zfill(padding)
        for suffix in ["_voxel_label.npy", "_voxel.npy", ".npy", "_voxel_label.label", "_voxel.label", ".label"]:
            path = os.path.join(GT_DIR, f"{padded_id}{suffix}")
            if os.path.exists(path):
                return path
    
    return None


def find_rgb_depth_files(frame_id):
    """Find RGB and depth files for a given frame ID."""
    rgb_dir = os.path.join(GT_ROOT, SEQUENCE, f"image_{CAMERA}")
    depth_dir = os.path.join(GT_ROOT, SEQUENCE, f"depth_{CAMERA}")
    
    rgb_candidates = [
        os.path.join(rgb_dir, f"{frame_id}_{CAMERA}.png"),
        os.path.join(rgb_dir, f"{frame_id}.png"),
    ]
    depth_candidates = [
        os.path.join(depth_dir, f"{frame_id}_{CAMERA}_depth.png"),
        os.path.join(depth_dir, f"{frame_id}_depth.png"),
        os.path.join(depth_dir, f"{frame_id}.png"),
    ]
    
    # Also try with different padding
    frame_num = int(frame_id) if frame_id.isdigit() else 0
    for padding in [6, 5, 4, 3]:
        padded_id = str(frame_num).zfill(padding)
        rgb_candidates.extend([
            os.path.join(rgb_dir, f"{padded_id}_{CAMERA}.png"),
            os.path.join(rgb_dir, f"{padded_id}.png"),
        ])
        depth_candidates.extend([
            os.path.join(depth_dir, f"{padded_id}_{CAMERA}_depth.png"),
            os.path.join(depth_dir, f"{padded_id}_depth.png"),
            os.path.join(depth_dir, f"{padded_id}.png"),
        ])
    
    rgb_path = None
    depth_path = None
    
    for candidate in rgb_candidates:
        if os.path.exists(candidate):
            rgb_path = candidate
            break
    
    for candidate in depth_candidates:
        if os.path.exists(candidate):
            depth_path = candidate
            break
    
    return rgb_path, depth_path


def visualize_single(pred_path, output_dir="./vis_output", view_type="bev"):
    """Visualize a single prediction with its GT, RGB, and depth."""
    
    # Extract frame ID from prediction filename
    basename = os.path.basename(pred_path)
    frame_id = os.path.splitext(basename)[0].split('_')[0]
    
    print(f"Frame ID: {frame_id}")
    print(f"Looking for GT in: {GT_DIR}")
    
    # Find GT file
    gt_path = find_gt_file(frame_id)
    
    if gt_path is None:
        print(f"ERROR: Could not find GT file for frame {frame_id}")
        print(f"Searched in: {GT_DIR}")
        return None
    
    print(f"Found GT: {gt_path}")
    print(f"Prediction: {pred_path}")
    
    # Find RGB and depth files
    rgb_path, depth_path = find_rgb_depth_files(frame_id)
    
    if rgb_path:
        print(f"Found RGB: {rgb_path}")
    else:
        print("RGB image not found")
    
    if depth_path:
        print(f"Found Depth: {depth_path}")
    else:
        print("Depth image not found")
    
    # Create output path
    save_path = os.path.join(output_dir, SEQUENCE, f"{frame_id}_comparison.png")
    
    # Visualize
    visualize_side_by_side(
        gt_path=gt_path,
        pred_path=pred_path,
        save_path=save_path,
        view_type=view_type,
        title_gt=f"Ground Truth - Frame {frame_id}",
        title_pred=f"Prediction - Frame {frame_id}",
        rgb_path=rgb_path,
        depth_path=depth_path,
    )
    
    return save_path


def visualize_all(pred_root, output_dir="./vis_output", view_type="bev", max_frames=None):
    """Visualize all predictions in a directory."""
    
    visualize_sequence_comparison(
        gt_root=GT_ROOT,
        pred_root=pred_root,
        save_root=output_dir,
        sequence=SEQUENCE,
        camera=CAMERA,
        view_type=view_type,
        max_frames=max_frames,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TartanAir P008 predictions vs ground truth with RGB and depth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction (.label format) - auto-finds GT, RGB, and depth
  python visualize_comparison.py --pred_file predictions/000000.label
  
  # Single prediction (.npy format)  
  python visualize_comparison.py --pred_file predictions/000000.npy
  
  # All predictions in a directory
  python visualize_comparison.py --pred_root ./predictions --all
  
  # First 10 frames only
  python visualize_comparison.py --pred_root ./predictions --all --max_frames 10
  
  # Different view
  python visualize_comparison.py --pred_file pred.label --view side

Output layout:
  +------------------+------------------+
  |   Ground Truth   |    Prediction    |
  |    (voxels)      |     (voxels)     |
  +------------------+------------------+
  |    RGB Image     |   Depth Image    |
  +------------------+------------------+
        """
    )
    
    parser.add_argument('--pred_file', type=str, help='Single prediction .npy file')
    parser.add_argument('--pred_root', type=str, help='Directory containing predictions')
    parser.add_argument('--all', action='store_true', help='Process all predictions in pred_root')
    parser.add_argument('--output_dir', type=str, default='./vis_output', help='Output directory')
    parser.add_argument('--view', type=str, default='bev', 
                        choices=['bev', 'side', 'front', 'top'],
                        help='Camera view type')
    parser.add_argument('--max_frames', type=int, help='Maximum frames to process')
    parser.add_argument('--gt_root', type=str, help=f'Override GT root (default: {GT_ROOT})')
    parser.add_argument('--sequence', type=str, help=f'Override sequence (default: {SEQUENCE})')
    
    args = parser.parse_args()
    
    # Override global settings if provided
    # global GT_ROOT, SEQUENCE, GT_DIR
    # if args.gt_root:
    #     GT_ROOT = args.gt_root
    # if args.sequence:
    #     SEQUENCE = args.sequence
    GT_DIR = os.path.join(GT_ROOT, SEQUENCE, f"voxel_label_{CAMERA}")
    
    # Check GT directory exists
    if not os.path.exists(GT_DIR):
        alt_gt_dir = os.path.join(GT_ROOT, SEQUENCE, f"voxel_{CAMERA}")
        if os.path.exists(alt_gt_dir):
            GT_DIR = alt_gt_dir
        else:
            print(f"ERROR: GT directory not found: {GT_DIR}")
            print(f"Please update GT_ROOT in this script or use --gt_root")
            return
    
    print(f"GT Directory: {GT_DIR}")
    
    if args.pred_file:
        visualize_single(args.pred_file, args.output_dir, args.view)
    
    elif args.pred_root and args.all:
        visualize_all(args.pred_root, args.output_dir, args.view, args.max_frames)
    
    elif args.pred_root:
        # Find first prediction and visualize it
        pred_files = sorted([f for f in os.listdir(args.pred_root) if f.endswith(('.npy', '.label', '.bin'))])
        if pred_files:
            pred_path = os.path.join(args.pred_root, pred_files[0])
            print(f"Visualizing first prediction: {pred_path}")
            print("Use --all to process all predictions")
            visualize_single(pred_path, args.output_dir, args.view)
        else:
            print(f"No prediction files (.npy, .label, .bin) found in {args.pred_root}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()