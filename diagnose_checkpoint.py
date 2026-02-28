#!/usr/bin/env python3
"""Diagnose checkpoint loading issues between mmcv and refactored model.

Usage:
    python diagnose_checkpoint.py --config_path configs/voxdet-semantickitti-cam.py \
        --ckpt_path ckpts/voxdet-semantickitti_cam/tensorboard/version_0/checkpoints/best.ckpt

This script loads a checkpoint and the model, then compares all keys to find:
1. Keys that match directly
2. Keys that are successfully remapped
3. Checkpoint keys that fail to match any model key (DROPPED weights)
4. Model keys with no checkpoint entry
5. Shape mismatches between checkpoint and model
"""

import sys
import os
import torch
import argparse
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voxdet_core import Config, build_model
from voxdet_models import *  # register all modules


def load_raw_state_dict(filename):
    """Load checkpoint and extract raw state_dict with prefix stripping."""
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip common prefixes (same as checkpoint.py)
    new_state_dict = {}
    for k, v in state_dict.items():
        orig_k = k
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('model.'):
            k = k[6:]
        new_state_dict[k] = (v, orig_k)

    return new_state_dict


def diagnose(config_path, ckpt_path):
    cfg = Config.fromfile(config_path)
    model = build_model(cfg['model'])
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    ckpt_sd = load_raw_state_dict(ckpt_path)

    print(f"Model keys: {len(model_keys)}")
    print(f"Checkpoint keys (after prefix strip): {len(ckpt_sd)}")
    print()

    # Categorize keys
    direct_match = {}       # ckpt key matches model key directly
    remapped_match = {}     # ckpt key -> new key matches model key
    dropped = {}            # ckpt key (or remapped) has NO match in model
    shape_mismatch = {}     # key matches but shape differs
    skipped = {}            # num_batches_tracked etc.

    matched_model_keys = set()

    for k, (v, orig_k) in ckpt_sd.items():
        if k.endswith('.num_batches_tracked'):
            skipped[k] = 'num_batches_tracked'
            continue

        # Direct match?
        if k in model_keys:
            model_v = model_sd[k]
            if v.shape == model_v.shape:
                direct_match[k] = k
                matched_model_keys.add(k)
            else:
                shape_mismatch[k] = (v.shape, model_v.shape)
            continue

        # Try remapping
        new_k = k
        transforms_applied = []

        if '.bn.' in new_k:
            new_k = new_k.replace('.bn.', '.norm.')
            transforms_applied.append('.bn. -> .norm.')
        if '.gn.' in new_k:
            new_k = new_k.replace('.gn.', '.norm.')
            transforms_applied.append('.gn. -> .norm.')
        if '.conv_offset.' in new_k:
            new_k = new_k.replace('.conv_offset.', '.offset_conv.')
            transforms_applied.append('.conv_offset. -> .offset_conv.')
        if 'depth_conv.' in new_k and (new_k.endswith('.weight') or new_k.endswith('.bias')):
            parts = new_k.split('.')
            for i, part in enumerate(parts):
                if part == 'depth_conv' and i + 2 < len(parts):
                    next_part = parts[i + 1]
                    after_next = parts[i + 2]
                    if next_part.isdigit() and after_next in ('weight', 'bias') and i + 2 == len(parts) - 1:
                        parts.insert(i + 2, 'dcn')
                        new_k = '.'.join(parts)
                        transforms_applied.append('depth_conv.N.X -> depth_conv.N.dcn.X')
                        break

        if new_k != k and new_k in model_keys:
            model_v = model_sd[new_k]
            if v.shape == model_v.shape:
                remapped_match[k] = (new_k, transforms_applied)
                matched_model_keys.add(new_k)
            else:
                shape_mismatch[new_k] = (v.shape, model_v.shape, k)
        else:
            dropped[k] = (new_k, transforms_applied if transforms_applied else ['none'])

    missing_model_keys = model_keys - matched_model_keys

    # Report
    print("=" * 80)
    print(f"DIRECT MATCHES: {len(direct_match)}")
    print("=" * 80)
    if len(direct_match) <= 20:
        for ck in sorted(direct_match.keys()):
            print(f"  {ck}")
    else:
        # Group by module prefix
        prefixes = defaultdict(int)
        for ck in direct_match:
            prefix = '.'.join(ck.split('.')[:3])
            prefixes[prefix] += 1
        for p in sorted(prefixes.keys()):
            print(f"  {p}.*  ({prefixes[p]} keys)")

    print()
    print("=" * 80)
    print(f"REMAPPED MATCHES: {len(remapped_match)}")
    print("=" * 80)
    for ck in sorted(remapped_match.keys()):
        new_k, transforms = remapped_match[ck]
        print(f"  {ck}")
        print(f"    -> {new_k}  [{', '.join(transforms)}]")

    print()
    print("=" * 80)
    print(f"SHAPE MISMATCHES: {len(shape_mismatch)}")
    print("=" * 80)
    for k in sorted(shape_mismatch.keys()):
        info = shape_mismatch[k]
        if len(info) == 2:
            print(f"  {k}: ckpt={info[0]} vs model={info[1]}")
        else:
            print(f"  {k} (from {info[2]}): ckpt={info[0]} vs model={info[1]}")

    print()
    print("=" * 80)
    print(f"DROPPED CHECKPOINT KEYS (no model match): {len(dropped)}")
    print("=" * 80)
    for ck in sorted(dropped.keys()):
        new_k, transforms = dropped[ck]
        if new_k != ck:
            print(f"  {ck}  (remapped to: {new_k})  [{', '.join(transforms)}]")
        else:
            print(f"  {ck}")

    # Try fuzzy matching for dropped keys
    if dropped:
        print()
        print("  --- Fuzzy match suggestions for dropped keys ---")
        for ck in sorted(dropped.keys()):
            new_k = dropped[ck][0]
            # Find closest model keys
            base = new_k.split('.')[-1]  # e.g., 'weight', 'bias'
            prefix_parts = new_k.split('.')[:-1]

            # Try progressively shorter prefixes
            suggestions = []
            for length in range(len(prefix_parts), max(0, len(prefix_parts) - 3), -1):
                prefix = '.'.join(prefix_parts[:length])
                matches = [mk for mk in model_keys if mk.startswith(prefix) and mk.endswith(base)]
                if matches and len(matches) <= 5:
                    suggestions = matches
                    break

            if suggestions:
                print(f"  {ck} -> maybe: {suggestions}")

    print()
    print("=" * 80)
    print(f"MISSING MODEL KEYS (no checkpoint entry): {len(missing_model_keys)}")
    print("=" * 80)
    # Group by module prefix
    prefixes = defaultdict(list)
    for mk in sorted(missing_model_keys):
        prefix = '.'.join(mk.split('.')[:3])
        prefixes[prefix].append(mk)
    for p in sorted(prefixes.keys()):
        keys = prefixes[p]
        if len(keys) <= 3:
            for k in keys:
                print(f"  {k}")
        else:
            print(f"  {p}.*  ({len(keys)} keys)")
            # Show first 2
            for k in keys[:2]:
                print(f"    e.g. {k}")

    print()
    print("=" * 80)
    print(f"SKIPPED: {len(skipped)}")
    print("=" * 80)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_ckpt = len(ckpt_sd) - len(skipped)
    loaded = len(direct_match) + len(remapped_match)
    print(f"  Total checkpoint params (excl. num_batches_tracked): {total_ckpt}")
    print(f"  Successfully loaded: {loaded} ({100*loaded/max(total_ckpt,1):.1f}%)")
    print(f"  Shape mismatches: {len(shape_mismatch)}")
    print(f"  DROPPED (not loaded): {len(dropped)}")
    print(f"  Model params without checkpoint: {len(missing_model_keys)}")

    if dropped:
        print()
        print("  *** WARNING: Dropped checkpoint keys mean trained weights are LOST ***")
        print("  *** This is likely causing the mIoU regression ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/voxdet-semantickitti-cam.py')
    parser.add_argument('--ckpt_path', required=True)
    args = parser.parse_args()
    diagnose(args.config_path, args.ckpt_path)
