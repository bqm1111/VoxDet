#!/usr/bin/env python3
"""Run a tiny end-to-end training sanity check for VoxDet.

This script creates a temporary config derived from a base config, overrides it
for a tiny run (mini split + few steps), and launches ``main.py``.
"""

import argparse
import os
import pprint
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path


def _load_py_config(path):
    namespace = runpy.run_path(path)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def _dump_py_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in cfg.items():
            f.write(f"{key} = {pprint.pformat(value, width=120)}\n")


def _enable_mini_split(cfg_dict):
    data = cfg_dict.get("data", {})
    if not isinstance(data, dict):
        return

    for split in ("train", "val", "test"):
        ds_cfg = data.get(split)
        if isinstance(ds_cfg, dict) and ds_cfg.get("type") == "SemanticKITTIDataset":
            ds_cfg["mini_split"] = True


def _set_dataloader(cfg_dict, train_batch_size, train_workers, eval_batch_size, eval_workers):
    train_loader = cfg_dict.get("train_dataloader_config")
    if isinstance(train_loader, dict):
        train_loader["batch_size"] = train_batch_size
        train_loader["num_workers"] = train_workers

    test_loader = cfg_dict.get("test_dataloader_config")
    if isinstance(test_loader, dict):
        test_loader["batch_size"] = eval_batch_size
        test_loader["num_workers"] = eval_workers


def _drop_invalid_load_from(cfg):
    load_from = cfg.get("load_from", None)
    if isinstance(load_from, str) and os.path.isfile(load_from):
        return
    cfg.pop("load_from", None)


def build_sanity_config(base_config, out_config, steps, train_batch_size, train_workers, eval_batch_size, eval_workers, force_mini_split, keep_load_from):
    cfg = _load_py_config(base_config)

    if force_mini_split:
        _enable_mini_split(cfg)

    _set_dataloader(
        cfg,
        train_batch_size=train_batch_size,
        train_workers=train_workers,
        eval_batch_size=eval_batch_size,
        eval_workers=eval_workers,
    )

    cfg["training_steps"] = steps
    cfg["accumulate_grad_batches"] = 1
    cfg["check_val_every_n_epoch"] = max(9999, int(cfg.get("check_val_every_n_epoch", 1)))

    if not keep_load_from:
        _drop_invalid_load_from(cfg)

    _dump_py_config(cfg, out_config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tiny VoxDet training sanity check")
    parser.add_argument("--base-config", default="configs/voxdet-semantickitti-cam.py", help="Base config to derive from")
    parser.add_argument("--log-folder", default="logs/sanity-train", help="Output log folder")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--train-batch-size", type=int, default=1, help="Train batch size")
    parser.add_argument("--train-workers", type=int, default=0, help="Train dataloader workers")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Eval batch size")
    parser.add_argument("--eval-workers", type=int, default=0, help="Eval dataloader workers")

    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES value if not already set")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging (recommended for sanity runs)")
    parser.add_argument("--no-mini-split", action="store_true", help="Do not force mini_split on SemanticKITTIDataset")
    parser.add_argument("--keep-load-from", action="store_true", help="Keep 'load_from' from base config even if file is missing")
    parser.add_argument("--dry-run", action="store_true", help="Only print command and generated config path")
    return parser.parse_args()


def main():
    args = parse_args()

    base_config = Path(args.base_config).resolve()
    if not base_config.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="voxdet_sanity_"))
    sanity_config = tmp_dir / "sanity_config.py"

    build_sanity_config(
        base_config=str(base_config),
        out_config=str(sanity_config),
        steps=args.steps,
        train_batch_size=args.train_batch_size,
        train_workers=args.train_workers,
        eval_batch_size=args.eval_batch_size,
        eval_workers=args.eval_workers,
        force_mini_split=not args.no_mini_split,
        keep_load_from=args.keep_load_from,
    )

    cmd = [
        sys.executable,
        "main.py",
        "--config_path",
        str(sanity_config),
        "--log_folder",
        args.log_folder,
        "--seed",
        str(args.seed),
        "--log_every_n_steps",
        "1",
    ]

    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in env and args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.disable_wandb:
        env["WANDB_MODE"] = "disabled"

    print(f"Generated sanity config: {sanity_config}")
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
