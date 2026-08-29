#!/usr/bin/env python3
"""Train YOLO26 pose for bbox + 2 keypoints (pelvis, pelvis_ground)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER, YAML


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bbox+2-point localization model")
    p.add_argument("--model", type=str, default="yolo26x-pose.pt", help="Model .pt or architecture .yaml")
    p.add_argument("--init-weights", type=str, default="", help="Optional pretrained .pt to load into --model YAML")
    p.add_argument(
        "--data",
        type=str,
        default="ultralytics/cfg/datasets/pelvis-proj-2pt.yaml",
        help="Dataset YAML with kpt_shape=[2,3]",
    )
    p.add_argument("--imgsz", type=int, default=960, help="Training image size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--device", type=str, default="0", help="CUDA device, e.g. 0 or 0,1 or cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=str, default="runs/pelvis-proj")
    p.add_argument("--name", type=str, default="yolo26x-pelvis-proj-960-e20-b64")
    p.add_argument("--rect", action=argparse.BooleanOptionalAction, default=True, help="Use rectangular batching")
    p.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable AMP mixed precision"
    )
    p.add_argument("--freeze", type=int, default=0, help="Freeze first N layers")
    p.add_argument("--optimizer", type=str, default="MuSGD", help="Optimizer: MuSGD/SGD/AdamW/... ")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--pose", type=float, default=18.0, help="Keypoint regression loss gain")
    p.add_argument("--kobj", type=float, default=2.0, help="Keypoint objectness loss gain")
    p.add_argument("--kpt1-weight", type=float, default=2.0, help="Relative weight for keypoint index 1")
    p.add_argument("--refine", type=float, default=None, help="Refinement loss gain (phase-2 models)")
    p.add_argument("--refine-prior", type=float, default=None, help="Refinement geometric prior gain")
    p.add_argument("--mosaic", type=float, default=0.2, help="Mosaic augmentation probability")
    p.add_argument("--translate", type=float, default=0.03, help="Translation augmentation fraction")
    p.add_argument("--scale", type=float, default=0.2, help="Scale augmentation gain")
    p.add_argument("--shear", type=float, default=0.0, help="Shear augmentation")
    p.add_argument("--perspective", type=float, default=0.0, help="Perspective augmentation")
    p.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation")
    p.add_argument("--close-mosaic", type=int, default=5, help="Disable mosaic in final N epochs")
    p.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every N epochs; disabled if < 1")
    p.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume training from")
    p.add_argument(
        "--allow-oob-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow out-of-bounds/zero-area labels and avoid clipping predictions",
    )
    return p.parse_args()


def validate_data_yaml(path: Path) -> None:
    cfg = YAML.load(path)
    kpt_shape = cfg.get("kpt_shape", None)
    if list(kpt_shape or []) != [2, 3]:
        raise ValueError(f"{path} must define kpt_shape: [2, 3], got: {kpt_shape}")

    for key in ("path", "train", "val"):
        if key not in cfg:
            raise ValueError(f"{path} missing required key: '{key}'")

    LOGGER.info(f"Dataset: {path}")
    LOGGER.info(f"Data root: {cfg['path']}")
    LOGGER.info("Model output will be: bbox + 2 keypoints (pelvis, pelvis_ground)")
    LOGGER.info("Pose training keeps bbox loss by default (box/cls/dfl + pose/kobj).")


def main() -> None:
    args = parse_args()

    if args.optimizer.lower() != "musgd":
        raise ValueError(
            f"This training pipeline is locked to MuSGD for YOLO original-weight fine-tuning. "
            f"Got optimizer={args.optimizer!r}"
        )

    model_path = Path(args.model)
    data_path = Path(args.data)
    init_weights = Path(args.init_weights) if args.init_weights else None
    resume_ckpt = Path(args.resume) if args.resume else None
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if init_weights is not None and not init_weights.exists():
        raise FileNotFoundError(f"Init weights not found: {init_weights}")
    if resume_ckpt is not None and not resume_ckpt.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")

    validate_data_yaml(data_path)

    model = YOLO(str(model_path))
    if init_weights is not None:
        LOGGER.info(f"Loading init weights: {init_weights}")
        model = model.load(str(init_weights))

    train_args = dict(
        task="pose",
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        rect=args.rect,
        amp=args.amp,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer="MuSGD",
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        pose=args.pose,
        kobj=args.kobj,
        kpt1_weight=args.kpt1_weight,
        refine=args.refine,
        refine_prior=args.refine_prior,
        allow_oob_labels=args.allow_oob_labels,
        mosaic=args.mosaic,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        degrees=args.degrees,
        freeze=args.freeze if args.freeze > 0 else None,
        close_mosaic=args.close_mosaic,
        save_period=args.save_period,
        resume=str(resume_ckpt) if resume_ckpt is not None else None,
        plots=True,
    )

    # Drop None-valued args for cleaner Ultralytics cfg merge.
    train_args = {k: v for k, v in train_args.items() if v is not None}

    LOGGER.info("Starting training with args:")
    for k, v in train_args.items():
        LOGGER.info(f"  {k}: {v}")

    model.train(**train_args)


if __name__ == "__main__":
    main()
