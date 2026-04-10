"""Test / evaluate the YOLOv11 BEV pose model against the sskit LocSim metric.

This script mirrors the behaviour of mmpose/tools/test.py for the BEV task:

  mmpose flow                             this script
  ─────────────────────────────────────   ──────────────────────────────────────
  runner.val()   (val split)           →  run_val()  saves *_val_stats.json
  runner.test()  (test split)          →  run_test() loads score_threshold from val
  --challenge                          →  --split challenge
  results.json + metadata.json + zip   →  same outputs in save_dir/

LocSim evaluation is provided by sskit (pip install sskit>=0.2.0).
  - iou_type='locsim'       →  primary BEV AP metric
  - iou_type='locsim_bbox'  →  secondary BBox LocSim metric
  - position_from_keypoint_index=1 → keypoint 1 (pelvis_ground) is the BEV position

Usage examples:
    # Step 1 – always run val first (gets optimal score_threshold):
    python test_bev.py \\
        --weights runs/pose-bev/yolo11m-pose-bev-640/weights/best.pt \\
        --data    ultralytics/cfg/datasets/soccernet-synloc.yaml \\
        --split   val --imgsz 640

    # Step 2 – run test (uses score_threshold from step 1):
    python test_bev.py \\
        --weights runs/pose-bev/yolo11m-pose-bev-640/weights/best.pt \\
        --data    ultralytics/cfg/datasets/soccernet-synloc.yaml \\
        --split   test --imgsz 640

    # Challenge submission (format_only, no GT required):
    python test_bev.py \\
        --weights runs/pose-bev/yolo11m-pose-bev-640/weights/best.pt \\
        --data    ultralytics/cfg/datasets/soccernet-synloc.yaml \\
        --split   challenge --imgsz 640
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

# Make sure the repo root is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from ultralytics.models.yolo.pose.val_bev import BEVPoseValidator
from ultralytics.utils import LOGGER


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test/evaluate YOLOv11 BEV pose model with sskit LocSim metric"
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default="soccernet-synloc.yaml",
                        help="Path to dataset YAML (soccernet-synloc.yaml)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test", "challenge"],
                        help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--batch", type=int, default=32,
                        help="Inference batch size")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for NMS")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 0, '0,1', 'cpu', etc.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save results (default: auto from weights path)")
    parser.add_argument("--rect", action="store_true",
                        help="Rectangular inference — pad to actual aspect ratio instead of square")
    parser.add_argument("--run-val-first", action="store_true",
                        help="Automatically run val before test/challenge to get score_threshold")
    parser.add_argument(
        "--run-val-first-with-postprocess",
        action="store_true",
        help=(
            "When used with --run-val-first and --bev-postprocess-stats, run the warm-up VAL pass with "
            "y-band postprocess enabled. Default behavior runs VAL without postprocess to keep the original "
            "score_threshold estimation."
        ),
    )
    parser.add_argument(
        "--bev-postprocess-stats",
        type=str,
        default=None,
        help="Optional path to BEV val-calibrated y-band post-process stats JSON.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(weights: str, data: str, split: str, imgsz: int, batch: int,
             conf: float, iou: float, device: str, workers: int,
             save_dir: Path, rect: bool = False, bev_postprocess_stats: str | None = None) -> dict:
    """Run one evaluation pass for the given split.

    Mirrors a single runner.val() or runner.test() call in mmpose/tools/test.py.

    Args:
        weights:  Path to .pt checkpoint.
        data:     Dataset YAML path.
        split:    'val', 'test', or 'challenge'.
        imgsz:    Input image size.
        batch:    Inference batch size.
        conf:     Confidence threshold.
        iou:      IoU threshold for NMS.
        device:   Torch device string or None for auto.
        workers:  Dataloader workers.
        save_dir: Directory to write outputs to.
        rect:     Rectangular batching (pad to aspect ratio, not square).
        bev_postprocess_stats: Optional path to BEV post-processing stats JSON.

    Returns:
        dict of metric name → value (includes LocSim metrics if sskit is installed).
    """
    model = YOLO(weights)

    # Validator args — save_json=True causes predictions.json to be written,
    # which BEVPoseValidator.eval_json() then feeds to sskit LocSimCOCOeval.
    val_args = dict(
        data=data,
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        workers=workers,
        save_json=True,
        save_dir=str(save_dir),
        verbose=True,
        rect=rect,
    )
    if device is not None:
        val_args["device"] = device

    # phase is passed separately — not inside val_args — because ultralytics
    # get_cfg() rejects any key not in its known argument schema.
    validator = BEVPoseValidator(args=val_args, phase=split, bev_postprocess_stats=bev_postprocess_stats)
    validator(model=model.model)

    return validator.metrics


# ──────────────────────────────────────────────────────────────────────────────
# Submission packaging  (mirrors mmpose/tools/test.py lines 175-181)
# ──────────────────────────────────────────────────────────────────────────────

def package_submission(save_dir: Path, split: str):
    """Write results.json + metadata.json and zip them for challenge submission.

    Mirrors mmpose/tools/test.py:
        shutil.copyfile(prefix + '.keypoints.json', 'results.json')
        json.dump(dict(score_threshold=th, position_from_keypoint_index=1), ...)
        zipfile.ZipFile(...)
    """
    pred_json = save_dir / "predictions.json"
    if not pred_json.exists():
        LOGGER.warning(f"predictions.json not found at {pred_json}; skipping submission packaging.")
        return

    results_json = save_dir / "results.json"
    shutil.copyfile(pred_json, results_json)
    LOGGER.info(f"Copied predictions → {results_json}")

    # Load score_threshold from val stats (matches mmpose test.py line 177)
    val_stats_file = save_dir / "locsim_val_stats.json"
    score_threshold = None
    if val_stats_file.exists():
        with open(val_stats_file) as f:
            score_threshold = json.load(f)["stats"]["score_threshold"]
        LOGGER.info(f"Loaded score_threshold={score_threshold:.4f} from {val_stats_file}")
    else:
        LOGGER.warning(
            f"{val_stats_file} not found. metadata.json will not include score_threshold. "
            "Run --split val first."
        )

    metadata = {"position_from_keypoint_index": 1}
    if score_threshold is not None:
        metadata["score_threshold"] = score_threshold
    metadata_json = save_dir / "metadata.json"
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)
    LOGGER.info(f"Wrote metadata → {metadata_json}")

    zip_name = "challenge_submission.zip" if split == "challenge" else "test_submission.zip"
    zip_path = save_dir / zip_name
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(results_json, "results.json")
        zf.write(metadata_json, "metadata.json")
    LOGGER.info(f"Submission archive → {zip_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = weights_path.parent.parent / "eval" / args.split
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Saving results to {save_dir}")

    # ── Step 1: always run val first for test/challenge to get score_threshold ──
    # Mirrors mmpose test.py lines 172-173:  runner.val(); runner.test()
    if args.split in ("test", "challenge") and args.run_val_first:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("Running VAL pass first (to obtain score_threshold) ...")
        LOGGER.info("=" * 60)
        val_save_dir = save_dir.parent / "val"
        val_save_dir.mkdir(parents=True, exist_ok=True)
        val_bev_stats = args.bev_postprocess_stats
        if args.bev_postprocess_stats and not args.run_val_first_with_postprocess:
            LOGGER.info(
                "run-val-first: temporarily disabling BEV postprocess on VAL so score_threshold "
                "stays consistent with the original inference pipeline."
            )
            val_bev_stats = None
        evaluate(
            weights=args.weights, data=args.data, split="val",
            imgsz=args.imgsz, batch=args.batch, conf=args.conf,
            iou=args.iou, device=args.device, workers=args.workers,
            save_dir=val_save_dir, rect=args.rect, bev_postprocess_stats=val_bev_stats,
        )
        # Copy val_stats files into save_dir so BEVPoseValidator can find them
        for stats_file in val_save_dir.glob("*_val_stats.json"):
            dst = save_dir / stats_file.name
            shutil.copyfile(stats_file, dst)
            LOGGER.info(f"Copied {stats_file.name} → {save_dir}")

    # ── Step 2: run target split ──────────────────────────────────────────────
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info(f"Running {args.split.upper()} evaluation ...")
    LOGGER.info("=" * 60)
    metrics = evaluate(
        weights=args.weights, data=args.data, split=args.split,
        imgsz=args.imgsz, batch=args.batch, conf=args.conf,
        iou=args.iou, device=args.device, workers=args.workers,
        save_dir=save_dir, rect=args.rect, bev_postprocess_stats=args.bev_postprocess_stats,
    )

    # ── Step 3: package submission zip (for test / challenge) ─────────────────
    if args.split in ("test", "challenge"):
        # Look for val_stats in the val dir if --run-val-first was used,
        # otherwise expect them already in save_dir
        if args.run_val_first:
            val_stats_src = save_dir.parent / "val" / "locsim_val_stats.json"
            val_stats_dst = save_dir / "locsim_val_stats.json"
            if val_stats_src.exists() and not val_stats_dst.exists():
                shutil.copyfile(val_stats_src, val_stats_dst)
        package_submission(save_dir, args.split)

    # ── Summary ───────────────────────────────────────────────────────────────
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Evaluation complete.")
    LOGGER.info(f"Results saved to: {save_dir}")
    if hasattr(metrics, "results_dict"):
        for k, v in metrics.results_dict.items():
            LOGGER.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
