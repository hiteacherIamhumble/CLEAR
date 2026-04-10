#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.head import Pose26MLPRefine, Pose26Refine  # noqa: F401
from ultralytics.nn.modules.transformer import P3CrossScaleDeformAttn  # noqa: F401


@dataclass
class GTObject:
    bbox_xyxy: list[float]
    pelvis: list[float]
    pelvis_ground: list[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate first 10 pelvis-proj test images for GT, baseline, and stage1/two-stage")
    p.add_argument("--images-dir", type=Path, default=ROOT / "my_database/images/test")
    p.add_argument("--labels-dir", type=Path, default=ROOT / "my_database/labels/test")
    p.add_argument("--stage1-ckpt", type=Path, required=True)
    p.add_argument("--baseline-ckpt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="1")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--num-images", type=int, default=10)
    return p.parse_args()


def first_n_images(images_dir: Path, n: int) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]


def load_gt(label_path: Path, img_w: int, img_h: int) -> list[GTObject]:
    gts: list[GTObject] = []
    if not label_path.exists():
        return gts
    text = label_path.read_text().strip()
    if not text:
        return gts
    for line in text.splitlines():
        vals = line.strip().split()
        if len(vals) < 11:
            continue
        cx, cy, bw, bh = map(float, vals[1:5])
        px, py, pv = map(float, vals[5:8])
        gx, gy, gv = map(float, vals[8:11])
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
        gts.append(
            GTObject(
                bbox_xyxy=[x1, y1, x2, y2],
                pelvis=[px * img_w, py * img_h, pv],
                pelvis_ground=[gx * img_w, gy * img_h, gv],
            )
        )
    return gts


def run_inference(model_path: Path, image_paths: list[Path], device: str, imgsz: int, conf: float, iou: float) -> dict[str, list[dict]]:
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(p) for p in image_paths],
        task="pose",
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        allow_oob_labels=True,
        verbose=True,
        stream=False,
    )
    out: dict[str, list[dict]] = {}
    for idx, r in enumerate(results):
        dets: list[dict] = []
        if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None and r.keypoints.data is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            kpts = r.keypoints.data.cpu().numpy()
            for i in range(len(xyxy)):
                dets.append(
                    {
                        "bbox_xyxy": [float(v) for v in xyxy[i]],
                        "score": float(confs[i]),
                        "class_id": int(cls[i]),
                        "pelvis": [float(kpts[i, 0, 0]), float(kpts[i, 0, 1]), float(kpts[i, 0, 2])],
                        "pelvis_ground": [float(kpts[i, 1, 0]), float(kpts[i, 1, 1]), float(kpts[i, 1, 2])],
                    }
                )
        out[image_paths[idx].name] = dets
    return out


def draw_pair(img: np.ndarray, p1: list[float], p2: list[float], line_color: tuple[int, int, int], p1_color: tuple[int, int, int], p2_color: tuple[int, int, int], thickness: int) -> None:
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))
    cv2.line(img, (x1, y1), (x2, y2), line_color, thickness, cv2.LINE_AA)
    cv2.circle(img, (x1, y1), thickness + 2, p1_color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2, y2), thickness + 2, p2_color, -1, cv2.LINE_AA)


def draw_box(img: np.ndarray, xyxy: list[float], color: tuple[int, int, int], thickness: int) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def add_header(img: np.ndarray, title: str) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (18, 18), (860, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
    cv2.putText(img, title, (34, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        "GT pair: red/orange | Pred pair: blue/yellow | Pred box: green",
        (34, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )


def save_gt_image(src: np.ndarray, gts: list[GTObject], out_path: Path, title: str) -> None:
    img = src.copy()
    thickness = max(2, round(min(img.shape[:2]) / 900))
    for gt in gts:
        draw_box(img, gt.bbox_xyxy, (0, 0, 255), thickness)
        if gt.pelvis[2] > 0 and gt.pelvis_ground[2] > 0:
            draw_pair(img, gt.pelvis, gt.pelvis_ground, (50, 50, 220), (0, 0, 255), (0, 180, 255), thickness)
    add_header(img, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def save_pred_image(src: np.ndarray, gts: list[GTObject], preds: list[dict], out_path: Path, title: str) -> None:
    img = src.copy()
    thickness = max(2, round(min(img.shape[:2]) / 900))
    for gt in gts:
        if gt.pelvis[2] > 0 and gt.pelvis_ground[2] > 0:
            draw_pair(img, gt.pelvis, gt.pelvis_ground, (50, 50, 220), (0, 0, 255), (0, 180, 255), thickness)
    for pred in preds:
        draw_box(img, pred["bbox_xyxy"], (0, 255, 0), thickness)
        if pred["pelvis"][2] > 0 and pred["pelvis_ground"][2] > 0:
            draw_pair(img, pred["pelvis"], pred["pelvis_ground"], (255, 200, 0), (255, 120, 0), (255, 255, 0), thickness)
    add_header(img, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def pair_distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def match_preds_to_gt(gts: list[GTObject], preds: list[dict]) -> dict:
    remaining = set(range(len(gts)))
    matched = []
    for pred_idx, pred in sorted(enumerate(preds), key=lambda x: x[1]["score"], reverse=True):
        best_gt = None
        best_score = None
        for gt_idx in remaining:
            gt = gts[gt_idx]
            iou = iou_xyxy(gt.bbox_xyxy, pred["bbox_xyxy"])
            if iou < 0.3:
                continue
            dist = pair_distance(gt.pelvis_ground, pred["pelvis_ground"])
            score = (iou, -dist)
            if best_score is None or score > best_score:
                best_score = score
                best_gt = gt_idx
        if best_gt is None:
            continue
        gt = gts[best_gt]
        remaining.remove(best_gt)
        matched.append(
            {
                "gt_index": best_gt,
                "pred_index": pred_idx,
                "iou": iou_xyxy(gt.bbox_xyxy, pred["bbox_xyxy"]),
                "pelvis_ground_dist": pair_distance(gt.pelvis_ground, pred["pelvis_ground"]),
            }
        )
    mean_dist = float(sum(m["pelvis_ground_dist"] for m in matched) / len(matched)) if matched else None
    return {
        "gt_count": len(gts),
        "pred_count": len(preds),
        "matched_count": len(matched),
        "missed_gt_count": len(gts) - len(matched),
        "fp_count": len(preds) - len(matched),
        "mean_pelvis_ground_dist": mean_dist,
        "matches": matched,
    }


def select_best_improvement(per_image: list[dict]) -> dict:
    return max(
        per_image,
        key=lambda item: (
            item["two_stage"]["matched_count"] - item["baseline"]["matched_count"],
            item["baseline"]["missed_gt_count"] - item["two_stage"]["missed_gt_count"],
            item["baseline"]["pred_count"] - item["two_stage"]["pred_count"],
            -(item["two_stage"]["mean_pelvis_ground_dist"] or 1e9),
        ),
    )


def main() -> None:
    args = parse_args()
    image_paths = first_n_images(args.images_dir, args.num_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_preds = run_inference(args.baseline_ckpt, image_paths, args.device, args.imgsz, args.conf, args.iou)
    stage1_preds = run_inference(args.stage1_ckpt, image_paths, args.device, args.imgsz, args.conf, args.iou)

    summary = {
        "images": [],
        "config": {
            "images_dir": str(args.images_dir),
            "labels_dir": str(args.labels_dir),
            "baseline_ckpt": str(args.baseline_ckpt),
            "two_stage_ckpt": str(args.stage1_ckpt),
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
        },
    }

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h, w = img.shape[:2]
        filename = img_path.name
        gts = load_gt(args.labels_dir / f"{img_path.stem}.txt", w, h)
        base_dets = baseline_preds.get(filename, [])
        stage1_dets = stage1_preds.get(filename, [])

        gt_path = args.output_dir / "gt" / filename
        base_path = args.output_dir / "baseline" / filename
        stage1_path = args.output_dir / "two_stage" / filename

        save_gt_image(img, gts, gt_path, f"{filename} | GT")
        save_pred_image(img, gts, base_dets, base_path, f"{filename} | Baseline")
        save_pred_image(img, gts, stage1_dets, stage1_path, f"{filename} | Two-stage")

        image_summary = {
            "file_name": filename,
            "gt_count": len(gts),
            "baseline": match_preds_to_gt(gts, base_dets),
            "two_stage": match_preds_to_gt(gts, stage1_dets),
            "outputs": {
                "gt": str(gt_path),
                "baseline": str(base_path),
                "two_stage": str(stage1_path),
            },
        }
        summary["images"].append(image_summary)

    best = select_best_improvement(summary["images"])
    summary["best_improvement"] = {
        "file_name": best["file_name"],
        "reason": {
            "baseline_matched": best["baseline"]["matched_count"],
            "two_stage_matched": best["two_stage"]["matched_count"],
            "baseline_missed_gt": best["baseline"]["missed_gt_count"],
            "two_stage_missed_gt": best["two_stage"]["missed_gt_count"],
            "baseline_pred_count": best["baseline"]["pred_count"],
            "two_stage_pred_count": best["two_stage"]["pred_count"],
            "baseline_mean_pelvis_ground_dist": best["baseline"]["mean_pelvis_ground_dist"],
            "two_stage_mean_pelvis_ground_dist": best["two_stage"]["mean_pelvis_ground_dist"],
        },
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output_dir": str(args.output_dir), "summary_json": str(summary_path), "best_improvement": summary["best_improvement"]}, indent=2))


if __name__ == "__main__":
    main()
