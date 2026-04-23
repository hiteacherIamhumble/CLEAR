# Pelvis Projection Task + Current Solution Handoff

Updated: 2026-04-09 (UTC)

This document is a summary of:
1. The task definition and dataset/meta/evaluation setup in this folder.
2. The current `yolo26x-pose` solution, including code modifications, finetuning recipe, and latest metrics.
3. Key contents from referenced files/artifacts.

---

## 1) Task Definition

### 1.1 Problem statement

The current pipeline is a **pose-style detection task with 2 keypoints per person**:
- Keypoint 0: `pelvis`
- Keypoint 1: `pelvis_ground` (ground projection of pelvis)

Each detection outputs:
- `bbox` (person box)
- `pelvis` keypoint `(x, y, v)`
- `pelvis_ground` keypoint `(x, y, v)`

This is not COCO-17 keypoint pose; the model head is adapted to `kpt_shape=[2,3]`.

### 1.2 Input format and sizes

Dataset config (`ultralytics/cfg/datasets/pelvis-proj-2pt.yaml`) defines:

```yaml
path: /root/autodl-tmp/ultralytics/my_database
train: images/train
val: images/val
test: images/test

kpt_shape: [2, 3]

allow_oob_keypoints: true
allow_oob_labels: true
flip_idx: [0, 1]
sigmas: [0.089, 0.089]

names:
  0: person

kpt_names:
  0:
    - pelvis
    - pelvis_ground
```

Label row format from YAML comments:

```text
class cx cy w h p_x p_y p_v g_x g_y g_v
```

Input size used for training/inference in current runs:
- `imgsz=960`

### 1.3 Dataset split metadata (train/val/test)

Current folder layout:
- `my_database/images/{train,val,test}`
- `my_database/labels/{train,val,test}`

Images in `my_database/images/*` are symlinks (example):
- `my_database/images/train/000048.jpg -> /root/autodl-tmp/sskit/database/train/000048.jpg`

Split statistics:

| Split | Images | Label files | Labeled instances (label rows) | Image storage size (real target dirs) |
|---|---:|---:|---:|---:|
| train | 42,504 | 42,504 | 668,259 | 59G |
| val | 6,777 | 6,777 | 109,351 | 9.6G |
| test | 9,309 | 9,309 | 148,164 | 14G |

Split roles:
- `train`: optimization/finetuning
- `val`: checkpoint monitoring and LocSim threshold selection
- `test`: final report (LocSim + YOLO metrics)

### 1.4 Final evaluation metrics used

The project evaluates both standard YOLO and custom LocSim metrics.

Standard YOLO metrics:
- `precision(B)`, `recall(B)`, `mAP50(B)`, `mAP50-95(B)` for bbox
- `precision(P)`, `recall(P)`, `mAP50(P)`, `mAP50-95(P)` for keypoints

Custom metrics:
- `precision`, `recall`, `f1` at LocSim=0.5 with selected score threshold
- `frame_accuracy`
- `mAP-LocSim` over LocSim IoU-like range (`0.50:0.95`)
- Also `LOCSIM_BBOX` variant is reported

---

## 2) Current Solution

### 2.1 Base model and adaptation

Base checkpoint:
- `yolo26x-pose.pt`

Adaptation behavior seen in training log:
- `Overriding model.yaml kpt_shape=[17, 3] with kpt_shape=[2, 3]`
- Pose head line shows 2-keypoint setup:
  - `Pose26 [1, [2, 3], ...]`
- Pretrained transfer:
  - `Transferred 1239/1263 items from pretrained weights`

### 2.2 How `yolo26x-pose` is modified in code

Modified files (`git status --short`):
- `ultralytics/cfg/default.yaml`
- `ultralytics/data/dataset.py`
- `ultralytics/data/utils.py`
- `ultralytics/models/yolo/detect/predict.py`
- `ultralytics/models/yolo/detect/val.py`
- `ultralytics/models/yolo/pose/predict.py`
- `ultralytics/models/yolo/pose/val.py`
- `ultralytics/utils/ops.py`
- plus new `tools/*`, `my_database/*`, and dataset YAML.

Key modifications:

1) New runtime switch for out-of-bounds labels

```yaml
# ultralytics/cfg/default.yaml
allow_oob_labels: False
```

2) Dataset cache/version and verification path carry OOB flag

```python
# ultralytics/data/dataset.py
DATASET_CACHE_VERSION = "1.0.4"
...
repeat(bool(self.data.get("allow_oob_labels", self.data.get("allow_oob_keypoints", False))))
```

```python
# ultralytics/data/utils.py (verify_image_label)
if len(args) == 9:
    ..., allow_oob_keypoints = args
...
if not allow_oob_keypoints:
    assert box_xywh.max() <= 1.01
    assert box_xywh.min() >= -0.01
    assert kpt_points.max() <= 1.01
    assert kpt_points.min() >= -0.01
```

3) Geometry scaling now supports optional no-clipping

```python
# ultralytics/utils/ops.py
def scale_boxes(..., clip: bool = True):
...
if xywh or not clip:
    return boxes
return clip_boxes(boxes, img0_shape)
```

```python
# ultralytics/utils/ops.py
def scale_coords(..., clip: bool = True):
...
if clip:
    coords = clip_coords(coords, img0_shape)
```

4) Predict/val paths propagate `allow_oob_labels`

```python
# detect/predict.py
allow_oob = bool(getattr(self.args, "allow_oob_labels", False))
pred[:, :4] = ops.scale_boxes(..., clip=not allow_oob)
```

```python
# detect/val.py
allow_oob = bool(self.data.get("allow_oob_labels", self.data.get("allow_oob_keypoints", False)))
"bboxes": ops.scale_boxes(..., clip=not allow_oob)
```

```python
# pose/predict.py
allow_oob = bool(getattr(self.args, "allow_oob_labels", False))
pred_kpts = ops.scale_coords(..., clip=not allow_oob)
```

```python
# pose/val.py
allow_oob = bool(self.data.get("allow_oob_labels", self.data.get("allow_oob_keypoints", False)))
"kpts": ops.scale_coords(..., clip=not allow_oob)
```

### 2.3 Finetuning procedure currently used

Main training driver:
- `tools/train_pelvis_proj_4k.py`

Key training script contents:

```python
# defaults in parse_args()
--model yolo26x-pose.pt
--data ultralytics/cfg/datasets/pelvis-proj-2pt.yaml
--imgsz 960
--epochs 20
--batch 64
--optimizer MuSGD
--lr0 0.01
--lrf 0.1
--pose 18.0
--kobj 2.0
--mosaic 0.2
--translate 0.03
--scale 0.2
--close-mosaic 5
```

```python
# explicit lock in script
if args.optimizer.lower() != "musgd":
    raise ValueError("pipeline is locked to MuSGD ...")
```

```python
# train call
model.train(task="pose", ..., optimizer="MuSGD", pose=args.pose, kobj=args.kobj, imgsz=args.imgsz, ...)
```

Main run artifact:
- `/root/autodl-tmp/ultralytics/runs/pose/runs/pelvis-proj/yolo26x-pelvis-proj-960-e20-b64-pose18.0-kobj2.0-optMuSGD-m0.2-20260407_235150`

`args.yaml` key values from that run:

```yaml
task: pose
mode: train
model: /root/autodl-tmp/ultralytics/yolo26x-pose.pt
data: /root/autodl-tmp/ultralytics/ultralytics/cfg/datasets/pelvis-proj-2pt.yaml
epochs: 20
batch: 64
imgsz: 960
optimizer: MuSGD
lr0: 0.01
lrf: 0.1
pose: 18.0
kobj: 2.0
mosaic: 0.2
translate: 0.03
scale: 0.2
close_mosaic: 5
rect: true
amp: true
patience: 5
allow_oob_labels: false
```

### 2.4 Current metrics/results

#### A) Main training run final validation metrics (epoch 20)

From `results.csv` final row:

```csv
20,...,metrics/precision(B)=0.97073,metrics/recall(B)=0.97415,metrics/mAP50(B)=0.98984,metrics/mAP50-95(B)=0.87477,metrics/precision(P)=0.97502,metrics/recall(P)=0.97915,metrics/mAP50(P)=0.99306,metrics/mAP50-95(P)=0.99236,...
```

Log summary (rounded) at end:

```text
all 6725 images, 108619 instances
Box:  P=0.971 R=0.974 mAP50=0.990 mAP50-95=0.875
Pose: P=0.975 R=0.979 mAP50=0.993 mAP50-95=0.992
```

#### B) LocSim evaluation results (val/test)

LocSim eval run:
- `/root/autodl-tmp/ultralytics/runs/pose-bev/locsim_eval_20260408_042737`
- log: `runs/pelvis-proj/nohup_locsim_20260408_042737.log`

VAL LocSim (from log block):
- precision: `0.9365`
- recall: `0.9100`
- f1: `0.9231`
- frame_accuracy: `0.4830`
- mAP-LocSim: `0.8324`
- score_threshold: `0.4912`

TEST LocSim (from log block):
- precision: `0.9427`
- recall: `0.8700`
- f1: `0.9049`
- frame_accuracy: `0.4648`
- mAP-LocSim: `0.7834`
- score_threshold: `0.4912`

TEST LOCSIM_BBOX (from log block):
- precision: `0.8713`
- recall: `0.8100`
- f1: `0.8395`
- frame_accuracy: `0.2460`
- mAP-LocSim: `0.4772`
- score_threshold: `0.4104`

Also reported in same test run (YOLO metrics):
- `precision(B)=0.9733`
- `recall(B)=0.9595`
- `mAP50(B)=0.9854`
- `mAP50-95(B)=0.8612`
- `precision(P)=0.9666`
- `recall(P)=0.9513`
- `mAP50(P)=0.9756`
- `mAP50-95(P)=0.8773`
- `fitness=1.7385`

### 2.5 Important artifact caveat

In `locsim_eval_20260408_042737`, these pairs are identical (overwritten with test values):
- `locsim_val_stats.json` == `locsim_test_stats.json`
- `locsim_bbox_val_stats.json` == `locsim_bbox_test_stats.json`

So for true VAL LocSim values, use the **log text** (`nohup_locsim_20260408_042737.log`) instead of `*_val_stats.json`.

---

## 3) Referenced Files and Key Contents

### 3.1 Core dataset/config files

- `ultralytics/cfg/datasets/pelvis-proj-2pt.yaml`
  - Defines split roots, `kpt_shape: [2,3]`, keypoint names, and OOB settings.

- `ultralytics/cfg/default.yaml`
  - Adds/uses `allow_oob_labels`.

### 3.2 Core training/inference scripts

- `tools/train_pelvis_proj_4k.py`
  - Enforces MuSGD pipeline.
  - Validates `kpt_shape=[2,3]`.
  - Trains YOLO pose with custom losses (`pose=18.0`, `kobj=2.0`) and augmentation settings.

- `tools/predict_pelvis_proj.py`
  - Runs pose inference.
  - Validates that each detection has exactly 2 keypoints.
  - Exports JSON rows with `bbox_xyxy`, `score`, `class_id`, `pelvis`, `pelvis_ground`.

Example key export structure:

```python
{
  "bbox_xyxy": [...],
  "score": ...,
  "class_id": ...,
  "pelvis": [x, y, v],
  "pelvis_ground": [x, y, v]
}
```

### 3.3 Core Ultralytics internal modifications

- `ultralytics/data/dataset.py` (cache version + OOB flag propagation)
- `ultralytics/data/utils.py` (label verification supports optional OOB coords)
- `ultralytics/utils/ops.py` (`scale_boxes` and `scale_coords` support `clip` toggle)
- `ultralytics/models/yolo/detect/{predict.py,val.py}` (pass `clip=not allow_oob`)
- `ultralytics/models/yolo/pose/{predict.py,val.py}` (pass `clip=not allow_oob`)

### 3.4 Run/eval artifacts used for current numbers

- Training log:
  - `runs/pelvis-proj/nohup_train_eval_e20_b64_MuSGD_20260407_235150.log`
- LocSim eval log:
  - `runs/pelvis-proj/nohup_locsim_20260408_042737.log`
- Main run args/results:
  - `/root/autodl-tmp/ultralytics/runs/pose/runs/pelvis-proj/yolo26x-pelvis-proj-960-e20-b64-pose18.0-kobj2.0-optMuSGD-m0.2-20260407_235150/args.yaml`
  - `/root/autodl-tmp/ultralytics/runs/pose/runs/pelvis-proj/yolo26x-pelvis-proj-960-e20-b64-pose18.0-kobj2.0-optMuSGD-m0.2-20260407_235150/results.csv`
- LocSim output directory:
  - `/root/autodl-tmp/ultralytics/runs/pose-bev/locsim_eval_20260408_042737/`

---

## 4) One-paragraph transfer summary

The current project finetunes `yolo26x-pose.pt` into a 2-keypoint pelvis localization model (`pelvis`, `pelvis_ground`) using `imgsz=960`, MuSGD, and custom pose/kobj loss gains on a large train/val/test split (42.5k/6.8k/9.3k images). The codebase has been modified to optionally allow out-of-bounds labels and disable clipping during coordinate scaling, which is wired through detect/pose predict and validation paths. The latest strong validation metrics are Box mAP50-95 `0.8748` and Pose mAP50-95 `0.9924`; LocSim reports `frame_accuracy=0.4830` on val and `0.4648` on test with `mAP-LocSim=0.8324` (val) and `0.7834` (test), with a known caveat that `locsim_val_stats.json` appears overwritten by test values in the saved artifacts.
