# CLEAR Pipeline

This document explains the pipeline used by:

- `/root/autodl-tmp/ultralytics/tools/run_stage2_continue30_more_eval.sh`

It covers:

- training pipeline
- inference / evaluation pipeline
- how image input flows through the model
- where camera parameters are used
- how validation gets the threshold
- how test reuses that threshold
- train / val / test sample counts

## 1. Entry Script Overview

The stage-2 entrypoint is:

- `tools/run_stage2_continue30_more_eval.sh`

What it does:

1. Continues stage-2 fine-tuning from an existing checkpoint (`INIT_WEIGHTS`).
2. Trains for `MORE_EPOCHS` more epochs, default `30`.
3. Saves `best.pt` and `last.pt`.
4. Runs LocSim evaluation for both checkpoints by calling `tools/test_bev_safe.py`.
5. Forces a validation pass before test via `--run-val-first`, so test can use the threshold computed on val.
6. Writes a summary JSON with checkpoint paths and eval metrics.

Default settings in this script:

- train batch: `48`
- eval batch: `16`
- image size: `960`
- more epochs: `30`
- patience: `8`
- final split: `test`
- train optimizer: `MuSGD`
- train LR: `0.0005`

## 2. Dataset and Sample Counts

### Training dataset config

Training uses:

- `ultralytics/cfg/datasets/pelvis-proj-2pt.yaml`

This dataset defines:

- root: `/root/autodl-tmp/ultralytics/my_database`
- train: `images/train`
- val: `images/val`
- test: `images/test`
- `kpt_shape: [2, 3]`

The two keypoints are:

1. `pelvis`
2. `pelvis_ground`

Label format is YOLO pose format:

- `class cx cy w h p_x p_y p_v g_x g_y g_v`

### Eval dataset config

BEV / LocSim evaluation uses:

- `ultralytics/cfg/datasets/soccernet-synloc.yaml`

This points to the same image/label root, but evaluation additionally expects COCO-style annotation JSON files under:

- `/root/autodl-tmp/ultralytics/my_database/annotations`

### Sample counts in the current dataset

From the current repo data:

- train images: `42,504`
- train label files: `42,504`
- val images: `6,777`
- val label files: `6,777`
- val annotation JSON images: `6,777`
- val annotations: `109,351`
- test images: `9,309`
- test label files: `9,309`
- test annotation JSON images: `9,309`
- test annotations: `148,164`

Important detail:

- `train` is standard YOLO image + label training data.
- `val` and `test` also have COCO-style annotation JSONs for BEV evaluation.

## 2A. Full Two-Stage Training Flow

If you want the full training story, the main orchestration script is also:

- `/root/autodl-tmp/ultralytics/tools/run_stage1_stage2_locsim_strict.sh`

This script trains and evaluates two stages in sequence:

1. Stage 1 training
2. Stage 2 training initialized from Stage 1 best checkpoint
3. LocSim evaluation for both stage outputs

### Stage 1 training details

Stage 1 uses:

- model: `ultralytics/cfg/models/26/yolo26x-pose-aifi-p5-p3csda.yaml`
- init weights: `runs/pelvis-proj/stageA_best.pt`
- epochs: `20` by default (`STAGE1_EPOCHS`)
- batch: `48`
- imgsz: `960`
- patience: `5`
- optimizer: `MuSGD`
- lr0: `0.01`
- lrf: `0.1`
- pose: `18.0`
- kobj: `2.0`
- kpt1-weight: `2.0`
- mosaic: `0.2`
- translate: `0.03`
- scale: `0.2`
- close-mosaic: `5`

So Stage 1 is the first task-specific training step on top of `stageA_best.pt`.

### Stage 2 training details

Stage 2 uses:

- model: `ultralytics/cfg/models/26/yolo26x-pose-aifi-p5-p3csda-mlp-refine.yaml`
- init weights: Stage 1 best checkpoint
- epochs: `20` by default (`STAGE2_EPOCHS`)
- batch: `48`
- imgsz: `960`
- patience: `5`
- optimizer: `MuSGD`
- lr0: `0.0005`
- lrf: `0.1`
- pose: `18.0`
- kobj: `2.0`
- kpt1-weight: `2.0`
- refine: `0.5`
- refine-prior: `0.1`
- mosaic: `0.2`
- translate: `0.03`
- scale: `0.2`
- close-mosaic: `0`

So in the strict two-stage script, the default training schedule is:

- Stage 1: `20` epochs
- Stage 2: `20` epochs
- total across both stages: `40` epochs

### Relation to `run_stage2_continue30_more_eval.sh`

The other script you originally pointed to:

- `tools/run_stage2_continue30_more_eval.sh`

is a continuation stage-2 script. It starts from an already-trained stage-2 checkpoint and adds:

- `30` more epochs by default

So that script is not the original Stage 1 + Stage 2 schedule. It is a later continuation of Stage 2.

## 4A. Stage 1 vs Stage 2 Architecture Difference

For the strict two-stage pipeline, the actual stage pair is:

- Stage 1: `yolo26x-pose-aifi-p5-p3csda.yaml`
- Stage 2: `yolo26x-pose-aifi-p5-p3csda-mlp-refine.yaml`

### What Stage 1 added

Relative to a more standard YOLO26 pose head, Stage 1 adds two main architectural ideas:

1. `AIFI` on P5
2. `P3CrossScaleDeformAttn` before the pose head

In other words, Stage 1 is the first architecture that introduces:

- stronger high-level context at P5 through `AIFI`
- explicit cross-scale fusion into the P3 branch through deformable attention

Its final head is still the base pose head:

- `Pose26`

So Stage 1 = feature/context upgrade, but no refine head yet.

### What Stage 2 added

Stage 2 keeps the full Stage 1 structure unchanged:

- same backbone
- same P5 `AIFI`
- same `P3CrossScaleDeformAttn`
- same P3/P4/P5 feature path

The only architectural replacement is at the final head:

- Stage 1 head: `Pose26`
- Stage 2 head: `Pose26MLPRefine`

So Stage 2 adds:

1. an MLP-based keypoint refinement head
2. refinement-specific training losses (`refine`, `refine-prior`)
3. a second training phase initialized from Stage 1 best

### Minimal architecture diff

Stage 1 final lines:

- `[[16, 19, 23], 1, P3CrossScaleDeformAttn, [256, 8, 4, 4.0]]`
- `[[24, 19, 23], 1, Pose26, [nc, kpt_shape]]`

Stage 2 final lines:

- `[[16, 19, 23], 1, P3CrossScaleDeformAttn, [256, 8, 4, 4.0]]`
- `[[24, 19, 23], 1, Pose26MLPRefine, [nc, kpt_shape]]`

So the structural diff is intentionally small and targeted:

- Stage 1 adds the stronger feature fusion stack
- Stage 2 adds the refine head on top of that stack

### Practical interpretation

You can think of the two stages like this:

- Stage 1 teaches the network to produce a stronger coarse bbox + 2-keypoint prediction using the improved AIFI + cross-scale-attention architecture.
- Stage 2 starts from that Stage 1 solution and adds a lightweight local correction module to refine keypoint coordinates, especially the BEV-critical `pelvis_ground` point.

## 3. What Is Actually Input to the Model?

### Short answer

The model input is the image tensor.

The camera parameters are not passed into the neural network forward path in this repo.

### Evidence from the code

Training uses the normal Ultralytics pose dataloader:

- `ultralytics/data/build.py`
- `ultralytics/data/dataset.py`

That dataloader builds batches containing image tensors and YOLO pose labels:

- image
- class
- bbox
- keypoints

There is no batch field carrying:

- `camera_matrix`
- `dist_poly`
- `undist_poly`

Those camera fields exist only in the COCO-style `annotations/val.json` and `annotations/test.json` files used by the BEV evaluator.

## 4. Stage-2 Model Structure

In `run_stage2_continue30_more_eval.sh`, Stage 2 uses:

- `ultralytics/cfg/models/26/yolo26x-pose-aifi-p5-p3csda-mlp-refine.yaml`

Key points:

- backbone builds multi-scale image features
- P5 goes through `AIFI`
- P3/P4/P5 are fused by `P3CrossScaleDeformAttn`
- final head is `Pose26MLPRefine`

The YAML still says `kpt_shape: [17, 3]`, but the trainer overrides it with the dataset shape:

- dataset shape: `[2, 3]`
- actual trained keypoints: `pelvis`, `pelvis_ground`

This override happens in `PoseModel` construction, where `data_kpt_shape` replaces the YAML `kpt_shape`.

## 5. Training Pipeline

### 5.1 Launcher

The shell script calls:

- `python tools/train_pelvis_proj_4k.py`

That script:

1. loads the model YAML
2. loads `INIT_WEIGHTS`
3. validates that dataset `kpt_shape` is `[2, 3]`
4. calls `model.train(...)` with Ultralytics pose training

### 5.2 Training inputs

Each training sample contains:

- one image
- one or more YOLO pose labels
- each label has:
  - bbox
  - keypoint 0: `pelvis`
  - keypoint 1: `pelvis_ground`

The dataset allows out-of-bounds labels/keypoints:

- `allow_oob_labels: true`
- `allow_oob_keypoints: true`

This matters because partially visible players can have pelvis / ground-projection points outside the image.

### 5.3 Image preprocessing and augmentation

For training:

- rectangular batching is enabled (`--rect`)
- augmentations come from standard Ultralytics pose transforms
- the stage-2 script sets:
  - `mosaic=0.0`
  - `translate=0.01`
  - `scale=0.05`
  - `close_mosaic=5`

So this run is a relatively conservative continuation fine-tune, not an aggressive augmentation run.

### 5.4 Forward path

For one image:

1. image is loaded by `YOLODataset`
2. labels are parsed from YOLO pose txt files
3. image is transformed and batched
4. model backbone produces multi-scale feature maps
5. head fuses P3/P4/P5 features
6. head predicts:
   - bbox
   - class score
   - 2 keypoints
7. `Pose26MLPRefine` runs an extra lightweight MLP refiner on top-scoring anchors

The refine head behavior:

- decodes coarse keypoints into image coordinates
- samples local P3 features around those keypoints
- concatenates sampled feature with local relative grid coordinates
- predicts a 2D delta with an MLP
- adds the delta to the coarse keypoint coordinates

The top-k refine limits for this script are:

- train top-k: `256`
- eval top-k: `100`
- hidden dim: `128`

These are passed by environment variables:

- `POSE26_MLP_REFINE_TOPK_TRAIN`
- `POSE26_MLP_REFINE_TOPK_EVAL`
- `POSE26_MLP_REFINE_HIDDEN`

### 5.5 Training losses

The run enables standard pose losses plus refinement losses:

- box loss
- cls loss
- dfl loss
- pose loss
- keypoint objectness loss
- refine loss
- refine prior loss

The shell script sets:

- `pose=18.0`
- `kobj=2.0`
- `kpt1-weight=2.0`
- `refine=0.1`
- `refine-prior=0.1`

`kpt1-weight=2.0` is especially important because keypoint index 1 is `pelvis_ground`, which is the point later used for BEV localization.

## 6. Inference Pipeline

### 6.1 Plain model inference

For an input image:

1. image is resized / padded to the configured inference size (`960` here)
2. model produces bbox, class/confidence, and 2 keypoints
3. the refine head replaces the top-scoring coarse keypoints with refined keypoints
4. predictions are scaled back to original image coordinates

At this stage, the output is still image-space prediction:

- bbox in image coordinates
- `pelvis` in image coordinates
- `pelvis_ground` in image coordinates

The model itself does not convert to BEV coordinates.

### 6.2 JSON export for BEV evaluation

During BEV evaluation, `BEVPoseValidator` writes:

- `predictions.json`

Each prediction contains normal COCO-style detection fields, including keypoints.

For BEV scoring:

- `position_from_keypoint_index = 1`

So the evaluator uses the second keypoint:

- `pelvis_ground`

as the image-space point that should represent the player location on the field.

## 7. Where Camera Parameters Enter

### Important clarification

Camera parameters do not enter model training or model inference in this codebase.

They enter only in BEV evaluation.

### Where they live

The `val.json` / `test.json` annotation files contain per-image fields such as:

- `camera_matrix`
- `dist_poly`
- `undist_poly`

Ground-truth annotations also contain:

- `position_on_pitch`
- `keypoints_3d`

### How they are used

Inside `sskit/sskit/sskit/coco.py`, `LocSimCOCOeval.computeIoU()` does:

1. takes the predicted image-space keypoint selected by `position_from_keypoint_index`
2. normalizes it using image width/height
3. calls `image_to_ground(camera_matrix, undist_poly, ...)`
4. gets predicted BEV / pitch coordinates
5. compares them with GT `position_on_pitch`
6. converts BEV distance into a LocSim similarity

So the camera model is used by the evaluator to project predicted image coordinates onto the field, not by the network to produce the prediction.

## 8. Validation Pipeline

### 8.1 What script is called

The stage-2 shell script calls:

- `tools/test_bev_safe.py`

That wrapper only ensures custom model classes are registered, then runs:

- `test_bev.py`

### 8.2 Validation flow

For validation:

1. `test_bev.py` loads the checkpoint with `YOLO(weights)`
2. constructs `BEVPoseValidator`
3. runs the model over the `val` split
4. writes `predictions.json`
5. calls `sskit` LocSim evaluation through `BEVPoseValidator.eval_json()`

### 8.3 How validation gets the threshold

This is the key logic.

During validation, `LocSimCOCOeval.summarize()` in `sskit` does:

1. compute precision / recall / F1 on LocSim at threshold `0.5`
2. if no score threshold was pre-specified:
   - find the index `i` where `f1_50` is maximal
   - set
     - `threshold = (scores_50[i] + scores_50[i+1]) / 2`
3. append that threshold to the output stats

So the validation threshold is automatically chosen from the validation predictions as the score threshold that maximizes F1 at LocSim `0.5`.

This threshold is then written by `BEVPoseValidator` to:

- `locsim_val_stats.json`

and similarly for bbox-locsim:

- `locsim_bbox_val_stats.json`

The important field is:

- `stats.score_threshold`

## 9. How Test Uses the Validation Threshold

This repo intentionally mirrors the mmpose flow:

1. run validation first
2. get optimal threshold from val
3. run test using that threshold

### 9.1 The forcing mechanism

The shell script always runs eval with:

- `--run-val-first`

So when `test_bev.py` is called for `--split test`, it first:

1. runs a val pass into a sibling `val/` directory
2. copies `*_val_stats.json` files into the test save directory

### 9.2 How test loads the threshold

Inside `BEVPoseValidator._run_locsim()`:

- for `phase in ("test", "challenge")`
- if a base val stats file exists
- it loads:
  - `th = json.load(f)["stats"]["score_threshold"]`
- then sets:
  - `coco_eval.params.score_threshold = th`

So the evaluator on test uses the score threshold found on validation.

### 9.3 How test predictions are also filtered with that threshold

This repo goes one step further for consistency.

In `BEVPoseValidator.init_metrics()`:

- it reads `locsim_val_stats.json`
- stores that threshold as `_analysis_score_threshold`

Then for baseline test/challenge runs with no BEV postprocess:

- `_final_prediction_threshold = _analysis_score_threshold`

During `update_metrics()`:

- predictions with `conf < _final_prediction_threshold` are removed before export

So for the normal baseline test flow:

1. val computes the threshold
2. test loads the threshold
3. test filters predictions using that threshold
4. LocSim evaluator also uses that threshold

This keeps:

- exported `predictions.json`
- test metrics
- submission metadata

aligned to the same validation-derived threshold.

## 10. Final Outputs

After the full shell script finishes, the main outputs are:

- training log
- `best.pt`
- `last.pt`
- LocSim result directories for both checkpoints
- summary JSON

The summary includes:

- checkpoint paths
- best eval metrics
- last eval metrics
- extracted `score_threshold`

Evaluation directories contain files such as:

- `predictions.json`
- `locsim_val_stats.json`
- `locsim_bbox_val_stats.json`
- `locsim_test_filtered_stats.json` or `locsim_test_stats.json`
- `metadata.json`
- submission zip

## 11. End-to-End Summary

If you want the shortest possible description of the pipeline:

1. Train a 2-keypoint YOLO pose model on image + YOLO labels:
   - keypoint 0 = pelvis
   - keypoint 1 = pelvis_ground
2. Continue fine-tuning stage-2 from an existing checkpoint.
3. At eval time, run val first.
4. Val writes `locsim_val_stats.json`, including the best `score_threshold`.
5. Test loads that `score_threshold`.
6. Test filters predictions with it and evaluates with it.
7. LocSim uses the predicted `pelvis_ground` image point plus per-image camera parameters from annotation JSON to project to BEV and compare against GT field position.

## 12. Most Important Clarification

If your question is specifically:

- "Is the model input image + camera params?"

The answer in this codebase is:

- model input: image only
- camera params: used later by BEV evaluation, not by model forward

That is the main architectural fact to keep in mind when reading this pipeline.
