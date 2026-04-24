# Pelvis Projection Two-Stage Workflow

This repository is currently trimmed to a focused custom workflow for the pelvis-projection task:

- two-stage training and LocSim evaluation: `tools/run_stage1_stage2_locsim_strict.sh`
- qualitative visualization and baseline comparison: `tools/annotate_first10_pelvis_compare.py`

Custom helper scripts outside these two paths have been removed from `tools/`.

## Kept Custom Entry Points

- `tools/run_stage1_stage2_locsim_strict.sh`
  Main two-stage pipeline. It trains Stage 1, initializes Stage 2 from the Stage 1 best checkpoint, and then runs LocSim evaluation.
- `tools/train_pelvis_proj_4k.py`
  Training entrypoint used by the strict pipeline.
- `tools/test_bev_safe.py`
  Safe LocSim wrapper used by the strict pipeline. It forwards execution to the repository-level `test_bev.py`.
- `tools/annotate_first10_pelvis_compare.py`
  Generates GT / baseline / improved-model visualization sets.
- `test_bev.py`
  Actual LocSim evaluation entrypoint.

## Default Configs

The workflow uses these configs by default:

- dataset: `ultralytics/cfg/datasets/pelvis-proj-2pt.yaml`
- LocSim dataset: `ultralytics/cfg/datasets/soccernet-synloc.yaml`
- Stage 1 model: `ultralytics/cfg/models/26/yolo26x-pose-aifi-p5-p3csda.yaml`
- Stage 2 model: `ultralytics/cfg/models/26/yolo26x-pose-aifi-p5-p3csda-mlp-refine.yaml`

The default training data root from `pelvis-proj-2pt.yaml` is:

```yaml
path: /root/autodl-tmp/ultralytics/my_database
```

LocSim evaluation additionally expects:

- `my_database/annotations/val.json`
- `my_database/annotations/test.json` when `LOCSIM_SPLIT=test`

## Run Two-Stage Training and LocSim

From the repository root:

```bash
bash tools/run_stage1_stage2_locsim_strict.sh
```

Common environment overrides:

```bash
TRAIN_DEVICE=1 \
  EVAL_DEVICE=1 \
  BATCH=48 \
  IMGSZ=960 \
  STAGE1_EPOCHS=20 \
  STAGE2_EPOCHS=20 \
  LOCSIM_SPLIT=test \
  bash tools/run_stage1_stage2_locsim_strict.sh
```

Main outputs:

- training logs: `logs/`
- exported best checkpoints: `results/<exp>_stage1_best.pt`, `results/<exp>_stage2_best.pt`
- LocSim outputs: `results/locsim_<exp>_stage1/`, `results/locsim_<exp>_stage2/`
- summary: `results/<exp>_summary.json`

## Run Visualization

The visualization script generates three outputs per image:

- `gt/`: GT player boxes and GT pelvis / pelvis-ground points
- `baseline/`: GT pelvis / pelvis-ground plus baseline predicted boxes and points
- `two_stage/`: GT pelvis / pelvis-ground plus improved-model predicted boxes and points

Example:

```bash
python tools/annotate_first10_pelvis_compare.py \
  --stage1-ckpt /root/autodl-tmp/ultralytics/results/two_stage_gpu1_fixed_stage1_best.pt \
  --baseline-ckpt /root/autodl-tmp/ultralytics/runs/pelvis-proj/yolo26x_mlp_fullft_e3_b64_gpu1_20260409_235143/weights/best.pt \
  --output-dir /root/autodl-tmp/ultralytics/results/first100_compare_stage1_vs_baseline \
  --device 1 \
  --imgsz 960 \
  --conf 0.25 \
  --iou 0.7 \
  --num-images 100
```

The output directory contains:

- `gt/*.jpg`
- `baseline/*.jpg`
- `two_stage/*.jpg`
- `summary.json`

`summary.json` records, for each image:

- GT count
- baseline and improved-model prediction counts
- matched GT count
- missed GT count
- an automatically selected `best_improvement` example

## Notes

- `results/` is ignored by Git.
- In `tools/annotate_first10_pelvis_compare.py`, the `--stage1-ckpt` argument is simply the improved-model checkpoint used for comparison. You can replace it with any other improved checkpoint if needed.
