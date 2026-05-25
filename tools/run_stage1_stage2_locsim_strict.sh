#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/ultralytics"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

LOG_DIR="$ROOT/logs"
RESULT_DIR="$ROOT/results"
RUNS_DIR="$ROOT/runs/pelvis-proj"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$RUNS_DIR"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

# -------- User knobs --------
TRAIN_DEVICE="${TRAIN_DEVICE:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-1}"
WORKERS="${WORKERS:-8}"
LOCSIM_WORKERS="${LOCSIM_WORKERS:-8}"

BATCH="${BATCH:-48}"
IMGSZ="${IMGSZ:-960}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"

LOCSIM_BATCH="${LOCSIM_BATCH:-16}"
LOCSIM_SPLIT="${LOCSIM_SPLIT:-test}"

DATA_YAML="${DATA_YAML:-$ROOT/ultralytics/cfg/datasets/pelvis-proj-2pt.yaml}"
LOCSIM_DATA="${LOCSIM_DATA:-$ROOT/ultralytics/cfg/datasets/soccernet-synloc.yaml}"

# Stage1 = AIFI+P3CSDA
MODEL_STAGE1="${MODEL_STAGE1:-$ROOT/ultralytics/cfg/models/26/yolo26x-pose-aifi-p5-p3csda.yaml}"
# Stage2 = Stage1 structure + MLP refine (initialized from Stage1 best)
MODEL_STAGE2="${MODEL_STAGE2:-$ROOT/ultralytics/cfg/models/26/yolo26x-pose-aifi-p5-p3csda-mlp-refine.yaml}"

BASE_WEIGHTS="${BASE_WEIGHTS:-$ROOT/runs/pelvis-proj/stageA_best.pt}"
LOCSIM_SCRIPT="${LOCSIM_SCRIPT:-$ROOT/tools/test_bev_safe.py}"

STAGE2_LR0="${STAGE2_LR0:-0.0005}"
STAGE2_REFINE="${STAGE2_REFINE:-0.5}"
STAGE2_REFINE_PRIOR="${STAGE2_REFINE_PRIOR:-0.1}"

EXP_PREFIX="${EXP_PREFIX:-strict_two_stage_${TS}}"
RUN1="${RUN1:-${EXP_PREFIX}_stage1_e${STAGE1_EPOCHS}}"
RUN2="${RUN2:-${EXP_PREFIX}_stage2_e${STAGE2_EPOCHS}}"

LOG_STAGE1_TRAIN="$LOG_DIR/${RUN1}_train.log"
LOG_STAGE1_EVAL="$LOG_DIR/${RUN1}_locsim.log"
LOG_STAGE2_TRAIN="$LOG_DIR/${RUN2}_train.log"
LOG_STAGE2_EVAL="$LOG_DIR/${RUN2}_locsim.log"
SUMMARY_JSON="$RESULT_DIR/${EXP_PREFIX}_summary.json"
STAGE1_CKPT_FIXED="$RESULT_DIR/${EXP_PREFIX}_stage1_best.pt"
STAGE2_CKPT_FIXED="$RESULT_DIR/${EXP_PREFIX}_stage2_best.pt"

resolve_ckpt() {
  local run_name="$1"
  local p1="$RUNS_DIR/$run_name/weights/best.pt"
  local p2="$RUNS_DIR/$run_name/weights/last.pt"
  if [[ -f "$p1" ]]; then
    echo "$p1"
    return 0
  fi
  if [[ -f "$p2" ]]; then
    echo "$p2"
    return 0
  fi
  local found
  found="$(find "$ROOT" -type f -path "*/${run_name}/weights/best.pt" | head -n 1 || true)"
  if [[ -n "$found" ]]; then
    echo "$found"
    return 0
  fi
  found="$(find "$ROOT" -type f -path "*/${run_name}/weights/last.pt" | head -n 1 || true)"
  if [[ -n "$found" ]]; then
    echo "$found"
    return 0
  fi
  return 1
}

preflight_ckpt() {
  local ckpt="$1"
  python - "$ckpt" << 'PY'
import sys
from pathlib import Path
# Explicit registration to avoid torch.load class errors.
from ultralytics.nn.modules.transformer import P3CrossScaleDeformAttn  # noqa: F401
from ultralytics.nn.modules.head import Pose26MLPRefine, Pose26Refine  # noqa: F401
from ultralytics import YOLO

ckpt = Path(sys.argv[1])
if not ckpt.is_file():
    raise FileNotFoundError(ckpt)
_ = YOLO(str(ckpt))
print(f"[OK] ckpt load preflight: {ckpt}")
PY
}

must_have_locsim_stats() {
  local d="$1"
  for f in \
    locsim_val_stats.json \
    locsim_bbox_val_stats.json \
    locsim_test_filtered_stats.json \
    locsim_test_stats.json \
    locsim_bbox_test_filtered_stats.json \
    locsim_bbox_test_stats.json \
    locsim_test_postprocess_stats.json \
    locsim_bbox_test_postprocess_stats.json; do
    if [[ -f "$d/$f" ]]; then
      echo "$d/$f"
      return 0
    fi
  done
  echo "[ERROR] no locsim stats in $d" >&2
  return 1
}

run_locsim() {
  local tag="$1"
  local ckpt="$2"
  local out_dir="$RESULT_DIR/locsim_${EXP_PREFIX}_${tag}"
  local out_log="$LOG_DIR/${EXP_PREFIX}_${tag}_locsim.log"
  mkdir -p "$out_dir"

  python "$LOCSIM_SCRIPT" \
    --weights "$ckpt" \
    --data "$LOCSIM_DATA" \
    --split "$LOCSIM_SPLIT" \
    --imgsz "$IMGSZ" \
    --batch "$LOCSIM_BATCH" \
    --device "$EVAL_DEVICE" \
    --workers "$LOCSIM_WORKERS" \
    --run-val-first \
    --save-dir "$out_dir" > "$out_log" 2>&1

  must_have_locsim_stats "$out_dir" > /dev/null
  echo "$out_dir"
}

extract_metrics() {
  local d="$1"
  python - "$d" << 'PY'
import json, os, sys
base = sys.argv[1]
cands = [
    'locsim_val_stats.json',
    'locsim_bbox_val_stats.json',
    'locsim_test_filtered_stats.json',
    'locsim_test_stats.json',
    'locsim_bbox_test_filtered_stats.json',
    'locsim_bbox_test_stats.json',
    'locsim_test_postprocess_stats.json',
    'locsim_bbox_test_postprocess_stats.json',
]
p = None
for c in cands:
    x = os.path.join(base, c)
    if os.path.isfile(x):
        p = x
        break
out = {'stats_file': p}
if p:
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    out.update({
        'AP': d.get('AP'),
        'AP50': d.get('AP .5'),
        'AP75': d.get('AP .75'),
        'precision': d.get('precision'),
        'recall': d.get('recall'),
        'f1': d.get('f1'),
        'frame_accuracy': d.get('frame_accuracy'),
        'score_threshold': d.get('score_threshold'),
    })
print(json.dumps(out, ensure_ascii=False))
PY
}

[[ -f "$BASE_WEIGHTS" ]] || {
  echo "[ERROR] BASE_WEIGHTS not found: $BASE_WEIGHTS"
  exit 2
}
[[ -f "$MODEL_STAGE1" ]] || {
  echo "[ERROR] MODEL_STAGE1 not found: $MODEL_STAGE1"
  exit 2
}
[[ -f "$MODEL_STAGE2" ]] || {
  echo "[ERROR] MODEL_STAGE2 not found: $MODEL_STAGE2"
  exit 2
}
[[ -f "$DATA_YAML" ]] || {
  echo "[ERROR] DATA_YAML not found: $DATA_YAML"
  exit 2
}
[[ -f "$LOCSIM_SCRIPT" ]] || {
  echo "[ERROR] LOCSIM_SCRIPT not found: $LOCSIM_SCRIPT"
  exit 2
}
[[ -f "$LOCSIM_DATA" ]] || {
  echo "[ERROR] LOCSIM_DATA not found: $LOCSIM_DATA"
  exit 2
}

ANNOT_DIR="${ANNOT_DIR:-$ROOT/my_database/annotations}"
[[ -f "$ANNOT_DIR/val.json" ]] || {
  echo "[ERROR] Missing annotation: $ANNOT_DIR/val.json"
  exit 2
}
if [[ "$LOCSIM_SPLIT" == "test" || "$LOCSIM_SPLIT" == "challenge" ]]; then
  [[ -f "$ANNOT_DIR/test.json" ]] || {
    echo "[ERROR] Missing annotation: $ANNOT_DIR/test.json"
    exit 2
  }
fi

# ---------------- Stage1 train ----------------
echo "[INFO] Stage1 train: $RUN1"
python tools/train_pelvis_proj_4k.py \
  --model "$MODEL_STAGE1" \
  --init-weights "$BASE_WEIGHTS" \
  --data "$DATA_YAML" \
  --imgsz "$IMGSZ" \
  --epochs "$STAGE1_EPOCHS" \
  --batch "$BATCH" \
  --device "$TRAIN_DEVICE" \
  --workers "$WORKERS" \
  --project "$RUNS_DIR" \
  --name "$RUN1" \
  --optimizer MuSGD \
  --lr0 0.01 \
  --lrf 0.1 \
  --patience "$PATIENCE" \
  --pose 18.0 \
  --kobj 2.0 \
  --kpt1-weight 2.0 \
  --mosaic 0.2 \
  --translate 0.03 \
  --scale 0.2 \
  --close-mosaic 5 \
  --rect \
  --amp \
  --allow-oob-labels > "$LOG_STAGE1_TRAIN" 2>&1

CKPT1="$(resolve_ckpt "$RUN1")"
echo "[INFO] Stage1 ckpt raw: $CKPT1"
cp -f "$CKPT1" "$STAGE1_CKPT_FIXED"
echo "[INFO] Stage1 ckpt fixed: $STAGE1_CKPT_FIXED"
preflight_ckpt "$STAGE1_CKPT_FIXED"

# ---------------- Stage2 train ----------------
echo "[INFO] Stage2 train: $RUN2"
python tools/train_pelvis_proj_4k.py \
  --model "$MODEL_STAGE2" \
  --init-weights "$STAGE1_CKPT_FIXED" \
  --data "$DATA_YAML" \
  --imgsz "$IMGSZ" \
  --epochs "$STAGE2_EPOCHS" \
  --batch "$BATCH" \
  --device "$TRAIN_DEVICE" \
  --workers "$WORKERS" \
  --project "$RUNS_DIR" \
  --name "$RUN2" \
  --optimizer MuSGD \
  --lr0 "$STAGE2_LR0" \
  --lrf 0.1 \
  --patience "$PATIENCE" \
  --pose 18.0 \
  --kobj 2.0 \
  --kpt1-weight 2.0 \
  --refine "$STAGE2_REFINE" \
  --refine-prior "$STAGE2_REFINE_PRIOR" \
  --mosaic 0.2 \
  --translate 0.03 \
  --scale 0.2 \
  --close-mosaic 0 \
  --rect \
  --amp \
  --allow-oob-labels > "$LOG_STAGE2_TRAIN" 2>&1

CKPT2="$(resolve_ckpt "$RUN2")"
echo "[INFO] Stage2 ckpt raw: $CKPT2"
cp -f "$CKPT2" "$STAGE2_CKPT_FIXED"
echo "[INFO] Stage2 ckpt fixed: $STAGE2_CKPT_FIXED"
preflight_ckpt "$STAGE2_CKPT_FIXED"

# ---------------- LocSim (after both stages) ----------------
LOCSIM1_DIR="$(run_locsim stage1 "$STAGE1_CKPT_FIXED")"
METRIC1="$(extract_metrics "$LOCSIM1_DIR")"

LOCSIM2_DIR="$(run_locsim stage2 "$STAGE2_CKPT_FIXED")"
METRIC2="$(extract_metrics "$LOCSIM2_DIR")"

python - "$SUMMARY_JSON" "$RUN1" "$STAGE1_CKPT_FIXED" "$LOG_STAGE1_TRAIN" "$LOCSIM1_DIR" "$METRIC1" "$RUN2" "$STAGE2_CKPT_FIXED" "$LOG_STAGE2_TRAIN" "$LOCSIM2_DIR" "$METRIC2" << 'PY'
import json, sys
(
    out_json,
    run1, ckpt1, log1, loc1, m1,
    run2, ckpt2, log2, loc2, m2,
) = sys.argv[1:]
summary = {
    'stage1': {
        'run': run1,
        'ckpt': ckpt1,
        'train_log': log1,
        'locsim_dir': loc1,
        'locsim': json.loads(m1),
    },
    'stage2': {
        'run': run2,
        'ckpt': ckpt2,
        'train_log': log2,
        'locsim_dir': loc2,
        'locsim': json.loads(m2),
    },
}
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[DONE] $SUMMARY_JSON"
