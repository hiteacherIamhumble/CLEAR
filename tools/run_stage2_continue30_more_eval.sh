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

TRAIN_DEVICE="${TRAIN_DEVICE:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-0}"
WORKERS="${WORKERS:-8}"
LOCSIM_WORKERS="${LOCSIM_WORKERS:-8}"

BATCH="${BATCH:-48}"
IMGSZ="${IMGSZ:-960}"
MORE_EPOCHS="${MORE_EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
LOCSIM_BATCH="${LOCSIM_BATCH:-16}"
FINAL_SPLIT="${FINAL_SPLIT:-test}"

DATA_YAML="${DATA_YAML:-$ROOT/ultralytics/cfg/datasets/pelvis-proj-2pt.yaml}"
LOCSIM_DATA="${LOCSIM_DATA:-$ROOT/ultralytics/cfg/datasets/soccernet-synloc.yaml}"
MODEL_STAGE2="${MODEL_STAGE2:-$ROOT/ultralytics/cfg/models/26/yolo26x-pose-aifi-p5-p3csda-mlp-refine.yaml}"
INIT_WEIGHTS="${INIT_WEIGHTS:-$ROOT/runs/pelvis-proj/stage2_dual_30e_20260422_011423_continue_to30/weights/last.pt}"
LOCSIM_SCRIPT="${LOCSIM_SCRIPT:-$ROOT/tools/test_bev_safe.py}"

STAGE2_LR0="${STAGE2_LR0:-0.0005}"
STAGE2_REFINE="${STAGE2_REFINE:-0.1}"
STAGE2_REFINE_PRIOR="${STAGE2_REFINE_PRIOR:-0.1}"
MOSAIC="${MOSAIC:-0.0}"
TRANSLATE="${TRANSLATE:-0.01}"
SCALE="${SCALE:-0.05}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-5}"
SAVE_PERIOD="${SAVE_PERIOD:-1}"
REFINE_TOPK_TRAIN="${REFINE_TOPK_TRAIN:-256}"
REFINE_TOPK_EVAL="${REFINE_TOPK_EVAL:-100}"
REFINE_HIDDEN="${REFINE_HIDDEN:-128}"

EXP_PREFIX="${EXP_PREFIX:-stage2_continue30_more_${TS}}"
RUN_NAME="${RUN_NAME:-${EXP_PREFIX}_e${MORE_EPOCHS}}"
TRAIN_LOG="$LOG_DIR/${RUN_NAME}_train.log"
SUMMARY_JSON="$RESULT_DIR/${EXP_PREFIX}_summary.json"

validate_device() {
  local device="$1"
  python - "$device" << PY
import sys
import torch

device = sys.argv[1].strip()
if device in {"", "cpu"}:
    raise SystemExit(0)
if "," in device:
    for part in device.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise SystemExit(f"[ERROR] Non-numeric CUDA device entry: {part!r}")
        if int(part) >= torch.cuda.device_count():
            raise SystemExit(f"[ERROR] Requested CUDA device {part} but only {torch.cuda.device_count()} visible device(s) exist")
    raise SystemExit(0)
if not device.isdigit():
    raise SystemExit(f"[ERROR] Unsupported device string: {device!r}")
idx = int(device)
count = torch.cuda.device_count()
if idx >= count:
    raise SystemExit(f"[ERROR] Requested CUDA device {idx} but only {count} visible device(s) exist")
PY
}

preflight_ckpt() {
  local ckpt="$1"
  python - "$ckpt" << PY
import sys
from pathlib import Path
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

extract_locsim_stats() {
  local stats_json="$1"
  python - "$stats_json" << PY
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    d = json.load(f)
stats = d.get('stats', d)
out = {
    'AP': stats.get('AP'),
    'AP50': stats.get('AP .5'),
    'AP75': stats.get('AP .75'),
    'precision': stats.get('precision'),
    'recall': stats.get('recall'),
    'f1': stats.get('f1'),
    'frame_accuracy': stats.get('frame_accuracy'),
    'score_threshold': stats.get('score_threshold'),
}
print(json.dumps(out, ensure_ascii=False))
PY
}

run_eval() {
  local tag="$1"
  local ckpt="$2"
  local out_dir="$RESULT_DIR/locsim_${EXP_PREFIX}_${tag}_${FINAL_SPLIT}"
  local out_log="$LOG_DIR/${EXP_PREFIX}_${tag}_${FINAL_SPLIT}.log"
  mkdir -p "$out_dir"
  POSE26_MLP_REFINE_TOPK_TRAIN="$REFINE_TOPK_TRAIN" \
    POSE26_MLP_REFINE_TOPK_EVAL="$REFINE_TOPK_EVAL" \
    POSE26_MLP_REFINE_HIDDEN="$REFINE_HIDDEN" \
    PYTHONPATH="$ROOT:${PYTHONPATH:-}" python "$LOCSIM_SCRIPT" \
    --weights "$ckpt" \
    --data "$LOCSIM_DATA" \
    --split "$FINAL_SPLIT" \
    --imgsz "$IMGSZ" \
    --batch "$LOCSIM_BATCH" \
    --device "$EVAL_DEVICE" \
    --workers "$LOCSIM_WORKERS" \
    --run-val-first \
    --save-dir "$out_dir" > "$out_log" 2>&1

  local stats_json="$out_dir/locsim_${FINAL_SPLIT}_stats.json"
  if [[ "$FINAL_SPLIT" == "test" && ! -f "$stats_json" ]]; then
    stats_json="$out_dir/locsim_test_filtered_stats.json"
  fi
  if [[ -n "$stats_json" && -f "$stats_json" ]]; then
    local metrics
    metrics="$(extract_locsim_stats "$stats_json")"
    echo "$stats_json|$metrics|$out_dir|$out_log"
  else
    echo "|{}|$out_dir|$out_log"
  fi
}

validate_device "$TRAIN_DEVICE"
validate_device "$EVAL_DEVICE"

[[ -f "$INIT_WEIGHTS" ]] || {
  echo "[ERROR] INIT_WEIGHTS not found: $INIT_WEIGHTS"
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
if [[ "$FINAL_SPLIT" == "test" || "$FINAL_SPLIT" == "challenge" ]]; then
  [[ -f "$ANNOT_DIR/test.json" ]] || {
    echo "[ERROR] Missing annotation: $ANNOT_DIR/test.json"
    exit 2
  }
fi

preflight_ckpt "$INIT_WEIGHTS"
echo "[INFO] Continue fine-tune from: $INIT_WEIGHTS"
echo "[INFO] New run: $RUN_NAME for $MORE_EPOCHS more epochs"

POSE26_MLP_REFINE_TOPK_TRAIN="$REFINE_TOPK_TRAIN" \
  POSE26_MLP_REFINE_TOPK_EVAL="$REFINE_TOPK_EVAL" \
  POSE26_MLP_REFINE_HIDDEN="$REFINE_HIDDEN" \
  python tools/train_pelvis_proj_4k.py \
  --model "$MODEL_STAGE2" \
  --init-weights "$INIT_WEIGHTS" \
  --data "$DATA_YAML" \
  --imgsz "$IMGSZ" \
  --epochs "$MORE_EPOCHS" \
  --batch "$BATCH" \
  --device "$TRAIN_DEVICE" \
  --workers "$WORKERS" \
  --project "$RUNS_DIR" \
  --name "$RUN_NAME" \
  --optimizer MuSGD \
  --lr0 "$STAGE2_LR0" \
  --lrf 0.1 \
  --patience "$PATIENCE" \
  --pose 18.0 \
  --kobj 2.0 \
  --kpt1-weight 2.0 \
  --refine "$STAGE2_REFINE" \
  --refine-prior "$STAGE2_REFINE_PRIOR" \
  --mosaic "$MOSAIC" \
  --translate "$TRANSLATE" \
  --scale "$SCALE" \
  --close-mosaic "$CLOSE_MOSAIC" \
  --rect \
  --amp \
  --allow-oob-labels \
  --save-period "$SAVE_PERIOD" > "$TRAIN_LOG" 2>&1

WEIGHTS_DIR="$RUNS_DIR/$RUN_NAME/weights"
if [[ ! -d "$WEIGHTS_DIR" ]]; then
  FOUND_DIR="$(find "$RUNS_DIR" -maxdepth 2 -type f -path "*/${RUN_NAME}*/weights/last.pt" -printf '%h\n' | sort -V | tail -n 1)"
  if [[ -n "$FOUND_DIR" ]]; then
    WEIGHTS_DIR="$FOUND_DIR"
  fi
fi
BEST_CKPT="$WEIGHTS_DIR/best.pt"
LAST_CKPT="$WEIGHTS_DIR/last.pt"
[[ -f "$BEST_CKPT" ]] || {
  echo "[ERROR] Missing best checkpoint: $BEST_CKPT"
  exit 3
}
[[ -f "$LAST_CKPT" ]] || {
  echo "[ERROR] Missing last checkpoint: $LAST_CKPT"
  exit 3
}
preflight_ckpt "$BEST_CKPT"
preflight_ckpt "$LAST_CKPT"

BEST_INFO="$(run_eval best "$BEST_CKPT")"
LAST_INFO="$(run_eval last "$LAST_CKPT")"

python - "$SUMMARY_JSON" "$RUN_NAME" "$INIT_WEIGHTS" "$TRAIN_LOG" "$BEST_CKPT" "$LAST_CKPT" "$BEST_INFO" "$LAST_INFO" << PY
import json, sys
summary_json, run_name, init_weights, train_log, best_ckpt, last_ckpt, best_info, last_info = sys.argv[1:]

def decode(part):
    stats_file, metrics, eval_dir, eval_log = part.split('|', 3)
    return {
        'stats_file': stats_file or None,
        'metrics': json.loads(metrics),
        'eval_dir': eval_dir,
        'eval_log': eval_log,
    }
summary = {
    'continue_more': {
        'run': run_name,
        'init_weights': init_weights,
        'train_log': train_log,
        'best_ckpt': best_ckpt,
        'last_ckpt': last_ckpt,
        'best_eval': decode(best_info),
        'last_eval': decode(last_info),
    }
}
with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[DONE] $SUMMARY_JSON"
