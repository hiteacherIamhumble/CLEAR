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
TRAIN_DEVICE="${TRAIN_DEVICE:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-0}"
WORKERS="${WORKERS:-8}"
LOCSIM_WORKERS="${LOCSIM_WORKERS:-8}"

BATCH="${BATCH:-48}"
IMGSZ="${IMGSZ:-960}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-12}"
PATIENCE="${PATIENCE:-8}"

LOCSIM_BATCH="${LOCSIM_BATCH:-16}"
SELECT_SPLIT="${SELECT_SPLIT:-val}"
FINAL_SPLIT="${FINAL_SPLIT:-test}"

DATA_YAML="${DATA_YAML:-$ROOT/ultralytics/cfg/datasets/pelvis-proj-2pt.yaml}"
LOCSIM_DATA="${LOCSIM_DATA:-$ROOT/ultralytics/cfg/datasets/soccernet-synloc.yaml}"
MODEL_STAGE2="${MODEL_STAGE2:-$ROOT/ultralytics/cfg/models/26/yolo26x-pose-aifi-p5-p3csda-mlp-refine.yaml}"
INIT_WEIGHTS="${INIT_WEIGHTS:-$ROOT/results/two_stage_gpu1_fixed_stage1_best.pt}"
LOCSIM_SCRIPT="${LOCSIM_SCRIPT:-$ROOT/tools/test_bev_safe.py}"

# Safer stage-2 defaults.
STAGE2_LR0="${STAGE2_LR0:-0.0005}"
STAGE2_REFINE="${STAGE2_REFINE:-0.1}"
STAGE2_REFINE_PRIOR="${STAGE2_REFINE_PRIOR:-0.1}"
MOSAIC="${MOSAIC:-0.0}"
TRANSLATE="${TRANSLATE:-0.01}"
SCALE="${SCALE:-0.05}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-5}"
SAVE_PERIOD="${SAVE_PERIOD:-1}"

EXP_PREFIX="${EXP_PREFIX:-stage2_locsim_safe_${TS}}"
RUN2="${RUN2:-${EXP_PREFIX}_e${STAGE2_EPOCHS}}"

LOG_STAGE2_TRAIN="$LOG_DIR/${RUN2}_train.log"
SUMMARY_JSON="$RESULT_DIR/${EXP_PREFIX}_summary.json"
BEST_CKPT_FIXED="$RESULT_DIR/${EXP_PREFIX}_best_locsim.pt"

preflight_ckpt() {
  local ckpt="$1"
  python - <<PY "$ckpt"
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
  python - <<PY "$stats_json"
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
stats = d.get("stats", d)
out = {
    "AP": stats.get("AP"),
    "AP50": stats.get("AP .5"),
    "AP75": stats.get("AP .75"),
    "precision": stats.get("precision"),
    "recall": stats.get("recall"),
    "f1": stats.get("f1"),
    "frame_accuracy": stats.get("frame_accuracy"),
    "score_threshold": stats.get("score_threshold"),
}
print(json.dumps(out, ensure_ascii=False))
PY
}

select_metric_value() {
  local stats_json="$1"
  python - <<PY "$stats_json"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
stats = d.get("stats", d)
print(float(stats["AP"]))
PY
}

[[ -f "$INIT_WEIGHTS" ]] || { echo "[ERROR] INIT_WEIGHTS not found: $INIT_WEIGHTS"; exit 2; }
[[ -f "$MODEL_STAGE2" ]] || { echo "[ERROR] MODEL_STAGE2 not found: $MODEL_STAGE2"; exit 2; }
[[ -f "$DATA_YAML" ]] || { echo "[ERROR] DATA_YAML not found: $DATA_YAML"; exit 2; }
[[ -f "$LOCSIM_SCRIPT" ]] || { echo "[ERROR] LOCSIM_SCRIPT not found: $LOCSIM_SCRIPT"; exit 2; }
[[ -f "$LOCSIM_DATA" ]] || { echo "[ERROR] LOCSIM_DATA not found: $LOCSIM_DATA"; exit 2; }

validate_device() {
  local device="$1"
  python - <<PY "$device"
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

validate_device "$TRAIN_DEVICE"
validate_device "$EVAL_DEVICE"

ANNOT_DIR="${ANNOT_DIR:-$ROOT/my_database/annotations}"
[[ -f "$ANNOT_DIR/val.json" ]] || { echo "[ERROR] Missing annotation: $ANNOT_DIR/val.json"; exit 2; }
if [[ "$FINAL_SPLIT" == "test" || "$FINAL_SPLIT" == "challenge" ]]; then
  [[ -f "$ANNOT_DIR/test.json" ]] || { echo "[ERROR] Missing annotation: $ANNOT_DIR/test.json"; exit 2; }
fi

echo "[INFO] Stage2-only train: $RUN2"
python tools/train_pelvis_proj_4k.py \
  --model "$MODEL_STAGE2" \
  --init-weights "$INIT_WEIGHTS" \
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
  --mosaic "$MOSAIC" \
  --translate "$TRANSLATE" \
  --scale "$SCALE" \
  --close-mosaic "$CLOSE_MOSAIC" \
  --rect \
  --amp \
  --allow-oob-labels \
  --save-period "$SAVE_PERIOD" > "$LOG_STAGE2_TRAIN" 2>&1

RUN_DIR="$RUNS_DIR/$RUN2"
WEIGHTS_DIR="$RUN_DIR/weights"
[[ -d "$WEIGHTS_DIR" ]] || { echo "[ERROR] Missing weights dir: $WEIGHTS_DIR"; exit 3; }

BEST_AP="-1"
BEST_EPOCH=""
BEST_EPOCH_CKPT=""
BEST_STATS_JSON=""
BEST_SELECT_DIR=""

mapfile -t EPOCH_CKPTS < <(find "$WEIGHTS_DIR" -maxdepth 1 -type f -name 'epoch*.pt' | sort -V)
if [[ ${#EPOCH_CKPTS[@]} -eq 0 ]]; then
  echo "[ERROR] No per-epoch checkpoints found in $WEIGHTS_DIR. Check save_period." >&2
  exit 4
fi

for ckpt in "${EPOCH_CKPTS[@]}"; do
  epoch_name="$(basename "$ckpt" .pt)"
  epoch_num="${epoch_name#epoch}"
  sel_dir="$RESULT_DIR/locsim_${EXP_PREFIX}_${epoch_name}_${SELECT_SPLIT}"
  sel_log="$LOG_DIR/${EXP_PREFIX}_${epoch_name}_${SELECT_SPLIT}.log"
  mkdir -p "$sel_dir"

  echo "[INFO] LocSim select ${epoch_name} on ${SELECT_SPLIT}"
  python "$LOCSIM_SCRIPT" \
    --weights "$ckpt" \
    --data "$LOCSIM_DATA" \
    --split "$SELECT_SPLIT" \
    --imgsz "$IMGSZ" \
    --batch "$LOCSIM_BATCH" \
    --device "$EVAL_DEVICE" \
    --workers "$LOCSIM_WORKERS" \
    --save-dir "$sel_dir" > "$sel_log" 2>&1

  stats_json="$sel_dir/locsim_${SELECT_SPLIT}_stats.json"
  [[ -f "$stats_json" ]] || { echo "[ERROR] Missing stats: $stats_json" >&2; exit 5; }
  ap="$(select_metric_value "$stats_json")"
  echo "[INFO] ${epoch_name} ${SELECT_SPLIT} LocSim AP=${ap}"

  if python - <<PY "$ap" "$BEST_AP"
import sys
cur = float(sys.argv[1])
best = float(sys.argv[2])
raise SystemExit(0 if cur > best else 1)
PY
  then
    BEST_AP="$ap"
    BEST_EPOCH="$epoch_num"
    BEST_EPOCH_CKPT="$ckpt"
    BEST_STATS_JSON="$stats_json"
    BEST_SELECT_DIR="$sel_dir"
  fi
done

[[ -n "$BEST_EPOCH_CKPT" ]] || { echo "[ERROR] Failed to select best epoch checkpoint"; exit 6; }
cp -f "$BEST_EPOCH_CKPT" "$BEST_CKPT_FIXED"
echo "[INFO] Best LocSim epoch: $BEST_EPOCH"
echo "[INFO] Best ${SELECT_SPLIT} LocSim AP: $BEST_AP"
echo "[INFO] Best ckpt fixed: $BEST_CKPT_FIXED"
preflight_ckpt "$BEST_CKPT_FIXED"

FINAL_DIR="$RESULT_DIR/locsim_${EXP_PREFIX}_best_${FINAL_SPLIT}"
FINAL_LOG="$LOG_DIR/${EXP_PREFIX}_best_${FINAL_SPLIT}.log"
mkdir -p "$FINAL_DIR"

echo "[INFO] Final LocSim eval on ${FINAL_SPLIT} using best LocSim-selected checkpoint"
python "$LOCSIM_SCRIPT" \
  --weights "$BEST_CKPT_FIXED" \
  --data "$LOCSIM_DATA" \
  --split "$FINAL_SPLIT" \
  --imgsz "$IMGSZ" \
  --batch "$LOCSIM_BATCH" \
  --device "$EVAL_DEVICE" \
  --workers "$LOCSIM_WORKERS" \
  --run-val-first \
  --save-dir "$FINAL_DIR" > "$FINAL_LOG" 2>&1

FINAL_STATS_JSON="$FINAL_DIR/locsim_${FINAL_SPLIT}_stats.json"
if [[ "$FINAL_SPLIT" == "test" && ! -f "$FINAL_STATS_JSON" ]]; then
  FINAL_STATS_JSON="$FINAL_DIR/locsim_test_filtered_stats.json"
fi
if [[ "$FINAL_SPLIT" == "challenge" && ! -f "$FINAL_STATS_JSON" ]]; then
  FINAL_STATS_JSON=""
fi

BEST_SELECT_METRICS="$(extract_locsim_stats "$BEST_STATS_JSON")"
if [[ -n "$FINAL_STATS_JSON" && -f "$FINAL_STATS_JSON" ]]; then
  FINAL_METRICS="$(extract_locsim_stats "$FINAL_STATS_JSON")"
else
  FINAL_METRICS="{}"
fi

python - <<PY "$SUMMARY_JSON" "$RUN2" "$INIT_WEIGHTS" "$BEST_CKPT_FIXED" "$BEST_EPOCH" "$BEST_AP" "$LOG_STAGE2_TRAIN" "$BEST_SELECT_DIR" "$BEST_STATS_JSON" "$BEST_SELECT_METRICS" "$FINAL_DIR" "$FINAL_LOG" "$FINAL_METRICS"
import json, sys
(
    out_json,
    run2, init_weights, best_ckpt, best_epoch, best_ap, train_log,
    select_dir, select_stats_json, select_metrics,
    final_dir, final_log, final_metrics,
) = sys.argv[1:]
summary = {
    "stage2_only": {
        "run": run2,
        "init_weights": init_weights,
        "train_log": train_log,
        "best_locsim_epoch": int(best_epoch),
        "best_locsim_ap": float(best_ap),
        "best_ckpt": best_ckpt,
        "selection_dir": select_dir,
        "selection_stats_file": select_stats_json,
        "selection_metrics": json.loads(select_metrics),
        "final_eval_dir": final_dir,
        "final_eval_log": final_log,
        "final_metrics": json.loads(final_metrics),
    }
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[DONE] $SUMMARY_JSON"
