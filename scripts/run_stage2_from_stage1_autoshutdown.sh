#!/usr/bin/env bash
set -Eeuo pipefail

# Stage2-only joint training, starting from user's stage1 sidepath best.pth
# (R1=75.3 from runs/csci_vjepa_sidepath_v2_residual_3gpu_20260523_020859).
# Backbone NOT frozen — full joint training. jepa_side / jepa_id_refine
# are warm-started from stage1 (already learned on image-mode bin-0 tokens).

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

ROOT=${ROOT:-/root/jepaintbpr/CSCI-VJEPA}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/MEVID}
CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/jepa_lazy_cache_vjepa2_1_g}
STAGE1_WEIGHT=${STAGE1_WEIGHT:-/root/autodl-tmp/runs/csci_vjepa_sidepath_v2_residual_3gpu_20260523_020859/stage1_image_sidepath/ez_eva02_vid_hybrid_extra_best.pth}
RUN_ROOT=$1
STAGE2_OUT=$RUN_ROOT/stage2_video_from_stage1
LOG_DIR=$RUN_ROOT/logs
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
GPUS=${GPUS:-0,1}
NUM_GPU=${NUM_GPU:-2}
MASTER_PORT=${MASTER_PORT:-12451}
SEED=${SEED:-1244}
COLOR=${COLOR:-17}
MAX_EPOCHS=${MAX_EPOCHS:-60}
BATCH_SIZE=${BATCH_SIZE:-8}
TEST_BATCH=${TEST_BATCH:-8}
SMOKE_ITERS=${SMOKE_ITERS:-0}   # >0 = smoke mode (early-exit), 0 = full run
SHUTDOWN_AT_END=${SHUTDOWN_AT_END:-1}

export TORCH_HOME=${TORCH_HOME:-/root/autodl-tmp/torch_cache}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

mkdir -p "$STAGE2_OUT" "$LOG_DIR"

{
  echo "==== stage2 from-stage1 driver ===="
  echo "run_root=$RUN_ROOT"
  echo "root=$ROOT"
  echo "stage1_weight=$STAGE1_WEIGHT"
  echo "data_root=$DATA_ROOT"
  echo "cache_dir=$CACHE_DIR"
  echo "gpus=$GPUS num_gpu=$NUM_GPU batch=$BATCH_SIZE test_batch=$TEST_BATCH"
  echo "max_epochs=$MAX_EPOCHS seed=$SEED color=$COLOR"
  echo "smoke_iters=$SMOKE_ITERS shutdown_at_end=$SHUTDOWN_AT_END"
  echo "start_time=$(date '+%F %T')"
  echo "head_commit=$(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "dirty_files=$(cd "$ROOT" && git status --short 2>/dev/null | tr '\n' ';')"
  sha1sum "$ROOT/model/ez_eva_custom.py" "$ROOT/train_two_step.py" "$ROOT/config/defaults.py" "$ROOT/model/jepa_side_path.py" 2>/dev/null || true
  nvidia-smi
  df -h /root/autodl-tmp || true
  du -sh "$CACHE_DIR" 2>/dev/null || true
} | tee -a "$LOG_DIR/driver.log"

if [[ ! -s "$STAGE1_WEIGHT" ]]; then
  echo "[$(date '+%F %T')] missing stage1 weight: $STAGE1_WEIGHT" | tee -a "$LOG_DIR/driver.log"
  exit 1
fi

cd "$ROOT"

# Smoke mode: cap MAX_EPOCHS to 1 and use a tiny EVAL_PERIOD so a smoke run
# only does dataset build + a few iters before the user kills it.
if [[ "$SMOKE_ITERS" -gt 0 ]]; then
  echo "[$(date '+%F %T')] SMOKE mode: limiting max_epochs=1 (kill manually after a few iters)" | tee -a "$LOG_DIR/driver.log"
  MAX_EPOCHS=1
fi

set +e
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -W ignore -m torch.distributed.launch \
  --nproc_per_node="$NUM_GPU" --master_port "$MASTER_PORT" \
  train_two_step.py \
  --env nccl --resume \
  --config_file configs/mevid_eva02_l_cloth_jepa.yml \
  DATA.ROOT "$DATA_ROOT" \
  MODEL.DIST_TRAIN True \
  TRAIN.TRAIN_VIDEO True \
  MODEL.MOTION_LOSS True \
  TRAIN.TEACH1 mevid \
  TEST.WEIGHT "$STAGE1_WEIGHT" \
  TRAIN.HYBRID True \
  TRAIN.DIR_TEACH1 "$DATA_ROOT" \
  TRAIN.TEACH1_MODEL None \
  TRAIN.TEACH1_LOAD_AS_IMG True \
  TRAIN.TEACH_DATASET_FIX color_adv \
  TRAIN.COLOR_ADV True \
  MODEL.NAME ez_eva02_vid_hybrid_extra \
  MODEL.JEPA_ENABLE True \
  MODEL.JEPA_SIDE_PATH True \
  MODEL.JEPA_ROOT /root/jepaintbpr/vjepa2 \
  MODEL.JEPA_CACHE_DIR "$CACHE_DIR" \
  MODEL.JEPA_WRITE_IMAGE_CACHE False \
  MODEL.JEPA_WRITE_TRAIN_CACHE False \
  MODEL.JEPA_RELEASE_ENCODER_AFTER_BATCH False \
  TRAIN.COLOR_PROFILE "$COLOR" \
  SOLVER.SEED "$SEED" \
  SOLVER.MAX_EPOCHS "$MAX_EPOCHS" \
  SOLVER.EVAL_PERIOD 2 \
  SOLVER.LOG_PERIOD 100 \
  DATA.BATCH_SIZE "$BATCH_SIZE" \
  DATA.NUM_INSTANCES 2 \
  DATA.TEST_BATCH "$TEST_BATCH" \
  OUTPUT_DIR "$STAGE2_OUT" \
  2>&1 | tee "$LOG_DIR/stage2_video_from_stage1.log"
stage2_rc=${PIPESTATUS[0]}
set -e

echo "[$(date '+%F %T')] stage2 exited with code $stage2_rc" | tee -a "$LOG_DIR/driver.log"
df -h /root/autodl-tmp | tee -a "$LOG_DIR/driver.log" || true
if [[ "$stage2_rc" -eq 0 && "$SHUTDOWN_AT_END" == "1" && "$SMOKE_ITERS" -eq 0 ]]; then
  echo "[$(date '+%F %T')] stage2 finished successfully; shutting down" | tee -a "$LOG_DIR/driver.log"
  sync
  shutdown -h now
else
  echo "[$(date '+%F %T')] stage2 rc=$stage2_rc smoke=$SMOKE_ITERS; shutdown skipped (SHUTDOWN_AT_END=$SHUTDOWN_AT_END)" | tee -a "$LOG_DIR/driver.log"
fi
exit "$stage2_rc"
