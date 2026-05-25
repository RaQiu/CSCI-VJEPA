#!/usr/bin/env bash
set -Eeuo pipefail

# Stage2-only with CSCI-V backbone FROZEN.
# Loads original CSCI-V 79.7 video weight, trains only JEPA side path
# (+ heads/bottleneck/classifier) on top via zero-init residual.
# Step 0 == baseline 79.7 (because jepa_id_refine final layer is zero-init).

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

ROOT=${ROOT:-/root/jepaintbpr/CSCI-VJEPA}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/MEVID}
CACHE_DIR=${CACHE_DIR:-$DATA_ROOT/jepa_lazy_cache_vjepa2_1_g}
PAPER_WEIGHT=${PAPER_WEIGHT:-/root/autodl-tmp/models/csci/mevid-17-1244/ez_eva02_vid_hybrid_extra_best.pth}
RUN_ROOT=$1
STAGE2_OUT=$RUN_ROOT/stage2_video_freeze
LOG_DIR=$RUN_ROOT/logs
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
GPUS=${GPUS:-0,1}
NUM_GPU=${NUM_GPU:-2}
MASTER_PORT=${MASTER_PORT:-12450}
SEED=${SEED:-1244}
COLOR=${COLOR:-17}
MAX_EPOCHS=${MAX_EPOCHS:-60}
BATCH_SIZE=${BATCH_SIZE:-4}
TEST_BATCH=${TEST_BATCH:-4}
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
  echo "==== freeze stage2 driver ===="
  echo "run_root=$RUN_ROOT"
  echo "root=$ROOT"
  echo "paper_weight=$PAPER_WEIGHT"
  echo "data_root=$DATA_ROOT"
  echo "cache_dir=$CACHE_DIR"
  echo "gpus=$GPUS num_gpu=$NUM_GPU batch=$BATCH_SIZE test_batch=$TEST_BATCH"
  echo "max_epochs=$MAX_EPOCHS seed=$SEED color=$COLOR"
  echo "shutdown_at_end=$SHUTDOWN_AT_END"
  echo "start_time=$(date '+%F %T')"
  echo "head_commit=$(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "dirty_files=$(cd "$ROOT" && git status --short 2>/dev/null | tr '\n' ';')"
  sha1sum "$ROOT/model/ez_eva_custom.py" "$ROOT/train_two_step.py" "$ROOT/config/defaults.py" "$ROOT/model/jepa_side_path.py" 2>/dev/null || true
  nvidia-smi
  df -h /root/autodl-tmp || true
  du -sh "$CACHE_DIR" 2>/dev/null || true
} | tee -a "$LOG_DIR/driver.log"

if [[ ! -s "$PAPER_WEIGHT" ]]; then
  echo "[$(date '+%F %T')] missing paper weight: $PAPER_WEIGHT" | tee -a "$LOG_DIR/driver.log"
  exit 1
fi

cd "$ROOT"

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
  TEST.WEIGHT "$PAPER_WEIGHT" \
  TRAIN.HYBRID True \
  TRAIN.DIR_TEACH1 "$DATA_ROOT" \
  TRAIN.TEACH1_MODEL None \
  TRAIN.TEACH1_LOAD_AS_IMG True \
  TRAIN.TEACH_DATASET_FIX color_adv \
  TRAIN.COLOR_ADV True \
  MODEL.NAME ez_eva02_vid_hybrid_extra \
  MODEL.JEPA_ENABLE True \
  MODEL.JEPA_SIDE_PATH True \
  MODEL.JEPA_FREEZE_BACKBONE True \
  MODEL.JEPA_FREEZE_KEEP_HEAD True \
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
  2>&1 | tee "$LOG_DIR/stage2_video_freeze.log"
stage2_rc=${PIPESTATUS[0]}
set -e

echo "[$(date '+%F %T')] stage2 exited with code $stage2_rc" | tee -a "$LOG_DIR/driver.log"
df -h /root/autodl-tmp | tee -a "$LOG_DIR/driver.log" || true
if [[ "$stage2_rc" -eq 0 && "$SHUTDOWN_AT_END" == "1" ]]; then
  echo "[$(date '+%F %T')] freeze stage2 finished successfully; shutting down" | tee -a "$LOG_DIR/driver.log"
  sync
  shutdown -h now
else
  echo "[$(date '+%F %T')] stage2 rc=$stage2_rc; shutdown skipped (SHUTDOWN_AT_END=$SHUTDOWN_AT_END)" | tee -a "$LOG_DIR/driver.log"
fi
exit "$stage2_rc"
