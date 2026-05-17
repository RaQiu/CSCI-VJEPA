

##### NoAd
# vid-ez 4 frames 
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True >> outputs/"$DATASET"_4T_NoAd_e2e.txt

# vid-ez 8 frames 
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True DATA.F8 True MODEL.TIM_DIM 8 >> outputs/"$DATASET"_8T_NoAd_e2e.txt

# vid-ez E2E (w/ pretrained)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --resume --config_file $CONFIG DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt  >> outputs/"$DATASET"_4T_NoAd_e2e_pre.txt

# vid-ez E2E (w/ pretrained) + 8 frames
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --resume --config_file $CONFIG DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt DATA.F8 True MODEL.TIM_DIM 8 >> outputs/"$DATASET"_8T_NoAd_e2e_pre.txt

#### NoAd + Motion LOSS
vid-ez E2E (w/ pretrained)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --resume --config_file $CONFIG DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt MODEL.MOTION_LOSS True >> outputs/"$DATASET"_4T_NoAd_e2e_pre_ml.txt

# vid-ez 4 frames 
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True MODEL.MOTION_LOSS True >> outputs/"$DATASET"_4T_NoAd_e2e_ml.txt

# vid-ez 8 frames 
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True DATA.F8 True MODEL.TIM_DIM 8 MODEL.MOTION_LOSS True >> outputs/"$DATASET"_8T_NoAd_e2e_ml.txt

# vid-ez E2E (w/ pretrained) + 8 frames
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --resume --config_file $CONFIG DATA.ROOT $ROOT \
    MODEL.NAME 'ez_eva02_vid' TRAIN.TRAIN_VIDEO True TEST.WEIGHT $wt DATA.F8 True MODEL.TIM_DIM 8 MODEL.MOTION_LOSS True >> outputs/"$DATASET"_8T_NoAd_e2e_pre.txt


