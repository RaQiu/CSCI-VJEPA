conda activate bert
cd ~/ICCV-CSCI-Person-ReID/
NUM_GPU=1
PORT=12355
RUN_NO=1
ENV='nccl'

#################### LTCC ####################
ltcc=/data/priyank/synthetic/LTCC/
CONFIG=configs/ltcc_eva02_l_cloth.yml
DATASET="ltcc"
ROOT=$ltcc
COLOR=26


###############################################################################################    
#################### PRCC ####################
prcc=/data/priyank/synthetic/PRCC/
CONFIG=configs/prcc_eva02_l_cloth.yml
DATASET="prcc"
ROOT=$prcc
COLOR=9


################################ # PROPOSED (IMAGE COLORS) ###################################################
# VANILL TRAIN (no color no cloth)
SEED=1244
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT \
    OUTPUT_DIR $DATASET"_ONLY_IMG" SOLVER.SEED $SEED >> ucf_output/"$DATASET"_img_nocloth-RUN-$SEED.txt    


# #### COLOR
SEED=1234
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    OUTPUT_DIR $DATASET+"_Co-$COLOR" >> outputs/"$DATASET"-CO-$COLOR-UCF2-RUN-$SEED-FINAL.txt






###############################################################################################
################################ # Training ABLATION ###################################################
##### TRADITIONAL SELF_ATTENTION 
##### Img + COLOR (Extra Token) [Traditional Unified Self-Attention]
SEED=1245
COLOR=5
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    MODEL.UNIFIED_DIST True >> outputs/"$DATASET"-CO-$COLOR-TRAD-SELF-ATTN-$SEED.txt


##### Masked SELF_ATTENTION 
##### Img + COLOR (Extra Token) [Masked Self-Attention]
SEED=1245
COLOR=5
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET MODEL.NAME 'eva02_img_extra_token' \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    MODEL.MASKED_SEP_ATTN True >> outputs/"$DATASET"-CO-$COLOR-MASK-SELF-ATTN-$SEED.txt
    
    

##### FEED COLORS 
##### Img + COLOR (Extra Token) [FEED COLORS]
SEED=1245
COLOR=5
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET \
    TRAIN.COLOR_ADV True DATA.DATASET_FIX 'color_adv' TRAIN.COLOR_PROFILE $COLOR SOLVER.SEED $SEED \
    MODEL.NAME 'eva02_img_extra_token_feed' MODEL.ATT_AS_INPUT True >> outputs/"$DATASET"-CO-$COLOR-Feed.txt

    
##### GREY
SEED=1245
COLOR=5
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT DATA.DATASET $DATASET \
    SOLVER.SEED $SEED DATA.GREY_SCALE True >> outputs/"$DATASET"_img_GREY-$SEED.txt    



################################ # Testing ###################################################    
# Img STATS GFLOP AND NUMBER OF PARAMS  
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $PORT \
    train.py --config_file $CONFIG DATA.ROOT $ROOT SOLVER.SEED $SEED TEST.MODE True \
    ANALYSIS_STATS True 
    
    
    
    












