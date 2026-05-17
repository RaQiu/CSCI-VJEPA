from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# Name of backbone
_C.MODEL.TYPE = 'eva02_cloth'
# Model name
_C.MODEL.NAME = 'eva02_l_cloth'


# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = True 
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = True
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'off'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False
# Dimension of the attribute list
_C.MODEL.META_DIMS = []
_C.MODEL.CLOTH_XISHU = 3
# Add attributes in model, options: 'True', 'False'
_C.MODEL.ADD_META = True
# Mask cloth attributes, options: 'True', 'False'
_C.MODEL.MASK_META = False
# Add cloth embedding only, options: 'True', 'False'
_C.MODEL.CLOTH_ONLY = True 
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'


_C.MODEL.TIM_DIM = 4
_C.MODEL.Joint = None
_C.MODEL.Adapter = None
_C.MODEL.MOTION_LOSS = None 
_C.MODEL.SPATIAL_AVG = None 
_C.MODEL.TEMPORAL_AVG = None 
_C.MODEL.TEMPORAL_AVG = None 
_C.MODEL.PRETRAIN = True   
_C.MODEL.EMBED_DIM = 1024  

_C.MODEL.EXTRA_DIM = 1024  
_C.MODEL.EXTRA_DISENTANGLE = None  
_C.MODEL.CLOTH_EMBED = None  
_C.MODEL.UNIFIED_DIST = None 
_C.MODEL.MASKED_SEP_ATTN = None 
_C.MODEL.ATT_AS_INPUT = None 
_C.MODEL.ATT_DIRECT = None 

_C.MODEL.RETURN_EARLY = None 

_C.MODEL.JEPA_ENABLE = False
_C.MODEL.JEPA_CACHE_DIR = ''
_C.MODEL.JEPA_ROOT = ''
_C.MODEL.JEPA_CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt'
_C.MODEL.JEPA_CHECKPOINT_FILE = 'vjepa2_1_vitG_384.pt'
_C.MODEL.JEPA_DEVICE = 'cuda'
_C.MODEL.JEPA_DTYPE = 'float16'
_C.MODEL.JEPA_IN_DIM = 1664
_C.MODEL.JEPA_FUSION_MODE = 'dense_patch'
_C.MODEL.JEPA_CONTEXT_TOKENS = 4
_C.MODEL.JEPA_ADAPTER_HIDDEN = 4096
_C.MODEL.JEPA_FUSION_HEADS = 8
_C.MODEL.JEPA_FUSION_LAYERS = 1
_C.MODEL.JEPA_GATE_LOGIT_INIT = -4.59512
_C.MODEL.JEPA_CACHE_REQUIRED = True
_C.MODEL.JEPA_RELEASE_ENCODER_AFTER_BATCH = False
_C.MODEL.JEPA_WRITE_TRAIN_CACHE = False

# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1

_C.TRAIN.TRAIN_VIDEO = None
_C.TRAIN.E2E = True 
_C.TRAIN.DEBUG = None 

_C.TRAIN.TEACH1 = None 
_C.TRAIN.DIR_TEACH1 = None
_C.TRAIN.TEACH1_MODEL = None 
_C.TRAIN.TEACH1_MODEL_WT = None 
_C.TRAIN.TEACH_METHOD = None 
_C.TRAIN.TEACH_DATASET_FIX = None 
_C.TRAIN.TEACH1_NUMCLASSES = None 

_C.TRAIN.PAIR_MSE = None
_C.TRAIN.COLOR_ADV = None
_C.TRAIN.COLOR_LOSS = None
_C.TRAIN.POSE_ONLY = None
_C.TRAIN.POSE = None
_C.TRAIN.LAYER_DISESNTANGLE = None

_C.TRAIN.COLOR_PROFILE = None
_C.TRAIN.GENDER = None

_C.TRAIN.HYBRID = None
_C.TRAIN.TEACH1_LOAD_AS_IMG = None 

_C.TRAIN.CONT_ONLY = None
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 8
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_HEIGHT = 224
_C.DATA.IMG_WIDTH = 224
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4
# Data root
_C.DATA.ROOT = '../Data'
# Number of instances
_C.DATA.NUM_INSTANCES = 2 #8
# Batch size during testing
_C.DATA.TEST_BATCH = 128
# Data sampling strategy
_C.DATA.SAMPLER = 'softmax_triplet'
# Extract data containing attributes during data processing, options: 'True', 'False'
_C.DATA.AUX_INFO = True
# Filename containing attributes
_C.DATA.META_DIR = 'PAR_PETA_105.txt'

_C.DATA.ADD_META = False
_C.DATA.MASK_META = False

_C.DATA.DENNIS_MODE = False

_C.DATA.F8 = None
_C.DATA.DATASET_FIX = None

_C.DATA.SAMPLING_PERCENTAGE = None 
_C.DATA.DATASET_SAMPLING_PERCENTAGE = None 
_C.DATA.RANDOM_FRAMES = None
_C.DATA.GREY_SCALE = None

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()

# Random crop prob
_C.AUG.RC_PROB = 0.5
# Random erase prob
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5

_C.AUG.TEMPORAL_SAMPLING_MODE = 'stride'
_C.AUG.SEQ_LEN = 8 
_C.AUG.SAMPLING_STRIDE = 4

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 60
# Base learning rate
_C.SOLVER.BASE_LR = 2e-5
_C.SOLVER.WARMUP_LR = 7.8125e-07
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 2
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.05
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.05

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 60)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 20
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 60
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Path to trained model
_C.TEST.WEIGHT = ""
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Test using images only
_C.TEST.TYPE = 'image_only'

_C.TEST.MODE= None 
_C.TEST.CONCAT_COLORS= None 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
_C.TENSORBOARD = None
_C.TAG = None
_C.TRAIN_DUMP = None
_C.GRAD_CAM = None
_C.ANALYSIS_STATS = None
_C.AUX_DUMP = None
