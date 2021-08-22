'''
    超参数
'''

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
MODEL_NET = 'convnet' # dilatedconv、pretrain
MODEL_INCHANNEL = 3

# -----------------------------------------------------------------------------
# Images
# -----------------------------------------------------------------------------
IMG_DIR = "/root/Dataset25/train/image"
LABLE_DIR = "/root/Dataset25/train/label"

# -----------------------------------------------------------------------------
# Patch
# -----------------------------------------------------------------------------
PATCH_SIZE = 25
TUANJU_DIR = "/root/Dataset25/pixel/patches/tuanju"
OTHER_DIR = "/root/Dataset25/pixel/patches/others"

# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
OUTPUTDIR_MODEL = "/root/Dataset25/pixel/output/models"
OUTPUTDIR_LOOGER = "/root/Dataset25/pixel/output"

# -----------------------------------------------------------------------------
# Train721参数
# -------------------------------------------------------.----------------------
TRAIN721_MODEL_NET = 'convnet'
TRAIN721_DEVICE = "cuda:0"

TRAIN721_EPOCH = 10
TRAIN721_BATCH_SIZE = 64

TRAIN721_OPTIMIZER_NAME = "Adam"
TRAIN721_LR = 1e-4

# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------
MERGE_THRESHOLD = 0.7  # 在predict和merging中用到



# -----------------------------------------------------------------------------
# Multi_gpu
# -----------------------------------------------------------------------------
MULTIGPU_MODEL_NET = 'dilatedconv'
MULTIGPU_DEVICE = "cuda:0"
MULTIGPU_DEVICE_IDS = [1,2,3]

MULTIGPU_EPOCH = 20
MULTIGPU_BATCH_SIZE = 256

MULTIGPU_OPTIMIZER_NAME = "Adam"
MULTIGPU_LR = 1e-4

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
PREDICT_DEVICE = "cuda:0"
PREDICT_BATCH_SIZE = 64

PREDICT_DIR = "/root/Dataset25/pixel/output/pred"

# PREDICT_SLIC = 100000  # slic超像素分割数量（约等）
# PREDICT_MERGE = MERGE_THRESHOLD  # 融合阈值

# PREDICT_MODEL_NAME = '/home/baiyu/Maotai/Baseline/models/cv_img_s2/fold3/model_5.pt'  # 训练好的模型
PREDICT_MODEL_NAME = "/root/Dataset25/pixel/output/models/model_10.pt"


