import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import tokenizers
from transformers import AutoTokenizer, AutoConfig
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
comp_name = 'chaii-hindi-and-tamil-question-answering'
my_impl = 'chaii-impl'
my_model_dataset = 'chaii-xlmrbs2-c1-toext-diff-hns'
if is_kaggle:
    TRAINING_FILE = f'../input/{my_impl}/data/train_folds_external_cleaned_dropped.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'.'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/deepset-xlm-roberta-base-squad2'
else: #colab
    repo_name = 'kaggle_chaii'
    drive_name = 'Chaii'
    model_save = 'xlmrbs2_c1_toext_diff_hns'
    
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds_external_cleaned_dropped.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    TRAINING_FILE_PICKLE = f'/content/gdrive/MyDrive/Dataset/{drive_name}/oof_prob/xlmrbs2_c1_toext_diff.pkl'
    PRETRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/xlmrbs2_c1_toext_diff'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = 'deepset/xlm-roberta-base-squad2'

# Model params
SEEDS = [14, 27, 44, 1002, 2023]
N_FOLDS = 5
EPOCHS = 1
NEGATIVE_POSITIVE_RATIO = 1.0

PATIENCE = None
EARLY_STOPPING_DELTA = None
N_REINIT_LAST_LAYERS = 0
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
ACCUMULATION_STEPS = 1
MAX_LEN = 384
DOC_STRIDE = 128

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_CONFIG)

CONF = AutoConfig.from_pretrained(
    MODEL_CONFIG)

N_LAST_HIDDEN = [10,  6, 12,  8,  4]
BERT_DROPOUT = 0.1
HIGH_DROPOUT = [0.5 , 0.35, 0.3 , 0.45, 0.4 ]
SOFT_ALPHA = [0.8, 0.7, 1. , 0.6, 0.9]
WARMUP_RATIO = 0.1

USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

SAVE_CHECKPOINT_TYPE = 'last_epoch' #'best_iter', 'best_epoch' or 'last_epoch'
EVAL_SCHEDULE = [
                (10., 200*ACCUMULATION_STEPS),
                ]


#Layer wise learning rate
HEAD_LEARNING_RATE = 3e-5
LEARNING_RATE_LAYERWISE_TYPE = 'exponential' #'linear' or 'exponential'
LEARNING_RATES_RANGE = [3e-5, 3e-5]
WEIGHT_DECAY = 0.001