import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import tokenizers
from transformers import AutoTokenizer, AutoConfig
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
comp_name = 'chaii-hindi-and-tamil-question-answering'
my_impl = 'chaii-impl'
my_model_dataset = 'chaii-muril-c1-hns-pretrained-ce'
if is_kaggle:
    TRAINING_FILE = f'../input/{my_impl}/data/train_folds_cleaned_dropped.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'.'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/google-muril-base-cased'
else: #colab
    repo_name = 'kaggle_chaii'
    drive_name = 'Chaii'
    model_save = 'muril_c1_hns_pretrained_ce'
    
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds_cleaned_dropped.csv'
    TRAINING_FILE_PICKLE = f'/content/gdrive/MyDrive/Dataset/{drive_name}/oof_prob/muril_c1_u_pretrained_ce.pkl'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    PRETRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/pretrained/muril_combo1_external_ce/model.bin'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = 'google/muril-base-cased'

# Model params
SEEDS = [1000]
N_FOLDS = 5
EPOCHS = 2
NEGATIVE_POSITIVE_RATIO = 2.0

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

N_LAST_HIDDEN = 4
BERT_DROPOUT = 0
HIGH_DROPOUT = 0.5
SOFT_ALPHA = 1.0
WARMUP_RATIO = 0.1

USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

SAVE_CHECKPOINT_TYPE = 'last_epoch' #'best_iter', 'best_epoch' or 'last_epoch'
EVAL_SCHEDULE = [
                (10., 60*ACCUMULATION_STEPS),
                ]


#Layer wise learning rate
HEAD_LEARNING_RATE = 1e-4
LEARNING_RATE_LAYERWISE_TYPE = 'exponential' #'linear' or 'exponential'
LEARNING_RATES_RANGE = [1e-5, 5e-5]
WEIGHT_DECAY = 0.01