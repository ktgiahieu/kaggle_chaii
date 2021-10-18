import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import tokenizers
from transformers import AutoTokenizer, AutoConfig
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
comp_name = 'chaii-hindi-and-tamil-question-answering'
my_impl = 'chaii-impl'
my_model_dataset = 'chaii-3f-muril-ls2-5e6-c1-ext-diff'
if is_kaggle:
    TRAINING_FILE = f'../input/{my_impl}/data/external_dropped.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'.'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/google-muril-large-cased'
else: #colab
    repo_name = 'kaggle_chaii'
    drive_name = 'Chaii'
    model_save = '3f_muril_ls2_5e6_c1_ext_diff'
    
    TRAINING_FILE = f'/content/{repo_name}/data/external_dropped.csv'
    VALID_FILE = f'/content/{repo_name}/data/3f_train_folds_external_cleaned_dropped.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/pretrained/google-muril-large-cased-squad2-1ep-5e6'

# Model params
SEEDS = [43, 1101, 2022]
N_FOLDS = 3
EPOCHS = 2
#NEGATIVE_POSITIVE_RATIO = 3.0

PATIENCE = None
EARLY_STOPPING_DELTA = None
N_REINIT_LAST_LAYERS = 0
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
ACCUMULATION_STEPS = 1
MAX_LEN = 384
DOC_STRIDE = 128

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_CONFIG)

CONF = AutoConfig.from_pretrained(
    MODEL_CONFIG)

N_LAST_HIDDEN = [6,  9, 12]
BERT_DROPOUT = 0.1
HIGH_DROPOUT = [0.45, 0.4, 0.3]
SOFT_ALPHA = [0.9, 0.8, 0.7]
WARMUP_RATIO = 0.1

USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

SAVE_CHECKPOINT_TYPE = 'best_epoch' #'best_iter', 'best_epoch' or 'last_epoch'
EVAL_SCHEDULE = [
                (10., 200*ACCUMULATION_STEPS),
                ]


#Layer wise learning rate
HEAD_LEARNING_RATE = 1e-5
LEARNING_RATE_LAYERWISE_TYPE = 'exponential' #'linear' or 'exponential'
LEARNING_RATES_RANGE = [2e-6, 5e-6]
WEIGHT_DECAY = 0.001