import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import tokenizers
from transformers import AutoTokenizer, AutoConfig
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
comp_name = 'chaii-hindi-and-tamil-question-answering'
my_impl = 'chaii-impl'
my_model_dataset = 'chaii-remberts2-1e6-c1-ext-wu-hns'
if is_kaggle:
    TRAINING_FILE = f'../input/{my_impl}/data/train_folds_external_cleaned_dropped.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'.'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/google-rembert'
else: #colab
    repo_name = 'kaggle_chaii'
    drive_name = 'Chaii'
    model_save = 'remberts2_1e6_c1_ext_wu_hns'
    
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds_external_cleaned_dropped.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    TRAINING_FILE_PICKLE = f'/content/gdrive/MyDrive/Dataset/{drive_name}/oof_prob/remberts2_1e6_c1_ext_wu.pkl'
    PRETRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/remberts2_1e6_c1_ext_wu'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/pretrained/google-rembert-squad2-4ep-1e6/checkpoint-150000'
# Model params
SEEDS = [15, 28, 45, 1003, 2024]
N_FOLDS = 5
EPOCHS = 5
NEGATIVE_POSITIVE_RATIO = 1.0

SAVE_CHECKPOINT = True
PATIENCE = None
EARLY_STOPPING_DELTA = None
N_REINIT_LAST_LAYERS = 0
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
ACCUMULATION_STEPS = 8
MAX_LEN = 384
DOC_STRIDE = 128

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_CONFIG)

CONF = AutoConfig.from_pretrained(
    MODEL_CONFIG)

N_LAST_HIDDEN = [20,  8, 24, 16, 12]
BERT_DROPOUT = 0.1
HIGH_DROPOUT = [0.35, 0.5 , 0.4 , 0.3 , 0.45]
SOFT_ALPHA = [1.0, 0.6, 0.9, 0.8, 0.7 ]
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
LEARNING_RATES_RANGE = [5e-6, 5e-6]
WEIGHT_DECAY = 0.001