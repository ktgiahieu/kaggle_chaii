import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import tokenizers
from transformers import AutoTokenizer, AutoConfig
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
comp_name = 'chaii-hindi-and-tamil-question-answering'
my_impl = 'chaii-impl'
my_model_dataset = 'chaii-3f-xlmrls2-c1-toext-diff1'
if is_kaggle:
    TRAINING_FILE = f'../input/{my_impl}/data/3f_train_folds_external_cleaned_dropped.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'.'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/deepset-xlm-roberta-large-squad2'
else: #colab
    repo_name = 'kaggle_chaii'
    drive_name = 'Chaii'
    model_save = '3f_xlmrls2_c1_toext_diff1'
    
    TRAINING_FILE = f'/content/{repo_name}/data/3f_train_folds_external_cleaned_dropped.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = 'deepset/xlm-roberta-large-squad2'

# Model params
SEEDS = [42, 1000, 2021]
N_FOLDS = 3
EPOCHS = 1
#NEGATIVE_POSITIVE_RATIO = 3.0

PATIENCE = None
EARLY_STOPPING_DELTA = None
N_REINIT_LAST_LAYERS = 0
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
MAX_LEN = 384
DOC_STRIDE = 128

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_CONFIG)

CONF = AutoConfig.from_pretrained(
    MODEL_CONFIG)

N_LAST_HIDDEN = [8,12,16]
BERT_DROPOUT = 0.1
HIGH_DROPOUT = [0.3, 0.4, 0.5]
SOFT_ALPHA = [0.6, 0.8, 1.0]
WARMUP_RATIO = 0.1

USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

SAVE_CHECKPOINT_TYPE = 'last_epoch' #'best_iter', 'best_epoch' or 'last_epoch'
EVAL_SCHEDULE = [
                (10., 200*ACCUMULATION_STEPS),
                ]


#Layer wise learning rate
HEAD_LEARNING_RATE = 2e-5
LEARNING_RATE_LAYERWISE_TYPE = 'exponential' #'linear' or 'exponential'
LEARNING_RATES_RANGE = [5e-6, 1e-5]
WEIGHT_DECAY = 0.01