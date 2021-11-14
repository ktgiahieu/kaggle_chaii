import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import tokenizers
from transformers import AutoTokenizer, AutoConfig
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
comp_name = 'chaii-hindi-and-tamil-question-answering'
my_impl = 'chaii-impl'
my_model_dataset = 'chaii-muril-ls2-apex-know'
if is_kaggle:
    TRAINING_FILE = f'../input/{my_impl}/data/train_folds_external_cleaned_dropped.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'.'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/ggoogle-muril-large-cased-squad2-1ep-5e6'

else: #colab
    repo_name = 'kaggle_chaii'
    drive_name = 'Chaii'
    model_save = 'muril_ls2_apex_know'
    
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds_external_cleaned_dropped.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    TEACHER_PICKLE_FILE = f'/content/gdrive/MyDrive/Dataset/{drive_name}/teacher_logit/ensemble_v1_13_train_logit.pkl'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG =f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/pretrained/google-muril-large-cased-squad2-1ep-5e6'

DEBUG = False
USE_APEX = False
# Model params
SEEDS = [43, 1101, 2022, 12, 45,22, 5]
N_FOLDS = 7
EPOCHS = 1
NEGATIVE_POSITIVE_RATIO = 1.0

SHUFFLE_AUGMENT_RATE = 0.0

PATIENCE = None
EARLY_STOPPING_DELTA = None
N_REINIT_LAST_LAYERS = 0
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 32
ACCUMULATION_STEPS = 1
MAX_LEN = 384
DOC_STRIDE = 128

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_CONFIG)

CONF = AutoConfig.from_pretrained(
    MODEL_CONFIG)

N_LAST_HIDDEN = [6,  9, 12, 4, 8, 5, 10]
BERT_DROPOUT = 0.1
HIGH_DROPOUT = [0.45, 0.4, 0.3, 0.5, 0.35, 0.25, 0.55]
SOFT_ALPHA = [0.9, 0.8, 0.7, 0.5, 0.55, 0.75, 0.6]
WARMUP_RATIO = 0.1

USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

SAVE_CHECKPOINT_TYPE = 'best_epoch' #'best_iter', 'best_epoch' or 'last_epoch'
EVAL_SCHEDULE = [
                (10., 2000*ACCUMULATION_STEPS),
                ]