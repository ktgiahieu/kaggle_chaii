import os
import pickle
import numpy as np
import pandas as pd
import transformers
import torch
import torchcontrib
from torch.utils.tensorboard import SummaryWriter
writer = None

import config
import dataset
import models
import engine
import utils


def run(fold, seed):
    dfx = pd.read_csv(config.TRAINING_FILE)
    
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    with open(config.TRAINING_FILE_PICKLE, 'rb') as handle:
        hns_features = pickle.load(handle)

    hns_features = [x for x in hns_features if x['kfold'] != fold]

    train_dataset = dataset.ChaiiDataset(
        ids=df_train.id.values,
        contexts=df_train.context.values,
        questions=df_train.question.values,
        answers=df_train.answer_text.values,
        answer_starts=df_train.answer_start.values,
        hns_features=hns_features,
        mode='train')

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
        shuffle=True)

    valid_dataset = dataset.ChaiiDataset(
        ids=df_valid.id.values,
        contexts=df_valid.context.values,
        questions=df_valid.question.values,
        answers=df_valid.answer_text.values,
        answer_starts=df_valid.answer_start.values,
        mode='valid')

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
        shuffle=False)

    device = torch.device('cuda')
    model_config = config.MODEL_CONF
    model_config.output_hidden_states = True
    ##
    model_config.hidden_dropout_prob = config.BERT_DROPOUT
    ##
    model = models.ChaiiModel(conf=model_config)
    model = model.to(device)

    model.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH, map_location="cuda"))

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': config.WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    base_opt = utils.create_optimizer(model)
    optimizer = torchcontrib.optim.SWA(
        base_opt,
        swa_start=int(num_train_steps * config.SWA_RATIO),
        swa_freq=config.SWA_FREQ,
        swa_lr=None)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_train_steps * config.WARMUP_RATIO),
        num_training_steps=num_train_steps)

    if not os.path.isdir(f'{config.MODEL_SAVE_PATH}'):
        os.makedirs(f'{config.MODEL_SAVE_PATH}')

    print(f'Training is starting for fold={fold}')

    score = engine.train_fn(train_data_loader, valid_data_loader, model, optimizer,
                    device, writer, f'{config.MODEL_SAVE_PATH}/model_{fold}_{seed}.bin', scheduler=scheduler, df_valid=df_valid, valid_dataset=valid_dataset)

    if config.USE_SWA:
        optimizer.swap_swa_sgd()

    return score


if __name__ == '__main__':
    for seed in config.SEEDS:
        utils.seed_everything(seed=seed)
        print(f"Training with SEED={seed}")
        fold_scores = []
        for i in range(config.N_FOLDS):
            writer = SummaryWriter(f"logs/fold{i}_seed{seed}")
            fold_score = run(i, seed)
            fold_scores.append(fold_score)
            writer.close()

        if len(fold_scores)==config.N_FOLDS and fold_scores[0] is not None:
            print('\nScores without SWA:')
            for i in range(config.N_FOLDS):
                print(f'Fold={i}, Score = {fold_scores[i]}')
            print(f'Mean = {np.mean(fold_scores)}')
            print(f'Std = {np.std(fold_scores)}')
