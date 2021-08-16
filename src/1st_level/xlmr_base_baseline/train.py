import os
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
    
    dfx.rename(columns={'excerpt': 'text', 'target': 'label'}, inplace=True)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_dataset = dataset.CommonlitDataset(
        texts=df_train.text.values,
        labels=df_train.label.values)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
        shuffle=True)

    valid_dataset = dataset.CommonlitDataset(
        texts=df_valid.text.values,
        labels=df_valid.label.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
        shuffle=False)

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True
    ##
    model_config.hidden_dropout_prob = config.BERT_DROPOUT
    ##
    model = models.CommonlitModel(conf=model_config)
    model = model.to(device)

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
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_train_steps * config.WARMUP_RATIO),
        num_training_steps=num_train_steps)

    if not os.path.isdir(f'{config.MODEL_SAVE_PATH}'):
        os.makedirs(f'{config.MODEL_SAVE_PATH}')

    print(f'Training is starting for fold={fold}')

    rmse_score = engine.train_fn(train_data_loader, valid_data_loader, model, optimizer,
                    device, writer, f'{config.MODEL_SAVE_PATH}/model_{fold}_{seed}.bin', scheduler=scheduler)

    if config.USE_SWA:
        optimizer.swap_swa_sgd()

    return rmse_score


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

        print('\nScores without SWA:')
        for i in range(config.N_FOLDS):
            print(f'Fold={i}, RMSE = {fold_scores[i]}')
        print(f'Mean = {np.mean(fold_scores)}')
        print(f'Std = {np.std(fold_scores)}')
