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


def run(seed):
    dfx = pd.read_csv(config.TRAINING_FILE)
    
    df_train = dfx
    df_valid = dfx

    train_dataset = dataset.ChaiiDataset(
        ids=df_train.id.values,
        contexts=df_train.context.values,
        questions=df_train.question.values,
        answers=df_train.answer_text.values,
        answer_starts=df_train.answer_start.values)

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
        answer_starts=df_valid.answer_start.values)

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

    score = engine.train_fn(train_data_loader, valid_data_loader, model, optimizer,
                    device, writer, f'{config.MODEL_SAVE_PATH}/model.bin', scheduler=scheduler, df_valid=df_valid, valid_dataset=valid_dataset)

    if config.USE_SWA:
        optimizer.swap_swa_sgd()

    return score


if __name__ == '__main__':
    for seed in config.SEEDS:
        utils.seed_everything(seed=seed)
        print(f"Training with SEED={seed}")
        writer = SummaryWriter(f"logs/seed{seed}")
        fold_score = run(seed)
        writer.close()

        print(f'Score = {fold_score}')
