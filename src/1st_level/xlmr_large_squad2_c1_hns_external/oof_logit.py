import sys
import os
import gc
import pickle
import torch
import numpy as np
import pandas as pd
import transformers
import tqdm.autonotebook as tqdm

import utils
import config
import models
import dataset
import engine
heatmap_logit = {}

def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    seed_models = []
    for seed in config.SEEDS:
        model = models.ChaiiModel(conf=model_config)
        model.to(device)
        model.load_state_dict(torch.load(
            f'{config.TRAINED_MODEL_PATH}/model_{fold}_{seed}.bin'),
            strict=False)
        model.eval()
        seed_models.append(model)

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

    
    losses = utils.AverageMeter()
    predicted_labels_per_fold_start = []
    predicted_labels_per_fold_end = []
    with torch.no_grad():
      
        tk0 = tqdm.tqdm(valid_data_loader, total=len(valid_data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            mask = d['mask']
            start_labels = d['start_labels']
            end_labels = d['end_labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            start_labels = start_labels.to(device, dtype=torch.float)
            end_labels = start_labels.to(device, dtype=torch.float)

            outputs_seeds_start = []
            outputs_seeds_end = []
            for i in range(len(config.SEEDS)):
                outputs_start, outputs_end = seed_models[i](ids=ids, mask=mask)

                outputs_seeds_start.append(outputs_start)
                outputs_seeds_end.append(outputs_end)

            outputs_start = sum(outputs_seeds_start) / (len(config.SEEDS))
            outputs_end = sum(outputs_seeds_end) / (len(config.SEEDS))
            
            loss = engine.loss_fn(outputs_start, outputs_end,
                           start_labels, end_labels)
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)

            outputs_start = outputs_start.cpu().detach()
            outputs_end = outputs_end.cpu().detach()

            predicted_labels_per_fold_start.append(outputs_start)
            predicted_labels_per_fold_end.append(outputs_end)
    
    # Raw predictions
    predicted_labels_per_fold_start = torch.cat(
        tuple(x for x in predicted_labels_per_fold_start), dim=0)
    predicted_labels_per_fold_end = torch.cat(
        tuple(x for x in predicted_labels_per_fold_end), dim=0)

    # Heatmap logit
    heatmap_logit_per_fold = utils.postprocess_heatmap_logit(df_valid, valid_dataset.features, 
                                                   (predicted_labels_per_fold_start, predicted_labels_per_fold_end))
    heatmap_logit.update(heatmap_logit_per_fold)


if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please specify output pickle name."
    utils.seed_everything(seed=config.SEEDS[0])
    for i in range(config.N_FOLDS):
        run(i)
        torch.cuda.empty_cache()
        gc.collect()

    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')

    pickle_name = sys.argv[1]
    with open(f'{config.INFERED_PICKLE_PATH}/{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(heatmap_logit, handle)
