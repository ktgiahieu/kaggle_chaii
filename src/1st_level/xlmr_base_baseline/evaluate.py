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
predicted_labels = []

def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)
    dfx.rename(columns={'excerpt': 'text', 'target': 'label'}, inplace=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    seed_models = []
    for seed in config.SEEDS:
        model = models.CommonlitModel(conf=model_config)
        model.to(device)
        model.load_state_dict(torch.load(
            f'{config.TRAINED_MODEL_PATH}/model_{fold}_{seed}.bin'),
            strict=False)
        model.eval()
        seed_models.append(model)

    valid_dataset = dataset.CommonlitDataset(
        texts=df_valid.text.values,
        labels=df_valid.label.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
        shuffle=False)

    
    losses = utils.AverageMeter()

    with torch.no_grad():
      
        tk0 = tqdm.tqdm(valid_data_loader, total=len(valid_data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            mask = d['mask']
            labels = d['labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            outputs_seeds = []
            for i in range(len(config.SEEDS)):
                outputs = seed_models[i](ids=ids, mask=mask)

                outputs_seeds.append(outputs)

            outputs = sum(outputs_seeds) / len(config.SEEDS)
            
            loss = engine.loss_fn(outputs, labels)
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=np.sqrt(losses.avg))

            outputs = outputs.cpu().detach().numpy()
            predicted_labels.extend(outputs.squeeze(-1).tolist())
    print(f'RMSE = {np.sqrt(losses.avg)}')
    return np.sqrt(losses.avg)


if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please specify output pickle name."
    utils.seed_everything(seed=config.SEEDS[0])
    fold_scores = []
    for i in range(config.N_FOLDS):
        fold_score = run(i)
        fold_scores.append(fold_score)
        torch.cuda.empty_cache()
        gc.collect()

    for i in range(config.N_FOLDS):
        print(f'Fold={i}, RMSE = {fold_scores[i]}')
    print(f'Mean = {np.mean(fold_scores)}')
    print(f'Std = {np.std(fold_scores)}')

    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')

    pickle_name = sys.argv[1]
    with open(f'{config.INFERED_PICKLE_PATH}/{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(predicted_labels, handle)
