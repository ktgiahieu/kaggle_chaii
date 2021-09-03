import sys
import pickle
import os
import gc

import numpy as np
import pandas as pd
import torch
import transformers
import tqdm

import config
import models
import dataset
import utils


def run():
    df_test = pd.read_csv(config.TEST_FILE)
    df_test.loc[:, 'answer_start'] = 0
    df_test.loc[:, 'answer_text'] = ''

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    test_dataset = dataset.ChaiiDataset(
        ids=df_test.id.values,
        contexts=df_test.context.values,
        questions=df_test.question.values,
        answers=df_test.answer_text.values,
        answer_starts=df_test.answer_start.values,
        mode='valid')

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1)
    
    predicted_labels = []
    all_models = []
    m = torch.nn.Sigmoid()
    for seed in config.SEEDS:
        model = models.ChaiiClassifierModel(conf=model_config)
        model.to(device)
        model.eval()
        all_models.append(model)
    for i in range(config.N_FOLDS):  
        for s, seed in enumerate(config.SEEDS):
            if config.is_kaggle:
                if i<=2:
                    model_path = f'{config.TRAINED_MODEL_PATH}-p1/model_{i}_{seed}.bin'
                else:
                    model_path = f'{config.TRAINED_MODEL_PATH}-p2/model_{i}_{seed}.bin'
            else:
                model_path = f'{config.TRAINED_MODEL_PATH}/model_{i}_{seed}.bin'
            all_models[s].load_state_dict(torch.load(model_path, map_location="cuda"))

        predicted_labels_per_fold = []
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
            for bi, d in enumerate(tk0):
                ids = d['ids']
                mask = d['mask']
                classifier_labels = d['classifier_labels']

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                classifier_labels = classifier_labels.to(device, dtype=torch.float)


                outputs_seeds = []
                for s in range(len(config.SEEDS)):
                    outputs = all_models[s](ids=ids, mask=mask)
                    outputs_seeds.append(outputs)

                outputs = sum(outputs_seeds) / (len(config.SEEDS))

                outputs = outputs.cpu().detach().numpy()
                predicted_labels_per_fold.extend(outputs.squeeze(-1).tolist())
        predicted_labels.append(predicted_labels_per_fold)
    predicted_labels = (m(torch.mean(
        torch.tensor(predicted_labels, dtype=torch.float), 
        dim=0)) > config.CLASSIFIER_THRESHOLD).tolist()

    df_test['predicted_labels'] = predicted_labels
    filter_df = df_test[df_test['predicted_labels']==1]

    filter_df.to_csv('train_filtered.csv', index=False)

    del test_dataset
    del data_loader
    del all_models
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please specify output pickle name."
    run()

    #            outputs_start = outputs_start.cpu().detach().numpy()
    #            outputs_end = outputs_end.cpu().detach().numpy()

    #            predicted_labels_per_fold_start.append(outputs_start)
    #            predicted_labels_per_fold_end.append(outputs_end)
        
    #    predicted_labels_per_fold_start = np.concatenate(
    #        tuple(x for x in predicted_labels_per_fold_start), axis=0)
    #    predicted_labels_per_fold_end = np.concatenate(
    #        tuple(x for x in predicted_labels_per_fold_end), axis=0)

    #    predicted_labels_start.append(predicted_labels_per_fold_start)
    #    predicted_labels_end.append(predicted_labels_per_fold_end)
    
    ## Raw predictions
    #predicted_labels_start = np.stack(
    #    tuple(x for x in predicted_labels_start), axis=0)
    #predicted_labels_start = np.mean(predicted_labels_start, axis=0)
    #predicted_labels_start = torch.softmax(predicted_labels_start, dim=-1)

    #predicted_labels_end = np.stack(
    #    tuple(x for x in predicted_labels_end), axis=0)
    #predicted_labels_end = np.mean(predicted_labels_end, axis=0)
    #predicted_labels_end = torch.softmax(predicted_labels_end, dim=-1)