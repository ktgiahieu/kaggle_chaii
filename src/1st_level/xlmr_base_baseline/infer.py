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

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    test_dataset = dataset.ChaiiDataset(
        ids=df_test.id.values,
        contexts=df_test.context.values,
        questions=df_test.question.values,)

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1)
    
    predicted_labels_start = []
    predicted_labels_end = []
    all_models = []
    for seed in config.SEEDS:
        model = models.CommonlitModel(conf=model_config)
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

        predicted_labels_per_fold_start = []
        predicted_labels_per_fold_end = []
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
            for bi, d in enumerate(tk0):
                ids = d['ids']
                mask = d['mask']
                start_labels = d['start_labels']
                end_labels = d['end_labels']

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                start_labels = start_labels.to(device, dtype=torch.float)
                end_labels = start_labels.to(device, dtype=torch.float)


                outputs_seeds_start, outputs_seeds_end = []
                for s in range(len(config.SEEDS)):
                    outputs_start, outputs_end = all_models[s](ids=ids, mask=mask)

                    outputs_seeds_start.append(outputs_start)
                    outputs_seeds_end.append(outputs_end)

                outputs_start = sum(outputs_seeds_start) / (len(config.SEEDS))
                outputs_end = sum(outputs_seeds_end) / (len(config.SEEDS))

                outputs_start = outputs_start.cpu().detach().numpy()
                outputs_end = outputs_end.cpu().detach().numpy()

                predicted_labels_per_fold_start.append(outputs_start)
                predicted_labels_per_fold_end.append(outputs_end)
        
        predicted_labels_per_fold_start = torch.cat(
            tuple(x for x in predicted_labels_per_fold_start), dim=0)
        predicted_labels_per_fold_end = torch.cat(
            tuple(x for x in predicted_labels_per_fold_end), dim=0)

        predicted_labels_start.append(predicted_labels_per_fold_start)
        predicted_labels_end.append(predicted_labels_per_fold_end)
    
    # Raw predictions
    predicted_labels_start = torch.stack(
        tuple(x for x in predicted_labels_start), dim=0)
    predicted_labels_start = torch.mean(predicted_labels_start, dim=0)
    predicted_labels_start = torch.softmax(predicted_labels_start, dim=-1)

    predicted_labels_end = torch.stack(
        tuple(x for x in predicted_labels_end), dim=0)
    predicted_labels_end = torch.mean(predicted_labels_end, dim=0)
    predicted_labels_end = torch.softmax(predicted_labels_end, dim=-1)

    #Post process 
    #(predictions = {'id': 'predicted_text', ...} )
    predictions = utils.postprocess_qa_predictions(df_test, test_dataset.features, 
                                                   (predicted_labels_start, predicted_labels_end))

    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')
        
    pickle_name = sys.argv[1]
    with open(f'{config.INFERED_PICKLE_PATH}/{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(predictions, handle)

    del test_dataset
    del data_loader
    del all_models
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please specify output pickle name."
    run()
