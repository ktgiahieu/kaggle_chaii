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
    df_test['context'] = df_test['context'].apply(lambda x: ' '.join(x.split()))
    df_test['question'] = df_test['question'].apply(lambda x: ' '.join(x.split()))
    df_test = df_test[df_test['language']=='tamil'].reset_index(drop=True)

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    test_dataset = dataset.ChaiiDataset(
        fold=0,
        ids=df_test.id.values,
        contexts=df_test.context.values,
        questions=df_test.question.values,
        answers=df_test.answer_text.values,
        answer_starts=df_test.answer_start.values,
        languages=df_test.language.values,
        mode='infer')

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.INFER_BATCH_SIZE,
        num_workers=1)
    
    predicted_labels_start = []
    predicted_labels_end = []

        
    for i in range(config.N_FOLDS): 
        print(f'Infer fold {i+1}')
        seed = config.SEEDS[i]
        model = models.ChaiiModel(conf=model_config, fold=i)
        model.to(device)
        model.eval()
        if config.is_kaggle:
            model_path = f'{config.TRAINED_MODEL_PATH}/model_{i+1}_{seed}.bin'
        else:
            model_path = f'{config.TRAINED_MODEL_PATH}/model_{i+1}_{seed}.bin'
        model.load_state_dict(torch.load(model_path, map_location="cuda"))

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

                outputs_start, outputs_end = model(ids=ids, mask=mask)

                outputs_start = outputs_start.cpu().detach()
                outputs_end = outputs_end.cpu().detach()

                predicted_labels_per_fold_start.append(outputs_start)
                predicted_labels_per_fold_end.append(outputs_end)
        
        predicted_labels_per_fold_start = torch.cat(
            tuple(x for x in predicted_labels_per_fold_start), dim=0)
        predicted_labels_per_fold_end = torch.cat(
            tuple(x for x in predicted_labels_per_fold_end), dim=0)

        predicted_labels_start.append(predicted_labels_per_fold_start)
        predicted_labels_end.append(predicted_labels_per_fold_end)

        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Raw predictions
    predicted_labels_start = torch.stack(
        tuple(x for x in predicted_labels_start), dim=0)
    predicted_labels_start = torch.mean(predicted_labels_start, dim=0)

    predicted_labels_end = torch.stack(
        tuple(x for x in predicted_labels_end), dim=0)
    predicted_labels_end = torch.mean(predicted_labels_end, dim=0)

    # Heatmap 
    heatmap_logit = utils.postprocess_char_prob(df_test, test_dataset.features, 
                                                   (predicted_labels_start, predicted_labels_end))

    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')
        
    pickle_name = sys.argv[1]
    with open(f'{config.INFERED_PICKLE_PATH}/{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(heatmap_logit, handle)

    del test_dataset
    del data_loader
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please specify output pickle name."
    run()