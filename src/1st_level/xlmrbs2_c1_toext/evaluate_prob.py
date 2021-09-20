import sys
import os
import gc
import pickle
import torch
import numpy as np
import pandas as pd
import transformers
import tqdm.autonotebook as tqdm
from sklearn.metrics import confusion_matrix


import utils
import config
import models
import dataset
import engine


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
    true_labels = []
    predicted_labels = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    m = torch.nn.Sigmoid()
    with torch.no_grad():
      
        tk0 = tqdm.tqdm(valid_data_loader, total=len(valid_data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            mask = d['mask']
            start_labels = d['start_labels']
            end_labels = d['end_labels']
            classifier_labels = start_labels[:,0] == 0

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            start_labels = start_labels.to(device, dtype=torch.float)
            end_labels = start_labels.to(device, dtype=torch.float)
            classifier_labels = classifier_labels.to(device, dtype=torch.float)

            outputs_seeds = []
            for i in range(len(config.SEEDS)):
                outputs_start, outputs_end = seed_models[i](ids=ids, mask=mask)
                outputs = 1 - (torch.softmax(outputs_start[:,:-1],dim=1)[:,0] + torch.softmax(outputs_end[:,:-1],dim=1)[:,0])/2
                outputs_seeds.append(outputs)

            outputs = sum(outputs_seeds) / (len(config.SEEDS))

            print(outputs)
            
            loss = engine.classifier_loss_fn(outputs, classifier_labels)

            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)

            if len(outputs.size()) > 1:
                outputs = outputs.squeeze(-1)
            outputs = outputs.cpu().detach().numpy() # 0 - 1

            if len(classifier_labels.size()) > 1:
                classifier_labels = classifier_labels.squeeze(-1)
            classifier_labels = classifier_labels.cpu().detach().numpy()
            
            true_labels.extend(classifier_labels.tolist())
            predicted_labels.extend(outputs.tolist())

            outputs = outputs > 0.5
            tn, fp, fn, tp = confusion_matrix(classifier_labels, outputs, labels=[0, 1]).ravel()

            TP += tp
            TN += tn
            FP += fp
            FN += fn
    
    all_features = valid_dataset.features
    for i,feature in enumerate(all_features):
        feature['true_labels'] = true_labels[i]
        feature['predicted_labels'] = predicted_labels[i]
        feature['kfold'] = fold
        all_features[i] = feature

    print(f'Loss = {losses.avg}')
    positive_rate = (TP+FP)/(TP+TN+FP+FN)
    print(f'Val positive_rate = {positive_rate}')
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print(f'Val accuracy = {accuracy}')
    recall = TP/(TP+FN)
    print(f'Val recall = {recall}')

    return all_features


if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please specify output pickle name."
    utils.seed_everything(seed=config.SEEDS[0])
    final_features = []
    for i in range(5):
        all_features = run(i)
        final_features.extend(all_features)
        torch.cuda.empty_cache()
        gc.collect()

    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')

    pickle_name = sys.argv[1]
    with open(f'{config.INFERED_PICKLE_PATH}/{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(final_features, handle)