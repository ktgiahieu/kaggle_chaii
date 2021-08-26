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
true_labels = []
predicted_labels = []

def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    seed_models = []
    for seed in config.SEEDS:
        model = models.ChaiiClassifierModel(conf=model_config)
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
    predicted_labels_per_fold = []
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
            classifier_labels = d['classifier_labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            classifier_labels = classifier_labels.to(device, dtype=torch.float)

            outputs_seeds = []
            for i in range(len(config.SEEDS)):
                outputs = seed_models[i](ids=ids, mask=mask)

                outputs_seeds.append(outputs)

            outputs = sum(outputs_seeds) / (len(config.SEEDS))
            
            loss = engine.classifier_loss_fn(outputs, classifier_labels)

            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)

            outputs = (m(outputs.squeeze(-1)) > config.CLASSIFIER_THRESHOLD).cpu().detach().numpy() # 0 or 1
            tn, fp, fn, tp = confusion_matrix(classifier_labels.squeeze(-1).cpu().detach().numpy(), 
                                              outputs, labels=[0, 1]).ravel()

            TP += tp
            TN += tn
            FP += fp
            FN += fn

            true_labels.extend(classifier_labels.squeeze(-1).cpu().detach().numpy().tolist())
            predicted_labels.extend(outputs.tolist())
    
    print(f'Loss = {losses.avg}')
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print(f'Val accuracy = {accuracy}')
    recall = TP/(TP+FN)
    print(f'Val recall = {recall}')

    return recall


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
        print(f'Fold={i}, Recall = {fold_scores[i]}')
    print(f'Mean = {np.mean(fold_scores)}')
    print(f'Std = {np.std(fold_scores)}')

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()
    oof_accuracy = (tp+tn)/(tp+tn+fp+fn)
    print(f'OOF accuracy = {oof_accuracy}')
    oof_recall = tp/(tp+fn)
    print(f'OOF recall = {oof_recall}')

    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')

    pickle_name = sys.argv[1]
    with open(f'{config.INFERED_PICKLE_PATH}/{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(predicted_labels, handle)
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
true_labels = []
predicted_labels = []

def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    seed_models = []
    for seed in config.SEEDS:
        model = models.ChaiiClassifierModel(conf=model_config)
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
        answer_starts=df_valid.answer_start.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
        shuffle=False)

    
    losses = utils.AverageMeter()
    predicted_labels_per_fold = []
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
            classifier_labels = d['classifier_labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            classifier_labels = classifier_labels.to(device, dtype=torch.float)

            outputs_seeds = []
            for i in range(len(config.SEEDS)):
                outputs = seed_models[i](ids=ids, mask=mask)

                outputs_seeds.append(outputs)

            outputs = sum(outputs_seeds) / (len(config.SEEDS))
            
            loss = engine.classifier_loss_fn(outputs, classifier_labels)

            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)

            outputs = (m(outputs.squeeze(-1)) > config.CLASSIFIER_THRESHOLD).cpu().detach().numpy() # 0 or 1
            tn, fp, fn, tp = confusion_matrix(classifier_labels.squeeze(-1).cpu().detach().numpy(), 
                                              outputs, labels=[0, 1]).ravel()

            TP += tp
            TN += tn
            FP += fp
            FN += fn

            true_labels.extend(classifier_labels.squeeze(-1).cpu().detach().numpy().tolist())
            predicted_labels.extend(outputs.tolist())
    
    print(f'Loss = {losses.avg}')
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print(f'Val accuracy = {accuracy}')
    recall = TP/(TP+FN)
    print(f'Val recall = {recall}')

    return recall


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
        print(f'Fold={i}, Recall = {fold_scores[i]}')
    print(f'Mean = {np.mean(fold_scores)}')
    print(f'Std = {np.std(fold_scores)}')

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()
    oof_accuracy = (tp+tn)/(tp+tn+fp+fn)
    print(f'OOF accuracy = {oof_accuracy}')
    oof_recall = tp/(tp+fn)
    print(f'OOF recall = {oof_recall}')

    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')

    pickle_name = sys.argv[1]
    with open(f'{config.INFERED_PICKLE_PATH}/{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(predicted_labels, handle)
