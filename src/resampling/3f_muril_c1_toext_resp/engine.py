from shutil import copyfile

import numpy as np
import torch
import tqdm
import gc

import config
import utils

from string import punctuation

def loss_fn(start_logits, end_logits,
            start_positions, end_positions):
    m = torch.nn.LogSoftmax(dim=1)
    loss_fct = torch.nn.KLDivLoss()
    start_loss = loss_fct(m(start_logits), start_positions)
    end_loss = loss_fct(m(end_logits), end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss

def classifier_loss_fn(logits, labels):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    loss = loss_fct(m(logits), labels)
    return loss

def train_fn(train_dataset, valid_dataset, model, optimizer, device, writer, model_path, scheduler=None, df_valid=None, train_data_loader_for_hns=None, train_dataset_for_hns=None):  
    model_path_filename = model_path.split('/')[-1]
    best_val_score = None
    step = 0
    last_eval_step = 0
    hns_features = None
    eval_period = config.EVAL_SCHEDULE[0][1]   

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
        shuffle=False)

    for epoch in range(config.EPOCHS):
        if hns_features is not None:
            train_dataset.resample_hns(hns_features)

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            num_workers=4,
            shuffle=True)

        losses = utils.AverageMeter()
        tk0 = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
        model.zero_grad()
        for bi, d in enumerate(tk0):
            ids = d['ids']
            mask = d['mask']
            start_labels = d['start_labels']
            end_labels = d['end_labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            start_labels = start_labels.to(device, dtype=torch.float)
            end_labels = end_labels.to(device, dtype=torch.float)

            model.train()
            
            outputs_start, outputs_end = model(ids=ids, mask=mask)
        
            loss = loss_fn(outputs_start, outputs_end,
                           start_labels, end_labels)

            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)

            loss = loss / config.ACCUMULATION_STEPS   
            loss.backward()

            if (bi+1) % config.ACCUMULATION_STEPS    == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                scheduler.step()
                model.zero_grad()                           # Reset gradients tensors
                if config.SAVE_CHECKPOINT_TYPE == 'best_iter':
                    if step >= last_eval_step + eval_period or epoch*len(train_data_loader) + bi +1 == config.EPOCHS*len(train_data_loader):
                        val_score = eval_fn(valid_data_loader, model, device, epoch*len(train_data_loader) + bi, writer, df_valid, valid_dataset)                           
                        last_eval_step = step
                        for score, period in config.EVAL_SCHEDULE:
                            if val_score <= score:
                                eval_period = period
                                break                               
                
                        if not best_val_score or val_score > best_val_score:                    
                            best_val_score = val_score
                            best_epoch = epoch
                            torch.save(model.state_dict(), f'./{model_path_filename}')
                            print(f"New best_val_score: {best_val_score:0.4}")
                        else:       
                            print(f"Still best_val_score: {best_val_score:0.4}",
                                    f"(from epoch {best_epoch})")                                    
            step += 1

            del ids, mask, start_labels, end_labels

            torch.cuda.empty_cache()
            gc.collect()

        writer.add_scalar('Loss/train',losses.avg, (epoch+1)*len(train_data_loader))
        if config.SAVE_CHECKPOINT_TYPE == 'best_epoch' or config.SAVE_CHECKPOINT_TYPE == 'best_iter':
            val_score = eval_fn(valid_data_loader, model, device, (epoch+1)*len(train_data_loader), writer, df_valid, valid_dataset)
            if not best_val_score or val_score > best_val_score:                    
                best_val_score = val_score
                best_epoch = epoch
                torch.save(model.state_dict(), f'./{model_path_filename}')
                print(f"New best_val_score: {best_val_score:0.4}")
            else:       
                print(f"Still best_val_score: {best_val_score:0.4}",
                        f"(from epoch {best_epoch})") 
        if config.SAVE_CHECKPOINT_TYPE == 'last_epoch':
            val_score = eval_fn(valid_data_loader, model, device, (epoch+1)*len(train_data_loader), writer, df_valid, valid_dataset)
            print(f"val_score: {val_score:0.4}")
            torch.save(model.state_dict(), f'./{model_path_filename}')
        if not config.is_kaggle: #colab
            copyfile(f'./{model_path_filename}', model_path)
            print("Copied best checkpoint to google drive.")

        hns_features = get_hns_features(train_data_loader_for_hns, train_dataset_for_hns, valid_dataset, model, device)
    return best_val_score

# IN PROGRESS
def eval_fn(data_loader, model, device, iteration, writer, df_valid=None, valid_dataset=None):
    model.eval()
    losses = utils.AverageMeter()
    predicted_labels_start = []
    predicted_labels_end = []
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
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

            outputs_start, outputs_end = model(ids=ids, mask=mask)

            #Continue with outputs start/end
            loss = loss_fn(outputs_start, outputs_end,
                           start_labels, end_labels)
            outputs_start = outputs_start.cpu().detach()
            outputs_end = outputs_end.cpu().detach()

            losses.update(loss.item(), ids.size(0))

            predicted_labels_start.append(outputs_start)
            predicted_labels_end.append(outputs_end)
    
    # Raw predictions
    predicted_labels_start = torch.cat(
        tuple(x for x in predicted_labels_start), dim=0)
    predicted_labels_end = torch.cat(
        tuple(x for x in predicted_labels_end), dim=0)

    #Post process 
    #Baseline
    #predicted_labels_start = torch.softmax(predicted_labels_start, dim=-1).numpy()
    #predicted_labels_end = torch.softmax(predicted_labels_end, dim=-1).numpy()
    #predictions = utils.postprocess_qa_predictions(df_valid, valid_dataset.features, 
    #                                               (predicted_labels_start, predicted_labels_end))
    # Heatmap 
    predictions = utils.postprocess_heatmap(df_valid, valid_dataset.features, 
                                                   (predicted_labels_start, predicted_labels_end))  


    df_valid['PredictionString'] = df_valid['id'].map(predictions).apply(utils.postprocess)
    eval_score = df_valid.apply(lambda row: utils.jaccard(row['PredictionString'],row['answer_text']), axis=1).mean()

    
    writer.add_scalar('Loss/val', losses.avg, iteration)
    print(f'Val loss iter {iteration}= {losses.avg}')

    writer.add_scalar('Score/val', eval_score, iteration)
    print(f'Val Jaccard score iter {iteration}= {eval_score}')
    return eval_score

# IN PROGRESS
def get_hns_features(train_data_loader_for_hns, train_dataset_for_hns, valid_dataset, model, device):
    print("Creating hns features.....")
    model.eval()
    losses = utils.AverageMeter()
    true_labels_cls = []
    predicted_labels_cls = []
    with torch.no_grad():
        for bi, d in enumerate(train_data_loader_for_hns):
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

            outputs_start, outputs_end = model(ids=ids, mask=mask)

            # HNS features
            outputs_cls = 1 - (torch.softmax(outputs_start[:,:-1],dim=1)[:,0] + torch.softmax(outputs_end[:,:-1],dim=1)[:,0])/2

            if len(outputs_cls.size()) > 1:
                outputs_cls = outputs_cls.squeeze(-1)
            outputs_cls = outputs_cls.cpu().detach().numpy() # 0 - 1

            if len(classifier_labels.size()) > 1:
                classifier_labels = classifier_labels.squeeze(-1)
            classifier_labels = classifier_labels.cpu().detach().numpy()
            
            true_labels_cls.extend(classifier_labels.tolist())
            predicted_labels_cls.extend(outputs_cls.tolist())

    hns_features = train_dataset_for_hns.sampled_features
    for i,feature in enumerate(hns_features):
        feature['true_labels'] = true_labels_cls[i]
        feature['predicted_labels'] = predicted_labels_cls[i]
        hns_features[i] = feature

    return hns_features
