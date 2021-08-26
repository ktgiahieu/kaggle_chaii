from shutil import copyfile
from sklearn.metrics import confusion_matrix
import random

import numpy as np
import torch
import tqdm
import gc

import config
import utils


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


def train_fn(train_data_loader, valid_data_loader, model, optimizer, device, writer, model_path, scheduler=None):  
    model_path_filename = model_path.split('/')[-1]
    best_val_score = None
    step = 0
    last_eval_step = 0
    eval_period = config.EVAL_SCHEDULE[0][1]   
    for epoch in range(config.EPOCHS):
        losses = utils.AverageMeter()
        tk0 = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
        model.zero_grad()
        for bi, d in enumerate(tk0):
            torch.cuda.empty_cache()
            gc.collect()
            if random.random() > d['sampling_rate']:
                continue

            ids = d['ids']
            mask = d['mask']
            classifier_labels = d['classifier_labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            classifier_labels = classifier_labels.to(device, dtype=torch.float)

            model.train()
            
            outputs = model(ids=ids, mask=mask)
        
            loss = classifier_loss_fn(outputs, classifier_labels)

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
                        val_score = eval_fn(valid_data_loader, model, device, epoch*len(train_data_loader) + bi, writer)                           
                        last_eval_step = step
                        for score, period in config.EVAL_SCHEDULE:
                            if val_score <= score:
                                eval_period = period
                                break                               
                
                        if not best_val_score or val_score > best_val_score:                    
                            best_val_score = val_score
                            best_epoch = epoch
                            torch.save(model.state_dict(), f'/content/{model_path_filename}')
                            print(f"New best_val_score: {best_val_score:0.4}")
                        else:       
                            print(f"Still best_val_score: {best_val_score:0.4}",
                                    f"(from epoch {best_epoch})")                                    
            step += 1

        writer.add_scalar('Loss/train',losses.avg, (epoch+1)*len(train_data_loader))
        if config.SAVE_CHECKPOINT_TYPE == 'best_epoch':
            val_score = eval_fn(valid_data_loader, model, device, (epoch+1)*len(train_data_loader), writer)
            if not best_val_score or val_score > best_val_score:                    
                best_val_score = val_score
                best_epoch = epoch
                torch.save(model.state_dict(), f'/content/{model_path_filename}')
                print(f"New best_val_score: {best_val_score:0.4}")
            else:       
                print(f"Still best_val_score: {best_val_score:0.4}",
                        f"(from epoch {best_epoch})") 
        if config.SAVE_CHECKPOINT_TYPE == 'last_epoch':
            torch.save(model.state_dict(), f'/content/{model_path_filename}')
        copyfile(f'/content/{model_path_filename}', model_path)
        print("Copied best checkpoint to google drive.")
    return best_val_score

# IN PROGRESS
def eval_fn(data_loader, model, device, iteration, writer):
    model.eval()
    losses = utils.AverageMeter()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    m = torch.nn.Sigmoid()

    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            ids = d['ids']
            mask = d['mask']
            classifier_labels = d['classifier_labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            classifier_labels = classifier_labels.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
        
            loss = classifier_loss_fn(outputs, classifier_labels)

            outputs = (m(outputs.squeeze(-1)) > config.CLASSIFIER_THRESHOLD).cpu().detach().numpy() # 0 or 1
            tn, fp, fn, tp = confusion_matrix(classifier_labels.squeeze(-1).cpu().detach().numpy(), 
                                              outputs, labels=[0, 1]).ravel()
            TP += tp
            TN += tn
            FP += fp
            FN += fn

            losses.update(loss.item(), ids.size(0))
    
    writer.add_scalar('Loss/val', losses.avg, iteration)
    print(f'Val loss iter {iteration}= {losses.avg}')
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print(f'Val accuracy {iteration}= {accuracy}')
    recall = TP/(TP+FN)
    print(f'Val recall {iteration}= {recall}')
    return recall
