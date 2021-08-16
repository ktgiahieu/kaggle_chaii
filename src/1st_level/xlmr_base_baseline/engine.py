from shutil import copyfile

import numpy as np
import torch
import tqdm
import gc

import config
import utils


def loss_fn(outputs, labels):
    loss_fct = torch.nn.MSELoss()
    return loss_fct(outputs, labels)


def train_fn(train_data_loader, valid_data_loader, model, optimizer, device, writer, model_path, scheduler=None):
    model_path_filename = model_path.split('/')[-1]
    best_val_rmse = None
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
            ids = d['ids']
            mask = d['mask']
            labels = d['labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            model.train()
            
            outputs = \
                model(ids=ids, mask=mask)
        
            loss = loss_fn(outputs, labels)

            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=np.sqrt(losses.avg))

            loss = loss / config.ACCUMULATION_STEPS   
            loss.backward()



            if (bi+1) % config.ACCUMULATION_STEPS    == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                scheduler.step()
                model.zero_grad()                           # Reset gradients tensors
                if step >= last_eval_step + eval_period:
                    val_rmse = eval_fn(valid_data_loader, model, device, epoch*len(train_data_loader) + bi, writer)                           
                    last_eval_step = step
                    for rmse, period in config.EVAL_SCHEDULE:
                        if val_rmse >= rmse:
                            eval_period = period
                            break                               
                
                    if not best_val_rmse or val_rmse < best_val_rmse:                    
                        best_val_rmse = val_rmse
                        best_epoch = epoch
                        torch.save(model.state_dict(), f'/content/{model_path_filename}')
                        print(f"New best_val_rmse: {best_val_rmse:0.4}")
                    else:       
                        print(f"Still best_val_rmse: {best_val_rmse:0.4}",
                                f"(from epoch {best_epoch})")                                    
            step += 1

        writer.add_scalar('Loss/train', np.sqrt(losses.avg), (epoch+1)*len(train_data_loader))
        copyfile(f'/content/{model_path_filename}', model_path)
        print("Copied best checkpoint to google drive.")

        rmse_score = eval_fn(valid_data_loader, model, device, (epoch+1)*len(train_data_loader), writer)
    return rmse_score

def eval_fn(data_loader, model, device, iteration, writer):
    model.eval()
    losses = utils.AverageMeter()

    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            ids = d['ids']
            mask = d['mask']
            labels = d['labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            outputs = \
                model(ids=ids, mask=mask)
            loss = loss_fn(outputs, labels)

            losses.update(loss.item(), ids.size(0))
    
    writer.add_scalar('Loss/val', np.sqrt(losses.avg), iteration)
    print(f'RMSE iter {iteration}= {np.sqrt(losses.avg)}')
    return np.sqrt(losses.avg)
