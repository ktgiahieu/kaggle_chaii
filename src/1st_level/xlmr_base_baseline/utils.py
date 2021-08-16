import os
import random
import re

import torch
import numpy as np

import config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_optimizer(model):
    num_layers = 6   #distil
    if config.model_type.split('-')[-1] == 'base':
        num_layers = 12
    elif config.model_type == 'funnel-transformers-large':
        num_layers = 26
    elif config.model_type.split('-')[-1]=='large' or config.model_type == 'deberta-v2-xlarge':
        num_layers = 24
    elif  config.model_type == 'deberta-xlarge' or config.model_type == 'deberta-v2-xxlarge':
        num_layers = 48
    

    named_parameters = list(model.named_parameters()) 
    automodel_parameters = list(model.automodel.named_parameters())
    head_parameters = named_parameters[len(automodel_parameters):]
        
    head_group = [params for (name, params) in head_parameters]

    parameters = []
    parameters.append({"params": head_group, "lr": config.HEAD_LEARNING_RATE})

    last_lr = config.LEARNING_RATES_RANGE[0]
    for name, params in automodel_parameters:
        weight_decay = 0.0 if "bias" in name else config.WEIGHT_DECAY
        lr = None
        layer_num = None

        if config.model_type.split('-')[0] == 'bart':
            found_layer_num_encoder = re.search('(?<=encoder\.layer).*', name)
            if found_layer_num_encoder:
                found_a_number = re.search('(?<=\.)\d+(?=\.)',found_layer_num_encoder.group(0))#fix encoder.layernorm.weight bug
                if found_a_number:
                    layer_num = int(found_a_number.group(0))
            else:
                found_layer_num_decoder = re.search('(?<=decoder\.layer).*', name)
                if found_layer_num_decoder:
                    found_a_number = re.search('(?<=\.)\d+(?=\.)',found_layer_num_decoder.group(0))#fix encoder.layernorm.weight bug
                    if found_a_number:
                        if num_layers == 12:
                            layer_num = int(found_a_number.group(0)) + 6
                        elif num_layers == 24:
                            layer_num = int(found_a_number.group(0)) + 12
                        else:
                            raise ValueError("Bart model has insufficient num_layers: %d (must be 12 or 24)" % num_layers)
        elif config.model_type == 'funnel-transformers-large':
            found_block_encoder = re.search('(?<=encoder\.blocks).*', name)
            if found_block_encoder:
                block_num, subblock_num = tuple([int(x) for x in re.findall('(?<=\.)\d+(?=\.)',found_block_encoder.group(0))])
                layer_num = block_num*8 + subblock_num
            else:
                found_layer_decoder = re.search('(?<=decoder\.layers).*', name)
                if found_layer_decoder:
                    layer_num = 24 + int(re.search('(?<=\.)\d+(?=\.)',found_layer_decoder.group(0)).group(0))
        else:
            found_layer_num = re.search('(?<=encoder\.layer).*', name)
            if found_layer_num:
                layer_num = int(re.search('(?<=\.)\d+(?=\.)',found_layer_num.group(0)).group(0))

        if layer_num is None:
            lr = last_lr
        else:
            if config.LEARNING_RATE_LAYERWISE_TYPE == 'linear':
                lr = config.LEARNING_RATES_RANGE[0] + (layer_num+1) * (config.LEARNING_RATES_RANGE[1] - config.LEARNING_RATES_RANGE[0])/num_layers
            elif config.LEARNING_RATE_LAYERWISE_TYPE == 'exponential':
                lr = config.LEARNING_RATES_RANGE[0] * (config.LEARNING_RATES_RANGE[1]/config.LEARNING_RATES_RANGE[0])**((layer_num+1)/num_layers)
            else:
                raise ValueError("config.LEARNING_RATE_LAYERWISE_TYPE must be 'linear' or 'exponential'")
        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})
        last_lr = lr
    return torch.optim.AdamW(parameters, eps=1e-06)