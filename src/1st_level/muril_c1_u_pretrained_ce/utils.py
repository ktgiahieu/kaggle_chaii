import os
import random
import re
import collections

import torch
import numpy as np
from scipy.special import softmax

import config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def postprocess(pred):
    pred = " ".join(pred.split())
    pred = pred.strip(punctuation)

    bad_starts = [".", ",", "(", ")", "-", "–",  ",", ";"]
    bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

    if pred == "":
        return pred
    while any([pred.startswith(y) for y in bad_starts]):
        pred = pred[1:]
    while any([pred.endswith(y) for y in bad_endings]):
        if pred.endswith("..."):
            pred = pred[:-3]
        else:
            pred = pred[:-1]

    return pred

def reinit_last_layers(model, reinit_layers=4):
    if reinit_layers > 0:
        print(f'Reinitializing Last {reinit_layers} Layers ...')
        encoder_temp = getattr(model, 'automodel')
        for layer in encoder_temp.encoder.layer[-reinit_layers:]:
            for module in layer.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=config.CONF.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, torch.nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=config.CONF.initializer_range)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, torch.nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
        encoder_temp = getattr(model, 'classifier')
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.CONF.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        print('Done reinitializing.!')
    return model

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_ids"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []
        
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            features[feature_index]["offsets"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offsets"])
            ]
            offsets = features[feature_index]["offsets"]
            cls_index = features[feature_index]["ids"].index(config.TOKENIZER.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offsets)
                        or end_index >= len(offsets)
                        or offsets[start_index] is None
                        or offsets[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]
        
        
    return predictions

def postprocess_heatmap(examples, features, raw_predictions, n_best_size = 20, max_answer_char_length = 50):
    all_start_logits, all_end_logits = raw_predictions
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_ids"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []
        
        context = example["context"]

        answer_start_sum_logits = np.zeros(len(context), dtype=np.float)
        answer_end_sum_logits = np.zeros(len(context), dtype=np.float)

        answer_start_num_logits = np.zeros(len(context), dtype=np.float)
        answer_end_num_logits = np.zeros(len(context), dtype=np.float)

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            features[feature_index]["offsets"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offsets"])
            ]
            offsets = features[feature_index]["offsets"]
            cls_index = features[feature_index]["ids"].index(config.TOKENIZER.cls_token_id)

            answer_start_sum_current_logit = np.zeros(len(context), dtype=np.float)
            answer_end_sum_current_logit = np.zeros(len(context), dtype=np.float)
            answer_start_num_current_logit = np.zeros(len(context), dtype=np.float)
            answer_end_num_current_logit = np.zeros(len(context), dtype=np.float)

            for start_index in range(len(start_logits)):
                if offsets[start_index] is None:
                    continue
                start_char = offsets[start_index][0]

                answer_start_sum_current_logit[start_char] = start_logits[start_index]
                answer_start_num_current_logit[start_char] = 1

            for end_index in range(len(end_logits)):
                if offsets[end_index] is None:
                    continue
                end_char = offsets[end_index][1] - 1

                answer_end_sum_current_logit[end_char] = end_logits[end_index]
                answer_end_num_current_logit[end_char] = 1
            
            answer_start_sum_logits = answer_start_sum_logits + answer_start_sum_current_logit
            answer_end_sum_logits = answer_end_sum_logits + answer_end_sum_current_logit
            answer_start_num_logits = answer_start_num_logits + answer_start_num_current_logit
            answer_end_num_logits = answer_end_num_logits + answer_end_num_current_logit


        answer_start_sum_logits[answer_start_num_logits==0] = -np.inf         
        answer_end_sum_logits[answer_end_num_logits==0] = -np.inf
        answer_start_num_logits[answer_start_num_logits==0] = 1
        answer_end_num_logits[answer_end_num_logits==0] = 1

        answer_start_sum_logits = answer_start_sum_logits / answer_start_num_logits
        answer_end_sum_logits = answer_end_sum_logits / answer_end_num_logits

        best_start_chars = np.argsort(answer_start_sum_logits)[-1 : -n_best_size - 1 : -1].tolist()
        best_end_chars = np.argsort(answer_end_sum_logits)[-1 : -n_best_size - 1 : -1].tolist()
        for start_char in best_start_chars:
            for end_char in best_end_chars:
                # Don't consider answers with a length that is either < 0 or > max_answer_char_length.
                if end_char <= start_char or end_char - start_char + 1 > max_answer_char_length:
                    continue

                valid_answers.append(
                    {
                        "score": answer_start_sum_logits[start_char] + answer_end_sum_logits[end_char],
                        "text": context[start_char: end_char+1]
                    }
                )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]
        
        
    return predictions

def postprocess_heatmap_logit(examples, features, raw_predictions, n_best_size = 20, max_answer_char_length = 50):
    all_start_logits, all_end_logits = raw_predictions
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_ids"]]].append(i)

    heatmap_logit = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []
        
        context = example["context"]

        answer_start_sum_logits = np.zeros(len(context), dtype=np.float)
        answer_end_sum_logits = np.zeros(len(context), dtype=np.float)

        answer_start_num_logits = np.zeros(len(context), dtype=np.float)
        answer_end_num_logits = np.zeros(len(context), dtype=np.float)

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            features[feature_index]["offsets"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offsets"])
            ]
            offsets = features[feature_index]["offsets"]
            cls_index = features[feature_index]["ids"].index(config.TOKENIZER.cls_token_id)

            answer_start_sum_current_logit = np.zeros(len(context), dtype=np.float)
            answer_end_sum_current_logit = np.zeros(len(context), dtype=np.float)
            answer_start_num_current_logit = np.zeros(len(context), dtype=np.float)
            answer_end_num_current_logit = np.zeros(len(context), dtype=np.float)

            for start_index in range(len(start_logits)):
                if offsets[start_index] is None:
                    continue
                start_char = offsets[start_index][0]

                answer_start_sum_current_logit[start_char] = start_logits[start_index]
                answer_start_num_current_logit[start_char] = 1

            for end_index in range(len(end_logits)):
                if offsets[end_index] is None:
                    continue
                end_char = offsets[end_index][1] - 1

                answer_end_sum_current_logit[end_char] = end_logits[end_index]
                answer_end_num_current_logit[end_char] = 1
            
            answer_start_sum_logits = answer_start_sum_logits + answer_start_sum_current_logit
            answer_end_sum_logits = answer_end_sum_logits + answer_end_sum_current_logit
            answer_start_num_logits = answer_start_num_logits + answer_start_num_current_logit
            answer_end_num_logits = answer_end_num_logits + answer_end_num_current_logit


        answer_start_sum_logits[answer_start_num_logits==0] = -np.inf         
        answer_end_sum_logits[answer_end_num_logits==0] = -np.inf
        answer_start_num_logits[answer_start_num_logits==0] = 1
        answer_end_num_logits[answer_end_num_logits==0] = 1

        answer_start_sum_logits = answer_start_sum_logits / answer_start_num_logits
        answer_end_sum_logits = answer_end_sum_logits / answer_end_num_logits

        answer_start_sum_logits = softmax(answer_start_sum_logits)
        answer_end_sum_logits = softmax(answer_end_sum_logits)

        heatmap_logit[example["id"]] = (answer_start_sum_logits, answer_end_sum_logits)

    return heatmap_logit

def token_level_to_char_level(text, offsets, preds):
    probas_char = np.zeros(len(text))
    for i, offset in enumerate(offsets):
        if offset[0] or offset[1]:
            probas_char[offset[0]:offset[1]] = preds[i]

    return probas_char


def jaccard(str1, str2):
    """Original metric implementation."""
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_best_start_end_idx(start_logits, end_logits,
                           orig_start, orig_end):
    """Return best start and end indices following BERT paper."""
    best_logit = -np.inf
    best_idxs = None
    start_logits = start_logits[orig_start:orig_end + 1]
    end_logits = end_logits[orig_start:orig_end + 1]
    for start_idx, start_logit in enumerate(start_logits):
        for end_idx, end_logit in enumerate(end_logits[start_idx:]):
            logit_sum = start_logit + end_logit
            if logit_sum > best_logit:
                best_logit = logit_sum
                best_idxs = (orig_start + start_idx,
                             orig_start + start_idx + end_idx)
    return best_idxs


def calculate_jaccard(original_tweet, target_string,
                      start_logits, end_logits,
                      orig_start, orig_end,
                      offsets, 
                      verbose=False):
    """Calculates final Jaccard score using predictions."""
    start_idx, end_idx = get_best_start_end_idx(
        start_logits, end_logits, orig_start, orig_end)

    filtered_output = ''
    for ix in range(start_idx, end_idx + 1):
        filtered_output += original_tweet[offsets[ix][0]:offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += ' '

    # Return orig tweet if it has less then 2 words
    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    if len(filtered_output.split()) == 1:
        filtered_output = filtered_output.replace('!!!!', '!')
        filtered_output = filtered_output.replace('..', '.')
        filtered_output = filtered_output.replace('...', '.')

    filtered_output = filtered_output.replace('ïï', 'ï')
    filtered_output = filtered_output.replace('¿¿', '¿')

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output

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
    num_layers = config.CONF.num_hidden_layers

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

        if 'bart' in re.split('/|-', config.MODEL_CONFIG):
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
        elif 'funnel' in re.split('/|-', config.MODEL_CONFIG) \
            and 'transformer' in re.split('/|-', config.MODEL_CONFIG):
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