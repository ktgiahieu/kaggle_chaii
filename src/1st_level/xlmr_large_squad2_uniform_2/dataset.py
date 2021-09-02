import numpy as np
import torch
import random

import config

import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"

def uniform_negative_sampling(features, num_positive):
    num_negative = len(features) - num_positive
    num_negative_preferred = num_positive * config.NEGATIVE_POSITIVE_RATIO
    negative_sampling_rate = num_negative_preferred / num_negative
    sampled_features = []

    current_document_features = []
    for i in range(len(features)):
        feature = features[i]
        if feature['classifier_labels'] == [1]:
            sampled_features.append(feature)
            continue
        if len(current_document_features)==0:
            current_document_features.append(feature)
            continue
        if current_document_features[0]['example_ids'] == feature['example_ids']:
            current_document_features.append(feature)
            continue

        prob = 1.0/(len(current_document_features) * num_positive)
        print(len(current_document_features))
        print(prob)
        for i, document_feature in enumerate(current_document_features):
            if random.random() < prob*config.NEGATIVE_POSITIVE_RATIO:
                sampled_features.append(document_feature)

        current_document_features = []
        current_document_features.append(feature)

    prob = 1.0/(len(current_document_features) * num_positive)
    for i, document_feature in enumerate(current_document_features):
        if random.random() < prob*config.NEGATIVE_POSITIVE_RATIO:
            sampled_features.append(document_feature)

    #for i in range(len(features)):
    #    feature = features[i]
    #    if feature['classifier_labels'] == [1] or random.random() < negative_sampling_rate:
    #        sampled_features.append(feature)
    print(f"num_positive: {num_positive}")
    print(f"len(sampled_features): {len(sampled_features)}")
    print(f"num_negative: {num_negative}")
    print(f"num_negative_preferred: {num_negative_preferred}")
    return sampled_features

def hard_negative_sampling(hns_features):
    sampled_features = []
    current_document_features = []
    for i in range(len(hns_features)):
        feature = hns_features[i]
        if feature['classifier_labels'] == [1]:
            sampled_features.append(feature)
            continue
        if len(current_document_features)==0:
            current_document_features.append(feature)
            continue
        if current_document_features[0]['example_ids'] == feature['example_ids']:
            current_document_features.append(feature)
            continue

        probs = [x['predicted_labels'] for x in current_document_features]
        norm_probs = [float(x)/sum(probs) for x in probs]
        for i, document_feature in enumerate(current_document_features):
            if random.random() < norm_probs[i]*config.NEGATIVE_POSITIVE_RATIO:
                sampled_features.append(document_feature)

        current_document_features = []
        current_document_features.append(feature)

    probs = [x['predicted_labels'] for x in current_document_features]
    norm_probs = [float(x)/sum(probs) for x in probs]
    for i, document_feature in enumerate(current_document_features):
        if random.random() < norm_probs[i]*config.NEGATIVE_POSITIVE_RATIO:
            sampled_features.append(document_feature)
    
    print(f"hns_features: {len(hns_features)}")
    print(f"sampled_features: {len(sampled_features)}")
    return sampled_features

def jaccard_array(a, b):
    """Calculates Jaccard on arrays."""
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def preprocess_data(tokenizer, ids, contexts, questions, answers, answer_starts):
    features = []
    for id, context, question, answer, answer_start in zip(ids, contexts, questions, answers, answer_starts):
        tokenized_example = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=config.MAX_LEN,
            stride=config.DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = tokenized_example.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_example["input_ids"][i]
            attention_mask = tokenized_example["attention_mask"][i]

            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_example.sequence_ids(i) #1 for answer, 0 for question, None for special tokens.

            if answer == '':
                feature = {'example_ids': id,
                       'ids': input_ids,
                       'mask': attention_mask,
                       'start_labels': [0],
                       'end_labels': [0],
                       'classifier_labels':[0],
                       'offsets': offsets,
                       'sequence_ids': sequence_ids}
                features.append(feature)
                continue

            start_char = answer_start
            end_char = answer_start + len(answer)

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_answer_start_index = token_start_index

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            token_answer_end_index = token_end_index

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                targets_start = cls_index
                targets_end = cls_index

                start_labels = [1] + [0]*(len(input_ids) - 1)
                end_labels = [1] + [0]*(len(input_ids) - 1)

                classifier_labels = 0
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                targets_start = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                targets_end = token_end_index + 1

                # Soft Jaccard labels
                # ----------------------------------
                n = len(input_ids)
                sentence_array = np.arange(n)
                answer_array = sentence_array[targets_start:targets_end + 1]

                start_labels = np.zeros(n)
                for i in range(token_answer_start_index, targets_end + 1):
                    jac = jaccard_array(answer_array, sentence_array[i:targets_end + 1])
                    start_labels[i] = jac + jac ** 2
                start_labels = (1 - config.SOFT_ALPHA) * start_labels / start_labels.sum()
                start_labels[targets_start] += config.SOFT_ALPHA

                end_labels = np.zeros(n)
                for i in range(targets_start, token_answer_end_index + 1):
                    jac = jaccard_array(answer_array, sentence_array[targets_start:i + 1])
                    end_labels[i] = jac + jac ** 2
                end_labels = (1 - config.SOFT_ALPHA) * end_labels / end_labels.sum()
                end_labels[targets_end] += config.SOFT_ALPHA

                start_labels = list(start_labels)
                end_labels = list(end_labels)

                classifier_labels = 1

            feature = {'example_ids': id,
                       'ids': input_ids,
                       'mask': attention_mask,
                       'offsets': offsets,
                       'start_labels': start_labels,
                       'end_labels': end_labels,
                       'classifier_labels':[classifier_labels],
                       'orig_answer': answer,
                       'sequence_ids': sequence_ids,}
            features.append(feature)
    return features


class ChaiiDataset:
    def __init__(self, ids, contexts, questions, answers, answer_starts, mode='train', hns_features=None):
        self.tokenizer = config.TOKENIZER
        self.mode = mode
        self.features = None
        self.sampled_features = None
        if hns_features is not None:
            self.features = hns_features
            self.sampled_features = hard_negative_sampling(self.features)
        else:
            self.features = preprocess_data(self.tokenizer, ids, contexts, questions, answers, answer_starts)
            if mode=='train':
                self.sampled_features = uniform_negative_sampling(self.features, len(ids))
        
    def __len__(self):
        return len(self.sampled_features) if self.mode == 'train' else len(self.features)

    def __getitem__(self, item):
        """Returns preprocessed data sample as dict with
        data converted to tensors.
        """
        if self.mode == 'train':
            data = self.sampled_features[item]

            return {'ids': torch.tensor(data['ids'], dtype=torch.long),
                    'mask': torch.tensor(data['mask'], dtype=torch.long),
                    'start_labels': torch.tensor(data['start_labels'],
                                                 dtype=torch.float),
                    'end_labels': torch.tensor(data['end_labels'],
                                               dtype=torch.float),
                    'classifier_labels':torch.tensor(data['classifier_labels'],
                                               dtype=torch.float),}
        else: #valid
            data = self.features[item]
            
            return {'ids': torch.tensor(data['ids'], dtype=torch.long),
                    'mask': torch.tensor(data['mask'], dtype=torch.long),
                    'start_labels': torch.tensor(data['start_labels'],
                                                 dtype=torch.float),
                    'end_labels': torch.tensor(data['end_labels'],
                                               dtype=torch.float),
                    'classifier_labels':torch.tensor(data['classifier_labels'],
                                               dtype=torch.float),}