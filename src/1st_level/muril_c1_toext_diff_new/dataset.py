import numpy as np
import torch
import random
from indicnlp.tokenize import sentence_tokenize

import config

import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"

def uniform_negative_sampling(features, num_positive):
    num_negative = len(features) - num_positive
    num_negative_preferred = num_positive * config.NEGATIVE_POSITIVE_RATIO
    negative_sampling_rate = num_negative_preferred / num_negative
    sampled_features = []
    for i in range(len(features)):
        feature = features[i]
        if feature['classifier_labels'] == [1] or random.random() < negative_sampling_rate:
            sampled_features.append(feature)
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

def preprocess_data(tokenizer, ids, orig_contexts, orig_questions, orig_answers, orig_answer_starts, languages, fold):
    features = []
    for id, orig_context, orig_question, orig_answer, orig_answer_start, language in zip(ids, orig_contexts, orig_questions, orig_answers, orig_answer_starts, languages):
        orig_question = orig_question.strip()

        all_aug_contexts = [orig_context]
        all_aug_questions = [orig_question]
        all_aug_answers = [orig_answer]
        all_aug_answer_starts = [orig_answer_start]


        if random.random() < 1.0:
            # Split context to sentences
            sentences_raw=sentence_tokenize.sentence_split(orig_context, lang='hi' if language=='hindi' else 'ta')
            sentences = []
            for sent in sentences_raw:
                if sent[0] == ')':
                    sentences[-1] = sentences[-1] + sent
                elif sent == ' P. B) என்ற முன்னெழுத்துகளால் பரவலாக அறியப்படுகிறார்.':
                    sentences[-1] = sentences[-1] + sent
                elif sent == 'அமர்நாத் ராமகிருஷ்ணன் கூறியுள்ளார்.':
                    sentences[-1] = sentences[-1] + ' ' + sent
                elif sent == 'கிமீ (573 ச. மைல்) பரப்பளவு கொண்டது.':
                    sentences.append('[2] புவியியலும் தட்பவெப்பநிலையும் தில்லி தேசிய தலைநகரப் பகுதி, 1,484 ச.கிமீ (573 ச.மைல்) பரப்பளவு கொண்டது.')

                elif sent == 'கிமீ (270 ச. மைல்) பகுதி நகர்ப்புறப் பகுதியாகவும் உள்ளது.':
                    sentences.append('இதில் 783 ச.கிமீ (302 ச.மைல்) பரப்பளவு கொண்ட பகுதி நாட்டுப்புறப் பகுதியாகவும், 700 ச.கிமீ (270 ச.மைல்) பகுதி நகர்ப்புறப் பகுதியாகவும் உள்ளது.')


                elif sent == 'கிமீ (3,011 ச. மைல்) நீர்ப் பரப்பும் ஆகும்.':
                    sentences.append('357,021 ச.கிமீ (137,847 ச.மைல்) பரப்பளவு கொண்ட இந்நாட்டில் 349,223 ச.கிமீ (134,836 ச.மைல்) நிலப் பரப்பும், 7,798 ச.கிமீ (3,011 ச.மைல்) நீர்ப் பரப்பும் ஆகும்.')
                elif sent == '[1] ஐரோப்பா கண்டமானது, 10,180,000 ச. கி;மீகள் பரப்பளவைக் கொண்டது.':
                    sentences.append('[1] ஐரோப்பா கண்டமானது, 10,180,000 ச.கி;மீகள் பரப்பளவைக் கொண்டது.')
                elif sent == 'இன்றைய பாகிஸ்தானிலுள்ள சிந்து நதியை அண்டித் தழைத்தோங்கியிருந்த இந்த நாகரிகம் மிகப் பரந்ததொரு பிரதேசத்தில் செல்வாக்குச் செலுத்திவந்தது. கி. மு 3000 க்கும் கி.':
                    sentences.append('இன்றைய பாகிஸ்தானிலுள்ள சிந்து நதியை அண்டித் தழைத்தோங்கியிருந்த இந்த நாகரிகம் மிகப் பரந்ததொரு பிரதேசத்தில் செல்வாக்குச் செலுத்திவந்தது. கி.மு 3000 க்கும் கி.')
                elif sent == 'வி இசையில் எல். ஆர். ஈஸ்வரியோடு இணைந்து அத்தானோடு இப்படியிருந்து ௭த்தனை நாளாச்சு ௭ன்ற பாடலைப் பாடினார்.':
                    sentences.append('[7][8][9][10][11] இவர் தமிழில் முதலில் பாடியது ஹோட்டல் ரம்பா திரைப்படத்தில் மெல்லிசை மன்னர்௭ம்.௭ஸ்.')
                    sentences.append('வி இசையில் எல். ஆர். ஈஸ்வரியோடு இணைந்து அத்தானோடு இப்படியிருந்து ௭த்தனை நாளாச்சு ௭ன்ற பாடலைப் பாடினார்.')
                elif sent == 'மீ நீளம் கொண்ட இது, தான்சானியா, உகாண்டா, ருவாண்டா, புருண்டி, காங்கோ, கென்யா, எத்தியோப்பியா, எரித்திரியா, தெற்கு சூடான், சூடான், எகிப்து ஆகிய பதினோரு நாடுகளின் வழியாகப் பாய்ந்து நடுநிலக் கடலில் கலக்கின்றது[2].':
                    sentences[-1] = sentences[-1] + sent
                else:
                    sentences.append(sent)
            # Find answer_start in sentences
            total_len = 0
            answer_sen = 0
            answer_sen_start = 0
            for i_sen, sen in enumerate(sentences):
              len_sen = len(sen)
              if orig_answer_start < total_len + len_sen:
                answer_sen = i_sen
                answer_sen_start = orig_answer_start - total_len
                break
              else:
                total_len+= len_sen
                if orig_context[total_len] == ' ':
                    total_len+=1

            if sentences[answer_sen][answer_sen_start:answer_sen_start+len(orig_answer)]!=orig_answer:
                for plus in range(-20, 20):
                    if sentences[answer_sen][answer_sen_start+plus:answer_sen_start+plus+len(orig_answer)]==orig_answer:
                        answer_sen_start+=plus
                        break

            print(sentences[answer_sen])
            print(sentences[answer_sen][answer_sen_start:answer_sen_start+len(orig_answer)])
            print(orig_answer)
            print(id)

            assert(sentences[answer_sen][answer_sen_start:answer_sen_start+len(orig_answer)]==orig_answer)

            # Shuffle neighbor sentences
            max_seq_len = len(sentences)
            idx = list(enumerate(sentences))
            ws = np.random.choice([2, 4, 5])
            num = max_seq_len // ws

            for i in range(num):
                sub_list = idx[i * ws:(i + 1) * ws]
                random.shuffle(sub_list)
                idx[i * ws:(i + 1) * ws] = sub_list

            indices, sentences_shuffled = zip(*idx)

            # Find new answer_start after shuffling
            new_answer_start = 0
            total_len = 0
            for i_sen, sen in zip(indices, sentences_shuffled):
              len_sen = len(sen)
              if i_sen == answer_sen:
                new_answer_start = total_len + answer_sen_start
              else:
                total_len += len_sen+1

            new_context = ' '.join(sentences_shuffled)
            assert(new_context[new_answer_start:new_answer_start+len(orig_answer)]==orig_answer)

            all_aug_contexts.append(new_context)
            all_aug_questions.append(orig_question)
            all_aug_answers.append(orig_answer)
            all_aug_answer_starts.append(new_answer_start)

        for context, question, answer, answer_start in zip(all_aug_contexts, all_aug_questions, all_aug_answers, all_aug_answer_starts):
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
                    start_labels = (1 - config.SOFT_ALPHA[fold]) * start_labels / start_labels.sum()
                    start_labels[targets_start] += config.SOFT_ALPHA[fold]

                    end_labels = np.zeros(n)
                    for i in range(targets_start, token_answer_end_index + 1):
                        jac = jaccard_array(answer_array, sentence_array[targets_start:i + 1])
                        end_labels[i] = jac + jac ** 2
                    end_labels = (1 - config.SOFT_ALPHA[fold]) * end_labels / end_labels.sum()
                    end_labels[targets_end] += config.SOFT_ALPHA[fold]

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
    def __init__(self, fold, ids, contexts, questions, answers, answer_starts, languages, mode='train', hns_features=None):
        self.fold = fold
        self.tokenizer = config.TOKENIZER
        self.mode = mode
        self.features = None
        self.sampled_features = None
        if hns_features is not None:
            self.features = hns_features
            self.sampled_features = hard_negative_sampling(self.features)
        else:
            self.features = preprocess_data(self.tokenizer, ids, contexts, questions, answers, answer_starts, languages, fold)
            if mode=='train':
                #self.sampled_features = uniform_negative_sampling(self.features, len(ids))
                self.sampled_features = self.features
        
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