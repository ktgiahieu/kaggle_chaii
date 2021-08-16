import numpy as np
import torch

import config

def process_data(text, label,
                 tokenizer, max_len):
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
    encoded_dict = tokenizer.encode_plus(
        text,                      # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = config.MAX_LEN,           # Pad & truncate all sentences.
        padding = 'max_length',
		return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
        truncation = True,
    )
    # ----------------------------------

    # Input for BERT
    input_ids = np.squeeze(encoded_dict['input_ids'],0)
    # Mask of input without padding
    mask = np.squeeze(encoded_dict['attention_mask'],0)

    return {'ids': input_ids,
            'mask': mask,
            'labels': [label]}


class CommonlitDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """Returns preprocessed data sample as dict with
        data converted to tensors.
        """
        data = process_data(self.texts[item],
                            self.labels[item],
                            self.tokenizer,
                            self.max_len)

        return {'ids': torch.tensor(data['ids'], dtype=torch.long),
                'mask': torch.tensor(data['mask'], dtype=torch.long),
                'labels': torch.tensor(data['labels'], dtype=torch.float),}
