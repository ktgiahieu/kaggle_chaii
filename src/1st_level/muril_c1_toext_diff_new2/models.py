import torch
import transformers

import config

class ChaiiModel(transformers.BertPreTrainedModel):
    def __init__(self, conf, fold):
        self.fold = fold
        super(ChaiiModel, self).__init__(conf)
        self.automodel = transformers.AutoModel.from_pretrained(
            config.MODEL_CONFIG,
            config=conf)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(conf.hidden_size*2, conf.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(conf.hidden_size, 2),
        )

        self.detect_answer_classifier = torch.nn.Sequential(
            torch.nn.Linear(conf.hidden_size, conf.hidden_size),
            torch.nn.tanh(),
            torch.nn.Linear(conf.hidden_size, 1),
        )

        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT[self.fold])

        if isinstance(self.classifier, torch.nn.Linear):
            self.classifier.weight.data.normal_(mean=0.0, std=conf.initializer_range)
            if self.classifier.bias is not None:
                self.classifier.bias.data.zero_()

        if isinstance(self.detect_answer_classifier, torch.nn.Linear):
            self.detect_answer_classifier.weight.data.normal_(mean=0.0, std=conf.initializer_range)
            if self.detect_answer_classifier.bias is not None:
                self.detect_answer_classifier.bias.data.zero_()

    def forward(self, ids, mask):
        out = self.automodel(ids, attention_mask=mask)

        #Detect answer classifier
        low_level_hidden_state = out.hidden_states[-6 -1][:,0,:]
        classifier_logits = self.detect_answer_classifier(low_level_hidden_state).squeeze(-1)

        # Mean-max pooler
        high_level_hidden_state = out.last_hidden_state
        combined_hidden_state = torch.cat([low_level_hidden_state, high_level_hidden_state], dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([
            self.classifier(self.high_dropout(combined_hidden_state))
            for _ in range(5)
        ], dim=0), dim=0)

        start_logits, end_logits = logits.split(1, dim=-1)

        # (batch_size, num_tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, classifier_logits
