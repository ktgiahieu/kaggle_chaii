import torch
import transformers

import config

class ChaiiModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(ChaiiModel, self).__init__(conf)
        self.automodel = transformers.AutoModel.from_pretrained(
            config.MODEL_CONFIG,
            config=conf)

        #self.classifier = torch.nn.Sequential(
        #    torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
        #    torch.nn.Linear(config.HIDDEN_SIZE*3, config.HIDDEN_SIZE*2),
        #    torch.nn.GELU(),
        #    torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
        #    torch.nn.Linear(config.HIDDEN_SIZE*2, 1),
        #)

        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)
        self.classifier = torch.nn.Linear(config.HIDDEN_SIZE, 2)

        torch.nn.init.normal_(self.classifier.weight, std=conf.initializer_range)

    def forward(self, ids, mask):
        out = self.automodel(ids, attention_mask=mask)

        ## Mean-max pooler
        #out = out.hidden_states
        #out = torch.stack(
        #    tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        #out_mean = torch.mean(out, dim=0)
        #out_max, _ = torch.max(out, dim=0)
        #pooled_last_hidden_states = torch.cat((out_mean, out_max), dim=-1)

        ## Multisample Dropout: https://arxiv.org/abs/1905.09788
        #logits = torch.mean(torch.stack([
        #    self.classifier(self.high_dropout(pooled_last_hidden_states))
        #    for _ in range(5)
        #], dim=0), dim=0)
        out = out.last_hidden_state

        self.classifier(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        # (batch_size, num_tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
