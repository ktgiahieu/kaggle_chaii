import torch
import transformers

import config

class ChaiiClassifierModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(ChaiiClassifierModel, self).__init__(conf)
        self.automodel = transformers.AutoModel.from_pretrained(
            config.MODEL_CONFIG,
            config=conf)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(conf.hidden_size*2, conf.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(conf.hidden_size, 1),
        )

        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)

    def forward(self, ids, mask):
        out = self.automodel(ids, attention_mask=mask)

        # Mean-max pooler
        out = out.hidden_states
        out = torch.stack(
            tuple(out[-i - 1][:,0,:] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        pooled_last_hidden_states = torch.cat((out_mean, out_max), dim=-1).squeeze(1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([
            self.classifier(self.high_dropout(pooled_last_hidden_states))
            for _ in range(5)
        ], dim=0), dim=0)

        return logits
