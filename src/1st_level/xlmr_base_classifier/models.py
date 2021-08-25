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
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE*6, config.HIDDEN_SIZE*4),
            torch.nn.GELU(),
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE*4, 1),
        )

    def forward(self, ids, mask):
        out = self.automodel(ids, attention_mask=mask)

        # Vector 0 pooler
        out = out.hidden_states
        out = torch.cat(
            tuple(out[-i - 1][:,0,:] for i in range(config.N_LAST_HIDDEN)), dim=-1)
        out = out.squeeze(1)

        logits = classifier(pooled_last_hidden_states)
        return logits
