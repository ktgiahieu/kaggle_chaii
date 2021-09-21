import torch
import transformers

import config

class ChaiiModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(ChaiiModel, self).__init__(conf)
        self.automodel = transformers.AutoModelForQuestionAnswering.from_pretrained(
            config.MODEL_CONFIG,
            config=conf)

        #self.classifier = torch.nn.Sequential(
        #    torch.nn.Linear(conf.hidden_size*2, conf.hidden_size),
        #    torch.nn.GELU(),
        #    torch.nn.Linear(conf.hidden_size, 2),
        #)

        #self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)
        self.classifier = torch.nn.Linear(conf.hidden_size, 2)
        if isinstance(self.classifier, torch.nn.Linear):
            self.classifier.weight.data.normal_(mean=0.0, std=conf.initializer_range)
            if self.classifier.bias is not None:
                self.classifier.bias.data.zero_()
        #torch.nn.init.normal_(self.classifier.weight, std=0.02)
        

    def forward(self, ids, mask):
        out = self.automodel(ids, attention_mask=mask)

        start_logits, end_logits = out.start_logits, out.end_logits 

        return start_logits, end_logits
