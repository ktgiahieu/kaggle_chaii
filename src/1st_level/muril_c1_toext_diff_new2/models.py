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
            torch.nn.GELU()
        )

        self.head_start_end = torch.nn.Sequential(
            torch.nn.Linear(conf.hidden_size, 2),
        )

        self.head_variance = torch.nn.Sequential(
            torch.nn.Linear(conf.hidden_size, 1),
            torch.nn.Softplus()
        )

        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT[self.fold])

        if isinstance(self.classifier, torch.nn.Linear):
            self.classifier.weight.data.normal_(mean=0.0, std=conf.initializer_range)
            if self.classifier.bias is not None:
                self.classifier.bias.data.zero_()

        if isinstance(self.head_start_end, torch.nn.Linear):
            self.head_start_end.weight.data.normal_(mean=0.0, std=conf.initializer_range)
            if self.head_start_end.bias is not None:
                self.head_start_end.bias.data.zero_()

        if isinstance(self.head_variance, torch.nn.Linear):
            self.head_variance.weight.data.normal_(mean=0.0, std=conf.initializer_range)
            if self.head_variance.bias is not None:
                self.head_variance.bias.data.zero_()

    def forward(self, ids, mask):
        out = self.automodel(ids, attention_mask=mask)

        # Mean-max pooler
        out = out.hidden_states
        out = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN[self.fold])), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        pooled_last_hidden_states = torch.cat((out_mean, out_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        all_logits = []
        for _ in range(5):
            out_n = self.classifier(self.high_dropout(pooled_last_hidden_states))
            start_end_n = self.head_start_end(out_n)
            var_n = self.head_variance(out_n)
            all_logits.append(torch.cat([start_end_n, var_n], dim=-1))
        logits = torch.mean(torch.stack(
            all_logits, dim=0), dim=0)

        start_logits, end_logits, variance = logits.split(1, dim=-1)

        # (batch_size, num_tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        variance = variance.squeeze(-1)

        return start_logits, end_logits, variance
