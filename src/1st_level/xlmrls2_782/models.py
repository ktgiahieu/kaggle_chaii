import torch
import torch.nn as nn
import transformers

class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = transformers.AutoModel.from_pretrained(modelname_or_path, config=config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights(self.qa_outputs)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self, 
        ids, 
        mask=None, 
    ):
        outputs = self.xlm_roberta(
            ids,
            attention_mask=mask,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        
        # sequence_output = self.dropout(sequence_output)
        qa_logits = self.qa_outputs(sequence_output)
        
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        return start_logits, end_logits
