from transformers import BertConfig, BertModel, BertPreTrainedModel
import torch
import json
from torch import nn
from typing import List, Optional, Tuple, Union


class BertModelCustom(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = self.get_embeddings()
        self.encoder = self.get_encoder()
        self.pooler = self.get_pooler()

        for name, param in self.named_parameters():
            if name.startswith("embeddings") and "layernorm" not in name.lower() and "word_embeddings" not in name.lower():
                param.detach_()
                param.requires_grad = False
                param.data.fill_(0.0)

    def get_embeddings(self):
        return self.embeddings

    def get_encoder(self):
        return self.encoder

    def get_pooler(self):
        return self.pooler

    def forward(
        self,
        inputs_embeds,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return super().forward(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class BertForContrastiveLearning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModelCustom(config)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.mlp(pooled_output)

        return logits

    def encode(self, inputs_embeds: torch.Tensor, normalize=True, output_tensor=False) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            embeddings = self.bert(inputs_embeds=inputs_embeds)[1]
            embeddings = embeddings.detach()
            if normalize:
                embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        self.train()
        if output_tensor:
            return embeddings
        else:
            return embeddings.tolist()
    
    def get_device(self):
        return self.bert.device
