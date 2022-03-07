# -*- coding: utf-8 -*-
"""
@author: bdchen

"""

import torch
from torch import nn
import transformers
from transformers import BertPreTrainedModel
from transformers.modeling_bert import BertLMPredictionHead
from transformers import (
  WEIGHTS_NAME,
  BertModel,
  BertConfig,
  BertTokenizer,
)


class My_Model_f2l(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
     
        self.bert = BertModel(config)

        
        self.MLP_2 = nn.Sequential(nn.Linear(2*config.hidden_size,2*config.hidden_size),nn.ReLU(),nn.Linear(2*config.hidden_size,2*config.hidden_size))

        self.lm_head = BertLMPredictionHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mlm_input_ids=None,
        mlm_labels=None,
        token_pos=None,
    ):
        # for MLM train
        mlm_outputs = None
        if mlm_input_ids is not None:
            #mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_outputs = self.bert(
                mlm_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds
            )
        if mlm_outputs is not None and mlm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            #mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            prediction_scores = self.lm_head(mlm_outputs[0])
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        else:
            masked_lm_loss = None

        # for CL train
        N,S = attention_mask.shape

        res = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        mask_setences =  attention_mask.unsqueeze(dim=2).expand(N,S,768)
        layer1 = res[2][1]*mask_setences
        layer12 = res[2][12]*mask_setences

        # for token
        token_mask_sentences = token_pos.unsqueeze(dim=2).expand(N,S,768)
        layer1_token = res[2][1]*token_mask_sentences
        layer12_token = res[2][12]*token_mask_sentences

        layer_concat = torch.cat((layer1,layer12),dim=1)
        (batch,tokens,embeddings) = layer_concat.shape

        tokens_numbers = torch.sum(attention_mask,dim=1).view(batch,1).expand(batch,embeddings)
        sentence_embeddings = torch.sum(layer_concat,dim=1) / tokens_numbers
        
        # for token_embeddings
        layer_token_concat = torch.cat((layer1_token,layer12_token),dim=1)
        tokens_pos_numbers = torch.sum(token_pos,dim=1).view(batch,1).expand(batch,embeddings)
        token_embeddings = torch.sum(layer_token_concat,dim=1) / tokens_pos_numbers

        # for the concat_embeddings
        concat_embeddings = torch.cat((sentence_embeddings,token_embeddings),dim=1)

        outputs = self.MLP_2(concat_embeddings)
        
        return outputs,masked_lm_loss