# -*- coding: utf-8 -*-
"""
@author: bdchen

"""

from __future__ import division
import numpy as np
import datetime
import time, random, copy, json
import re
import torch
import os
import codecs
import logging
import transformers
import sys
import pdb
import torch.nn as nn

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    BertModel,
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
)
from transformers.modeling_bert import BertLMPredictionHead

from config import config
import my_model
from my_model import My_Model_f2l
import linecache

model_file = './model/bert-base-multilingual-cased/' # download bert-base-multilingual-cased from Huggingface
train_file = './data/out_file_para_all/choose25M/all_files.txt'

max_length = 128
mlm_probability = 0.15
lambda_mlm = 0.1 #0.1
# QUEUE_LENGTH = 17360



def get_inputs(train_batch_sentences,tokenizer,train_batch_sentences_pos):
    input_ids_all = []
    attention_mask_all = []
    token_type_ids_all = []

    token_pos_all = []
    
    for i,sentence in enumerate(train_batch_sentences):
        text_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, return_attention_mask=True, max_length=max_length)

        input_ids, token_type_ids, attention_mask = text_dict['input_ids'], text_dict['token_type_ids'], text_dict['attention_mask']

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        start_pos = int(train_batch_sentences_pos[i].split('\t')[0])
        end_pos = int(train_batch_sentences_pos[i].split('\t')[1])
        token_pos_snt = [0]*start_pos + [1]*(end_pos - start_pos + 1) + [0]*(max_length - end_pos - 1)
        token_pos_all.append(token_pos_snt)

        input_ids_all.append(input_ids)
        attention_mask_all.append(attention_mask)
        token_type_ids_all.append(token_type_ids)

    input_ids_tensor = torch.tensor(input_ids_all, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask_all, dtype=torch.long)
    token_type_ids_tensor = torch.tensor(token_type_ids_all, dtype=torch.long)

    token_pos_tensor = torch.tensor(token_pos_all, dtype=torch.long)

    #sentence_embeddings = model(input_ids_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_type_ids_tensor)

    #sentence_embeddings_ids = torch.tensor(train_batch, dtype=torch.long).to('cuda')

    return input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, token_pos_tensor


def get_sentence_embeddings(train_batch_sentences, model, tokenizer, train_batch_sentences_pos):
    input_ids_all = []
    attention_mask_all = []
    token_type_ids_all = []

    token_pos_all = []
    
    for i,sentence in enumerate(train_batch_sentences):
        text_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, return_attention_mask=True, max_length=max_length)

        input_ids, token_type_ids, attention_mask = text_dict['input_ids'], text_dict['token_type_ids'], text_dict['attention_mask']

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        # join token_pos
        start_pos = int(train_batch_sentences_pos[i].split('\t')[0])
        end_pos = int(train_batch_sentences_pos[i].split('\t')[1])
        token_pos_snt = [0]*start_pos + [1]*(end_pos - start_pos + 1) + [0]*(max_length - end_pos - 1)
        token_pos_all.append(token_pos_snt)

        input_ids_all.append(input_ids)
        attention_mask_all.append(attention_mask)
        token_type_ids_all.append(token_type_ids)

    input_ids_tensor = torch.tensor(input_ids_all, dtype=torch.long).to('cuda')
    attention_mask_tensor = torch.tensor(attention_mask_all, dtype=torch.long).to('cuda')
    token_type_ids_tensor = torch.tensor(token_type_ids_all, dtype=torch.long).to('cuda')

    token_pos_tensor = torch.tensor(token_pos_all, dtype=torch.long).to('cuda')

    #sentence_embeddings = model(input_ids_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_type_ids_tensor)

    concat_embeddings = model(input_ids_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_type_ids_tensor,token_pos=token_pos_tensor)

    #sentence_embeddings_ids = torch.tensor(train_batch, dtype=torch.long).to('cuda')

    #return sentence_embeddings
    return concat_embeddings





def my_get_special_tokens_mask(token_ids_0):
    return list(map(lambda x: 1 if x in [101, 102, 0] else 0, token_ids_0))


def mask_tokens(inputs: torch.Tensor, tokenizer, special_tokens_mask = None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            my_get_special_tokens_mask(val) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y, temp=None):
        if temp is None:
            temp = self.temp
        return self.cos(x, y) / temp


def main():
    max_seq_length = config.max_seq_length

    neg_threshold = config.neg_threshold
    total_train_examples = config.total_train_examples
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    warmup_proportion = config.warmup_proportion
    log_dir = config.log_dir

    lr = config.lr

    num_train_steps = int(total_train_examples / batch_size * num_epochs)
    # print(num_train_steps)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    # print(num_warmup_steps)


    # ------------------------------- initialize config -------------------

    random_seed = 2021
    rng = random.Random(random_seed)

    bert_config = BertConfig.from_pretrained(model_file, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_file)
    # model = BertModel.from_pretrained(model_file, config=config)

    # ----------------------------- define a logger -------------------------------
    logger = logging.getLogger("execute")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_dir + "log_1.tf", mode="w")
    fh.setLevel(logging.INFO)

    fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    n_gpu = torch.cuda.device_count()
    print('Start training...')  ##rewrite

    # ------------------------ prepare training data ------------------------

    # LANG = ['ar','en','es','fr','ru','zh']
    # Dict_all_train_samples = {}
    # for lg in LANG:
    #     print(lg)
    #     train_file_name = os.path.join(train_file,f'UNv1.0.6way.{lg}')
    #     str1 = linecache.getlines(train_file_name)
    #     for num,snt in enumerate(str1):
    #         str1[num] = (snt.split('\n')[0])
    #     Dict_all_train_samples[lg] = str1
    #     str1 = []

    Dict_all_train_samples = []
    Dict_all_train_samples = linecache.getlines(train_file)
    print('Finish data-process')

    # ----------------------------------- begin to train -----------------------------------

    model = My_Model_f2l.from_pretrained(model_file, config=bert_config)

    model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)  # (model_q.parameters(), lr=0.0005, momentum=0.9, nesterov=True)#, weight_decay=1e-5)
    per = config.num_epochs // 6
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[per * 2, per * 4, per * 5], gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss()

    sentence_cloud_ids = [i for i in range(25000000)]  # 平行语料组唯一id

    count_sum = 0
    count_train = 0

    cross_metrics = torch.cat((torch.eye(batch_size).to('cuda'),torch.zeros(batch_size,batch_size).to('cuda')),dim=1)
    right_cross_metrics = torch.cat((torch.zeros(batch_size,batch_size).to('cuda'),torch.eye(batch_size).to('cuda')),dim=1)
    re_cross_metrics = torch.ones(batch_size,2*batch_size).to('cuda') - cross_metrics - right_cross_metrics

    cos_sim = Similarity(0.07)

    for epoch in range(num_epochs):
        start = time.time()

        random.shuffle(sentence_cloud_ids)  # 随机打乱ids

        model.train()

        for i in range(0, len(sentence_cloud_ids), batch_size):
            # try:
            train_batch = sentence_cloud_ids[i:i + batch_size]
            # 构造train_batch_sentences
            train_batch_sentences = []
            train_batch_sentences_ans = []

            train_batch_sentences_pos = []
            train_batch_sentences_ans_pos = []

            for idx in train_batch:
                train_batch_sentences.append(Dict_all_train_samples[idx].strip().split('\t')[0])
                train_batch_sentences_pos.append(Dict_all_train_samples[idx].strip().split('\t')[2] + '\t' + Dict_all_train_samples[idx].strip().split('\t')[3])
                
                train_batch_sentences_ans.append(Dict_all_train_samples[idx].strip().split('\t')[4])
                train_batch_sentences_ans_pos.append(Dict_all_train_samples[idx].strip().split('\t',6)[-1])

            count_train += 1

            if len(train_batch) != batch_size:
                continue

            if True:
                print("start training officially")

                # 正例需要mask


                input_ids_tensor_q, attention_mask_tensor_q, token_type_ids_tensor_q, token_pos_tensor_q = get_inputs(train_batch_sentences, tokenizer, train_batch_sentences_pos)
                mlm_input_ids, mlm_labels = mask_tokens(input_ids_tensor_q, tokenizer)
                input_ids_tensor_q = input_ids_tensor_q.to('cuda')
                attention_mask_tensor_q = attention_mask_tensor_q.to('cuda')
                token_type_ids_tensor_q = token_type_ids_tensor_q.to('cuda')
                token_pos_tensor_q = token_pos_tensor_q.to('cuda')
                mlm_input_ids = mlm_input_ids.to('cuda')
                mlm_labels = mlm_labels.to('cuda')

                query_sent_array_q , mlm_loss = model(input_ids_tensor_q, attention_mask=attention_mask_tensor_q, token_type_ids=token_type_ids_tensor_q, mlm_input_ids=mlm_input_ids, mlm_labels=mlm_labels, token_pos=token_pos_tensor_q)                                                             
                
                input_ids_tensor_k, attention_mask_tensor_k, token_type_ids_tensor_k, token_pos_tensor_k = get_inputs(train_batch_sentences_ans, tokenizer, train_batch_sentences_ans_pos)
                mlm_input_ids_k, mlm_labels_k = mask_tokens(input_ids_tensor_k, tokenizer)
                input_ids_tensor_k = input_ids_tensor_k.to('cuda')
                attention_mask_tensor_k = attention_mask_tensor_k.to('cuda')
                token_type_ids_tensor_k = token_type_ids_tensor_k.to('cuda')
                token_pos_tensor_k = token_pos_tensor_k.to('cuda')
                mlm_input_ids_k = mlm_input_ids_k.to('cuda')
                mlm_labels_k = mlm_labels_k.to('cuda')

                ans_sent_array_k, none_loss = model(input_ids_tensor_k, attention_mask=attention_mask_tensor_k, token_type_ids=token_type_ids_tensor_k, mlm_input_ids=mlm_input_ids_k, mlm_labels=mlm_labels_k, token_pos=token_pos_tensor_k)

                #sim = cos_sim(query_sent_array_q.unsqueeze(1),ans_sent_array_k.unsqueeze(0))

                sim_out = cos_sim(query_sent_array_q.unsqueeze(1),ans_sent_array_k.unsqueeze(0)) # B_q * B_k
                sim_in = cos_sim(query_sent_array_q.unsqueeze(1),query_sent_array_q.unsqueeze(0)) # B_q * B_k

                sim = torch.cat((sim_out,sim_in),dim=1) # B_q * (B_k + B_q)
                # labels = torch.arange(sim.size(0)).long().to(sim.device)
                # cl_loss = criterion(sim, labels)
                sim_pos = torch.sum(sim*cross_metrics,dim=-1)
                expsim = torch.exp(sim)
                sim_nag = torch.log(torch.sum(expsim*re_cross_metrics,dim=-1))
                cl_loss = torch.mean(sim_nag - sim_pos)

                total_mlm_loss = 0.5*(mlm_loss + none_loss)
                loss = cl_loss + lambda_mlm*total_mlm_loss

                # update model_q
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.info("cls_loss %s, mlm_loss %s, total_loss %s, count_train %s" % (cl_loss.item(), total_mlm_loss.item(), loss.item(), count_train))
                print(loss.item())
            if count_train % 250 == 0:
                os.makedirs(f'./model_save/{count_train}/')
                model.save_pretrained(f'./model_save/{count_train}/')
                # torch.save(model_q.state_dict(), './model_save/' + str(count_train) + '_' + 'model.pt')
                # 加载时用
                # model = BertModel.from_pretrained(model_file, config=bert_config)
                # model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
                # _,_=valid(count_train,model_q,file,batch_size, tokenizer,max_seq_length, out_dim,output_path)
                # except:
                #     print('next epoch')
                #     #import pdb; pdb.set_trace()
                #     continue

        scheduler.step()

    end = time.time()


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
