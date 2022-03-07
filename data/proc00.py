# -*- coding: utf-8 -*-
"""
@author: bdchen

"""

# 1. Download The United Nations Parallel Corpus v1.0 as Zemiski and unzip it into ./Zemiski/6way/ and download bert-base-multilingual-cased from Huggingface to ../model/
# 2. Split the parallel sentences into 100 pieces in order to deal with it faster
# 3. This is a demo to show how to process the first 10 pieces
# 4. After process the whole dataset, choose 5M sentences from each para-languages and mix them up to build a 25M para-sentences corpus for training


import linecache
import codecs
import os
import torch
import transformers
from transformers import (
  BertConfig,
  BertTokenizer,
  BertForTokenClassification,
)
import time


model_file = '../model/bert-base-multilingual-cased/'
max_length = 150

train_file = './Zemiski/split6way/'
LANG = ['ar','en','es','fr','ru','zh']

word_file = './usual_word/'


def get_word_position(line1,tokenizer,word1):
  sentence1 = line1.strip()
  
  text_dict1 = tokenizer.encode_plus(sentence1, add_special_tokens=True, return_attention_mask=True, max_length=max_length)
  input_ids1 = text_dict1['input_ids']
  #print(input_ids1)
  len_text1 = len(input_ids1)
  if len_text1 < 128:
    word_dict1 = tokenizer.encode_plus(word1, add_special_tokens=True, return_attention_mask=True, max_length=max_length)
    input_word1 = word_dict1['input_ids']
    input_word1.pop(0)
    input_word1.pop(-1)
    len_word1 = len(input_word1)
    #print(input_word1)
    
    for j,tex in enumerate(input_ids1):
      if tex==input_word1[0]:
        if len_word1==1:
          return j,j,1
        elif len_text1>(len_word1+j-1):
          flag = 1
          for i in range(len_word1-1):
            if input_ids1[j+i+1]==input_word1[i+1]:
              continue
            else:
              flag = 0
          if flag:
            return j,j+len_word1-1,1

  return 0,0,0


def main():

  bert_config = BertConfig.from_pretrained(model_file)
  tokenizer = BertTokenizer.from_pretrained(model_file)

  all_cur_word = {}
  for lg in LANG:
    f = codecs.open(os.path.join(word_file,f'{lg}_uni.txt'),'r','utf-8')
    str_word = []
    for s in f:
      str_word.append(s.strip().lower())
    all_cur_word[lg] = str_word
    f.close()

  for numberlg in range(10):

    Dict_all_train_samples = {}
    for lg in LANG:
        print(lg)
        train_file_name = os.path.join(train_file,f'{lg}_00{numberlg}')
        str1 = linecache.getlines(train_file_name)
        #str1 = linecache.getlines(f'{lg}.txt')
        Dict_all_train_samples[lg] = str1
        str1 = []

    all_para_samples = {}

    #time_start=time.time()

    for snt_id,en_snt in enumerate(Dict_all_train_samples['en']):
      all_para_samples[snt_id] = []
      en_snt = en_snt.lower().strip()
      for word_id,en_word in enumerate(all_cur_word['en']):
        if en_snt.count(en_word) == 1:
          flag = 0
          begin_pos1,end_pos1,flag_snt1 = get_word_position(en_snt,tokenizer,en_word)
          if flag_snt1:
            for lg in LANG:
              if lg == 'en':
                continue
              snt_lg = Dict_all_train_samples[lg][snt_id].lower().strip()
              word_lg = all_cur_word[lg][word_id]
              if snt_lg.count(word_lg) == 1:  
                begin_pos2,end_pos2,flag_snt2 = get_word_position(snt_lg,tokenizer,word_lg)
                if flag_snt2:
                  all_para_samples[snt_id].append(snt_lg + '\t' + word_lg + '\t' + str(begin_pos2) + '\t' + str(end_pos2) +'\n')
                  flag +=1
                
            if flag:
              all_para_samples[snt_id].append(en_snt + '\t' + en_word + '\t' + str(begin_pos1) + '\t' + str(end_pos1) +'\n'+'\n')

    print('finish find, start writing')

    out_file = './out_file_paraall/'
    

    f_out = codecs.open(os.path.join(out_file,f'00{numberlg}.txt'),'w','utf-8')
    for snt_id in range(len(Dict_all_train_samples['en'])):
      if all_para_samples[snt_id]:
        for snt in all_para_samples[snt_id]:
          f_out.write(snt)
        
    f_out.close()

    #time_end=time.time()
    #print('totally cost',time_end-time_start)



if __name__ == '__main__':

    main()
  


