# -*- coding: utf-8 -*-
"""
@author: bdchen

"""

import os
import shutil
import codecs
import openpyxl
import linecache

import matplotlib.pyplot as plt

filename = 'log_1.tf'
str1 = linecache.getlines(filename)

batch = []
cl_loss = []
mlm_loss = []
total_loss = []
for i,snt in enumerate(str1):
    snt = snt.strip().split(',')
    cl_loss.append(float(snt[0].split(' ')[-1]))
    mlm_loss.append(float(snt[1].split(' ')[-1]))
    total_loss.append(float(snt[2].split(' ')[-1]))
    batch.append(i+1)

print(len(cl_loss))
print(len(mlm_loss))

print(len(total_loss))
print(len(batch))

# print(batch[10],loss[10])
# print(batch[10000],loss[10000])
# aa = (str1[59260].strip().split(',')[0]).split(' ')[-1]
# print(format(float(aa),'5f'))


# batch = [float(i) for i in range(62499)]
# loss = [float(62499-i) for i in range(62499)]

# plt.plot(batch,cl_loss,color='blue',label='cl_loss')
# plt.plot(batch,mlm_loss,color='red',label='mlm_loss')
plt.plot(batch,total_loss,color='black',label='total_loss')
plt.xlabel(u'batch number')
plt.ylabel(u'loss')
plt.title(u'loss figure of simclr_mlm')
plt.legend()
plt.savefig('test.jpg')





# wb = openpyxl.Workbook()
# sh = wb.active
# file_xlx = 'try.xlsx'

# for i,snt in enumerate(str1):
#     loss = (snt.strip().split(',')[0]).split(' ')[-1]
#     loss_num = format(float(loss),'5f')
#     sh.cell(i+1,2,loss_num)

# wb.save(file_xlx)
# wb.close


