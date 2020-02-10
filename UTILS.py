
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import f1_score
from numpy import argmax
import os
import re
import sys

# In[2]:

def load_paths(tegg):
    path_lists = []
    if tegg == 'rus':
        for path in ["_rus.pickle", "_rus.json", "_rus.h5", "_rus.npy", '_both.npy']:
            path_lists.append(sorted([name for name in os.listdir(".") if name.endswith(path)]))
        return path_lists
    if tegg == 'eng':
        for path in ["_eng.pickle", "_eng.json", "_eng.h5", "_eng.npy", '_both.npy']:
            path_lists.append(sorted([name for name in os.listdir(".") if name.endswith(path)]))
        return path_lists
    if tegg == 'both':
        path = sorted([name for name in os.listdir(".") if name.endswith('_both.npy')])[0]
        return path
    else:
        print('Unknow tegg: {}. Loading of paths id failed'.format(tegg)) 
        #sys.exit()

#In[3]:
def concat_statement(path, process, tegg, args = None):
    with open(path, 'r') as f:
        n = f.readlines()
    n = [x.strip('\n') for x in n]
    stat = []
    if process == 'train':
        i = 0
        for x in n:
            while n[i] != '|||||':
                stat.append(n[i])
                i+=1
            if x == '|||||':
                break
        stat = ' '.join(stat)
        if tegg == 'eng':
            stat = re.sub('!= 3.0', '= 3.0', stat)
            stat_1, stat_2 = stat.split('___')
            return stat_1, stat_2
        else:
            stat_1, stat_2 = stat.split('___')
            return stat_1, stat_2
    if process == 'prediction':
        for i, x in enumerate(n):
            if x == '|||||':
                stat.extend(n[i+1: len(n)])
        stat = ' '.join(stat)
        new_date = args.date+' '+args.time
        stat = re.sub('input_date' , new_date, stat)
        if tegg == 'eng':
            stat = re.sub('!= 3.0', '= 3.0', stat)
            stat_1, stat_2 = stat.split('___')
            return stat_1, stat_2
        else:
            stat_1, stat_2 = stat.split('___')
            return stat_1, stat_2
        
#In[4]: 

def load_params(process):
    list_params = []
    for i in sorted([name for name in os.listdir(".") if name.endswith('.txt')])[:3]:
        if i.endswith('TNS.txt'):
            with open(i, 'r') as f:
                x = f.readlines()
            dict_params = dict(re.sub(' ', '', line).rstrip().split('=') for line in x) 
        else:
            with open(i, 'r') as f:
                n = f.readlines()
            list_params.append({x.strip('\n').split(':')[0]: int(x.strip('\n').split(':')[1]) for x in n})
    if process == 'create':
        return dict_params, list_params[1]
    if process == 'train':
        return dict_params, list_params[0], list_params[1]
    if process == 'prediction':
        return dict_params, list_params[0], list_params[1]    
    
