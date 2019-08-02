#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sema

Generate substitutability scores from a .p file

1. Preprocess data 
2. Compute substitutabilityScore 
    score =  [jaccardIndex, jaccardIndex2]
3. Output in the form of a dict

"""
import pickle
import datetime
import logging
date = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
filename = 'logs/XP_' + date + '.log'
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename=filename, 
                    format = '%(asctime)s %(message)s', 
                    level=logging.INFO)

import pandas as pd
from substitutability import *

filepath_dico = 'data/dict_cod.p'
filepath_data = 'data/conso_ad.p'
level = 'codsougr'
max_meal = 10
score = jaccardIndex

### Import data
dict_cod = openPickleFile(filepath_dico)
conso = openPickleFile(filepath_data)
dico = dict_cod[level]

res = main(conso, level, dico, max_meal, score)
res_dict = saveToDict(res)

with open('results/subScoreAllMeals.p', 'wb') as handle:
    pickle.dump(res_dict, handle)