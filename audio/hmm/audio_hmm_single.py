#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, GridSearchCV
import pickle
from UrbanHMM import UrbanHMMClassifier

import multiprocessing

print(multiprocessing.cpu_count())

modelspath = 'models'
if not os.path.exists(modelspath):
    os.makedirs(modelspath)


fulldatasetpath = '../downsampled/'


metadata = pd.read_csv('../UrbanSound8K.csv')


le = LabelEncoder()
le.fit(metadata['class'])
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

Nstates = 5
Ncoef = 20

urban_hmm = UrbanHMMClassifier(class_map = class_mapping, num_states = Nstates, num_cep_coef = Ncoef)
urban_hmm.fit(X = list(fulldatasetpath + metadata[metadata['fold'] != 10]['slice_file_name'].astype(str)), 
               y = le.transform(metadata[metadata['fold'] != 10]['class']))

scored = urban_hmm.score(X = list(fulldatasetpath + metadata[metadata['fold'] == 10]['slice_file_name'].astype(str)), 
                           y = le.transform(metadata[metadata['fold'] == 10]['class']))

print("Fold 10 Score")
print(scored)

fname = "./models/hmm_fold10test_s{}_c{}.pkl".format(Nstates, Ncoef)
pickle.dump(grid_search.best_estimator_, open(name, "wb"))

