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

parameters = {
    'num_cep_coef': [17,20,25,30,40], 
    'num_states':[3,4,5]
}

gKFold = GroupKFold(n_splits = 10)
urban_hmm = UrbanHMMClassifier(class_map = class_mapping)
grid_search = GridSearchCV(urban_hmm, parameters, cv = gKFold, n_jobs = -1, verbose = 5)
grid_search.fit(X = list(fulldatasetpath + metadata['slice_file_name'].astype(str)), 
               y = le.transform(metadata['class']),
               groups = metadata['fold'])

best_filename = "./models/hmm_cvbest_f1_{}.pkl".format(str(grid_search.best_score_)[2:10])
pickle.dump(grid_search.best_estimator_, open(best_filename, "wb"))


cv_filename = "./models/hmm_cv_f1_{}.pkl".format(str(grid_search.best_score_)[2:10])
pickle.dump(grid_search, open(cv_filename, "wb"))



