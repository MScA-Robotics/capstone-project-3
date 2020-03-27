#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import classification_report
import pickle
from UrbanHMM import UrbanHMMClassifier


modelspath = 'models'
if not os.path.exists(modelspath):
    os.makedirs(modelspath)


fulldatasetpath = '../downsampled/'


metadata = pd.read_csv('../UrbanSound8K.csv')


le = LabelEncoder()
le.fit(metadata['class'])
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

parameters = {
    'num_cep_coef': [25,30,35,40,45,50], 
    'num_states':[2,3,4,5,6]
}

gKFold = GroupKFold(n_splits = 10)
urban_hmm = UrbanHMMClassifier(class_map = class_mapping)
grid_search = GridSearchCV(urban_hmm, parameters, cv = gKFold, n_jobs = -1, verbose = 1)
grid_search.fit(X = list(fulldatasetpath + metadata['slice_file_name'].astype(str)), 
               y = le.transform(metadata['class']),
               groups = metadata['fold'])

best_filename = "./models/hmm_cvbest_f1_{}.pkl".format(str(grid_search.best_score_)[2:10])
pickle.dump(grid_search.best_estimator_, open(best_filename, "wb"))


cv_filename = "./models/hmm_cv_f1_{}.pkl".format(str(grid_search.best_score_)[2:10])
pickle.dump(grid_search, open(cv_filename, "wb"))


print("\n-----------GRID SEARCH RANKING----------\n")
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

print("\n---------FULL TRAIN REPORT------------\n")
y_true = le.transform(metadata['class'])
predict  = grid_search.best_estimator_.predict(
    X = list(fulldatasetpath + metadata['slice_file_name'].astype(str)))
print(classification_report(y_true, predict, target_names=sorted(class_mapping, key=class_mapping.get)))
