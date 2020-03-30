#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import warnings
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc, logfbank
from hmmlearn import hmm
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

class UrbanHMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, class_map, num_cep_coef = 10, num_states = 5, num_iter = 1000):
        self.class_map = class_map
        self.num_states = num_states
        self.num_cep_coef = num_cep_coef
        self.num_iter = num_iter
        self.cov_type = 'diag'
        self.model_name = 'GaussianHMM'
        self._initialize_ensemble()

    def _initialize_ensemble(self):
        self.class_models = {key: hmm.GaussianHMM(n_components=self.num_states, covariance_type=self.cov_type, n_iter=self.num_iter) for key in self.class_map.keys()}

    def _check_input_shape(self, input_):
        if type(input_) == str:
            return np.array(input_).reshape(1,-1)
        if type(input_) == list:
            return np.array(input_).reshape(-1,1)
        elif input_.shape == ():
            return input_.reshape(1,-1)
        elif (len(input_.shape) == 1) & (input_.shape[0] >= 1):
            return input_.reshape(-1,1)
        elif (len(input_.shape) == 2) & (input_.shape[0] >= 1):
            return input_
        else:
            raise ValueError("Array is not in the correct shape. {}".format(input_.shape))

    def _mfcc(self, class_files):
        X = np.array([])
        class_files = self._check_input_shape(class_files)

        for file in class_files:
            file = file[0]
            # Extract the current filepath and read the file
            try:
                sampling_freq, signal = wavfile.read(file)
            except Exception as e:
                print(e)
                print("Failed to read {}".format(file))
                break
            # Extract features
            # Default values:
            # winlen=0.025, winstep=0.01, nfilt=26, nfft=512,
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features_mfcc = mfcc(signal, sampling_freq, numcep= self.num_cep_coef)
            # Append features to the variable X
            if len(X) == 0:
                X = features_mfcc
            else:
                X = np.append(X, features_mfcc, axis=0)
        return self._check_input_shape(X)

    def fit(self, X, y, verbose = False):
        self.train_files = np.array(X)
        self.train_classes = np.array(y)

        if set(self.train_classes) != set(self.class_map.values()):
            raise ValueError("Training data does not have same classes as class_map")

#        progress = []
        for key, model in self.class_models.items():
            _training_data = self._mfcc(self.train_files[np.where(self.train_classes == self.class_map[key])[0]])
#           if verbose:
#                progress.append(self.class_map[key])
#                print("Fitting Classes {}".format(progress), end="\r", flush=True)
            self.class_models[key] = model.fit(_training_data)

        return self

    def predict(self, X, y = None, prediction_type = "ids"):

        if prediction_type not in ("labels", "ids"):
            raise ValueError('prediction_type must be "labels" or "ids"')

        def _predict_one(file):
            scores = {class_: model.score(self._mfcc(file)) for class_, model in self.class_models.items()}
            if prediction_type == 'labels':
                return max(scores, key=scores.get)
            else:
                return self.class_map[max(scores, key=scores.get)]

        X = self._check_input_shape(X)

        predicted_classes = np.array([], dtype = int)

        for file in X:
            predicted_classes = np.append(predicted_classes, _predict_one(file))
        return predicted_classes

    def predict_llik(self, X, y = None):
        predicted_llik = pd.DataFrame(index = X, columns = list(class_map.keys()))

        def _predict_llik_one(file):
            scores = {class_: model.score(self._mfcc(file)) for class_, model in self.class_models.items()}
            return scores

        X = self._check_input_shape(X)

        for file in X:
            predicted_llik.loc[file] = _predict_llik_one(file)
        return predicted_llik

    def score(self, X, y):
        y_pred = self.predict(X)
        return(f1_score(y, y_pred, average = 'macro'))
