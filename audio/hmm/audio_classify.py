# Example of how to classify an audio clip using the UrbanHMM class and an existing trained model

import pickle
import os
from UrbanHMM import *

audio_model_dir = 'models'
model_name = 'hmm_cvbest_f1_56437703.pkl'

model_path = os.path.join(audio_model_dir, model_name)
print(model_path)

with open(model_path, 'rb') as model_file:
    urban_hmm = pickle.load(model_file)

urban_hmm.predict('data/traffic_03252020205759.wav')
