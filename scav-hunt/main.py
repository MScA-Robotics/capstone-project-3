import config
from coneutils import calibrate

from scavbot import ScavBot
from scavear import Scavear, Listener

import multiprocessing as mp

boundaries_dict = calibrate.load_boundaries('coneutils/boundaries.json')

bot = ScavBot(
    image_model_dir='models/visual/custom_model',
    image_dir='data/images',
    params=config.params,
    boundaries=boundaries_dict
)

ear = Scavear(model_dir='models/audio', model_name='hmm_cvbest_f1_56437703.pkl')
ear.listen_record_classify_log()

# Wait for start signal
listener = Listener()
listener.listen()

p1 = mp.Process(target=bot.main)
p2 = mp.Process(target=ear.listen_record_classify_log, args=(150,))
