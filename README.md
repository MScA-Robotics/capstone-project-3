# capstone-project-3
MScA Robotics Capstone for Group 3 staring Fall 2019


# Setup

## Prerequisites

This setup requires a Raspberry Pi3 or Pi4 with the OS Raspberrian for 
Robots by DexterOS installed. It also requires Python3 and OpenCV. 

Instructions for Installing:
  - [Raspberrian for Robots](https://www.dexterindustries.com/howto/install-raspbian-for-robots-image-on-an-sd-card/)
  - OpenCV ([Pi3](https://pimylifeup.com/raspberry-pi-opencv/), [Pi4](https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/))
  - [Python3](https://medium.com/@isma3il/install-python-3-6-or-3-7-and-pip-on-raspberry-pi-85e657aadb1e) (needs to be 3.6 or greater)


Next, for the audio portion there are additional needs: the low-level virual machine and pyaudio. Both are 
better installed through apt-get then they are through pip.

    sudo apt-get install llvm
    sudo apt-get install python3-pyaudio
    
## Directory Setup
    
Clone the repository and install required packages.

    git clone git@github.com:MScA-Robotics/capstone-project-3.git
    cd capstone-project-3
    pip3 install -r requirements.txt

There are required directories and models that are not committed to
the repo. These are needed for the models to run and save data.

Within the scav-hunt directory, create a models directory and
a noise directory. Within models create an audio and visual directory.

    scav-hunt/
    |-- models/
    |   |-- audio/
    |   |-- visual/
    |-- noise/    
    
## Models 

There are two models needed for the scavenger hunt. An audio model
and a visual model. The audio model needs to be a pickle file or the 
UrbanAudio class. The visual model is a tensorflow-lite model and 
needs to be a .tflite file with an associated labelmap.txt in the 
same directory. Download both and put them in the associtated 
directory structure.

Also needed are the four numpy array files that make up the noise
masking for the audio file.

    scav-hunt/
    |-- models/
    |   |-- audio/
    |   |   |-- audio_model_file.pkl
    |   |-- visual/
    |   |   |-- visual_model/
    |   |   |   |-- model.tflite
    |   |   |   |-- labels.txt
    |-- noise/
    |   |-- mean_freq.npy
    |   |-- noise_db.npy
    |   |-- noise_thresh.npy
    |   |-- std_freq.npy

Finally, copy the example boundaries file into it's own 
boundaries file.

    cd scav-hunt/coneutils
    cp boundaries.json.example boundaries.json    

Setup is complete and the bot should run. Test by running the 
scavbot.py and scavear.py scripts. A microphone needs to be plugged
in for scavear and a camera plugged in for scavbot.
  
    python3 scavbot.py
    python3 scavear.py
