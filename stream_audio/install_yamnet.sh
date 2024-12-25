#!/bin/bash

#update package list and install prerequisites
sudo apt-get update
sudo apt-get install -y wget git libsndfile1-dev curl

#download and install miniconda for ARM architecture
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O Miniforge3.sh
bash Miniforge3.sh -b -p $HOME/miniforge3
rm Miniforge3.sh

#initialize conda
source $HOME/miniforge3/bin/activate
conda init
source ~/.bashrc

#create a new conda environment with Python 3.9
conda create -n yamnet_env python=3.9 -y
conda activate yamnet_env

#upgrade pip and wheel
pip install --upgrade pip wheel

#install required Python packages
pip install numpy resampy tensorflow soundfile matplotlib tf_keras

#clone the TensorFlow models repository
git clone https://github.com/tensorflow/models.git

#copy the YAMNet project to the current directory
cp -r models/research/audioset/yamnet .

#remove the model dire
rm -rf models

#navigate to the YAMNet directory
cd yamnet

#download the YAMNet model weights and example audio file
curl -O https://storage.googleapis.com/audioset/yamnet.h5
curl -O https://storage.googleapis.com/audioset/speech_whistling2.wav
curl -O https://storage.googleapis.com/audioset/miaow_16k.wav
curl -O https://storage.googleapis.com/audioset/yamnet.tflite


echo "Installation complete. To use the YAMNet model, activate the environment with 'conda activate yamnet_env' and run your inference script."

echo "Testing..."
python3 yamnet_test.py

echo "Trying to make an inference..."
python3 inference.py speech_whistling2.wav

echo "You can now make an inference with the miaow_16k.wav audio file running the follofing command: python3 inference miaow_16k.wav"
