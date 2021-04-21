import numpy as np
import matplotlib.pyplot as plt

import IPython.display as ipd

from librosa.feature import melspectrogram as librosa_mel
from librosa.core import load as librosa_load

import tensorflow as tf

from yamnet.research.audioset.yamnet import params, yamnet

from os import path
import requests

model_path = "yamnet/research/audioset/yamnet/yamnet.h5"
yamnet_url = "https://storage.googleapis.com/audioset/yamnet.h5"

if not path.exists(model_path):
    print(f"Downloading yamnet pre-trained model from {yamnet_url} ... ")
    # pretrained model
    try:
        r = requests.get(yamnet_url, allow_redirects=True)
        with open(model_path, 'wb') as f:
            f.write(r.content)
        print("Done.")
    except Exception as e: print(e)

import dreamsound