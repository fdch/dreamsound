# DreamSound

Repository for the dreamsound python package.

## Description

Inspired by the [DeepDream](https://www.tensorflow.org/tutorials/generative/deepdream) project, DreamSound plays a sound file to 
[Yamnet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet), a pre-trained neural network, and Yamnet returns a dreamed sound. 

Internally, DreamSound takes the gradients of a class from the pre-trained yamnet model and filters them with an original sound with some combination technique.

## Install
Dreamsound depends on the following pip packages you can `pip install`:
```
requests
numpy
matplotlib
IPython
librosa
tensorflow
soundfile
```

First, install the dependencies
```sh
python3 -m pip install -r requirements.txt
```

Install the dreamsound package using pip:

```sh
python3 -m pip install dreamsound
```

**NOTE: you need to download the yamnet model before importing the dreamsound module. Please, continue reading.**


## Prepare

Create a directory for your project and relocate there.

```sh
mkdir dream_test
cd dream_test
```

### Run the Yamnet Downloader.

The Yamnet Downloader file *does not* come with the pip distribution. However, it is distributed on this repository. If you do not want to clone this repository, simply do: `curl -O https://raw.githubusercontent.com/fdch/dreamsound/main/yamnet_downloader.py` on this same directory, and run:
```sh
python3 yamnet_downloader.py
```
Alternatively, you can get Yamnet yourself, crudely like this:

```sh
git clone https://github.com/fdch/models.git models
mv models/research/audioset/yamnet/* .
rm -rf models
curl -O https://storage.googleapis.com/audioset/yamnet.h5
```

## Usage example
You must have the yamnet model on the same directory. Now, you can import the dreamsound module and use the class. This code loads some files from disk and passes them to the `DreamSound` class from the `dreamsound` module. This looks something like this:

```python
>>> import dreamsound
INFO:tensorflow:Enabling eager execution
INFO:tensorflow:Enabling v2 tensorshape
INFO:tensorflow:Enabling resource variables
INFO:tensorflow:Enabling tensor equality
INFO:tensorflow:Enabling control flow v2
>>> ds = dreamsound.DreamSound(["../audio/original.wav", "../audio/cat.wav"])
Loading audio files...
Done.
I have now 2 audio files in memory.
Using last layer.
Yamnet loaded, using layer:activation_1
Dreamer started.
```
## Filtering
There are two types of filtering, auto or targetted:

### Auto Filtering:
Filter the first audio with it's dreamed self

```python
>>> ds(audio_index=0)
Running step 0, class: Whistling...
...
Writing ./audio/Whistle-9-orig.wav...
Writing ./audio/Whistle-9-diff.wav...
Writing ./audio/Whistle-9-filt.wav...
Writing ./audio/Whistle-9-hard.wav...
Writing ./audio/Whistle-9-grad.wav...
```

### Targetted Filtering:
Filter the first with a dreamed target
```python
>>> ds(audio_index=0, tgt=1)
Target class: Animal...
Running step 0, class: Whistling...
Running step 1, class: Whistle...
Running step 2, class: Whistle...
Running step 3, class: Whistle...
Running step 4, class: Whistle...
Running step 5, class: Whistle...
Running step 6, class: Whistle...
Running step 7, class: Whistle...
Running step 8, class: Flute...
Running step 9, class: Whistle...
Writing ./audio/Whistle-9-orig-tgt-Animal.wav...
Writing ./audio/Whistle-9-diff-tgt-Animal.wav...
Writing ./audio/Whistle-9-filt-tgt-Animal.wav...
Writing ./audio/Whistle-9-hard-tgt-Animal.wav...
Writing ./audio/Whistle-9-grad-tgt-Animal.wav...
```

## Recurse
Finally, you can pass no arguments to continue filtering recursively
```python
>>> ds()
Target class: Animal...
Running step 10, class: Whistle...
Running step 11, class: Whistle...
Running step 12, class: Wind instrument, woodwind instrument...
Running step 13, class: Wind instrument, woodwind instrument...
Running step 14, class: Flute...
Running step 15, class: Flute...
Running step 16, class: Wind instrument, woodwind instrument...
Running step 17, class: Whistle...
Running step 18, class: Whistle...
Running step 19, class: Music...
Writing ./audio/Music-19-orig-tgt-Animal.wav...
Writing ./audio/Music-19-diff-tgt-Animal.wav...
Writing ./audio/Music-19-filt-tgt-Animal.wav...
Writing ./audio/Music-19-hard-tgt-Animal.wav...
Writing ./audio/Music-19-grad-tgt-Animal.wav...
```

## Class Variables

You can change any of the following before or after calling the class:

```python
sr          = 22050
max_dur     = 10
patch_hop   = 0.1
win_length  = 2048
hop_length  = 128
pad_end     = False
loss_power  = 0.001
plot_every  = 10
figsize     = (10,8)
top_db      = 80.0
step_size   = 0.95
output_type = 3
steps       = 10
threshold   = 1e-07
classid     = None
maxloss     = True
elapsed     = 0
recurse     = False
target      = None
power       = 1.0
audio_dir   = "./audio/"
image_dir   = "./image/"
```

# Authors

Fede Camara Halac (https://github.com/fdch)
Matias Delgadino (https://github.com/zaytam)

# Acknowledgements

YamNet
AudioSet
