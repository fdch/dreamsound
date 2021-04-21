# DreamSound

Repository for the dreamsound python package.

## Install
Install using pip

```sh
python3 -m pip install dreamsound
```

**NOTE: you need to download the yamnet model before importing the dreamsound module. Please, continue reading.**


## Get Yamnet

Create a directory for your project

```sh
mkdir dream_test
cd dream_test
```
Run the yamnet downloader file distributed on this repository

```sh
python3 ../yamnet_downloader.py
```

Alternatively, you can get yamnet crudely like this:
```sh
git clone https://github.com/fdch/models.git models
mv models/research/audioset/yamnet/* .
rm -rf models
curl -O https://storage.googleapis.com/audioset/yamnet.h5
```

## Usage
You must have the yamnet model on the same directory. Now, you can import the dreamsound module and use the class. This code loads some files from disk and passes them to the `DreamSound` class from the `dreamsound` module. This looks something like this:

```python
import dreamsound

# Instantiate the DreamSound class
ds = dreamsound.DreamSound(["file1.wav", "file2.wav"])
```
## Filtering
There are two types of filtering, auto or targetted:

### Auto Filtering:
Filter the first audio with it's dreamed self

```python
ds(audio_index=0) 
```

### Targetted Filtering:
Filter the first with a dreamed target
```python
ds(audio_index=0, target=1) 
```

## Recurse
Finally, you can pass no arguments to continue filtering recursively
```python
ds()
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
