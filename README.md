# DreamSound

Repository for the dreamsound python package. 

## install

```shell
pip3 install dreamsound
```

## usage

This code loads some files from disk and passes them to the `DreamSound` class from the `dreamsound` module.

### import the module

```python
import dreamsound as d
```

### instantiate the DreamSound class
```python
ds = d.DreamSound(["file1.wav", "file2.wav"])
```

When you call the class, you can pass an audio index specifying which audio you are using. Calling the class will run the model for 10 steps (the default)

```python
ds(audio_index=0)
```

## parameters

You can change any of the following parameters before calling the class:

```python
sr = 22050
max_dur = 10
patch_hop = 0.1 # in seconds

# fft params
win_length = 2048
hop_length = 128
pad_end = False
norm_factor = 5.999975341271492

# loss power
loss_power = 0.001

# plotting
plot_every = 10
figsize = (10,8)
top_db = 80.0

# perfom
step_size = 0.95
output_type = 3
steps = 10
threshold = 1e-07
classid = None
maxloss = True
elapsed_steps = 0

# prevent loading recursively on first run
enable_recursion = False  

# filter
w_tgt = None
power = 1.0
```
