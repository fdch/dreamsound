###############################################################################
#
#  DreamSound class
#
###############################################################################

import os
import sys
import pathlib

__init__dir   = str(pathlib.Path(__file__).parent.absolute())
__models__dir = "/models"
__yamnet__dir = "/research/audioset/yamnet"
__weights__   = "yamnet.h5"
__classes__   = "yamnet_class_map.csv"
__yamnet__ = {
    "url"    : "https://github.com/tensorflow/models.git",
    "wurl"   : "https://storage.googleapis.com/audioset/" + __weights__,
    "path"   : __init__dir + __models__dir,
    "model"  : __init__dir + __models__dir + __yamnet__dir,
    "weights": __init__dir + __models__dir + __yamnet__dir + "/" + __weights__,
    "classes": __init__dir + __models__dir + __yamnet__dir + "/" + __classes__,
    "init"   : __init__dir + __models__dir + __yamnet__dir + "/__init__.py"
}
audio_dir = __init__dir + "/../audio/"
images_dir = __init__dir + "/../images/"
###############################################################################
# first time only: check if model is downloaded, place it next to this file   #
###############################################################################

if not os.path.exists(__yamnet__["path"]):
    print(f"Downloading tensorflow models from {__yamnet__['url']} ... ")
    try:
        os.system(f"git clone {__yamnet__['url']} {__yamnet__['path']}")
        print("Done.")
    except Exception as e: print(e)   

os.chdir(__yamnet__["model"])

if not os.path.exists(__yamnet__["weights"]):
    print(f"Downloading tensorflow models from {__yamnet__['wurl']} ... ")
    try:
        os.system(f"curl -O {__yamnet__['wurl']}")
        print("Done.")
    except Exception as e: print(e)   

sys.path.append(__yamnet__["model"])
import params, yamnet

os.chdir(__init__dir)

###############################################################################
# got yamnet                                                                 #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from librosa.feature import melspectrogram as librosa_mel
from librosa.core import load as librosa_load
import tensorflow as tf
import soundfile as sf

class DreamSound:

    sr = 22050
    max_dur = 10
    patch_hop = 0.1 # seconds

    # fft params
    win_length = tf.constant(2048)
    hop_length = tf.constant(128)
    pad_end = False
    norm_factor = tf.constant(5.999975341271492)

    # loss power
    loss_power = tf.constant(0.001)
    
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

    def __init__(self, paths=None, layer=None):

        # fill in array 'audio' with sounds as audio data 
        self.audio = []
        if paths is not None:
            self.load_audio(paths)

        self.load_model(layer)
    
    def load_model(self, layer):
        # load its class names
        self.class_names = yamnet.class_names(__yamnet__["classes"])
        # load model parameters and get model
        self.params = params.Params(sample_rate=self.sr,
            patch_hop_seconds=self.patch_hop)
        self.model = yamnet.yamnet_frames_model(self.params)
        # load model weigths
        self.model.load_weights(__yamnet__["weights"])
        if layer is not None:
            self.layername = layer
        else:
            print("Using last layer.")
            self.layername = self.model.layers[-1].name
        print(f"Yamnet loaded, using layer:{self.layername}")

        # Maximize the activations of these layers
        self.layers = self.model.get_layer(self.layername).output
        # Create the feature extraction model
        self.dreamer = tf.keras.Model(inputs=self.model.input, outputs=self.layers)

        print("Dreamer started.")

    def load_audio(self, paths):
        print("Loading audio files...")
        for p in paths:
            y,_ = librosa_load(p, sr=self.sr, duration=self.max_dur)
            self.audio.append(y)
        print("Done.")
        print(f"I have now {len(self.audio)} audio files in memory.")

    def clear_audio(self):
        del self.audio
        self.audio = []
        print(f"I have now {len(self.audio)} audio files in memory.")

    def get_class(self, classid):
        if isinstance(classid, str):
            res = []
            for i, x in enumerate(self.class_names):
                if classid in x:
                    print(i,x)
                    res.append(i)
            return res
        elif isinstance(classid,int):
            return self.class_names[classid]
        else:
            print(f"Can't parse: {type(classid)}")
    
    def summary(self):
        self.model_name.summary()
    
    def stft(self, x):
        return tf.signal.stft(x,
                              self.win_length,
                              self.hop_length,
                              pad_end=self.pad_end)
        
    def istft(self, x):
        return tf.signal.inverse_stft(x, 
                                      self.win_length, 
                                      self.hop_length)

    def complex_mul(self, x, y, norm=False):
        x_imag = tf.math.real(x)
        x_real = tf.math.imag(x)
        if norm:    
            # normalize real and imag
            x_real /= tf.math.reduce_max(tf.math.abs(x_real))
            x_imag /= tf.math.reduce_max(tf.math.abs(x_imag))
        
        conv = tf.dtypes.complex(self.clip_func(x_real,y,tf.math.multiply),
                                 self.clip_func(x_imag,y,tf.math.multiply))
        return conv
    
    def clip_func(self, x, y, fun):
        if x.shape[0] < y.shape[0] : y = y[:x.shape[0]]
        if y.shape[0] < x.shape[0] : x = x[:y.shape[0]]
        return fun(x,y)


    def combine_spectra(self, x, y):
        
        # take short time fourier transform
        X = self.stft(x)
        Y = self.stft(y)
        
        tf.print(tf.math.reduce_mean(tf.math.abs(X)))
        tf.print(tf.math.reduce_std(tf.math.abs(X)))
        
        # take magnitude
        Yabs = tf.math.abs(y) ** 2
        
        # get rid of tiny values
        Hfilt = Yabs * 0.5 * ( tf.math.sign(Yabs - self.threshold) + 1 )
        Hfilt *= self.step_size
    
        # multiply the magnitude with the complex pair
        Yfilt = self.complex_mul(Y, Hfilt, norm=True)
    
        # combine complex values and take inverse fourier
        combined = istft(self.clip_func(X,Yfilt,tf.math.add))

        return combined    
    
    def combine_spectra_2(self, x, y, target=None):
        
        # x is filtered
        # y is the filter if target is none

        X = self.stft(x)

        if target is not None:
            F = self.stft(target)
            Fabs = tf.math.abs(F) ** self.power
            y_filter = Fabs
        else:
            Y = self.stft(y)
            Yabs = tf.math.abs(Y) ** self.power
            y_filter = Yabs

        norm_y_filter = y_filter / tf.math.reduce_max(y_filter)
        y_filter_thresh = norm_y_filter - self.threshold

        y_filter *= (tf.math.sign(y_filter_thresh) + 1) * 0.5
        y_filter *= self.step_size

        X_y_filtered = self.complex_mul(X, y_filter, norm=True)

        x_real = self.istft(X_y_filtered)
        combined = self.clip_func(x_real, y, tf.math.add)
        y_filter_real = self.istft(
                tf.dtypes.complex(y_filter,
                    tf.zeros(y_filter.shape)))
 
        return combined, x_real, y_filter_real

    def calc_loss(self, data):

        layer_activations, argmax = self.class_from_audio(data)

        if self.classid is not None:
            loss =  tf.reduce_sum(layer_activations[:,self.classid])
        else:
            loss =  tf.reduce_sum(layer_activations[:,argmax])

        return loss ** self.loss_power, self.class_names[argmax]

    def class_from_audio(self, waveform):
        # Pass forward the data through the model to retrieve the activations.
        layer_activations = self.dreamer(waveform)
        reduced = tf.reduce_mean(layer_activations, axis=0)
        argmax = tf.argmax(reduced)
        return layer_activations, argmax

    def dream_big(self, source, target=None):
        
        wt =  tf.convert_to_tensor(source)

        if target is not None:
            wt_tgt = tf.convert_to_tensor(target)
            _, self.classid = self.class_from_audio(wt_tgt)
            tf.print(f"Target class: { self.class_names[self.classid] }...")
        else:
            wt_tgt = None

        wr_ = None
        wf_ = None

        loss = tf.constant(0.0)

        for i in tf.range(self.steps):

            with tf.GradientTape() as tape:
                tape.watch(wt)
                loss, c = self.calc_loss(wt)
           
            tf.print(f"Running step {self.elapsed_steps}, class: {c}...")

            gradients = tape.gradient(loss, wt)
            gradients /= tf.math.reduce_std(gradients)

            if self.output_type == 0:
                # combine spectra
                wt = self.combine_spectra(wt, gradients)
            elif self.output_type == 1:
                # default
                # In gradient ascent, the "loss" is maximized so that 
                # the input image increasingly "excites" the layers.
                # You can update the image by directly adding the gradients
                wt += gradients * self.step_size
            elif self.output_type == 2:
                # combine spectra, flipped
                wt = self.combine_spectra(gradients, wt)
            elif self.output_type == 3:
                # combine spectra, filtering by the original sound
                wt, wr_, wf_ = self.combine_spectra_2(gradients, wt, target=wt_tgt)
            elif self.output_type == 4:
                # combine spectra, filtering by the original sound FLIPPEd
                wt, wr_, wf_ = self.combine_spectra_2(wt, gradients, target=wt_tgt)
            elif self.output_type == 5:
                # return gradients only
                wt = gradients
            else:
                # return only wt
                wt = wt 

            if (i+1) % self.plot_every == 0 and i > 0:
                
                # store data in self arrays                
                self.difference = self.clip_func(wt,source,tf.math.subtract).numpy()
                self.gradients = gradients.numpy()
                self.wavetensor = wt.numpy()
                
                # plot
                plots = []
                fig_title = f"{c}-{self.elapsed_steps}"
                plots.append(self.wavetensor)
                plots.append("orig")
                plots.append(self.difference)
                plots.append("diff")
                if wr_ is not None:
                    self.inverse = tf.transpose(wr_).numpy()
                    plots.append(self.inverse)
                    plots.append("filt")
                if wf_ is not None:
                    self.filter = tf.transpose(wf_).numpy()
                    plots.append(self.filter)
                    plots.append("hard")
                plots.append(self.gradients)
                plots.append("grad")

                self.plotter(plots, file=fig_title)

            # end main loop
            self.elapsed_steps += 1 # increment count

        return wt

    def plotter(self, plots, file):

        num_plots = len(plots)//2

        fig = plt.figure(figsize=self.figsize)

        for i in range(num_plots):
            plt.subplot(num_plots//2, 3, i+1)
            plot = plots[i * 2]
            label = plots[i * 2 + 1]
            self.plot_and_listen(s=plot, label=file+"-"+label, play=False)

        plt.savefig(images_dir + file+".png")
        plt.close()



    def plot_and_listen(self,s=None,label='',wave=True, spec=True, play=True):
        
        if s is None:
            s = self.audio[0]

        if wave:
            self.plot(s,image=False, label=label)
            
        if spec:
            ss = librosa_mel(s)
            ss = np.log10(np.maximum(1e-05,ss))*10.0
            ss = np.maximum(ss, ss.max()-self.top_db)
            self.plot(ss, image=True, label=label)
        
        fname = audio_dir + label.replace(" ","_") + ".wav"
        print(f"Writing {fname}...")
        sf.write(fname, s, self.sr, subtype="PCM_24")
        
        if play:
            os.system(f"ffplay {fname}")
    

    def play(self):

        self.plot_and_listen()
    
    def plot(self, data, image=True, label=None):

        if image:
            plt.imshow(data, origin="lower", aspect="auto")
        else:
            plt.plot(data)
        if label is not None:
            plt.title(label)

    def print(self):
        print(vars(self))

    def __call__(self, audio_index=None, target=None):

        # first time, no index given
        if audio_index is None and not self.enable_recursion:
            w = self.audio[0] 
        
        # recurse, no index given
        elif audio_index is None and self.enable_recursion:
            w = self.x 
        
        # do not recurse, specific index given
        else:
            # reset elapsed
            self.elapsed_steps = 0
            w = self.audio[audio_index]
        
        # was a target provided?
        if target is not None:
            self.w_tgt = self.audio[target]

        self.x = self.dream_big(w, target=self.w_tgt)

        # enable recursion after first run
        self.enable_recursion = True

# end DreamSound class


#Get YAMNet

# filenames = ["/content/drive/MyDrive/python_scratch/audio/newfiles/audio/simple_piano.wav",
#              'speech_whistling2.wav']

# ds = DreamSound(filenames)

# print(ds.get_class(466))

# c = ds.get_class("Whistling")

# ds.power= 1
# ds.plot_every = 10
# ds.steps = 20
# ds.step_size = 0.95
# ds.threshold = 1e-07
# ds.output_type = 3 
# ds.classid = None
# # ds.classid = 76 # 'Cat' class

# ds(audio_index=0) # use first audio file

# ds.plot_and_listen(ds.gradients, label='Gradients')

# ds()

# ds.plot_and_listen(ds.gradients, label='More evolved Gradients')

# ds()

# ds()

# ds()