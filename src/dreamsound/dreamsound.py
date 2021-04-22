###############################################################################
#
#  This file is part of the dreamsound package.
#  
#  DreamSound class definition
#
###############################################################################

try:
    import yamnet, params
except ModuleNotFoundError: 
    print("Please, run `python yamnet_downloader.py` first. Exiting.")
    quit()

import os
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from librosa.feature import melspectrogram as librosa_mel
from librosa.core import load as librosa_load
import tensorflow as tf
import soundfile as sf

class DreamSound(object):
    """DreamSound class definition
    
    Description
    -----------
    Inspired by the Deep Dream project, DreamSound plays a sound file to 
    Yamnet, a pre-trained neural network, and Yamnet returns a dreamed sound.
        
    Parameters: (optional) [paths], [layer]
    -----------
    paths: array holding audio file paths
    layer: string identifying which layer to choose from yamnet 
    (default is the last activation layer)

    Returns
    -----------
    An instance of the DreamSound class

    Call
    -----------
    When called, the class computes `steps` times
    the `dream()` function, which takes the gradients of a class
    from the pre-trained yamnet model and filters them with an
    original sound with some combination technique defined
    in the `output_type` variable. Note that files will be
    written to disc at a rate of `plot_every` steps.

    Class Variables
    -----------
    sr          : (int) default samplerate (Hz) (22050) 
    max_dur     : (float) max soundfile duration (seconds) (10) 
    patch_hop   : (float) hop_size for yamnet (seconds) (0.1) 
    win_length  : (int) FFT window size (2048) 
    hop_length  : (int) STFT hop size (128) 
    pad_end     : (bool) apply padding (tf.signal.stft)  (False) 
    loss_power  : (float) small power to elevate the loss  (0.001) 
    plot_every  : (int) epochs to skip before plotting (10) 
    figsize     : (int,int) plot total width and height ((10,8)) 
    top_db      : (float) top decibel to clip to in displays (80.0) 
    step_size   : (float) apply filtering at some rate (0.95) 
    output_type : (int) choose filtering technique, see (*) (3) 
    steps       : (int) number of steps to recurse (10) 
    threshold   : (float) tiny threshold for filtering (1e-07) 
    classid     : (str|int) use specific class or auto-classify (None) 
    maxloss     : (bool) get the class of the maximum loss (True) 
    elapsed     : (int) how many steps performed so far (0) 
    recurse     : (bool) prevent loading recursively on first run   (False) 
    target      : (int) if specified, that indexed audio is the filter (None) 
    power       : (float) amplitude or power spectrum (default 1) (1.0) 
    audio_dir   : (str) path to output audio files ("./audio/")
    image_dir   : (str) path to output image files ("./image/")

    Methods
    -----------
    load_model(layer)
        loads the yamnet model with specified layer as string
    load_audio(paths)
        loads the array of sound files using librosa
    clear_audio()
        empties the internal sound arrays
    get_class(classid)
        pass the class name given either an int or a string
    summary()
        returns a summary of the model
    stft(x)
        wrapper for tf.stft
    istft(x)
        wrapper for tf.istft
    complex_mul(x, y, norm=False)
        wrapper for complex multiplication
    clip_func(x, y, fun)
        make broadcasting easier but nastier; 
        apply a function `fun` to x and y, clipping the smallest array
    combine_spectra(x, y)
        one type of spectral combination
    combine_spectra_2(x, y, target=None)
        yet another spectral combination
    calc_loss(data)
        get loss function
    class_from_audio(waveform)
        get the class of an audio file
    dream(source, tgt=None)
        perform the dreaming. if a tgt is specified, use that as filter
    plotter(plots, file)
        wrapper to plot using suplots
    plot_and_listen(s=None,label='',wave=True, spec=True, play=True)
        a plotting wrapper for matplotlib
    play()
        play back audio using ipython
    plot(data, image=True, label=None)
        wrap the plot functions
    print()
        print all internal parameters

    (*) Combinations 
    -----------
    You can define the following for `output_type`:

    0 - combine spectra
    1 - In gradient ascent, the "loss" is maximized so that 
    the input image increasingly "excites" the layers. 
    You can update the image by directly adding the gradients
    2 - combine spectra, flipped
    3 - combine spectra, filtering by the original sound
    4 - combine spectra, filtering by the original sound flipped
    5 - return gradients only
    
    anything else will only recurse on the original audio
    """
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
    tgt_class   = None
    power       = 1.0
    audio_dir   = "./audio/"
    image_dir   = "./image/"

    def __init__(self, paths=None, layer=None):

        # fill in array 'audio' with sounds as audio data 
        self.audio = []
        if paths is not None:
            self.load_audio(paths)

        self.load_model(layer)

        self.__mkdir__(self.audio_dir)
        self.__mkdir__(self.image_dir)

    def __mkdir__(self, path):
        if not os.path.exists(path):
            print(f"Making {path} directory...")
            os.mkdir(path)

    def load_model(self, layer):
        """loads the yamnet model with specified layer as string
        """

        # load its class names
        self.class_names = yamnet.class_names("yamnet_class_map.csv")
        # load model parameters and get model
        self.params = params.Params(sample_rate=self.sr,
            patch_hop_seconds=self.patch_hop)
        self.model = yamnet.yamnet_frames_model(self.params)
        # load model weigths
        self.model.load_weights("yamnet.h5")
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
        """loads the array of sound files using librosa
        """
        
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

    def dream(self, source, target=None):
        
        wt =  tf.convert_to_tensor(source)

        if target is not None:
            wt_tgt = tf.convert_to_tensor(target)
            _, self.classid = self.class_from_audio(wt_tgt)
            self.tgt_class = self.class_names[self.classid]
            tf.print(f"Target class: { self.tgt_class }...")
        else:
            wt_tgt = None

        wr_ = None
        wf_ = None

        loss = tf.constant(0.0)

        for i in tf.range(self.steps):

            with tf.GradientTape() as tape:
                tape.watch(wt)
                loss, c = self.calc_loss(wt)
           
            tf.print(f"Running step {self.elapsed}, class: {c}...")

            gradients = tape.gradient(loss, wt)
            gradients /= tf.math.reduce_std(gradients)

            if self.output_type == 0:
                # combine spectra
                wt = self.combine_spectra(wt, gradients)
            elif self.output_type == 1:
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
                fig_title = f"{c}-{self.elapsed}"
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
            self.elapsed += 1 # increment count

        return wt

    def plotter(self, plots, file):

        num_plots = len(plots)//2

        fig = plt.figure(figsize=self.figsize)

        for i in range(num_plots):
            plt.subplot(num_plots//2, 3, i+1)
            plot = plots[i * 2]
            label = plots[i * 2 + 1]
            self.plot_and_listen(s=plot, label=file+"-"+label, play=False)

        plt.savefig(self.image_dir + file+".png")
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
        
        label = label.replace(" ","_")

        if self.target is not None:
            label = label + "-tgt-" + self.tgt_class.replace(" ","_")
        
        fname = self.audio_dir + label + ".wav"
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

    def __call__(self, audio_index=None, tgt=None):

        # first time, no index given
        if audio_index is None and not self.recurse:
            w = self.audio[0] 
        
        # recurse, no index given
        elif audio_index is None and self.recurse:
            w = self.x 
        
        # do not recurse, specific index given
        else:
            # reset elapsed
            self.elapsed = 0
            w = self.audio[audio_index]
        
        # was a target provided?
        if tgt is not None:
            self.target = self.audio[tgt]

        self.x = self.dream(w, target=self.target)

        # enable recursion after first run
        self.recurse = True

# end DreamSound class

