###############################################################################
#
#  This file is part of the dreamsound package.
#  
#  DreamSound class definition
#
###############################################################################
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False
try:
    import yamnet, params
except ModuleNotFoundError: 
    from .downloader import YamnetDownloader
    yn = YamnetDownloader()()
    import yamnet, params
import numpy as np
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram as librosa_mel
from librosa.core import load as librosa_load

if IN_COLAB:
    from IPython.display import display
    from IPython.display import Audio
else:
    from os.path import exists
    from os import mkdir
    from os import system

TF_DTYPE, TF_CTYPE = tf.float64, tf.complex128

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
    play        : (bool) play audio file at `plot_every` steps (False)
    show        : (bool) show spectrogram plots at `plot_every` steps (False)
    save        : (bool) write PCM and PNG files at `plot_every` steps (True)
    power       : (float) amplitude or power spectrum (default 1) (1.0 / 8.0) 
    verbose     : (int) set verbosity level for printing (1)
    audio_dir   : (str) path to write audio files ("./audio/")
    image_dir   : (str) path to write image files ("./image/")
    class_file  : (str) path to yamnet class file ("yamnet_class_map.csv")
    weights_file: (str) path to yamnet pre-trained weights ("yamnet.h5")

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
    hard_resize(x, y)
        make both tensors the size of the minimum
    combine_1(x, y)
        one type of spectral combination
    combine_2(x, y)
        yet another spectral combination
    combine_3(x, y, target)
        yet another spectral combination with a target class
    calc_loss(data)
        get loss function
    class_from_audio(waveform)
        get the class of an audio file
    dream(source, tgt=None)
        perform the dreaming. if a tgt is specified, use that as filter
    plotter(plots, file)
        wrapper to plot using suplots
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
    power       = 1.0 / 8.0
    verbose     = 1
    play        = False
    show        = False
    save        = True
    audio       = []
    audio_dir   = "./audio/"
    image_dir   = "./image/"
    class_file  = "yamnet_class_map.csv"
    weights_file= "yamnet.h5"

    def __init__(self, paths=None, layer=None):

        self.dreamer = self.load_model(layer)
        
        if paths is not None:
            self.load_audio(paths)

        if "mkidr" in dir():
            if not exists(self.audio_dir): mkdir(self.audio_dir)
            if not exists(self.image_dir): mkdir(self.audio_dir)

    def __call__(self, audio_index=None, target=None):

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
        if target is not None:
            self.target = self.audio[target]

        self.x = self.dream(w, target=self.target)

        # enable recursion after first run
        self.recurse = True

    def __print__(self, msg):
        if self.verbose:
            print(msg)
        else:
            pass

    def load_model(self, layer=None):
        """
        This function loads the yamnet model with a specified layer and
        returns a 'dreamer' model that returns the activations of such layer
        
        Parameters
        -----------
        layer (string) : a specified layer

        If `layer` is not specified, the last layer is used instead.

        Returns
        ----------
        (tf.keras.Model) : the dreamer model
        
        """

        # load its class names
        self.class_names = yamnet.class_names(self.class_file)
        self.class_names_tensor = tf.constant(self.class_names)
        # load model parameters and get model
        self.params = params.Params(
                                    sample_rate=self.sr,
                                    patch_hop_seconds=self.patch_hop
                                    )
        self.model = yamnet.yamnet_frames_model(self.params)
        # load model weigths
        self.model.load_weights(self.weights_file)
        if layer is not None:
            self.layername = layer
        else:
            self.__print__("Using last layer.")
            self.layername = self.model.layers[-1].name
        self.__print__(f"Yamnet loaded, using layer:{self.layername}")
        # Get the specified layer
        self.layers = self.model.get_layer(self.layername).output
        # Finally, create the dreamer model
        self.dreamer = tf.keras.Model(
                                      inputs  = self.model.input, 
                                      outputs = self.layers
                                      )
        self.__print__("Dreamer started.")
        return self.dreamer

    def load_audio(self, paths):
        """loads the array of sound files using librosa
        """
        
        self.__print__("Loading audio files...")
        for p in paths:
            y,_ = librosa_load(p, sr=self.sr, duration=self.max_dur)
            self.audio.append(y)
        self.__print__("Done.")
        self.__print__(f"DreamSound has {len(self.audio)} audio files in memory.")

    def clear_audio(self):
        del self.audio
        self.audio = []
        self.__print__(f"DreamSound {len(self.audio)} audio files in memory.")

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
            self.__print__(f"Can't parse: {type(classid)}")
    
    def summary(self):
        self.model_name.summary()
    
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
        ]
    )
    def hard_resize(self, x, y):
        if tf.shape(x)[0] < tf.shape(y)[0] : y=tf.slice(y,[0],[tf.shape(x)[0]])
        if tf.shape(y)[0] < tf.shape(x)[0] : x=tf.slice(x,[0],[tf.shape(y)[0]])
        return x, y


    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_DTYPE)]
    )
    def stft(self, x):
        return tf.signal.stft(x,
                              self.win_length,
                              self.hop_length,
                              pad_end=self.pad_end)
        
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_CTYPE)]
    )
    def istft(self, x):
        return tf.signal.inverse_stft(x, 
                                      self.win_length, 
                                      self.hop_length)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=TF_CTYPE),
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
        ]
    )
    def complex_mul(self, x, y):
        x_imag = tf.math.real(x)
        x_real = tf.math.imag(x)
        real, y = self.hard_resize(x_real,y)
        imag, y = self.hard_resize(x_imag,y)
        conv = tf.dtypes.complex(tf.math.multiply(real,y),
                                 tf.math.multiply(imag,y))
        return conv

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
        ]
    )
    def combine_1(self, x, y):
        
        # take short time fourier transform
        X_mag, X_pha = self.magphase(self.stft(x))
        Y_mag, Y_pha = self.magphase(self.stft(y))
        
        # get rid of tiny values
        Y_mag *= ( 1 + tf.math.sign(Y_mag - self.threshold) ) * 0.5
        Y_mag *= self.step_size

        # multiply x by y
        X_mag *= Y_mag

        # take inverse fourier
        X_real = self.istft(self.complex_mul(X_pha, X_mag)) 

        return X_real    
    
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
        ]
    )
    def combine_2(self, x, y):
        
        # x is filtered
        # y is the filter if target is none

        X = self.stft(x)
        Y = self.stft(y)
        y_filter, y_pha = self.magphase(Y)

        norm_y_filter = y_filter / tf.math.reduce_max(y_filter)
        y_filter_thresh = norm_y_filter - self.threshold

        y_filter *= (tf.math.sign(y_filter_thresh) + 1) * 0.5
        y_filter *= self.step_size

        X_y_filtered = self.complex_mul(X, y_filter)

        x_real = self.istft(X_y_filtered)
        x_real, y = self.hard_resize(x_real, y)
        combined = tf.math.add(x_real, y)
        y_filter_real = self.istft(self.complex_mul(y_pha,y_filter))
 
        return combined, x_real, y_filter_real
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_CTYPE)]
    )
    def phase(self, x):
        im = tf.constant(1.0, dtype=TF_DTYPE)
        return tf.complex(tf.math.exp(tf.math.angle(x)), im)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_CTYPE)]
    )
    def magphase(self, x):
        return tf.math.pow(tf.math.abs(x),self.power), self.phase(x)


    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
        ]
    )
    def combine_3(self, x, y, target):
        
        # x is filtered
        # y is a mask
        # target is the filter

        X_mag, X_pha = self.magphase(self.stft(x))
        Y_mag, Y_pha = self.magphase(self.stft(y))
        T_mag, T_pha = self.magphase(self.stft(target))

        # filter out the target's magnitude with a mask
        T_mag_norm = T_mag / tf.math.reduce_max(T_mag)
        mask = T_mag_norm - self.threshold
        T_mag *= (tf.math.sign(mask) + 1) * 0.5 * self.step_size

        # filter out y's magnitude with a mask
        Y_mag_norm = Y_mag / tf.math.reduce_max(Y_mag)
        mask = Y_mag_norm - self.threshold
        Y_mag *= (tf.math.sign(mask) + 1) * 0.5 * self.step_size

        # combine all magnitudes and add phase of X
        combined = (X_mag * T_mag) 
        combined = tf.math.pow(combined, 1/self.power)
        # combined += Y_mag

        output = self.istft(self.complex_mul(Y_pha, combined))
        output, y = self.hard_resize(output, y)
        output += y
        
        T_real = self.istft(self.complex_mul(T_pha, T_mag))

        Y_real = self.istft(self.complex_mul(Y_pha, Y_mag))
 
        return output, Y_real, T_real

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_DTYPE)]
    )
    def calc_loss(self, wavetensor):

        act, argmax = self.class_from_audio(wavetensor)
        
        def a(): return tf.math.reduce_sum(act[:,self.classid])
        def b(): return tf.math.reduce_sum(act[:,argmax])
        
        loss = tf.cond(self.use_target, a, b) 
        loss **= self.loss_power

        return loss, self.class_names_tensor[argmax]

        
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_DTYPE)]
    )
    def class_from_audio(self, wavetensor):
        # Pass forward the data through the model to retrieve the activations.
        layer_activations = self.dreamer(wavetensor)
        reduced = tf.math.reduce_mean(layer_activations, axis=0)
        argmax = tf.math.argmax(reduced)
        return layer_activations, argmax

    def clip_or_pad(self, x, y):
        """Clips or Pads X based on the size of Y

        If x is greater than y, then clip x based on y
        if x is smaller than y, then pad with zeros to match y
        """
        xdim = x.shape[0]
        ydim = y.shape[0]
        if xdim >= ydim:
            x = x[:ydim]
        else:
            pad = np.zeros((ydim-xdim,y.shape[1]))
            x = np.concatenate([x, pad])
        return x

    def dream(self, source, target=None):
        
        wt   = tf.convert_to_tensor(source, dtype=TF_DTYPE)
        wr_  = None
        wf_  = None
        loss = tf.constant(0.0)

        if target is not None:
            target = self.clip_or_pad(target,source)
            target = tf.convert_to_tensor(target, dtype=TF_DTYPE)
            _, self.classid = self.class_from_audio(target)
            self.tgt_class = self.class_names[self.classid]
            self.use_target = tf.constant(True, dtype=tf.bool)
            if self.verbose:
                tf.print(f"Target class: { self.tgt_class } ...")
        else:
            self.use_target = tf.constant(False, dtype=tf.bool)

        # begin loop
        for i in tf.range(self.steps):

            # get the gradients
            with tf.GradientTape() as tape:
                tape.watch(wt)
                loss, c = self.calc_loss(wt)
           
            if self.verbose:
                tf.print(f"Running step {self.elapsed}, class: {c}...")

            g_  = tape.gradient(loss, wt)
            g_ /= tf.math.reduce_std(g_)

            # combine spectra
            if self.output_type == 0:
                wt = self.combine_1(wt, g_)

            elif self.output_type == 1:
                wt += (g_ * self.step_size)

            elif self.output_type == 2:
                wt = self.combine_1(g_, wt)

            elif self.output_type == 3:
                if target is None:
                    wt, wr_, wf_ = self.combine_2(g_, wt)
                else:
                    wt, wr_, wf_ = self.combine_3(g_, wt, target)

            elif self.output_type == 4:
                if target is None:
                    wt, wr_, wf_ = self.combine_2(wt, g_)
                else:
                    wt, wr_, wf_ = self.combine_3(wt, g_, target)

            elif self.output_type == 5:
                # return gradients only
                wt = g_
            else:
                # return only wavetensor
                wt = wt

            # plot and save
            if (i+1) % self.plot_every == 0 and i > 0:
                
                # store data in self arrays          
                w_min, s_min    = self.hard_resize(wt,source)
                self.difference = tf.math.subtract(w_min, s_min)
                self.difference = self.difference.numpy()
                self.gradients  = g_.numpy()
                self.wavetensor = wt.numpy()
                
                # plot
                plots = [
                    {0:"orig", 1:self.wavetensor},
                    {0:"diff", 1:self.difference},
                    {0:"grad", 1:self.gradients}
                ]

                if wr_ is not None:
                    self.inverse = tf.transpose(wr_).numpy()
                    plots.append({0:"filt", 1:self.inverse})
                if wf_ is not None:
                    self.filter = tf.transpose(wf_).numpy()
                    plots.append({0:"hard", 1:self.filter})
                
                self.plot_and_save(
                    plots = plots, 
                    file  = f"{c}-{self.output_type}-{self.elapsed}"
                )

            # end main loop
            self.elapsed += 1 # increment count

        return wt.numpy()

    def plot_and_save(self, plots, file):

        image_file = self.image_dir + file + ".png"

        fig = plt.figure(figsize=self.figsize)

        for i in range(len(plots)):
            rows = len(plots) // 2
            cols = 3
            plt.subplot(rows, cols, i + 1)
            
            waveform  = plots[i][1]
            label = file.replace(" ","_") + "-" + plots[i][0]

            mel_spec = librosa_mel(waveform)
            mel_spec = np.log10(np.maximum(1e-05, mel_spec)) * 10.0
            mel_spec = np.maximum(mel_spec, mel_spec.max()-self.top_db)
            
            plt.imshow(mel_spec, origin="lower", aspect="auto")
            plt.title(label)
            
            if self.target is not None:
                label += "-tgt-" + self.tgt_class.replace(" ","_")
            
            audio_file = self.audio_dir + label + ".wav"
            
            if self.save:
                self.__print__(f"Writing {audio_file} ...")
                sf.write(audio_file, waveform, self.sr, subtype="PCM_24")
        
        if self.save:
            self.__print__(f"Writing {image_file} ...")
            plt.savefig(image_file)

        if self.play:
            if IN_COLAB:
                display(Audio(waveform, rate=self.sr))
            else:
                system(f"ffplay -autoexit {audio_file}")
        
        if self.show:
            plt.show()

        plt.close()

    def print(self):
        print(vars(self))

# end DreamSound class