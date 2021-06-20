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
    3 - Filter the gradient with the original sound
    5 - return gradients only
    
    anything else will only recurse on the original audio
    """
    sr          = 22050
    max_dur     = 10
    patch_hop   = 0.1
    fft_length  = 4096
    win_length  = 2048
    hop_length  = 128
    pad_end     = False
    loss_power  = 0.001
    plot_every  = 10
    figsize     = (10,8)
    top_db      = 80.0
    step_size   = 0.2
    output_type = 0
    steps       = 10
    threshold   = 0.5
    classid     = None
    elapsed     = 0
    recurse     = False
    argmax      = False
    flip        = False
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
        
        self.i_window_fn = tf.signal.inverse_stft_window_fn(self.hop_length)


    def __call__(self, audio_index=None, target=None, argmax=False, flip=False):

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

        self.argmax = argmax
        self.flip = flip

        self.x = self.dream(w, target=self.target, argmax=self.argmax)

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
        if len(self.audio) > 0:
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
                              frame_length=self.win_length,
                              frame_step=self.hop_length,
                              fft_length=self.fft_length,
                              pad_end=self.pad_end)
        
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_CTYPE)]
    )
    def istft(self, x):
        return tf.signal.inverse_stft(x,
                              frame_length=self.win_length,
                              frame_step=self.hop_length,
                              fft_length=self.fft_length)

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
            tf.TensorSpec(shape=None, dtype=TF_DTYPE)
        ]
    )
    def normalize(self, x):
        """Normalize x with x.max()
        """
        return x / tf.math.reduce_max(x)


    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
        ]
    )
    def combine_2(self, x, y):
        """Filter the gradient with the original sound (keeping its phase)

        Filtra el gradiente con el sonido original
        Usando la phase del sonido original
        
        Description
        -----------

        1. The hard-cut filter is made of the magnitude of the original sound, 
        - 1. normalizing the magnitude of the original sound
        - 2. offsetting the magnitude down by `threshold`
        - 3. hard-cutting the magnitude to values 0 or 1 depending on its sign.
        
        2. Apply the hard-cut filter to the magnitude of the gradient by 
        - 1. complex multiplication, and 
        - 2. rephase with the original sound's phase

        3. Inverse FFT
        - 1. Compute the inverse sftf of the filtered gradient, and 
        - 2. Add the attenuated gradients (by `step_size`) to the (real) original sound
        - 3. compute the inverse stft of the hard-cut filter (and rephase)
        
        Parameters
        -----------
        x = gradient
        y = original sound
    
        Returns
        -----------
        return output, hard_cut_real

        output        =   the new gradient with the added sound
        hard_cut_real =   the hard cut filter

        """
        
        X = self.stft(x)
        Y = self.stft(y)
        X_mag, X_pha = self.magphase(X)
        Y_mag = tf.math.abs(Y)

        # normalize
        Y_mag_norm = self.normalize(Y_mag)
        # offset
        Y_mag_offset = Y_mag_norm - self.threshold
        # hard cut based on sign
        hard_cut = (tf.math.sign(Y_mag_offset) + 1) * 0.5
        # apply the hard-cut filter to the magnitude of the gradient
        X_mag_cut = hard_cut * X_mag
        # apply the phase of the original sound to the filtered gradient mag
        X_mag_rephased = self.complex_mul(X_pha, X_mag_cut)
        # compute the inverse stft on the cut and rephased magnitude
        x_new = self.istft(X_mag_rephased)
        # resize either x_new or y to min length so that we can add them
        x_new, y = self.hard_resize(x_new, y)
        # add a small amount of the sound to the new (real) gradient 
        output = tf.math.add(x_new * self.step_size, y)
        # inverse fft of the hard cut
        hard_cut_real = self.istft(self.complex_mul(X_pha,hard_cut))
 
        return output, hard_cut_real

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_CTYPE)]
    )
    def magphase(self, x):
        """Return the magnitude and the phase of x
        """
        a = tf.math.abs(x)
        ang = tf.math.angle(x)
        p = tf.complex(tf.math.cos(ang), tf.math.sin(ang))
        return tf.math.pow(a,self.power), p

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
            tf.TensorSpec(shape=None, dtype=TF_DTYPE),
        ]
    )
    def combine_3(self, x, y, target):
        
        X = self.stft(x)

        X_mag = tf.math.abs(X)
        Y_mag, Y_pha = self.magphase(self.stft(target))

        # normalize
        Y_mag_norm = self.normalize(Y_mag)
        # offset
        Y_mag_offset = Y_mag_norm - self.threshold
        # hard cut based on sign
        hard_cut = (tf.math.sign(Y_mag_offset) + 1) * 0.5
        # here we can either 
        # a. apply the hard cut to the magnitude: `hard_cut *= Y_mag`, or
        # b. keep the hard_cut as is: `hard_cut = hard_cut`
        # the former lets the original sound in, the latter does not
        # we are going with 'a'
        # hard_cut *= Y_mag
        # apply the hard-cut filter to the magnitude of the gradient
        X_mag_cut = hard_cut * X_mag
        # apply the phase of the original sound to the filtered gradient mag
        X_mag_rephased = self.complex_mul(Y_pha, X_mag_cut)
        # compute the inverse stft on the cut and rephased magnitude
        x_new = self.istft(X_mag_rephased)
        # resize either x_new or y to min length so that we can add them
        x_new, y = self.hard_resize(x_new, y)
        # add a small amount of the sound to the new (real) gradient 
        output = tf.math.add(x_new * self.step_size, y)
        # inverse fft of the hard cut
        hard_cut_real = self.istft(self.complex_mul(Y_pha,hard_cut))
 
        return output, hard_cut_real

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_DTYPE)]
    )
    def calc_loss(self, wavetensor):

        act, argmax, losses = self.class_from_audio(wavetensor)
        
        def classact(): return tf.math.reduce_sum(act[:,self.classid])
        def argmaxact(): return tf.math.reduce_sum(act[:,argmax])
        def sumlosses(): return tf.math.reduce_sum(losses)
        def notarg(): return tf.cond(self.use_argmax, argmaxact, sumlosses)

        loss = tf.cond(self.use_target, classact, notarg) 
        loss **= self.loss_power

        return loss, argmax

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_DTYPE)]
    )
    def calc_gradients(self, input_tensor):
        
        # get the normalized gradients
        
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(input_tensor)
            loss, argmax = self.calc_loss(input_tensor)

        gradients  = tape.gradient(loss, input_tensor)
        gradients /= tf.math.reduce_std(gradients) + 1e-8 

        return gradients, argmax

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=TF_DTYPE)]
    )
    def class_from_audio(self, wavetensor):
        # Pass forward the data through the model to retrieve the activations.
        layer_activations = self.dreamer(wavetensor)
        reduced = tf.math.reduce_mean(layer_activations, axis=0)
        argmax = tf.math.argmax(reduced)
        losses = tf.map_fn(lambda x:tf.math.reduce_mean(x),layer_activations)
        return layer_activations, argmax, losses

    # def clip_or_pad(self, x, y):
    #     """Clips or Pads X based on the size of Y

    #     If x is greater than y, then clip x based on y
    #     if x is smaller than y, then pad with zeros to match y
    #     """
    #     xdim = x.shape[0]
    #     ydim = y.shape[0]
    #     if xdim >= ydim:
    #         x = x[:ydim]
    #     else:
    #         pad = np.zeros((ydim-xdim,y.shape[1]))
    #         x = np.concatenate([x, pad])
    #     return x

    def dream(self, source, target=None, argmax=False):
        
        wt   = tf.convert_to_tensor(source, dtype=TF_DTYPE)
        wr_  = None
        wf_  = None
        loss = tf.constant(0.0)

        if target is not None:
            target = tf.convert_to_tensor(target, dtype=TF_DTYPE)
            target, wt = self.hard_resize(target, wt)
            _, self.classid, _ = self.class_from_audio(target)
            self.tgt_class = self.class_names[tf.math.reduce_max(self.classid)]
            self.use_target = tf.constant(True, dtype=tf.bool)
            if self.verbose:
                tf.print(f"Target class: { self.tgt_class } ...")
        else:
            self.use_target = tf.constant(False, dtype=tf.bool)
        
        self.use_argmax = tf.constant(argmax, dtype=tf.bool)


        # begin loop
        for i in tf.range(self.steps):

            # get the gradients and the class name
            wt_g, wt_argmax = self.calc_gradients(wt)

            wt_c = self.class_names_tensor[wt_argmax]
            
            if self.verbose:
                tf.print(f"Running step {self.elapsed}, class: {wt_c} ...")
            
            
            # Activation Maximization Function Options

            if self.output_type == 0:
                if self.flip:
                    wt_g += (wt * self.step_size)
                else:
                    wt += (wt_g * self.step_size)

            

            elif self.output_type == 1:
                if self.flip:
                    wt = self.combine_1(wt_g, wt)
                else:
                    wt = self.combine_1(wt, wt_g)

            

            elif self.output_type == 2:
                if target is None:
                    if self.flip:
                        wt, wf_ = self.combine_2(wt, wt_g)
                    else:
                        wt, wf_ = self.combine_2(wt_g, wt)
                else:
                    if self.flip:
                        wt, wf_ = self.combine_3(wt_g, wt, target)
                    else:
                        wt, wf_ = self.combine_3(wt, wt_g, target)

            

            elif self.output_type == 3:
                if target is None:
                    # return gradients only
                    wt = wt_g
                else:
                    # take the gradient of the target loss 
                    # and point towards an obscure direction
                    tgt_g_, tgt_argmax = self.calc_gradients(target)

                    tgt_c = self.class_names_tensor[tgt_argmax]
                    
                    if self.verbose:
                        tf.print(f"Target class: {tgt_c} ...")

                    if self.flip:
                        tgt_wt, wf_ = self.combine_3(target, wt_g, tgt_g_)
                    else:
                        tgt_wt, wf_ = self.combine_3(wt_g, target, tgt_g_)
                    
                    tgt_wt, wt = self.hard_resize(tgt_wt, wt)
                    
                    wt = tf.math.add(wt, 
                            tf.math.multiply(tgt_wt,
                                self.step_size))
            
            
            else:
                # return only wavetensor, unchanged
                wt = wt

                # if target is not None:
                #     # take the gradient of the target loss against the input
                #     # and point in that direction
                #     with tf.GradientTape() as tgt_tape:
                #         tgt_tape.watch([wt, target])
                #         wt = wt
                #         tgt_loss, tgt_argmax = self.calc_loss(target)

                #     tgt_c = self.class_names_tensor[tgt_argmax]
                    
                #     if self.verbose:
                #         tf.print(f"Target class: {tgt_c} ...")
                    
                #     tgt_g_ = tgt_tape.gradient(tgt_loss, wt)
                #     tgt_g_ /= tf.math.reduce_std(tgt_g_) + 1e-8
                    
                #     tgt_g_, wt = self.hard_resize(tgt_g_, wt)

                #     wt = tf.math.add(wt, 
                #             tf.math.multiply(tgt_g_,
                #                 self.step_size))

            
            
            # plot and save
            

            if (i+1) % self.plot_every == 0 and i > 0:
                
                # store data in self arrays          
                w_min, s_min    = self.hard_resize(wt,source)
                self.difference = tf.math.subtract(w_min, s_min)
                self.difference = self.difference.numpy()
                self.gradients  = wt_g.numpy()
                self.wavetensor = wt.numpy()
                
                # plot
                plots = [
                    {0:"orig", 1:self.wavetensor},
                    {0:"diff", 1:self.difference},
                    {0:"grad", 1:self.gradients}
                ]

                if wf_ is not None:
                    self.filter = tf.transpose(wf_).numpy()
                    plots.append({0:"hard", 1:self.filter})
                
                self.plot_and_save(
                    plots = plots, 
                    file  = f"{wt_c}-{self.output_type}-{self.elapsed}"
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
                display(Audio(self.wavetensor, rate=self.sr))
            else:
                system(f"ffplay -autoexit {audio_file}")
        
        if self.show:
            plt.show()

        plt.close()

    def print(self):
        print(vars(self))

# end DreamSound class