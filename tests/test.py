import dreamsound
    
ds = dreamsound.DreamSound(["../audio/original.wav", "../audio/cat.wav"])

ds.plot_every = 100
ds.steps = 10

ds(audio_index=0, tgt=1)

assert ds.elapsed == 10
