import dreamsound
    
ds = dreamsound.DreamSound(["../audio/original.wav", "../audio/cat.wav"])

ds.plot_every = 100
ds.steps = 10

ds(audio_index=0, tgt=1)

assert ds.elapsed == 10

ds.steps = 3
ds(audio_index=0)

assert ds.elapsed == 3

ds.steps = 5
ds()

assert ds.elapsed == 8

for i in range(6):
    print(i)
    ds.steps = 2
    ds.output_type = i
    ds(audio_index=1)
    assert ds.elapsed == 2

