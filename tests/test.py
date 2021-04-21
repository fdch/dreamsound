import dreamsound as d

def test():
    
    ds = d.dreamsound.DreamSound(["../audio/original.wav"])
    ds.plot_every = 40
    ds.steps = 40
    ds()

test()