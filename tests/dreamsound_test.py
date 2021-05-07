import dreamsound

def test_all():
    ds = dreamsound.DreamSound(["../audio/original.wav", "../audio/cat.wav"])

    ds.plot_every = 100
    ds.steps = 10

    ds(audio_index=0, target=1)

    assert ds.elapsed == 10

    ds.steps = 3
    ds(audio_index=0)

    assert ds.elapsed == 3

    ds.steps = 5
    ds()

    assert ds.elapsed == 8

    for i in range(6):
        print(i, 'auto')
        ds.steps = 2
        ds.output_type = i
        ds(audio_index=1)
        assert ds.elapsed == 2


    print(3, 'targetted')
    ds.steps = 2
    ds.output_type = 3
    ds(audio_index=0, target=1)
    assert ds.elapsed == 2

test_all()