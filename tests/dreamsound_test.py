import dreamsound

# audio_indices = [0]
# output_types  = [0,1,2,3,4]
# step_sizes    = [0.2]
# thresholds    = [0.5]
# flipped       = [True, False]
# argmax        = [True, False]
# targets       = [None, 1]
# win_lengths   = [2048]
# hop_lengths   = [128]
# fft_lengths   = [4096]

# cmd=[]
# for audio_index in audio_indices:
#     for ot in output_types:
#         for step_size in step_sizes:
#             for threshold in thresholds:
#                 for win_length in win_lengths:
#                     for fft_length in fft_lengths:
#                         for hop_length in hop_lengths:
#                             for flip in flipped:
#                                 for am in argmax:
#                                     for target in targets:
#                                         cmd.append([audio_index, ot, step_size, threshold, win_length, fft_length, hop_length, flip, am, target])


def test_all():

    # print(len(cmd) * ds.steps // ds.plot_every)

    # for i,c in enumerate(cmd):
        # ds = dreamsound.DreamSound(["../audio/original.wav","../audio/cat.wav"])
    #     ds.steps      = 100
    #     ds.plot_every = 50
    #     ds.save       = True
    #     ds.show       = False
    #     ds.play       = False
    #     ds.plot_every = 100
    #     ds.steps = 10
    #     print("-"*80)
    #     print("Iteration", i)
    #     print(f"""
    #     audio_index={str(c[0])}, 
    #     output_type={str(c[1])}, 
    #     step_size={str(c[2])}, 
    #     threshold={str(c[3])}, 
    #     win_length={str(c[4])}, 
    #     fft_length={str(c[5])}, 
    #     hop_length={str(c[6])}, 
    #     flip={str(c[7])}, 
    #     argmax={str(c[8])}, 
    #     target={str(c[9])}
    #     """)
    #     print("-"*80)
    #     ds.output_type  = c[1]
    #     ds.step_size    = c[2]
    #     ds.threshold    = c[3]
    #     ds.win_length   = c[4]
    #     ds.fft_length   = c[5]
    #     ds.hop_length   = c[6]
    #     ds(audio_index=c[0], target=c[9], argmax=c[8], flip=c[7]) 
    #     assert ds.elapsed == 10
    #     del ds

    ds = dreamsound.DreamSound(["../audio/original.wav","../audio/cat.wav"])

    # ds(audio_index=0, target=None, argmax=False, flip=False)
    # ds.steps = 10
    # assert ds.elapsed == 10

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