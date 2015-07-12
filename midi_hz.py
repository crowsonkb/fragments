import numpy as np

def hz_to_midi(x):
    return 69 + 12*np.log2(x/440)

def midi_to_hz(x):
    return 440 * 2**((x-69)/12)
