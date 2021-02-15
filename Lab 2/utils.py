import numpy as np
import scipy.io.wavfile  as wavfile
import os

# The following function generates a continuous time sinusoid
# given the amplitude A, F (cycles/seconds), Fs=sampling rate, start and endtime
def fnGenSampledSinusoid(A,Freq,Phi,Fs,sTime,eTime):
    # Showing off how to use numerical python library to create arange
    n = np.arange(sTime,eTime,1.0/Fs)
    y = A*np.cos(2 * np.pi * Freq * n + Phi)
    return [n,y]

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))

def convolve(x, h):
    #Our function follow the numpy convolution with no truncation
    #First step is to get the final array with size as x.size + h.size -1
    x_length = np.size(x)
    h_length = np.size(h)

    y = np.zeros(x_length + h_length - 1)

    #As we loop through the range of 1 signal at each sample index
    #Create the other signal starting at that sample index
    #Once created, sum all the amplitude of all the signals at that particular index
    #Here we can do it directly in the same for loop sinze we initialized the result array to 0
    for i in np.arange(x_length):
        for j in np.arange(h_length):
            y[i + j] = y[i + j] + x[i] * h[j]
        #print("Loop (on x_signal): ", i)

    #print("Done")
    return y

def save_sound(file_name, sampling_frequency, bits):
    wavfile.write(file_name, sampling_frequency, bits)

def play_sound(file_name):
    os.system("aplay " + file_name)

def read_sound(file_name):
    return wavfile.read(file_name)

def delta(n):
    if n == 0:
        return 1
    else:
        return 0