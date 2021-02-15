import numpy as np
import time
import matplotlib.pyplot as plt
from utils import convolve, play_sound, fnNormalize16BitToFloat, read_sound, fnNormalizeFloatTo16Bit, save_sound, delta, fnGenSampledSinusoid
from scipy import signal

ipcleanfilename = '/home/boom/CE3007/Lab1_Manual/Lab1_Example/testIp_16bit.wav'

def q2_5_f():
    x = np.zeros(10, dtype=float)
    x[0] = 1
    B = [1, -0.7653668, 0.99999]
    A = [1, -0.722744, 0.888622]
    y_ifil = signal.lfilter(B, A, x)
    print(y_ifil)

"""
Generate the output again and compare output vs input
Plot all kind of things to study the relationship
"""
def q2_5_d():
    #Load file
    ipnoisyfilename = '/home/boom/CE3007/Lab2_Manual/Lab2_Example/helloworld_noisy_16bit.wav'
    _, sampleX_16bit = read_sound(ipnoisyfilename)
    #To float
    x = fnNormalize16BitToFloat(sampleX_16bit)
    #Coeff for filter
    B = [1, -0.7653668, 0.99999]
    A = [1, -0.722744, 0.888622]

    #Filtering
    y_ifil = signal.lfilter(B, A, x)
    y_ifil_16bit = fnNormalizeFloatTo16Bit(y_ifil)
    y_ifil_name = 'y_ifil.wav'
    save_sound(y_ifil_name, 16000, y_ifil_16bit)
    play_sound(y_ifil_name)

    #Plotting Spectrogram
    [f1, t1, Sxx1] = signal.spectrogram(x, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    [f2, t2, Sxx2] = signal.spectrogram(y_ifil, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    _, ax = plt.subplots(2, 1)

    ax[0].pcolormesh(t1, f1, 10*np.log10(Sxx1))
    ax[0].set_ylabel('Frequency [Hz]')
    ax[0].set_xlabel('Time [sec]')
    ax[0].set_title('spectrogram of output signal y1')

    ax[1].pcolormesh(t2, f2, 10*np.log10(Sxx2))
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [sec]')
    ax[1].set_title('spectrogram of output signal y2')
    plt.show()

"""
Given the difference equation
Calculate the output based on the input with a look up table
"""
def q2_5_c():
    #Load sound file
    ipnoisyfilename = '/home/boom/CE3007/Lab2_Manual/Lab2_Example/helloworld_noisy_16bit.wav'
    _, sampleX_16bit = read_sound(ipnoisyfilename)
    #Convert to float
    x = fnNormalize16BitToFloat(sampleX_16bit)
    
    #Output should be same size as input
    y = np.zeros(len(x), dtype=float)

    #Filter coeff
    B = [1, -0.7653668, 0.99999]
    A = [1, -0.722744, 0.888622]

    #Loop through input x axis space (and not the output one just in case)
    #Assume system is relax at n < 0
    for n in range(len(x)):
        if n == 0:
            #Only input, only output, the rest of the term are all 0 
            y[n] = 1 * x[n]
        elif n == 1:
            #Only input at n=1, output at n=0 (in delayed term), input at n=0(in delayed term)
            y[n] = 1 * x[n] + (-0.7653668) * x[n - 1] - (-0.722744) * y[n - 1]
        else:
            #Should have all the necessary input, input delayed once and delayed twice
            #Should calculate based on delayed output once and twice
            y[n] = 1 * x[n] + (-0.7653668) * x[n - 1] + 0.99999 * x[n - 2] - (-0.722744) * y[n - 1] - 0.888622 * y[n - 2]
    
    #For comparison
    y_ifil = signal.lfilter(B, A, x)
    print(len(y_ifil))
    print(len(y))

    #Check
    for i in range(len(y)):
        if y[i] == y_ifil[i]:
            print("Very Good!")
            continue
        else:
            print("=========WARN========")
            print("Ours: ",y[i])
            print("IFIL: ",y_ifil[i])
            print("=====================")

"""
Given a noise corrupted audio file, plot the spectrogram
"""
def q2_5_a():
    #Load the sound file
    ipnoisyfilename = '/home/boom/CE3007/Lab2_Manual/Lab2_Example/helloworld_noisy_16bit.wav'
    _, sampleX_16bit = read_sound(ipnoisyfilename)
    x = fnNormalize16BitToFloat(sampleX_16bit)
    
    #Spectrogram stuffs (always in float)
    [f, t, Sxx] = signal.spectrogram(x, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    _, ax = plt.subplots(2, 1)
    t1 = np.arange(0, len(x), 1) * (1 / 16000)

    ax[0].plot(t1, x)
    ax[0].grid()
    ax[0].set_title('Signal in time domain')
    ax[0].set_xlabel('Time(s)')
    ax[0].set_ylabel('Amplitude')

    ax[1].pcolormesh(t, f, 10*np.log10(Sxx))
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [sec]')
    ax[1].set_title('spectrogram of noisy signal')

    plt.show()

""" In this question part:
- We will be passing a compound sinuisodal signal through the above two system
- Use spectrogram to investigate in the frequency domain
- Effect of filtering by each system can be observed 
"""
def q2_4_c():
    #Given system characteristics
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype='float')
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523], dtype='float')

    #Generate the signals
    #Amplitude, frequency (Hz), sampling frequency, start time, stop time
    #Meant to generate for 1 sec but may cut down so easier to verify
    _, x1 = fnGenSampledSinusoid(0.1, 700, 0, 16000, 0, 1.0 + 1/16000)
    _, x2 = fnGenSampledSinusoid(0.1, 3333, 0, 16000, 0, 1.0 + 1/16000)

    #Superpose them together
    x = x1 + x2

    #Obtain the output through the two systems and from there, get the time
    y1 = np.convolve(x, h1)
    t1 = np.arange(0, len(y1), 1) * (1 / 16000) #in seconds since period 1 sample = 1/16000 s
    y2 = np.convolve(x, h2)
    t2 = np.arange(0, len(y2), 1) * (1 / 16000) #in seconds since period 1 sample = 1/16000 s

    #Spectrogram of input
    [f, t, Sxx] = signal.spectrogram(x, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    plt.pcolormesh(t, f, 10*np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('spectrogram of input signal')
    plt.show()

    ############################################
    ####### Investigate the output of system ###
    ############################################

    #Plot the output of the system (time domain)
    _, ax = plt.subplots(2, 1)
    ax[0].plot(t1, y1)
    ax[0].grid()
    ax[0].set_title("Output y1")
    ax[1].plot(t2, y2)
    ax[1].grid()
    ax[1].set_title("Output y2")
    plt.show()

    #Spectrogram of generated output (frequency domain)
    [f1, t1, Sxx1] = signal.spectrogram(y1, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    [f2, t2, Sxx2] = signal.spectrogram(y2, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    _, ax = plt.subplots(2, 1)

    ax[0].pcolormesh(t1, f1, 10*np.log10(Sxx1))
    ax[0].set_ylabel('Frequency [Hz]')
    ax[0].set_xlabel('Time [sec]')
    ax[0].set_title('spectrogram of output signal y1')

    ax[1].pcolormesh(t2, f2, 10*np.log10(Sxx2))
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [sec]')
    ax[1].set_title('spectrogram of output signal y2')

    plt.show()

    #Audio stuffs
    x_16bit = fnNormalizeFloatTo16Bit(x)
    y1_16bit = fnNormalizeFloatTo16Bit(y1)
    y2_16bit = fnNormalizeFloatTo16Bit(y2)
    x_file_name = 'x_16bit.wav'
    y1_file_name = 'y1_16bit.wav'
    y2_file_name = 'y2_16bit.wav'
    save_sound(x_file_name, 16000, x_16bit)
    save_sound(y1_file_name, 16000, y1_16bit)
    save_sound(y2_file_name, 16000, y2_16bit)
    play_sound(x_file_name)
    play_sound(y1_file_name)
    play_sound(y2_file_name)

"""
Implement filter function with arguments:
- Impulse response h
- Input signal x
- Return signal y

Making use of our convolution maybe?
"""
def q2_4_b():
    # Update Impulse response here if needed
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype='float')
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523], dtype='float')

    #Range of sample index (0 to 16)
    #Create the input signal
    n = np.arange(0, 16)
    x = np.zeros(len(n))
    for i in range(len(n)):
        #At n[15]=15, delta(15-15) = delta(0)
        x[i] = delta(n[i]) - 2 * delta(n[i] - 15)

    #Sanity check of input signal with a stem plot
    plt.figure(1)
    plt.grid()
    plt.stem(n,x,'g-o',use_line_collection=True)
    plt.title("Input signal")

    #Result for comparison
    results_1_1 = convolve(x, h1)[0:len(n)]
    results_1_2 = np.convolve(x, h1)[0:len(n)]
    results_1_3 = signal.lfilter(h1, [1], x)
    print(results_1_1)
    print(results_1_2)
    print(results_1_3)

    plt.figure(2)
    plt.grid()
    plt.stem(results_1_1,use_line_collection=True)
    plt.title("Output y1 signal")

    results_2_1 = convolve(x, h2)[0:len(n)]
    results_2_2 = np.convolve(x, h2)[0:len(n)]
    results_2_3 = signal.lfilter(h2, [1], x)
    print(results_2_1)
    print(results_2_2)
    print(results_2_3)

    plt.figure(3)
    plt.grid()
    plt.stem(results_2_1,use_line_collection=True)
    plt.title("Output y2 signal")

    plt.show()
    
"""
Given two filters and their impulse response, plot it out
"""
def q2_4_a():
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype='float')
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523], dtype='float')
    _, ax = plt.subplots(2, 1)
    ax[0].stem(h1)
    ax[0].grid()
    ax[1].stem(h2)
    ax[1].grid()
    plt.show()

"""
Using our convolution method defined in part A to generate an output signal given the impulse response echo
Save the file and investigate the result
"""
def q2_3_b():
    #Define the impulse response specs here
    impulseH = np.zeros(8000)
    impulseH[1] = 1
    impulseH[4000] = 0.5
    impulseH[7900] = 0.3
    
    #Clean file
    play_sound(ipcleanfilename)

    #Read in the sound, convert to float
    Fs, sampleX_16bit = read_sound(ipcleanfilename)
    sampleX_float = fnNormalize16BitToFloat(sampleX_16bit)
    
    #print(sampleX_float.shape)

    #Filter with our convolution
    start = time.time()
    y = np.convolve(sampleX_float, impulseH)
    print("Time: ",time.time()-start)
    #Downsampled and save
    y_16bit = fnNormalizeFloatTo16Bit(y)
    save_file_name = "/home/boom/CE3007/Labs/t2_16bit.wav"
    save_sound(save_file_name, Fs, y_16bit)
    play_sound(save_file_name)
    

"""
In this question, an impules response that following echo response (a spike delta every far intervals)
The routine to do convolution by hand is located in utils.py
Explanation on how to do it also there
"""
def q2_3_a():
    # # Assumed finite with this number of sample index
    impulseH = np.zeros(8000)
    impulseH[1] = 1
    impulseH[4000] = 0.5
    impulseH[7900] = 0.3

    # Plotting the impulse response / that characterise the lti system/ filter.
    plt.stem(impulseH)
    plt.grid()
    plt.show()

    #Routine is meant for example, the question answer is located in q2_3_b
    #First get the input signal and impulse response
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    h = np.array([1, 2, 3, 4, 5, 6])
    #convolve test with our implementation
    y_test = convolve(x, h)
    #test with numpy function
    y = np.convolve(x, h)
    comparison = y == y_test
    assert comparison.all()
    

"""
This question is about passing a sinuisodal wave through an LTI system given a characteristic impulse respones
It meant to demonstrate how sine waves are eigen function (modified amplitude and phaseshift but frequency is the same)
It also shown how convolution with impulses signals are meant to look like.
Lastly it investigate how assuming a sine wave as finite might not be good for our analysis.
"""
def q2_1_b():
    #Since 0.1 pi per sample index, to cover 2 pi (1 cycle) -> 20 sample index
    #0-19 is one sample, 20 - 39 is the next sample, so on.
    #Here we choose 102 (non-inclusive) => just nice finish the last cycle and 1 more.

    #Here is the sample index space
    n = np.arange(0, 101)

    #Here is the signal
    x = np.cos(0.1 * np.pi * n)

    #Here is the filter/impuse response coeff
    h = np.array([0.2, 0.3, -0.5])

    #Convolution in Python
    y = np.convolve(x, h)

    #Plot the resultsj
    _, ax = plt.subplots(2, 1)
    ax[0].stem(x)
    ax[0].grid()
    ax[1].stem(y)
    ax[1].grid()
    plt.show()

def main():
    #q2_1_b()
    #q2_3_a()
    #q2_3_b()
    #q2_4_a()
    #q2_4_b()
    #q2_4_c()
    #q2_5_a()
    #q2_5_c()
    #q2_5_d()
    q2_5_f()

if __name__ == "__main__":
    main()