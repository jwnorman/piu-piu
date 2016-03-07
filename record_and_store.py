# record and store a song view sounddevice
# this is to be able to plot the recording
# of sounddevice and compare it to a plot
# of the recording as a .wav file (via GarageBand)

from __future__ import division
from scipy.io import wavfile
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn
import operator
import glob
import sys
import fnmatch
import os
import re
import shutil
import uuid
import pipes
import sounddevice as sd
import pickle

duration = 397  # seconds of riri
fs = 44100
sd.wait()

# Sounddevice recording
# myrecording = sd.rec(duration * fs, samplerate=fs, channels=2)
# pickle.dump(myrecording, open("/Users/jacknorman1/Documents/USF/MSAN/Module3/ML2/Project/piu-piu/riri/riri.pickle", "wb"))
myrecording = pickle.load(open("/Users/jacknorman1/Documents/USF/MSAN/Module3/ML2/Project/piu-piu/riri/riri.pickle", "rb"))
myrecording_channel0 = myrecording[:,0]
plt.subplot(2,2,1)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Live-Stream Recording')
plt.plot(myrecording_channel0)

fd_myrecording = abs(np.fft.fft(myrecording_channel0))
freq_myrecording = abs(np.fft.fftfreq(len(fd_myrecording), 1/float(44100)))
plt.subplot(2,2,2)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Live-Stream Recording (Fast Fourier Transformed)')
# plt.axis([0, 3000, 0, 12000])

plt.plot(freq_myrecording, fd_myrecording)
#plt.show()

# GarageBand recording
# fs_gb, data_gb = wavfile.read("/Users/jacknorman1/Documents/USF/MSAN/Module3/ML2/Project/piu-piu/riri/riri_recording_via_computer.wav")
# plt.subplot(3,2,3)
# channel0_gb = data_gb[:,0]
# plt.plot(channel0_gb)

# fd_gb = abs(np.fft.fft(channel0_gb))
# freq_gb = abs(np.fft.fftfreq(len(fd_gb), 1/float(44100)))
# plt.subplot(3,2,4)
# plt.axis([0, 3000, 0, 500000000])
# plt.plot(freq_gb, fd_gb)
# #plt.show()

# Raw file
fs_raw, data_raw = wavfile.read("/Users/jacknorman1/Documents/USF/MSAN/Module3/ML2/Project/piu-piu/riri/riri_regular.wav")
plt.subplot(2,2,3)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Audio File')
channel0_raw = data_raw[:,0]
plt.plot(channel0_raw)

fd_raw = abs(np.fft.fft(channel0_raw))
freq_raw = abs(np.fft.fftfreq(len(fd_raw), 1/float(44100)))
plt.subplot(2,2,4)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Audio File (Fast Fourier Transformed)')
# plt.axis([0, 3000, 0, 5000000000])
plt.plot(freq_raw, fd_raw)

# plt.savefig("/Users/jacknorman1/Documents/USF/MSAN/Module3/ML2/Project/piu-piu/riri/riri_plot")
plt.show()





