import pyaudio
from pyaudio import PyAudio, paContinue, paComplete
import wave
from time import sleep
from matplotlib import pyplot as plt
from threading import Lock
import numpy as np

PYAUDIO_FORMAT = pyaudio.paInt16
NP_FORMAT = np.int16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORDED_SIZE = CHUNK * CHANNELS
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "file.wav"
 
def get_spectrum(recorded_data):
    N = int(RECORDED_SIZE / 2)
    spectrum = np.fft.fft(recorded_data)[:N]
    spectrum = np.abs(spectrum / 500000)
    return spectrum

def get_frequencies():
    N = int(RECORDED_SIZE / 2)
    return np.fft.fftfreq(RECORDED_SIZE, d=1.0/(RATE * CHANNELS))[:N]

def callback(in_data, frame_count, time_info, flag):
    if flag:
        print("Playback Error: %i" % flag)

    callback.recroded_data_mutex.acquire()
    try:
        callback.recroded_data = np.fromstring(in_data, dtype=NP_FORMAT)
    finally:
        callback.recroded_data_mutex.release()

    callback.frames.append(in_data)

    if len(callback.frames) >= RATE / CHUNK * RECORD_SECONDS:
        return None, paComplete

    return None, paContinue

callback.frames = []
callback.buffer = []
callback.recroded_data = []
callback.recroded_data_mutex = Lock()

audio = pyaudio.PyAudio()

fig, ax = plt.subplots(1, 1)
plt.ion()
freq = get_frequencies()
N = int(RECORDED_SIZE / 2)
y = [0] * N
ax.set_ylim(0, 10)
ax.set_xlim(0, 1000)
graph, = ax.plot(freq, y)
 
# start Recording
stream = audio.open(format=PYAUDIO_FORMAT, 
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback = callback)
print("recording...")
 
data = []
while stream.is_active():
    callback.recroded_data_mutex.acquire()
    try:
        data = list(callback.recroded_data)
    finally:
        callback.recroded_data_mutex.release()

    if len(data) == RECORDED_SIZE:
        spectrum = get_spectrum(data)
        graph.set_ydata(spectrum)

    plt.pause(0.1)

stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(PYAUDIO_FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(callback.frames))
waveFile.close()