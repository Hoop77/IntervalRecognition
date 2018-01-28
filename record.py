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
    spectrum = np.abs(np.fft.fft(recorded_data))
    spectrum /= 500000
    freq = np.fft.fftfreq(len(recorded_data), d=1/(RATE * CHANNELS))
    return freq, spectrum

def callback(in_data, frame_count, time_info, flag):
    if flag:
        print("Playback Error: %i" % flag)

    callback.curr_data_mutex.acquire()
    try:
        callback.curr_data = np.fromstring(in_data, dtype=NP_FORMAT)
    finally:
        callback.curr_data_mutex.release()

    callback.frames.append(in_data)

    if len(callback.frames) >= RATE / CHUNK * RECORD_SECONDS:
        return None, paComplete

    return None, paContinue

callback.frames = []
callback.curr_data = []
callback.curr_data_mutex = Lock()

audio = pyaudio.PyAudio()

fig, ax = plt.subplots(1, 1)
plt.ion()
x = list(range(2048))
y = [0] * 2048
ax.set_ylim(0, 10)
graph, = ax.plot(y)
 
# start Recording
stream = audio.open(format=PYAUDIO_FORMAT, 
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback = callback)
print("recording...")
 
curr_data = []
while stream.is_active():
    callback.curr_data_mutex.acquire()
    try:
        curr_data = list(callback.curr_data)
    finally:
        callback.curr_data_mutex.release()

    if len(curr_data) == RECORDED_SIZE:
        freq, spectrum = get_spectrum(curr_data)
        graph.set_data(freq, spectrum)

    plt.pause(0.1)

stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(PYAUDIO_FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(callback.frames))
waveFile.close()