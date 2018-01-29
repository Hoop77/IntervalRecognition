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
INPUT_SIZE = CHUNK * CHANNELS
RECORDED_FRAMES = 32
RECORDED_SIZE = RECORDED_FRAMES * INPUT_SIZE
RECORD_SECONDS = 20
WAVE_OUTPUT_FILENAME = "file.wav"

def freq_of_tone(tone):
    return 440 * (np.power(np.power(2, 1./12.), tone-49))

tone_labels = [
    "C0", "C#0", "D0", "D#0", "E0", "F0", "F#0", "G0", "G#0", "A0", "A#0", "B0",
    "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1",
    "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
    "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6",
    "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7",
    "C8"
]
tones = [freq_of_tone(tone) for tone in range(-8, 88+1)]
label_of_tone = {tones[i]: tone_labels[i] for i in range(len(tones))}

def comp(freq, l, r):
    m = l + int((r - l) / 2)
    if l == m:
        if np.abs(freq - tones[l]) <= np.abs(freq - tones[r]):
            return tones[l]
        else: 
            return tones[r]
    elif freq <= tones[l]:
        return tones[l]
    elif freq > tones[l] and freq <= tones[m]:
        return comp(freq, l, m)
    elif freq > tones[m] and freq < tones[r]:
        return comp(freq, m, r)
    else:
        return tones[r]

def get_nearest_tone(freq):
    return comp(freq, 0, len(tones) - 1)

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

    if len(callback.frames ) >= RECORDED_FRAMES:
        callback.recorded_data_mutex.acquire()
        try:
            callback.recorded_data = np.array([])
            for frame in callback.frames[-RECORDED_FRAMES:]:
                callback.recorded_data = np.append(callback.recorded_data, np.fromstring(frame, dtype=NP_FORMAT))
        finally:
            callback.recorded_data_mutex.release()

    callback.frames.append(in_data)

    if len(callback.frames) >= RATE / CHUNK * RECORD_SECONDS:
        return None, paComplete

    return None, paContinue

callback.frames = []
callback.recorded_data = np.array([])
callback.recorded_data_mutex = Lock()

audio = pyaudio.PyAudio()

fig, ax = plt.subplots(1, 1)
plt.ion()
plt.tight_layout()
frequencies = get_frequencies()
N = int(RECORDED_SIZE / 2)
y = [0] * N
ax.set_ylim(0, 10)
ax.set_xlim(0, 4500)
graph, = ax.plot(frequencies, y)

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
    callback.recorded_data_mutex.acquire()
    try:
        data = list(callback.recorded_data)
    finally:
        callback.recorded_data_mutex.release()

    if len(data) == RECORDED_SIZE:
        spectrum = get_spectrum(data)
        strongest_freq = frequencies[np.argmax(spectrum)]
        true_freq = get_nearest_tone(strongest_freq)
        tone_label = label_of_tone[true_freq]
        print("current tone: {} @{:.02f} Hz (perfect tone @{:.02f} Hz)".format(tone_label, strongest_freq, true_freq))

        #graph.set_ydata(spectrum)
        #plt.pause(0.01)

    sleep(0.01)

stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(PYAUDIO_FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(callback.frames))
waveFile.close()