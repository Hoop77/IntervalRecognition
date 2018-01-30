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
FRAME_SIZE = CHUNK * CHANNELS
RECORDED_FRAMES = 64
RECORDED_SIZE = RECORDED_FRAMES * FRAME_SIZE
AMPLITUDE_THRESHOLD = 2.0
WINDOW_SIZE = 10
SLEEP_TIME = 0.01
UPDATE_UI = True
INTERVAL_TOLERANCE = 100. # in cent (100 cent == minor second)
FREQUENCIES = int(RECORDED_SIZE / 2)

def freq_of_tone_idx(tone_idx):
    return 440 * (np.power(np.power(2, 1./12.), tone_idx-49))

def interval_to_cent(freq1, freq2):
    high_freq = max(freq1, freq2)
    low_freq = min(freq1, freq2)
    if low_freq == 0.:
        return float('inf')
    return 1200. * np.log2(high_freq / low_freq)

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
tones = [freq_of_tone_idx(tone) for tone in range(-8, 88+1)]
label_of_tone = {tones[i]: tone_labels[i] for i in range(len(tones))}

tone_series_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
tone_series = [tones[i::12] for i in range(12)]

series_idx_of_tone = {tone: (i % 12) for i, tone in enumerate(tones)}

intervals = [
    "prime",
    "minor second",
    "major second",
    "minor third",
    "major third",
    "perfect forth",
    "diminished fifth",
    "perfect fifth",
    "minor sixth",
    "major sixth",
    "minor seventh",
    "major seventh",
    "octave"
]

def print_tone_series():
    for idx, series in enumerate(tone_series):
        for i, tone in enumerate(series):
            print("{}{}: {:.2f}Hz".format(tone_series_labels[idx], i, tone))

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
    spectrum = np.fft.fft(recorded_data)[:FREQUENCIES]
    spectrum = np.abs(spectrum / 500000)
    return spectrum

def make_frequencies():
    return np.fft.fftfreq(RECORDED_SIZE, d=1.0/(RATE * CHANNELS))[:FREQUENCIES]

def make_nearest_tone_of_frequency_dict(frequencies):
    result = {}
    for freq in frequencies:
        nearest_tone = get_nearest_tone(freq)
        if interval_to_cent(nearest_tone, freq) > INTERVAL_TOLERANCE:
            nearest_tone = None
        result[freq] = nearest_tone
    return result

def linear_filter(x, freq=(50, 100), gain=(0., 1.)):
    start_freq = freq[0]
    end_freq = freq[1]
    start_gain = gain[0]
    end_gain = gain[1]

    if x <= start_freq:
        return start_gain
    elif x >= end_freq:
        return end_gain
    else:
        return start_gain + ((end_gain - start_gain) * ((x - start_freq) / (end_freq - start_freq)))

def linear_highpass_filter(frequencies, spectrum, freq=(0, 500), gain=(0., 1.)):
    for i, x in enumerate(frequencies):
        spectrum[i] *= linear_filter(x, freq, gain)

def logarithmic_filter(x, intensity=1., release=1.):
    return np.log2(np.power((1./intensity) * x, release) + 1)

def logarithmic_highpass_filter(frequencies, spectrum, intensity=1., release=1.):
    maxval = logarithmic_filter(frequencies[-1], intensity, release)
    for i, x in enumerate(frequencies):
        spectrum[i] *= logarithmic_filter(x, intensity, release) / maxval

def make_highpass_filter(frequencies):
    filter_spectrum = np.array([1.] * len(frequencies))
    logarithmic_highpass_filter(frequencies, filter_spectrum, intensity=1., release=1.)
    return filter_spectrum

frequencies = make_frequencies()
highpass_filter = make_highpass_filter(frequencies)
nearest_tone_of_frequency = make_nearest_tone_of_frequency_dict(frequencies)

def get_tone_series_idx_from_spectrum(frequencies, spectrum):
    # Add the amplitude of each tone series together and return the series with the maximum sum.
    # This might be the one with the most impact on a certain tone.
    series_sum = {series_idx: 0 for series_idx in range(12)}
    for i, freq in enumerate(frequencies):
        nearest_tone = nearest_tone_of_frequency[freq]
        if nearest_tone is None:
            continue
        nearest_series_idx = series_idx_of_tone[nearest_tone]
        series_sum[nearest_series_idx] += spectrum[i]
    return max(series_sum, key=series_sum.get)

class Window:
    def __init__(self, size=WINDOW_SIZE):
        self.size = size
        self.data = [None] * size

    def put(self, tone):
        # rotate left
        self.data[:-1] = self.data[1:]
        self.data[-1] = tone

    def get(self):
        weights = {}
        # currently uniform weights
        w = 1. / self.size
        # assign weights to each element in data
        for el in self.data:
            if el not in weights:
                weights[el] = w
            else:
                weights[el] += w
        return max(weights, key=weights.get)

def threshold_filter(spectrum, threshold):
    for i, y in enumerate(spectrum):
        if y < threshold:
            spectrum[i] = 0.

def plot_tone_series(series_label, color=(1, 0, 1, 0.5)):
    tone_series_idx = tone_series_labels.index(series_label)
    lines = []
    for tone in tone_series[tone_series_idx]:
        line = plt.axvline(x=tone, color=color, linewidth=1.5)
        lines.append(line)
    return lines

def update_lines(lines):
    for line in lines:
        line.set_ydata(line.get_ydata())

def callback(in_data, frame_count, time_info, flag):
    if flag:
        print("Playback Error: %i" % flag)

    callback.recorded_data_mutex.acquire()
    try:
        # rotate left
        callback.recorded_data[:-FRAME_SIZE] = callback.recorded_data[FRAME_SIZE:]
        callback.recorded_data[-FRAME_SIZE:] = np.fromstring(in_data, dtype=NP_FORMAT)
    finally:
        callback.recorded_data_mutex.release()

    return None, paContinue

callback.recorded_data = np.zeros(RECORDED_SIZE)
callback.recorded_data_mutex = Lock()
window_function = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, RECORDED_SIZE, False)))

window = Window()
num_tones_recognized = 0
last_series_idx = None

audio = pyaudio.PyAudio()

fig, ax = plt.subplots(1, 1)
plt.ion()
plt.tight_layout()

y = [0] * FREQUENCIES
threshold = [AMPLITUDE_THRESHOLD] * len(frequencies)

ax.set_ylim(0, 50)
ax.set_xlim(0, 2000)

spectrum_graph, = ax.plot(frequencies, y, color=(0, 0, 1, 0.5))
threshold_graph, = ax.plot(frequencies, threshold, color=(1, 0, 0, 1))
highpass_filter_graph, = ax.plot(frequencies, highpass_filter, color=(0, 1, 0, 1))
glines = plot_tone_series("G")

# start Recording
stream = audio.open(format=PYAUDIO_FORMAT, 
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback = callback)
print("recording...")

data = np.array([])
while stream.is_active():
    callback.recorded_data_mutex.acquire()
    try:
        data = np.array(callback.recorded_data)
    finally:
        callback.recorded_data_mutex.release()

    if len(data) == RECORDED_SIZE:
        spectrum = get_spectrum(data * window_function)
        spectrum *= highpass_filter
        max_amplitude = max(spectrum)

        series_idx = get_tone_series_idx_from_spectrum(frequencies, spectrum)

        if max_amplitude < AMPLITUDE_THRESHOLD:
            window.put(None)
        else:
            window.put(series_idx)
            #print("current tone: {}".format(tone_series_labels[series_idx]))
        
        curr_series_idx = window.get()
        if curr_series_idx is not last_series_idx:
            if curr_series_idx is None:
                print("Silence")
            else:
                print("#{} Current tone: {}".format(num_tones_recognized, tone_series_labels[curr_series_idx]))
                num_tones_recognized += 1
            last_series_idx = curr_series_idx

        if UPDATE_UI:
            update_lines(glines)
            spectrum_graph.set_ydata(spectrum)
            threshold_graph.set_ydata(threshold)
            highpass_filter_graph.set_ydata(highpass_filter)

            plt.pause(SLEEP_TIME)
        else:
            sleep(SLEEP_TIME)

stream.close()
audio.terminate()