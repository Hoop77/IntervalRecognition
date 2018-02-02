import pyaudio
from pyaudio import PyAudio, paContinue, paComplete
import wave
from time import sleep
from matplotlib import pyplot as plt
from threading import Lock
import numpy as np
from eq import eq

PYAUDIO_FORMAT = pyaudio.paInt16
NP_FORMAT = np.int16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FRAME_SIZE = CHUNK * CHANNELS
RECORDED_FRAMES = 32
RECORDED_SIZE = RECORDED_FRAMES * FRAME_SIZE
AMPLITUDE_THRESHOLD = 2.0
WINDOW_SIZE = 10
SLEEP_TIME = 0.01
UPDATE_UI = True
INTERVAL_TOLERANCE = 100. # in cent (100 cent == minor second)
FREQUENCIES = int(RECORDED_SIZE / 2)

def freq_of_note_idx(note_idx):
    return 440 * (np.power(np.power(2, 1./12.), note_idx-49))

def interval_to_cent(freq1, freq2):
    high_freq = max(freq1, freq2)
    low_freq = min(freq1, freq2)
    if low_freq == 0.:
        return float('inf')
    return 1200. * np.log2(high_freq / low_freq)

note_labels = [
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
notes = [freq_of_note_idx(note) for note in range(-8, 88+1)]
label_of_note = {notes[i]: note_labels[i] for i in range(len(notes))}

note_series_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
note_series = [notes[i::12] for i in range(12)]

series_idx_of_note = {note: (i % 12) for i, note in enumerate(notes)}

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

def note_idx_to_interval(note_series_idx0, note_series_idx1):
    if note_series_idx0 <= note_series_idx1:
        return intervals[note_series_idx1 - note_series_idx0]
    else:
        return intervals[len(note_series) - (note_series_idx0 - note_series_idx1)]

def print_note_series():
    for idx, series in enumerate(note_series):
        for i, note in enumerate(series):
            print("{}{}: {:.2f}Hz".format(note_series_labels[idx], i, note))

def comp(freq, l, r):
    m = l + int((r - l) / 2)
    if l == m:
        if np.abs(freq - notes[l]) <= np.abs(freq - notes[r]):
            return notes[l]
        else: 
            return notes[r]
    elif freq <= notes[l]:
        return notes[l]
    elif freq > notes[l] and freq <= notes[m]:
        return comp(freq, l, m)
    elif freq > notes[m] and freq < notes[r]:
        return comp(freq, m, r)
    else:
        return notes[r]

def get_nearest_note(freq):
    return comp(freq, 0, len(notes) - 1)

def get_spectrum(recorded_data):
    spectrum = np.fft.fft(recorded_data)[:FREQUENCIES]
    spectrum = np.abs(spectrum / 500000)    # TODO: Change factor!
    return spectrum

def make_frequencies():
    return np.fft.fftfreq(RECORDED_SIZE, d=1.0/(RATE * CHANNELS))[:FREQUENCIES]

def make_nearest_note_of_frequency_dict(frequencies):
    result = {}
    for freq in frequencies:
        nearest_note = get_nearest_note(freq)
        if interval_to_cent(nearest_note, freq) > INTERVAL_TOLERANCE:
            nearest_note = None
        result[freq] = nearest_note
    return result

frequencies = make_frequencies()

eq_points = {
    0: 0,
    100: 1,
}
eq_filter = eq(frequencies, eq_points)

nearest_note_of_frequency = make_nearest_note_of_frequency_dict(frequencies)

def get_note_series_idx_from_spectrum(frequencies, spectrum):
    # Add the amplitude of each note series together and return the series with the maximum sum.
    # This might be the one with the most impact on a certain note.
    series_sum = {series_idx: 0 for series_idx in range(12)}
    for i, freq in enumerate(frequencies):
        nearest_note = nearest_note_of_frequency[freq]
        if nearest_note is None:
            continue
        nearest_series_idx = series_idx_of_note[nearest_note]
        series_sum[nearest_series_idx] += spectrum[i]
    return max(series_sum, key=series_sum.get)

class Window:
    def __init__(self, size=WINDOW_SIZE):
        self.size = size
        self.data = [None] * size

    def put(self, note):
        # rotate left
        self.data[:-1] = self.data[1:]
        self.data[-1] = note

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

def plot_note_series(series_label, color=(1, 0, 1, 0.5)):
    note_series_idx = note_series_labels.index(series_label)
    lines = []
    for note in note_series[note_series_idx]:
        line = plt.axvline(x=note, color=color, linewidth=1.5)
        lines.append(line)
    return lines

def update_lines(lines):
    for line in lines:
        line.set_ydata(line.get_ydata())

def callback(in_data, frame_count, time_info, flag):
    if flag:
        print("Playback Error: %i" % flag)

    # rotate left
    callback.recorded_data[:-FRAME_SIZE] = callback.recorded_data[FRAME_SIZE:]
    callback.recorded_data[-FRAME_SIZE:] = np.fromstring(in_data, dtype=NP_FORMAT)

    return None, paContinue

callback.recorded_data = np.zeros(RECORDED_SIZE)

window_function = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, RECORDED_SIZE, False)))

window = Window()
num_notes_recognized = 0
last_series_idx = None

audio = pyaudio.PyAudio()

fig, ax = plt.subplots(1, 1)
plt.ion()
plt.tight_layout()

y = [0] * FREQUENCIES
threshold = [AMPLITUDE_THRESHOLD] * len(frequencies)

ax.set_ylim(0, 50)
ax.set_xlim(0, 2000)

original_spectrum_graph, = ax.plot(frequencies, y, color=(0, 0, 0, 0.3))
modified_spectrum_graph, = ax.plot(frequencies, y, color=(0, 0, 1, 0.5))
threshold_graph, = ax.plot(frequencies, threshold, color=(1, 0, 0, 1))
eq_graph, = ax.plot(frequencies, eq_filter, color=(0, 1, 0, 1))
glines = plot_note_series("G")

# start Recording
stream = audio.open(format=PYAUDIO_FORMAT, 
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback = callback)
print("recording...")

while stream.is_active():
        spectrum = get_spectrum(callback.recorded_data * window_function)

        if UPDATE_UI:
            original_spectrum = np.copy(spectrum)

        spectrum *= eq_filter
        max_amplitude = max(spectrum)

        series_idx = get_note_series_idx_from_spectrum(frequencies, spectrum)

        if max_amplitude < AMPLITUDE_THRESHOLD:
            window.put(None)
        else:
            window.put(series_idx)
            #print("current note: {}".format(note_series_labels[series_idx]))
        
        curr_series_idx = window.get()
        if curr_series_idx is not last_series_idx:
            if curr_series_idx is None:
                print("Silence")
            else:
                if last_series_idx is not None:
                    print("#{} Current note: {}, interval: {}".format(num_notes_recognized, note_series_labels[curr_series_idx], note_idx_to_interval(last_series_idx, curr_series_idx)))
                else:
                    print("#{} Current note: {}".format(num_notes_recognized, note_series_labels[curr_series_idx]))

                num_notes_recognized += 1
            last_series_idx = curr_series_idx

        if UPDATE_UI:
            update_lines(glines)
            original_spectrum_graph.set_ydata(original_spectrum)
            modified_spectrum_graph.set_ydata(spectrum)
            threshold_graph.set_ydata(threshold)
            eq_graph.set_ydata(eq_filter)

            plt.pause(SLEEP_TIME)
        else:
            sleep(SLEEP_TIME)

stream.close()
audio.terminate()