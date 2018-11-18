from scipy import signal
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import os
import wave
import tensorflow as tf

f = "./audio/Drumtracks_GhostBitch_Separated/splits/totals/Drumtracks_GhostBitch_totals_0_6197.wav"
with wave.open(f, "rb") as wave_file:
	# frame_rate = wave_file.getframerate()

	# song = AudioSegment.from_file(f, format='wav')
	# samples = song.get_array_of_samples()
	# samples = np.array(samples)

	# f, t, Sxx = signal.spectrogram(samples, frame_rate)
	# plt.pcolormesh(t, f, Sxx)
	# plt.ylabel('Frequency [Hz]')
	# plt.xlabel('Time [sec]')
	# plt.show()

	