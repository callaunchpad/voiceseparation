from scipy import signal
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import wave
import os

def _trivial__enter__(self):
    return self
def _self_close__exit__(self, exc_type, exc_value, traceback):
    self.close()

wave.Wave_read.__exit__ = wave.Wave_write.__exit__ = _self_close__exit__
wave.Wave_read.__enter__ = wave.Wave_write.__enter__ = _trivial__enter__

#input_song = pyd.AudioSegment.from_wav("audio/AE3_ImComingHome-Full Session_Separated/instrumentals.wav")

f = "./audio/AE3_ImComingHome-Full Session_Separated/instrumentals.wav"
song = AudioSegment.from_file(f, format='wav')
samples = song.get_array_of_samples()
samples = np.array(samples)

with wave.open(f, "rb") as wave_file:
    frame_rate = wave_file.getframerate()

#fs = 10e3
# N = 1e5
# amp = 2 * np.sqrt(2)
# noise_power = 0.01 * fs / 2
# time = np.arange(N) / float(fs)
# mod = 500*np.cos(2*np.pi*0.25*time)
# carrier = amp * np.sin(2*np.pi*3e3*time + mod)
# noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# noise *= np.exp(-time/5)
# x = carrier + noise

f, t, Sxx = signal.spectrogram(samples,frame_rate)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
