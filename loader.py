import os
from os import listdir, makedirs
from os.path import isfile, join
import numpy as np
from scipy.io.wavfile import read, write
import random

def snr_to_weight(snr):
	return 10 ** (snr / 20)

def simulate_mixture(hparams, vocals, instrumentals):
	snr = hparams.max_input_snr * np.random.random()
	snrs = [snr, -snr]
	random.shuffle(snrs)

	vocals = snr_to_weight(snrs[0]) * vocals
	instrumentals = snr_to_weight(snrs[1]) * instrumentals

	mixture = vocals + instrumentals
	return mixture

class Loader(object):
	def __init__(self, hparams):
		self.hparams = hparams
		self.data_dir = hparams.data_dir

		data = [join(self.data_dir, f).lower() for f in listdir(self.data_dir) if
				 isfile(join(self.data_dir, f)) and f.endswith('.wav')]
		self.data = data

	def build_data(self, val = False):

		vocals = np.zeros((self.hparams.batch_size, self.hparams.waveform_size))
		instrumentals = np.zeros((self.hparams.batch_size, self.hparams.waveform_size))
		mixture = np.zeros((self.hparams.batch_size, self.hparams.waveform_size))

		data_names = []

		for b in range(self.hparams.batch_size):
			if val:
				audio_name = random.choice(self.data[int(len(self.data) * self.hparams.val_split):])
				Fs, audio_arr = read(audio_name)
			else:
				audio_name = random.choice(self.data[:int(len(self.data) * self.hparams.val_split)])
				Fs, audio_arr = read(audio_name)
			assert(Fs == self.hparams.Fs)

			start = np.random.randint(len(audio_arr) - self.hparams.waveform_size)
			audio_arr = audio_arr[start:start+self.hparams.waveform_size]

			vocals[b, :] = audio_arr[:, 1]
			instrumentals[b, :] = audio_arr[:, 0]
			mixture[b, :] = simulate_mixture(self.hparams, audio_arr[:, 1], audio_arr[:, 0])

			data_names.append(audio_name)

		return vocals, instrumentals, mixture, data_names
