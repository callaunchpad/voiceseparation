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
        self.mix_dir = hparams.mix_dir
        self.voc_dir = hparams.voc_dir
        self.inst_dir = hparams.inst_dir

        self.mix_data = [join(self.mix_dir, f).lower() for f in listdir(self.mix_dir) if
                         isfile(join(self.mix_dir, f)) and f.endswith('.wav')]
        self.voc_data = [join(self.voc_dir, f).lower() for f in listdir(self.voc_dir) if
                         isfile(join(self.voc_dir, f)) and f.endswith('.wav')]
        self.inst_data = [join(self.inst_dir, f).lower() for f in listdir(self.inst_dir) if
                          isfile(join(self.inst_dir, f)) and f.endswith('.wav')]
        self.data_len = len(self.mix_data)

    def build_data(self, val=False):

        vocals = np.zeros((self.hparams.batch_size, self.hparams.waveform_size))
        instrumentals = np.zeros((self.hparams.batch_size, self.hparams.waveform_size))
        mixture = np.zeros((self.hparams.batch_size, self.hparams.waveform_size))

        data_names = []

        for b in range(self.hparams.batch_size):
            if val:
                mix_train = self.mix_data[int(len(self.mix_data) * self.hparams.val_split):]
                voc_train = self.voc_data[int(len(self.voc_data) * self.hparams.val_split):]
                inst_train = self.inst_data[int(len(self.inst_data) * self.hparams.val_split):]

                ind_choice = random.randint(len(mix_train))

                mix_name = mix_train[ind_choice]
                voc_name = voc_train[ind_choice]
                inst_name = inst_train[ind_choice]

                Fs_mix, audio_arr_mix = read(mix_name)
                Fs_voc, audio_arr_voc = read(voc_name)
                Fs_inst, audio_arr_inst = read(inst_name)
            else:
                mix_train = self.mix_data[:int(len(self.mix_data) * self.hparams.val_split)]
                voc_train = self.voc_data[:int(len(self.voc_data) * self.hparams.val_split)]
                inst_train = self.inst_data[:int(len(self.inst_data) * self.hparams.val_split)]

                ind_choice = random.randint(len(mix_train))

                mix_name = mix_train[ind_choice]
                voc_name = voc_train[ind_choice]
                inst_name = inst_train[ind_choice]

                Fs_mix, audio_arr_mix = read(mix_name)
                Fs_voc, audio_arr_voc = read(voc_name)
                Fs_inst, audio_arr_inst = read(inst_name)
            assert ((Fs_mix == self.hparams.Fs) and (Fs_voc == self.hparams.Fs) and (Fs_inst == self.hparams.Fs))

            start = np.random.randint(len(audio_arr_mix) - self.hparams.waveform_size)

            audio_arr_mix = audio_arr_mix[start:start + self.hparams.waveform_size]
            audio_arr_voc = audio_arr_voc[start:start + self.hparams.waveform_size]
            audio_arr_inst = audio_arr_inst[start:start + self.hparams.waveform_size]

            vocals[b, :] = audio_arr_voc
            instrumentals[b, :] = audio_arr_inst
            mixture[b, :] = simulate_mixture(self.hparams, audio_arr_voc, audio_arr_inst)

            data_names.append([mix_name, voc_name, inst_name])

        return vocals, instrumentals, mixture, data_names
