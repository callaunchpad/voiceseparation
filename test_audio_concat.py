import numpy as np
from pydub import AudioSegment
import os
from os import listdir, makedirs
from os.path import isfile, join

audio_path = './audio/'
song_names = listdir(audio_path) #list of directories of song names
print(song_names)
vocal_keyword = 'vox'

output_dir = audio_path + "training_data/"
instrumentals_output_path = output_dir + 'instrumentals/'
vocals_output_path = output_dir + 'vocals/'
totals_output_path = output_dir + 'total_track/'

# track length in milliseconds
min_split_track_length = 3000
max_split_track_length = 5000

for song in song_names:
    song_instrumentals_output_path = instrumentals_output_path + song + '_instrumentals.wav'
    song_vocals_output_path = vocals_output_path + song + '_vocals.wav'
    song_totals_output_path = totals_output_path + song + '_total_track.wav'
    track_dir_path = audio_path + song

    files = [join(track_dir_path, f).lower() for f in listdir(track_dir_path) if
             isfile(join(track_dir_path, f)) and f.endswith('.wav')]
    data = {f: AudioSegment.from_file(f, format='wav') for f in files}

    instrumentals = {f: data.get(f) for f in list(data.keys()) if vocal_keyword not in f}
    vocals = {f: data.get(f) for f in list(data.keys()) if vocal_keyword in f}

    max_duration = max([len(track) for track in data.values()])

    instrumentals_track = AudioSegment.silent(duration=max_duration)
    for track in instrumentals.values():
        instrumentals_track = instrumentals_track.overlay(track, position=0)

    vocals_track = AudioSegment.silent(duration=max_duration)
    for track in vocals.values():
        vocals_track = vocals_track.overlay(track, position=0)

    totals_track = instrumentals_track.overlay(vocals_track)

    #split the instrumentals, vocals, totals
    split_times = [0]
    total_time = 0
    while total_time < max_duration:
        time = np.random.rand(1)[0] * (max_split_track_length - min_split_track_length) + min_split_track_length
        total_time += time
        split_times.append(split_times[-1] + time)

    split_instrumentals = [instrumentals_track[split_times[i - 1]:split_times[i]] for i in range(1, len(split_times))]
    split_vocals = [vocals_track[split_times[i - 1]:split_times[i]] for i in range(1, len(split_times))]
    split_totals = [totals_track[split_times[i - 1]:split_times[i]] for i in range(1, len(split_times))]

    #create a totals, instr, and voc folder in /audio/training_data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(instrumentals_output_path):
        os.makedirs(instrumentals_output_path)
    if not os.path.exists(vocals_output_path):
        os.makedirs(vocals_output_path)
    if not os.path.exists(totals_output_path):
        os.makedirs(totals_output_path)

    #export split audio clips
    for i in range(len(split_totals)):
        path = instrumentals_output_path + song + '_instrumentals_' + str(i) + '_' + str(
            len(split_instrumentals[i])) + '.wav'
        with open(path, 'wb') as f:
            split_instrumentals[i].export(f, format='wav')
        path = vocals_output_path + song + '_vocals_' + str(i) + '_' + str(len(split_vocals[i])) + '.wav'
        with open(path, 'wb') as f:
            split_vocals[i].export(f, format='wav')
        path = totals_output_path + song + '_totals_' + str(i) + '_' + str(len(split_totals[i])) + '.wav'
        with open(path, 'wb') as f:
            split_totals[i].export(f, format='wav')
