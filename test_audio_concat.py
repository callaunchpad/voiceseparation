import numpy as np
from pydub import AudioSegment
import os
from os import listdir, makedirs
from os.path import isfile, join

song_name = 'Drumtracks_GhostBitch'
vocal_keyword = 'Vox'
audio_path = './audio/'

track_dir_path = audio_path + song_name
output_dir = audio_path + song_name + '_Separated'

instrumentals_output_path = output_dir + '/instrumentals.wav'
vocals_output_path = output_dir + '/vocals.wav'
totals_output_path = output_dir + '/total_track.wav'

splits_dir = output_dir + '/splits'
vocal_splits_dir = splits_dir + '/vocals'
instrumental_splits_dir = splits_dir + '/instrumentals'
total_splits_dir = splits_dir + '/totals'

# track length in milliseconds
min_split_track_length = 5000
max_split_track_length = 15000

files = [join(track_dir_path, f) for f in listdir(track_dir_path) if isfile(join(track_dir_path, f)) and f.endswith('.wav')]
data = {f : AudioSegment.from_file(f, format='wav') for f in files}

instrumentals = {f : data.get(f) for f in list(data.keys()) if vocal_keyword not in f}
vocals = {f : data.get(f) for f in list(data.keys()) if vocal_keyword in f}

max_duration = max([len(track) for track in data.values()])

instrumentals_track = AudioSegment.silent(duration = max_duration)
for track in instrumentals.values():
	instrumentals_track = instrumentals_track.overlay(track, position = 0)

vocals_track = AudioSegment.silent(duration = max_duration)
for track in vocals.values():
	vocals_track = vocals_track.overlay(track, position = 0)

totals_track = instrumentals_track.overlay(vocals_track)

split_times = [0]
total_time = 0
while total_time < max_duration:
	time = np.random.rand(1)[0]*(max_split_track_length - min_split_track_length) + min_split_track_length
	total_time += time
	split_times.append(split_times[-1] + time)

split_instrumentals = [instrumentals_track[split_times[i-1]:split_times[i]] for i in range(1, len(split_times))]
split_vocals = [vocals_track[split_times[i-1]:split_times[i]] for i in range(1, len(split_times))]
split_totals = [totals_track[split_times[i-1]:split_times[i]] for i in range(1, len(split_times))]

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

with open(instrumentals_output_path, 'wb') as f:
	instrumentals_track.export(f, format = 'wav')
with open(vocals_output_path, 'wb') as f:
	vocals_track.export(f, format = 'wav')
with open(totals_output_path, 'wb') as f:
	totals_track.export(f, format = 'wav')

if not os.path.exists(splits_dir):
	os.makedirs(splits_dir)
if not os.path.exists(instrumental_splits_dir):
	os.makedirs(instrumental_splits_dir)
if not os.path.exists(vocal_splits_dir):
	os.makedirs(vocal_splits_dir)
if not os.path.exists(total_splits_dir):
	os.makedirs(total_splits_dir)

for i in range(len(split_totals)):
	path = instrumental_splits_dir + '/' + song_name + '_instrumentals_' + str(i) + '_' + str(len(split_instrumentals[i])) + '.wav'
	with open(path, 'wb') as f:
		split_instrumentals[i].export(f, format = 'wav')
	path = vocal_splits_dir  + '/' + song_name + '_vocals_' + str(i) + '_' + str(len(split_vocals[i])) + '.wav'
	with open(path, 'wb') as f:
		split_vocals[i].export(f, format = 'wav')
	path = total_splits_dir  + '/' + song_name + '_totals_' + str(i) + '_' + str(len(split_totals[i])) + '.wav'
	with open(path, 'wb') as f:
		split_totals[i].export(f, format = 'wav')
	