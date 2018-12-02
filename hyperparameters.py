import tensorflow as tf

FS = 16000
TIME_LENGTH = 1
FRAME_LENGTH = 512

hparams = tf.contrib.training.HParams(

	# Logging Parameters
	log_dir = "logs/",
	log_file_simple = "logs/loss_mir1k.txt",

	# Data Parameters
	# data_dir = "audio/",
	data_dir = "MIR-1K/Wavfile/",
	val_split = 0.8,
	max_input_snr = 5,

	# Waveform Parameters
	Fs = FS,
	time_length = TIME_LENGTH,
	frame_length = FRAME_LENGTH,
	use_log = True,
	magnitude_offset = 1e-6,

	frame_step = int(FRAME_LENGTH / 4),
	num_freqs = int(FRAME_LENGTH / 2) + 1,

	## Make sure the waveform size is a multiple of frame_length
	waveform_size = int(FS * TIME_LENGTH / FRAME_LENGTH) * FRAME_LENGTH,

	model_name = "RNN",

	# RNN Parameters
	use_lstm = False,
	num_layers = 4,
	layer_size = 200,
	clip_gradient = 200,
	num_fc_layers = 3,
	fc_layer_size = 300,

	# AE Parameters


	# Training Parameters
	max_steps = int(2e6),
	learning_rate = 10e-4,
	batch_size = 32,
	optimizer_name = "RMSProp",
	eval_loss_frequency = 8,

	print_loss_frequency = 20,
	save_model_interval = 60,
	save_dir = "checkpoints/",

	)