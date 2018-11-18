import tensorflow as tf

hparams = tf.contrib.training.HParams(

	# Logging Parameters
	log_dir = "logs/",

	# Data Parameters
	data_dir = "audio/",

	# Waveform Parameters
	FS = 8000,
	TIME_LENGTH = 2,
	FRAME_LENGTH = 512,
	use_log = True,
	magnitude_offset = 1e-6,

	N_C = 64,

	frame_step = int(FRAME_LENGTH / 4),
	num_freqs = int(FRAME_LENGTH / 2) + 1,

	## Make sure the waveform size is a multiple of FRAME_LENGTH
	waveform_size = int(FS * TIME_LENGTH / FRAME_LENGTH) * FRAME_LENGTH,

	model_name = "RNN",

	# RNN Parameters
	use_lstm = False,
	num_layers = 3,
	layer_size = 200,
	clip_gradient = 200,

	# AE Parameters


	# Training Parameters
	max_steps = int(2e6),
	learning_rate = 10e-4,
	batch_size = 16,
	optimizer_name = "RMSProp",

	print_loss_frequency = 20,
	save_model_interval = 600,
	save_dir = "checkpoints/"

	)