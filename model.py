import numpy as np
import tensorflow as tf

# Pre/postprocessing functions
def preprocess_stft(hparams, waveform):
	STFT_waveform = tf.contrib.signal.stft(waveform, hparams.frame_length, hparams.frame_step)
	X_waveform, P_waveform = tf.abs(STFT_waveform), tf.angle(STFT_waveform)
	if hparams.use_log:
		X_waveform = tf.log(X_waveform + hparams.magnitude_offset)
	return X_waveform, P_waveform

# Net related functions
def make_rnn(hparams, X_mixture):
	cells = []
	for _ in range(hparams.num_layers):
		if hparams.use_lstm:
			cells.append(tf.contrib.rnn.BasicLSTMCell(hparams.layer_size))
		else:
			cells.append(tf.contrib.rnn.BasicRNNCell(hparams.layer_size))
	multi_rnn = tf.contrib.rnn.MultiRNNCell(cells)
	# may need to add initial state here
	outputs, last_out = tf.nn.dynamic_rnn(
							multi_rnn,
							X_mixture,
							dtype = tf.float32
						)
	return outputs

def make_net(hparams, X_mixture):
	if hparams.model_name == "RNN":
		output_mask = make_rnn(hparams, X_mixture)
	elif hparams.model_name == "AE":
		output_mask = make_ae(hparams, X_mixture)
	else:
		raise Exception("Unrecognized model name in hyperparameters.")
	return output_mask

# Optimizer function
def create_optimizer(hparams, loss):
	global_step = tf.Variable(0, trainable = False)
	learning_rate = hparams.learning_rate

	if hparams.optimizer_name == "RMSProp":
		optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
	elif hparams.optimizer_name == "Adam":
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	else:
		raise Exception("Unrecognized optimizer name in hyperparameters.")

	variables = tf.trainable_variables()
	gradients = tf.gradients(loss, variables)
	if hparams.clip_gradient > -1:
		gradients = [None if gradient is None else 
					 tf.clip_by_norm(gradient, hparams.clip_gradient) for gradient in gradients]

	train_op = optimizer.apply_gradients(zip(gradients, variables), global_step = global_step)

	return global_step, learning_rate, train_op

# Build and return graph
def build_graph(hparams):

	# Defining tensor for input mixture
	mixture_shape = [hparams.batch_size, hparams.waveform_size]
	mixture = tf.placeholder(tf.float32, shape = mixture_shape)

	# Defining tensors for label vocals and instrumentals
	vocals_shape = [hparams.batch_size, hparams.waveform_size]
	vocals = tf.placeholder(tf.float32, shape = vocals_shape)

	instrumentals_shape = [hparams.batch_size, hparams.waveform_size]
	instrumentals = tf.placeholder(tf.float32, shape = instrumentals_shape)

	# Preprocess input mixture with STFT
	X_mixture, P_mixture = preprocess_stft(hparams, mixture)

	# Preprocess label vocals and instrumentals with STFT
	X_vocals, P_vocals = preprocess_stft(hparams, vocals)
	X_instrumentals, P_instrumentals = preprocess_stft(hparams, instrumentals)

	# Build Neural Network
	output_mask = make_net(hparams, X_mixture)

	# Get vocals and instrumentals estimate from mask
	X_vocals_estimate = output_mask * X_mixture
	X_instrumentals_estimate = (tf.ones(instrumentals_shape) - output_mask) * X_mixture

	# Loss
	loss = tf.reduce_mean(tf.square(X_vocals - X_vocals_estimate) + tf.square(X_instrumentals - X_instrumentals_estimate))

	# Optimization
	global_step, learning_rate, train_op = create_optimizer(hparams, loss)

	# Create input tuple
	inputs = (mixture, vocals, instrumentals)

	return inputs, loss, train_op




