import numpy as np
import tensorflow as tf
import time

from loader import Loader

# Construction of feed dict with loader
def make_feed_dict(loader, inputs, val = False):
	mixture, vocals, instrumentals = inputs
	np_vocals, np_instrumentals, np_mixture, data_names = loader.build_data(val)

	return {mixture: np_mixture, vocals: np_vocals, instrumentals: np_instrumentals}

# Logging functions
def setup(hparams, sess, saver):
	sess.run(tf.global_variables_initializer())

	train_writer = tf.summary.FileWriter(hparams.log_dir + "/train", sess.graph)
	eval_writer = tf.summary.FileWriter(hparams.log_dir + "/eval", sess.graph)

	return train_writer, eval_writer

def log_hparams(hparams, writer):
	value = get_hparams_text(hparams.values())
	text_tensor = tf.make_tensor_proto(value, dtype=tf.string)
	meta = tf.SummaryMetadata()
	meta.plugin_data.plugin_name = "text"
	summary = tf.Summary()
	summary.value.add(tag="hyperparameters", metadata=meta, tensor=text_tensor)
	writer.add_summary(summary)

# Main train loop
def run_train(hparams, inputs, loss, train_op):
	# Decode inputs
	mixture, vocals, instrumentals = inputs

	# Prepare data
	loader = Loader(hparams)

	# Set config for GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Setup saver for saving checkpoints
	saver = tf.train.Saver()

	with tf.Session(config = config) as sess:

		# Setup train/eval writers and log hparams
		# train_writer, eval_writer = setup(hparams, sess, saver)
		# log_hparams(hparams, train_writer)
		sess.run(tf.global_variables_initializer())

		with open(hparams.log_file_simple, 'a+') as f:
			f.write('\n**************** NEW TRAIN SESSION ****************\n')

		start = time.time()
		n_evals = 1
		n_saves = 1

		# Iterate through max number of training steps
		for step in range(hparams.max_steps):
			feed_dict = make_feed_dict(loader, inputs)

			# Run session to get loss and gradients
			result = sess.run([loss, train_op], feed_dict = feed_dict)

			raw_loss, _ = result
 
			# Print loss per time step
			if step % hparams.print_loss_frequency == 0:
				print ('%d (%d %d%%) Loss: %.8f' % (time.time() - start, step, float(step) / hparams.max_steps * 100, raw_loss))
				with open(hparams.log_file_simple, 'a+') as f:
					f.write('%d (%d %d%%) Loss: %.8f\n' % (time.time() - start, step, float(step) / hparams.max_steps * 100, raw_loss))

			# Run Eval
			if step % hparams.eval_loss_frequency == 0:
				feed_dict_eval = make_feed_dict(loader, inputs, val = True)
				result = sess.run(loss, feed_dict = feed_dict_eval)
				raw_loss = result
				n_evals += 1

			if step > 10 and (time.time() - start) // hparams.save_model_interval >= n_saves:
				print ("Saving Model")
				saver.save(sess, hparams.save_dir + ('/%s/' % str(step)), step)
				n_saves += 1

			# train_writer.flush()
			# eval_writer.flush()







