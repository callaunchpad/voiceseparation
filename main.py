import model
import train
import hyperparameters

def main():

	hparams = hyperparameters.hparams

	print("************************************")
	print("*********** Begin Train ************")
	print("************************************")

	print("Model: %s" % hparams.model_name)
	print("Optimizer: %s" % hparams.optimizer_name)
	print("Data directory: %s" % hparams.data_dir)
	print("Log directory: %s" % hparams.log_dir)

	inputs, loss, train_op = model.build_graph(hparams)
	train.run_train(hparams, inputs, loss, train_op)

if __name__ == '__main__':
	main()