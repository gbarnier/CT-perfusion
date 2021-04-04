import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import copy

################################################################################
################################## Training ####################################
################################################################################
def train_model_2d(project_folder, project_name):

	# Display information
	info = config.info
	debug = config.debug
	start_time = time.time()
	if info:
		print("User has requested to display information")
		print("Project name: ", project_name)
		print("Project folder: ", project_folder)
		print("Training mode")

	###################### Load data + pre-processing ##########################
	# Load training data
	if config.include_dev:
		train_data, train_labels, dev_data, dev_labels = CTP_utils_2d.load_data_2d(config.include_dev)
		if info: print("Successfully loaded train and dev sets")
	else:
		train_data, train_labels = CTP_utils_2d.load_data_2d(config.include_dev)
		if info: print("Successfully loaded train set")

	### At this point we have the train data/labels loaded into one array
	n_train_data = train_data.shape[0]
	n_time = train_data.shape[2]

	# Display info
	if info:
		print("Number of training examples: ", n_train_data)
		if config.include_dev: print("Number of dev examples: ", dev_data.shape[0])

	# Normalize data
	train_stats = {}
	train_stats['mean'] = np.mean(train_data)
	train_stats['std'] = np.std(train_data)
	train_stats['n_train'] = n_train_data
	train_data = (train_data - train_stats['mean']) / train_stats['std']
	if config.include_dev: dev_data = (dev_data - train_stats['mean']) / train_stats['std']

	if info:
		print("Mean training examples: ", train_stats['mean'])
		print("Standard deviation of training examples: ", train_stats['std'])
		print("Min value train data: ", np.min(train_data))
		print("Max value train data: ", np.max(train_data))
		print("Min value train labels: ", np.min(train_labels))
		print("Max value train labels: ", np.max(train_labels))
		if config.include_dev:
			print("Min value dev data: ", np.min(dev_data))
			print("Max value dev data: ", np.max(dev_data))
			print("Min value dev labels: ", np.min(dev_labels))
			print("Max value dev labels: ", np.max(dev_labels))

	############################# Torch ########################################
	### Convert data and labels into PyTorch tensors
	train_data_torch = CTP_utils.np2torch(train_data, config.device)
	train_labels_torch = CTP_utils.np2torch(train_labels, config.device)
	all_data = {}
	all_labels = {}
	all_data['train'] = train_data_torch
	all_labels['train'] = train_labels_torch

	### Create data dictionary
	if config.include_dev:
		dev_data_torch = CTP_utils.np2torch(dev_data, config.device)
		dev_labels_torch = CTP_utils.np2torch(dev_labels, config.device)
		all_data['dev'] = dev_data_torch
		all_labels['dev'] = dev_labels_torch

	########################### Model instanciation ############################
	# Create model
	model = CTP_models_2d.create_model_2d(config.model_type, n_time)
	model.

	# Get meory allocation info
	CTP_utils_2d.get_cuda_info()

	# Initialize weights
	CTP_models_2d.init_weights_xavier_2d(model)

	# Log type of model in train statistics dictionary
	train_stats['model_type'] = model.name
	train_stats['model_n_param'] = sum(p.numel() for p in model.parameters())

	################# QC #################
	if debug:
		print("len(model.network): ", len(model.network))
		print("len(model.network): ", model.network[0])
		print("len(model.network): ", type(model.network[0].weight.data))
		print("Before model.network[0].weight: ", model.network[0].weight)
		print("Before model.network[0].bias: ", model.network[0].bias)
		print("After model.network[0].weight: ", torch.mean(model.network[0].weight))
		print("After model.network[0].bias: ", torch.mean(model.network[0].bias))

	################################ Training ##################################
	# Train model
	if info: print("Launching training")
	info = False
	model_out, loss_train, loss_dev, accuracy_train, accuracy_dev, lr_curve = CTP_train.fit_model_2d(model.name, all_data, all_labels, model, info=info)

	# Record training information
	training_time = time.time()-start_time
	train_stats['training time'] = training_time
	train_stats['device'] = config.device
	train_stats['optim_method'] = config.optim_method
	train_stats['batch_size'] = config.batch_size
	train_stats['lr_decay_strategy'] = config.lr_decay_strategy
	train_stats['loss_function'] = config.loss_function
	train_stats['seed'] = config.seed
	print("Training time: %2.1f [s]" %(training_time))

	############################ Saving + plotting #############################
	# Plot and save train/dev loss functions
	if config.save_model:
		if info: print("Saving model + loss functions + accuracy")
		CTP_utils.save_model(model_out, train_stats, project_folder, project_name)
		CTP_utils.save_results(loss_train, loss_dev, accuracy_train, accuracy_dev, lr_curve, project_folder, project_name, save=True)


if __name__ == '__main__':

	############################################################################
	############################### Main #######################################
	############################################################################
	# Parse command line
	parser = argparse.ArgumentParser(description=' ')
	parser.add_argument("mode", help="Select between train/test/predict modes", type=str)
	parser.add_argument("project_name", help="Name of the experiment", type=str)
	parser.add_argument("--patient_file", help="Patient's file (h5 format)", type=str)
	parser.add_argument("--patient_id", help="Patient's id", type=str)
	parser.add_argument("--n_epochs", help="Number of epochs for training", type=int)
	parser.add_argument("--optim", help="Optimization method", type=str)
	parser.add_argument("--lr", help="Learning rate", type=float)
	parser.add_argument("--model", help="Model type", type=str)
	parser.add_argument("--at", help="Accuracy threashold (in tenth of a sec)", type=float)
	parser.add_argument("--batch_size", help="Batch size (set to -1 for batch gradient)", type=int)
	parser.add_argument('--lr_decay', type=str)
	parser.add_argument('--device', help="Device for computation ('cpu' or 'cuda')", type=str)
	parser.add_argument('--train_file', help="Name of file for training", type=str)
	parser.add_argument('--train_file_list', help="Name of file for training", type=str)
	parser.add_argument('--dev_file', help="Name of file for dev", type=str)
	parser.add_argument('--dev_file_list', help="Name of file for dev", type=str)
	parser.add_argument('--test_file', help="Name of file for test", type=str)
	parser.add_argument('--test_file_list', help="Name of file for test", type=str)
	parser.add_argument('--decay_rate', help="Decay rate for the decay lr schedule", type=float)
	parser.add_argument('--decay_gamma', help="Decay rate for the exp decay lr schedule", type=float)
	parser.add_argument('--step_size', help="Learning rate update frequency (for step schedule)", type=float)
	parser.add_argument('--loss', help="Loss function", type=str)
	parser.add_argument('--l2_reg_lambda', help="Trade-off parameter for L2-regularization", type=float)
	parser.add_argument('--seed', help="Use a seed to obtain deterministic results", type=int)
	args = parser.parse_args()
	curr_dir = os.getcwd()
	project_folder = curr_dir + '/models2d/' + args.project_name

	# Override config parameters
	if args.n_epochs != None: config.num_epochs = args.n_epochs
	if args.optim != None: config.optim_method = args.optim
	if args.lr != None: config.learning_rate = args.lr
	if args.model != None: config.model_type = args.model
	if args.at != None: config.ac_threshold = args.at
	if args.batch_size != None:
		config.batch_size = None if args.batch_size == -1 else args.batch_size
	if args.at != None: config.ac_threshold = args.at
	config.lr_decay_strategy = args.lr_decay
	if args.device != None: config.device = args.device
	if args.decay_rate != None: config.decay_rate = args.decay_rate
	if args.decay_gamma != None: config.decay_gamma = args.decay_gamma
	if args.step_size != None: config.decay_step_size = args.step_size
	if args.loss != None: config.loss_function = args.loss
	if args.l2_reg_lambda != None: config.l2_reg_lambda = args.l2_reg_lambda
	if args.seed > 0:
		config.seed = args.seed
		CTP_utils.seed_everything(config.seed)

	# Training data information
	config.train_file_list = []
	if args.train_file != None:
		config.train_file_list.append(args.train_file)
	elif args.train_file_list != None:
		with open(args.train_file_list) as f:
			for row in f:
				config.train_file_list.append(row.rstrip('\n') )

	# Dev data information
	if args.dev_file != None:
		config.dev_file_list.append(args.dev_file)
	elif args.dev_file_list != None:
		with open(args.dev_file_list) as f:
			for row in f:
				config.dev_file_list.append(row.rstrip('\n') )

	# Test data information
	if args.test_file != None:
		config.test_file_list.append(args.test_file)
	elif args.test_file_list != None:
		with open(args.test_file_list) as f:
			for row in f:
				config.test_file_list.append(row.rstrip('\n') )

	# Display info
	if config.info:
		print("----Parameters for training----")
		print("Model type: ", config.model_type)
		print("Number of epochs: ", config.num_epochs)
		print("Optimization method: ", config.optim_method)
		print("Learning rate: ", config.learning_rate)
		print("Accuracy threshold: ", config.ac_threshold)
		print("Accuracy stability value: ", config.eps_stability)
		print("Learning rate decay strategy: ", config.lr_decay_strategy)
		print("Device: ", config.device)
		print("Train file_list: ", config.train_file_list)
		print("Dev file_list: ", config.dev_file_list)
		print("Test file_list: ", config.test_file_list)
		print("Batch size: ", config.batch_size)
		print("Loss function: ", config.loss_function)
		if config.seed != None: print("Seed for deterministic results: ", config.seed)

	# Train mode
	if args.mode == 'train':

		# Create working directory for this model
		if path.exists(project_folder): sys.exit('Path for experiment already exist, please choose another name')
		os.mkdir(project_folder)

		# Train model
		train_model_2d(project_folder, args.project_name)
