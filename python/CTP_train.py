import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import CTP_utils
from CTP_config import config
from torch.cuda import amp

################################################################################
################################ Optimization ##################################
################################################################################

## Update model for one epoch
def update_model_epoch(train_data, train_labels, model, batch_size, optimizer, loss_function, info=False):

	# Compute total number of batches
	if batch_size == None:
		n_batch = 1
		batch_size = train_data.shape[0]
	else:
		n_batch_full = train_data.shape[0]//batch_size
		if train_data.shape[0]%batch_size != 0: n_batch = n_batch_full+1

	# Create permutation
	permutation = torch.randperm(train_data.shape[0])

	# If train_data size not divisible by batch size
	if info:
		if train_data.shape[0]%batch_size != 0:
			print("Warning: size of training data (", train_data.shape[0], ") is not divisible by batch size (", batch_size, ")")
			print("Number of batches of full size: ", n_batch-1)
			print("Size of last batch: ", train_data.shape[0]-n_batch_full*batch_size)
		else:
			print("Size of training data (", train_data.shape[0], ") is divisible by batch size (", batch_size, ")")
			print("Number of batches of full size: ", n_batch)

	# Loop over mini-batches
	for i_batch in range(n_batch):

		# Get indices for permuation array
		idx_min = i_batch*batch_size
		idx_max = min( (i_batch+1)*batch_size, train_data.shape[0] )
		indices = permutation[idx_min:idx_max]

		############## QC ############
		# print("i batch: ", i_batch)
		# nsamp+=idx_max-idx_min
		# print("i_batch = ", i_batch)
		# print("ind_inf = ", ind_inf)
		# print("ind_sup = ", ind_sup)
		##############################

		# Extract the batch from data and label
		train_data_batch = train_data[indices,:,:]
		train_labels_batch = train_labels[indices,:]

		########################### Single precision ###########################
		# Set gradients to zero
		optimizer.zero_grad()

		# Compute forward pass
		print("train_data_batch shape: ", train_data_batch.shape)
		y_pred_batch = model.forward(train_data_batch)
		print("y_pred_batch shape: ", y_pred_batch.shape)

		# Compute loss
		loss = loss_function(y_pred_batch, train_labels_batch)

		# Compute gradient
		loss.backward()

		# Update model parameters
		optimizer.step()
		########################################################################
		########################### Half precision #############################
		# with amp.autocast():
		#
		# 	# Compute forward pass
		# 	y_pred_batch = model.forward(train_data_batch)
		# 	# Compute loss
		# 	loss = loss_function(y_pred_batch, train_labels_batch)
		#
		# # Set gradients to zero
		# optimizer.zero_grad()
		#
		# # Compute gradient
		# scaler.scale(loss).backward()
		#
		# # Step and update parameters
		# scaler.step(optimizer)
		# scaler.update()
		########################################################################

	return

## Optimziation method
def get_optim(optim_method, model, learning_rate, weight_decay):

	if optim_method == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	elif optim_method == 'sgd':
		# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0002)
		optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	elif optim_method == 'RMSprop':
		optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	else:
		sys.exit('[get_optim] Error: please select a valid optimization method -- exiting')

	return optimizer

## Learning rate decay scheduler
def create_scheduler(optimizer, num_epochs):

	# lr_new = gamma * lr_old if epoch%step_size = 0
	# lr_new = lr_old otherwise
	if config.lr_decay_strategy == 'step':
		scheduler = optim.lr_scheduler.StepLR(optimizer, config.decay_step_size, config.decay_gamma)

	# lr_new = lr_init * lambda(epoch)
	elif config.lr_decay_strategy == 'decay':
		print("decay rate: ", config.decay_rate)
		scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (1.0 + config.decay_rate * epoch))

	# lr_new = lr_init * lambda(epoch)
	elif config.lr_decay_strategy == 'sqrt':
		scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (1.0 + np.sqrt(epoch)))

	# lr_new = gamma * lr_old
	elif config.lr_decay_strategy == 'exp':
		print("decay rate: ", config.decay_gamma)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.decay_gamma)

	else:
		sys.exit('[create_scheduler] Error: please select a valid learning rate scheduler -- exiting')

	return scheduler

### Loss function
def get_loss_fn(loss_fn):

	############################################################################
	# Info from: https://medium.com/analytics-vidhya/a-comprehensive-guide-to-loss-functions-part-1-regression-ff8b847675d6
	############################################################################

	################################## MSE #####################################
	# Mean Squared Error (MSE) = 1/n Sum || y_pred - y ||^2
	# Advantages: for small errors, MSE helps converge to the minima efficiently, as the gradient reduces gradually
	# Drawbacks:
	# - squaring the values does increases the rate of training, but at the same time, an extremely large loss may lead to a drastic jump during backpropagation, which is not desirable
	if loss_fn == 'mse':
		loss_function = nn.MSELoss()

	################################## MAE #####################################
	# Mean Absolute Error (MAE) = 1/n Sum | y_pred - y |
	# Advantages:
	# Drawbacks:
	#            - MAE calculates loss by considering all the errors on the same scale.
	#            - For example, if one of the output is on the scale of hundred while other is on the scale of thousand, our network won’t be able to distinguish between them just based on MAE, and so, it’s hard to alter weights during backpropagation
	#            - MAE is a linear scoring method, i.e. all the errors are weighted equally while calculating the mean. This means that while backpropagation, we may just jump past the minima due to MAE’s steep
	elif loss_fn == 'mae':
		loss_function = nn.L1Loss()

	################################# Huber ####################################
	# Huber loss: L = (y_pre-y)^2 for |y_pred-y| < delta
	#               = 2*delta |y_pred-y| - delta^2 otherwise
	# Advantages:
	#            - Modifiable hyperparameter delta
	#            - Linearity above delta ensures fair weightage to outliers (Not as extreme as in MSE).
	#            - Curved nature below delta ensures the right length of steps during backpropagation
	# Drawbacks:
	#            - Due to additional conditionals and comparisons, Huber loss is comparatively expensive in terms of computations, especially if your dataset is large.
	#            - To get the best results, δ also needs to be optimized, which increases training requirementses the right length of steps during backpropagation
	elif loss_fn == 'huber':
		loss_function = nn.SmoothL1Loss()

	################################ Log-cosh ##################################
	# - Log-cosh is quite similar to Huber loss as it is also a combination of linear and quadratic scorings.
	# - One difference that sets this apart is that it is double differentiable. Some optimization algorithms like XGBoost prefer such functions over Huber, which is differentiable only once. Log-cosh calculates the log of hyperbolic cosine of the error
	# L = sum log (cosh(y - p_pred))

	elif loss_fn == 'log_cosh':
		sys.exit('[get_loss_fn] Error: log cosh loss function not implemented yet -- exiting')

	############################################################################
	else:
		sys.exit('[get_loss_fn] Error: please select a valid loss function -- exiting')

	return loss_function

### Compute loss function
def compute_loss(data, labels, model, loss_function):
	model.eval() # Set eval mode (used for prediction: turns off dropout, etc.)
	with torch.no_grad():
		y_pred = forward_by_batch(data, model, config.batch_size)
		accuracy = compute_accuracy(y_pred, labels, config.ac_threshold)
		loss_value = loss_function(y_pred, labels)
	model.train() # Set mode back to training
	return loss_value, accuracy

### Forward by batch
def forward_by_batch(data, model, batch_size):

	# Only for forward computation
	with torch.no_grad():

		# Allocate output prediction array on device
		y_pred = torch.zeros((data.shape[0],1), device=config.device)

		# Compute total number of batches
		if batch_size == None:
			n_batch = 1
			batch_size = data.shape[0]
		else:
			n_batch_full = data.shape[0]//batch_size
			if data.shape[0]%batch_size != 0: n_batch = n_batch_full+1

		# Loop over mini-batches
		for i_batch in range(n_batch):

			# Get min/max index
			idx_min = i_batch*batch_size
			idx_max = min( (i_batch+1)*batch_size, data.shape[0] )

			# Compute forward pass for this batch
			y_pred[idx_min:idx_max,:] = model.forward(data[idx_min:idx_max,:,:])

	return y_pred

### Compute accuracy
def compute_accuracy(pred, labels, ac_threshold):

	accuracy = torch.sum( torch.abs( (labels-pred)/(labels+config.eps_stability) ).le(ac_threshold) ).item()
	# accuracy = torch.sum( torch.abs( (labels-pred)).le(ac_threshold) ).item()
	accuracy /= pred.shape[0]
	return accuracy

################################################################################
########################### Training and testing ###############################
################################################################################
def train_assess_model(name, all_data, all_labels, model, info=False):

	# Create pointers to data/labels
	train_data = all_data['train']
	train_labels = all_labels['train']

	if 'dev' in all_data.keys():
		dev = True
		dev_data = all_data['dev']
		dev_labels = all_labels['dev']
	else: dev = False

	# Get other parameters from config
	num_epochs = config.num_epochs
	learning_rate = config.learning_rate
	l2_reg_lambda = config.l2_reg_lambda
	batch_size = config.batch_size
	optim_method = config.optim_method
	optimizer = get_optim(optim_method, model, learning_rate, l2_reg_lambda)

	# Learning rate decay scheduler
	print("Before optimizer.param_groups[0]['lr']: ", optimizer.param_groups[0]['lr'])
	if config.lr_decay_strategy != None:
		lr_decay = True
		lr_scheduler = create_scheduler(optimizer, num_epochs)
	else:
		lr_decay = False
	print("After optimizer.param_groups[0]['lr']: ", optimizer.param_groups[0]['lr'])

	# Display optimization parameters
	if info:
		print("-"*30)
		print("Model name: ", name)
		print("Data type: ", type(all_data))
		print("Labels type: ", type(all_labels))
		print("Model: ", model)
		print("Optimization method: ", optim_method)
		print("Number of epochs: ", num_epochs)
		print("Using a learning rate decay strategy")
		print("Batch size: ", batch_size)
		print("Computing loss for dev data: ", dev)
		print("-"*30)

	### Loss function
	loss_function = get_loss_fn(config.loss_function)

	### Inversion metrics
	loss_train = []
	loss_dev = []
	accuracy_train = []
	accuracy_dev = []
	lr_curve =[]

	# Compute initial train loss
	CTP_utils.get_cuda_info()
	loss_value, accuracy = compute_loss(train_data, train_labels, model, loss_function)
	if 'cuda' in config.device: loss_value = loss_value.cpu()
	loss_train.append(loss_value)
	accuracy_train.append(accuracy)

	# Compute initial dev loss
	if dev:
		loss_value_dev, dev_accuracy = compute_loss(dev_data, dev_labels, model, loss_function)
		if 'cuda' in config.device: loss_value_dev = loss_value_dev.cpu()
		loss_dev.append(loss_value_dev)
		accuracy_dev.append(dev_accuracy)
		if config.debug:
			print("Initial dev loss: ", loss_value_dev.item())
			print("Initial dev accuracy: ", accuracy)
	lr_curve.append(optimizer.param_groups[0]['lr'])

	# Display information
	print("Initial learning rate: ", optimizer.param_groups[0]['lr'])
	print("Epoch #", 0, "/", num_epochs)
	print("Initial train loss: ", loss_value.item())
	print("Initial train accuracy: ", accuracy)
	if dev: print("Initial dev loss: ", loss_value_dev.item())
	if dev: print("Initial dev accuracy: ", dev_accuracy)

	# For FP16
	# scaler = amp.GradScaler()

	### Train model for num_epoch
	for i_epoch in range(num_epochs):

		# Update model parameters for one epoch
		update_model_epoch(train_data, train_labels, model, batch_size, optimizer, loss_function, info)

		# Compute loss function on the full train set
		loss_value, accuracy = compute_loss(train_data, train_labels, model, loss_function)
		if 'cuda' in config.device: loss_value = loss_value.cpu()
		loss_train.append(loss_value)
		accuracy_train.append(accuracy)

		# Compute loss function on the full dev batch
		if dev:
			loss_value_dev, dev_accuracy = compute_loss(dev_data, dev_labels, model, loss_function)
			if 'cuda' in config.device: loss_value_dev = loss_value_dev.cpu()
			loss_dev.append(loss_value_dev)
			accuracy_dev.append(dev_accuracy)

		if (i_epoch+1)%config.rec_freq == 0:
			print("-"*40)
			print("Epoch #", i_epoch+1, "/", num_epochs)
			print("Train loss: ", loss_value.item())
			print("Train accuracy: ", accuracy)
			print("Dev loss: ", loss_value_dev.item())
			print("Dev accuracy: ", dev_accuracy)
			print("Learning rate value: ", optimizer.param_groups[0]['lr'])
			CTP_utils.get_cuda_info("GPU info for epoch #"+str(i_epoch))

		if lr_decay:
			lr_scheduler.step()
		lr_curve.append(optimizer.param_groups[0]['lr'])

	return model, loss_train, loss_dev, accuracy_train, accuracy_dev, lr_curve
