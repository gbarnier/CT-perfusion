from CTP_config import config
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import ast
from numpy import savetxt
import CTP_models
import pyVector
import os
from CTP_config import config
from scipy.ndimage import uniform_filter
import random

# Function taken from CS 234 code
def np2torch(np_array, device, cast_double_to_float=True):

	"""
	Utility function that accepts a numpy array and does the following:
		1. Convert to torch tensor
		2. Move it to the GPU (if CUDA is available)
		3. Optionally casts float64 to float32 (torch is picky about types)
	"""

	# Move to CPU/GPU if requested
	np_array = torch.from_numpy(np_array).to(device)

	if cast_double_to_float and np_array.dtype is torch.float64:
		np_array = np_array.float()
	return np_array

def get_cuda_info(msg=None):
	print("-"*10, "GPU info", "-"*10)
	if msg != None: print(msg)
	if torch.cuda.is_available(): print("A GPU device is available for this session")
	print("Device requested by user: ", config.device)
	print("Current device id: ", torch.cuda.current_device())
	print("Current device name: ", torch.cuda.get_device_name())
	print("Number of available GPUs: ", torch.cuda.device_count())
	print("Allocated memory on GPU: %3.2f [GB]" % (torch.cuda.memory_allocated(device=config.device)/1024**3))
	print("Reserved memory on GPU: %3.2f [GB]" % (torch.cuda.memory_reserved(device=config.device)/1024**3))
	print("Max allocated memory on GPU: %3.2f [GB]" % (torch.cuda.max_memory_allocated(device=config.device)/1024**3))
	print("Max reserved memory on GPU: %3.2f [GB]" % (torch.cuda.max_memory_reserved(device=config.device)/1024**3))
	print("-"*21+"-"*len("GPU info"))
	############################ Other information #############################
	# print("Memory statistics: ", torch.cuda.memory_stats())
	# print("torch.cuda.list_gpu_processes(device=config.device): ",torch.cuda.list_gpu_processes(device=config.device))
	# print("torch.cuda.memory_summary(device=config.device, abbreviated=False)):", torch.cuda.memory_summary(device=config.device, abbreviated=False))
	# print("torch.cuda.memory_snapshot(): ", torch.cuda.memory_snapshot())
	############################################################################

# Plot and save objective functions for loss and accuracy
def save_results(loss_train, loss_dev, accuracy_train, accuracy_dev, lr_curve, exp_folder, exp_name, show=False, save=True):

	# Convert list to numpy array
	loss_train = np.array(loss_train)
	accuracy_train = np.array(accuracy_train)
	lr_curve = np.array(lr_curve)
	if len(loss_dev) > 0:
		loss_dev = np.array(loss_dev)
		accuracy_dev = np.array(accuracy_dev)
		dev = True
	else: dev = False

	# Get paths
	save_path = exp_folder + '/' + exp_name

	# Save non-normalized train/dev loss functions values
	if save == True:
		savetxt(save_path+'_loss_train.csv', loss_train, delimiter=',')
		savetxt(save_path+'_accuracy_train.csv', accuracy_train, delimiter=',')
		savetxt(save_path+'_lr_curve.csv', lr_curve, delimiter=',')
		if dev:
			savetxt(save_path+'_loss_dev.csv', loss_dev, delimiter=',')
			savetxt(save_path+'_accuracy_dev.csv', accuracy_dev, delimiter=',')

	# Create iteration axis
	num_epoch = loss_train.shape[0]
	epoch = np.arange(num_epoch)

	# Non-normalized objective functions
	f1 = plt.figure()
	plt.plot(epoch, loss_train, "b", label="Train loss", linewidth=config.linewidth)
	if dev: plt.plot(epoch, loss_dev, "r", label="Dev loss", linewidth=config.linewidth)
	plt.xlabel("Epochs")
	plt.ylabel(config.loss_function+" loss")
	plt.legend(loc="upper right")
	plt.title("Loss function ("+exp_name+")")
	max_val = np.max(np.maximum(loss_train, loss_dev))
	plt.ylim(0, max_val*1.05)
	plt.grid()
	if show: plt.show()
	f1.savefig(save_path + '_loss_fn.pdf', bbox_inches='tight')

	# Accuracy
	f1 = plt.figure()
	plt.plot(epoch, accuracy_train, "b", label="Accuracy train", linewidth=config.linewidth)
	if dev: plt.plot(epoch, accuracy_dev, "r", label="Accuracy dev", linewidth=config.linewidth)
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend(loc="upper right")
	plt.title("Model accuracy ("+exp_name+")")
	plt.ylim(0,1.05)
	plt.grid()
	if show: plt.show()
	f1.savefig(save_path + '_accuracy_fn.pdf', bbox_inches='tight')

	# Learning rate curve
	f1 = plt.figure()
	# plt.plot(epoch, lr_curve, "-", label="Learning rate schedule")
	plt.plot(epoch, lr_curve, "-")
	plt.xlabel("Epochs")
	plt.ylabel("Learning rate")
	# plt.legend(loc="upper right")
	plt.title("Learning rate ("+exp_name+")")
	plt.ylim(0,lr_curve[0]*1.05)
	plt.grid()
	if show: plt.show()
	f1.savefig(save_path + '_lr_curve.pdf', bbox_inches='tight')

	f1 = plt.figure()
	lr_curve /= lr_curve[0]
	# plt.plot(epoch, lr_curve, "-", label="Normalized learning rate schedule")
	plt.plot(epoch, lr_curve, "-")
	plt.xlabel("Epochs")
	plt.ylabel("Normalized learning rate")
	# plt.legend(loc="upper right")
	plt.title("Normalized learning rate  ("+exp_name+")")
	plt.ylim(0,1.05)
	plt.grid()
	if show: plt.show()
	f1.savefig(save_path + '_lr_curve_norm.pdf', bbox_inches='tight')

	# Normalized objective functions
	loss_train /= loss_train[0]
	if dev: loss_dev /= loss_dev[0]
	f2 = plt.figure()
	plt.plot(epoch, loss_train, "b", label="Normalized train loss", linewidth=config.linewidth)
	if dev: plt.plot(epoch, loss_dev, "r", label="Normalized dev loss", linewidth=config.linewidth)
	plt.xlabel("Epochs")
	plt.ylabel("Normalized"+config.loss_function+" loss")
	plt.legend(loc="upper right")
	plt.title("Normalized loss function ("+exp_name+")")
	plt.ylim(0,1.05)
	plt.grid()
	if show: plt.show()
	f2.savefig(save_path + '_loss_fn_norm.pdf', bbox_inches='tight')

# Save model parameters
def save_model(model, train_stats, exp_folder, exp_name):

	# Save model parameters
	save_path_model = exp_folder + '/' + exp_name + '_model.mod'
	torch.save(model.state_dict(), save_path_model)

	# Save statistics
	save_path_stats = exp_folder + '/' + exp_name + '_model_stats.csv'
	f = open(save_path_stats,"w")
	f.write(str(train_stats))
	f.close()

# Save accuracy
def save_accuracy(accuracy, exp_folder, exp_name):

	save_path_accuracy = exp_folder + '/' + exp_name + '_test_time_accuracy.csv'
	f = open(save_path_accuracy,"w")
	f.write(str(accuracy))
	f.close()

# Save loss
def save_loss(loss, exp_folder, exp_name):

	save_path_loss = exp_folder + '/' + exp_name + '_test_time_loss.csv'
	f = open(save_path_loss,"w")
	f.write(str(loss))
	f.close()

# Save model parameters
def load_model(exp_folder, exp_name, n_time):

	# Load dictionary
	load_path_stats = exp_folder +'/'+ exp_name + '_model_stats.csv'
	print("load_path_stats: ", load_path_stats)
	file = open(load_path_stats, "r")
	contents = file.read()
	model_stats = ast.literal_eval(contents)
	file.close()

	# Load model
	load_path_model = exp_folder + '/'+ exp_name + '_model.mod'
	print("load_path_model: ", load_path_model)
	model = CTP_models.create_model(config.model_type, n_time)
	model.load_state_dict(torch.load(load_path_model))
	model.eval()
	return model_stats, model

# Function loads train and dev data
def load_data_labels(include_dev, include_test=False, debug=True):


	################################# Training #################################

	################################# Debug ####################################
	# Case for 1 training file
	# print("type config.train_file_list[0]: ", type(config.train_file_list[0]))
	# print("config.train_file_list[0]: ", len(config.train_file_list[0]))
	# print("config.train_file_list[0]: ", len('/net/server2/sep/gbarnier/projects/CTperf/dat/S00243-train-win1.h5'))
	# print("config.train_file_list[0]: ", config.train_file_list[0])
	# print("config.train_file_list[0]: ", '/net/server2/sep/gbarnier/projects/CTperf/dat/S00243-train-win1.h5')
	# hf_train = h5py.File('/net/server2/sep/gbarnier/projects/CTperf/dat/S00243-train-win1.h5', 'r')
	################################# End debug ################################

	hf_train = h5py.File(config.train_file_list[0], 'r')
	train_data = np.array(hf_train.get("data_train"), dtype=np.float32)
	n_train = train_data.shape[0]
	train_labels = np.array(hf_train.get('tmax_train'), dtype=np.float32)
	print("Length of train file list: ", len(config.train_file_list))

	# Case where we have more than one file to train on
	if len(config.train_file_list) > 1:

		for i_file in range(len(config.train_file_list)-1):

			# Read h5 file corresponding to train file list
			hf_train = h5py.File(config.train_file_list[i_file+1], 'r')
			# Get numpy array for training data
			train_data_temp = np.array(hf_train.get("data_train"), dtype=np.float32)
			# Concatenate old/new arrays
			train_data = np.concatenate((train_data, train_data_temp), axis=0)
			# Update number of training examples
			n_train += train_data_temp.shape[0]
			# Same stuff for labels
			train_labels_temp = np.array(hf_train.get("tmax_train"), dtype=np.float32)
			train_labels = np.concatenate((train_labels, train_labels_temp), axis=0)
			print("Loading ", config.train_file_list[i_file+1])


	if debug:
		print("train_data shape: ", train_data.shape)
		print("n_train: ", n_train)

	# Shuffle total data
	np.random.seed(config.seed_id)
	p = np.random.permutation(n_train)
	train_data = train_data[p,:,:]
	train_labels = train_labels[p,:]

	################################## Dev #####################################
	if include_dev:

		# Reading training data hf
		hf_dev = h5py.File(config.dev_file_list[0], 'r')
		dev_data = np.array(hf_dev.get("data_train"), dtype=np.float32)
		dev_labels = np.array(hf_dev.get('tmax_train'), dtype=np.float32)
		n_dev = dev_data.shape[0]
		print("Length of dev file list: ", len(config.dev_file_list))

		# Case where we have more than one file to train on
		if len(config.dev_file_list) > 1:

			for i_file in range(len(config.dev_file_list)-1):

				# Read h5 file corresponding to train file list
				hf_dev = h5py.File(config.dev_file_list[i_file+1], 'r')
				# Get numpy array for training data
				dev_data_temp = np.array(hf_dev.get("data_train"), dtype=np.float32)
				# Concatenate old/new arrays
				dev_data = np.concatenate((dev_data, dev_data_temp), axis=0)
				# Update number of training examples
				n_dev += dev_data_temp.shape[0]
				# Same stuff for labels
				dev_labels_temp = np.array(hf_dev.get("tmax_train"), dtype=np.float32)
				dev_labels = np.concatenate((dev_labels, dev_labels_temp), axis=0)
				print("Loading ", config.dev_file_list[i_file+1])

		if debug:
			print("dev_data shape: ", dev_data.shape)
			print("n_dev: ", n_dev)
			print("shape dev data: ", dev_data.shape)
			print("shape dev labels: ", dev_labels.shape)

	################################## Test ####################################
	if include_test:

		# Reading training data hf
		hf_test = h5py.File(config.test_file_list[0], 'r')
		test_data = np.array(hf_test.get("data_train"), dtype=np.float32)
		test_labels = np.array(hf_test.get('tmax_train'), dtype=np.float32)
		n_test = test_data.shape[0]
		print("Length of test file list: ", len(config.test_file_list))

		# Case where we have more than one file to train on
		if len(config.test_file_list) > 1:

			for i_file in range(len(config.test_file_list)-1):

				# Read h5 file corresponding to train file list
				hf_test = h5py.File(config.test_file_list[i_file+1], 'r')
				# Get numpy array for training data
				test_data_temp = np.array(hf_test.get("data_train"), dtype=np.float32)
				# Concatenate old/new arrays
				test_data = np.concatenate((test_data, test_data_temp), axis=0)
				# Update number of training examples
				n_test += test_data_temp.shape[0]
				# Same stuff for labels
				test_labels_temp = np.array(hf_test.get("tmax_train"), dtype=np.float32)
				test_labels = np.concatenate((test_labels, test_labels_temp), axis=0)
				print("Loading ", config.test_file_list[i_file+1])

		if debug:
			print("test_data shape: ", test_data.shape)
			print("n_test: ", n_test)
			print("shape test data: ", test_data.shape)
			print("shape test labels: ", test_labels.shape)

	################################## Output ##################################
	if include_dev and include_test:
		return train_data, train_labels, dev_data, dev_labels, test_data, test_labels

	elif include_dev and not include_test:
		return train_data, train_labels, dev_data, dev_labels

	else:
		return train_data, train_labels

# Function loads train and dev data
def load_patient_data(patient_file, patient_id, info=True):

	# Reading patient's file (.h5)
	print("patient_file: ", patient_file)
	hf_patient = h5py.File(patient_file, 'r')

	# Load CT data
	data_patient_4d = np.array(hf_patient.get("4d_cube"), dtype=np.float32)

	# Load tmax if available
	if "tmax_train" in hf_patient.keys():

		# Read data/labels
		tmax_patient_2d = np.array(hf_patient.get("tmax_train"), dtype=np.float32)
		data_patient_3d = np.array(hf_patient.get("data_train"), dtype=np.float32)
		xyz_index = np.array(hf_patient.get("xyz_index"), dtype=np.float32)
		if info: print("Patient recognized: ",  patient_id, "sucessfully loaded patient's data and labels (in training format)")

		return data_patient_3d, tmax_patient_2d, xyz_index

	else:

		# Creating the mask based on CT values
		mask = 1.0 * (np.mean(data_patient_4d, axis=0) < config.ct_sup)
		mask *= 1.0 * (np.mean(data_patient_4d, axis=0) > config.ct_inf)

		# Compute the number of non-zero value in the mask
		n_data = int(np.sum(mask))
		if info: print("number of data points predicted for patient: ", n_data)
		data_patient_3d = np.zeros((n_data, 1, data_patient_4d.shape[0])) # 2D array that contains the CT image for all voxels
		xyz_index = np.zeros((n_data, 3)) # 2D arrays that contain the (x,y,z) index

		# Reshape array for prediction (PyTorh format)
		nx = data_patient_4d.shape[3]
		ny = data_patient_4d.shape[2]
		nz = data_patient_4d.shape[1]
		i_train = 0
		for iz in range(nz):
			for iy in range(ny):
				for ix in range(nx):
					if ct_mask[iz,iy,ix] > 0.0:
						data_patient_3d[i_train, 0, :] = data_patient_4d[:,iz,iy,ix]
						xyz_index[i_train,2] = iz # Record the index on the z-axis for this example
						xyz_index[i_train,1] = iy # Record the index on the y-axis for this example
						xyz_index[i_train,0] = ix # Record the index on the x-axis for this example
						i_train+=1

		# Deallocate memory
		del data_patient_4d
		tmax_patient_2d = []

		return data_patient_3d, tmax_patient_2d, xyz_index


# Function loads train and dev data
def save_predicted_tmax(tmax_pred, labels_2d, xyz_index, project_folder, project_name, patient_file, patient_id, info=True):

	# Allocate 4d numpy array and set values to -1
	hf_patient = h5py.File(patient_file, 'r')
	x_axis = np.array(hf_patient.get("x_axis"), dtype=np.float32)
	y_axis = np.array(hf_patient.get("y_axis"), dtype=np.float32)
	z_axis = np.array(hf_patient.get("z_axis"), dtype=np.float32)

	# Get axes parameters
	ox = x_axis[0]
	oy = y_axis[0]
	oz = z_axis[0]
	dx=1; dy=1; dz=1;
	if len(x_axis)>1: dx = x_axis[1]-x_axis[0]
	if len(y_axis)>1: dy = y_axis[1]-y_axis[0]
	if len(z_axis)>1: dz = z_axis[1]-z_axis[0]

	# Allocate array for output tmax map
	tmax_pred_map = np.zeros((len(z_axis), len(y_axis), len(x_axis)))
	tmax_diff_zone = np.zeros((len(z_axis), len(y_axis), len(x_axis)))
	mask_used = np.zeros((len(z_axis), len(y_axis), len(x_axis)))

	# Fill in values from tmax_pred -> tmax_pred_map
	idx = 0
	for idx in range(xyz_index.shape[0]):
		x_index = int(xyz_index[idx,0])
		y_index = int(xyz_index[idx,1])
		z_index = int(xyz_index[idx,2])
		tmax_pred_map[z_index,y_index,x_index] = tmax_pred[idx,0]
		mask_used[z_index,y_index,x_index] = 1.0
		diff_temp = np.abs(tmax_pred[idx,0]-labels_2d[idx,0])/(labels_2d[idx,0]+config.eps_stability)
		if diff_temp < config.ac_threshold_zone1:
			tmax_diff_zone[z_index,y_index,x_index]=0
		elif diff_temp >= config.ac_threshold_zone1 and diff_temp < config.ac_threshold_zone2:
			tmax_diff_zone[z_index,y_index,x_index]=1
		elif diff_temp >= config.ac_threshold_zone2 and diff_temp < config.ac_threshold_zone3:
			tmax_diff_zone[z_index,y_index,x_index]=2
		else:
			tmax_diff_zone[z_index,y_index,x_index]=3
	### debug
	# true_tmax = np.array(hf_patient.get("tmax"), dtype=np.float32)
	# for iz in range(len(z_axis)):
	# 	for iy in range(len(y_axis)):
	# 		for ix in range(len(x_axis)):
	# 			if tmax_pred_map[iz, iy, ix] <= 0.0:
	# 				tmax_pred_map[iz,iy,ix] = true_tmax[iz,iy,ix]

	# Apply spatial averaging on predicted TMax
	tmax_pred_map_avg = spatial_ave_image3d(tmax_pred_map, config.avg_size)

	# HF files
	out_file = project_folder+'/'+project_name+'_'+patient_id+'_pred_tmax.h5'
	if info: print("Writing predicted TMax map to h5 format: ", out_file)
	hf = h5py.File(out_file, 'w')
	hf.create_dataset('tmax', data=tmax_pred_map)
	hf.create_dataset('tmax_avg', data=tmax_pred_map_avg)
	hf.create_dataset('x_axis', data=x_axis)
	hf.create_dataset('y_axis', data=y_axis)
	hf.create_dataset('z_axis', data=z_axis)
	hf.create_dataset('mask_used', data=mask_used)
	hf.create_dataset('tmax_diff_zone', data=tmax_diff_zone)
	hf.close()

	# SEP file
	out_file_sep = project_folder+'/'+project_name+'_'+patient_id+'_pred_tmax_sep.H'
	out_file_avg_sep = project_folder+'/'+project_name+'_'+patient_id+'_pred_tmax_avg_sep.H'
	out_file_sep_mask = project_folder+'/'+project_name+'_'+patient_id+'_mask_used_sep.H'
	out_file_sep_zone = project_folder+'/'+project_name+'_'+patient_id+'_pred_tmax_diff_zone.H'
	if info: print("Writing predicted TMax map to SEP format: ", out_file_sep)
	hf = h5py.File(out_file, 'r')
	tmax_pred_map = np.array(hf.get("tmax"), dtype=np.float32)
	vec = pyVector.vectorIC(tmax_pred_map)
	vec.writeVec(out_file_sep)
	command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file_sep
	os.system(command)
	tmax_pred_avg_map = np.array(hf.get("tmax_avg"), dtype=np.float32)
	vec = pyVector.vectorIC(tmax_pred_avg_map)
	vec.writeVec(out_file_avg_sep)
	command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file_avg_sep
	os.system(command)
	mask_used = np.array(hf.get("mask_used"), dtype=np.float32)
	vec = pyVector.vectorIC(mask_used)
	vec.writeVec(out_file_sep_mask)
	tmax_diff_zone = np.array(hf.get("tmax_diff_zone"), dtype=np.float32)
	vec = pyVector.vectorIC(tmax_diff_zone)
	vec.writeVec(out_file_sep_zone)
	command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file_sep_zone
	os.system(command)

# Function loads train and dev data
def load_patient_data_debug(patient_file, info=True):

	# Reading patient's file (.h5)
	print("patient_file: ", patient_file)
	hf_patient = h5py.File(patient_file, 'r')

	# Load mask computed with tmax
	mask = np.array(hf_patient.get("tmax_mask"), dtype=np.float32)
	print("shape mask: ", mask.shape)

	# Load CT data
	data_patient_4d = np.array(hf_patient.get("4d_cube"), dtype=np.float32)
	patient_tmax_3d = np.array(hf_patient.get("tmax"), dtype=np.float32)
	patient_train_data = np.array(hf_patient.get("data_train"), dtype=np.float32)
	patient_train_tmax = np.array(hf_patient.get("tmax_train"), dtype=np.float32)
	print("data_patient_4d shape: ", data_patient_4d.shape)
	print("patient_tmax_3d shape: ", patient_tmax_3d.shape)
	print("patient_train_data shape: ", patient_train_data.shape)
	print("patient_train_tmax shape: ", patient_train_tmax.shape)

	# Compute the number of non-zero value in the mask
	n_data = int(np.sum(mask))
	if info: print("n data: ", n_data)
	data_patient_3d = np.zeros((n_data, 1, data_patient_4d.shape[0])) # 3D array that contains the CT image for all voxels
	tmax_2d = np.zeros((n_data, 1, 1)) # 3D array that contains tmax for all voxels
	xyz_index = np.zeros((n_data, 3)) # 2D arrays that contain the (x,y,z) index

	# Reshape array for prediction (PyTorh format)
	nx = data_patient_4d.shape[3]
	ny = data_patient_4d.shape[2]
	nz = data_patient_4d.shape[1]
	i_train = 0
	for iz in range(nz):
		for iy in range(ny):
			for ix in range(nx):
				if mask[iz,iy,ix] > 0.0:
					data_patient_3d[i_train, 0, :] = data_patient_4d[:,iz,iy,ix]
					tmax_2d[i_train, 0, :] = patient_tmax_3d[iz,iy,ix]
					xyz_index[i_train,2] = iz # Record the index on the z-axis for this example
					xyz_index[i_train,1] = iy # Record the index on the y-axis for this example
					xyz_index[i_train,0] = ix # Record the index on the x-axis for this example
					i_train+=1

	# Deallocate memory
	del data_patient_4d

	return data_patient_3d, tmax_2d, xyz_index

def spatial_ave_image3d(image, size=2):
	"""Function to spatially average TMax map"""
	output_slice = uniform_filter(image[0, :, :], size=size, mode="constant")
	output_image = np.zeros((image.shape[0], output_slice.shape[0], output_slice.shape[1]))
	print("output_slice shape:", output_slice.shape)
	print("output_image shape:", output_image.shape)
	for depth_idx in range(image.shape[0]):
		output_image[depth_idx, :, :] = uniform_filter(image[depth_idx, :, :], size=size)
	return output_image

# Function that computes mean and standard deviation
def get_mean_std(loader):
	# Compute mean and standard deviation by batches
	mean_data = 0
	std_data = 0
	n_batch = 0

	for data,_ in loader:
		mean_data += torch.mean(data)
		std_data += torch.mean(data**2)
		n_batch += 1

	mean_data = mean_data / n_batch
	std_data = (std_data / n_batch - mean_data**2)**0.5
	return mean_data, std_data

# Set the same random numbers
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
