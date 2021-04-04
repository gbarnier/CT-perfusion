import os
from torch.utils.data import Dataset
import torch
import h5py
import numpy as np

class ctp_dataset(Dataset):

	def __init__(self, data_file_list):

		# Load data
		hf_file = h5py.File(data_file_list[0], 'r')
		self.data = np.array(hf_file.get("data_train"), dtype=np.float32)
		self.labels = np.array(hf_file.get('tmax_train'), dtype=np.float32)
		print("Length of data file list: ", len(data_file_list))
		if len(data_file_list) > 1:
			for i_file in range(len(data_file_list)-1):
				hf_file = h5py.File(data_file_list[i_file+1], 'r')
				data_temp = np.array(hf_file.get("data_train"), dtype=np.float32)
				self.data = np.concatenate((self.data, data_temp), axis=0)
				labels_temp = np.array(hf_file.get("tmax_train"), dtype=np.float32)
				self.labels = np.concatenate((self.labels, labels_temp), axis=0)
				print("Loading ", data_file_list[i_file+1])

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):

		data_example = self.data[index, :, :]
		label_example = self.labels[index, :]
		return data_example, label_example


	############ Code example on how to iterate of the data loader #############
	# train_data_new = test_data_loader.ctDataset(config.train_file_list)
	# dev_data_new = test_data_loader.ctDataset(config.dev_file_list)
	# config.batch_size = 1024
	# print("config.batch_size: ", config.batch_size)
	# train_loader = DataLoader(dataset=train_data_new, batch_size=config.batch_size, shuffle=True)
	# dev_loader = DataLoader(dataset=dev_data_new, batch_size=config.batch_size, shuffle=True)
	# avg, std = CTP_utils.get_mean_std(train_loader)
	# print("avg: ", avg)
	# print("std: ", std)

	# i=0
	# for data_batch, label_batch in train_loader:
	# 	i+=1
	# 	print("i: ", i)
	# 	print("data_batch: ", data_batch)
	# 	print("label_batch: ", label_batch)
