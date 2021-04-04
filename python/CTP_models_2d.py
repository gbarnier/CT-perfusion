import torch
import torch.nn as nn
import os, sys
# from CTP_config import config
import CTP_utils

################################################################################
################################## Model definition(s) #########################
################################################################################
# Function that creates neural network
def create_model_2d(model_type, nt, device, info=True):

	print("model_type: ", model_type)

	if model_type == 'EGNET':
		model = EGNET(nt)
		model.to(device)

	else: sys.exit("Model requested: ", model_type ,"Please provide a valid model type")

	# Set number of parameters
	if info:
		print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
		print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

	return model

class time_encoder(nn.Module):

	def __init__(self, nt):

		# Call parent class constructor
		super(time_encoder, self).__init__()

		################################ Layer 1 ###############################
		# Conv1
		print("Input conv1d l1: ", nt)
		n_channels_in = 1 # Number of input channels for the time signals
		nf_conv1d_l1 = 32
		f_conv1d_l1 = 4
		s_conv1d_l1 = 1
		p_conv1d_l1 = 0
		conv1d_l1 = nn.Conv1d(n_channels_in, nf_conv1d_l1, f_conv1d_l1, stride=s_conv1d_l1, padding=p_conv1d_l1)
		n_out_conv1d_l1 = int( (nt+2*p_conv1d_l1-f_conv1d_l1) / s_conv1d_l1) + 1
		print("Output conv1d l1: ", n_out_conv1d_l1)

		# MaxPool1
		print("Input pool l1: ", n_out_conv1d_l1)
		f_pool_l1 = 2
		s_pool_l1 = f_pool_l1
		p_pool_l1 = 0
		max_pool_l1 = nn.MaxPool1d(f_pool_l1, stride=s_pool_l1, padding=p_pool_l1)
		n_out_pool_l1 = int( (n_out_conv1d_l1+2*p_pool_l1-f_pool_l1)/s_pool_l1 ) + 1
		print("Output pool l1: ", n_out_pool_l1)

		################################ Layer 2 ###############################
		# Conv2
		print("Input conv1d l2: ", n_out_pool_l1)
		nf_conv1d_l2 = 64
		f_conv1d_l2 = 4
		s_conv1d_l2 = 1
		p_conv1d_l2 = 0
		conv1d_l2 = nn.Conv1d(nf_conv1d_l1, nf_conv1d_l2, f_conv1d_l2, stride=s_conv1d_l2, padding=p_conv1d_l2)
		n_out_conv1d_l2 = int( (n_out_pool_l1+2*p_conv1d_l2-f_conv1d_l2) / s_conv1d_l2) + 1
		print("Output conv1d l2: ", n_out_conv1d_l2)

		# MaxPool2
		print("Input pool l2: ", n_out_conv1d_l2)
		f_pool_l2 = 2
		s_pool_l2 = f_pool_l2
		p_pool_l2 = 0
		max_pool_l2 = nn.MaxPool1d(f_pool_l2, stride=s_pool_l2, padding=p_pool_l2)
		n_out_pool_l2 = int( (n_out_conv1d_l2+2*p_pool_l2-f_pool_l2)/s_pool_l2 ) + 1
		print("Output pool l2: ", n_out_pool_l2)

		################################ Layer 3 ###############################
		# Conv3
		print("Input conv1d l3: ", n_out_pool_l2)
		nf_conv1d_l3 = 64
		f_conv1d_l3 = 4
		s_conv1d_l3 = 1
		p_conv1d_l3 = 0
		conv1d_l3 = nn.Conv1d(nf_conv1d_l2, nf_conv1d_l3, f_conv1d_l3, stride=s_conv1d_l3, padding=p_conv1d_l3)
		n_out_conv1d_l3 = int( (n_out_pool_l2+2*p_conv1d_l3-f_conv1d_l3) / s_conv1d_l3) + 1
		print("Output conv1d l3: ", n_out_conv1d_l3)

		# MaxPool3
		print("Input pool l3: ", n_out_conv1d_l3)
		f_pool_l3 = 2
		s_pool_l3 = f_pool_l3
		p_pool_l3 = 0
		max_pool_l3 = nn.MaxPool1d(f_pool_l3, stride=s_pool_l3, padding=p_pool_l3)
		n_out_pool_l3 = int( (n_out_conv1d_l3+2*p_pool_l3-f_pool_l3)/s_pool_l3 ) + 1
		print("Output pool l3: ", n_out_pool_l3)

		################################ Layer 4 ###############################
		# Conv4
		print("Input conv1d l4: ", n_out_pool_l3)
		nf_conv1d_l4 = 64
		f_conv1d_l4 = 4
		s_conv1d_l4 = 1
		p_conv1d_l4 = 0
		conv1d_l4 = nn.Conv1d(nf_conv1d_l3, nf_conv1d_l4, f_conv1d_l4, stride=s_conv1d_l4, padding=p_conv1d_l4)
		n_out_conv1d_l4 = int( (n_out_pool_l3+2*p_conv1d_l4-f_conv1d_l4) / s_conv1d_l4) + 1
		print("Output conv1d l4: ", n_out_conv1d_l4)

		# MaxPool4
		print("Input pool l4: ", n_out_conv1d_l4)
		f_pool_l4 = 4
		s_pool_l4 = 1
		p_pool_l4 = 0
		max_pool_l4 = nn.MaxPool1d(f_pool_l4, stride=s_pool_l4, padding=p_pool_l4)
		n_out_pool_l4 = int( (n_out_conv1d_l4+2*p_pool_l4-f_pool_l4)/s_pool_l4 ) + 1
		print("Output pool l4: ", n_out_pool_l4)

		# Model construction
		self.time_encoder_network = nn.Sequential(
			conv1d_l1,nn.ReLU(),max_pool_l1,
			conv1d_l2,nn.ReLU(),max_pool_l2,
			conv1d_l3,nn.ReLU(),max_pool_l3,
			conv1d_l4,nn.ReLU(),max_pool_l4,
			nn.Flatten()
		)

	# Define forward pass
	def forward(self, x):
		return self.time_encoder_network(x)

class double_conv2d(nn.Module):

	def __init__(self, in_channels, out_channels):

		# Call parent class constructor
		super(double_conv2d, self).__init__()
		test = nn.BatchNorm2d(out_channels)

		# Define one conv block
		f = 3 # Filter size
		s = 1
		p = 1 # => 'same' convolution
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, f, stride=s, padding=p, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, f, stride=s, padding=p, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
			)

	# Define forward pass
	def forward(self, x):
		return self.conv(x)

class EGNET(nn.Module):

	# Constructor
	def __init__(self, nt):

		# Call to the __init__ function of the super class
		super(EGNET, self).__init__()

		########################### Model parameters ###########################
		self.name = 'EGNET'
		self.nt = nt
		self.n_channels_in = 64

		########################### Time encoder ###############################
		self.time_encoding = time_encoder(nt)
		print("Number of parameters for time encoder: ", sum(p.numel() for p in self.time_encoding.parameters()))
		print("Number of trainable parameters for time encoder: ", sum(p.numel() for p in self.time_encoding.parameters() if p.requires_grad))
		######################## Spatial encoder/decoder #######################
		# Encoding
		self.down = double_conv2d(self.n_channels_in, 128)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		print("self.n_channels_in: ", self.n_channels_in)
		print("Number of parameters for down: ", sum(p.numel() for p in self.down.parameters()))
		print("Number of trainable parameters for down: ", sum(p.numel() for p in self.down.parameters() if p.requires_grad))

		# Bottleneck
		self.bottleneck = double_conv2d(128, 128)
		print("Number of parameters for bottleneck: ", sum(p.numel() for p in self.bottleneck.parameters()))
		print("Number of trainable parameters for bottleneck: ", sum(p.numel() for p in self.bottleneck.parameters() if p.requires_grad))
		# Decoding
		self.ups = nn.ModuleList()
		print("self.n_channels_in: ", self.n_channels_in)
		self.ups.append(nn.ConvTranspose2d(128, self.n_channels_in, kernel_size=2, stride=2, padding=0))
		self.ups.append(double_conv2d(192, self.n_channels_in))
		print("Number of trainable parameters for ups: ", sum(p.numel() for p in self.ups.parameters() if p.requires_grad))

		# Last layer
		self.last_conv_layer = nn.Sequential(
			nn.Conv2d(self.n_channels_in, 1, 1, stride=1, padding=0),
			nn.ReLU()
		)
		print("Number of parameters in last layer: ", sum(p.numel() for p in self.last_conv_layer.parameters()))
		print("Number of trainable parameters in last layer: ", sum(p.numel() for p in self.last_conv_layer.parameters() if p.requires_grad))

		# Display dimensions for QC purposes
		print("self.n_channels_in: ", self.n_channels_in)
		print("self.nt: ", self.nt)

	# Forward pass
	def forward(self, x):

		# Check dimensions of input
		n_batch = x.shape[0]
		ny = x.shape[1]
		nx = x.shape[2]
		nt = x.shape[3]
		if nt != self.nt: sys.exit('Please provide a consistent number of time steps')

		# Applying time encoding
		x = torch.reshape(x, (n_batch*ny*nx, 1, nt))
		x = self.time_encoding(x)
		print("shape x after time encoding: ", x.shape)
		if x.shape[1] != self.n_channels_in: sys.exit('The number of output channels after time encoding is not consistent')
		x = torch.reshape(x, (n_batch, ny, nx, x.shape[1]))
		x = x.permute(0,3,1,2)
		print("shape x after time encoding and reshape: ", x.shape)

		# Apply first convolutional layer
		x = self.down(x)
		print("x shape after down: ", x.shape)

		# Save skip connection
		skip_connections = []
		skip_connections.append(x)
		print("skip_connections[0] shape:  ", skip_connections[0].shape)

		# Apply pooling layer
		x = self.pool(x)
		print("x shape after pool: ", x.shape)

		# Apply bottleneck
		x = self.bottleneck(x)
		print("x shape after bottleneck: ", x.shape)

		# Apply transpose
		x = self.ups[0](x)
		print("x shape after ups[0]: ", x.shape)

		# Concatenate along the channel dimension
		# The input shape is (batch, channel, heigth, width)
		concat_skip = torch.cat((skip_connections[0], x), dim=1)
		print("concat_skip shape: ", concat_skip.shape)
		# Apply double conv
		x = self.ups[1](concat_skip)
		print("x shape after ups[1]: ", x.shape)

		# Apply last convolutional layer
		x = self.last_conv_layer(x)
		print("x shape last: ", x.shape)

		return x

	def initialize_weights(self):

		for m in self.modules():

			print("m: ", m)

			if isinstance(m, nn.Conv1d):
				nn.init.xavier_uniform_(m.weight)

			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)

				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

			elif isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight)
				nn.init.constant_(m.bias, 0)

	# def forward_old(self, x):
	# 	print("x shape: ", x.shape)
	# 	print("x type: ", type(x))
	# 	print("x mean: ", torch.mean(x))
	# 	print("x var: ", torch.var(x))
	# 	# print("ntotal: ", ntotal)
	# 	output = self.time_encoder(x)
	# 	output = torch.reshape(output, (2, 3, 2, 168))
	# 	# n_channels_out = output.shape[1]
	# 	# n_batch = output.shape[0]
	# 	# x = torch.reshape(x, (,))
	# 	#
	# 	print("output mean: ", torch.mean(output))
	# 	print("output var: ", torch.var(output))
	# 	print("output shape: ", output.shape)
	#
	# 	return output

################################################################################
############################ Parameter initialization ##########################
################################################################################
# def init_weights_xavier_2d(model):
#
# 	# Loop over blocks
# 	for block in model.time_encoding:
# 		if isinstance(block, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
#
# 			# Initialize the weights
# 			nn.init.xavier_uniform_(block.weight, gain=1.0)
#
# 			# Initialize the biases
# 			if block.bias != None:
# 				nn.init.zeros_(block.bias)
