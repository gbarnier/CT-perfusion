import torch
import torch.nn as nn
import os, sys
from CTP_config import config
import CTP_utils

################################################################################
################################## Model definition(s) #########################
################################################################################
# Function that creates neural network
def create_model(model_type, input_size, info=True):

	print("model_type: ", model_type)

	if model_type == 'baseline':
		model = baseline(input_size)
		model.to(config.device)
		if info: print("User has chosen the baseline model: two FC layers (one hidden layer)")

	elif model_type == 'fc6':
		model = fc_deep(input_size)
		model.to(config.device)
		if info: print("User has chosen a fully connected layers model: six FC layers (five hidden layers)")

	elif model_type == 'gui':
		model = gui(input_size)
		model.to(config.device)
		if info: print("User has chosen Gui model")

	elif model_type == 'blogMed':
		model = blogMed(input_size)
		model.to(config.device)
		if info: print("User has chosen blogMed model")

	elif model_type == 'blogDeep':
		model = blogDeep(input_size)
		model.to(config.device)
		if info: print("User has chosen blogDeep model")

	elif model_type == 'blog':
		model = cnn_blog(input_size)
		model.to(config.device)
		if info: print("User has chosen CNN modified from https://blog.goodaudience.com")

	elif model_type == 'kiranyaz':
		model = cnn_kiranyaz(input_size)
		model.to(config.device)
		if info: print("User has chosen CNN modified from Kiranyaz et al. (2019)")

	elif model_type == 'blogLarge':
		model = blogLarge(input_size)
		model.to(config.device)
		if info: print("User has chosen blogLarge")

	elif model_type == 'blog_test':
		model = blog_test(input_size)
		model.to(config.device)
		if info: print("User has chosen 'blog_test'")

	elif model_type == 'debug':
		model = debug(input_size)
		model.to(config.device)
		if info: print("User has chosen 'debug'")


	else: sys.exit("Model requested: ", model_type ,"Please provide a valid model type")

	# Set number of parameters

	if info:
		print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
		print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

	return model


# Baseline model: 1 FC hidden layer followed by ReLU
class debug(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(debug, self).__init__()

		# Model name
		self.name = 'debug'

		# Conv layer
		nc0 = 1 # Number of input channels
		n0 = input_size
		nf_conv1 = 30
		f_conv1 = 10
		s_conv1 = 1
		p_conv1 = 0
		conv1 = nn.Conv1d(nc0, nf_conv1, f_conv1, stride=s_conv1, padding=p_conv1)
		n1 = int( (n0+2*p_conv1-f_conv1)/s_conv1 ) + 1
		print("input conv 1: ", n0)
		print("output conv 1: ", n1)

		# FC layer
		n_fc1_in = n1*nf_conv1
		n_fc1_out = 1
		print("input fc 1: ", n_fc1_in)
		print("output fc 1: ", n_fc1_out)
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		# Model construction
		self.network = nn.Sequential(
			conv1,nn.ReLU(),nn.Flatten(),fc1,nn.ReLU()
		)

	def forward(self, x):
		output = self.network(x)
		# print("shape output: ", output.shape)
		output = torch.reshape(output, (output.shape[0], output.shape[1]))
		return output

# Baseline model: 1 FC hidden layer followed by ReLU
class baseline(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(baseline, self).__init__()

		# Model name
		self.name = 'baseline'

		################################ Layer 1 ###############################
		# FC1
		n_fc1_in=input_size
		n_fc1_out = config.baseline_n_hidden
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input FC 1 : ", n_fc1_in)
		print("output FC 1 : ", n_fc1_out)

		# FC2
		n_fc2_in=n_fc1_out
		n_fc2_out = 1
		fc2 = nn.Linear(n_fc2_in, n_fc2_out)
		print("input FC 2 : ", n_fc2_in)
		print("output FC 2 : ", n_fc2_out)
		print("Leaky slope: ", config.leaky_slope)

		# Model construction
		self.network = nn.Sequential(
			fc1,nn.ReLU(),
			fc2,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		output = torch.reshape(output, (output.shape[0], output.shape[1]))
		return output

class fc_deep(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(fc_deep, self).__init__()

		# Model name
		self.name = 'fc_deep'

		################################ Layer 1 ###############################
		# FC1
		n_fc1_in=input_size
		n_fc1_out = config.fc6_n_hidden[0]
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input FC 1: ", n_fc1_in)
		print("output FC 1: ", n_fc1_out)

		# FC2
		n_fc2_in=n_fc1_out
		n_fc2_out = config.fc6_n_hidden[1]
		fc2 = nn.Linear(n_fc2_in, n_fc2_out)
		print("input FC 2: ", n_fc2_in)
		print("output FC 2: ", n_fc2_out)

		# FC3
		n_fc3_in=n_fc2_out
		n_fc3_out = config.fc6_n_hidden[2]
		fc3 = nn.Linear(n_fc3_in, n_fc3_out)
		print("input FC 3: ", n_fc3_in)
		print("output FC 3: ", n_fc3_out)

		# FC4
		n_fc4_in=n_fc3_out
		n_fc4_out = config.fc6_n_hidden[3]
		fc4 = nn.Linear(n_fc4_in, n_fc4_out)
		print("input FC 4: ", n_fc4_in)
		print("output FC 4: ", n_fc4_out)

		# FC5
		n_fc5_in=n_fc4_out
		n_fc5_out = config.fc6_n_hidden[4]
		fc5 = nn.Linear(n_fc5_in, n_fc5_out)
		print("input FC 5: ", n_fc5_in)
		print("output FC 5: ", n_fc5_out)

		# FC6
		n_fc6_in=n_fc5_out
		n_fc6_out = 1
		fc6 = nn.Linear(n_fc6_in, n_fc6_out)
		print("input FC 6: ", n_fc6_in)
		print("output FC 6: ", n_fc6_out)

		# Model construction
		self.network = nn.Sequential(
			fc1,nn.ReLU(),
			fc2,nn.ReLU(),
			fc3,nn.ReLU(),
			fc4,nn.ReLU(),
			fc5,nn.ReLU(),
			fc6,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		output = torch.reshape(output, (output.shape[0], output.shape[1]))
		return output

# Model modified from:
# https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
class cnn_blog(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(cnn_blog, self).__init__()

		# Model name
		self.name = 'blog'

		# Conv1
		nc0 = 1 # Number of input channels
		n0 = input_size
		# nf_conv1 = 100
		nf_conv1 = 30
		f_conv1 = 10
		s_conv1 = 1
		p_conv1 = 0
		conv1 = nn.Conv1d(nc0, nf_conv1, f_conv1, stride=s_conv1, padding=p_conv1)
		print("input conv 1: ", n0)

		# Conv2
		n1 = int( (n0+2*p_conv1-f_conv1)/s_conv1 ) + 1
		# nf_conv2 = 100
		nf_conv2 = 30
		f_conv2 = 10
		s_conv2 = 1
		p_conv2 = 0
		conv2 = nn.Conv1d(nf_conv1, nf_conv2, f_conv2, stride=s_conv2, padding=p_conv2)
		print("input conv 2: ", n1)

		# MaxPool1
		n2 = int( (n1+2*p_conv2-f_conv2) / s_conv2) + 1
		f_pool1 = 3
		s_pool1 = f_pool1
		p_pool1 = 0
		max_pool1 = nn.MaxPool1d(f_pool1, stride=s_pool1, padding=p_pool1)
		print("input max pool 1: ", n2)

		# Conv3
		n3 = int( (n2+2*p_pool1-f_pool1)/s_pool1 ) + 1
		# nf_conv3 = 160
		nf_conv3 = 50
		f_conv3 = 10
		s_conv3 = 1
		p_conv3 = 0
		conv3 = nn.Conv1d(nf_conv2, nf_conv3, f_conv3, stride=s_conv3, padding=p_conv3)
		print("input conv 3 : ", n3)

		# Conv4
		n4 = int( (n3+2*p_conv3-f_conv3)/s_conv3 ) + 1
		# nf_conv4 = 160
		nf_conv4 = 50
		f_conv4 = 10
		s_conv4 = 1
		p_conv4 = 0
		conv4 = nn.Conv1d(nf_conv3, nf_conv4, f_conv4, stride=s_conv4, padding=p_conv4)
		print("input conv 4 : ", n4)

		# Average pooling
		n5 = int( (n4+2*p_conv4-f_conv4) / s_conv4) + 1
		f_pool2 = 2
		s_pool2 = f_pool2
		p_pool2 = 0
		avg_pool2 = nn.AvgPool1d(f_pool2, stride=s_pool2, padding=p_pool2)
		print("input avg pool 1 : ", n5)

		# Dropout1
		# p_drop1 = 0.5 # Probability of zeroing out activation
		# dropout1 = nn.Dropout(p_drop1)

		# FC1
		n_fc1_in = int( (n5+2*p_pool2-f_pool2) / s_pool2) + 1
		print("n_fc1_in before:", n_fc1_in)
		n_fc1_in*=nf_conv4
		n_fc1_out = 1
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input FC 1 : ", n_fc1_in)
		print("output FC 1 : ", n_fc1_out)

		# Model construction
		self.network = nn.Sequential(
			conv1,nn.ReLU(),
			conv2,nn.ReLU(),
			max_pool1,
			conv3,nn.ReLU(),
			conv4,nn.ReLU(),
			avg_pool2,nn.ReLU(),
			nn.Flatten(),
			fc1,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		return output

class blogSmall(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(blogSmall, self).__init__()

		# Model name
		self.name = 'blogSmall'

		# Conv1
		nc0 = 1 # Number of input channels
		n0 = input_size
		# nf_conv1 = 100
		nf_conv1 = 30
		f_conv1 = 10
		s_conv1 = 1
		p_conv1 = 0
		conv1 = nn.Conv1d(nc0, nf_conv1, f_conv1, stride=s_conv1, padding=p_conv1)
		print("input conv 1: ", n0)

		# Conv2
		n1 = int( (n0+2*p_conv1-f_conv1)/s_conv1 ) + 1
		# nf_conv2 = 100
		nf_conv2 = 30
		f_conv2 = 10
		s_conv2 = 1
		p_conv2 = 0
		conv2 = nn.Conv1d(nf_conv1, nf_conv2, f_conv2, stride=s_conv2, padding=p_conv2)
		print("input conv 2: ", n1)

		# MaxPool1
		n2 = int( (n1+2*p_conv2-f_conv2) / s_conv2) + 1
		f_pool1 = 3
		s_pool1 = f_pool1
		p_pool1 = 0
		max_pool1 = nn.MaxPool1d(f_pool1, stride=s_pool1, padding=p_pool1)
		print("input max pool 1: ", n2)

		# Conv3
		n3 = int( (n2+2*p_pool1-f_pool1)/s_pool1 ) + 1
		# nf_conv3 = 160
		nf_conv3 = 50
		f_conv3 = 10
		s_conv3 = 1
		p_conv3 = 0
		conv3 = nn.Conv1d(nf_conv2, nf_conv3, f_conv3, stride=s_conv3, padding=p_conv3)
		print("input conv 3 : ", n3)

		# Conv4
		n4 = int( (n3+2*p_conv3-f_conv3)/s_conv3 ) + 1
		# nf_conv4 = 160
		nf_conv4 = 50
		f_conv4 = 10
		s_conv4 = 1
		p_conv4 = 0
		conv4 = nn.Conv1d(nf_conv3, nf_conv4, f_conv4, stride=s_conv4, padding=p_conv4)
		print("input conv 4 : ", n4)

		# Average pooling
		n5 = int( (n4+2*p_conv4-f_conv4) / s_conv4) + 1
		f_pool2 = 2
		s_pool2 = f_pool2
		p_pool2 = 0
		avg_pool2 = nn.AvgPool1d(f_pool2, stride=s_pool2, padding=p_pool2)
		print("input avg pool 1 : ", n5)

		# Dropout1
		# p_drop1 = 0.5 # Probability of zeroing out activation
		# dropout1 = nn.Dropout(p_drop1)

		# FC1
		n_fc1_in = int( (n5+2*p_pool2-f_pool2) / s_pool2) + 1
		print("n_fc1_in before:", n_fc1_in)
		n_fc1_in*=nf_conv4
		n_fc1_out = 1
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input FC 1 : ", n_fc1_in)
		print("output FC 1 : ", n_fc1_out)

		# Model construction
		self.network = nn.Sequential(
			conv1,nn.ReLU(),
			conv2,nn.ReLU(),
			max_pool1,
			conv3,nn.ReLU(),
			conv4,nn.ReLU(),
			avg_pool2,nn.ReLU(),
			nn.Flatten(),
			fc1,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		return output

# Model modified from:
# 1D Convolutional Neural Networks and Applications – A Survey
# Kiranyaz et al., 2019
class cnn_kiranyaz(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(cnn_kiranyaz, self).__init__()

		# Model name
		self.name = 'kiranyaz'

		################################ Layer 1 ###############################
		# Conv1
		nc0 = 1 # Number of input channels
		n0 = input_size
		nf_conv1 = 24
		f_conv1 = 4
		s_conv1 = 1
		p_conv1 = 0
		conv1 = nn.Conv1d(nc0, nf_conv1, f_conv1, stride=s_conv1, padding=p_conv1)
		print("input conv 1: ", n0)

		# MaxPool1
		n1 = int( (n0+2*p_conv1-f_conv1) / s_conv1) + 1
		f_pool1 = 2
		s_pool1 = f_pool1
		p_pool1 = 0
		max_pool1 = nn.MaxPool1d(f_pool1, stride=s_pool1, padding=p_pool1)
		print("input max pool 1: ", n1)

		################################ Layer 2 ###############################
		# Conv2
		n2 = int( (n1+2*p_pool1-f_pool1)/s_pool1 ) + 1
		nf_conv2 = 24
		f_conv2 = 4
		s_conv2 = 1
		p_conv2 = 0
		conv2 = nn.Conv1d(nf_conv1, nf_conv2, f_conv2, stride=s_conv2, padding=p_conv2)
		print("input conv 2: ", n2)

		# MaxPool2
		n3 = int( (n2+2*p_conv2-f_conv2) / s_conv2) + 1
		f_pool2 = 2
		s_pool2 = f_pool2
		p_pool2 = 0
		max_pool2 = nn.MaxPool1d(f_pool2, stride=s_pool2, padding=p_pool2)
		print("input max pool 2: ", n3)

		################################ Layer 3 ###############################
		# Conv3
		n4 = int( (n3+2*p_pool2-f_pool2)/s_pool2 ) + 1
		nf_conv3 = 24
		f_conv3 = 4
		s_conv3 = 1
		p_conv3 = 0
		conv3 = nn.Conv1d(nf_conv2, nf_conv3, f_conv3, stride=s_conv3, padding=p_conv3)
		print("input conv 3: ", n4)

		# MaxPool3
		n5 = int( (n4+2*p_conv3-f_conv3) / s_conv3) + 1
		f_pool3 = 2
		s_pool3 = f_pool3
		p_pool3 = 0
		max_pool3 = nn.MaxPool1d(f_pool3, stride=s_pool3, padding=p_pool3)
		print("input max pool 3: ", n5)

		################################ Layer 4 ###############################
		# FC1
		n_fc1_in = int( (n5+2*p_pool3-f_pool3) / s_pool3) + 1
		n_fc1_in *= nf_conv3
		n_fc1_out = n_fc1_in
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input fc 1: ", n_fc1_in)

		################################ Layer 5 ###############################
		# FC2
		n_fc2_in = n_fc1_out
		n_fc2_out = 1
		fc2 = nn.Linear(n_fc2_in, n_fc2_out)
		print("input fc 2: ", n_fc2_out)

		# Model construction
		self.network = nn.Sequential(
			conv1,nn.ReLU(),
			max_pool1,
			conv2,nn.ReLU(),
			max_pool2,
			conv3,nn.ReLU(),
			max_pool3,
			nn.Flatten(),
			fc1,nn.ReLU(),
			fc2,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		return output

class gui(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(gui, self).__init__()

		# Model name
		self.name = 'gui'

		################################ Layer 1 ###############################
		# Conv1
		nc0 = 1 # Number of input channels
		n0 = input_size
		nf_conv1 = 100
		f_conv1 = 10
		s_conv1 = 1
		p_conv1 = 0
		conv1 = nn.Conv1d(nc0, nf_conv1, f_conv1, stride=s_conv1, padding=p_conv1)
		print("input conv 1: ", n0)

		# MaxPool1
		n1 = int( (n0+2*p_conv1-f_conv1) / s_conv1) + 1
		f_pool1 = 2
		s_pool1 = f_pool1
		p_pool1 = 0
		max_pool1 = nn.MaxPool1d(f_pool1, stride=s_pool1, padding=p_pool1)
		print("input max pool 1: ", n1)

		################################ Layer 2 ###############################
		# Conv2
		n2 = int( (n1+2*p_pool1-f_pool1)/s_pool1 ) + 1
		nf_conv2 = 150
		f_conv2 = 5
		s_conv2 = 1
		p_conv2 = 0
		conv2 = nn.Conv1d(nf_conv1, nf_conv2, f_conv2, stride=s_conv2, padding=p_conv2)
		print("input conv 2: ", n2)

		# MaxPool2
		n3 = int( (n2+2*p_conv2-f_conv2) / s_conv2) + 1
		f_pool2 = 2
		s_pool2 = f_pool2
		p_pool2 = 0
		max_pool2 = nn.MaxPool1d(f_pool2, stride=s_pool2, padding=p_pool2)
		print("input max pool 2: ", n3)

		################################ Layer 3 ###############################
		# Conv3
		n4 = int( (n3+2*p_pool2-f_pool2)/s_pool2 ) + 1
		nf_conv3 = 150
		f_conv3 = 3
		s_conv3 = 2
		p_conv3 = 0
		conv3 = nn.Conv1d(nf_conv2, nf_conv3, f_conv3, stride=s_conv3, padding=p_conv3)
		print("input conv 3: ", n4)

		# MaxPool3
		n5 = int( (n4+2*p_conv3-f_conv3) / s_conv3) + 1
		f_pool3 = 2
		s_pool3 = f_pool3
		p_pool3 = 0
		max_pool3 = nn.MaxPool1d(f_pool3, stride=s_pool3, padding=p_pool3)
		print("input max pool 3: ", n5)

		################################ Layer 4 ###############################
		# FC1
		n_fc1_in = int( (n5+2*p_pool3-f_pool3) / s_pool3) + 1
		n_fc1_in *= nf_conv3
		n_fc1_out = 100
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input fc 1: ", n_fc1_in)

		################################ Layer 5 ###############################
		# FC2
		n_fc2_in = n_fc1_out
		n_fc2_out = 1
		fc2 = nn.Linear(n_fc2_in, n_fc2_out)
		print("input fc 2: ", n_fc2_out)

		# Model construction
		self.network = nn.Sequential(
			conv1,nn.ReLU(),
			max_pool1,
			conv2,nn.ReLU(),
			max_pool2,
			conv3,nn.ReLU(),
			max_pool3,
			nn.Flatten(),
			fc1,nn.ReLU(),
			fc2,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		output = torch.reshape(output, (output.shape[0], output.shape[1]))
		return output

class blogMed(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(blogMed, self).__init__()

		# Model name
		self.name = 'blogMed'

	# Conv1
		nc0 = 1 # Number of input channels
		n0 = input_size
		# nf_conv1 = 100
		nf_conv1 = 30
		f_conv1 = 3
		s_conv1 = 1
		p_conv1 = 0
		conv1 = nn.Conv1d(nc0, nf_conv1, f_conv1, stride=s_conv1, padding=p_conv1)
		print("input conv 1: ", n0)

		# Conv2
		n1 = int( (n0+2*p_conv1-f_conv1)/s_conv1 ) + 1
		nf_conv2 = 50
		f_conv2 = 3
		s_conv2 = 1
		p_conv2 = 0
		conv2 = nn.Conv1d(nf_conv1, nf_conv2, f_conv2, stride=s_conv2, padding=p_conv2)
		print("input conv 2: ", n1)

		# MaxPool1
		n2 = int( (n1+2*p_conv2-f_conv2) / s_conv2) + 1
		f_pool1 = 4
		s_pool1 = f_pool1
		p_pool1 = 0
		max_pool1 = nn.MaxPool1d(f_pool1, stride=s_pool1, padding=p_pool1)
		print("input max pool 1: ", n2)

		# Conv3
		n3 = int( (n2+2*p_pool1-f_pool1)/s_pool1 ) + 1
		# nf_conv3 = 160
		nf_conv3 = 80
		f_conv3 = 3
		s_conv3 = 1
		p_conv3 = 0
		conv3 = nn.Conv1d(nf_conv2, nf_conv3, f_conv3, stride=s_conv3, padding=p_conv3)
		print("input conv 3 : ", n3)

		# Conv4
		n4 = int( (n3+2*p_conv3-f_conv3)/s_conv3 ) + 1
		# nf_conv4 = 160
		nf_conv4 = 80
		f_conv4 = 5
		s_conv4 = 2
		p_conv4 = 0
		conv4 = nn.Conv1d(nf_conv3, nf_conv4, f_conv4, stride=s_conv4, padding=p_conv4)
		print("input conv 4 : ", n4)

		# Max pooling
		n5 = int( (n4+2*p_conv4-f_conv4) / s_conv4) + 1
		f_pool2 = 2
		s_pool2 = f_pool2
		p_pool2 = 0
		max_pool2 = nn.MaxPool1d(f_pool2, stride=s_pool2, padding=p_pool2)
		print("input max pool 1 : ", n5)

		################################ Layer 4 ###############################
		# FC1
		n_fc1_in = int( (n5+2*p_pool2-f_pool2) / s_pool2) + 1
		n_fc1_in *= nf_conv4
		n_fc1_out = 100
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input fc 1: ", n_fc1_in)

		################################ Layer 5 ###############################
		# FC2
		n_fc2_in = n_fc1_out
		n_fc2_out = 1
		fc2 = nn.Linear(n_fc2_in, n_fc2_out)
		print("input fc 2: ", n_fc2_in)

		# Model construction
		self.network = nn.Sequential(
			conv1,nn.ReLU(),conv2,max_pool1,nn.ReLU(),
			conv3,nn.ReLU(),conv4,max_pool2,nn.ReLU(),
			nn.Flatten(),
			fc1,nn.ReLU(),
			fc2,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		return output

class blogDeep(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(blogDeep, self).__init__()

		# Model name
		self.name = 'blogDeep'

		# Conv1
		nc0 = 1 # Number of input channels
		n0 = input_size
		# nf_conv1 = 100
		nf_conv1 = 50
		f_conv1 = 3
		s_conv1 = 1
		p_conv1 = 0
		conv1 = nn.Conv1d(nc0, nf_conv1, f_conv1, stride=s_conv1, padding=p_conv1)
		print("input conv 1: ", n0)

		# Conv2
		n1 = int( (n0+2*p_conv1-f_conv1)/s_conv1 ) + 1
		# nf_conv2 = 100
		nf_conv2 = 50
		f_conv2 = 3
		s_conv2 = 1
		p_conv2 = 0
		conv2 = nn.Conv1d(nf_conv1, nf_conv2, f_conv2, stride=s_conv2, padding=p_conv2)
		print("input conv 2: ", n1)

		# MaxPool1
		n2 = int( (n1+2*p_conv2-f_conv2) / s_conv2) + 1
		f_pool1 = 2
		s_pool1 = f_pool1
		p_pool1 = 0
		max_pool1 = nn.MaxPool1d(f_pool1, stride=s_pool1, padding=p_pool1)
		print("input max pool 1: ", n2)

		# Conv3
		n3 = int( (n2+2*p_pool1-f_pool1)/s_pool1 ) + 1
		# nf_conv3 = 160
		nf_conv3 = 100
		f_conv3 = 3
		s_conv3 = 1
		p_conv3 = 0
		conv3 = nn.Conv1d(nf_conv2, nf_conv3, f_conv3, stride=s_conv3, padding=p_conv3)
		print("input conv 3: ", n3)

		# Conv4
		n4 = int( (n3+2*p_conv3-f_conv3)/s_conv3 ) + 1
		# nf_conv4 = 160
		nf_conv4 = 100
		f_conv4 = 3
		s_conv4 = 1
		p_conv4 = 0
		conv4 = nn.Conv1d(nf_conv3, nf_conv4, f_conv4, stride=s_conv4, padding=p_conv4)
		print("input conv 4: ", n4)

		# Max pooling
		n5 = int( (n4+2*p_conv4-f_conv4) / s_conv4) + 1
		f_pool2 = 2
		s_pool2 = f_pool2
		p_pool2 = 0
		max_pool2 = nn.MaxPool1d(f_pool2, stride=s_pool2, padding=p_pool2)
		print("input max pool 2: ", n5)

		# Conv5
		n6 = int( (n5+2*p_pool2-f_pool2)/s_pool2 ) + 1
		# nf_conv4 = 160
		nf_conv5 = 100
		f_conv5 = 3
		s_conv5 = 1
		p_conv5 = 0
		conv5 = nn.Conv1d(nf_conv4, nf_conv5, f_conv5, stride=s_conv5, padding=p_conv5)
		print("input conv 5: ", n6)

		# Conv6
		n7 = int( (n6+2*p_conv5-f_conv5)/s_conv5 ) + 1
		# nf_conv4 = 160
		nf_conv6 = 100
		f_conv6 = 3
		s_conv6 = 1
		p_conv6 = 0
		conv6 = nn.Conv1d(nf_conv5, nf_conv6, f_conv6, stride=s_conv6, padding=p_conv6)
		print("input conv 6: ", n7)

		# Max pooling
		n8 = int( (n7+2*p_conv6-f_conv6) / s_conv6) + 1
		f_pool3 = 2
		s_pool3 = f_pool3
		p_pool3 = 0
		max_pool3 = nn.MaxPool1d(f_pool3, stride=s_pool3, padding=p_pool3)
		print("input max pool 3: ", n8)

		# FC1
		n_fc1_in = int( (n8+2*p_pool3-f_pool3) / s_pool3) + 1
		print("n_fc1_in before:", n_fc1_in)
		n_fc1_in*=nf_conv6
		n_fc1_out = 100
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input FC 1: ", n_fc1_in)
		print("output FC 1: ", n_fc1_out)

		# FC2
		n_fc2_in = n_fc1_out
		print("n_fc2_in:", n_fc2_in)
		n_fc2_out = 1
		fc2 = nn.Linear(n_fc2_in, n_fc2_out)
		print("input FC 1: ", n_fc2_in)
		print("output FC 1: ", n_fc2_out)

		# Model construction
		self.network = nn.Sequential(
			conv1,nn.ReLU(),
			conv2,nn.ReLU(),
			max_pool1,
			conv3,nn.ReLU(),
			conv4,nn.ReLU(),
			max_pool2,nn.ReLU(),
			conv5,nn.ReLU(),
			conv6,nn.ReLU(),
			max_pool2,nn.ReLU(),
			nn.Flatten(),
			fc1,nn.ReLU(),
			fc2,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		return output


class blogLarge(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(blogLarge, self).__init__()

		# Model name
		self.name = 'blogLarge'

		################# Layer 1 #################
		# Conv1
		nc0 = 1 # Number of input channels
		n0 = input_size
		nf_conv1 = 100
		f_conv1 = 3
		s_conv1 = 1
		p_conv1 = 0
		conv1 = nn.Conv1d(nc0, nf_conv1, f_conv1, stride=s_conv1, padding=p_conv1)
		print("input conv 1: ", n0)

		# MaxPool1
		n1 = int( (n0+2*p_conv1-f_conv1) / s_conv1) + 1
		f_pool1 = 2
		s_pool1 = f_pool1
		p_pool1 = 0
		max_pool1 = nn.MaxPool1d(f_pool1, stride=s_pool1, padding=p_pool1)
		print("input max pool 1: ", n1)

		################# Layer 2 #################
		# Conv2
		n2 = int( (n1+2*p_pool1-f_pool1)/s_pool1 ) + 1
		nf_conv2 = 150
		f_conv2 = 3
		s_conv2 = 1
		p_conv2 = 0
		conv2 = nn.Conv1d(nf_conv1, nf_conv2, f_conv2, stride=s_conv2, padding=p_conv2)
		print("input conv 2: ", n2)

		# MaxPool2
		n3 = int( (n2+2*p_conv2-f_conv2) / s_conv2) + 1
		f_pool2 = 2
		s_pool2 = f_pool2
		p_pool2 = 0
		max_pool2 = nn.MaxPool1d(f_pool2, stride=s_pool2, padding=p_pool2)
		print("input max pool 2: ", n3)

		################# Layer 3 #################
		# Conv3
		n4 = int( (n3+2*p_pool2-f_pool2)/s_pool2 ) + 1
		nf_conv3 = 200
		f_conv3 = 3
		s_conv3 = 1
		p_conv3 = 0
		conv3 = nn.Conv1d(nf_conv2, nf_conv3, f_conv3, stride=s_conv3, padding=p_conv3)
		print("input conv 3: ", n4)

		# MaxPool3
		n5 = int( (n4+2*p_conv3-f_conv3) / s_conv3) + 1
		f_pool3 = 2
		s_pool3 = f_pool3
		p_pool3 = 0
		max_pool3 = nn.MaxPool1d(f_pool3, stride=s_pool3, padding=p_pool3)
		print("input max pool 3: ", n5)

		################# Layer 4 #################
		# Conv4
		n6 = int( (n5+2*p_pool3-f_pool3)/s_pool3 ) + 1
		nf_conv4 = 200
		f_conv4 = 3
		s_conv4 = 1
		p_conv4 = 0
		conv4 = nn.Conv1d(nf_conv3, nf_conv4, f_conv4, stride=s_conv4, padding=p_conv4)
		print("input conv 3: ", n4)

		# MaxPool4
		n7 = int( (n6+2*p_conv4-f_conv4) / s_conv4) + 1
		f_pool4 = 2
		s_pool4 = f_pool4
		p_pool4 = 0
		max_pool4 = nn.MaxPool1d(f_pool4, stride=s_pool4, padding=p_pool4)
		print("input max pool 4: ", n7)

		################# Dense layers #################
		# FC1
		n_fc1_in = int( (n7+2*p_pool4-f_pool4) / s_pool4) + 1
		print("n_fc1_in before:", n_fc1_in)
		n_fc1_in*=nf_conv4
		n_fc1_out = 100
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input FC 1: ", n_fc1_in)
		print("output FC 1: ", n_fc1_out)

		# FC2
		n_fc2_in = n_fc1_out
		print("n_fc2_in:", n_fc2_in)
		n_fc2_out = 1
		fc2 = nn.Linear(n_fc2_in, n_fc2_out)
		print("input FC 1: ", n_fc2_in)
		print("output FC 1: ", n_fc2_out)

		# Model construction
		self.network = nn.Sequential(
			conv1,nn.ReLU(),
			max_pool1,
			conv2,nn.ReLU(),
			max_pool2,
			conv3,nn.ReLU(),
			max_pool3,
			conv4,nn.ReLU(),
			max_pool4,
			nn.Flatten(),
			fc1,nn.ReLU(),
			fc2,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		return output

class blog_test(nn.Module):

	def __init__(self, input_size):

		# Call to the __init__ function of the super class
		super(blog_test, self).__init__()

		# Model name
		self.name = 'blog_test'

		# Conv1
		nc0 = 1 # Number of input channels
		n0 = input_size
		# nf_conv1 = 100
		nf_conv1 = 30
		f_conv1 = 3
		s_conv1 = 1
		p_conv1 = 0
		conv1 = nn.Conv1d(nc0, nf_conv1, f_conv1, stride=s_conv1, padding=p_conv1)
		print("input conv 1: ", n0)

		# Conv2
		n1 = int( (n0+2*p_conv1-f_conv1)/s_conv1 ) + 1
		# nf_conv2 = 100
		nf_conv2 = 30
		f_conv2 = 3
		s_conv2 = 1
		p_conv2 = 0
		conv2 = nn.Conv1d(nf_conv1, nf_conv2, f_conv2, stride=s_conv2, padding=p_conv2)
		print("input conv 2: ", n1)

		# MaxPool1
		n2 = int( (n1+2*p_conv2-f_conv2) / s_conv2) + 1
		f_pool1 = 3
		s_pool1 = f_pool1
		p_pool1 = 0
		max_pool1 = nn.MaxPool1d(f_pool1, stride=s_pool1, padding=p_pool1)
		print("input max pool 1: ", n2)

		# Conv3
		n3 = int( (n2+2*p_pool1-f_pool1)/s_pool1 ) + 1
		# nf_conv3 = 160
		nf_conv3 = 50
		f_conv3 = 3
		s_conv3 = 1
		p_conv3 = 0
		conv3 = nn.Conv1d(nf_conv2, nf_conv3, f_conv3, stride=s_conv3, padding=p_conv3)
		print("input conv 3 : ", n3)

		# Conv4
		n4 = int( (n3+2*p_conv3-f_conv3)/s_conv3 ) + 1
		# nf_conv4 = 160
		nf_conv4 = 50
		f_conv4 = 3
		s_conv4 = 1
		p_conv4 = 0
		conv4 = nn.Conv1d(nf_conv3, nf_conv4, f_conv4, stride=s_conv4, padding=p_conv4)
		print("input conv 4 : ", n4)

		# Average pooling
		n5 = int( (n4+2*p_conv4-f_conv4) / s_conv4) + 1
		f_pool2 = 2
		s_pool2 = f_pool2
		p_pool2 = 0
		avg_pool2 = nn.AvgPool1d(f_pool2, stride=s_pool2, padding=p_pool2)
		print("input avg pool 1 : ", n5)

		# Dropout1
		# p_drop1 = 0.5 # Probability of zeroing out activation
		# dropout1 = nn.Dropout(p_drop1)

		# FC1
		n_fc1_in = int( (n5+2*p_pool2-f_pool2) / s_pool2) + 1
		print("n_fc1_in before:", n_fc1_in)
		n_fc1_in*=nf_conv4
		n_fc1_out = 1
		fc1 = nn.Linear(n_fc1_in, n_fc1_out)
		print("input FC 1 : ", n_fc1_in)
		print("output FC 1 : ", n_fc1_out)

		# Model construction
		self.network = nn.Sequential(
			conv1,nn.ReLU(),
			conv2,nn.ReLU(),
			max_pool1,
			conv3,nn.ReLU(),
			conv4,nn.ReLU(),
			avg_pool2,nn.ReLU(),
			nn.Flatten(),
			fc1,nn.LeakyReLU(config.leaky_slope)
		)

	def forward(self, x):
		output = self.network(x)
		return output

################################################################################
############################ Parameter initialization ##########################
################################################################################
def init_weights_xavier(model):

	# Loop over blocks
	for block in model.network:
		if isinstance(block, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):

			# Initialize the weights
			nn.init.xavier_uniform_(block.weight, gain=1.0)

			# Initialize the biases
			if block.bias != None:
				nn.init.zeros_(block.bias)
