import numpy as np
import torch
import torch.nn as nn
import CTP_models_2d
import CTP_utils

CTP_utils.seed_everything(10)
nbatch=2
ny=64
nx=64
nt=80
device = 'cuda:0'

input = np.random.rand(nbatch, ny, nx, nt)
print("input shape: ", input.shape)

for idb in range(nbatch):
    for idy in range(ny):
        for idx in range(nx):
            input[idb,idy,idx,:] = 1.0*idb

print("input shape: ", input.shape)
# print("input[0,:]", input[0,:])
# print("input[1,:]", input[1,:])
# input_reshape = input.reshape(nbatch*ny*nx,1,nt)
# input_reshape_torch = CTP_utils.np2torch(input_reshape, device)
# print("input_reshape shape:", input_reshape.shape)
# print("input_reshape: ", input_reshape[0:ny*nx,0,:])
# print("input_reshape: ", input_reshape[ny*nx*(nbatch-1):,0,:])

m = nn.BatchNorm2d(100)
# m = nn.BatchNorm2d(100, affine=False)
# input = torch.randn(20, 100, 35, 45)
# output = m(input)
# print("output shape: ", output.shape)
# print("Number of parameters: ", sum(p.numel() for p in m.parameters()))
# print("Number of trainable parameters: ", sum(p.numel() for p in m.parameters() if p.requires_grad))

input_torch = CTP_utils.np2torch(input, device)

print("input_torch type: ", type(input_torch))
print("input_torch shape: ", input_torch.shape)

model = CTP_models_2d.create_model_2d('EGNET', nt, device)
model.initialize_weights()
print("input mean: ", torch.mean(input_torch))
print("input variance: ", torch.var(input_torch))


output = model.forward(input_torch)
# output_old = model.forward_old(input_reshape_torch)
#
#
# print("output shape: ", output.shape)
# print("output_old shape: ", output_old.shape)
#
# print("output mean: ", torch.mean(output))
# print("output_old mean: ", torch.mean(output_old))
#
# print("output variance: ", torch.var(output))
# print("output_old variance: ", torch.var(output_old))
#
# print("output max: ", torch.max(output))
# print("output min: ", torch.min(output))
# print("output_old max: ", torch.max(output_old))
# print("output_old min: ", torch.min(output_old))
