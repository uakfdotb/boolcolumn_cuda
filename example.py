# method 1: our extension
import numpy
import time
import torch
import boolcolumn_cuda

mats = numpy.random.uniform(size=(256, 16, 512))
mats_torch = torch.from_numpy(mats).cuda()
t0 = time.time()
output = boolcolumn_cuda.forward(mats_torch)
x = output.cpu().numpy()
print(time.time()-t0)

# method 2: try torch_semiring_einsum extension
# (it can reduce memory usage, but is quite slow)
import numpy
import time
import torch
import torch_semiring_einsum

def dominate_semiring(equation, *args, block_size):
	def func(compute_sum):
		def add_in_place(a, b):
			a[:, :, :] = torch.minimum(a, b)
		def sum_block(a, dims):
			if not dims:
				return a
			return a.amin(dim=dims)
		def multiply_in_place(a, b):
			a[:, :, :] = (a >= b).float()
		return compute_sum(add_in_place, sum_block, multiply_in_place)
	return torch_semiring_einsum.semiring_einsum_forward(equation, args, block_size, func)

equation = 'bij,bik->bjk'
equation = torch_semiring_einsum.compile_equation(equation)
mats = numpy.random.uniform(size=(256, 16, 256))
mats_torch = torch.from_numpy(mats).cuda()
t0 = time.time()
output = dominate_semiring(equation, mats_torch, mats_torch, block_size=10)
x1 = output.cpu().numpy()
print(time.time()-t0)

# method 3: broadcast
import numpy
import time
import torch

mats = numpy.random.uniform(size=(256, 16, 512))
mats_torch = torch.from_numpy(mats).cuda()
t0 = time.time()
left_broad = torch.transpose(mats_torch, 1, 2).unsqueeze(-1)
right_broad = mats_torch.unsqueeze(-3)
stack = torch.stack(torch.broadcast_tensors(left_broad, right_broad), 0)
conj = stack[0, :, :] >= stack[1, :, :]
output = torch.amin(conj, -2)
x2 = output.cpu().numpy()
print(time.time()-t0)
