import numpy
import time
import torch
import matmul_viterbi_cuda

class ViterbiFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, a, b):
		outputs, max_indices = matmul_viterbi_cuda.forward(a, b)
		ctx.save_for_backward(*[a, b, outputs, max_indices])
		return outputs

	@staticmethod
	def backward(ctx, grad_outputs):
		d_a, d_b = matmul_viterbi_cuda.backward(*ctx.saved_tensors, grad_outputs.contiguous())
		return d_a, d_b

mats_a = numpy.random.uniform(size=(1, 16, 32))
mats_b = numpy.random.uniform(size=(1, 32, 24))
mats_a_torch = torch.from_numpy(mats_a).cuda()
mats_b_torch = torch.from_numpy(mats_b).cuda()
mats_a_torch = torch.tensor(mats_a_torch, requires_grad=True)
mats_b_torch = torch.tensor(mats_b_torch, requires_grad=True)
t0 = time.time()
output = ViterbiFunction.apply(mats_a_torch, mats_b_torch)
f = torch.sigmoid(output).sum()
f.backward()
cuda_output = (
    output.detach().cpu().numpy(),
    mats_a_torch.grad.cpu().numpy(),
    mats_b_torch.grad.cpu().numpy(),
)
print('cuda time', time.time()-t0)

# broadcast
import numpy
import time
import torch

#mats_a = numpy.random.uniform(size=(1, 16, 32))
#mats_b = numpy.random.uniform(size=(1, 32, 24))
mats_a_torch = torch.from_numpy(mats_a).cuda()
mats_b_torch = torch.from_numpy(mats_b).cuda()
mats_a_torch = torch.tensor(mats_a_torch, requires_grad=True)
mats_b_torch = torch.tensor(mats_b_torch, requires_grad=True)
t0 = time.time()
left_broad = mats_a_torch.unsqueeze(-1)
right_broad = mats_b_torch.unsqueeze(-3)
stack = torch.stack(torch.broadcast_tensors(left_broad, right_broad), 0)
conj = stack[0, :, :] * stack[1, :, :]
output = torch.amax(conj, -2)
f = torch.sigmoid(output).sum()
f.backward()
broadcast_output = (
    output.detach().cpu().numpy(),
    mats_a_torch.grad.cpu().numpy(),
    mats_b_torch.grad.cpu().numpy(),
)
print('broadcast time', time.time()-t0)

print(numpy.abs(broadcast_output[0] - cuda_output[0]).max())
print(numpy.abs(broadcast_output[1] - cuda_output[1]).max())
print(numpy.abs(broadcast_output[2] - cuda_output[2]).max())
