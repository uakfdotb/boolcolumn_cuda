#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define block_size 16

// compute the output at one output row/col
// to do so, we need to sum over the product of the corresponding input rows
template <typename scalar_t>
__global__ void matmul_viterbi_cuda_forward_kernel(
	const scalar_t* __restrict__ a,
	const scalar_t* __restrict__ b,
	scalar_t* __restrict__ output,
	int64_t* __restrict__ max_indices,
	size_t sz1,
	size_t sz2,
	size_t sz3,
	size_t batch_size) {

	const int64_t out_row = blockIdx.x * block_size + threadIdx.x;
	const int64_t out_col = blockIdx.y * block_size + threadIdx.y;
	const int64_t batch = blockIdx.z;
	if(out_row < sz1 && out_col < sz3 && batch < batch_size) {
		scalar_t sum = 0;
		int64_t src = 0;
		for(int64_t k = 0; k < sz2; k++) {
			// a[out_row, k]
			scalar_t left = a[batch*sz1*sz2 + out_row*sz2 + k];
			// b[k, out_col]
			scalar_t right = b[batch*sz2*sz3 + k*sz3 + out_col];

			scalar_t prod = left*right;
			if(prod > sum) {
				sum = prod;
				src = k;
			}
		}
		output[batch*sz1*sz3 + out_row*sz3 + out_col] = sum;
		max_indices[batch*sz1*sz3 + out_row*sz3 + out_col] = src;
	}
}

std::vector<torch::Tensor> matmul_viterbi_cuda_forward(
	torch::Tensor a,
	torch::Tensor b) {

	const auto batch_size = a.size(0);
	const auto sz1 = a.size(1);
	const auto sz2 = a.size(2);
	const auto sz3 = b.size(2);

	auto output = torch::zeros({batch_size, sz1, sz3}, a.options());
	auto max_indices = torch::zeros({batch_size, sz1, sz3}, torch::TensorOptions().dtype(torch::kInt64).device(a.device()));

	// each thread computes value at one output cell
	const dim3 dimBlock(block_size, block_size);
	const dim3 dimGrid((sz1 + block_size - 1) / block_size, (sz3 + block_size - 1) / block_size, batch_size);

	AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_viterbi_forward_cuda", ([&] {
		matmul_viterbi_cuda_forward_kernel<scalar_t><<<dimGrid, dimBlock>>>(
			a.data<scalar_t>(),
			b.data<scalar_t>(),
			output.data<scalar_t>(),
			max_indices.data<int64_t>(),
			sz1,
			sz2,
			sz3,
			batch_size
		);
	}));
	return {output, max_indices};
}

template <typename scalar_t>
__global__ void matmul_viterbi_a_cuda_backward_kernel(
	const scalar_t* __restrict__ a,
	const scalar_t* __restrict__ b,
	const scalar_t* __restrict__ output,
	const int64_t* __restrict__ max_indices,
	const scalar_t* __restrict__ d_output,
	scalar_t* __restrict__ d_a,
	size_t sz1,
	size_t sz2,
	size_t sz3,
	size_t batch_size) {

	const int64_t row = blockIdx.x * block_size + threadIdx.x;
	const int64_t col = blockIdx.y * block_size + threadIdx.y;
	const int64_t batch = blockIdx.z;
	if(row < sz1 && col < sz2 && batch < batch_size) {
		// we're at a[row, col], so each out[row, k] could potentially depend on us
		scalar_t sum = 0;
		for(int k = 0; k < sz3; k++) {
			int64_t idx = max_indices[batch*sz1*sz3 + row*sz3 + k];
			if(idx == col) {
				sum += d_output[batch*sz1*sz3 + row*sz3 + k] * b[batch*sz2*sz3 + col*sz3 + k];
			}
		}
		d_a[batch*sz1*sz2 + row*sz2 + col] = sum;
	}
}

template <typename scalar_t>
__global__ void matmul_viterbi_b_cuda_backward_kernel(
	const scalar_t* __restrict__ a,
	const scalar_t* __restrict__ b,
	const scalar_t* __restrict__ output,
	const int64_t* __restrict__ max_indices,
	const scalar_t* __restrict__ d_output,
	scalar_t* __restrict__ d_b,
	size_t sz1,
	size_t sz2,
	size_t sz3,
	size_t batch_size) {

	const int64_t row = blockIdx.x * block_size + threadIdx.x;
	const int64_t col = blockIdx.y * block_size + threadIdx.y;
	const int64_t batch = blockIdx.z;
	if(row < sz2 && col < sz3 && batch < batch_size) {
		// we're at b[row, col], so each out[k, col] could potentially depend on us
		scalar_t sum = 0;
		for(int k = 0; k < sz1; k++) {
			int64_t idx = max_indices[batch*sz1*sz3 + k*sz3 + col];
			if(idx == row) {
				sum += d_output[batch*sz1*sz3 + k*sz3 + col] * a[batch*sz1*sz2 + k*sz2 + row];
			}
		}
		d_b[batch*sz2*sz3 + row*sz3 + col] = sum;
	}
}

std::vector<torch::Tensor> matmul_viterbi_cuda_backward(
	torch::Tensor a,
	torch::Tensor b,
	torch::Tensor output,
	torch::Tensor max_indices,
	torch::Tensor d_output) {

	const auto batch_size = a.size(0);
	const auto sz1 = a.size(1);
	const auto sz2 = a.size(2);
	const auto sz3 = b.size(2);

	auto d_a = torch::zeros_like(a);
	auto d_b = torch::zeros_like(b);

	const dim3 aDimBlock(block_size, block_size);
	const dim3 aDimGrid((sz1 + block_size - 1) / block_size, (sz2 + block_size - 1) / block_size, batch_size);
	const dim3 bDimBlock(block_size, block_size);
	const dim3 bDimGrid((sz2 + block_size - 1) / block_size, (sz3 + block_size - 1) / block_size, batch_size);

	AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_viterbi_a_backward_cuda", ([&] {
		matmul_viterbi_a_cuda_backward_kernel<scalar_t><<<aDimGrid, aDimBlock>>>(
			a.data<scalar_t>(),
			b.data<scalar_t>(),
			output.data<scalar_t>(),
			max_indices.data<int64_t>(),
			d_output.data<scalar_t>(),
			d_a.data<scalar_t>(),
			sz1,
			sz2,
			sz3,
			batch_size
		);
	}));
	AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_viterbi_b_backward_cuda", ([&] {
		matmul_viterbi_b_cuda_backward_kernel<scalar_t><<<bDimGrid, bDimBlock>>>(
			a.data<scalar_t>(),
			b.data<scalar_t>(),
			output.data<scalar_t>(),
			max_indices.data<int64_t>(),
			d_output.data<scalar_t>(),
			d_b.data<scalar_t>(),
			sz1,
			sz2,
			sz3,
			batch_size
		);
	}));
	return {d_a, d_b};
}
