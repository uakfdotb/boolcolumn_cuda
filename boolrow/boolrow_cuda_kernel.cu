#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// compute the output at one output row/col
// to do so, we need to sum over the product of the corresponding input rows
template <typename scalar_t>
__global__ void boolrow_cuda_forward_kernel(
	const scalar_t* __restrict__ input,
	scalar_t* __restrict__ output,
	const int block_size,
	size_t num_input_rows,
	size_t num_input_cols) {

	const int out_row = blockIdx.x * block_size + threadIdx.x;
	const int out_col = blockIdx.y * block_size + threadIdx.y;
	const int batch = blockIdx.z;
	scalar_t sum = 1;
	if(out_row < num_input_rows && out_col < num_input_rows) {
		for(int in_col = 0; in_col < num_input_cols; in_col++) {
			scalar_t left = input[batch*num_input_rows*num_input_cols + out_row*num_input_cols + in_col];
			scalar_t right = input[batch*num_input_rows*num_input_cols + out_col*num_input_cols + in_col];
			if(left < right) {
				sum = 0;
			}
		}
		output[batch*num_input_rows*num_input_rows + out_row*num_input_rows + out_col] = sum;
	}
}

torch::Tensor boolrow_cuda_forward(
	torch::Tensor input) {

	const auto batch_size = input.size(0);
	const auto num_input_rows = input.size(1);
	const auto num_input_cols = input.size(2);
	auto output = torch::zeros({batch_size, num_input_rows, num_input_rows}, input.options());

	// our kernel computes output at certain cells
	// each "block" is a set of threads running in parallel on the GPU
	// so we run the kernel over 2D square blocks in parallel, over both rows/columns
	// then we also have to iterate over the batch size
	const int block_size = 32;
	const dim3 dimBlock(block_size, block_size);
	const dim3 dimGrid((num_input_rows + block_size - 1) / block_size, (num_input_rows + block_size - 1) / block_size, batch_size);

	AT_DISPATCH_FLOATING_TYPES(input.type(), "boolrow_forward_cuda", ([&] {
		boolrow_cuda_forward_kernel<scalar_t><<<dimGrid, dimBlock>>>(
			input.data<scalar_t>(),
			output.data<scalar_t>(),
			block_size,
			num_input_rows,
			num_input_cols
		);
	}));
	return output;
}
