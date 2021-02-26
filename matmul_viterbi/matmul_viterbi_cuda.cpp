#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> matmul_viterbi_cuda_forward(
	torch::Tensor a,
	torch::Tensor b);

std::vector<torch::Tensor> matmul_viterbi_cuda_backward(
	torch::Tensor a,
	torch::Tensor b,
	torch::Tensor output,
	torch::Tensor max_indices,
	torch::Tensor d_output);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> matmul_viterbi_forward(
	torch::Tensor a,
	torch::Tensor b) {

	CHECK_INPUT(a);
	CHECK_INPUT(b);
	return matmul_viterbi_cuda_forward(a, b);
}

std::vector<torch::Tensor> matmul_viterbi_backward(
	torch::Tensor a,
	torch::Tensor b,
	torch::Tensor output,
	torch::Tensor max_indices,
	torch::Tensor d_output) {

	CHECK_INPUT(a);
	CHECK_INPUT(b);
	CHECK_INPUT(output);
	CHECK_INPUT(max_indices);
	CHECK_INPUT(d_output);
	return matmul_viterbi_cuda_backward(a, b, output, max_indices, d_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &matmul_viterbi_forward, "matmul viterbi forward (CUDA)");
	m.def("backward", &matmul_viterbi_backward, "matmul viterbi backward (CUDA)");
}
