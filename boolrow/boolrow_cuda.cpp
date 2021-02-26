#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor boolrow_cuda_forward(
	torch::Tensor input);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor boolrow_forward(
	torch::Tensor input) {
	CHECK_INPUT(input);

	return boolrow_cuda_forward(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &boolrow_forward, "boolrow forward (CUDA)");
}
