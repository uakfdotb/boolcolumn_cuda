from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='matmul_viterbi',
    ext_modules=[
        CUDAExtension('matmul_viterbi_cuda', [
            'matmul_viterbi_cuda.cpp',
            'matmul_viterbi_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
