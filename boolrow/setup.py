from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='boolrow',
    ext_modules=[
        CUDAExtension('boolrow_cuda', [
            'boolrow_cuda.cpp',
            'boolrow_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
