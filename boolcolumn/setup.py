from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='boolcolumn',
    ext_modules=[
        CUDAExtension('boolcolumn_cuda', [
            'boolcolumn_cuda.cpp',
            'boolcolumn_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
