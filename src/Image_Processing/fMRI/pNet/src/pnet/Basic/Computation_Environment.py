# Setup the preferred computation requirement
# Yuncong Ma, 8/15/2023


import platform
import os

PNET_OS = platform.system()

if PNET_OS == 'Darwin':
    PNET_OS = 'macOS'
# PNET_OS = 'Windows'
# PNET_OS = 'Linux'


# CPU or GPU mode
PNET_MODE = 'CPU'

# Parallel settings for CPU based computation
PNET_CPU_PARALLEL = 0
PNET_CPU_CORE = 1

# Parallel computation for matrix operation using Numpy
OMP_NUM_THREADS = 1
OPENBLAS_NUM_THREADS = 1

# Parallel settings for GPU based computation
PNET_GPU_PARALLEL = 0
PNET_GPU_CORE = 1

os.system('export OMP_NUM_THREADS=1')
os.system('export OPENBLAS_NUM_THREADS=1')


def default_computation_environment():
    PNET_MODE = 'CPU'
    PNET_CPU_PARALLEL = 0
    PNET_CPU_CORE = 1

    # Parallel computation for matrix operation using Numpy
    OMP_NUM_THREADS = 1
    OPENBLAS_NUM_THREADS = 1

    # Parallel settings for GPU based computation
    PNET_GPU_PARALLEL = 0
    PNET_GPU_CORE = 1


def set_computation_environment():
    os.system('export OMP_NUM_THREADS=1')
    os.system('export OPENBLAS_NUM_THREADS=1')


