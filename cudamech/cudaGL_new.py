import os
import sys
os.environ['NUMBA_CUDA_LOG_LEVEL'] = 'DEBUG'

from numba import cuda
import cupy
import math
import random

from cudamech.Mechanism.Parameter_Optimization import parameter_optimization
from cudamech.generate import *
from cudamech.Mechanism.Mapping import mapping_fromRealToL
from cudamech.Mapping import listmapping_inverse_fromLToReal

import warnings
from numba.core.errors import NumbaWarning
from torch.utils.dlpack import to_dlpack
import matplotlib.pyplot as plt

# Suppress all Numba warnings
warnings.simplefilter('ignore', category=NumbaWarning)


def random_pdf(x_axis, P_list, array_size):
    # Old code for CPU random
    return_list = []
    for i in range(array_size):
        P_section = P_list[i * 10001: (i + 1)*10001]
        P_sum = cupy.sum(P_section)
        prob_list = P_section / P_sum
        cumu_probs = cupy.cumsum(prob_list).get().tolist()
        random_num = random.uniform(0, 1)
        for j in range(len(cumu_probs)):
            cp = cumu_probs[j]
            if random_num <= cp:
                return_list.append(x_axis[j])
                break
    return return_list

def random_pdf_kernel(x_axis, P_list, array_size):
    BLOCK_SIZE = 32
    total_blocks = math.ceil(array_size / BLOCK_SIZE)
    # Create cumu_probs
    all_cumu_probs = cupy.empty(0, dtype=cupy.float64)
    P_list_new = P_list.reshape(array_size, 10001) / 1000.0
    row_sums = cupy.sum(P_list_new, axis=1, keepdims=True)
    prob = P_list_new / row_sums
    all_cumu_probs = cupy.cumsum(prob, axis=1).ravel()

    # Kernel Call
    output_array = cupy.zeros(array_size, dtype=cupy.float64)
    rand_nums = cupy.random.uniform(0, 1, (10001,))
    random_element_rp_new[total_blocks, BLOCK_SIZE](x_axis, all_cumu_probs, array_size, rand_nums, output_array)
    return output_array.get().tolist()

def perturb_array(A, ep, sensitivity, lower, index, b_size=256, BLOCK_SIZE=32):
    # A is a normal 1D python array
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = open(os.devnull, 'w') # Comment these lines to show output stream and error
    sys.stderr = open(os.devnull, 'w')

    k_best, m_best, y_best = parameter_optimization(ep, index)
    results = cupy.empty(0)
    for b in range(0, len(A), b_size):
        array = A[b:b + b_size]

        print("Filling CP")

        Cp_list = cupy.zeros(len(array), dtype=cupy.float64)
        array_gpu = cupy.array(array, dtype=cupy.float64)
        
        total_blocks = math.ceil(len(array) / BLOCK_SIZE)
        cp_map[total_blocks, BLOCK_SIZE](array_gpu, sensitivity, lower, ep, k_best, m_best, y_best, index, len(array), Cp_list)
        
        print("Done filling CP")
        Cp_list = cupy.array(Cp_list)
        X_axis = cupy.zeros(10001 * (len(array)), dtype=cupy.float64) # 10001 comes from divid + 1 from generate_perturbed_list
        P_list = cupy.zeros(10001 * (len(array)), dtype=cupy.float64)
        total_blocks = math.ceil(len(P_list) / BLOCK_SIZE)
        print("GPL Kernel Start")
        GPL_PerX[total_blocks, BLOCK_SIZE](ep, k_best, m_best, y_best, Cp_list, index, X_axis, P_list, len(P_list))
        print("GPL Kernel Stop")
        del Cp_list

        print("Shuffle Start")
        total_blocks = math.ceil(len(array) / BLOCK_SIZE)
   
        x_list = cupy.array(random_pdf_kernel(X_axis, P_list, len(array)))
       
        print("Shuffle Stop")
        
        # Inverse map
        del X_axis, P_list
        print("Inverse Mapping Kernel Start")
        total_blocks = math.ceil(len(x_list) / BLOCK_SIZE)
        listmapping_inverse_fromLToReal[total_blocks, BLOCK_SIZE](x_list, sensitivity, lower, ep, k_best, m_best, y_best, index, len(x_list))
        print("Inverse Mapping Kernel Stop")

        print("Copy results")
        results = cupy.concatenate([results, x_list])
        print("Done, deleting list")
        
        print("Batch done")
        cupy._default_memory_pool.free_all_blocks()

    cupy._default_memory_pool.free_all_blocks()
    results = cupy.round(results, 5)
    results = results.get().tolist()

    # Revert output streams
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    return results
