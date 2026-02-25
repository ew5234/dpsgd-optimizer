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

# Suppress all Numba warnings
warnings.simplefilter('ignore', category=NumbaWarning)

def random_pdf(x_axis, P_list, array_size):
    return_list = []
    for i in range(array_size):
        P_section = P_list[i * 10001: (i + 1)*10001]
        P_sum = cupy.sum(P_section)
        prob_list = P_section / P_sum
        cumu_probs = cupy.cumsum(prob_list)
        random = math.random(0, 100)
        for j, cp in cumu_probs:
            if random <= cp:
                return_list.append(x_axis[j])
                break
    return return_list
def perturb_array(A, ep, sensitivity, lower, index, b_size=256, BLOCK_SIZE=32):
    # A is a normal 1D python array
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    #sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    k_best, m_best, y_best = parameter_optimization(ep, index)
    results = cupy.empty(0)
    for b in range(0, len(A), b_size):
        array = A[b:b + b_size]

        Cp_list = []
        print("Filling CP")
        for element in array:
            Cp = mapping_fromRealToL(element, sensitivity, lower, ep, k_best, m_best, y_best, index)
            Cp_list.append(Cp)
        print("Done filling CP")
        # Allocate Memory
        Cp_list = cupy.array(Cp_list)
        X_axis = cupy.zeros(10001 * (len(array)), dtype=cupy.float32) # 10001 comes from divid + 1 from generate_perturbed_list
        P_axis = cupy.zeros(10001 * (len(array)), dtype=cupy.float32)
        P_list = cupy.zeros(10001 * (len(array)), dtype=cupy.float32)
        total_blocks = math.ceil(len(P_list) / BLOCK_SIZE)
        print("GPL Kernel Start")
        #GPL[total_blocks, BLOCK_SIZE](ep, k_best, m_best, y_best, Cp_list, index, X_axis, P_axis, P_list, len(array))
        GPL_PerX[total_blocks, BLOCK_SIZE](ep, k_best, m_best, y_best, Cp_list, index, X_axis, P_list, len(P_list))
        print("GPL Kernel Stop")
        del P_axis, Cp_list
        # Fill List for PDF
  
        rp_list_size = int(cupy.sum(P_list).item())
        BLOCK_SIZE = 1024
        print(rp_list_size)
        total_blocks = math.ceil(len(P_list) / BLOCK_SIZE)
        
        rp_list = cupy.zeros(rp_list_size, dtype=cupy.float16)
        cumulated_sum = cupy.zeros(len(P_list) + 1)
        cumulated_sum[1:] = cupy.cumsum(P_list)
        print("RP Fill Kernel Start")
        fill_rp[total_blocks, BLOCK_SIZE](P_list, X_axis, cumulated_sum, len(P_list), rp_list)
        # seg_size = 1000
        # fill_rp_coursened[total_blocks, BLOCK_SIZE](X_axis, cumulated_sum, len(P_list), rp_list, rp_list_size, seg_size)
        print("RP Fill Kernel Stop")
        print(rp_list)
        print("Shuffle Start")
        rp_list_new = cupy.zeros(len(array))
        total_blocks = math.ceil(len(array) / BLOCK_SIZE)
        # random_element_rp_new[total_blocks, BLOCK_SIZE](rp_list_new, cumulated_sum, len(array), rp_list, rp_list_size, random.randint(1,1000))
        x_list = cupy.array(random_pdf(X_axis, P_list, len(array)))
                
        print("Shuffle Stop")
        rp_list = rp_list_new
        
        # Inverse map
        del X_axis, P_list
        print("Inverse Mapping Kernel Start")
        total_blocks = math.ceil(len(x_list) / BLOCK_SIZE)
        listmapping_inverse_fromLToReal[total_blocks, BLOCK_SIZE](x_list, sensitivity, lower, ep, k_best, m_best, y_best, index, len(x_list))
        print("Inverse Mapping Kernel Stop")

        print("Copy results")
        # P_indices = np.arange(len(array)) * 10001
        # first_rp_indices = cumulated_sum[P_indices].astype(int)
        # curr_results = rp_list[first_rp_indices]

        # results = cupy.concatenate([results, curr_results])
        results = cupy.concatenate([results, x_list])
        print("Done, deleting list")
        del rp_list, x_list
        print("Batch done")
        cupy._default_memory_pool.free_all_blocks()

    cupy._default_memory_pool.free_all_blocks()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    results = cupy.round(results, 5)
    results = results.get().tolist()
    return results
