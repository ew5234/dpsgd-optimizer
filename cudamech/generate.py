from numba import cuda
import numba
import math

from cudamech.Constraints import *
from cudamech.Mapping import mapping_fromRealToL

@cuda.jit(device=True)
def PDF_fun(x, ep, k, m, y, Cp, index):
    L = LValue(ep, k, m, y, index)
    a = aValue(ep, k, m, y, Cp, index)
    t = (y + k) / math.e ** ep

    if (index == 1):
        P = 0
        if (x >= -L) and (x < a):
            P = y
        if (x >= a) and (x < a + m):
            P = y + k
        if (x >= a + m) and (x <= L):
            P = y

        return float3f(P)

    if (index == 2):
        P = 0
        if (x >= -L) and (x < a):
            P = y
        if (x >= a) and (x < a + m):
            P = y + k * math.sin(math.pi / m * (x - a))
        if (x >= a + m) and (x <= L):
            P = y
        return float3f(P)

    if (index == 3):
        P = 0
        if (x >= -L) and (x < a):
            P = y
        if (x >= a) and (x < a + m / 2):
            P = y + 2 * k / m * x - 2 * a * k / m
        if (x >= a + m / 2) and (x < a + m):
            P = y - 2 * k / m * x + 2 * (a + m) * k / m
        if (x >= a + m) and (x <= L):
            P = y
        return float3f(P)

    if (index == 4):
        P = 0
        if (x >= -L) and (x < a):
            P = -(y - t) / L ** 4 * x ** 4 + y
        if (x >= a) and (x < a + m):
            P = -(y - t) / L ** 4 * x ** 4 + y + k
        if (x >= a + m) and (x <= L):
            P = -(y - t) / L ** 4 * x ** 4 + y
        return float3f(P)

    if (index == 5):
        P = 0
        if (x >= -L) and (x < a):
            P = -(y - t) / L ** 4 * x ** 4 + y
        if (x >= a) and (x < a + m):
            P = -(y - t) / L ** 4 * x ** 4 + y + k * math.sin(math.pi / m * (x - a))
        if (x >= a + m) and (x <= L):
            P = -(y - t) / L ** 4 * x ** 4 + y
        return float3f(P)

    if (index == 6):
        P = 0
        if (x >= -L) and (x < a):
            P = -(y - t) / L ** 4 * x ** 4 + y
        if (x >= a) and (x < a + m / 2):
            P = -(y - t) / L ** 4 * x ** 4 + y + 2 * k / m * x - 2 * a * k / m
        if (x >= a + m / 2) and (x < a + m):
            P = -(y - t) / L ** 4 * x ** 4 + y - 2 * k / m * x + 2 * (a + m) * k / m
        if (x >= a + m) and (x <= L):
            P = -(y - t) / L ** 4 * x ** 4 + y
        return float3f(P)
    
    return 1

@cuda.jit
def random_element_rp_new(x_axis, all_cumu_probs, array_size, rand_nums, output_array):
    i = cuda.grid(1)
    if i < array_size:
        cumu_id = i * 10001
        for j in range(10001):
            if rand_nums[i] <= all_cumu_probs[j + cumu_id]:
                output_array[i] = x_axis[j]
                break
    
@cuda.jit
def fill_rp(P_list, X_axis, cumulative_sum_list, list_len, output):
    i = cuda.grid(1)
    if i < list_len:
        value = int(P_list[i])
        c_sum_val = int(cumulative_sum_list[i])
        for j in range(value):
            output[j + c_sum_val] = X_axis[i]
        cuda.syncthreads() 
            
@cuda.jit(device=True)
def bin_search(list, i, start, end):
    for _ in range(end):
        mid = int(start + math.ceil((end - start - 1)/2))
        c_val = list[mid] - 1
        if i <= c_val:
            end = mid + 1
        elif i > c_val:
            start = mid + 1

        dif_len = end - start - 1
        if dif_len == 1:
            if i <= list[start]:
                end = start + 1
            else:
                start = end - 1
            return start
        if dif_len == 0:
            return start
    
    return 0 # Necessary for numba to consider this function non-optional

@cuda.jit
def fill_rp_coursened(X_axis, cumulative_sum_list, list_len, output, rp_list_len, seg_len):
    i = cuda.grid(1) * seg_len
    
    for seg_i in range(seg_len):
        i = int(i + seg_i)
        if i < rp_list_len:
            section_id = bin_search(cumulative_sum_list, i, 0, list_len)
            output[i] = X_axis[section_id]

@cuda.jit
def GPL_PerX(ep, k, m, y, Cp_list, index_f, X_axis, P_list, input_size):
    i = cuda.grid(1)
    if i < input_size:
        section_id = int(math.floor(i / 10001))
        Cp = Cp_list[section_id]
        
        L = LValue(ep, k, m, y, index_f)
        a = aValue(ep, k, m, y, Cp, index_f)
        
        divid = 10000
        step = float(2 * L / divid)
        
        x_count = -L + ((i % 10001) * step)
        
        P_x = PDF_fun(x_count, ep, k, m, y, Cp, index_f)
        X_axis[i] = x_count
        
        P_list[i] = int(P_x * 1000)
            
    cuda.syncthreads()

    
@cuda.jit
def GPL(ep, k, m, y, Cp_list, index_f, X_axis, P_axis, P_list, input_size):
    i = cuda.grid(1)
    if i < input_size:
        Cp = Cp_list[i]
        
        L = LValue(ep, k, m, y, index_f)
        a = aValue(ep, k, m, y, Cp, index_f)
        
        divid = 10000
        step = float(2 * L / divid)
        x_count = -L
        index = 0
        while (x_count <= L):   
            P_x = PDF_fun(x_count, ep, k, m, y, Cp, index_f)
            
            P_axis[index + i * (divid + 1)] = P_x
            X_axis[index + i * (divid + 1)] = x_count        
            index = int(index + 1)
            x_count = float(x_count + step)
        
        for i2 in range(divid + 1): # Size of X_Axis is always divid + 1
            rp = P_axis[i2 + i * (divid + 1)]
            rp = int(rp * 1000)
            P_list[i2 + i * (divid + 1)] = rp
            
        
        cuda.syncthreads()
        
import numba
@cuda.jit
def random_element_rp(output, cumulated_sum, array_size, rp_list, rp_list_size, seed):
    i = cuda.grid(1)
    if i < array_size:
        rp_index = i * 10001
        start_index = int(cumulated_sum[rp_index])
        end_index = int(cumulated_sum[rp_index + 10000])
        end_index = min(end_index, rp_list_size)

        #GPU random
        rng = i + seed
        rng ^= rng << 13
        rng ^= rng >> 17
        rng ^= rng << 5
        rand_offset = int(rng % (end_index - start_index))
        
        output[i] = rp_list[start_index + rand_offset]

@cuda.jit
def cp_map(input_array, sensitivity, lower, ep, k_best, m_best, y_best, index, input_len, output_array):
    i = cuda.grid(1)
    if i < input_len:
        input = input_array[i] # Fetch input from memory first
        output = mapping_fromRealToL(input, sensitivity, lower, ep, k_best, m_best, y_best, index)
        output_array[i] = output