from cudamech.Constraints import *
from numba import cuda
# ==================Mapping=================#
@cuda.jit(device=True)
def sensitivity_Cp_fun(ep, k, m, y, index):
    L = LValue(ep, k, m, y, index)
    
    if (index == 1) or (index == 4):
        retVal = k * m * (2 * L - m)
        return retVal
    if (index == 2) or (index == 5):
        retVal = 2 * k * m * (2 * L - m) / math.pi
        return retVal
    if (index == 3) or (index == 6):
        retVal = k * m * (2 * L - m) / 2
        return retVal
    return 1.0

@cuda.jit(device=True)
def mapping_fromRealToL(input_value, sensitivity_f, lower, ep, k, m, y, index):
    L = LValue(ep, k, m, y, index)
    sensitivity_Cp = sensitivity_Cp_fun(ep, k, m, y, index)
    C = sensitivity_Cp / sensitivity_f
    mapped_value = (input_value - lower) * C - L

    return mapped_value

@cuda.jit(device=True)
def mapping_inverse_fromLToReal(input_value, sensitivity_f, lower, ep, k, m, y, index, i):
    L = LValue(ep, k, m, y, index)
    sensitivity_Cp = sensitivity_Cp_fun(ep, k, m, y, index)
    C = sensitivity_Cp / sensitivity_f
    mapped_inverse_value = (input_value + L) / C + lower

    return mapped_inverse_value

@cuda.jit
def listmapping_inverse_fromLToReal(input_list, sensitivity_f, lower, ep, k, m, y, index, list_len):
    i = cuda.grid(1)
    if i < list_len:
        input_val = input_list[i]
        final_val = mapping_inverse_fromLToReal(input_val, sensitivity_f, lower, ep, k, m, y, index, i)
        input_list[i] = float(final_val)
        cuda.syncthreads()


# ==========================================#

