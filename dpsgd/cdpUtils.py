import math
import numpy
import tensorflow as tf
from cudamech.cudaGL_new import *
from cudamech.Perturbation_Mechanism_v2_GPU import *

def BatchClipByL2norm(t, upper_bound, name=None):
    #params
    ep = 1
    c = 2 #90
    sensitivity = 2*c/128
    lower = -c
    index = 1

    assert upper_bound > 0
    saved_shape = tf.shape(t)
    #batch_size = tf.slice(saved_shape, [0], [1])
    batch_size = tf.constant(1.0)
    shape = tf.concat(0, [[tf.constant(128)], [-1]])
    t2 = tf.reshape(t, shape)
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                            tf.constant(1.0/upper_bound))
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.math.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
    scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
    return clipped_t

def AddGaussianNoise(t, ep, sensitivity, lower, index, name=None): # Main noise function for CDP, not Gaussian
  
  noisy_t = perturb_array(tf.reshape(tf.reduce_mean(t), [1]), ep, sensitivity, lower, index, b_size=4096, BLOCK_SIZE=32, same_best=True)
  # noisy_t = perturbation_fun_optimized_oneCall_v2_gpu(ep, float(tf.reduce_mean(t)), sensitivity, lower, index)
  # noisy_t = t + tf.reshape(noisy_t, [-1, 1])
  
  noisy_t = t + tf.fill(t.shape, noisy_t)
  return noisy_t

def GetTensorOpName(x):
  t = x.name.rsplit(":", 1)
  if len(t) == 1:
    return x.name
  else:
    return t[0]
