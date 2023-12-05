import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from PhaseScreenGen import PhaseScreenGen
import keras as k

@tf.function
def TF_ft_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):
    

    del_f = 1./(N*delta)

    fx = tf.linspace(-N//2,N//2-1, N) * del_f
    [fx, fy] = tf.meshgrid(fx, fx)

    f = tf.sqrt(fx**2 + fy**2)


    
    fm = 5.92/l0/(2*np.pi)  # Inner Scale Frequency
    f0 = 1./L0               # outer scale frequency [1/m]


    
    PSD_phi = 0.023 * r0 ** (-5. / 3.) * tf.exp(-(f/fm) ** 2) / (f**2+ f0**2) ** (11/6)
    
    tf.tensor_scatter_nd_update(PSD_phi, [[N//2,N//2]], [0])
    cn = tf.complex(tf.random.normal( (N,N) ), tf.random.normal( (N,N) )) * tf.cast(tf.math.sqrt(PSD_phi) * del_f, tf.complex64)
    phz = tf.math.real(tf_ift2(cn, 1))
    # print("Tracing!")

    return phz

def tf_ift2(g, delta_f):
    N = np.size(g,0) # assume square
    g = tf.cast(g, tf.complex64)
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(g))) * (N * delta_f)**2



