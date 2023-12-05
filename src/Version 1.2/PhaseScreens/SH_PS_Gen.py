import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numba import jit
from timeit import default_timer as timer
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import warnings

def tf_ift2(g, delta_f):
    N = np.size(g,0) # assume square
    g = tf.cast(g, tf.complex64)
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(g))) * (N * delta_f)**2



class PhaseScreenGen:
    def __init__(self, r0, N, delta, L0, l0):
        self.r0 = tf.constant(r0, dtype=tf.double)
        self.N = tf.constant(N, dtype=tf.int32)
        self.delta_f = tf.constant(1./(N*delta), dtype=tf.double)
        self.l0 = tf.constant(l0, dtype=tf.double)
        self.L0 = tf.constant(L0, dtype=tf.double)


        fx =  tf.linspace(-N//2,N//2-1, N) * self.delta_f
        [fx, fy] = tf.meshgrid(fx, fx)

        self.f = tf.sqrt(fx**2 + fy**2)

        self.fm = 5.92/self.l0/(2*np.pi)  # Inner Scale Frequency
        self.f0 = 1./self.L0

        self.PSD_phi = 0.023 * self.r0 ** (-5. / 3.) * tf.exp(-(self.f/self.fm) ** 2) / (self.f**2+ self.f0**2) ** (11/6)
        
        tf.tensor_scatter_nd_update(self.PSD_phi, [[self.N//2,self.N//2]], [0])
        
        # print("Tracing!")
    @tf.function
    def generate_instance(self):
        cn = tf.complex(tf.random.normal( (self.N,self.N) ), tf.random.normal( (self.N,self.N) )) * tf.cast(tf.math.sqrt(self.PSD_phi) * self.delta_f, tf.complex64)
        phz = tf.math.real(tf_ift2(cn, 1))

        # print("Tracing!_inst")

        return phz
    
    


    
def main():
    r0 = 0.4  # Coherence parameter     # N number of grid points per side
    D = 2  # Length of a phase screen
    L0 = 100
    l0 = 0.01
    N=128
    delta = D/N  # Grid Spacing
    del_f = 1./(N*delta)
    a = PhaseScreenGen_SH(r0, N, delta, L0, l0)
    phz = a.subHarmPhaseScreen()
    plt.imshow(phz)
    plt.show()


    

if __name__ == "__main__":
    main()
