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
    def __init__(self, r0, N, delta, L0, l0, N_p = 3):
        self.r0 = r0
        self.N = N
        self.delta = delta
        self.L0 = L0
        self.l0 = l0
        self.N_p = N_p
        self.delta_f = 1./(self.N*self.delta)

        self.FFT_PS_one_time_setup()
        self.FFT_SH_PS_one_time_setup()

    def FFT_PS_one_time_setup(self):
        self.del_f = 1/(self.N*self.delta)
        x = np.arange(-self.N/2., self.N/2.)
        [fx, fy] = np.meshgrid(x, x)

        f = np.sqrt(fx**2 + fy**2)
        fm = 5.92 / (self.l0*2*np.pi)
        f0 = 1/self.L0

        self.PSD_phi = 0.023 * self.r0**(-5/3) * np.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6)
        self.PSD_phi[self.N//2][self.N//2] = 0

    def FFT_SH_PS_one_time_setup(self):

        self.del_f_SH = np.zeros(self.N_p)      # TO store N_P*del_fs for later

        # self.PSD_phi_SH = list(range(self.N_p))

        self.PSD_phi_SH = []

        self.const_SH_PS = list(range(self.N_p**2))
        x = np.arange(-self.N/2., self.N/2.)
        [self.x, self.y] = np.meshgrid(x, x)  
        self.x *= self.delta
        self.y *= self.delta
        for p in range(self.N_p):
            self.del_f_SH[p] = 1 / (self.N_p**(p+1)*(self.N*self.delta))
            x = np.arange(-self.N/2., self.N/2.)
            [fx_SH, fy_SH] = np.meshgrid(x, x)
            f_SH = np.sqrt(fx_SH**2 + fy_SH**2)
            fm_SH = 5.92 / (self.l0*2*np.pi)
            f0_SH = 1/self.L0

            # self.PSD_phi_SH[p] = ( 0.023 * self.r0**(-5/3) * np.exp(-(f_SH/fm_SH)**2) / \
            #         (f_SH**2 + f0_SH**2)**(11/6) )

            self.PSD_phi_SH.append( ( 0.023 * self.r0**(-5/3) * np.exp(-(f_SH/fm_SH)**2) / \
                    (f_SH**2 + f0_SH**2)**(11/6) ))
            self.PSD_phi_SH[p][self.N//2][self.N//2] = 0
            for ii in range(self.N_p**2):
                self.const_SH_PS[ii] = np.exp(1j*2*np.pi*(fx_SH[ii]*self.x + fy_SH[ii]*self.y))


    @tf.function
    def generate_instance(self):
        self.N = tf.convert_to_tensor(self.N, dtype=tf.int32)
        self.PSD_phi = tf.convert_to_tensor(self.PSD_phi, dtype=tf.float32)
        self.delta_f = tf.convert_to_tensor(self.delta_f, dtype=tf.float32)

        cn = tf.complex(tf.random.normal( (self.N,self.N) ), tf.random.normal( (self.N,self.N) )) * tf.cast(tf.math.sqrt(self.PSD_phi) * self.delta_f, tf.complex64)
        phz = tf.math.real(tf_ift2(cn, 1))

        # print("Tracing!_inst")

        return phz
    
    ### fix it from here! some has to be arrays some not
    

    @tf.function
    def generate_SH_instance(self):
        self.N = tf.convert_to_tensor(self.N, dtype=tf.int32)
        self.delta_f = tf.convert_to_tensor(self.delta_f, dtype=tf.float32)
        self.PSD_phi_SH = [tf.convert_to_tensor(arr, dtype=tf.float32) for arr in self.PSD_phi_SH]
        self.const_SH_PS = [tf.complex(tf.cast(arr, tf.float32), 0.0) for arr in self.const_SH_PS]
        
        phz_hi = tf.complex(self.generate_instance(), 0.0)
        phz_lo = tf.zeros_like(phz_hi, tf.complex64)

        # high frequency screen form FFT
        for p in range(self.N_p):
            cn = tf.complex(tf.random.normal( (self.N,self.N) ), tf.random.normal( (self.N,self.N) )) * tf.cast(tf.math.sqrt(self.PSD_phi_SH) * self.delta_f, tf.complex64)
            SH = tf.zeros((self.N,self.N), tf.complex64)
            for ii in range(self.N_p**2):
                r, c = [ii%self.N_p, ii//self.N_p]
                SH = SH + cn[r, c] * self.const_SH_PS[ii]
            phz_lo = phz_lo + SH
        phz_lo = phz_lo - tf.reduce_mean(phz_lo)
        phz_lo_real = tf.math.real(phz_lo)
        phz_lo_real = tf.complex(phz_lo_real, 0.0)
        phase_real = phz_lo_real + phz_hi

        

        return phase_real
        

    
    

    
def main():
    r0 = 0.4  # Coherence parameter     # N number of grid points per side
    D = 2  # Length of a phase screen
    L0 = 100
    l0 = 0.01
    N=128
    delta = D/N  # Grid Spacing
    del_f = 1./(N*delta)
    a = PhaseScreenGen(r0, N, delta, L0, l0)
    a.generate_SH_instance()
    
    

    

if __name__ == "__main__":
    main()
