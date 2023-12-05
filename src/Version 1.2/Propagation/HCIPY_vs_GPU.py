import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
import time
import timeit
from datetime import datetime
from hcipy import *
import os
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)
sys.path.append(parent_directory)

from PhaseScreens.PhaseScreenGen import *
from basic_funcs import *

def rect(x,width):
    return np.heaviside(x+width/2,0.5) - np.heaviside(x-width/2,0.5)

class propagator:
    def __init__(self, N , wvl, delta1, deltan, z):
        self.N = N
        x = np.arange(-N/2., N/2.)
        [nx, ny] = np.meshgrid(x, x)
        k = 2*np.pi/wvl
        nsq = nx**2 + ny**2

        w = 0.47 * N
        self.sg = np.exp(-nsq**8/w**16)

        z = np.array([0, *z])
        self.n = len(z)

        Delta_z = z[1:] - z[:self.n-1]

        alpha = z / z[-1]
        self.delta = (1-alpha) * delta1 + alpha * deltan

        m = self.delta[1:] / self.delta[:self.n-1]

        x1 = nx * self.delta[0]
        y1 = ny * self.delta[0]
        r1sq = x1**2 + y1**2

        self.Q1 = np.exp(1j*k/2*(1-m[0]) / Delta_z[0] * r1sq)
        self.Q2 = []
        self.m = m
        self.deltaf = []

        for idx in range(self.n-1):
            deltaf = 1/(N*self.delta[idx])
            self.deltaf.append(deltaf)

            fX = nx * deltaf
            fY = ny * deltaf

            fsq = fX**2 + fY**2
            Z = Delta_z[idx]

            Q2 = np.exp(-1j*(np.pi**2)*2*Z/(m[idx]*k)*fsq)
            
            self.Q2.append(Q2)
        self.Q2 = np.asarray(self.Q2)
        self.deltaf = np.asarray(self.deltaf)
        self.xn = nx * self.delta[-1]
        self.yn = ny * self.delta[-1]
        rnsq = self.xn**2 + self.yn**2
        self.Q3 = np.exp(1j*k/2*(m[-1]-1) / (m[-1] * Z) * rnsq)
        

    @tf.function
    def propagate(self,Uin, PS_arr):
        self.Q1 = tf.convert_to_tensor(self.Q1, dtype=tf.complex64)
        self.Q2 = tf.convert_to_tensor(self.Q2, dtype=tf.complex64)
        self.Q3 = tf.convert_to_tensor(self.Q3, dtype=tf.complex64)
        self.m = tf.convert_to_tensor(self.m, dtype=tf.float32)
        self.deltaf = tf.convert_to_tensor(self.deltaf, dtype=tf.float32)
        self.N = tf.convert_to_tensor(self.N, dtype=tf.float32)
        self.sg = tf.convert_to_tensor(self.sg, dtype=tf.float32)
        PS_arr = tf.convert_to_tensor(PS_arr, dtype=tf.float32)
        self.PS_arr = tf.complex(PS_arr, 0.0)
        print(f"Here is johny: {self.PS_arr.dtype}")
        print(f"Here is the shape: {self.PS_arr[0].get_shape()}")
        print(f"Here is the shape: {tf.complex(self.sg, 0.0).get_shape()}")
        
        print("tracing")

        Uin = self.Q1 * tf.cast(Uin, tf.complex64) 
        
        for idx in range(self.n-1):
            ff = tf_ift2(self.Q2[idx] * tf_ft2(Uin / tf.complex(self.m[idx], 0.0), self.delta[idx]), self.deltaf[idx])
            Uin = tf.math.multiply(tf.complex(self.sg, 0.0), self.PS_arr[idx]) * ff
        Uout = self.Q3 * Uin
        return Uout, self.xn, self.yn





def ang_spec_multi_vac_oneTime(Uin, wvl, delta1, deltan, z):

    N=len(Uin)
    
    x = np.arange(-N/2., N/2.)
    [nx, ny] = np.meshgrid(x, x)
    k = 2*np.pi/wvl

    nsq = nx**2 + ny**2
    
    w = 0.47 * N
    sg = np.exp(-nsq**8/w**16)

    z = np.array([0, *z])
    n = len(z)

    Delta_z = z[1:] - z[:n-1]

    alpha = z / z[-1]
    delta = (1-alpha) * delta1 + alpha * deltan
    m = delta[1:] / delta[:n-1]
    x1 = nx * delta[0]
    y1 = ny * delta[0]
    r1sq = x1**2 + y1**2
    # print(r1sq)
    Q1 = np.exp(1j*k/2*(1-m[0]) / Delta_z[0] * r1sq)
    Uin = Uin * Q1
    for idx in range(n-1):

        deltaf = 1/(N*delta[idx])
        fX = nx * deltaf
        fY = ny * deltaf

        fsq = fX**2 + fY**2
        Z = Delta_z[idx]

        Q2 = np.exp(-1j*(np.pi**2)*2*Z/(m[idx]*k)*fsq)
        Uin = sg * ift2(Q2 * ft2(Uin/m[idx], delta[idx]), deltaf)

    xn = nx * delta[-1]
    yn = ny * delta[-1]
    rnsq = xn**2 + yn**2
    Q3 = np.exp(1j*k/2*(m[-1]-1) / (m[-1] * Z) * rnsq)

    Uout = Q3 * Uin

    


def fresnel_prop_square_ap(x2, y2, D1, wvl, Dz):
    N_F = (D1/2)**2 / (wvl*Dz)

    bigX = x2 / np.sqrt(wvl*Dz)
    bigY = y2 / np.sqrt(wvl*Dz)
    alpha1 = -np.sqrt(2) * (np.sqrt(N_F) + bigX)
    alpha2 = np.sqrt(2) * (np.sqrt(N_F) - bigX)
    beta1 = -np.sqrt(2) * (np.sqrt(N_F) + bigY)
    beta2 = np.sqrt(2) * (np.sqrt(N_F) - bigY)

    sa1, ca1 = scipy.special.fresnel(alpha1)
    sa2, ca2 = scipy.special.fresnel(alpha2)
    sb1, cb1 = scipy.special.fresnel(beta1)
    sb2, cb2 = scipy.special.fresnel(beta2)

    U = 1/(2*1j) * ((ca2-ca1) + 1j * (sa2 - sa1)) * ((cb2 - cb1) + 1j * (sb2 - sb1))
    

    return U
def MultiLayerAtmosphere_no_evol(size, r0_array, v_array, time_window=100):
    pupil_grid = make_pupil_grid(size, 1)
    focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, 8, 16)
    wavelength = 1e-6
    L0 = 100

    layers = []

    n = len(r0_array)

    for i in range(n):

        layer = FiniteAtmosphericLayer(pupil_grid, 5, L0, v_array[i], 0)
        layers.append(layer)


    atmosphere = MultiLayerAtmosphere(layers, False)  # No Scintillation effect
    atmosphere.Cn_squared = Cn_squared_from_fried_parameter(1/40, wavelength)
    prop = AngularSpectrumPropagator(pupil_grid, 1)

    aperture = make_circular_aperture(1)(pupil_grid)
    wf = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
    wf2 = atmosphere.forward(wf)
    wf2.electric_field *= aperture
    img = Field(prop(wf2).intensity, focal_grid)

    plt.clf()
    plt.subplot(1,2,1)
    imshow_field(wf2.phase, cmap='RdBu')
    # plt.show()
    # plt.subplot(1,2,2)
    # imshow_field(np.log10(img / img.max()), vmin=-6)
    # plt.draw()


def main():
    
    

    """ VALIDATION of propagation! ---> Uncomment the whole section!  """

    # D1 = 2e-3
    # D2= 6e-3
    # wvl = 1e-6
    # k = 2*np.pi/wvl
    # z = 1
    # delta1 = D1/30
    # deltan=D2/30
    # N=128
    # n=5
    # z = np.linspace(1,n, n) * z / n
    # z_np = np.array([0 ,*z])
    # x = np.arange(-N/2., N/2.) * delta1
    # [x1, y1] = np.meshgrid(x, x) 
    # ap = rect(x1, D1) * rect(y1, D1)
    # a = propagator(N , wvl, delta1, deltan, z)
    # Uout, x2, y2 = a.propagate(ap)
    # # tf.config.run_functions_eagerly(True)
    # Uout = Uout.numpy()
    # x2 = x2.numpy()
    # y2 = y2.numpy()
    # Dz = z[-1]
    # UU = Uout[64]
    # plt.plot(x2[64], abs(UU))
    # Uout_an = fresnel_prop_square_ap(x2[64], 0, D1, wvl, Dz)
    # plt.plot(x2[64], abs(Uout_an), marker='*')

    # plt.show()



    """ Time managmenet  """

    # D1 = 2e-3
    # D2= 6e-3
    # wvl = 1e-6
    # k = 2*np.pi/wvl
    # z = 1
    # delta1 = D1/30
    # deltan=D2/30
    # N=256
    # n=5

    # z = np.linspace(1,n, n) * z / n
    # z_np = np.array([0 ,*z])
    # GPU_time_start = time.time()
    # x = np.arange(-N/2., N/2.) * delta1
    # [x1, y1] = np.meshgrid(x, x) 
    # ap = rect(x1, D1) * rect(y1, D1)
    # Iterations = 100

    # a = propagator(N , wvl, delta1, deltan, z)
    # One_time_setup = time.time()
    # # print(f"One time setup time: {GPU_time_start - One_time_setup}")
    # Iterations_time = time.time()
    # for _ in range(Iterations):
    #     Uout, x2, y2 = a.propagate(ap)
    # Finish_time_GPU = time.time()
    # print(f"Iterations time: {Finish_time_GPU - Iterations_time}")
    # print(f"One time setup time: {One_time_setup - GPU_time_start}")

    # CPU_Time = time.time()
    # for _ in range(Iterations):
    #     ang_spec_multi_vac_oneTime(ap , wvl, delta1, deltan, z)
    # CPU_Finish_Time = time.time()
    # print(f"CPU running time: {CPU_Finish_Time - CPU_Time}")
    PS_size = 7     #HERE's the size
    N = 2**PS_size
    GPU_Start = time.time()
    iterations = 500
    size = 128
    n=5
    r0 = 0.4  # Coherence parameter     # N number of grid points per side
    D = 2  # Length of a phase screen
    L0 = 100
    l0 = 0.01
    
    delta = D/N  # Grid Spacing
    del_f = 1./(N*delta)
    PS_size = 7
    a = PhaseScreenGen(r0, 2**(PS_size), delta, L0, l0)  # Change, 
    ps_list = []
    for i in range(n):
        ps_list.append(a.generate_instance())

    

    n=5
    r0 = 0.4  # Coherence parameter     # N number of grid points per side
    D = 2  # Length of a phase screen
    L0 = 100
    l0 = 0.01
    delta = D/N  # Grid Spacing
    del_f = 1./(N*delta)
    
    

    # PROP
    
    D1 = 2e-3
    D2= 6e-3
    wvl = 1e-6
    k = 2*np.pi/wvl
    z = 1
    delta1 = D1/30
    deltan=D2/30
    z = np.linspace(1,n, n) * z / n
    z_np = np.array([0 ,*z])
    x = np.arange(-N/2., N/2.) * delta1
    [x1, y1] = np.meshgrid(x, x) 
    ap = rect(x1, D1) * rect(y1, D1)


    a = propagator(N , wvl, delta1, deltan, z)
    for _ in range(iterations):
        a.propagate(ap, ps_list)

    Finish_GPU = time.time()
    GPU_OP = Finish_GPU - GPU_Start
    
    print(f"GPU-TF takes: {GPU_OP}s")

    r0_array = [0.4, 0.4, 0.4, 0.4, 0.4]
    v_array = [1, 1, 1, 1 ,1]
    
    size = 128      # Here's the size
    n = 5
    CPU_Time = time.time()
    for _ in range(iterations):
        MultiLayerAtmosphere_no_evol(size, r0_array, v_array) 
    CPU_Finish_Time = time.time()
    CPU_OP = CPU_Finish_Time - CPU_Time
    print(f"HCIPY takes: {CPU_OP}s")



if __name__ == '__main__':
    main()