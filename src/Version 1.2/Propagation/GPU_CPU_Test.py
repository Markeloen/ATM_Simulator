from Propagator import propagator
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



def CPU_Prop_HCIPY(size, r0_array, v_array, time_window=100):
    size = 2**size
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

    aperture = make_rectangular_aperture(1)(pupil_grid)
    wf = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
    wf2 = atmosphere.forward(wf)
    wf2.electric_field *= aperture
    img = Field(prop(wf2).intensity, focal_grid)

    # plt.clf()
    # plt.subplot(1,2,1)
    # imshow_field(wf2.phase, cmap='RdBu')
    # plt.show()
    # plt.subplot(1,2,2)
    # imshow_field(np.log10(img / img.max()), vmin=-6)
    # plt.draw()


def GPU_Prop_Test(size, iters, n_layers):
    

    N = 2**size

    #For Phase_Screen
    r0 = 0.4  # Coherence parameter     # N number of grid points per side
    D = 2  # Length of a phase screen
    L0 = 100
    l0 = 0.01
    
    delta = D/N  # Grid Spacing
    del_f = 1./(N*delta)
    # Generate Random PSs
    a = PhaseScreenGen(r0, 2**(size), delta, L0, l0)
    ps_list = []
    for i in range(n_layers):
        ps_list.append(a.generate_instance())

    # Propagation 

    D1 = 2e-3
    D2= 6e-3
    wvl = 1e-6
    k = 2*np.pi/wvl
    z = 1
    delta1 = D1/30
    deltan=D2/30
    z = np.linspace(1,n_layers, n_layers) * z / n_layers
    z_np = np.array([0 ,*z])
    x = np.arange(-N/2., N/2.) * delta1
    [x1, y1] = np.meshgrid(x, x) 
    ap = rect(x1, D1) * rect(y1, D1)

    a = propagator(N , wvl, delta1, deltan, z)
    for _ in range(iters):
        c = a.propagate(ap, ps_list)


def CPU_Vs_HCIPY(up_to_size = 7, num_of_planes = 5, iterations = 100, save_fig = True):
    for n in range(3):
        result_dict_CPU = {}
        result_dict_GPU_object = {}
        for i in range(up_to_size):
            Start_GPU = time.time()
            GPU_Prop_Test(i+1, iterations, 5*(n+1))
            Finish_GPU = time.time()
            GPU_Time = Finish_GPU - Start_GPU

            r0_array = [0.4 for _ in range (5*(n+1))]
            v_array = [1 for _ in range (5*(n+1))]
            
            CPU_Start = time.time()
            for _ in range(iterations):
                CPU_Prop_HCIPY(2**(i+1), r0_array, v_array) 
            CPU_Finish_Time = time.time()
            CPU_Time = CPU_Finish_Time - CPU_Start
            result_dict_GPU_object[i] = round(GPU_Time, 2)
            result_dict_CPU[i] = round(CPU_Time, 2)
            print(f"Printing running time for plane size: {2**(i+1)}")
            print(f"HCIPY takes: {CPU_Time}s")
            print(f"GPU takes: {GPU_Time}s")
        keys_GPU = list(result_dict_GPU_object.keys())
        custom_tick_labels = [2 ** (f+1) for f in keys_GPU]
        values_GPU = list(result_dict_GPU_object.values())
        values_CPU = list(result_dict_CPU.values())

        plt.plot(keys_GPU, values_GPU, marker='o', linestyle='-', color='blue', label='GPU')
        plt.plot(keys_GPU, values_CPU, marker='x', linestyle='-', color='red', label='CPU-HCIPY')
        plt.xticks(keys_GPU, custom_tick_labels)
        plt.xlabel('n*n size')
        plt.ylabel('Time (seconds)')
        plt.title(f"Running time comparison for {iterations} times propagation for {5*(n+1)} plane")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.show() 
        if save_fig:
            plt.savefig(f"Comparison_{5*(n+1)}.png")

def validate_prop():
    D1 = 2e-3
    D2= 6e-3
    wvl = 1e-6
    k = 2*np.pi/wvl
    z = 1
    delta1 = D1/30
    deltan=D2/30
    N=128
    n=5
    z = np.linspace(1,n, n) * z / n
    z_np = np.array([0 ,*z])
    x = np.arange(-N/2., N/2.) * delta1
    [x1, y1] = np.meshgrid(x, x) 
    ap = rect(x1, D1) * rect(y1, D1)
    a = propagator(N , wvl, delta1, deltan, z)
    Uout, x2, y2 = a.propagate(ap)
    # tf.config.run_functions_eagerly(True)
    Uout = Uout.numpy()
    x2 = x2.numpy()
    y2 = y2.numpy()
    Dz = z[-1]
    UU = Uout[64]
    plt.plot(x2[64], abs(UU))
    Uout_an = fresnel_prop_square_ap(x2[64], 0, D1, wvl, Dz)
    plt.plot(x2[64], abs(Uout_an), marker='*')

    plt.show()

def main():
    









    # Run when going home:
    iters = 100
    n = 10
    size = 8
    result_dict_CPU = {}
    result_dict_GPU_object = {}
    for i in range(size):
        
        start_gpu = time.time()
        print(f"GPU-Calculating size {2**(i+1)}............")
        GPU_Prop_Test(i+1, iters, n)
        finish_gpu = time.time()
        GPU_time = round(finish_gpu - start_gpu,2)
        print(f"It took GPU {GPU_time}")
        start_hcipy = time.time()
        r0_arr = [0.4 for i in range(n)]
        v0_arr = [1 for i in range(n)]
        print(f"CPU-Calculating size {2**(i+1)}............")
        for _ in range(iters):
            CPU_Prop_HCIPY(i+1, r0_arr, v0_arr)
        finish_HCIPY = time.time()
        CPU_time = round(finish_HCIPY - start_hcipy,2)
        print(f"It took HCIPY {CPU_time}")
        result_dict_CPU[i] = CPU_time
        result_dict_GPU_object[i] = GPU_time
    keys_GPU = list(result_dict_GPU_object.keys())
    custom_tick_labels = [2 ** (f+1) for f in keys_GPU]
    values_GPU = list(result_dict_GPU_object.values())
    values_CPU = list(result_dict_CPU.values())

    plt.plot(keys_GPU, values_GPU, marker='o', linestyle='-', color='blue', label='GPU')
    plt.plot(keys_GPU, values_CPU, marker='x', linestyle='-', color='red', label='CPU-HCIPY')
    plt.xticks(keys_GPU, custom_tick_labels)
    plt.xlabel('n*n size')
    plt.ylabel('Time (seconds)')
    plt.title(f"100 time propagation for 10 partial props / TF VS HCIPY")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show() 

if __name__ == '__main__':
    main()