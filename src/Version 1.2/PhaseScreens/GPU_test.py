import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from PhaseScreenGen import PhaseScreenGen
from phasescreen import *
from phasescreen_tf import *


def test_PSGEN_class(PS_size = 8, iterations_time = 1000, save_fig = False):
    print("HE")
    r0 = 0.4  # Coherence parameter     # N number of grid points per side
    D = 2  # Length of a phase screen
    L0 = 100
    l0 = 0.01
    N=2**PS_size
    delta = D/N  # Grid Spacing
    del_f = 1./(N*delta)

    result_dict_GPU_object = {}
    result_dict_GPU_func = {}
    result_dict_NOGPU = {}
    
    
    
    for n in range(PS_size):
        start = timer()
        for _ in range(iterations_time):
            ft_phase_screen(r0, 2**(n+1), delta, L0, l0)
        end = timer()
        NO_GPU = round(end - start,2)
        result_dict_NOGPU[n] = NO_GPU
        
        a = PhaseScreenGen(r0, 2**(n+1), delta, L0, l0)
        start = timer()
        for _ in range(iterations_time):
            c = a.generate_instance()
        end = timer()
        TF = round(end - start,2)
        result_dict_GPU_object[n] = TF

        start = timer()
        for _ in range(iterations_time):
            TF_ft_phase_screen(r0, 2**(n+1), delta, L0, l0)
        end = timer()
        GPU_with_Function = round(end - start,2)
        result_dict_GPU_func[n] = GPU_with_Function
    
    # print(f"Time for - {N}*{N} NO_GPU: {NO_GPU} GPU & Function: {GPU_with_Function} - VS GPU: {TF}")
    
    keys_GPU = list(result_dict_GPU_object.keys())
    custom_tick_labels = [2 ** (n+1) for n in keys_GPU]
    values_GPU_obj = list(result_dict_GPU_object.values())
    values_GPU_func = list(result_dict_GPU_func.values())
    values_NGPU = list(result_dict_NOGPU.values())
    plt.plot(keys_GPU, values_GPU_obj, marker='o', linestyle='-', color='blue', label='GPU - object')
    plt.plot(keys_GPU, values_GPU_func, marker='x', linestyle='-', color='red', label='GPU - func')
    plt.plot(keys_GPU, values_NGPU, marker='s', linestyle='--', color='orange', label='NO GPU')
    plt.xticks(keys_GPU, custom_tick_labels)
    plt.xlabel('n*n size')
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of Time with GPU and without GPU')
    plt.legend()

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show() 

    if save_fig:
        plt.savefig('Comparison.png')


def main():


    test_PSGEN_class(8, 1000, save_fig = True)





if __name__ == "__main__":
    main()