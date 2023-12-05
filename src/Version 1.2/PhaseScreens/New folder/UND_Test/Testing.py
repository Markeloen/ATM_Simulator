import numpy
import matplotlib.pylab as plt
from tqdm import tqdm, trange, tqdm_notebook
from astropy.io import fits
import time
import imageio
import os
# %matplotlib inline


from phasescreen import *
from infinitephasescreen import *

from matplotlib import animation, rc




def main():
    # Set up parameters for creating phase screens
    
    nx_size = 1024
    D = 100.
    pxl_scale = D/nx_size
    r0 = 2
    L0 = 10
    # wind_speed = 10 #m/s 
    # n_tests = 25 # 16
    # n_scrns = 100
    stencil_length_factor = 32

    s = time.time()
    phase_screen = PhaseScreenKolmogorov(nx_size, pxl_scale, r0, L0, stencil_length_factor=stencil_length_factor)
    print(round(time.time()-s,2))
    plt.figure()
    plt.imshow(phase_screen.scrn)
    cbar = plt.colorbar()
    cbar.set_label('Wavefront deviation (radians)', labelpad=8)


    #save it
    filenames = []
    frames = 100


    for i in range(frames):
        # s = time.time()
        # phase_screen.add_row()
        # print(round(time.time()-s,2))
        # # plt.imshow(phase_screen.scrn)
        # plt.draw()
        # plt.pause(1)


        filename = f'frame_{i}.png'
        # plt.savefig(filename)
        # plt.close()
        filenames.append(filename)

    # Create a GIF
    # with imageio.get_writer('my_animation.gif', mode='I', duration=0.1) as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    # Remove files
    for filename in filenames:
        os.remove(filename)

if __name__ == "__main__":
    main()