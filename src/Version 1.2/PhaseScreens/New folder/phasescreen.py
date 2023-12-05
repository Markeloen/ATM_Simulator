import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numba import jit
from timeit import default_timer as timer
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import warnings


def ft_sh_phaseScreen(r0, N, delta, L0, l0, FFT=None, seed=None):
    R = np.random.default_rng(seed)

    D = N * delta
    # high-frequency screen from FFT method
    phs_hi = ft_phase_screen(r0, N, delta, L0, l0, FFT, seed=seed)
    x = np.arange(-N/2., N/2.) * delta
    (x,y) = np.meshgrid(x,x)
    phs_lo = np.zeros(phs_hi.shape)

    # loop over frequency grids with spacing 1/(3^p*L)
    for p in range(1,4):
        # setup the PSD
        del_f = 1 / (3**p*D) #frequency grid spacing [1/m]
        fx = np.arange(-1,2) * del_f

        # frequency grid [1/m]
        fx, fy = np.meshgrid(fx,fx)
        f = np.sqrt(fx**2 +  fy**2) # polar grid

        fm = 5.92/l0/(2*np.pi) # inner scale frequency [1/m]
        f0 = 1./L0

        # outer scale frequency [1/m]
        # modified von Karman atmospheric phase PSD
        PSD_phi = (0.023*r0**(-5./3)
                    * np.exp(-1*(f/fm)**2) / ((f**2 + f0**2)**(11./6)) )
        PSD_phi[1,1] = 0

        # random draws of Fourier coefficients
        cn = ( (R.normal(size=(3,3))
            + 1j*R.normal(size=(3,3)) )
                        * np.sqrt(PSD_phi)*del_f )
        SH = np.zeros((N,N),dtype="complex")
        # loop over frequencies on this grid
        for i in range(0, 3):
            for j in range(0, 3):

                SH += cn[i,j] * np.exp(1j*2*np.pi*(fx[i,j]*x+fy[i,j]*y))

        phs_lo = phs_lo + SH
        # accumulate subharmonics

    phs_lo = phs_lo.real - phs_lo.real.mean()

    phs = phs_lo+phs_hi

    return phs


# 

def ft_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):
    delta = float(delta)
    r0 = float(r0)
    L0 = float(L0)
    l0 = float(l0)
    R = np.random.default_rng(seed)

    del_f = 1./(N*delta)

    fx = np.arange(-N/2., N/2.) * del_f
    (fx,fy) = np.meshgrid(fx,fx)
    # print(fx[0].size)

    f = np.sqrt(fx**2. + fy**2)

    
    fm = 5.92/l0/(2*np.pi)  # Inner Scale Frequency
    f0 = 1./L0               # outer scale frequency [1/m]

    
    PSD_phi = 0.023 * r0 ** (-5. / 3.) * np.exp(-(f/fm) ** 2) / (f**2+ f0**2) ** (11/6) # Von Karman PSD
    PSD_phi[int(N/2),int(N/2)] = 0

    cn = ((R.normal(size=(N, N))+1j * R.normal(size=(N, N))) * np.sqrt(PSD_phi)*del_f)
    phs = ift2(cn, 1).real
    return phs, fx, fy


# Make a SH phase screen

def example_1(N = 256, plots=False):
    # R = np.random.default_rng()
    R = tf.random.Generator.from_seed(2)
    r0 = 0.4  # Coherence parameter     # N number of grid points per side
    D = 2  # Length of a phase screen
    L0 = 100
    l0 = 0.01

    
    
    delta = D/N  # Grid Spacing
    del_f = 1./(N*delta)
    # spatial grid
    x = np.linspace(-N/2,N/2,N,dtype=float) * del_f
    (x,y) = np.meshgrid(x,x)
    start = timer()
    for i in range(100):
        phz = ft_sh_phaseScreen(r0, N, delta, L0, l0)
    end = timer()
    if(plots):
        plt.imshow(phz)
        fig = plt.figure(figsize=(6, 3.2))

        ax = fig.add_subplot(111)
        ax.set_title('colorMap')
        plt.imshow(phz)
        ax.set_aspect('equal')

        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
        plt.show()
    return end - start






# From AOTOOLS

def ift2(G, delta_f, FFT=None):
    """
    Wrapper for inverse fourier transform

    Parameters:
        G: data to transform
        delta_f: pixel seperation
        FFT (FFT object, optional): An accelerated FFT object
    """

    N = G.shape[0]

    if FFT:
        g = np.fft.fftshift(FFT(np.fft.fftshift(G))) * (N * delta_f) ** 2
    else:
        g = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(G))) * (N * delta_f) ** 2

    return g


    


def main():
    
    for i in range(4):
        start = timer()
        for _ in range(100):
            example_1(2**(i+1))
        end = timer()
        print(f"Time for creating size {2**(i+1)} phase screen: {end - start}")
    
    # See a phase screen
    # print(f"Time for creating size {N} phase screen: {example_1(N)}/")



    print(tf.config.list_physical_devices('GPU'))


if __name__ == "__main__":
    main()
