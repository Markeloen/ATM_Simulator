import numpy as np
import matplotlib.pyplot as plt
from phasescreen import *

def main():
    N = 128
    r0 = 0.4  # Coherence parameter     # N number of grid points per side
    D = 2  # Length of a phase screen
    L0 = 100
    l0 = 0.01
    delta = D/N  # Grid Spacing
    del_f = 1./(N*delta)
    # spatial grid
    x = np.linspace(-N/2,N/2,N,dtype=float) * del_f
    (x,y) = np.meshgrid(x,x)

    phz, fx, fy = ft_phase_screen(r0, N, delta, L0, l0)
    print((fx**2+fy**2)**(-11/12))
    alpha = 0.99
    P = 2*np.pi / D * N * r0**(-5/6) * (fx**2+fy**2)**(-11/12)
    random_noise = np.random.normal(0, 1, (N, N))
    T = 2
    vx = 1
    vy = 2
    theta = -2*np.pi*T*(fx*vx+fy*vy)
    theta = np.array(theta, dtype=np.float128)
    new_phz = alpha * np.exp(theta) * phz + ((1-alpha**2)**(.5) * P * random_noise)
    
    # fig = plt.figure(figsize=(6, 3.2))

    # ax = fig.add_subplot(111)
    # ax.set_title('colorMap')
    # plt.imshow(phz)
    # ax.set_aspect('equal')

    # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    # cax.get_xaxis().set_visible(False)
    # cax.get_yaxis().set_visible(False)
    # cax.patch.set_alpha(0)
    # cax.set_frame_on(False)
    # plt.colorbar(orientation='vertical')
    # plt.show()


if __name__ == "__main__":
    main()