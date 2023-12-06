import tensorflow as tf
import numpy as np
import scipy

def rect(x,width):
    return np.heaviside(x+width/2,0.5) - np.heaviside(x-width/2,0.5)


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

def ft2(data, delta):
    """
    A properly scaled 1-D FFT
    Parameters:
        data (ndarray): An array on which to perform the FFT
        delta (float): Spacing between elements
    Returns:
        ndarray: scaled FFT
    """
    DATA = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(data))) * delta**2
    return DATA

def tf_ft2(g, delta):
    g = tf.cast(g, tf.complex64)
    delta = tf.complex(tf.cast(delta, dtype=tf.float32), 0.0)
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(g))) * delta**2




def tf_ift2(g, delta_f):
    N = np.size(g,0) # assume square
    g = tf.cast(g, tf.complex64)
    delta_f = tf.complex(tf.cast(delta_f, dtype=tf.float32), 0.0)
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(g))) * (N * delta_f)**2


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


def circ(x,y,D):
    r = np.sqrt(x**2 + y**2)
    return r < D/2.0