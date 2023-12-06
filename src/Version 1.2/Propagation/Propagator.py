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
from .basic_funcs import *


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
        # print(f"Here is johny: {self.PS_arr.dtype}")
        # print(f"Here is the shape: {self.PS_arr[0].get_shape()}")
        # print(f"Here is the shape: {tf.complex(self.sg, 0.0).get_shape()}")
        
        # print("tracing")

        Uin = self.Q1 * tf.cast(Uin, tf.complex64) 
        
        for idx in range(self.n-1):
            ff = tf_ift2(self.Q2[idx] * tf_ft2(Uin / tf.complex(self.m[idx], 0.0), self.delta[idx]), self.deltaf[idx])
            Uin = tf.math.multiply(tf.complex(self.sg, 0.0), self.PS_arr[idx]) * ff
        Uout = self.Q3 * Uin
        return Uout, self.xn, self.yn