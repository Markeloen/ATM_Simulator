from aotools.turbulence import infinitephasescreen
from matplotlib import pyplot
from imageio import *
import numpy as np

def testVKInitScreen():

    scrn = infinitephasescreen.PhaseScreenVonKarman(128, 4./64, 0.2, 50, n_columns=4)

def testVKAddRow():

    scrn = infinitephasescreen.PhaseScreenVonKarman(128, 4./64, 0.2, 50, n_columns=4)
    scrn.add_row()



# Test of Kolmogoroc screen
def testKInitScreen():

    scrn = infinitephasescreen.PhaseScreenKolmogorov(128, 4./64, 0.2, 50, stencil_length_factor=4)

def testKAddRow():

    screen = infinitephasescreen.PhaseScreenKolmogorov(128, 4./64, 0.2, 50, stencil_length_factor=4)
    screen.add_row()
    pyplot.ion()
    pyplot.imshow(screen.stencil)

    pyplot.figure()
    pyplot.imshow(screen.scrn)
    pyplot.colorbar()
    for i in range(1000):
        screen.add_row()

        pyplot.clf()
        pyplot.imshow(screen.scrn)
        pyplot.colorbar()
        pyplot.draw()
        pyplot.pause(0.01)
        # pyplot.savefig(f'img_{i}.png', transparent = False,  facecolor = 'white')
    # frames = []


    # for t in np.linspace(0, 25, 25):
    #     image = imageio.v2.imread(f'img_{t}.png')
    #     frames.append(image)

        


    # imageio.mimsave('./example.gif', # output gif
    #                 frames,          # array of input frames
    #                 fps = 5)

def main():
    testKAddRow()

if __name__ == "__main__":


    main()
    # from matplotlib import pyplot


    # screen = infinitephasescreen.PhaseScreenVonKarman(32, 8./16, 0.2, 40, 2)

    # pyplot.ion()
    # pyplot.imshow(screen.stencil)

    # pyplot.figure()
    # pyplot.imshow(screen.scrn)
    # pyplot.colorbar()
    # for i in range(100):
    #     screen.add_row()

    #     pyplot.clf()
    #     pyplot.imshow(screen.scrn)
    #     pyplot.colorbar()
    #     pyplot.draw()
    #     pyplot.pause(0.01)
