import numpy as np
import matplotlib.pyplot as plt
import os,sys
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from OSV_lib import loadImage, displayImage, computeHistogram, displayHistogram


def interpolateImage(iImage, iSize, iOrder):
    iOrder = int(iOrder)
    Y, X = iImage.shape

    M, N = iSize

    oImage = np.zeros((N, M), dtype = iImage.dtype)
    
    dx = (X -1) / (M - 1)
    dy = (Y -1) / (N -1)

    for n in range(N):
        for m in range(M):
            s = 0

            pt = np.array((m * dx, n * dy))

            # 0. order interpolacije
            if iOrder == 0:
                # najdi najbližjega soseda
                px = np.round(pt).astype(np.uint16)
                s = iImage[px[1], px[0]]

            if iOrder == 1:
                px = np.floor(pt).astype(np.uint16)

                # pomikamo se po koordinatah okoli točke na sredini, ki bo novi pixl
                # kalkuliramo uteži okoliških pixlov kot površine kvadratov
                a = abs(pt[0] - (px[0] + 1) * abs(pt[1]) - (px[1] + 1))
                b = abs(pt[0] - (px[0] + 0) * abs(pt[1]) - (px[1] + 1))
                c = abs(pt[0] - (px[0] + 1) * abs(pt[1]) - (px[1] + 0))
                d = abs(pt[0] - (px[0] + 0) * abs(pt[1]) - (px[1] + 0))

                # sivinska vrednost pixlov okoli
                sa = iImage[px[1] + 0, px[0] + 0]
                sb = iImage[px[1] + 0, min(px[0] + 1, X - 1)]
                sc = iImage[min(px[1] + 1, Y - 1), px[0] + 0]
                sd = iImage[min(px[1] + 1, Y - 1), min(px[0] + 1, X - 1)]

                s = int(a * sa + b * sb + c * sc + d * sd)
                
            oImage[n, m] = s
    return oImage


if __name__ == "__main__":
    # plt.show() to display original image when run in terminal
    I = loadImage("C:/Users/ASUS/Desktop/OSV/vaja3/pumpkin-200x152-08bit.raw", (200, 152), np.uint8)
    displayImage(I, "originalna slika")
    plt.show()

    intSize = [I.shape[1] * 2, I.shape[0] * 2]

    interpolated_0_order = interpolateImage(I, intSize, 0)
    displayImage(interpolated_0_order, "Interpolirana slika 0. reda")
    plt.show()

    interpolated_0_order = interpolateImage(I, intSize, 1)
    displayImage(interpolated_0_order, "Interpolirana slika 1. reda")
    plt.show()