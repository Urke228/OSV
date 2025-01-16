import numpy as np
import matplotlib.pyplot as plt
from OSV_lib import displayImage, loadImage, transformImage, getParameters, getRadialValues


if __name__ == "__main__":
    # Naloži in prikaži originalno sliko
    imSize = [256, 512]
    pxDim = [2, 1]

    gX = np.arange(imSize[0]) * pxDim[0]
    gY = np.arange(imSize[1]) * pxDim[1]
    I = loadImage("C:/Users/ASUS/Desktop/OSV/vaja6/lena-256x512-08bit.raw", imSize, np.uint8)
    # displayImage(I, "Originalna slika", gX, gY) 
    
    T = getParameters("affine", rot=30)
    print(T)
    bgr = 63
    tImage = transformImage("affine", I, pxDim, np.linalg.inv(T), iBgr = bgr)
    # displayImage(tImage, "Affina preslikava", gX, gY)
    
    xy = np.array([[0, 0], [511, 0], [0, 511], [511, 511]])
    uv = np.array([[0, 0], [511, 0], [0, 511], [255, 255]])
    P = getParameters("radial", orig_pts = xy, mapped_pts = uv)
    print(P)

    rImage = transformImage("radial", I, pxDim, P, iBgr = bgr)
    # displayImage(rImage, "Radialna preslikava", gX, gY)
    # plt.show()


