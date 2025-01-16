import numpy as np
import matplotlib.pyplot as plt
from OSV_lib import displayImage, loadImage3D, getPlanarCrossSection, getPlanarProjection



if __name__ == "__main__":

    imSize = [512, 58, 907]
    pxDim = [0.597656, 3, 0.597656]
    I = loadImage3D(r"C:/Users/ASUS/Desktop/OSV/vaja4/spine-512x058x907-08bit.raw",
                    imSize,
                    np.uint8)
    # print(I.shape)
    # displayImage(I[:, 250, :], "prerez")

    # xc = 250 # št. slicea (0-512)
    # sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [1, 0, 0], xc)
    # displayImage(sagCS, "sagital crosssection", sagH, sagV)

    # xc = 30 # št. slicea (0-58)
    # sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [0, 1, 0], xc)
    # displayImage(sagCS, "coronal crosssection", sagH, sagV)

    # xc = 400 # št. slicea (0-907)
    # sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [0, 0, 1], xc)
    # displayImage(sagCS, "axial crosssection", sagH, sagV)

    func = np.max
    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [1, 0, 0], func)
    displayImage(sagP, "sagital projection", sagH, sagV)

    func = np.max
    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0, 1, 0], func)
    displayImage(sagP, "coronal projection", sagH, sagV)

    func = np.max
    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0, 0, 1], func)
    displayImage(sagP, "axil projection", sagH, sagV)
    plt.show()



