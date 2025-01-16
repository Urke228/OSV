import numpy as np
import matplotlib.pyplot as plt
from OSV_lib import displayImage, loadImage



def spatialFiltering(iType, iImage, iFilter, iStatFunc = None, iMorphOp = None):

    N, M = iFilter.shape
    
    # polovica velikosti vhodnega jedra: če imamo 3x3 jedro --> (3 - 1 / 2 = 1) 
    m = int((M - 1) / 2) 
    n = int((N - 1) / 2)

    # povečamo vhodno sliko za polovično velikost jedra v vse smeri
    iImage = changeSpatialDomain("enlarge", iImage, m, n)

    # inicializacija izhodne slike 
    Y, X = iImage.shape
    oImage = np.zeros((Y, X), dtype = float)

    # sprehajamo se cez vhodno sliko, stem da upoštevamo velikost jedra.
    # za polovično velikost jedra se premaknemo v sliko in tu začnemo.
    for y in range(n, Y - n):
        for x in range(m, X - m):
            # pridonimo odsek (patch) slike za določen slikovni element slike v velikosti vhodnega filtra
            patch = iImage[y - n : y + n + 1, x - m : x + m + 1]

            # patch filtriramo s poljubno metodo
            if iType == "kernel":             
                # zmnožimo patch in jedro, ter sestejemo vse vrednosti da dobimo novo sivinsko vrednosti slikovnega elementa    
                oImage[y, x] = (patch * iFilter).sum()

            if iType == "statistical":
                # Nad odsekom izvedemo statistično operacijo
                oImage[y, x] = iStatFunc(patch)

            if iType == "morphological":
                # upostevamo vrednsoti v odseku (patchu) kjer je vrednost jedra enaka 0
                R = patch[iFilter != 0]
                # da erodiramo, vzamemo najmanjso vrednost odseka
                if iMorphOp == "erosion":
                    oImage[y, x] = R.min()
                # da diliramo, vzamemo največjo vrednost odseka 
                if iMorphOp == "dilation":
                    oImage[y, x] = R.max()

    # iz slike izfiltriramo izhodno sliko
    oImage = changeSpatialDomain("reduce", oImage, m, n)
    return oImage



def changeSpatialDomain ( iType , iImage , iX , iY , iMode = None , iBgr =0) :
    Y, X = iImage.shape

    if iType == "enlarge":
        if iMode is None:
            # izhodno sliko razširimo na velikost vhodne slike + jedra
            oImage = np.zeros((Y + 2 * iY, X + 2 * iX))

            # v sredino vstavimo vhodno sliko
            oImage[iY : Y + iY, iX : X + iX] = iImage

    elif iType == "reduce":
        # iz sredine vzamemo originalno (filtrirano) sliko
        oImage = iImage[iY : Y - iY, iX : X - iX]

    return oImage






if __name__ == "__main__":
    I = loadImage("C:/Users/ASUS/Desktop/OSV/vaja7/cameraman-256x256-08bit.raw", [256, 256], np.uint8)
    #displayImage(I, "Originalna slika.")
    
    # poljubno nastavimo filter 
    Kernel = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
    ])
    KImage = spatialFiltering("kernel", I, iFilter = Kernel)
    #displayImage(KImage, "Filtrirana slika z laplaceovim filtrom.")
    

    SImage = spatialFiltering("statistical", I, iFilter = np.zeros((30, 30)), iStatFunc = np.median)
    displayImage(SImage, "Statistično filtrirana slika: Mediana.")


    MKernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ])

    MImage = spatialFiltering("morphological", I, iFilter = MKernel, iMorphOp = "dilation")
    #MImage = spatialFiltering("morphological", I, iFilter = MKernel, iMorphOp = "erosion")
    #displayImage(MImage, "Dilated Image.")
    #displayImage(MImage, "Erosioned Image.")

    PaddedImage = changeSpatialDomain("enlarge", I, 100, 100)
    #displayImage(PaddedImage, "Razširjena slika s 0 vrednostjo.")


    plt.show()


