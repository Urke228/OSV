import matplotlib.pyplot as plt
import numpy as np



def loadImage(iPath, iSize, iType):
    fid = open(iPath, 'rb')
    buffer = fid.read()
    buffer_len = len(np.frombuffer(buffer = buffer, dtype = iType))
    if buffer_len != np.prod(iSize):
        raise ValueError('Size of input does not match the specified size')
    else: 
        oImage_shape = (iSize[1], iSize[0])

    oImage = np.ndarray(oImage_shape, dtype = iType, buffer = buffer, order='F')
    return oImage



def displayImage(iImage, iTitle = '', iGridX = None, iGridY = None):
    
    
    fig = plt.figure()
    plt.title(iTitle)
    # vmin in vmax je oknjenje (če so večje bo ostalo 255, če manjše bo ostalo 0)
    
    if iGridX is not None and iGridY is not None:
        stepX = iGridX[1] - iGridX[0]
        stepY = iGridY[1] - iGridY[0]

        extent = (
            iGridX[0] - 0.5 * stepX,
            iGridX[-1] + 0.5 * stepX,
            iGridY[-1] + 0.5 * stepY,
            iGridY[0] - 0.5 * stepY
        )

    else:
        extent = (
            0 - 0.5,
            iImage.shape[1] - 0.5,
            iImage.shape[0] - 0.5,
            0 - 0.5
        )
    
    plt.imshow(iImage,
                cmap = 'gray', #cmap=plt.cm.gray
                vmin = 0,
                vmax = 255,
                aspect = 'equal',
                extent = extent #(0-0.5, iImage.shape[1]-0.5, iImage.shape[0]-0.5, 0-0.5)
    )
    return fig



def computeHistogram(iImage):
    mBits = (np.log2(iImage.max())) + 1

    oLevels = np.arange(0, 2 ** mBits, 1)

    iImage = iImage.astype(np.uint8)

    oHist = np.zeros(len(oLevels))

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            oHist[iImage[y, x]] = oHist[iImage[y, x]] + 1

    #Normaliziran Histogram
    oProb = oHist / iImage.size

    oCDF = np.zeros_like(oHist)
    for i in range(len(oProb)):
        oCDF[i] = oProb[: i + 1].sum()

    #Comulative distibution function
    return oHist, oProb, oCDF, oLevels



def displayHistogram(iHist, iLevels, iTitle):
    plt.figure()
    plt.title(iTitle)
    plt.bar(iLevels, iHist, width=1, edgecolor="darkred", color="red")
    plt.xlim((iLevels.min(), iLevels.max()))
    plt.ylim((0, 1.05 * iHist.max()))
    plt.show()



def equalizeHistogram(iImage):
    _, _, CDF, _ = computeHistogram(iImage)

    nBits = int(np.log2(iImage.max())) + 1

    max_intensity = 2 ** nBits + 1

    oImage = np.zeros_like(iImage)
    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            old_intensity = iImage[y, x]
            new_intensity = np.floor(CDF[old_intensity] * max_intensity)
            oImage[y, x] = new_intensity
    
    return oImage



def loadImage3D(iPath, iSize, iType):
    fid = open(iPath, "rb")
    im_shape = (iSize[1], iSize[0], iSize[2]) # Y, X, Z
    oImage = np.ndarray(shape = im_shape, dtype = iType, buffer = fid.read(), order = "F")
    fid.close()

    return oImage



def getPlanarCrossSection(iImage , iDim , iNormVec , iLoc):
    Y, X, Z = iImage.shape
    dx, dy, dz = iDim

    if iNormVec == [1, 0, 0]:
        oCS = iImage[:, iLoc, :].T #transponiranje matrike, ker je treba zamenjat dimenziji
        oH = np.arange(Y) * dy
        oV = np.arange(Z) * dz

    elif iNormVec == [0, 1, 0]:
        oCS = iImage[iLoc, :, :].T #transponiranje matrike, ker je treba zamenjat dimenziji
        oH = np.arange(X) * dx
        oV = np.arange(Z) * dz

    elif iNormVec == [0, 0, 1]:
        oCS = iImage[:, :, iLoc] #transponiranje matrike, ker je treba zamenjat dimenziji
        oH = np.arange(X) * dx
        oV = np.arange(Y) * dy

    return np.array(oCS) , oH , oV



def getPlanarProjection ( iImage , iDim , iNormVec , iFunc ) :
    Y, X, Z = iImage.shape
    dx, dy, dz = iDim

    if iNormVec == [1, 0, 0]:
        oP = iFunc(iImage, axis = 1).T
        oH = np.arange(Y) * dy
        oV = np.arange(Z) * dz

    elif iNormVec == [0, 1, 0]:
        oP = iFunc(iImage, axis = 0).T
        oH = np.arange(X) * dx
        oV = np.arange(Z) * dz
    
    elif iNormVec == [0, 0, 1]:
        oP = iFunc(iImage, axis = 2)
        oH = np.arange(X) * dx
        oV = np.arange(Y) * dy

    return oP , oH , oV



def scaleImage(iImage, a, b):
    oImage = np.array(iImage, dtype=float)
    oImage = iImage * a + b
    return oImage



def windowImage(iImage, iC, iW):
    ## inicializiramo izhodno sliko
    oImage = np.array(iImage, dtype = float)


    ## skaliramo vrednosti vhodne slike na skalo 0-255, ker želimo imeti bitno sliko. Nato premaknemo vrednosti vhodne slike na 0 (C - W/2)
    oImage = 255 / iW * (iImage - (iC - iW / 2))

    ## clipnemo sliko na range 0-255
    oImage[iImage < iC - iW / 2] = 0
    oImage[iImage > iC + iW / 2] = 255

    return oImage



def sectionalScaleImage(iImage, iS, oS):

    oImage = np.array(iImage, dtype = float)

    for i in range(len(iS) - 1):
        sL = iS[i]
        sH = iS[i + 1]

        # array z isto veliko mrežo pixlov kot slika, ki ima 1 na tistih mestih ki nas zanima in 0 na mestih ki nas ne zanima (temu lahko rečemo tudi maska)
        idx = np.logical_and(iImage >= sL, iImage <= sH)

        # scale factor (faktor skale = skala vhodne slike / skala izhodne slike)
        k = (oS[i + 1] - oS[i]) / (sH - sL)

        oImage[idx] = k * (iImage[idx] - sL) + oS[i]

    return oImage



def gammaImage(iImage, gama):
    
    oImage = np.array(iImage, dtype = float)
    oImage = 255**(1 - gama) * (iImage ** gama)
    
    return oImage



def getRadialValues(iXY, iCP):
    # št. kontrolnih tock
    K = iCP.shape[0]

    # instanciranje izhodnih radialnih uteži
    oValue = np.zeros(K)
    x_i, y_i = iXY
    
    for k in range(K):
        x_k, y_k = iCP[k]
        # razdalja vhodne tocke do k-te kontrolne točke
        r = np.sqrt((x_i - x_k) ** 2 + (y_i - y_k) ** 2)

        # apliciranje radialne funkcije na r
        if r > 0:
            oValue[k] = -(r ** 2) * np.log(r)

    return oValue



def getParameters(iType, scale=None, trans=None, rot=None, shear=None, orig_pts=None, mapped_pts=None):

    # default values
    oP = {}

    if iType == "affine":
        if scale is None:
            scale = [1, 1]
        if trans is None:
            trans = [0, 0]
        if rot is None:
            rot = 0
        if shear is None:
            shear = [0, 0]

        # skaliranje 
        Tk = np.array([
            [scale[0], 0, 0],
            [0, scale[1], 0],
            [0, 0, 1]
        ])

        # translacija
        Tt = np.array([
            [1, 0, trans[0]],
            [0, 1, trans[1]],
            [0, 0, 1]
        ])

        # rotacija
        phi = rot * np.pi / 180
        Tf = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])

        # strizna deformacija
        Tg = np.array([
            [1, shear[0], 0],
            [shear[1], 1, 0],
            [0, 0, 1]
        ])

        # mnozenje matrik med seboj, @ == np.dot (aka. skalarni produkt)
        oP = Tg @ Tf @ Tt @ Tk

    elif iType == "radial":
        assert orig_pts is not None, "manjka orig_pts"
        assert mapped_pts is not None, "manjka mapped_pts"

        K = orig_pts.shape[0]

        UU = np.zeros((K, K), dtype = float)
        
        for i in range(K):
            UU[i, :] = getRadialValues(orig_pts[i, :], orig_pts)

        oP["alphas"] = np.linalg.solve(UU, mapped_pts[:, 0])
        oP["betas"] = np.linalg.solve(UU, mapped_pts[:, 1])
        oP["pts"] = orig_pts

    return oP



def transformImage(iType, iImage, iDim, iP, iBgr = 0, iInterp = 0):
    Y, X = iImage.shape
    dx, dy = iDim

    oImage = np.ones((Y, X)) * iBgr

    for y in range(Y):
        for x in range(X):
            x_hat, y_hat = x * dx, y * dy
            
            if iType == "affine":
                x_hat, y_hat, _ = iP @ np.array([x_hat, y_hat, 1])
            
            if iType == "radial":
                U = getRadialValues([x_hat, y_hat], iP["pts"])
                x_hat = U @ iP["alphas"]
                y_hat = U @ iP["betas"]

            x_hat, y_hat = x_hat/dx, y_hat/dy

            if iInterp == 0:
                x_hat, y_hat = round(x_hat), round(y_hat)
                
                if 0 <= x_hat < X and 0 <= y_hat < Y:
                    oImage[y, x] = iImage[y_hat, x_hat]

    return oImage


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