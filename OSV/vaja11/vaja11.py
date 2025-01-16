import numpy as np
import matplotlib.pyplot as plt
from OSV_lib import displayImage, loadImage, transformImage, getParameters, getRadialValues
import cv2

def loadFrame(iVideo, IFrame):
    iVideo.set(1, IFrame - 1)
    ret, oFrame = iVideo.read()
    oFrame = oFrame[:, :, 0].astype(float)
    return oFrame

if __name__ == "__main__":
    cap = cv2.VideoCapture("C:/Users/ASUS/Desktop/OSV/vaja11/simple-video.avi")
    print(f"Število frameov v videu: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    N = 100
    I1 = loadFrame(cap, N)
    I2 = loadFrame(cap, N + 1)
    displayImage(I1, "Slika 1")
    # plt.show()
    displayImage(I2, "Slika 2")
    # plt.show()

def framePrediction(iFrame, iMV):
    iMV = np.array(iMV).astype(int)
    dx, dy = iMV

    # krožno premaknemo vrstice in stolpce
    oFrame = np.roll(iFrame, [dy, dx], axis = (0, 1))

    if dx >= 0:
        oFrame[:, :dx] = -1
    else:
        oFrame[:, dx:] = -1

    if dy >= 0:
        oFrame[:dy, :] = -1
    else:
        oFrame[dy:, :] = -1

    return oFrame

def blockMatching(iFrame1, iFrame2, iSize, iSearchSIze):

    Y, X = iFrame1.shape
    dx, dy = iSize

    M = int(X / dx)
    N = int(Y / dy)

    oMF = np.zeros((N, M, 2), dtype = int)
    # OCP - output center point
    oCP = np.zeros((N, M,2), dtype = float)
    Err = np.ones((N, M), dtype = float) * 255

    P = (iSearchSIze - 1) / 2
    PTS = np.array([
        [0, 0], 
        [1, 0], [-1, 0],
        [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
        ])
    
    for n in range(N):
        y_min = n * dy
        y_max = (n + 1) * dy
        y = np.arange(y_min, y_max)

        for m in range(M):
            x_min = m * dx
            x_max = (m + 1) * dx
            x = np.arange(x_min, x_max)

            oCP[n, m, 0] = x.mean()
            oCP[n, m, 1] = y.mean()

            # trenuten blok na frameu 2
            B2 = iFrame2[y_min:y_max, x_min:x_max]

            for i in range(1, 4):
                # logaritemsko skaliranje premika
                Pi = (P + 1) / (2 ** i)
                PTSi = PTS * Pi

                # prvi kandidat za iskanje vektorja premika
                d0 = oMF[n, m, :]

                for p in range(PTSi.shape[0]):
                    d = d0 + PTSi[p, :]
                    # predicted frame 2
                    pF2 = framePrediction(iFrame1, d)
                    # predicted block 2
                    pB2 = pF2[y_min:y_max, x_min:x_max]

                    # maska za odstranit bloke, ki so izven meja
                    msk = pB2 >= 0

                    # trenutna napaka blokov
                    bErr = np.mean(np.abs(B2[msk] - pB2[msk]))
                    
                    # če je trenutna napaka nižja, je vektor boljši
                    # ga shranimo
                    if bErr < Err[n, m]:
                        Err[n, m] = bErr
                        oMF[n, m, :] = d
    return oMF, oCP



def displayMotionField(iMF, iCP, iTitle, iImage=None):
    if iImage is None:
        fig = plt.figure()
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal")
        plt.title(iTitle)
    else:
        fig = displayImage(iImage, iTitle)
    
    plt.quiver(
        iCP[:, :, 0],
        iCP[:, :, 1], 
        iMF[:, :, 0], 
        iMF[:, :, 1], 
        color="r",
        units="xy",
        scale=0.5,
        angles="xy"
    )
    plt.show()
    return fig

if __name__ == "__main__":
    MF, CP = blockMatching(I1, I2, [8, 8], 15)
    fig1 = displayMotionField(MF, CP, "Vektorji premika")
    fig2 = displayMotionField(MF, CP, "Slika z vektorjem premika", I1)