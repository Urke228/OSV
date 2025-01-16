import numpy as np
import matplotlib.pyplot as plt
from OSV_lib import displayImage, loadImage, computeHistogram, windowImage, scaleImage, sectionalScaleImage, gammaImage

# Dodatno: Naloga 2
def thresholdImage(iImage, iT):
    Lg = 2 ** 8
    oImage = np.array(iImage, dtype=float)

    for i in range(iImage.shape[0]):
        for j in range(iImage.shape[1]):
            if iImage[i, j] <= iT:
                oImage[i, j] = 0
            else:
                oImage[i, j] = Lg - 1
        
    return oImage

if __name__ == "__main__":

## NAL 1: Naloži in prikaži sliko
    
    I = loadImage("C:/Users/ASUS/Desktop/OSV/vaja5/image-512x512-16bit.raw", [512, 512], np.int16)
    # displayImage(I, "Originalna slika")
#    print(f"Sivinske vrednosti originalne slike:\tmin={I.min()}, max={I.max()}") ## max in min sivinska vrednost v sliki 
    

## NAL 2: Linearna sivinska preslikava
    
    sImage = scaleImage(I, -0.125, 256)
    # displayImage(sImage, "Skalirana slika")
#    print(print(f"Sivinske vrednosti skalirane slike:\tmin={sImage.min()}, max={sImage.max()}"))
    

## NAL 3: Windowing/Oknjenje - Prikažemo le sivinske vrednosti znotraj določenega okna in te vrednosti skaliramo čez celo območje
    
    wImage = windowImage(sImage, 1000, 500)
#  displayImage(wImage, "Oknjena slika")
   

## NAL 4: Odsekoma linearna preslikava 

    sCP = np.array([[0, 85], [85, 0], [170, 255], [255, 170]])
    ssImage = sectionalScaleImage(wImage, sCP[:, 0], sCP[:, 1])
#   displayImage(ssImage, "Slika po odsekoma linearni preslikavi.")
    

## NAL 5: Gama preslikava

    gImage = gammaImage(wImage, 5)
#    displayImage(gImage, "Slika po gama preslikavi")


    tImmage = thresholdImage(wImage, 200)
    displayImage(tImmage, "Slika po upragovanju")
    plt.show()




