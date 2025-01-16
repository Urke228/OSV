import matplotlib.pyplot as plt
import numpy as np

from OSV_lib  import displayImage, loadImage, computeHistogram, displayHistogram, equalizeHistogram

if __name__ == '__main__':
    
    # Create gray image and all the histograms.

    image_2_gray = loadImage('C:/Users/ASUS/Desktop/OSV/vaja2/data/valley-1024x683-08bit.raw', [1024, 683], np.uint8)
    figure = displayImage(image_2_gray, 'Gray')
    
    hist, prob, CDF, levels = computeHistogram(image_2_gray)
    displayHistogram(hist, levels, "Histogram")
    displayHistogram(prob, levels, "Normaliziran histogram")
    displayHistogram(CDF, levels, "CDF histogram")
    # plt.show()


    # Equalize the gray image and show all the histograms

    image_equalized = equalizeHistogram(image_2_gray)
    figure = displayImage(image_equalized, 'Equalized Image')
    
    hist, prob, CDF, levels = computeHistogram(image_equalized)
    #displayImage(image_equalized, "Slika z izravnanim histogramom")
    displayHistogram(hist, levels, "Izravnan histogram")
    displayHistogram(prob, levels, "Normaliziran izravnan histogram")
    displayHistogram(CDF, levels, "CDF izravnanega histogram")
    plt.show()