import matplotlib.pyplot as plt
import numpy as np

from OSV_lib  import displayImage, loadImage

if __name__ == '__main__':


     image = plt.imread('C:/Users/ASUS/Desktop/OSV/vaja1/data/lena-color.png')

     plt.figure()
     plt.imshow(image)
     plt.show()
     plt.imsave("C:/Users/ASUS/Desktop/OSV/vaja1/data/lena-gray-color.jpg", image)

     image_2_gray = loadImage('C:/Users/ASUS/Desktop/OSV/vaja1/data/lena-color-512x410-08bit.raw', (512, 410, 3), np.uint8)
     figure = displayImage(image_2_gray, 'Lena Gray')
     plt.show()
