# Convolution, filter gray level image with a blur filter. Plot original image, blurred image and row 100 from the original image and row 100 from blurred image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage

img = 1.0*plt.imread("einstein_gray.jpg")

fig = plt.figure(figsize=(10,7))
rows = 2
cols = 2

fig.add_subplot(rows,cols,1)
plt.imshow(img, cmap="gray")
plt.axis("off")

fig.add_subplot(rows,cols,2)
plt.plot(img[100,:])
plt.axis("off")

print(type(img))
print(img.shape)

filt = 1/9*np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
img_blurr = ndimage.convolve(img, filt, mode ='reflect');
fig.add_subplot(rows,cols,2)
plt.imshow(img_blurr, cmap="gray")
plt.axis("off")



fig.add_subplot(rows,cols,4)
plt.plot(img_blurr[100,:])
plt.axis("off")
plt.show()


