import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("woman_park_1.png")
print(img.shape)

# open CV has the channels in BRG order need to be swapped from proper display
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# print some small portion of the image, from the first channel
print(img[0:18,0:18,1])

img_gray = (0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2])/255

plt.imshow(img_gray, cmap='gray')
plt.show()
