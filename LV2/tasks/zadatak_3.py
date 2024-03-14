import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("road.jpg")
plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()


plt.imshow(img,alpha=0.7)
plt.axis('off')
plt.show()

width=len(img[0])
quarter=int(width/4)
plt.imshow(img[:,1*quarter:2*quarter,:])
plt.axis('off')
plt.show()


plt.imshow(np.rot90(img,3))
plt.axis('off')
plt.show()


flipped_image = np.flip(img,1)
plt.imshow(flipped_image)
plt.show()












