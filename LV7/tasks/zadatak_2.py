import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("images/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

km=KMeans(n_clusters=5,init="k-means++",n_init=5,random_state=0)

km.fit(img_array_aprox)
labels=km.predict(img_array_aprox)
print(labels)

centroids=km.cluster_centers_
print(centroids)

print(img_array_aprox)
print(img_array_aprox[:,0])
print(img_array_aprox[:,1])
print(img_array_aprox[:,2])


print(centroids[labels][:, 0])
print(centroids[labels][:, 1])
print(centroids[labels][:, 2])


img_array_aprox[:, 0] = centroids[labels][:, 0]
img_array_aprox[:, 1] = centroids[labels][:, 1]
img_array_aprox[:, 2] = centroids[labels][:, 2]


img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(img)
axarr[1].imshow(img_array_aprox)
plt.tight_layout()
plt.show()