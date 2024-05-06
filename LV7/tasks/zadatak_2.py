import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("images\\test_5.jpg")

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

#1
unique_colors=np.unique(img_array_aprox,axis=0)
print("Broj razlicitih boja:",len(unique_colors))

#2
km=KMeans(n_clusters=2,n_init=5,init='random',random_state=0)
km.fit(img_array_aprox)
cluster_centers=km.cluster_centers_
labels=km.predict(img_array_aprox)

#3
img_array_aprox=cluster_centers[labels]
img_array_aprox=np.reshape(img_array_aprox,(w,h,d))

#4
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_array_aprox)
plt.show()

#na drugoj slici koju grade podatci koji su prosli kroz algoritam, vidimo manje nijansi boja. Najvise to vidimo po plavoj boji, Kada stavimo broj 2 vidimo jednu nijasnu plave i zute.

#5
#mozemo vidjeti da kada stavimo K=2 samo budu dvije nijanse boja na rezultantnoj slici 

#6

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="random",random_state=0,n_init=5)
    kmeans.fit(img_array.copy())
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

img_array_aprox=img_array.copy()
km=KMeans(n_clusters=4,n_init=5,init='random',random_state=0)
km.fit(img_array_aprox)
cluster_centers=km.cluster_centers_
labels=km.predict(img_array_aprox)

img_array_aprox=cluster_centers[labels]
img_array_aprox=np.reshape(img_array_aprox,(w,h,d))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_array_aprox)
plt.show()


#za sliku test_5 na kojoj sam radio testiranje najbolje vrijednosti K, mozemo vidjeti kako je  K=4 najbolja jer imamo 4 nijanse boja koje se pojavljuju u vise nijansi i slika izgleda nepromijenjeno

#7
labels = km.labels_

for cluster_id in range(2):  
    cluster_mask = labels.reshape(w, h) == cluster_id

    binary_image = np.zeros((w, h), dtype=np.uint8)
    binary_image[cluster_mask] = 255 

    plt.figure()
    plt.title("Binary Image for Cluster {}".format(cluster_id))
    plt.imshow(binary_image, cmap='gray')  
    plt.tight_layout()
    plt.show()
