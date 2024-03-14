import numpy as np
import matplotlib.pyplot as plt

zeros = np.zeros((50, 50))
ones = np.ones((50, 50))

top = np.hstack((ones, zeros))
bottom = np.hstack((zeros, ones))
matrix = np.vstack((top, bottom))

plt.figure()
plt.imshow(matrix,cmap="binary")
plt.show()