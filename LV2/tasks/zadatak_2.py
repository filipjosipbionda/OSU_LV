import numpy as np 
import matplotlib.pyplot as plt

data=np.loadtxt('data.csv',delimiter=",",dtype="str")
data=data[1::]
data=np.array(data,np.float64)
print(data)

print(f'Measurements were made on {len(data)} people ')#a

height=data[:,1]
weight=data[:,2]

mean = height.mean()
max = height.max()
min = height.min()
print(f"Max: {max}, min: {min}, mean: {mean}")

plt.scatter(height,weight,)
plt.show()

height=data[0::50,1]
weight=data[0::50,2]

plt.scatter(height,weight)
plt.show()

men=data[data[:,0]==1]
women=data[data[:,0]==0]

print(f"Stats for men: \nmax height: {np.max(men[:,1])}\nmin height: {np.min(men[:,1])}\naverage height: {np.mean(men[:,1])}")
print(f"Stats for women: \nmax height: {np.max(women[:,1])}\nmin height: {np.min(women[:,1])}\naverage height: {np.mean(women[:,1])}")












