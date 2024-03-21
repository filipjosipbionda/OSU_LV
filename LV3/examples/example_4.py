import pandas as pd 

data=pd.read_csv('data_C02_emission.csv')

print(len(data))
print(data)

print(data.head(5))

print(data.tail(3))

print(data.info())


print(data.describe())


print(data.max())
print(data.min())