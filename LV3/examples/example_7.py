import pandas as pd 

data=pd.read_csv('data_C02_emission.csv')

print(data.isnull().sum())

data.dropna(axis=0)
data . dropna ( axis =1 )
data.drop_duplicates()

data=data.reset_index(drop=True)