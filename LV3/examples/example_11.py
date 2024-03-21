import pandas as pd 

data=pd.read_csv('data_C02_emission.csv')

print(data.corr(numeric_only=True))