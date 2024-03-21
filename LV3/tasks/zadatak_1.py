import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data=pd.read_csv('data_C02_emission.csv')


#a
print(len(data))
print(data.info())

print(f'Rows with missing values:{data.isnull().sum()}')
print(f"Duplicated values:{data.duplicated().sum()}")

data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Model']=data['Model'].astype('category')
data['Make']=data['Make'].astype('category')
data['Transmission']=data['Transmission'].astype('category')
data['Fuel Type']=data['Fuel Type'].astype('category')


#b
most_consuming=data.nlargest(3,'Fuel Consumption City (L/100km)')
least_consuming=data.nsmallest(3,'Fuel Consumption City (L/100km)')

print('Most consuming:\n')
print(most_consuming[['Make','Model','Fuel Consumption City (L/100km)']])

print('Least consuming:\n')
print(least_consuming[['Make','Model','Fuel Consumption City (L/100km)']])

#c
targeted_data=data[(data['Engine Size (L)']>=2.5)&(data['Engine Size (L)']<=3.5)]
print(f'Number of cars that have size of motor between 2.5 and 3.5 L: {len(targeted_data)}')

print(f'Avetage C02 emission for this targeted vehicles is:{targeted_data['CO2 Emissions (g/km)'].mean()}')

#d 
audi_cars=data[data['Make']=='Audi']
print(len(audi_cars))

audi_cars_with_4_cylinder=audi_cars[audi_cars['Cylinders']==4]
print(f'Average C02 emissions for cars which manufacturer is Audi and which have 4 Cylinders are:{audi_cars_with_4_cylinder['CO2 Emissions (g/km)'].mean()}')

#e

cylinder_counts = data['Cylinders'].value_counts().sort_index()
print(cylinder_counts)

average_cylinder_emissions = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print(f'Average C02 emission based on number of Cylinders is: {average_cylinder_emissions}')

#f

diesel_cars=data[data['Fuel Type']=='D']
petrol_cars=data[data['Fuel Type']=='Z']

print(f'Average Fuel Consumption City (L/100km) for cars running on diesel is:{diesel_cars['Fuel Consumption City (L/100km)'].mean()}')
print(f'Average Fuel Consumption City (L/100km) for cars running on petrol is:{petrol_cars['Fuel Consumption City (L/100km)'].mean()}')

print(f'Medial Fuel Consumption City (L/100km) for cars running on diesel is:{diesel_cars['Fuel Consumption City (L/100km)'].median()}')
print(f'Medial Fuel Consumption City (L/100km) for cars running on petrol is:{petrol_cars['Fuel Consumption City (L/100km)'].median()}')

#g

four_cylinder_diesel_car=data[(data['Cylinders']==4)&(data['Fuel Type']=='D')]
print(f'4 Cylinder diesel car that has biggest Fuel Consumption City (L/100km) is:{four_cylinder_diesel_car.nlargest(1,'Fuel Consumption City (L/100km)')}')

#h
manual_cars=data[data['Transmission'].str[0]=='M']
print(f'Amount of cars with manual transmission is:{len(manual_cars)}')

#i
print(data.corr(numeric_only=True))








