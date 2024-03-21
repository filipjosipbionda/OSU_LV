import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

data=pd.read_csv('data_C02_emission.csv')

#a
plt.figure()
c02_emission=data['CO2 Emissions (g/km)'].plot(kind='hist',bins=20)
plt.show()


#b


colors={
    'Z':'brown',
    'X':'red',
    'E':'blue',
    'D':'black'
    }

data.plot.scatter(x='Fuel Consumption City (L/100km)',y='CO2 Emissions (g/km)',c=data['Fuel Type'].map(colors),s=50)
plt.show()

#c


data.boxplot(column='Fuel Consumption Hwy (L/100km)',by='Fuel Type')
plt.show()

#d

fuel_grouped_cars=data.groupby('Fuel Type')
fuel_grouped_cars.size().plot(kind='bar',xlabel='Fuel Type',ylabel='Number of cars',title='Cars by fuel type')
plt.show()

#e

cylinder_cars_grouped=data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print(cylinder_cars_grouped)
cylinder_cars_grouped.plot(kind='bar',x=cylinder_cars_grouped.index,y=cylinder_cars_grouped.values,xlabel='Cylinders', ylabel='CO2 emissions (g/km)', title='CO2 emissions by number of cylinders')
plt.show()


