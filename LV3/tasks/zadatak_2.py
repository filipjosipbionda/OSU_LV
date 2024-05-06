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

avg_C02_by_cylinders=data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()#grupira po cylindrima, zatim pristupa po vrijednostima cylindara svim vrijednostima za emisije CO2 koje se nalaze pod tim cilindrom i racuna njihovu srednju vrijednost
# i sprema to u polje srednjih vrijednosti 

avg_C02_by_cylinders.plot(kind='bar')

plt.title('Average CO2 Emissions by Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions (g/km)')
plt.xticks(rotation=0)
plt.show()


