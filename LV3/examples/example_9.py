import pandas as pd
import matplotlib . pyplot as plt
data = pd . read_csv ('data_C02_emission.csv')
grouped = data . groupby ('Cylinders')
grouped.boxplot ( column =['CO2 Emissions (g/km)'])
data . boxplot (column =['CO2 Emissions (g/km)'], by='Cylinders')
plt . show ()