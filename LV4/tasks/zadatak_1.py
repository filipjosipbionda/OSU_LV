import pandas as pd
import numpy as np
from sklearn . model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

#a
data=pd.read_csv('data_C02_emission.csv')

input_variables=[
    'Engine Size (L)',
    'Cylinders',
    'Fuel Consumption City (L/100km)',
    'CO2 Emissions (g/km)'
]

output='CO2 Emissions (g/km)'

X=data[input_variables]
y=data[output]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#b

plt.scatter(X_train['Cylinders'],y_train,c='Red')
plt.scatter(X_test['Cylinders'],y_test,c='Blue')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Emissions compared to engine size')
plt.show()


#c
sc=MinMaxScaler()

X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)

plt.hist(X_train['Cylinders'])
plt.show()

plt.hist(X_train_n[:,1])
plt.show()

#d
linearModel=lm.LinearRegression()
linearModel.fit(X_train_n,y_train)
print("Model parameters:")
print("Coefficient:", linearModel.coef_)
print("Intercept:", linearModel.intercept_)

#e
y_test_p = linearModel.predict(X_test_n)
plt.scatter(y_test,y_test_p)
plt.title('Comparision between real values and predicted values')
plt.xlabel('real values')
plt.ylabel('predicted values')
plt.show()

#f
MSE=mean_squared_error(y_test,y_test_p)

RMSE=mean_squared_error(y_test,y_test_p,squared=False)

MAE=mean_absolute_error(y_test,y_test_p)

MAPE=mean_absolute_percentage_error(y_test,y_test_p)

R_2=r2_score(y_test,y_test_p)

print(f"MAE: {MAE}, MSE: {MSE}, MAPE: {MAPE}, RMSE: {RMSE}, R2 SCORE: {R_2}")
