import pandas as pd
import numpy as np
from sklearn . model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

#a
data=pd.read_csv('data_C02_emission.csv')

#a
input_variables = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
output_variable = 'CO2 Emissions (g/km)'


X=data[input_variables]
y=data[output_variable]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#b
plt.figure()
plt.scatter(x=X_train['Engine Size (L)'],y=y_train,c='Red')
plt.scatter(x=X_test['Engine Size (L)'],y=y_test,c='Blue')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

#c
plt.hist(X_train['Engine Size (L)'])
sc=MinMaxScaler()
X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)
plt.hist
plt.hist(X_train_n[:,0])
plt.show()

#d
linearRegression_model=lm.LinearRegression()
linearRegression_model.fit(X_train_n,y_train)

parameters=linearRegression_model.coef_
print(parameters)


#e
y_test_p=linearRegression_model.predict(X_test)

plt.scatter(x=y_test,y=y_test_p)
plt.title("Real values compared to predicted values")
plt.xlabel("Real values")
plt.ylabel("Predicted values")
plt.show()

#f
MAE = mean_absolute_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
RMSE = mean_squared_error(y_test, y_test_p, squared=False)
R_TWO_SCORE = r2_score(y_test, y_test_p)

print(f"MAE: {MAE}, MSE: {MSE}, MAPE: {MAPE}, RMSE: {RMSE}, R2 SCORE: {R_TWO_SCORE}")