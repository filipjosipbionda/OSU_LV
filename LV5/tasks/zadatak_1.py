import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,precision_score,accuracy_score,recall_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#skaliranje moze i ne mora, nece odmoci 
sc=StandardScaler()
X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)

#a
plt.scatter(x=X_train_n[:,0],y=X_train_n[:,1],c='red')
plt.scatter(x=X_test_n[:,0],y=X_test_n[:,1],c='blue')
plt.show()

#b
logisticRegression_model=LogisticRegression()
logisticRegression_model.fit(X_train_n,y_train)

#c 
theta0=logisticRegression_model.intercept_
theta1,theta2=logisticRegression_model.coef_.T
print(theta0)
print(theta1)
print(theta2)

#x2=ax1+b

x1_values=np.linspace(np.min(X_train_n[:,0]),np.max(X_train_n[:,1]),100)
a=-theta1/theta2
b=-theta0/theta0
x2_values=a*x1_values+b
plt.scatter(x=X_train_n[:,0],y=X_train_n[:,1],c='red')
plt.plot(x1_values,x2_values,c='green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#d
y_predict=logisticRegression_model.predict(X_test_n)
disp=ConfusionMatrixDisplay(confusion_matrix(y_test,y_predict))
disp.plot()
plt.show()

print("Accuracy:",accuracy_score(y_test,y_predict))
print("Precision:",precision_score(y_test,y_predict))
print("Recall:",recall_score(y_test,y_predict))

#e
correcty_classified=X_test[y_predict==y_test]
misclassified=X_test[y_predict!=y_test]

plt.figure()
plt.scatter(x=correcty_classified[:,0],y=correcty_classified[:,1],c='green')
plt.scatter(x=misclassified[:,0],y=misclassified[:,1],c='black')
plt.show()