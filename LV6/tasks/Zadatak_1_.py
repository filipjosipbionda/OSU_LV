import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

#6.5.1.1-6.5.1.2

def run_task(K):
    KNN_model=KNeighborsClassifier(n_neighbors=K)
    KNN_model.fit(X_train_n,y_train)

    y_train_p_KNN=KNN_model.predict(X_train_n)
    y_test_p_KNN=KNN_model.predict(X_test_n)

    print("KNN: ")
    print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
    print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

    plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
    plt.tight_layout()
    plt.show()


run_task(5)
run_task(1)#overfit
run_task(100)#underfit

#6.5.2
from sklearn.model_selection import GridSearchCV
model=KNeighborsClassifier()
scores=cross_val_score(model,X_train_n,y_train,cv=5)
print(scores)

param_grid={
    'n_neighbors':range(1,20)
}

grid_search=GridSearchCV(model,param_grid,cv=5,scoring="accuracy",n_jobs=-1)
grid_search.fit(X_train_n,y_train)
k=grid_search.best_params_['n_neighbors']
print(k)

model.set_params(n_neighbors=k)
model.fit(X_train_n,y_train)
y_train_p_KNN=model.predict(X_train_n)
y_test_p_KNN=model.predict(X_test_n)

print("KNN: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()




#6.5.3
SVM_model=svm.SVC(kernel='rbf',gamma=0.1,C=10)
SVM_model.fit(X_train_n,y_train)

y_train_p_SVM=SVM_model.predict(X_train_n)
y_test_p_SVM=SVM_model.predict(X_test_n)

print("SVM: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))

plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()

#kada sam promijenio gammu s 1 na 0.1 i C s 0.1 na 10, granica odluke mi se dosta zaoblila i dala je nekakav oblik priblizan elipsi koji se nalazi u donjem lijevom kutu 

#6.5.4
SVM_model=svm.SVC(kernel='rbf')

param_grid={
    'C':[10,100,100],
    'gamma':[10,1,0.1,0.01]
}
svm_gscv=GridSearchCV(SVM_model,param_grid,cv=5,scoring="accuracy",n_jobs=-1)
svm_gscv.fit(X_train_n,y_train)

C=svm_gscv.best_params_['C']
gamma=svm_gscv.best_params_['gamma']

print("Optimalna vrijednosti:")
print("C:",C)
print("gamma:",gamma)
print ( svm_gscv . best_score_ )

