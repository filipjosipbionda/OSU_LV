import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,classification_report

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

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
                    edgecolor = 'w',
                    label=labels[cl])
        
    plt.show()

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()[:,0]

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#skaliranje moze i ne mora, nece odmoci 

sc=StandardScaler()
X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)

#a

train_uniques,train_counts=np.unique(X_train_n,return_counts=True)
test_uniques,test_counts=np.unique(X_test_n,return_counts=True)

plt.figure()
plt.bar(train_uniques,train_counts,color='Red',label='Train data')
plt.bar(test_uniques,test_counts,color='Blue',label='Test data')
plt.legend()
plt.show()


#b
logisticRegression_model=LogisticRegression()
logisticRegression_model.fit(X_train_n,y_train)


#c
intercept=logisticRegression_model.intercept_

coeficients=logisticRegression_model.coef_.T

print(intercept)
print(coeficients)

#razlika je ta sto imamo vise vrijednosti za intercept i u samim koeficijentima imamo array od 3 vrijednosti jer imamo 
#3 izlazne velicine 


#d
plot_decision_regions(X_train_n,y_train,logisticRegression_model)

#podatci su dosta precizno odredeni 

#e
y_predict=logisticRegression_model.predict(X_test_n)

disp=ConfusionMatrixDisplay(confusion_matrix(y_test,y_predict))
disp.plot()
plt.show()

print("Accuracy:",accuracy_score(y_test,y_predict))

print(classification_report(y_test,y_predict))

# #24 puta kad je bilo 0 je predicto 0
# 1 put kad je bilo 0 je predicto 1
# 2 put kad je bila 0 je predicto 2
# Itd