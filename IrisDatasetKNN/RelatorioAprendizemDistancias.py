import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KDTree

# Importando dataset Iris
iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

error = []

#%% Calculando erro para K valores entre 1 e 40
for j in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=j)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

##
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(12, 8))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro x Valor K')
plt.xlabel('Valor K')
plt.ylabel('Erro')
plt.show()

import pandas as pd
def populateAccDataFrame(splitter):
    accDf = pd.DataFrame({"k":[],"acc":[]})
    for j in range(1,40):
        for train_index, test_index in splitter.split(X):
            knn = KNeighborsClassifier(n_neighbors=j)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            accDf.loc[accDf.size] = [j, np.mean(pred_i == y_test)]
    return accDf

import seaborn as sns
def plotSns(df, title, **kwargs):
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(x="k", y="acc", data=df, **kwargs)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Valor K", fontsize=14)
    ax.set_ylabel("Acurácia", fontsize=14)
    plt.savefig("temp.png", dpi=150, bbox_inches='tight')
    return ax



#%% Subamostragem aleatória (10)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
acc = []

ss = ShuffleSplit(n_splits=50, test_size=0.20, random_state=847123891)
repeatedKFold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=847123891)
loo = LeaveOneOut()

accDf = populateAccDataFrame(ss)
accDf2 = populateAccDataFrame(repeatedKFold)
accDf4 = populateAccDataFrame(loo)
accDf4 = accDf4.groupby("k").mean().reset_index()
accDf["split"] = "random"
accDf2["split"] = "repeatedKFold"
accDf4["split"] = "leaveOneOut"
accDf3=(accDf.append(accDf2))
accDf3 = accDf3.append(accDf4)
ax = plotSns(accDf3, "Acurácia x Valor K", hue="split", style="split")

for i in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    acc.append(np.mean(pred_i == y_test))

plt.figure(figsize=(12, 8))
plt.plot(range(1, 11), acc, color='red', linestyle='dashed', marker='*',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Acurácia x Subamostragem aleatória (K=5)')
plt.xlabel('Subamostragem aleatória')
plt.ylabel('Acurácia')
plt.show()

#%% Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scores = []

# Calculando acurácia para K valores entre 1 e 40
for j in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=j)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)

    # CV (10)
    score = cross_val_score(knn, X, y, cv=10)
    scores.append(score.mean())

plt.figure(figsize=(12, 8))
plt.plot(range(1, 41), scores, color='red', linestyle='dashed', marker='*',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Acurácia Média x Valor K (10-fold CV)')
plt.xlabel('Valor K')
plt.ylabel('Acurácia Média')
plt.show()

# Leave one out
loo = LeaveOneOut()
#print(loo.get_n_splits(X))

scores = []
for train_index, test_index in loo.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(X_train, X_test, y_train, y_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    scores.append(pred_i == y_test)

plt.figure(figsize=(12, 8))
plt.plot(range(1, len(X)+1), scores, color='red', linestyle='dashed', marker='*',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Acurácia x Split (Leave one out / K=5)')
plt.xlabel('Split')
plt.ylabel('Acurácia')
plt.show()

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

error = []
#%% Calculando erro para K valores entre 1 e 40 (Distância de Manhattan)
for j in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=j, p=1)  # p=1 (Distância de Manhattan)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 8))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro Médio x Valor K (Distância de Manhattan)')
plt.xlabel('Valor K')
plt.ylabel('Erro Médio')
plt.show()

error = []
#%% Calculando erro para K valores entre 1 e 40 (Similaridade do Cosseno)
for j in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=j, metric='cosine')  # (Similaridade do Cosseno)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 8))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro Médio x Valor K (Distância de Cosseno)')
plt.xlabel('Valor K')
plt.ylabel('Erro Médio')
plt.show()

error = []
#%% Calculando erro para K valores entre 1 e 40 (Distância de Minkowski Ponderada)
for j in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=j, metric='wminkowski', p=2, metric_params={'w': np.random.random(X_train.shape[1])})  # (Distância de Minkowski Ponderada)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 8))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro Médio x Valor K (Distância de Minkowski Ponderada)')
plt.xlabel('Valor K')
plt.ylabel('Erro Médio')
plt.show()

error = []
#%% Calculando erro para K valores entre 1 e 40 (Brute)
for j in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=j, algorithm='brute')
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 8))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro Médio x Valor K (Brute)')
plt.xlabel('Valor K')
plt.ylabel('Erro Médio')
plt.show()

error = []
#%% Calculando erro para K valores entre 1 e 40 (Ball Tree)
for j in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=j, algorithm='ball_tree')
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 8))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro Médio x Valor K (Ball Tree)')
plt.xlabel('Valor K')
plt.ylabel('Erro Médio')
plt.show()

error = []
#%% Calculando erro para K valores entre 1 e 40 (KD Tree)
for j in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=j, algorithm='kd_tree')
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 8))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro Médio x Valor K (KD Tree)')
plt.xlabel('Valor K')
plt.ylabel('Erro Médio')
plt.show()
