# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
import clipboard
from matplotlib import pyplot as plt

np = pd.np
dataFrame=pd.read_csv('train.csv')

# =============================================================================
# 
# #Copiar nomes de colunas e tipos para clipboard
# columnTypes=["%s\t%s" % (c, t) for c,t in zip(dataFrame.columns, dataFrame.dtypes)]
# clipboard.copy(("\n".join(columnTypes)))
# 
# =============================================================================

# Remover coluna Id (nao utiliza na analise)
dataFrame = dataFrame.drop("Id", 1)


# =============================================================================
# Classificar colunas como CATEGORICAS: nominais ou ordinais
# =============================================================================
varDataFrame = pd.read_csv("variables.csv", delimiter=";")
nominalColumns=np.logical_and(varDataFrame.Categorical, np.logical_not(varDataFrame.Ordered))
col = varDataFrame.Name[nominalColumns][0]
varDataFrame.Name[nominalColumns]

# Nominais
for col in varDataFrame.Name[nominalColumns]:
    dataFrame[col] = pd.Categorical(dataFrame[col], ordered=False)

# Ordinais
ordinalColumns=pd.notnull(varDataFrame.Categories)
varDataFrame.Categories = varDataFrame.Categories[ordinalColumns].map(eval)
for col, cats in varDataFrame[ordinalColumns][["Name", "Categories"]].values:
    dataFrame[col] = pd.Categorical(dataFrame[col], categories=cats, ordered=True)
    
for col, cats in varDataFrame[ordinalColumns][["Name", "Categories"]].values:
    dataFrame[col] = dataFrame[col].cat.rename_categories(range(dataFrame[col].cat.categories.size))


# Checar dados faltantes
import missingno as msno
msno.matrix(dataFrame, labels=True, color=(0.5,0.5,1), sparkline=False)

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
plt.boxplot([dataFrame["SalePrice"][dataFrame.PoolArea>0], dataFrame["SalePrice"][dataFrame.PoolArea==0]], 
            labels=["Com Piscina", "Sem Piscina"])

import sklearn.ensemble
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

tudo=np.logical_not(nominalColumns)

dataFrame2 = dataFrame[dataFrame.columns.values[tudo.values.astype(bool)]]
dataFrame2=dataFrame2.fillna(0)
cols = list(dataFrame2.columns)
colPredict = "SalePrice"
cols.remove(colPredict)
rf = sklearn.ensemble.RandomForestRegressor()
rf.fit(dataFrame2[cols], dataFrame2[colPredict])
plt.xlabel("predict")
plt.ylabel("observed")
plt.plot(rf.predict(dataFrame2[cols]),dataFrame2[colPredict], 'ro')
plt.show()

dados=pd.DataFrame({"coluna":dataFrame2.columns[:-1], "importancia": rf.feature_importances_})
dados=dados.sort_values(by="importancia", ascending=False)
plt.xticks(rotation='vertical')
plt.rcParams["figure.figsize"] = (15,3)
plt.plot(dados.iloc[:,0], dados.iloc[:,1])
plt.show()