# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

#%%
import missingno as msno
import pandas as pd
# Importa as bibliotecas de Machine Learning
import sklearn.ensemble
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn import preprocessing

# Numpy by Pandas
np = pd.np

#%%
# Lê arquivo CSV de Treino
dataFrame=pd.read_csv('train.csv')

# =============================================================================
# 
# #Copiar nomes de colunas e tipos para clipboard
# columnTypes=["%s\t%s" % (c, t) for c,t in zip(dataFrame.columns, dataFrame.dtypes)]
# clipboard.copy(("\n".join(columnTypes)))
# 
# =============================================================================

# Remover coluna Id (nao utiliza na análise)
dataFrame = dataFrame.drop("Id", 1)

#%%
# =============================================================================
# Ajustar colunas QUALITATIVAS: nominais/categóricas ou ordinais
# =============================================================================
# Lê arquivo CSV com as variáveis e tipos
varDataFrame = pd.read_csv("variables.csv", delimiter=";")

# Separa as variáveis nominais/categóricas
nominalColumns=np.logical_and(varDataFrame.Categorical, np.logical_not(varDataFrame.Ordered))
#col = varDataFrame.Name[nominalColumns][0]

# Nominais/categóricas
for col in varDataFrame.Name[nominalColumns]:
    dataFrame[col] = pd.Categorical(dataFrame[col], ordered=False)


# Ordinais
# Busca as categorias ordinais não-nulas
ordinalColumns=pd.notnull(varDataFrame.Categories)
# Transforma as categorias ordinais de formato String para Objeto
varDataFrame.Categories = varDataFrame.Categories[ordinalColumns].map(eval)
# Ajusta as categorias ordinais para o tipo de dados "Categoria"
for col, cats in varDataFrame[ordinalColumns][["Name", "Categories"]].values:
    dataFrame[col] = pd.Categorical(dataFrame[col], categories=cats, ordered=True)
# Renomeia as categorias ordinais utilizando sequência de números
for col, cats in varDataFrame[ordinalColumns][["Name", "Categories"]].values:
    dataFrame[col] = dataFrame[col].cat.rename_categories(range(dataFrame[col].cat.categories.size))

#%%
# =============================================================================
# Checar dados faltantes em modo gráfico
# =============================================================================
msno.matrix(dataFrame, labels=True, color=(0.5,0.5,1), sparkline=False)

#%%
# Verificar graficamente valor de venda da casa com e sem piscina
fig1, ax1 = plt.subplots()
ax1 = fig1.add_subplot(1, 1, 1)
# ax1.set_facecolor("#1E1E1E")
bplot = plt.boxplot([dataFrame["SalePrice"][dataFrame.PoolArea>0], 
            dataFrame["SalePrice"][dataFrame.PoolArea==0]], 
            labels=["Com Piscina", "Sem Piscina"],
            patch_artist=True)  # fill with color
[patch.set_facecolor("#6666FF") for patch in bplot['boxes']]
plt.show()

#%%
# Ignora as 23 variáveis nominais/categóricas
tudo=np.logical_not(nominalColumns)


# Cria novo conjunto de dados sem as 23 variáveis nominais/categóricas
dataFrame2 = dataFrame[dataFrame.columns.values[tudo.values.astype(bool)]]

# Preenche as células nan com 0
dataFrame2=dataFrame2.fillna(0)

# Cria um list com as 57 variáveis restantes
cols = list(dataFrame2.columns)

# Seta a variável a ser predita
colPredict = "SalePrice"

# Remove a variável a ser predita
cols.remove(colPredict)


#%%
# Realiza a regressão com o algoritmo floresta aleatória e plota os valores encontrados x esperados
rf = sklearn.ensemble.RandomForestRegressor()
rf.fit(dataFrame2[cols], dataFrame2[colPredict])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlabel("predict")
plt.ylabel("observed")
plt.plot(rf.predict(dataFrame2[cols]),dataFrame2[colPredict], 'ro')
# ax.set_facecolor("#1E1E1E")
plt.show()

#%%
# Plota as variáveis mais importantes da regressão em ordem decrescente
dados=pd.DataFrame({"coluna":dataFrame2.columns[:-1], "importancia": rf.feature_importances_})
dados=dados.sort_values(by="importancia", ascending=False)
plt.xticks(rotation='vertical')
plt.rcParams["figure.figsize"] = (15,3)
plt.plot(dados.iloc[:,0], dados.iloc[:,1])
plt.show()

#%%
# Histograma de Preço de venda
plt.xlabel("Preço de Venda (US$)")
plt.ylabel("Quantidade")
plt.axes().set_xlim(0, 500000)
plt.hist(dataFrame2["SalePrice"], bins=40)
plt.show()

#%%
# Histogramas de preço de venda por bairro
figure(num=None, 
        figsize=(15, 10), 
        dpi=80, edgecolor='k')
bairros = dataFrame["Neighborhood"]
precoDeVenda = dataFrame["SalePrice"]
for i, bairro in enumerate(
            np.unique(bairros)):
    axs = plt.subplot(5, 5, i+1)
    axs.set_xlim(0, 500000)
    axs.set_ylim(0, 45)
    plt.xticks([0, 250000, 500000])
    plt.hist(
        precoDeVenda[bairros == bairro], 
        bins='auto', alpha=0.5, label='yes')
    plt.title(bairro)   
plt.silent_list
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

#%%
diff = abs(rf.predict(dataFrame2[cols])-dataFrame2[colPredict])
dataFrame2[colPredict].iloc[523]
rf.predict(dataFrame2[cols])[523]

#%%
diff = diff.sort_values(ascending = False)

#%%
from sklearn.preprocessing import OneHotEncoder
coder = OneHotEncoder(handle_unknown="ignore")

#%%
#Explorando a variavel mais importante (overall quality)
fig1, ax1 = plt.subplots()
ax1 = fig1.add_subplot(1, 1, 1)
# ax1.set_facecolor("#1E1E1E")
plt.axes().get_yaxis().set_visible(False)
ax1.get_yaxis().set_visible(True)
plt.xlabel("Qualidade Geral")
plt.ylabel("Preço de Venda (US$)")
ax1.get_xaxis().set_ticks(range(1,11))
ax1.violinplot(dataFrame.groupby("OverallQual")["SalePrice"].apply(list))
plt.show()

#%%
