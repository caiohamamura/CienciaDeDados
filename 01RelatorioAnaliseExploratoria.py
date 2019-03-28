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


# Importa dataFrame e variaveis pré processadas
from AnaliseInicial.TratamentoDados import (
    np,             #numpy
    dataFrame,      #dataFrame principal
    varsDataFrame   #dataFrame das variaveis
)


#%%
# =============================================================================
# Checar dados faltantes em modo gráfico
# =============================================================================
msno.matrix(dataFrame, labels=True, color=(0.5,0.5,1), sparkline=False)


#%%
# Ignora as 23 variáveis nominais/categóricas
nominalColumns = varsDataFrame.Ordered == False
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
# Estatísticas descritivas para variáveis exporta para csv
from scipy import stats
numericalCols = varsDataFrame.Name[pd.isna(varsDataFrame.Categorical)]
dataNumerical = dataFrame[numericalCols]
res=dataNumerical.apply(
    func=[np.mean, np.median, np.max, np.min, np.var], 
    axis=0)
res2 = stats.kurtosis(dataNumerical, axis=0, nan_policy="omit")
res3 = stats.skew(dataNumerical, axis=0, nan_policy="omit")
res4 = stats.mode(dataNumerical, axis=0, nan_policy="omit").mode
res5 = pd.DataFrame({
    "kurtosis":res2, "skew":res3, "mode": res4[0]
    })
res5.index = res.columns
res6 = res.transpose().join(res5)
res7 = res6[
    ["mean", "median", "mode", "amin", 
    "amax", "var", "skew", "kurtosis"]]
# res7.to_csv("descriptiveStats.csv", sep=";", decimal =",") # Salva para CSV
pd.options.display.float_format = '{:.2f}'.format

res7


#%%
import seaborn as sns
corr = dataNumerical.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig, ax = plt.subplots(figsize=(15,7))
ax.set_title("Correlograma das Variáveis Quantitativas")
sns.heatmap(corr, ax=ax, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


#%%
dataNumerical.cov()
cov = dataNumerical.cov()
mask = np.zeros_like(cov, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

np.percentile(cov.values.flatten(), 90)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig, ax = plt.subplots(figsize=(15,7))
ax.set_title("Mapa de Calor de Covariâncias")
ax = sns.heatmap(cov, ax=ax, cmap=cmap, vmin=-131, vmax=24000, center=12000,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([-131, 12000, 24000])
colorbar.set_ticklabels
plt.show()


#%%
