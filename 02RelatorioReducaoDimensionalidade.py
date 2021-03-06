import pandas as pd
import sklearn.ensemble
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from AnaliseInicial.TratamentoDados import (
    np,
    dataFrame2,
    dataFrame,
    varDataFrame
)

# Cria um list com as 57 variáveis restantes
cols = list(dataFrame.columns)

# Seta a variável a ser predita
colPredict = "SalePrice"


""" 
Analise de colinearidade
 """
# Usa Spearman porque tem variáveis ordinais e não
# assume normalidade dos dados
corr = dataFrame.corr(method="spearman") #Matriz de correlação
triu = np.triu(corr, True) #Remove triangulo inferior
triu = triu[:, :-1] #Remove SalePrice
triu[-1,:] = corr.values[-1,:-1] #Adiciona SalePrice na ultima linha

# Indices das matrizes para slicing
indices=np.indices(triu.shape)

# Nome das variáveis na linha e coluna para cada posição
row_names=corr.columns.values[indices[0]]
col_names=corr.columns.values[indices[1]]
row_col_names = np.dstack((row_names, col_names))

# Gera pares de variáveis autocorrelacionadas
mascara_correlacao = np.abs(triu)>0.7
col_pairs_autocorr_0_7 = row_col_names[mascara_correlacao]


# Remove as variáveis correlacionadas com SalePrice
col1_mask = col_pairs_autocorr_0_7[:,0] != colPredict
col_pairs_autocorr_0_7 = col_pairs_autocorr_0_7[col1_mask]

# Correlação de cada variável com a variável SalePrice
cols_1_values = corr.loc[colPredict][col_pairs_autocorr_0_7[:,0]]
cols_2_values = corr.loc[colPredict][col_pairs_autocorr_0_7[:,1]]
corr_values=np.dstack((cols_1_values,cols_2_values))[0]
cols_values=np.dstack((col_pairs_autocorr_0_7, corr_values))


# Exportar dataframe para Latex
# import clipboard
# clipboard.copy(pd.DataFrame({"Atributo1": col_pairs_autocorr_0_7[:,0], 
#     "Atributo2": col_pairs_autocorr_0_7[:,1], 
#     "Atr1_|r|_SalePrice": np.abs(cols_1_values.values),
#     "Atr2_|r|_SalePrice": np.abs(cols_2_values.values),
#     "|r|": triu[mascara_correlacao][col1_mask]
#     }).to_latex(index=False))


# Gráfico de colinearidade
# cmap = sns.color_palette("RdBu_r", 8)
# cmap.append("#000000")
# fig, ax = plt.subplots(figsize=(8,8))
# sns.heatmap(triu, ax=ax, cmap=cmap, center=0,
#             square=True, linewidths=1, 
#             cbar_kws={"shrink": .5, "ticks": [-0.35, 0, 0.35, 0.7]},
#             vmin=-0.4,vmax=0.9,
#             xticklabels=corr.columns[:-1], yticklabels=corr.columns
#             )
# plt.show()


# Pega variavel com menor autocorrelação para remoção
# mascara_remocao = cols_values[:,:,1].argmin(axis=1)
# col_pairs = cols_values[:,:,0]
# vars_menor_correlacao = [i[j] for i, j in zip(col_pairs, mascara_remocao)]

# Ajuste manual
# naoremove = ["KitchenQual", "GarageYrBlt", "Fireplaces", "1stFlrSF"] 
# [vars_menor_correlacao.remove(i) for i in naoremove]
# remove = ["FireplacesQu", "TotalBsmtSF"]
# [vars_menor_correlacao.append(i) for i in remove]

# Remover as colunas com |r| > 0,7
# dataFrame2 = dataFrame2.drop(vars_menor_correlacao,1)
# dataFrame = dataFrame.drop(vars_menor_correlacao,1)

""" 
 Análise de correlação entre variáveis nominais;
"""
#Colunas presentes
cols2 = dataFrame.columns.values


#Máscara de colunas que não são SalePrice
colsNotSalePrice2 = cols2[cols2!="SalePrice"]

from sklearn.feature_selection import chi2
import scipy.stats

dataFrameNominal = dataFrame[varDataFrame["Name"][varDataFrame["Ordered"]==False]]
dataFrameNominal = dataFrameNominal.join(dataFrame.iloc[:,-1])

from scipy import stats
for col in dataFrameNominal.columns[:-1]:
    dataFrameNominal[col].cat.rename_categories(np.arange(1,dataFrameNominal[col].cat.categories.size+1), inplace=True)

#Inicializa tabela de correlacao
corrTab=pd.DataFrame(index=dataFrameNominal.columns[:-1],columns=dataFrameNominal.columns[:-1])

# Roda o teste qui-quadrado para cada variável nominal
for col_i in range(dataFrameNominal.columns.size):
    for col_j in range(col_i+1, dataFrameNominal.columns.size):
        col1 = dataFrameNominal.columns[col_i]
        col2 = dataFrameNominal.columns[col_j]
        ct = pd.crosstab(dataFrameNominal[col2], dataFrameNominal[col1])
        res=stats.chi2_contingency(ct)[1]
        corrTab[col1][col2] = res



# Calcula a correlação entre cada variável nominal e a resposta
for col in dataFrameNominal.columns[:-1]:
    corrTab.loc["SalePrice", col] = scipy.stats.f_oneway(*dataFrameNominal.groupby(col)["SalePrice"].apply(np.array))[0]
    

corrTab.to_clipboard()

corr = corrTab
triu = np.tril(corr, -1) #Remove triangulo inferior
triu[0:23,0:23] = triu[0:23,0:23].T #Adiciona SalePrice na ultima linha

# Indices das matrizes para slicing
indices=np.indices(triu.shape)

# Nome das variáveis na linha e coluna para cada posição
row_names=corr.index.values[indices[0]]
col_names=corr.columns.values[indices[1]]
row_col_names = np.dstack((row_names, col_names))

# Gera pares de variáveis autocorrelacionadas
mascara_correlacao = np.triu(np.abs(triu)<1e-135, 1)
col_pairs_autocorr_0_7 = row_col_names[mascara_correlacao]


# Remove as variáveis correlacionadas com SalePrice
col1_mask = col_pairs_autocorr_0_7[:,0] != colPredict
col_pairs_autocorr_0_7 = col_pairs_autocorr_0_7[col1_mask]

# Correlação de cada variável com a variável SalePrice
cols_1_values = corr.loc[colPredict][col_pairs_autocorr_0_7[:,0]]
cols_2_values = corr.loc[colPredict][col_pairs_autocorr_0_7[:,1]]
corr_values=np.dstack((cols_1_values,cols_2_values))[0]
cols_values=np.dstack((col_pairs_autocorr_0_7, corr_values))

#Exporta atributos nominais escolhidos
# import clipboard
# exportDf = pd.DataFrame({"Atributo1": col_pairs_autocorr_0_7[:,0], 
#     "Atributo2": col_pairs_autocorr_0_7[:,1], 
#     "Atr1_|r|_SalePrice": np.abs(cols_1_values.values),
#     "Atr2_|r|_SalePrice": np.abs(cols_2_values.values),
#     "|r|": triu[mascara_correlacao][col1_mask]
#     })
# exportDf.to_clipboard(index=False)
# clipboard.copy(exportDf.to_latex(index=False))



""" 
Análise de correlação das numéricas e ordinais
 """
corr = dataFrame.corr(method="spearman") #Matriz de correlação
triu = np.triu(corr, True) #Remove triangulo inferior
triu = triu[:, :-1] #Remove SalePrice
triu[-1,:] = corr.values[-1,:-1] #Adiciona SalePrice na ultima linha

# Gráfico de colinearidade
cmap = sns.color_palette("RdBu_r", 8)
cmap.append("#000000")
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(triu, ax=ax, cmap=cmap, center=0,
            square=True, linewidths=1, 
            cbar_kws={"shrink": .5, "ticks": [-0.35, 0, 0.35, 0.7]},
            vmin=-0.4,vmax=0.9,
            xticklabels=corr.columns[:-1], yticklabels=corr.columns
            )
plt.show()

# Indices das matrizes para slicing
indices=np.indices(triu.shape)

# Nome das variáveis na linha e coluna para cada posição
row_names=corr.columns.values[indices[0]]
col_names=corr.columns.values[indices[1]]
row_col_names = np.dstack((row_names, col_names))

# Gera pares de variáveis autocorrelacionadas
mascara_correlacao = np.abs(triu)>0.7
col_pairs_autocorr_0_7 = row_col_names[mascara_correlacao]


# Remove as variáveis correlacionadas com SalePrice
col1_mask = col_pairs_autocorr_0_7[:,0] != colPredict
col_pairs_autocorr_0_7 = col_pairs_autocorr_0_7[col1_mask]

# Correlação de cada variável com a variável SalePrice
cols_1_values = corr.loc[colPredict][col_pairs_autocorr_0_7[:,0]]
cols_2_values = corr.loc[colPredict][col_pairs_autocorr_0_7[:,1]]
corr_values=np.dstack((cols_1_values,cols_2_values))[0]
cols_values=np.dstack((col_pairs_autocorr_0_7, corr_values))


#Exporta resultados
import clipboard
exportDf = pd.DataFrame({"Atributo1": col_pairs_autocorr_0_7[:,0], 
    "Atributo2": col_pairs_autocorr_0_7[:,1], 
    "Atr1_|r|_SalePrice": np.abs(cols_1_values.values),
    "Atr2_|r|_SalePrice": np.abs(cols_2_values.values),
    "|r|": triu[mascara_correlacao][col1_mask]
    })
# exportDf.to_clipboard(index=False)
# clipboard.copy(exportDf.to_latex(index=False))



""" 
 Seleção com KBest mutual_info
"""
# Transforma nominais em números 
# (não são afetados por mutual_info)
dataFrame3=dataFrame
colsCategory = dataFrame3.columns[(dataFrame3.dtypes=="category")]
for col in colsCategory:
    dataFrame3[col].cat.rename_categories(np.arange(1, dataFrame3[col].cat.categories.size+1, dtype='float64'), inplace=True)

dataFrame3.fillna(dataFrame3.median(), inplace=True)

# Algoritmos de escolha de variáveis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression

# Seletores (só ficou legal mutual_info_regression)
selectors = [mutual_info_regression]

for selector in selectors:
    # Criar modelo de seletor com 10 melhores
    selectorModel = SelectKBest(selector, k=10)
    
    # Ajustar modelo para as variáveis
    X_kbest = selectorModel.fit_transform(dataFrame3[colsNotSalePrice2], dataFrame3["SalePrice"])
    

    # Mostrar colunas selecionadas
    print("Selecionados por", selector.__name__)
    for i in colsNotSalePrice2[selectorModel.get_support()]:
        print(i, end=', ')
    print('\n', end='\n')

#pd.to_pickle(colsNotSalePrice2, "colsNotSalePrice2")
#dataFrame3.to_csv("dataFrame3")
# colsNotSalePrice2 = pd.read_pickle("colsNotSalePrice2")
# dataFrame3 = pd.read_csv("dataFrame3")


""" 
Seleção Wrapper RFE
"""
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

#Cria estimador
estimator3 = RandomForestRegressor()
#Cria seletor com validação cruzada 3-fold e 10 repetições
selector3 = RFECV(estimator3, min_features_to_select=1, step=1, cv=model_selection.RepeatedStratifiedKFold(n_splits=3, n_repeats=10),
              scoring='r2', n_jobs=-1)
selector3 = selector3.fit(dataFrame3[colsNotSalePrice2], dataFrame3["SalePrice"])
for col in colsNotSalePrice2[selector3.ranking_==1]:
    print("%s" % col, end=", ")
    


#Seleção por forward SequentialFeatureSelector 
from mlxtend import feature_selection

#SFS para random forests
sfs3=feature_selection.SequentialFeatureSelector(
    estimator3,
    k_features=79,
    forward=True,
    scoring="r2",
    cv=model_selection.RepeatedStratifiedKFold(3, 10),
    n_jobs=-1)


sfs4 = sfs3.fit(dataFrame3[colsNotSalePrice2], dataFrame3["SalePrice"])

#Gera gráficos com os resultados

# #DataFrame com resultados do SVM
# svmPlot=pd.DataFrame({
#     "Nº variáveis": range(5, len(selector3.grid_scores_) + 5), 
#     "algorithm":"svm", 
#     "Correlação": selector.grid_scores_})

#DataFrame com resultados do RF

# Le dos dados salvos
# selector3 = pd.read_pickle("./PickledObjects/selector3.pkl")

# Cria dataframe do RFE-rf para plotar
rfPlot=pd.DataFrame({
    "Nº variáveis": range(1, len(selector3.grid_scores_)+1), 
    "algorithm":"RFE-rf", 
    "Correlação": selector3.grid_scores_})

# Le dos dados salvos
# sfs4 = pd.read_pickle("./PickledObjects/sfs4.pkl")

# Popula dataframe do SFS-rf para plotar
dict4 = sfs4.get_metric_dict()
rfPlot2=pd.DataFrame(columns=["Nº variáveis", "algorithm", "Correlação"])
for key,value in dict4.items():
    rfPlot2.loc[rfPlot2.shape[0]] = {"Nº variáveis": key,
    "algorithm": "SFS-rf",
    "Correlação":value["avg_score"]
    }

#DataFrame conjunto
jointDf=rfPlot2.append(rfPlot)

import seaborn as sns

#Fundo branco
sns.set(style="white")

#Paleta de cores brilhante
palette=sns.color_palette("bright", 2)

#Cria gráfico de linha com amplitude total
ax1=sns.lineplot(x="Nº variáveis", y="Correlação",
             data=jointDf, hue="algorithm", palette=palette)
ax1.set(xticks=np.arange(0, 90, 10))
texto=ax1.legend().get_texts()
ax1.set_ylabel("Correlação r²")
texto[0].set_text("Legenda")
texto[0].set_x(-30)



#Cria seletor para mostrar colunas selecionadas para random forests
selector4 = RFE(estimator3, 16, step=1)
selector4 = selector4.fit(dataFrame[colsNotSalePrice2[selector3.support_]], dataFrame["SalePrice"])
for col in colsNotSalePrice2[selector3.support_][selector4.support_]:
    print("%s" % col, end=", ")
