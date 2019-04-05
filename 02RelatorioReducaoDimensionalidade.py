# =============================================================================
import pandas as pd
import sklearn.ensemble
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from AnaliseInicial.TratamentoDados import (
    np,
    dataFrame2,
    dataFrame
)

# Cria um list com as 57 variáveis restantes
cols = list(dataFrame.columns)

# Seta a variável a ser predita
colPredict = "SalePrice"


##############################
#Analise de colinearidade
##############################
# dataFrame sem colunas nominais binarias
corr = dataFrame.corr() #Matriz de correlação
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
# plt.savefig("figure.png",
# top=0.95,
# bottom=0.25,
# left=0.0,
# right=1.0,
# hspace=0.2,
# wspace=0.2)
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

import clipboard
clipboard.copy(pd.DataFrame({"Atributo1": col_pairs_autocorr_0_7[:,0], 
    "Atributo2": col_pairs_autocorr_0_7[:,1], 
    "Atr1_|r|_SalePrice": np.abs(cols_1_values.values),
    "Atr2_|r|_SalePrice": np.abs(cols_2_values.values),
    "|r|": triu[mascara_correlacao][col1_mask]
    }).to_latex())

# Pega variavel com menor autocorrelação para remoção
mascara_remocao = cols_values[:,:,1].argmin(axis=1)
col_pairs = cols_values[:,:,0]
vars_menor_correlacao = [i[j] for i, j in zip(col_pairs, mascara_remocao)]

#Remover kitchenQual que está por transição
vars_menor_correlacao.remove("KitchenQual")

# Remover as colunas com |r| > 0,7
dataFrame2 = dataFrame2.drop(vars_menor_correlacao,1)
dataFrame = dataFrame.drop(vars_menor_correlacao,1)


# Algoritmos de escolha de variáveis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import f_classif


#Colunas presentes
cols2 = dataFrame2.columns.values
#Máscara de colunas que não são SalePrice
colsNotSalePrice2 = cols2[cols2!="SalePrice"]

#FIXME: Completar NaN com 0 
dataFrame3=dataFrame2.fillna(0)

# Seletores
selectors = [chi2, f_regression]
colsNotSalePrice1 = cols1[cols1!="SalePrice"]

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
