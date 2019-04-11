# -*- coding: utf-8 -*-
'''
Exporta:

 - dataFrame: colunas categóricas e nominais 
    corrigidas;

 - dataFrame2: colunas nominais estão binarizadas

 - np: numpy

 - varDataFrame: dados das variáveis

'''

import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import OneHotEncoder
#import clipboard
from matplotlib import pyplot as plt

# Numpy by Pandas
np = pd.np
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

# =============================================================================
# Ajustar colunas QUALITATIVAS: nominais/categóricas ou ordinais
# =============================================================================
# Lê arquivo CSV com as variáveis e tipos
# transforma lista string em lista
literal_eval=lambda x:eval(x) if x else x
varDataFrame = pd.read_csv("variables.csv", delimiter=";", converters={"Categories": literal_eval, "Categories_Fixed": literal_eval})


# Separa as variáveis nominais/categóricas

nominalColumnsMask=np.logical_and(
    varDataFrame.Categorical, 
    varDataFrame.Ordered==False
)
nominalColumns = varDataFrame["Name"][nominalColumnsMask].tolist()

ordinalColumnsMask=np.logical_and(
    varDataFrame.Categorical, 
    varDataFrame.Ordered==True
)
#col = varDataFrame.Name[nominalColumns][0]

####################
# Ordinais
####################
# Ajusta as categorias ordinais para o tipo de dados "Categoria"
for col, cats in varDataFrame[ordinalColumnsMask][["Name", "Categories"]].values:
    dataFrame[col].fillna("NA", inplace=True)
    dataFrame[col] = pd.Categorical(dataFrame[col], categories=cats, ordered=True)
    # renomeia as categorias usando sequências de números
    dataFrame[col].cat.rename_categories(
        range(1, dataFrame[col].cat.categories.size+1),
        inplace=True
        )
    

########################
# Nominais
########################
for col, cats, cats_fixed in varDataFrame[["Name", "Categories", "Categories_Fixed"]][nominalColumnsMask].values:
    null_mask = pd.isnull(dataFrame[col])
    dataFrame[col] = dataFrame[col].fillna("NaN")
    dataFrame[col] = pd.Categorical(dataFrame[col], categories=cats, ordered=False)
    dataFrame[col] = dataFrame[col].cat.rename_categories(cats_fixed)

#############################
## Converte as colunas nominais em binario
#############################
#Lista com todas as categorias
catsList = []
for i in range(len(nominalColumns)):
    catsList += [nominalColumns[i]+"_"+str(j) for j in dataFrame[nominalColumns[i]].cat.categories.tolist()]

ohe = OneHotEncoder(sparse=False) 
ohe.fit(dataFrame[nominalColumns])
dataFrame2 = dataFrame.join(pd.DataFrame(ohe.fit_transform(dataFrame[nominalColumns]).astype(np.uint), columns = catsList))
dataFrame2 = dataFrame2.drop(nominalColumns, 1)

##################################################


