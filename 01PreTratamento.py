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


# Checar dados faltantes
import missingno as msno
msno.matrix(dataFrame, labels=True, color=(0.5,0.5,1), sparkline=False)
