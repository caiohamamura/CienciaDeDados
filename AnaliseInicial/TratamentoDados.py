
#%%
import pandas as pd
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
varsDataFrame = pd.read_csv("variables.csv", delimiter=";")

# Separa as variáveis nominais/categóricas
nominalColumns=np.logical_and(varsDataFrame.Categorical, np.logical_not(varsDataFrame.Ordered))

# Nominais/categóricas
for col in varsDataFrame.Name[nominalColumns]:
    dataFrame[col] = pd.Categorical(dataFrame[col], ordered=False)


# Ordinais
# Busca as categorias ordinais não-nulas
ordinalColumns=pd.notnull(varsDataFrame.Categories)
# Transforma as categorias ordinais de formato String para Objeto
varsDataFrame.Categories = varsDataFrame.Categories[ordinalColumns].map(eval)
# Ajusta as categorias ordinais para o tipo de dados "Categoria"
for col, cats in varsDataFrame[ordinalColumns][["Name", "Categories"]].values:
    dataFrame[col] = pd.Categorical(dataFrame[col], categories=cats, ordered=True)
# Renomeia as categorias ordinais utilizando sequência de números
for col, cats in varsDataFrame[ordinalColumns][["Name", "Categories"]].values:
    dataFrame[col] = dataFrame[col].cat.rename_categories(range(dataFrame[col].cat.categories.size))

varsDataFrame = pd.read_csv("variables2.csv", delimiter=";", decimal=",")


#%%
#Calcular categorias para variaveis nominais
def __main__():
    nominalColsNames = varDataFrame.Name[nominalColumns]

    for i, col in zip(nominalColsNames.index, nominalColsNames):
        varDataFrame["Categories"][i] = str(list(dataFrame[col].values.categories))

    varDataFrame.to_csv("variables2.csv", sep=";", decimal=",")