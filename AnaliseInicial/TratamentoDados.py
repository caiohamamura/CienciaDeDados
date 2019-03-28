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

# Lê arquivo CSV com as variáveis e tipos
#varsDataFrame2 = pd.read_csv("variables.csv", delimiter=";")

#%%
# =============================================================================
# Ajustar colunas QUALITATIVAS: nominais/categóricas ou ordinais
# =============================================================================
# Lê arquivo CSV com as variáveis e tipos
varsDataFrame = pd.read_csv("variables2.csv", delimiter=";", decimal=",")

# Quantitativas
quantitColumns=pd.isnull(varsDataFrame.Categorical)

# Qualitativas
qualitColumns=pd.notnull(varsDataFrame.Categorical)
# Separa as variáveis nominais/categóricas
nominalColumns=np.logical_and(varsDataFrame.Categorical, np.logical_not(varsDataFrame.Ordered))
# Separa as variáveis ordinais
ordinalColumns=np.logical_and(pd.notnull(varsDataFrame.Categorical), varsDataFrame.Ordered)

# =============================================================================
# Ajustar colunas com dados NaN
# =============================================================================
# Preenche as células quantitativas nan com 0
for col in varsDataFrame[quantitColumns][["Name"]].values:
    dataFrame[col]=dataFrame[col].fillna(0)

# Ajusta dados da variável Alley preenchidos como NA
colAlley = varsDataFrame.Name[nominalColumns][5]
dataFrame[colAlley]=dataFrame[colAlley].fillna('NA')

# Ajusta dados da variável GarType preenchidos como NA
colGarType = varsDataFrame.Name[nominalColumns][57]
dataFrame[colGarType]=dataFrame[colGarType].fillna('NA')

# Ajusta dados da variável MiscFeat preenchidos como NA
colMiscFeat = varsDataFrame.Name[nominalColumns][73]
dataFrame[colMiscFeat]=dataFrame[colMiscFeat].fillna('NA')

# Ajusta dados das variáveis ordinais preenchidos como NA
for col in varsDataFrame[ordinalColumns][["Name"]].values:
    dataFrame[col]=dataFrame[col].fillna('NA')
    
# Ajusta dados da variável MSZoning preenchidos como 'C (all)'
dataFrame['MSZoning'] = dataFrame['MSZoning'].replace('C (all)', 'C')

# Ajusta dados da variável BsmtFinType1 preenchidos como 'GLQ'
dataFrame['BsmtFinType1'] = dataFrame['BsmtFinType1'].replace('GLQ', 'GQL')

# Ajusta dados da variável BsmtFinType2 preenchidos como 'GLQ'
dataFrame['BsmtFinType2'] = dataFrame['BsmtFinType2'].replace('GLQ', 'GQL')

# Ajusta dados da variável Exterior2nd preenchidos como 'Brk Cmn'
dataFrame['Exterior2nd'] = dataFrame['Exterior2nd'].replace('Brk Cmn', 'BrkComm')

# Ajusta dados da variável Exterior2nd preenchidos como 'CmentBd'
dataFrame['Exterior2nd'] = dataFrame['Exterior2nd'].replace('CmentBd', 'CemntBd')

# Ajusta dados da variável BldgType preenchidos como 'Twnhs'
dataFrame['BldgType'] = dataFrame['BldgType'].replace('Twnhs', 'TwnhsI')

# Ajusta dados da variável BldgType preenchidos como 'Duplex'
dataFrame['BldgType'] = dataFrame['BldgType'].replace('Duplex', 'Duplx')

# Ajusta dados da variável BldgType preenchidos como '2fmCon'
dataFrame['BldgType'] = dataFrame['BldgType'].replace('2fmCon', '2FmCon')

# Ajusta dados da variável Neighborhood preenchidos como 'NAmes'
dataFrame['Neighborhood'] = dataFrame['Neighborhood'].replace('NAmes', 'Names')

# Transforma as variáveis qualitativas de formato String para Objeto
varsDataFrame.Categories = varsDataFrame.Categories[qualitColumns].map(eval)
    
# Ajusta as categorias nominais para o tipo de dados "Categoria"
for col, cats in varsDataFrame[nominalColumns][["Name", "Categories"]].values:
    dataFrame[col] = pd.Categorical(dataFrame[col], categories=cats, ordered=False)
# Renomeia as categorias nominais utilizando sequência de números 
for col, cats in varsDataFrame[nominalColumns][["Name", "Categories"]].values:
    dataFrame[col] = dataFrame[col].cat.rename_categories(range(dataFrame[col].cat.categories.size))

# Ajusta as categorias ordinais para o tipo de dados "Categoria"
for col, cats in varsDataFrame[ordinalColumns][["Name", "Categories"]].values:
    dataFrame[col] = pd.Categorical(dataFrame[col], categories=cats, ordered=True)
# Renomeia as categorias ordinais utilizando sequência de números
for col, cats in varsDataFrame[ordinalColumns][["Name", "Categories"]].values:
    dataFrame[col] = dataFrame[col].cat.rename_categories(range(dataFrame[col].cat.categories.size))
    
# Remove objetos NA restantes ("Wd Shng" => Exterior2nd, etc.)
dataFrame = dataFrame.dropna()

#import missingno as msno
#msno.matrix(dataFrame, labels=True, color=(0.5,0.5,1), sparkline=False)

#%%
#Calcular categorias para variaveis nominais (com base no arquivo de treino)
#def __main__():
#    nominalColsNames = varsDataFrame2.Name[nominalColumns]
#
#    for i, col in zip(nominalColsNames.index, nominalColsNames):
#        varsDataFrame2["Categories"][i] = str(list(dataFrame[col].values.categories))
#
#    varsDataFrame2.to_csv("variables2.csv", sep=";", decimal=",")