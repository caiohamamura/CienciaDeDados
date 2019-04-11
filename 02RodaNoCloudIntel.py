import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import sys
colsNotSalePrice2 = pd.read_pickle("./PickledObjects/colsNotSalePrice2.pkl")
dataFrame3 = pd.read_csv("./PickledObjects/dataFrame3.csv")

def write(val):
    with open("outlog%.d" % rank, "a+") as f:
        f.write(val)
        f.write("\n")


from mlxtend import feature_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

#Cria estimador
estimator3 = RandomForestRegressor()
#Cria seletor com validação cruzada 3-fold e 10 repetições
sfs3=feature_selection.SequentialFeatureSelector(
    estimator3,
    k_features=21,
    forward=True,
    scoring="r2",
    cv=model_selection.RepeatedStratifiedKFold(3, 10),
    n_jobs=-1)


sfs4 = sfs3.fit(dataFrame3[colsNotSalePrice2], dataFrame3["SalePrice"])
pd.to_pickle(sfs3, "./PickledObjects/sfs3.pkl")
pd.to_pickle(sfs4, "./PickledObjects/sfs4.pkl")

