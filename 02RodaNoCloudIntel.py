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
from sklearn.svm import SVR
from sklearn import model_selection

from mlxtend import feature_selection

estimator = SVR(kernel="linear")
sfs=feature_selection.SequentialFeatureSelector(
    estimator,
    k_features=21,
    forward=True,
    scoring="r2",
    cv=model_selection.RepeatedStratifiedKFold(3, 10),
    n_jobs=-1)


sfs2 = sfs.fit(dataFrame3[colsNotSalePrice2], dataFrame3["SalePrice"])

pd.to_pickle(sfs, "./PickledObjects/sfs.pkl")
pd.to_pickle(sfs2, "./PickledObjects/sfs2.pkl")

