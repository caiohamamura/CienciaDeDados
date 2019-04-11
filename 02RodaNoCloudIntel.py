from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()   
colsNotSalePrice2 = pd.read_pickle("colsNotSalePrice2")
dataFrame3 = pd.read_csv("dataFrame3")
corrMutualInfo = pd.DataFrame(columns=[colsNotSalePrice2])

def write(val):
    with open("outlog%.d" % rank, "a+") as f:
        f.write(val)
        f.write("\n")


for col in np.array_split(colsNotSalePrice2, size)[rank]:
    write("Processing %s" % col)
    corrMutualInfo.loc[col] = mutual_info_regression(dataFrame3[colsNotSalePrice2], dataFrame3[col], discrete_features=True)
     #print("Rank %d got column: %s" % (rank, col))
    
corrMutualInfo.to_csv("corrMutualInfo%.2d.csv" % rank)
