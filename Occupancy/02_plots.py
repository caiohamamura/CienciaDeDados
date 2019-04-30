import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Plot style
trainResult = pd.read_csv("data/trainResult.csv")
dataSet = pd.read_csv("data/normalizedData.csv")
X = dataSet.iloc[:,:-1]
y = dataSet.iloc[:,-1]

sns.set_style("whitegrid") 
palette=sns.color_palette("muted")[0:trainResult["network"].unique().size] 


# Plot
plt.figure(dpi=150)
ax = sns.lineplot(x="epoch", y="loss", hue="network", style="type", data=trainResult, palette=palette)


# Plot configs
params = {"loc":"upper right"}
leg = ax.legend()
handles = leg.legendHandles # Remove handler for type

handles[0].set_label("ANN")
handles[1].set_label("Log.5")
handles[2].set_label("Log.10")
handles[3].set_label("Log.15")
handles[4].set_label("Log.5.moment")
handles[5].set_label("RBF.5")
handles[6].set_label("Log.5.5")
handles[7].set_label("Set")
ax.legend(handles=handles, **params)
ax.set_xlabel(ax.get_xlabel().capitalize())
ax.set_ylabel(ax.get_ylabel().capitalize())
plt.xticks(pd.np.arange(0, trainResult["epoch"].max()+1, 2))
plt.xlabel("Epoch")
plt.ylabel("Log-loss")
plt.show()

from radarboxplot import radarboxplot
plt.figure(dpi=150)
axs=radarboxplot(X, y, X.columns.values, nrows=1, ncols=2)
plt.show()