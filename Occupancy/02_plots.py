import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from training
trainResult = pd.read_csv("data/trainResult.csv")
dataSet = pd.read_csv("data/normalizedData.csv")
X = dataSet.iloc[:,:-1]
y = dataSet.iloc[:,-1]

# Plot style
sns.set_style("whitegrid") 
# Palette to use
palette = sns.color_palette("muted")[0:trainResult["network"].unique().size] 

# Plot resolution
plt.figure(dpi=150)

# Seaborn lineplot by network and type (train/validation)
ax = sns.lineplot(
    x="epoch", y="loss", hue="network", style="type", 
    data=trainResult, palette=palette
)

# Plot configs
# Legend position
params = {"loc":"upper right"}
leg = ax.legend()

# Get each handle from legend and rename
handles = leg.legendHandles 
handles[0].set_label("ANN")
handles[1].set_label("Log.5")
handles[2].set_label("Log.10")
handles[3].set_label("Log.15")
handles[4].set_label("Log.5.moment")
handles[5].set_label("RBF.5")
handles[6].set_label("Log.5.5")
handles[7].set_label("Set")

# Set legend to renamed handles
ax.legend(handles=handles, **params)

# Put numbers in x axis each step
step = 2
plt.xticks(pd.np.arange(0, trainResult["epoch"].max()+1, step))

# Labels for x and y
plt.xlabel("Epoch")
plt.ylabel("Log-loss")
plt.show()


# Plot the radar boxplot
from radarboxplot import radarboxplot

plt.figure(dpi=150)
axs=radarboxplot(X, y, X.columns.values, nrows=1, ncols=2)
plt.show()