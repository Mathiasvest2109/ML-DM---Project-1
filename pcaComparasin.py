import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import svd

X = pd.read_csv("features.csv")
#X = X.fillna(0)
X = X.dropna()
chosenfeature = "target"
meow = X
X = X.drop(columns=chosenfeature)
attributeNames = X.columns

classLabels = pd.cut(meow[chosenfeature], [-1,0, np.inf],labels=['no disease', 'disease' ])

classnames=sorted(set(classLabels))
classDict = dict(zip(classnames, range(len(classnames))))

print(meow.min(axis=0)[chosenfeature],meow.max(axis=0)[chosenfeature])

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

N= len(y)
M= len(attributeNames)
C= len(classnames)


#r = np.arange(1, X.shape[1] + 1)
#plt.bar(r, np.std(X, 0))
#plt.xticks(r, attributeNames)
#plt.ylabel("Standard deviation")
#plt.xlabel("Attributes")
#plt.title("Heart disease with regard to "+chosenfeature+": attribute standard deviations")


## Investigate how standardization affects PCA

# Try this *later* (for last), and explain the effect
# X_s = X.copy() # Make a to be "scaled" version of X
# X_s[:, 2] = 100*X_s[:, 2] # Scale/multiply attribute C with a factor 100
# Use X_s instead of X to in the script below to see the difference.
# Does it affect the two columns in the plot equally?


# Subtract the mean from the data
Y1 = X.to_numpy() - np.ones((N, 1)) * X.mean().to_numpy()

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X.to_numpy() - np.ones((N, 1)) * X.mean().to_numpy()
Y2 = Y2 * (1 / np.std(Y2, 0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ["Zero-mean", "Zero-mean and unit variance"]
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.6)
plt.title("Heart disease with regard to "+chosenfeature+": Effect of standardization")
nrows = 10
ncols = 10

k=1
# Obtain the PCA solution by calculate the SVD of either Y1 or Y2
U, S, Vh = svd(Ys[k], full_matrices=False)
V = Vh.T  # For the direction of V to fit the convention in the course we transpose
# For visualization purposes, we flip the directionality of the
# principal directions such that the directions match for Y1 and Y2.

V = -V
U = -U

# Compute variance explained
rho = (S * S) / (S * S).sum()

# Compute the projection onto the principal components
Z = U * S

# Plot projection
for yaxis in range(10):
    for xaxis in range (10):
        plt.subplot(nrows, ncols, xaxis*10+yaxis+1)
        C = len(classnames)
        for c in range(C):
            plt.plot(Z[y == c, yaxis], Z[y == c, xaxis], ".", alpha=0.5)
        if (yaxis == 0): plt.ylabel("PC" + str(xaxis + 1))
        if (xaxis == 9): plt.xlabel("PC" + str(yaxis + 1))
        plt.xticks([])  
        plt.yticks([])
        
        #plt.legend(classnames)
        plt.axis("equal")
plt.suptitle(titles[k] + "\n" + "Projection")
plt.show()