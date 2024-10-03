import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import svd

X = pd.read_csv("features.csv")
#X = X.fillna(0)
X = X.dropna()
chosenfeature = "target"
X = X.drop(columns=chosenfeature)
attributeNames = X.columns
N=len(X["age"])
# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Ys = X.to_numpy() - np.ones((N, 1)) * X.mean().to_numpy()
Ys = Ys * (1 / np.std(Ys, 0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions
# of Y2

# Store the two in a cell, so we can just loop over them:
title= "Zero-mean and unit variance"
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(6, 5))

# Obtain the PCA solution by calculate the SVD of either Y1 or Y2
U, S, Vh = svd(Ys, full_matrices=False)
V = Vh.T  # For the direction of V to fit the convention in the course we transpose
# For visualization purposes, we flip the directionality of the
# principal directions such that the directions match for Y1 and Y2.

V = -V

# Plot attribute coefficients in principal component space
colors = plt.cm.tab20(np.linspace(0, 1, len(attributeNames)))
# plt.subplot(nrows, ncols, 1 + k)
for att in range(V.shape[1]):
    plt.arrow(0, 0, V[att, i], V[att, j], color=colors[att], head_width=0.03, head_length=0.03)

for att in range(V.shape[1]):
    print(str(attributeNames[att])+": ",V[att, i],"\n")

plt.legend(attributeNames, loc='upper left')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xlabel("PC" + str(i + 1))
plt.ylabel("PC" + str(j + 1))
plt.grid()
# Add a unit circle
plt.plot(
    np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01))
)
plt.title(title + "\n" + "Attribute coefficients")
plt.axis("equal")

plt.show()