import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import svd
import enum


X = pd.read_csv("features.csv")

attribute = "target"
#if(attribute != "age"): continue
X = pd.read_csv("features.csv")
#X = X.fillna(0)
X = X.dropna()
print(attribute)
chosenfeature = attribute
meow = X
X = X.drop(columns=chosenfeature)


attributeNames = X.columns

classLabels = pd.cut(meow[chosenfeature], [-1,0, np.inf],labels=['no disease', 'disease' ])


classnames=sorted(set(classLabels))
classDict = dict(zip(classnames, range(len(classnames))))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

N= len(y)
M= len(attributeNames)
C= len(classnames)

Y2 = X.to_numpy() - np.ones((N, 1)) * X.mean().to_numpy()
Y2 = Y2 * (1 / np.std(Y2, 0))

U, S, Vh = svd(Y2, full_matrices=False)
V = Vh.T  # For the direction of V to fit the convention in the course we transpose
# For visualization purposes, we flip the directionality of the
# principal directions such that the directions match for Y1 and Y2.

V = -V
U = -U

# Compute variance explained
rho = (S * S) / (S * S).sum()

# Compute the projection onto the principal components
Z = U * S

i = 0
j = 1
k = 2


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Plot projection
C = len(classnames)
for c in range(C):
    xs = Z[y == c, i]
    ys = Z[y == c, j]
    zs = Z[y == c, k]
    ax.scatter(xs, ys, zs, marker='o')
ax.set_xlabel("PC" + str(i + 1))
ax.set_ylabel("PC" + str(j + 1))
ax.set_zlabel("PC" + str(k + 1))
ax.set_title("Zero-mean and unit variance" + "\n" + "Projection"+"\n"+chosenfeature)
ax.legend(classnames)
ax.axis("equal")

plt.show()