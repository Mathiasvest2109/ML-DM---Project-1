import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import svd
import enum


X = pd.read_csv("features.csv")
attributeNames = X.columns
count=0
for i in range(len(X["oldpeak"])):
    if(X["oldpeak"][i]==X["target"][i] == 0): count+=1
print("ST-depression is 0\ncount: ",(X["oldpeak"]==0).sum(),"\n\nDisease is 0 \ncount: ",(X["target"]==0).sum(),"\n\nboth are 0\ncount: ",count)

plt.plot(X["oldpeak"],X["target"],'o',alpha=0.02)
plt.xlabel("oldpeak")
plt.ylabel("target")
# plt.title(attributeNames[i])
plt.show()
