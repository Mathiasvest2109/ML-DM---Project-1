import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
X = pd.read_csv("features.csv")
#X = X.fillna(0)
X = X.dropna()
attributeNames = X.columns
plt.figure()
plt.subplots_adjust(hspace=0.6)

def getName(input):
    switch={
        "age":"age",
        "sex":"Gender",
        "cp":"Chest pain type",
        "trestbps":"resting blood pressur",
        "fbs":"(fasting blood sugar > 120 mg/dl)\n (1 = true; 0 = false)",
        "restecg":"resting electrocardiographic results",
        "thalach":"maximum heart rate achieved",
        "exang":"exercise induced angina \n(1 = yes; 0 = no)",
        "oldpeak":"ST depression induced by\n exercise relative to rest",
        "slope":"the slope of the peak exercise\n ST segment(1 = up; 2 = flat; 3 = down)",
        "ca":"number of major vessels (0-3)",
        "thal":"thalassemia\n(3 = normal, 6 = fixed defect, 7 = reversible defect)",
        "chol":"serum cholesterol"
    }
    return switch.get(input,"Invalid")

for i in range(13):
        plt.subplot(4,4,i+1)
        interval = int(X.max(axis=0)[attributeNames[i]]-X.min(axis=0)[attributeNames[i]])+1
        plt.hist(X[attributeNames[i]],bins=X[attributeNames[i]].nunique() if X[attributeNames[i]].nunique() > 10 else X[attributeNames[i]].nunique()*2)
        
        plt.title(getName(attributeNames[i]))
plt.show()