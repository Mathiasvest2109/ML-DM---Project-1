import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import svd
import enum


X = pd.read_csv("features.csv")
#X = X.fillna(0)    
X = X.dropna()
X = X.drop(columns="target")
# Step 2: Compute the correlation matrix
corr_matrix = X.corr()

# Step 3: Create the heatmap using matplotlib
plt.figure(figsize=(12, 8))

# Create a colormap ranging from blue (negative) to red (positive)
cmap = plt.get_cmap('PiYG')

# Use imshow to display the matrix
plt.imshow(corr_matrix, cmap=cmap, aspect='auto',vmin=-1, vmax=1)

# Add a color bar
plt.colorbar()

# Step 4: Add labels for x and y axes
plt.xticks(ticks=np.arange(len(X.columns)), labels=X.columns, rotation=90)
plt.yticks(ticks=np.arange(len(X.columns)), labels=X.columns)

# Add annotations with correlation values on the heatmap
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

# Step 5: Display the heatmap
plt.title('Heatmap of Correlation Coefficients')
plt.tight_layout()
plt.show()