import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from principal_component_analysis import PCA

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names


# Apply PCA with 2 components
pca_model = PCA(num_components=2)
result_pca = pca_model.fit_transform(X)

# Scatter plot of the transformed data
colors = ["navy", "turquoise", "darkorange"]
for i in range(len(target_names)):
    indices = y == i
    plt.scatter(
        result_pca[indices, 0],
        result_pca[indices, 1],
        c=colors[i],
        label=target_names[i],
        edgecolor="k",
    )

plt.title("PCA of (Iris dataset)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend()
plt.show()
