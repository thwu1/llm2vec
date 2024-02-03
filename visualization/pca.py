from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X = np.random.rand(1000, 100)
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print(X_pca.shape)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.show()
