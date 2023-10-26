import numpy as np


class PCA:
    def __init__(self, num_components=None):
        """
        Initializes a new instance of the class.

        Parameters:
            num_components (int): The number of components.

        Returns:
            None
        """
        self.num_components = num_components
        self.mean = None
        self.components = None

    def fit_transform(self, X):
        """
        Fit the PCA model and transform the input data.

        Parameters:
        - X: NumPy array, shape (n_samples, n_features)
          Input data matrix.

        Returns:
        - X_pca: NumPy array, shape (n_samples, num_components)
          Data matrix after PCA transformation.
        """
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        """
        Fit the PCA model.

        Parameters:
        - X: NumPy array, shape (n_samples, n_features)
          Input data matrix.
        """
        # Standardize the data (mean centering)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors in descending order of eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues = eigenvalues[sorted_indices]

        # Choose the top k eigenvectors (principal components)
        if self.num_components is not None:
            self.components = eigenvectors[:, : self.num_components]
        else:
            self.components = eigenvectors

    def transform(self, X):
        """
        Transform the input data using the fitted PCA model.

        Parameters:
        - X: NumPy array, shape (n_samples, n_features)
          Input data matrix.

        Returns:
        - X_pca: NumPy array, shape (n_samples, num_components)
          Data matrix after PCA transformation.
        """
        if self.mean is None or self.components is None:
            raise ValueError("PCA model not fitted. Call the fit method first.")

        X_centered = X - self.mean
        X_pca = np.dot(X_centered, self.components)

        return X_pca
