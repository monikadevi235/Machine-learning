import numpy as np
import pandas as pd

X = np.array([
    [2, 0, 1],
    [3, 1, 2],
    [4, 0, 3],
    [5, 2, 4],
    [6, 1, 5]
])

print("Original Data (X):\n", X)

mean = np.mean(X, axis=0)
print("\nMean of each attribute:\n", mean)

X_centered = X - mean
print("\nMean Centered Data:\n", X_centered)

variance = np.var(X, axis=0, ddof=1)
print("\nVariance of each attribute:\n", variance)

cov_matrix = np.cov(X_centered.T)
print("\nCovariance Matrix:\n", cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

total_variance = np.sum(eigenvalues)
explained_variance = (eigenvalues / total_variance) * 100
print("\nPercentage of Variance Explained:\n", explained_variance)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nSorted Eigenvalues:\n", eigenvalues)
print("\nSorted Eigenvectors:\n", eigenvectors)

W = eigenvectors[:, :2]
print("\nProjection Matrix W:\n", W)

Y = np.dot(X_centered, W)
print("\nReduced Data (2D):\n", Y)
