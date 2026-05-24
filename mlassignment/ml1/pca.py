import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("songs.csv")

# =========================
# DATASET PREVIEW
# =========================
print("\nDataset Preview:\n")
print(data.head())

# =========================
# SELECT FEATURES
# =========================
X = data[[
    'danceability',
    'energy',
    'tempo',
    'valence',
    'acousticness',
    'liveness'
]]

# =========================
# TARGET COLUMN
# =========================
y = data['track_genre']

# =========================
# STANDARDIZE DATA
# =========================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# APPLY PCA
# Reduce dimensions to 2
# =========================
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Shape:")
print(X_pca.shape)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# CREATE KNN MODEL
# =========================
model = KNeighborsClassifier(n_neighbors=5)

# =========================
# TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test)

# =========================
# ACCURACY
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# =========================
# PCA VISUALIZATION
# =========================
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)

plt.title("PCA Visualization of Songs Dataset")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()