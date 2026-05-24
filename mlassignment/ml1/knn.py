# ==============================
# SONG LANGUAGE PREDICTION
# USING KNN ALGORITHM
# ==============================

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("songs.csv")

# Display first 5 rows
print("Dataset Preview:\n")
print(data.head())

# Select Spotify audio features
X = data[
    [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'valence',
        'tempo'
    ]
]

# Target column
y = data['language']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train model
knn.fit(X_train, y_train)

# Predict test data
y_pred = knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# PREDICT NEW SONG LANGUAGE
# ==============================

new_song = [[
    0.75,    # danceability
    0.80,    # energy
    -5.0,    # loudness
    0.04,    # speechiness
    0.15,    # acousticness
    0.00,    # instrumentalness
    0.85,    # valence
    125      # tempo
]]

# Scale new song data
new_song_scaled = scaler.transform(new_song)

# Predict language
prediction = knn.predict(new_song_scaled)

print("\nPredicted Song Language:")
print(prediction[0])