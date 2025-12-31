from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
texts = [
    "The movie was fantastic",
    "I hated the storyline",
    "Brilliant acting",
    "Very boring movie",
    "Loved the background music",
    "Worst film ever",
    "Enjoyed every scene",
    "Waste of money",
    "Truly inspiring movie",
    "Not interesting at all"
]

labels = [
    "Positive", "Negative", "Positive", "Negative", "Positive",
    "Negative", "Positive", "Negative", "Positive", "Negative"
]

# Step 1: Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Step 2: Train Naive Bayes
model = MultinomialNB()
model.fit(X, labels)

# Step 3: Test sentence
test_sentence = ["brilliant movie"]
test_vector = vectorizer.transform(test_sentence)

# Step 4: Prediction
prediction = model.predict(test_vector)
print("Predicted Sentiment:", prediction[0])
