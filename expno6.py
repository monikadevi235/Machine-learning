# Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Create the dataset
data = {
    'Age': ['Young', 'Young', 'Middle', 'Old', 'Old', 'Middle', 'Young', 'Old'],
    'Salary': ['High', 'Low', 'Medium', 'Medium', 'High', 'Low', 'Medium', 'Low'],
    'Experience': ['Low', 'Low', 'High', 'High', 'Medium', 'Medium', 'Low', 'High'],
    'Location': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'AcceptJob': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Step 2: Encode categorical data (CORRECT way)
encoder = LabelEncoder()
for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

# Step 3: Split input and output
X = df.drop('AcceptJob', axis=1)
y = df['AcceptJob']

# Step 4: Create and train Decision Tree model (ID3 - Entropy)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Step 5: Predict a new sample (manual encoding)
# Young=2, Medium=2, Low=1, Yes=1
new_sample = [[2, 2, 1, 1]]

prediction = model.predict(new_sample)
print("Predicted Output (Accept Job):", prediction)

# Step 6: Visualize the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True
)
plt.show()
