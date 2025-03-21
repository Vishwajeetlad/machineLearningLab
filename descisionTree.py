import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# 1. Load and Explore the Data
# ----------------------------
iris = load_iris()
X = iris.data  # Feature matrix
y = iris.target  # Target vector

# Convert to DataFrame for better visualization (optional)
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print("First 5 rows of the dataset:")
print(df.head())

# ----------------------------
# 2. Split the Data
# ----------------------------
# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# ----------------------------
# 3. Create and Train the Decision Tree Model
# ----------------------------
# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# ----------------------------
# 4. Make Predictions and Evaluate the Model
# ----------------------------
# Predict on the test data
y_pred = dt_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ----------------------------
# 5. Visualize the Decision Tree
# ----------------------------
plt.figure(figsize=(12,8))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()
