import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------
# 1. Load the Data and Inspect
# ----------------------------
# Load a sample dataset (Titanic dataset)
df = sns.load_dataset('titanic')
print("Dataset Shape:", df.shape)
print("First 5 rows:\n", df.head())

# ----------------------------
# 2. Handling Missing Data
# ----------------------------
# Check for missing values in each column
print("\nMissing values in each column:\n", df.isnull().sum())

# Fill missing values for 'age' with the median
df['age'].fillna(df['age'].median(), inplace=True)
# Fill missing values for 'embarked' with the mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
# Drop 'deck' column due to high number of missing values
df.drop('deck', axis=1, inplace=True)

# Verify that missing data has been handled
print("\nMissing values after imputation:\n", df.isnull().sum())

# ----------------------------
# 3. Correlation Analysis
# ----------------------------
# Calculate correlation matrix for numerical features
corr_matrix = df.select_dtypes(include=[np.number]).corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# Plot heatmap for the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------
# 4. Encoding Techniques
# ----------------------------
# Example: Label Encoding for the 'sex' column
le = LabelEncoder()
df['sex_encoded'] = le.fit_transform(df['sex'])

# Example: One-Hot Encoding for the 'embarked' column
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# View the encoded columns
print("\nData after encoding:\n", df.head())

# ----------------------------
# 5. Scaling Numerical Features
# ----------------------------
scaler = StandardScaler()
numerical_cols = ['age', 'fare']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the scaled numerical columns
print("\nScaled Numerical Columns:\n", df[numerical_cols].head())

# ----------------------------
# 6. Different Graphs for EDA
# ----------------------------

# Histogram: Distribution of Age
plt.figure(figsize=(8, 4))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Scaled Age")
plt.show()

# Bar Plot: Survival Counts
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df)
plt.title("Survival Counts")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# Scatter Plot: Age vs. Fare colored by Survival
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='fare', hue='survived', data=df)
plt.title("Age vs Fare Scatter Plot")
plt.xlabel("Scaled Age")
plt.ylabel("Scaled Fare")
plt.show()

# Box Plot: Age distribution across Passenger Classes
plt.figure(figsize=(8, 6))
sns.boxplot(x='class', y='age', data=df)
plt.title("Boxplot of Age by Class")
plt.show()
