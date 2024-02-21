import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Assuming 'credit_approval.csv' is your dataset file
df = pd.read_csv('credit_approval.csv')

# Summary statistics for continuous features
print(df.describe())

# Count of missing values per feature
print(df.isnull().sum())

plt.figure(figsize=(8, 6))
sns.histplot(df['A2'], kde=True, bins=20)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['A14'], kde=True, bins=20)
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='A7', data=df)
plt.title('Distribution of Employment Status')
plt.xlabel('Employment Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='A13', data=df)
plt.title('Credit History Distribution')
plt.xlabel('Credit History')
plt.ylabel('Count')
plt.show()

# Selecting continuous features for correlation analysis
continuous_features = df[['A2', 'A3', 'A8', 'A11', 'A14', 'A15']]  # Example, adjust based on your dataset

plt.figure(figsize=(10, 8))
corr = continuous_features.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Continuous Features')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='A16', data=df)
plt.title('Approval Status Distribution')
plt.xlabel('Approval Status')
plt.ylabel('Count')
plt.show()
