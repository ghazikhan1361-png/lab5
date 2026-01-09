# ============================================================
# LAB 05 — Treating a Classification Problem as ML Problem
# Compatible with Cursor IDE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import joblib

# ------------------------------------------------------------
# STEP 1 — Load CSV File
# ------------------------------------------------------------

csv_path = r"C:\Users\Admin\Desktop\Python Folder\sample-data.csv"
df = pd.read_csv(csv_path)

print("Dataset Loaded:", df.shape)
print(df.head())


# ------------------------------------------------------------
# STEP 2 — EDA
# ------------------------------------------------------------

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

sns.countplot(x='Survived', data=df)
plt.title("Survival Counts")
plt.show()


# ------------------------------------------------------------
# STEP 3 — Feature Engineering
# ------------------------------------------------------------

# Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Convert categorical to numeric
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


# ------------------------------------------------------------
# STEP 4 — Train Test Split + Scaling
# ------------------------------------------------------------

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
num_cols = ['Age', 'Fare', 'FamilySize']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# ------------------------------------------------------------
# STEP 5 — Train Multiple Models
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print("\n========== MODEL RESULTS ==========\n")

for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))


# ------------------------------------------------------------
# STEP 6 — Hyperparameter Tuning (Random Forest)
# ------------------------------------------------------------

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
}

rf = RandomForestClassifier(random_state=42)
gs = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")

print("\nFitting GridSearchCV...")
gs.fit(X_train, y_train)

print("\nBEST PARAMETERS:", gs.best_params_)


# ------------------------------------------------------------
# STEP 7 — Save Best Model
# ------------------------------------------------------------

joblib.dump(gs.best_estimator_, "titanic_best_model.joblib")
print("\n✔ Model saved as: titanic_best_model.joblib")
