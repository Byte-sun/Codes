# ===============================================
# QMH 2025 Diabetic Data Analysis (Simplified)
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ---------------- Load & Basic Clean ----------------
df = pd.read_excel("C:/Users/Dell/Downloads/diabetic_data_QMH_Club_Fest_2025.xlsx", na_values=["?", "None", "Unknown"])
print("Shape:", df.shape)
print(df.head())

# ---------------- Target Setup ----------------
target = "V50"   # readmission_status
df = df[df[target].isin(["NO", "<30", ">30"])]  # keep only valid targets
df[target] = df[target].replace({">30": 1, "<30": 1, "NO": 0})

# ---------------- Missing Values Overview ----------------
missing = df.isna().mean().sort_values(ascending=False) * 100
print("\nTop Missing Columns:\n", missing.head(10))

plt.figure(figsize=(10,5))
missing[missing > 0].plot(kind='barh', color='salmon')
plt.title("Percentage of Missing Values by Column")
plt.xlabel("Percent Missing")
plt.show()

# ---------------- Simple Crosstab Analysis ----------------
# Relationship between age and readmission
ct_age = pd.crosstab(df["V5"], df[target], normalize='index') * 100
print("\nCrosstab: Readmission by Age Band (%):\n", ct_age)

ct_age.plot(kind='bar', stacked=True, figsize=(8,5), colormap="coolwarm")
plt.title("Readmission Rate by Age Band (%)")
plt.ylabel("Percentage")
plt.legend(["No Readmission", "Readmitted (<30/>30)"])
plt.xticks(rotation=45)
plt.show()

# ---------------- Correlation for numeric features ----------------
num_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# ---------------- Split Data ----------------
X = df.drop(columns=[target])
y = df[target]

# Separate numerical & categorical columns
num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

# ---------------- Preprocessing Pipelines ----------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features)
])

# ---------------- Model ----------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ---------------- Evaluation ----------------
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ---------------- Confusion Matrix ----------------
cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  index=["Actual No", "Actual Yes"], 
                  columns=["Pred No", "Pred Yes"])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Readmission Model")
plt.show()
