# === Simplified, Visualization-rich Diabetic Readmission Analysis ===
# Requirements: scikit-learn, pandas, numpy, matplotlib, seaborn

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")

# ---------- PARAMETERS ----------
DATA_PATH = "C:/Users/Dell/Downloads/diabetic_data_QMH_Club_Fest_2025.xlsx"
OUTPUT_DIR = "./QMH_visual_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
MISSING_THRESHOLD = 0.95
MAX_OHE_UNIQUE = 50

# ---------- 1. Load Data ----------
df = pd.read_excel(DATA_PATH)
print("Initial shape:", df.shape)

# ---------- 2. Clean Missing/Invalid Data ----------
PLACEHOLDERS = ["None", "NONE", "none", "NA", "N/A", "NaN", "NULL",
                "Not Available", "Not available", "Not mapped", "Not Mapped",
                "?", "other", "Other", "Unknown", "Invalid", "unknown", "invalid"]
df = df.replace(PLACEHOLDERS, np.nan)

# Trim whitespace
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# ---------- 3. Drop Irrelevant Columns ----------
irrelevant_cols = ["V1","V2","V6","V28","V30","V37","V38","V39","V40","V41","V45","V46","V47"]
df.drop(columns=[c for c in irrelevant_cols if c in df.columns], inplace=True, errors="ignore")

# ---------- 4. Drop High Missingness ----------
missing_frac = df.isna().mean()
high_missing = missing_frac[missing_frac >= MISSING_THRESHOLD].index
df.drop(columns=high_missing, inplace=True)
print(f"Dropped {len(high_missing)} columns with high missingness.")

# ---------- 5. Target Encoding ----------
target_col = "V50"
valid_map = {">30":1, "<30":1, "NO":0}
df[target_col] = df[target_col].astype(str).str.strip().map(valid_map)
df = df.dropna(subset=[target_col])
y = df[target_col]
X = df.drop(columns=[target_col])

# ---------- 6. Separate Feature Types ----------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# ---------- 7. Handle High-cardinality Categoricals ----------
ohe_cols, freq_cols = [], []
for c in cat_cols:
    nuniq = X[c].nunique(dropna=True)
    (ohe_cols if nuniq <= MAX_OHE_UNIQUE else freq_cols).append(c)
for c in freq_cols:
    freq = X[c].value_counts(normalize=True)
    X[c] = X[c].map(freq).fillna(0.0)

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# ---------- 8. Preprocessing Pipelines ----------
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])
col_trans = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, ohe_cols)
])

# ---------- 9. Modeling Pipeline ----------
pipeline = Pipeline([
    ("preproc", col_trans),
    ("var_thresh", VarianceThreshold(threshold=1e-4)),
    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE))
])

# ---------- 10. Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=RANDOM_STATE)
print("Train/Test Split:", X_train.shape, X_test.shape)

# ---------- 11. Fit Model ----------
pipeline.fit(X_train, y_train)
print("Model trained successfully!")

# ---------- 12. Evaluate ----------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]
roc = roc_auc_score(y_test, y_prob)
print("\nROC AUC:", round(roc, 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------- 13. Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# ---------- 14. ROC Curve ----------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc:.3f}")
plt.plot([0,1], [0,1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

# ---------- 15. Pie Chart - Readmission Ratio ----------
plt.figure(figsize=(6,6))
plt.pie(y.value_counts(), labels=["No Readmission", "Readmission"],
        autopct="%1.1f%%", startangle=90, colors=["lightgreen", "tomato"])
plt.title("Overall Readmission Ratio")
plt.savefig(os.path.join(OUTPUT_DIR, "readmission_pie.png"))
plt.close()

# ---------- 16. Bar Chart - Categorical Distribution ----------
if "V7" in df.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x="V7", y=target_col, data=df, estimator=np.mean, ci=None, palette="viridis")
    plt.title("Average Readmission Rate by Admission Type (V7)")
    plt.ylabel("Readmission Rate")
    plt.savefig(os.path.join(OUTPUT_DIR, "bar_readmission_by_V7.png"))
    plt.close()

# ---------- 17. Scatter & Line Plot (Numeric vs Probability) ----------
num_for_plot = [c for c in num_cols if X_test[c].nunique() > 5][:3]  # pick 3 numeric features
X_test_plot = X_test.copy()
X_test_plot["pred_prob"] = y_prob

for c in num_for_plot:
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=c, y="pred_prob", data=X_test_plot, alpha=0.5)
    sns.lineplot(x=c, y="pred_prob", data=X_test_plot, color="red", lw=2)
    plt.title(f"Predicted Readmission Probability vs {c}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_line_{c}.png"))
    plt.close()

# ---------- 18. Correlation Heatmap ----------
num_for_corr = X[num_cols].select_dtypes(include=[np.number])
if num_for_corr.shape[1] > 1:
    corr = num_for_corr.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()

print("\n✅ Analysis complete! All plots saved to:", OUTPUT_DIR)
