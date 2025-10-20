# === Diabetic Readmission Analysis with Feature Insights & Visualizations ===
# Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import os, warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")

# ---------- PARAMETERS ----------
DATA_PATH = "C:/Users/Dell/Downloads/diabetic_data_QMH_Club_Fest_2025.xlsx"
OUTPUT_DIR = "./QMH_visual_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
MISSING_THRESHOLD = 0.95
MAX_OHE_UNIQUE = 50

# ---------- 1. Load & Clean ----------
df = pd.read_excel(DATA_PATH)
print("Initial shape:", df.shape)

PLACEHOLDERS = ["None","none","NA","N/A","NaN","NULL","Not Available",
                "Not available","Not mapped","Not Mapped","?","other",
                "Other","Unknown","Invalid","unknown","invalid"]
df = df.replace(PLACEHOLDERS, np.nan)
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

irrelevant = ["V1","V2","V6","V28","V30","V37","V38","V39","V40","V41","V45","V46","V47"]
df.drop(columns=[c for c in irrelevant if c in df.columns], inplace=True, errors="ignore")

missing_frac = df.isna().mean()
high_missing = missing_frac[missing_frac >= MISSING_THRESHOLD].index
df.drop(columns=high_missing, inplace=True)
print(f"Dropped {len(high_missing)} columns with ≥95% missing.")

# ---------- 2. Target setup ----------
target_col = "V50"
valid_map = {">30":1, "<30":1, "NO":0}
df[target_col] = df[target_col].astype(str).str.strip().map(valid_map)
df = df.dropna(subset=[target_col])
y = df[target_col]
X = df.drop(columns=[target_col])

# ---------- 3. Feature split ----------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# ---------- 4. Handle high-cardinality categoricals ----------
ohe_cols, freq_cols = [], []
for c in cat_cols:
    (ohe_cols if X[c].nunique(dropna=True) <= MAX_OHE_UNIQUE else freq_cols).append(c)
for c in freq_cols:
    freq = X[c].value_counts(normalize=True)
    X[c] = X[c].map(freq).fillna(0.0)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# ---------- 5. Preprocessing ----------
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))])
col_trans = ColumnTransformer([("num", num_pipe, num_cols),
                               ("cat", cat_pipe, ohe_cols)])

# ---------- 6. Modeling pipeline ----------
pipeline = Pipeline([
    ("preproc", col_trans),
    ("var_thresh", VarianceThreshold(1e-4)),
    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=RANDOM_STATE))
])

# ---------- 7. Train/Test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("Train/Test:", X_train.shape, X_test.shape)

# ---------- 8. Fit model ----------
pipeline.fit(X_train, y_train)
print("Model fitted.")

# ---------- 9. Evaluate ----------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]
roc = roc_auc_score(y_test, y_prob)
print("\nROC AUC:", round(roc,3))
print(classification_report(y_test, y_pred))

# ---------- 10. Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# ---------- 11. ROC Curve ----------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc:.3f}")
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

# ---------- 12. Pie Chart (Readmission Ratio) ----------
plt.figure(figsize=(6,6))
plt.pie(y.value_counts(), labels=["No Readmission","Readmission"],
        autopct="%1.1f%%", startangle=90, colors=["lightgreen","tomato"])
plt.title("Overall Readmission Distribution")
plt.savefig(os.path.join(OUTPUT_DIR,"readmission_pie.png"))
plt.close()

# ---------- 13. Bar Chart (Categorical Feature Impact) ----------
for c in ohe_cols[:2]:
    plt.figure(figsize=(8,5))
    sns.barplot(x=c, y=target_col, data=df, estimator=np.mean, ci=None, palette="viridis")
    plt.xticks(rotation=45)
    plt.title(f"Average Readmission Rate by {c}")
    plt.ylabel("Readmission Rate")
    plt.savefig(os.path.join(OUTPUT_DIR, f"bar_{c}.png"))
    plt.close()

# ---------- 14. Feature Importance Extraction ----------
preproc = pipeline.named_steps["preproc"]
try:
    ohe = preproc.named_transformers_["cat"].named_steps["ohe"]
    ohe_names = list(ohe.get_feature_names_out(ohe_cols))
except Exception:
    ohe_names = []
feature_names = num_cols + ohe_names
support = pipeline.named_steps["var_thresh"].get_support()
final_feats = [f for f,s in zip(feature_names,support) if s]
coef = pipeline.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature":final_feats,"coef":coef})
coef_df["abscoef"] = coef_df["coef"].abs()
top3 = coef_df.sort_values("abscoef", ascending=False).head(3)
print("\nTop 3 influential features:\n", top3)

# ---------- 15. Interpret Each Top Feature ----------
print("\n--- Interpretations ---")
for _, row in top3.iterrows():
    direction = "increases" if row["coef"] > 0 else "decreases"
    print(f"• Higher values of **{row['feature']}** {direction} the probability of patient readmission.")

# ---------- 16. Scatter + Line Plot (Top 3 Features) ----------
X_test_plot = X_test.copy()
X_test_plot["pred_prob"] = y_prob

for f in top3["feature"]:
    if f in X_test_plot.columns:
        plt.figure(figsize=(7,5))
        sns.scatterplot(x=f, y="pred_prob", data=X_test_plot, alpha=0.4)
        sns.lineplot(x=f, y="pred_prob", data=X_test_plot, color="red", lw=2)
        plt.title(f"Predicted Readmission Probability vs {f}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_line_{f}.png"))
        plt.close()

# ---------- 17. Correlation Heatmap ----------
if len(num_cols) > 1:
    corr = X[num_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.savefig(os.path.join(OUTPUT_DIR,"correlation_heatmap.png"))
    plt.close()

print("\n✅ Analysis complete. Plots and insights saved in:", OUTPUT_DIR)

