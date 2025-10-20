# ===============================================
# QMH 2025 Diabetic Data Analysis (Preprocess + Model + Plots)
# ===============================================
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ---------- PARAMETERS ----------
DATA_PATH = "C:/Users/Dell/Downloads/diabetic_data_QMH_Club_Fest_2025.xlsx"
OUTPUT_DIR = "./qmh_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
MISSING_THRESHOLD = 0.95
LOW_INFO_TOP_RATIO = 0.99
ENTROPY_THRESHOLD = 0.05
MAX_OHE_UNIQUE = 50
VAR_THRESH = 1e-4

# ---------- 1. Load ----------
df = pd.read_excel(DATA_PATH, na_values=["?", "None", "Unknown", "Not Available", "NULL"])
print("Initial shape:", df.shape)

# ---------- 2. Explicit initial drops (IDs / known irrelevant) ----------
irrelevant_cols = ["V1","V2","V6","V28","V30","V37","V38","V39","V40","V41","V45","V46","V47"]
present_irrelevant = [c for c in irrelevant_cols if c in df.columns]
if present_irrelevant:
    df.drop(columns=present_irrelevant, inplace=True)
    print(f"Dropped provided irrelevant columns ({len(present_irrelevant)}): {present_irrelevant}")

# ---------- 3. Normalize missing placeholders to np.nan ----------
PLACEHOLDERS = ["NONE","none","NA","N/A","NaN","Not mapped","Not Mapped","other","Other","Invalid","invalid"]
df = df.replace(PLACEHOLDERS, np.nan)

# Trim whitespace
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

# Show top missing columns
missing_frac = df.isna().mean().sort_values(ascending=False)
print("\nTop 10 columns by missing fraction:")
print(missing_frac.head(10))

# Save missing summary
missing_df = missing_frac.reset_index()
missing_df.columns = ["column", "missing_fraction"]
missing_df.to_csv(os.path.join(OUTPUT_DIR, "missing_summary.csv"), index=False)

# ---------- 4. Drop very high missingness ----------
high_missing_cols = missing_frac[missing_frac >= MISSING_THRESHOLD].index.tolist()
if high_missing_cols:
    df.drop(columns=high_missing_cols, inplace=True)
    print(f"\nDropped {len(high_missing_cols)} columns with >= {MISSING_THRESHOLD*100:.0f}% missing.")

# ---------- 5. Column-specific invalid-category removals ----------
invalid_category_map = {
    "V3": ["?", "other", "Other"],
    "V4": ["Unknown", "Invalid"],
    "V11": ["?"],
    "V12": ["?"],
    "V19": ["?"],
    "V20": ["?"],
    "V21": ["?"],
    "V23": ["None"],
    "V24": ["None"]
}
rows_before = len(df)
for col, bad_values in invalid_category_map.items():
    if col in df.columns:
        mask_bad = df[col].astype(str).isin(bad_values)
        n_bad = mask_bad.sum()
        if n_bad > 0:
            df = df.loc[~mask_bad]
            print(f"Removed {n_bad} rows with bad vals in {col}")
print(f"Rows removed by invalid-value filtering: {rows_before - len(df)} (remaining: {len(df)})")

# ---------- 6. Filter numeric code filters ----------
code_filters = {"V7":[5,6,8], "V8":[18,19,21,25,26], "V9":[9,15,17,20,21]}
for col, bad_codes in code_filters.items():
    if col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > 0:
            mask = coerced.isin(bad_codes)
            n_bad = mask.sum()
            if n_bad:
                df = df.loc[~mask]
                print(f"Filtered {col}: removed {n_bad} rows with codes {bad_codes}")

# ---------- 7. Convert diagnosis-coded features to string ----------
for c in ["V11","V12","V19","V20","V21"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# ---------- 8. Clean target ----------
target_col = "V50"
if target_col not in df.columns:
    raise KeyError("Target V50 not found.")
df[target_col] = df[target_col].astype(str).str.strip()
valid_map = {">30":1, "<30":1, "NO":0}
invalid_target_mask = ~df[target_col].isin(valid_map.keys())
if invalid_target_mask.sum() > 0:
    print(f"Warning: dropping {invalid_target_mask.sum()} rows with unexpected target values: {df.loc[invalid_target_mask, target_col].unique()}")
    df = df.loc[~invalid_target_mask]
df[target_col] = df[target_col].map(valid_map)

# ---------- 9. Build X and y ----------
y = df[target_col].copy()
X = df.drop(columns=[target_col])

# ---------- 10. Drop near-constant columns ----------
single_value_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if single_value_cols:
    X.drop(columns=single_value_cols, inplace=True)

# ---------- 11. Low-information drops ----------
n_rows = len(X)
low_info_cols = []
for c in X.columns:
    vc = X[c].value_counts(dropna=True)
    if vc.empty:
        low_info_cols.append(c); continue
    top_ratio = vc.iloc[0] / n_rows
    if top_ratio >= LOW_INFO_TOP_RATIO:
        low_info_cols.append(c)

def col_entropy(series):
    p = series.value_counts(normalize=True, dropna=True)
    return entropy(p, base=2) if len(p) > 1 else 0.0

entropies = X.apply(col_entropy)
low_entropy_cols = entropies[entropies < ENTROPY_THRESHOLD].index.tolist()
drop_low_info = sorted(set(low_info_cols + low_entropy_cols))
drop_low_info = [c for c in drop_low_info if c in X.columns]
if drop_low_info:
    X.drop(columns=drop_low_info, inplace=True)
    print(f"Dropped {len(drop_low_info)} low-information columns (examples): {drop_low_info[:20]}")

# ---------- 12. Prepare lists for pipeline ----------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"Final feature counts -> numeric: {len(num_cols)}, categorical: {len(cat_cols)}")

# ---------- 13. Cardinality handling ----------
ohe_cols, freq_cols = [], []
for c in cat_cols:
    nuniq = X[c].nunique(dropna=True)
    if nuniq <= MAX_OHE_UNIQUE:
        ohe_cols.append(c)
    else:
        freq_cols.append(c)
print(f"OHE columns: {len(ohe_cols)} ; Frequency-encoded columns: {len(freq_cols)}")

# Frequency-encode globally (simple approach)
for c in freq_cols:
    freq = X[c].value_counts(normalize=True)
    X[c] = X[c].map(freq).fillna(0.0)

# update numeric columns after freq-encoding
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# ---------- 14. ColumnTransformer and pipeline ----------
numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))])

transformers = []
if num_cols:
    transformers.append(("num", numeric_transformer, num_cols))
if ohe_cols:
    transformers.append(("cat_ohe", categorical_transformer, ohe_cols))
col_transformer = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

pipeline = Pipeline([
    ("preproc", col_transformer),
    ("var_thresh", VarianceThreshold(threshold=VAR_THRESH)),
    ("clf", LogisticRegression(penalty="l2", solver="saga", C=1.0, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE))
])

# ---------- 15. Train/test split ----------
X_full = X.copy(); y_full = y.copy()
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full)
print(f"Train/test sizes: {X_train.shape} / {X_test.shape}")

# ---------- 16. Fit pipeline ----------
pipeline.fit(X_train, y_train)
print("Pipeline fitted.")

# ---------- 17. FEATURE NAMES (after transforms) ----------
try:
    preproc = pipeline.named_steps["preproc"]
    feature_names = []
    if "num" in preproc.named_transformers_:
        feature_names += num_cols.copy()
    if "cat_ohe" in preproc.named_transformers_:
        ohe = preproc.named_transformers_["cat_ohe"].named_steps["ohe"]
        ohe_names = list(ohe.get_feature_names_out(ohe_cols))
        feature_names += ohe_names
except Exception as e:
    print("Warning while extracting feature names:", e)
    transformed = pipeline.named_steps["preproc"].transform(X_train)
    n_feats = transformed.shape[1]
    feature_names = [f"f_{i}" for i in range(n_feats)]

vt = pipeline.named_steps["var_thresh"]
support_mask = vt.get_support()
final_feature_names = [name for name, keep in zip(feature_names, support_mask) if keep]
print(f"Final number of features used by classifier: {len(final_feature_names)}")

# ---------- 18. Coefficients and top predictors ----------
coef = pipeline.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature": final_feature_names, "coef": coef}).assign(abscoef=lambda df_: df_.coef.abs()).sort_values("abscoef", ascending=False)
coef_df.to_csv(os.path.join(OUTPUT_DIR, "coef_importance.csv"), index=False)
print("\nTop 20 features by absolute coefficient:")
print(coef_df.head(20))

# ---------- 19. Evaluation ----------
y_pred = pipeline.predict(X_test)
unique_test_classes = np.unique(y_test)
if set(unique_test_classes) == {0,1}:
    y_prob = pipeline.predict_proba(X_test)[:,1]
    roc = roc_auc_score(y_test, y_prob)
    print("\nROC AUC:", roc)
else:
    roc = None
    print("\nWarning: test set missing a class in y_test. Test classes:", unique_test_classes)

print("\nClassification report (test):")
print(classification_report(y_test, y_pred))

# ---------- 20. Plot: Confusion matrix ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred No","Pred Yes"], yticklabels=["Actual No","Actual Yes"])
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), bbox_inches="tight")
plt.show()

# ---------- 21. Plot: ROC Curve ----------
if roc is not None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), bbox_inches="tight")
    plt.show()

# ---------- 22. Plot: Top model coefficients (feature importance) ----------
top_n = min(20, len(coef_df))
plt.figure(figsize=(8, max(4, top_n*0.3)))
sns.barplot(x="coef", y="feature", data=coef_df.head(top_n), orient="h")
plt.title("Top model coefficients (absolute importance ordering)")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_coefficients.png"), bbox_inches="tight")
plt.show()

# ---------- 23. Plot: Correlation heatmap (numeric features) ----------
num_for_corr = X.select_dtypes(include=[np.number]).columns.tolist()
if len(num_for_corr) >= 2:
    plt.figure(figsize=(10,8))
    corr = X[num_for_corr].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.show()

# ---------- 24. Regression-style plots (predicted prob vs numeric features) ----------
# Compute predicted probabilities for the full X_test (after pipeline transforms inside)
if set(unique_test_classes) == {0,1}:
    X_test_prep = X_test.copy()
    test_probs = pipeline.predict_proba(X_test)[:,1]
    X_test_prep["_pred_prob"] = test_probs
    
    # choose a few informative numeric features (if present)
    features_to_plot = []
    candidates = ["V10","V13","V14","V15","V16","V17","V18"]  # hospital_days, lab counts, med count, visits...
    for c in candidates:
        if c in X_test_prep.columns and np.issubdtype(X_test_prep[c].dtype, np.number):
            features_to_plot.append(c)
    features_to_plot = features_to_plot[:3]  # limit to 3 plots for brevity

    for feat in features_to_plot:
        plt.figure(figsize=(7,4))
        sns.regplot(x=feat, y="_pred_prob", data=X_test_prep, logistic=False, scatter_kws={"s":8, "alpha":0.4}, line_kws={"color":"red"})
        plt.ylim(-0.02,1.02)
        plt.title(f"Predicted readmission probability vs {feat}")
        plt.ylabel("Predicted probability of readmission")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"predprob_vs_{feat}.png"), bbox_inches="tight")
        plt.show()

# ---------- 25. Crosstab analyses and heatmaps ----------
# We'll compute percentage readmission (mean of target) by category for several pairs
crosstabs = {}
pairs_to_check = [
    ("V5","V7"),   # age band x admission type
    ("V5","V8"),   # age band x discharge type
    ("V7","V8"),   # adm type x discharge type
    ("V4","V50"),  # sex x readmission (single variable table)
    ("V3","V50"),  # ethnicity x readmission
]

# Single-variable percentage tables too
single_vars = ["V5","V7","V8","V3","V4"]

with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "crosstabs.xlsx")) as writer:
    # single variable readmission %
    for v in single_vars:
        if v in df.columns:
            tab = pd.crosstab(df[v], df[target_col], normalize="index")  # normalized per row
            tab.to_excel(writer, sheet_name=f"{v}_by_readmit")
            # heatmap
            plt.figure(figsize=(8, max(3, tab.shape[0]*0.25)))
            sns.heatmap(tab, annot=True, fmt=".2f", cmap="Reds")
            plt.title(f"Readmission % by {v}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"heat_{v}_readmit.png"), bbox_inches="tight")
            plt.show()

    # pairwise
    for a,b in [p for p in pairs_to_check if p[0] in df.columns and p[1] in df.columns]:
        # compute readmission rate grouped by the two categories -> pivot table
        pivot = df.groupby([a,b])[target_col].mean().unstack(fill_value=np.nan)  # mean = proportion readmitted
        pivot.to_excel(writer, sheet_name=f"{a}_vs_{b}")
        plt.figure(figsize=(10, max(4, pivot.shape[0]*0.25)))
        sns.heatmap(pivot, annot=False, cmap="Reds", linewidths=0.2)
        plt.title(f"Readmission rate (mean) by {a} and {b}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"heat_{a}_vs_{b}.png"), bbox_inches="tight")
        plt.show()

print(f"All outputs saved to {OUTPUT_DIR}")
