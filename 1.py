import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("/kaggle/input/testtrain/train.csv")
test_df = pd.read_csv("/kaggle/input/testtrain/test.csv")

train_df = train_df.drop_duplicates()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

TARGET = "Class"

X = train_df.drop(TARGET, axis=1)
y = train_df[TARGET]

mask = y.notna()
X = X[mask]
y = y[mask]

print("Rows after removing missing target:", X.shape[0])

label_encoder = None
if y.dtype == "object":
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

empty_cols = X.columns[X.isnull().all()]
X = X.drop(empty_cols, axis=1)
test_df = test_df.drop(empty_cols, axis=1)

low_variance_cols = [c for c in X.columns if X[c].nunique() <= 2]
X = X.drop(columns=low_variance_cols)
test_df = test_df.drop(columns=low_variance_cols, errors="ignore")

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

plt.figure(figsize=(6, 4))
sns.countplot(x=train_df[TARGET])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(train_df.isnull(), cbar=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
X[numeric_cols].boxplot(rot=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
corr = X[numeric_cols].corr()
sns.heatmap(corr, cmap="coolwarm", square=True, linewidths=0.5)
plt.show()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

param_grid = {
    "model__max_depth": [14, 18, None],
    "model__min_samples_split": [5, 10],
    "model__min_samples_leaf": [2, 4]
}

search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=12,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

valid_preds = search.predict(X_valid)
print("Validation accuracy:", accuracy_score(y_valid, valid_preds))

best_model = search.best_estimator_
best_model.fit(X, y)

test_preds = best_model.predict(test_df)

if label_encoder is not None:
    test_preds = label_encoder.inverse_transform(test_preds)

submission = pd.DataFrame({
    "id": test_df["id"],
    TARGET: test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created")
