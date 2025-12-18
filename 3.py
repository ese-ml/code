# Install required libraries
!pip install autogluon.tabular seaborn matplotlib

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from autogluon.tabular import TabularDataset, TabularPredictor

# Load data
train = TabularDataset('/kaggle/input/tt/train.csv')
test = TabularDataset('/kaggle/input/tt/test.csv')

label = 'Class'

# -------------------------------
# Handle missing values explicitly
# -------------------------------
def handle_missing(df):
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Numeric → median
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Categorical → mode
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

print("Missing values before handling:")
print(train.isnull().sum().sort_values(ascending=False).head())

train = handle_missing(train)
test = handle_missing(test)

print("\nMax missing values after handling:", train.isnull().sum().max())

# -------------------------------
# Data Visualization (EDA)
# -------------------------------
numeric_cols = train.select_dtypes(include=[np.number]).columns

# Boxplots
plt.figure(figsize=(14, 6))
train[numeric_cols].boxplot(rot=90)
plt.title("Boxplots of Numeric Features")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = train[numeric_cols].corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=False,
    square=True,
    linewidths=0.5
)

plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# -------------------------------
# Train AutoGluon Model
# -------------------------------
predictor = TabularPredictor(
    label=label,
    problem_type='multiclass',
    eval_metric='accuracy'
).fit(
    train,
    hyperparameters={
        'GBM': {},
        'CAT': {},
        'XGB': {},
        'RF': {},
        'XT': {},
    },
    time_limit=900
)

# -------------------------------
# Predict and create submission
# -------------------------------
preds = predictor.predict(test)
probs = predictor.predict_proba(test)

print(predictor.leaderboard(silent=True))

submission = pd.DataFrame({
    'id': test['id'],
    label: preds
})

submission.to_csv('submission.csv', index=False)
print("submission.csv saved")
