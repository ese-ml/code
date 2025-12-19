import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

train=pd.read_csv("/kaggle/input/mock-test-2-mse-2/train.csv")
test=pd.read_csv("/kaggle/input/mock-test-2-mse-2/test.csv")

test_ids = test['id'].copy()

train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
train.info()
train.isnull().sum()
test.isnull().sum()

y = train["Status"]
train= train.drop("Status", axis=1)

y = pd.DataFrame(y)
le=LabelEncoder()
y["Status"]=le.fit_transform(y["Status"])

cat=[col for col in train.columns if train[col].dtype=='object']
num =[num for num in train.columns if num not in cat]
train[num].hist(figsize=(15, 12), bins=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
sns.boxplot(data=train[num])
plt.xticks(rotation=90)
plt.show()
print(cat)
print(num)
for col in num:
    median = train[col].median()
    train[col] = train[col].fillna(median)
    test[col] = test[col].fillna(median)

for col in cat:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    test[col] = test[col].fillna(mode)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

enc_train = ohe.fit_transform(train[cat])
enc_test = ohe.transform(test[cat])

enc_train_df = pd.DataFrame(enc_train, columns=ohe.get_feature_names_out(cat), index=train.index)
enc_test_df = pd.DataFrame(enc_test, columns=ohe.get_feature_names_out(cat), index=test.index)

train = pd.concat([train.drop(cat, axis=1), enc_train_df], axis=1)
test = pd.concat([test.drop(cat, axis=1), enc_test_df], axis=1)

plt.figure(figsize=(20,10))
sns.heatmap(train.corr(), annot=True, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

train.isnull().sum()

x_train,x_test,y_train,y_test = train_test_split(train, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': [100,200,150,300],
    'max_depth': [None, 10,12,15,18,20],
    'min_samples_split': [2,3,4,5,10],
    'min_samples_leaf': [2,4,5],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
}

rf = RandomForestClassifier(random_state=42)

tuner = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    n_jobs=-1,
    cv=5,
    verbose=2,
    random_state=42
)

tuner.fit(x_train, y_train)
print("Best Parameters:", tuner.best_params_)

best_rf = tuner.best_estimator_
y_pred = best_rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

best_rf.fit(train, y)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

importances = pd.Series(best_rf.feature_importances_, index=train.columns)
importances.sort_values(ascending=False).head(20).plot(kind='bar', figsize=(12,6))
plt.ylabel("Importance")
plt.show()

pred = best_rf.predict_proba(test) # for multiclass 2 
# pred = best_rf.predict(test) #for simple 1 

pred
pred = (pred == pred.max(axis=1, keepdims=True)).astype(int)
# in multiclass classification output is 0 and 1 2.2

submission = pd.DataFrame({
    'id': test_ids,
    'Status_C': pred[:, 0],
    'Status_CL': pred[:, 1],
    'Status_D': pred[:, 2]
})

submission.to_csv("submission.csv", index=False)

a=pd.read_csv("/kaggle/working/submission.csv")
print(a)
