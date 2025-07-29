import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load merged data from Task 1
data = pd.read_csv('data/processed/merged_data.csv')

# Convert timestamps to datetime and derive time_since_signup
data['purchase_time'] = pd.to_datetime(data['purchase_time'])
data['signup_time'] = pd.to_datetime(data['signup_time'])
data['time_since_signup'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds() / 3600

# Separate features and target, drop original datetime columns
X = data.drop(['class', 'purchase_time', 'signup_time'], axis=1)
y = data['class']

# Identify categorical and numeric columns
categorical_cols = ['country']
numeric_cols = ['ip_int', 'time_since_signup']

# Encode Categorical Features
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded_cols = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

# Combine encoded categorical with numeric data
X = pd.concat([X.drop(categorical_cols, axis=1), encoded_df], axis=1)

# Data Preparation: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle Class Imbalance with SMOTE (on training data only)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Original class distribution:", y.value_counts())
print("Resampled class distribution:", pd.Series(y_train_res).value_counts())

# Normalization and Scaling
scaler = StandardScaler()
X_train_res[numeric_cols] = scaler.fit_transform(X_train_res[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Model Selection and Training
# Logistic Regression (Baseline)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_res, y_train_res)
lr_pred = lr_model.predict(X_test)

# XGBoost (Ensemble)
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_res, y_train_res)
xgb_pred = xgb_model.predict(X_test)

# Evaluation
# Precision-Recall Curve
lr_probs = lr_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_probs)
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_probs)

plt.figure(figsize=(10, 5))
plt.plot(recall_lr, precision_lr, marker='.', label='Logistic Regression')
plt.plot(recall_xgb, precision_xgb, marker='.', label='XGBoost')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('notebooks/images/pr_curve.png')
plt.close()

# F1-Score
lr_f1 = f1_score(y_test, lr_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
print(f"Logistic Regression F1-Score: {lr_f1:.3f}")
print(f"XGBoost F1-Score: {xgb_f1:.3f}")

# Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_pred)
cm_xgb = confusion_matrix(y_test, xgb_pred)

plt.figure(figsize=(10, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Logistic Regression Confusion Matrix')
plt.savefig('notebooks/images/cm_lr.png')
plt.close()

plt.figure(figsize=(10, 5))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('XGBoost Confusion Matrix')
plt.savefig('notebooks/images/cm_xgb.png')
plt.close()



# SHAP Explainability
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary Plot
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
plt.savefig('notebooks/images/shap_summary.png')
plt.close()

# Force Plot for first test instance
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
plt.savefig('notebooks/images/shap_force.png')
plt.close()