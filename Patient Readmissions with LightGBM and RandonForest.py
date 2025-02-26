import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Dataset
data_path = r"C:\Users\ishit\Downloads\hospital_readmissions\hospital_readmissions_modified.csv"
df = pd.read_csv(data_path)

# Target variable to binary
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == 'yes' else 0)

# Drop original target column
df.drop(columns=['readmitted'], inplace=True)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Label Encoding for categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for future reference

# Separate features and target variable
X = df.drop(columns=['readmitted_binary'])
y = df['readmitted_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# StandardScaler on Numerical Features
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# LightGBM Model
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1
}

lgb_model = lgb.train(params, train_data, valid_sets=[valid_data], valid_names=['valid'], num_boost_round=200)

# Predictions - LightGBM
y_pred_lgb = (lgb_model.predict(X_test) > 0.5).astype(int)
y_pred_proba_lgb = lgb_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predictions - Random Forest
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Compare model performance
results = pd.DataFrame({
    "Model": ["LightGBM", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, y_pred_lgb), accuracy_score(y_test, y_pred_rf)],
    "AUC-ROC": [roc_auc_score(y_test, y_pred_proba_lgb), roc_auc_score(y_test, y_pred_proba_rf)],
    "Precision": [precision_score(y_test, y_pred_lgb), precision_score(y_test, y_pred_rf)],
    "Recall": [recall_score(y_test, y_pred_lgb), recall_score(y_test, y_pred_rf)]
})

print(Fore.GREEN + "\nComparison of Model Performance:")
print(Style.RESET_ALL)
print(results)

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# LightGBM Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_lgb), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Confusion Matrix - LightGBM")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("Confusion Matrix - Random Forest")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Model Comparison 
plt.figure(figsize=(8,5))
results.set_index("Model").plot(kind="bar", figsize=(8,5), colormap="viridis")
plt.title("Model Comparison: LightGBM vs Random Forest")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="best")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
