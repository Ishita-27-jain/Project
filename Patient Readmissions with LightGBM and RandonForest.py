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
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import shap

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

# After model predictions and before polynomial features generation
# Add Feature Importance Analysis
plt.figure(figsize=(12, 6))
feature_imp = pd.DataFrame({
    'Feature': X.columns,  # Use original X instead of X_train
    'LightGBM': lgb_model.feature_importance(),
    'Random Forest': rf_model.feature_importances_
})

# Create subplots
plt.subplot(1, 2, 1)
feature_imp.nlargest(15, 'LightGBM').plot(
    x='Feature', y='LightGBM', kind='barh',
    title='Top 15 Features (LightGBM)'
)

plt.subplot(1, 2, 2)
feature_imp.nlargest(15, 'Random Forest').plot(
    x='Feature', y='Random Forest', kind='barh',
    title='Top 15 Features (Random Forest)'
)

plt.tight_layout()
plt.show()

poly = PolynomialFeatures(degree=2, interaction_only=True)
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
interaction_features = pd.DataFrame(
    poly.fit_transform(df[numerical_features]),
    columns=poly.get_feature_names_out(numerical_features)  # Updated this line
)
df = pd.concat([df, interaction_features.iloc[:, len(numerical_features)+1:]], axis=1)

# Then continue with the train-test split
X = df.drop(columns=['readmitted_binary'])
y = df['readmitted_binary']

# Add time-based features if admission_date exists
if 'admission_date' in df.columns:
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df['admission_day_of_week'] = df['admission_date'].dt.dayofweek
    df['admission_month'] = df['admission_date'].dt.month
    df['is_weekend'] = df['admission_day_of_week'].isin([5, 6]).astype(int)

# ... continue with existing preprocessing code until after model predictions ...

# After both models' predictions, add stacking
meta_features = pd.DataFrame({
    'lgb_pred': y_pred_proba_lgb,
    'rf_pred': y_pred_proba_rf
})

meta_classifier = LogisticRegression()
meta_classifier.fit(meta_features, y_test)

final_predictions = meta_classifier.predict(meta_features)
final_probabilities = meta_classifier.predict_proba(meta_features)[:, 1]

# Update results DataFrame to include stacked model
results = pd.DataFrame({
    "Model": ["LightGBM", "Random Forest", "Stacked Model"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lgb),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, final_predictions)
    ],
    "AUC-ROC": [
        roc_auc_score(y_test, y_pred_proba_lgb),
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, final_probabilities)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lgb),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, final_predictions)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lgb),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, final_predictions)
    ]
})

# ... existing performance comparison code ...
# Add Risk Score Analysis
def calculate_risk_score(probabilities):
    return pd.cut(probabilities, 
                 bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

risk_scores = pd.DataFrame({
    'LightGBM Risk': calculate_risk_score(y_pred_proba_lgb),
    'Random Forest Risk': calculate_risk_score(y_pred_proba_rf),
    'Stacked Model Risk': calculate_risk_score(final_probabilities)
})

plt.figure(figsize=(15, 5))
for i, col in enumerate(risk_scores.columns):
    plt.subplot(1, 3, i+1)
    risk_scores[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()

# Calculate potential cost savings
avg_readmission_cost = 10000  # Example cost in dollars
predicted_prevented = sum((y_test == 1) & (final_predictions == 0))
potential_savings = predicted_prevented * avg_readmission_cost
print(f"Potential cost savings: ${potential_savings:,}")

# Add SHAP values analysis for model interpretability
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
