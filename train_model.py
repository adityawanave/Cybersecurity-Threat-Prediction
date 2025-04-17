# train_model.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import shap
import joblib

# Load Dataset
df = pd.read_csv("cybersecurity_attacks.csv")
print("Original dataset shape:", df.shape)

# Drop unnamed column if exists
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Remove outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Define features and target
target_column = 'Threat_Level'  # <- Change if needed
X = df.drop(columns=[target_column])
y = df[target_column]

# Balance classes with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Feature selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(rf, n_features_to_select=5)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_].tolist()
print("Top 5 selected features:", selected_features)

X = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Training and Evaluation
results = {}
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    results[name] = {
        "Accuracy": acc,
        "ROC AUC": roc_auc
    }

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = model

# Save best model and scaler
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Best model and scaler saved!")

# Optional: SHAP feature importance
explainer = shap.Explainer(best_model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
