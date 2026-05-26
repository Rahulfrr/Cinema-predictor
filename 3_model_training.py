import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('imdb_cleaned_hit_flop.csv')

# Prepare features
X = df[['Released_Year', 'Runtime_mins', 'No_of_Votes', 'Meta_score']].copy()

# Convert Released_Year to numeric
X['Released_Year'] = pd.to_numeric(X['Released_Year'], errors='coerce')

y = df['Target'].copy()

# Handle missing values in Meta_score
X['Meta_score'] = X['Meta_score'].fillna(X['Meta_score'].median())

# Check for any remaining NaN
print("Missing values in features:")
print(X.isnull().sum())

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} movies")
print(f"Test set: {len(X_test)} movies")

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate model
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

print("\nTraining Set:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.3f}")
print(f"Precision: {precision_score(y_train, y_pred_train, pos_label='Hit'):.3f}")
print(f"Recall: {recall_score(y_train, y_pred_train, pos_label='Hit'):.3f}")
print(f"F1-Score: {f1_score(y_train, y_pred_train, pos_label='Hit'):.3f}")

print("\nTest Set:")
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, pos_label='Hit')
test_recall = recall_score(y_test, y_pred_test, pos_label='Hit')
test_f1 = f1_score(y_test, y_pred_test, pos_label='Hit')

print(f"Accuracy: {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall: {test_recall:.3f}")
print(f"F1-Score: {test_f1:.3f}")

# Feature importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test, labels=['Flop', 'Hit'])
print(cm)

# Save model
import pickle
with open('hit_flop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'hit_flop_model.pkl'")