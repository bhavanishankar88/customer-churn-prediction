import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load full dataset
df = pd.read_csv('data/customer_churn.csv')

print("Data Loaded!")
print("Shape:", df.shape)
print("Churn Rate:", round(df['churn'].mean()*100, 2), "%\n")

# Features and Target
X = df.drop('churn', axis=1)
y = df['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with better settings
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/churn_model.pkl')

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Training Completed!")
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
coefficients = model.coef_[0]
features = X.columns

print("\nFeature Importance:")
for feature, coef in zip(features, coefficients):
    print(f"{feature}: {coef:.4f}")