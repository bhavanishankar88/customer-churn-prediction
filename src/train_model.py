import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

df = pd.read_csv('data/synthetic_churn_data.csv')

print("Data Loaded:")
print("shape:",df.shape)
print("churn rate:", round(df['churn'].mean()*100,2), "%\n")

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

os.makedirs('models',exist_ok=True)
joblib.dump(model, 'models/churn_model.pkl')

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

print("Model Training completed!")
print(f"Accuracy: {accuracy*100:.2f}%")
print("\n classification report:")
print(classification_report(y_test,y_pred))

coefficients = model.coef_[0]
features = X.columns

print("\nFeature Importance:")

for feature,coef in zip(features,coefficients):
    print(f"{feature}: {coef:.4f}")