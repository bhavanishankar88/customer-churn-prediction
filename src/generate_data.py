import numpy as np
import pandas as pd
import os

np.random.seed(42)

num_customers = 1000

tenure          = np.random.uniform(1,72,num_customers)
monthly_charges = np.random.uniform(25,120,num_customers)
contract_type   = np.random.choice([0,1,2],num_customers)
internet_service = np.random.choice([0,1,2],num_customers)
payment_method  = np.random.choice([0,1,2,3],num_customers)
senior_citizen  = np.random.choice([0,1],num_customers)
total_charges   = tenure * monthly_charges 

churn_prob = (0.4 + (contract_type == 0) * 0.3 + (tenure < 12) * 0.25 - (monthly_charges > 80) * 0.1)

churn = np.random.binomial(1,churn_prob.clip(0,1))

df = pd.DataFrame({
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'contract_type'  : contract_type,
    'internet_service' : internet_service,
    'payment_method' : payment_method,
    'senior_citizen' : senior_citizen,
    'churn'          : churn
})

os.makedirs('data', exist_ok = True)
df.to_csv('data/synthetic_churn_data.csv', index = False)

print(f"customer churn dataset created!")
print(f"Total Customers:{len(df)}")
print(f"Churned Customers:{df['churn'].mean()*100:.2f}%")
print(df.head())