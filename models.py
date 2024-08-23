import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

fraud_df = pd.read_csv('fraud.csv')
fraud_encoded = pd.get_dummies(fraud_df, drop_first=True)

target = fraud_encoded['FraudFound_P']
cov = fraud_encoded.drop('FraudFound_P', axis = 1)
cov_train, covd_test, target_train, target_test = train_test_split(cov, target, test_size=0.2, random_state=23)

model = LogisticRegression(max_iter=10000)
model.fit(cov_train, target_train)

fraud_pred = model.predict(covd_test)
print(fraud_pred)
