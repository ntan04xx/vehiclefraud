import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import xgboost as xgb

# Data setup
fraud_df = pd.read_csv('fraud.csv')
fraud_encoded = pd.get_dummies(fraud_df, drop_first=True)

target = fraud_encoded['FraudFound_P']
cov = fraud_encoded[['Sex_Male', 'Age', 'AgeOfVehicle_3 years', 'AgeOfVehicle_4 years', 'AgeOfVehicle_5 years', 'AgeOfVehicle_6 years', 'AgeOfVehicle_7 years', 'AgeOfVehicle_more than 7', 'AgeOfVehicle_new', 'AccidentArea_Urban', 'DriverRating', 'WitnessPresent_Yes', 'Year']]
cov_train, covd_test, target_train, target_test = train_test_split(cov, target, test_size=0.2, random_state=23)

# Creating models
model = LogisticRegression(max_iter=10000) # This model has highest accuracy of 94%
model.fit(cov_train, target_train)

fraud_pred = model.predict(covd_test)

ridge_model = Ridge(alpha = 10)
ridge_model.fit(cov_train, target_train)

lasso_model = Lasso(alpha = 10)
lasso_model.fit(cov_train, target_train)

dtrain = xgb.DMatrix(cov_train, label = target_train)
dtest = xgb.DMatrix(covd_test, label = target_test)
params = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'learning_rate': 0.1,
    'eval_metric': 'logloss'
}

xgb_model = xgb.XGBClassifier(**params)

# Find cross validation accuracy
k_folds = KFold(10)
scores = cross_val_score(model, cov, target, cv = k_folds)
print(scores.mean())

ridge_scores = cross_val_score(ridge_model, cov, target, cv = k_folds)
print(ridge_scores.mean())

lasso_scores = cross_val_score(lasso_model, cov, target, cv = k_folds)
print(lasso_scores.mean())

xgb_scores = cross_val_score(xgb_model, cov, target, cv = k_folds)
print(xgb_scores.mean())
