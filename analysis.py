import pandas as pd
import matplotlib.pyplot as plt

fraud_df = pd.read_csv('fraud.csv')
# Males have more fraud, higher incidence of claims and more willing to abuse the system
sex_data = fraud_df.groupby('Sex')['FraudFound_P'].mean().reset_index()
sex_data.plot(x = 'Sex', y = 'FraudFound_P')

# Fraud highest among teenagers who have lower maturity and elderly that are more paranoid and abusing system
age_data = fraud_df.groupby('Age')['FraudFound_P'].mean().reset_index()
age_data.plot(x = 'Age', y = 'FraudFound_P')

# 3-4 year old vehicles have highest fraud along with new ones, more physical risk for familiar vehicles 
vehicle_data = fraud_df.groupby('AgeOfVehicle')['FraudFound_P'].mean().reset_index()
vehicle_data.plot(x = 'AgeOfVehicle', y = 'FraudFound_P')

# Rural areas have more fraud, possibly due to lax culture and smaller community
area_data = fraud_df.groupby('AccidentArea')['FraudFound_P'].mean().reset_index()
area_data.plot(x = 'AccidentArea', y = 'FraudFound_P')

# Fraud highest with higher rating drivers who on average are inexperienced or have claimed more frequentlu
rating_data = fraud_df.groupby('DriverRating')['FraudFound_P'].mean().reset_index()
rating_data.plot(x = 'DriverRating', y = 'FraudFound_P')

plt.show()
