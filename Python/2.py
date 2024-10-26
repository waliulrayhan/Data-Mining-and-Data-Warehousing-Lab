import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading the petrol consumption dataset
petrol_data = pd.read_csv('petrol_consumption.csv')

# Splitting the data for 70:30
X_petrol = petrol_data.drop(columns=['Petrol_Consumption'])
y_petrol = petrol_data['Petrol_Consumption']

# 70:30 Split
X_train_70, X_test_70, y_train_70, y_test_70 = train_test_split(X_petrol, y_petrol, test_size=0.3, random_state=42)
# 80:20 Split
X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(X_petrol, y_petrol, test_size=0.2, random_state=42)

# Model for 70:30
lr_70 = LinearRegression()
lr_70.fit(X_train_70, y_train_70)
y_pred_70 = lr_70.predict(X_test_70)

# Model for 80:20
lr_80 = LinearRegression()
lr_80.fit(X_train_80, y_train_80)
y_pred_80 = lr_80.predict(X_test_80)

# b) Results comparison
print("70:30 MAE:", mean_absolute_error(y_test_70, y_pred_70))
print("80:20 MAE:", mean_absolute_error(y_test_80, y_pred_80))

# c) Scatter plot of actual vs predicted for 80:20
plt.scatter(y_test_80, y_pred_80)
plt.xlabel('Actual Petrol Consumption')
plt.ylabel('Predicted Petrol Consumption')
plt.title('Actual vs Predicted Petrol Consumption (80:20 Split)')
plt.show()

# d) Mean Absolute Error for 80:20 split
mae_80 = mean_absolute_error(y_test_80, y_pred_80)
print("Mean Absolute Error (80:20 Split):", mae_80)