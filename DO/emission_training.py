import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

ds = pd.read_csv('emission_data.csv')
X = ds.iloc[:, :-4]
y = ds.iloc[:, -4]

print(X.head())
print(y.head())

X = X.drop(columns= ['customer_review', 'carrier_rank', 'atseal', 'traffic_level', 'road_condition'])
print(X.head())

le = LabelEncoder()
X['destination_category'] = le.fit_transform(X['destination_category'])


le2 = LabelEncoder()
X['mode'] = le.fit_transform(X['mode'])
le3 = LabelEncoder()
X['transport_vehicle'] = le.fit_transform(X['transport_vehicle'])
le4 = LabelEncoder()
X['fuel_type'] = le.fit_transform(X['fuel_type'])
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LinearRegression()
rf = RandomForestRegressor(max_depth= 20)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(y_pred)

# Calculate the absolute errors
errors = abs(y_pred - y_test)
print('\ndifferences', errors)
# Print out the mean absolute error (mae)
print('\nMean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('\nAccuracy:', round(accuracy, 2), '%.')

print('emission prediction result of random forest regressor on test set', rf.score(X_test, y_test))
print('emission prediction result of random forest regressor on training set', rf.score(X_train, y_train))
