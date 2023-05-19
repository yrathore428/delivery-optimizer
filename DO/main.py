import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pydot

ds = pd.read_csv('delivery_data.csv')
X = ds.iloc[:, :-3]
y1 = ds.iloc[:, -2]
y2 = ds.iloc[:, -1]
y3 = ds.iloc[:, -3]

# predicting y1: binary on whether delivery is ontime or not

le = LabelEncoder()
X['road_condition'] = le.fit_transform(X['road_condition'])

np.set_printoptions(threshold=sys.maxsize)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2)

# standard scalar will be applied to all the columns except those which have been onehot encoded, or the ones which are binary
sc = StandardScaler()
columns_to_scale = X_train[:, [7, 8, 9, 11, 12, 13]]
scaled_columns = sc.fit_transform(columns_to_scale)
X_train[:, [7, 8, 9, 11, 12, 13]] = scaled_columns

columns_to_scale = X_test[:, [7, 8, 9, 11, 12, 13]]
scaled_columns = sc.transform(columns_to_scale)
X_test[:, [7, 8, 9, 11, 12, 13]] = scaled_columns

classifier1 = RandomForestClassifier(n_estimators=20, criterion='gini', random_state=0)
classifier11 = DecisionTreeClassifier()
classifier1.fit(X_train, y1_train)
y1_pred = classifier1.predict(X_test)
print(y1_pred)

cm = confusion_matrix(y1_test, y1_pred)

print("confusion matrix \n", cm)
print("\n accuracy score: ", accuracy_score(y1_test, y1_pred))
print(classification_report(y1_test, y1_pred))

print('result on test set', classifier1.score(X_test, y1_test))
print('result on training set', classifier1.score(X_train, y1_train))


# predicting how many days it will take for delivery (regression)
# note: it makes more sense to use regression for number of days since time
# is a continuous variable and delivery can happen at any time

X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2)

sc = StandardScaler()
columns_to_scale = X_train[:, [7, 8, 9, 11, 12, 13]]
scaled_columns = sc.fit_transform(columns_to_scale)
X_train[:, [7, 8, 9, 11, 12, 13]] = scaled_columns

columns_to_scale = X_test[:, [7, 8, 9, 11, 12, 13]]
scaled_columns = sc.transform(columns_to_scale)
X_test[:, [7, 8, 9, 11, 12, 13]] = scaled_columns


pipeline = make_pipeline(StandardScaler(), Ridge(alpha=0.9))
lasso = Lasso(alpha = 0.09)
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor2 = RandomForestRegressor(max_depth= 5 ,n_estimators= 7, min_samples_split=11, max_leaf_nodes=14)
regressor.fit(X_poly, y2_train)
regressor2.fit(X_train, y2_train)
pipeline.fit(X_train, y2_train)
lasso.fit(X_train, y2_train)
y2_pred = regressor2.predict(X_test).round()
y22_pred = pipeline.predict(X_test)
y222_pred = regressor.predict(poly_reg.transform(X_test))


# print("accuracy score",accuracy_score(y2_test, y2_pred))
# cm = confusion_matrix(y2_test, y2_pred)
# print("confusion matrix: \n", cm)

print('result of random forest regressor on test set', regressor2.score(X_test, y2_test))
print('result of random forest regressor on training set', regressor2.score(X_train, y2_train))
print('result of ridge regression on test set', pipeline.score(X_test, y2_test))
print('result of ridge regression on training set', pipeline.score(X_train, y2_train))
# print('result of polynomial regression on test set', regressor.score(X_test, y2_test))
# print('result of polynomial regression on training set', regressor.score(X_train, y2_train))
print("\nscore of polyreg", r2_score(y2_test, y222_pred))
print('\npredictions (of regression model) and true value side by side, predictions first')
print(np.c_[(y2_pred), y2_test])
# Calculate the absolute errors
errors = abs(y2_pred - y2_test)
print('\ndifferences', errors)
# Print out the mean absolute error (mae)
print('\nMean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y2_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('\nAccuracy:', round(accuracy, 2), '%.')
# print(np.max(y2_train))



####trying CNNs to predict number of delivery days (categorical)

# num_classes = 27
# y_one_hot = tf.keras.utils.to_categorical(y2_train, num_classes=num_classes)
# print(tf.__version__)

# ann = tf.keras.models.Sequential()

# # input layer
# ann.add(tf.keras.layers.Dense(units = 64, activation='relu', input_dim = 9))

# # hidden layer
# ann.add(tf.keras.layers.Dense(units = 64, activation='relu'))

# # second hidden layer
# ann.add(tf.keras.layers.Dense(units=32, activation = 'relu'))

# # output layer
# ann.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# # compiling the ann
# ann.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# ann.fit(X_train, y_one_hot, batch_size=64, epochs = 500, verbose=False)

# y22_pred = ann.predict(X_test).round()

# score = ann.evaluate(X_train, y_one_hot, batch_size=32)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# np.set_printoptions(threshold=np.inf)
# print('\nprediction matrix, position of ones represent the nth day')
# print(y22_pred)
# print(y2_test)

# # issues with neural network, need to check it, max accuracy achieved is 90 percent on
# # regression model but overall accuracy is bad since regression is not the optimal option


X_train, X_test, y3_train, y3_test = train_test_split(X, y3, test_size=0.2)

sc = StandardScaler()
columns_to_scale = X_train[:, [7, 8, 9, 11, 12, 13]]
scaled_columns = sc.fit_transform(columns_to_scale)
X_train[:, [7, 8, 9, 11, 12, 13]] = scaled_columns

columns_to_scale = X_test[:, [7, 8, 9, 11, 12, 13]]
scaled_columns = sc.transform(columns_to_scale)
X_test[:, [7, 8, 9, 11, 12, 13]] = scaled_columns

regressor3 = RandomForestRegressor(max_depth= 5 ,n_estimators= 7, min_samples_split=11, max_leaf_nodes=14)
regressor3.fit(X_train, y3_train)
y3_pred = regressor3.predict(X_test)

print('cost prediction result of random forest regressor on test set', regressor3.score(X_test, y3_test))
print('cost prediction result of random forest regressor on training set', regressor3.score(X_train, y3_train))
print('\npredictions (of regression model) and true value side by side, predictions first')
print(np.c_[(y3_pred), y3_test])

# Calculate the absolute errors
errors = abs(y3_pred - y3_test)
print('\ndifferences', errors)
# Print out the mean absolute error (mae)
print('\nMean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y3_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('\nAccuracy:', round(accuracy, 2), '%.')