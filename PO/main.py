import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####DONT FORGET TO BALANCE THE TARGET VARIABLE AFTER CHECKING CLASS WEIGHTS
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####

ds = pd.read_csv('packaging_data_v2.csv')
X = ds.iloc[:, 0:7]
y1 = ds.iloc[:, -2]
y2 = ds.iloc[:, -3]
y3 = ds.iloc[:, -1]
y4 = ds.iloc[:, -4]

print(y2.value_counts())
print(y1.value_counts())
print(y3.value_counts())
print(y4.value_counts())

print('\n****************predicting extra protective packaging**************** \n')
le1 = LabelEncoder()
y1 = le1.fit_transform(y1)

# standard scaler will be applied to all colums except the column with binary values
scaler = StandardScaler()
columns_to_scale = X.iloc[:, [0, 1, 2, 3, 6]]
scaled_columns = scaler.fit_transform(columns_to_scale)
X.iloc[:, [0, 1, 2, 3, 6]] = scaled_columns
print(X)

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2)


#X_test[:, list(range(0, 5)) + list(range(6, 10))] = scaler.transform(X_test[:, list(range(0, 5)) + list(range(6, 10))])
# X_test = pd.DataFrame(X_test, columns= ["length", "width", "height", "weight", "fragility", "atseal", "stime", "alloy", "plastic", "glass"])

class_weights = compute_class_weight('balanced', classes=np.unique(y1_train), y=y1_train)
class_weights_dict = dict(enumerate(class_weights))
print('class weights: ', class_weights_dict)
# y = le.fit_transform(y)
# print(X)
# print(y)

####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
# Using logistic regression to predict weather to use extra protection or not and in case of yes, which kind of extra protection

# X_test = pd.DataFrame(X_test, columns= ["length", "width", "height", "weight", "fragility", "atseal", "stime", "alloy", "plastic", "glass"])
class_weights = compute_sample_weight('balanced', y1_train)
classifier1 = RandomForestClassifier(n_estimators=20, criterion='gini', random_state=0)
classifier11 = XGBClassifier()
# nb = GaussianNB()
classifier1.fit(X_train, y1_train, sample_weight=class_weights)
classifier11.fit(X_train, y1_train, sample_weight=class_weights)

y1_pred = classifier1.predict(X_test)

cm = confusion_matrix(y1_test, y1_pred)

print("confusion matrix \n", cm)
print("\naccuracy score: ", accuracy_score(y1_test, y1_pred))
print(classification_report(y1_test, y1_pred))

print("xgb result on test set",classifier11.score(X_test, y1_test))
print("xgb result on training set",classifier11.score(X_train, y1_train))
print("random forest result on test set",classifier1.score(X_test, y1_test))
print("random forest result on training set",classifier1.score(X_train, y1_train))

y1_pred = le1.inverse_transform(y1_pred)

filename = 'predict_protection.sav'
pickle.dump(classifier1, open(filename, 'wb'))
filename = 'protection_encoder.sav'
pickle.dump(le1, open(filename, 'wb'))

# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# ####
# using the random forest classifier to predict the type of packaging to be used
print('\n****************predicting the type of packaging material that should be used**************** \n')
le2 = LabelEncoder()
y2 = le2.fit_transform(y2)

X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2)

class_weights = compute_class_weight('balanced', classes=np.unique(y2_train), y=y2_train)
class_weights_dict = dict(enumerate(class_weights))
print('class weights', class_weights_dict)

# X_test = pd.DataFrame(X_test, columns= ["length", "width", "height", "weight", "fragility", "atseal", "stime", "alloy", "plastic", "glass"])

classifier2 = XGBClassifier()
classifier22 = RandomForestClassifier(n_estimators=20, criterion='gini', random_state=0)
class_weights = compute_sample_weight('balanced', y2_train)
classifier2.fit(X_train, y2_train, sample_weight = class_weights)
classifier22.fit(X_train, y2_train, sample_weight= class_weights)

y2_pred = classifier22.predict(X_test) #confusion matrix from random forest

cm = confusion_matrix(y2_test, y2_pred)
print("confusion matrix \n",cm)
print("\naccuracy score: ",accuracy_score(y2_test, y2_pred))
print(classification_report(y2_test, y2_pred))

# scores of random forest classifier
print("random forest result on test set:",classifier22.score(X_test, y2_test))
print("random forest result on training set:",classifier22.score(X_train, y2_train))

# scores of decision tree classifier
print("xgboost result on test set:",classifier2.score(X_test, y2_test))
print("xgboost result on training set:",classifier2.score(X_train, y2_train))


y2_pred = le2.inverse_transform(y2_pred)

filename = 'predict_packaging.sav'
pickle.dump(classifier22, open(filename, 'wb'))
filename = 'packaging_encoder.sav'
pickle.dump(le2, open(filename, 'wb'))
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
#visualizing the predicted results
# ax1 = plt.subplot(2,2,1)
# plt.hist(y4_pred)
# plt.title("packaging material")

# ax1 = plt.subplot(2,2,2)
# plt.hist(y3_pred)
# plt.title("package thickness class-0,1,2")

# ax1 = plt.subplot(2,2,3)
# plt.hist(y2_pred)
# plt.title("extra protection-binary")

# ax1 = plt.subplot(2,2,4)
# plt.hist(y1_pred)
# plt.title("package fragility-0,1,2")

# plt.show()


#predict for user input

df = pd.DataFrame()

length = input("\nenter the length of machine part:")
width = input("\nenter the width of machine part:")
height = input("\nenter the height of machine part:")
weight = input("\nenter the weight of machine part:")
fragility = input("\nenter the fragility of machine part(0 for non fragile and 1 for fragile):")
atseal = input("\nenter weather or not machine part is atmospherically sealed (0 for no 1 for yes):")
stime = input("\nenter the storage time of machine part in days from 1 to 180:")

user_input  = [length, width, height, weight, fragility, atseal, stime]
user_input = pd.to_numeric(user_input)
series = pd.Series(user_input) 
user_input = df._append(series, ignore_index = True)
print(user_input)


columns_to_scale = user_input.iloc[:, [0, 1, 2, 3, 6]]
scaled_columns = scaler.transform(columns_to_scale)
user_input.iloc[:, [0, 1, 2, 3, 6]] = scaled_columns

print(user_input)


y1_user = classifier1.predict(user_input) #predicts the need for extra protectionls

y2_user = classifier22.predict(user_input) #predicts the packaging material to be used

print("packaging material to be used:", y2_user, le2.inverse_transform(y2_user))
print("extra protection:", y1_user, le1.inverse_transform(y1_user))

filename = 'scaler_model.sav'
pickle.dump(scaler, open(filename, 'wb'))