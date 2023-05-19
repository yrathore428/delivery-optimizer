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

ds = pd.read_csv('packaging_data.csv')
X = ds.iloc[:, 0:10]
y1 = ds.iloc[:, -1]
y2 = ds.iloc[:, -2]
y3 = ds.iloc[:, -3]
y4 = ds.iloc[:, -4]

# standard scaler will be applied to all colums except the column with binary values
scaler = StandardScaler()
columns_to_scale = X.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 9]]
scaled_columns = scaler.fit_transform(columns_to_scale)
X.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 9]] = scaled_columns
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


# Using the random forest classifier to predict the packaged fragility (class 0 or 1 or 2)
print("\n****************predicting the packaged fragility**************** \n")
classifier1 = KNeighborsClassifier(n_neighbors= 10, metric= 'minkowski', p=2)
classifier11 = RandomForestClassifier(n_estimators= 20, criterion= 'gini', random_state=0)
print(np.unique(y1_train))
class_weights = compute_sample_weight('balanced', y1_train)

classifier11.fit(X_train, y1_train, sample_weight=class_weights)
classifier1.fit(X_train, y1_train)

y1_pred = classifier11.predict(X_test)

cm = confusion_matrix(y1_test, y1_pred)
print("confusion matrix \n",cm)

print("\naccuracy score: ",accuracy_score(y1_test, y1_pred))

print(classification_report(y1_test, y1_pred))

# random forest scores
print("random forest result on test set",classifier11.score(X_test, y1_test))
print("random forest result on training set", classifier11.score(X_train, y1_train))

# decision tree scores
print("knn neighbors result on test set",classifier1.score(X_test, y1_test))
print("knn neighbors result on training set", classifier1.score(X_train, y1_train))
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
# Using logistic regression to predict weather to use extra protection or not (binary)
print('\n****************predicting the need for extra protective packaging**************** \n')
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2)

class_weights = compute_class_weight('balanced', classes=np.unique(y2_train), y=y2_train)
class_weights_dict = dict(enumerate(class_weights))
print('class weights: ', class_weights_dict)

# X_test = pd.DataFrame(X_test, columns= ["length", "width", "height", "weight", "fragility", "atseal", "stime", "alloy", "plastic", "glass"])
class_weights = compute_sample_weight('balanced', y2_train)
classifier2 = RandomForestClassifier(n_estimators=20, criterion='gini', random_state=0)
lr = LogisticRegression(C=1e2, solver="sag", random_state=0)
nb = GaussianNB()
classifier2.fit(X_train, y2_train, sample_weight=class_weights)
lr.fit(X_train, y2_train, sample_weight=class_weights)
y2_pred = classifier2.predict(X_test)
cm = confusion_matrix(y2_test, y2_pred)

print("confusion matrix \n", cm)
print("\naccuracy score: ", accuracy_score(y2_test, y2_pred))
print(classification_report(y2_test, y2_pred))

print("logistic regression result on test set",lr.score(X_test, y2_test))
print("logistic regression result on training set",lr.score(X_train, y2_train))
print("random forest result on test set",classifier2.score(X_test, y2_test))
print("random forest result on training set",classifier2.score(X_train, y2_train))

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
# Using the random forest classifier to predict the packaging thickness (class 0 or 1 or 2)
print('\n****************predicting the thickness class for packaging**************** \n')
X_train, X_test, y3_train, y3_test = train_test_split(X, y3, test_size=0.2)

class_weights = compute_class_weight('balanced', classes=np.unique(y3_train), y=y3_train)
class_weights_dict = dict(enumerate(class_weights))
print("class weights: ", class_weights)
# X_test = pd.DataFrame(X_test, columns= ["length", "width", "height", "weight", "fragility", "atseal", "stime", "alloy", "plastic", "glass"])


classifier3 = XGBClassifier()
classifier33 = RandomForestClassifier(n_estimators=20, criterion='gini', random_state=0)
class_weights = compute_sample_weight('balanced', y3_train)
classifier33.fit(X_train, y3_train, sample_weight=class_weights)
classifier3.fit(X_train, y3_train, sample_weight=class_weights)

y3_pred = classifier33.predict(X_test)

cm = confusion_matrix(y3_test, y3_pred) #confusion matrix from xgboost
print("confusion matrix \n", cm)
print("\naccuracy score: ",accuracy_score(y3_test, y3_pred))
print(classification_report(y3_test, y3_pred))

# random forest scores
print("xgboost result on test set",classifier33.score(X_test, y3_test))
print("xgboost result on training set",classifier33.score(X_train, y3_train))

# decision tree scores
print("random forest result on test set",classifier3.score(X_test, y3_test))
print("random forest result on training set",classifier3.score(X_train, y3_train))

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
le = LabelEncoder()
y4 = le.fit_transform(y4)

X_train, X_test, y4_train, y4_test = train_test_split(X, y4, test_size=0.2)

class_weights = compute_class_weight('balanced', classes=np.unique(y4_train), y=y4_train)
class_weights_dict = dict(enumerate(class_weights))
print('class weights', class_weights_dict)

# X_test = pd.DataFrame(X_test, columns= ["length", "width", "height", "weight", "fragility", "atseal", "stime", "alloy", "plastic", "glass"])

classifier4 = XGBClassifier()
classifier44 = RandomForestClassifier(n_estimators=20, criterion='gini', random_state=0)
class_weights = compute_sample_weight('balanced', y4_train)
classifier4.fit(X_train, y4_train, sample_weight = class_weights)
classifier44.fit(X_train, y4_train, sample_weight= class_weights)

y4_pred = classifier44.predict(X_test) #confusion matrix from random forest

cm = confusion_matrix(y4_test, y4_pred)
print("confusion matrix \n",cm)
print("\naccuracy score: ",accuracy_score(y4_test, y4_pred))
print(classification_report(y4_test, y4_pred))

# scores of random forest classifier
print("random forest result on test set:",classifier44.score(X_test, y4_test))
print("random forest result on training set:",classifier44.score(X_train, y4_train))

# scores of decision tree classifier
print("xgboost result on test set:",classifier4.score(X_test, y4_test))
print("xgboost result on training set:",classifier4.score(X_train, y4_train))


y4_pred = le.inverse_transform(y4_pred)
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

length = input("\nenter the length of machine part:")
width = input("\nenter the width of machine part:")
height = input("\nenter the height of machine part:")
weight = input("\nenter the weight of machine part:")
fragility = input("\nenter the fragility of machine part(0 to 4, 0 is least 4 is most):")
atseal = input("\nenter weather or not machine part is atmospherically sealed (0 for no 1 for yes):")
stime = input("\nenter the storage time of machine part in days from 1 to 180:")
alloy = input("\nenter the composition of machine part, percentage of alloy:")
plastic = input("\nenter the composition of machine part, percentage of plastic/polymer:")
glass = input("\nenter the composition of machine part, percentage of glass:")

X_pred = [length,width,height,weight,fragility,stime,alloy,plastic,glass]
# X_pred = pd.to_numeric(X_pred)
print(X_pred)

X_pred = scaler.transform(X_pred)


y1_user = classifier11.predict([X_pred])
y2_user = lr.predict([X_pred])
y3_user = classifier33.predict([X_pred])
y4_user = classifier44.predict([X_pred])

print("packaging material to be used:", le.inverse_transform(y4_user))
print("package thickness class:", y3_user)
print("extra protection:", y2_user)
print("package fragility:", y1_user)

