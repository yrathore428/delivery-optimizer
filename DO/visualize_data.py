import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv("delivery_data.csv")
X = ds.iloc[:, :-2]
y1 = ds.iloc[:, -2]
y2 = ds.iloc[:, -1]

print(X.head())
print(y1.head())
print(y2.head())

# visualize distribution of target variables

print(type(y1))
print(type(y2))
print(type(X))



ax1 = plt.subplot(2,1,1)
plt.hist(y1)
plt.title('delivery ontime')

ax2 = plt.subplot(2,1,2)
plt.scatter(y2 ,X["destination_category"])
plt.title('predicted delivery days')

plt.show()
print(y1.describe())

