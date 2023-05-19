import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ds = pd.read_csv('packaging_data.csv')
X = ds.iloc[:, 0:10]
y1 = ds.iloc[:, -1]
y2 = ds.iloc[:, -2]
y3 = ds.iloc[:, -3]
y4 = ds.iloc[:, -4]



print(X.describe())

# print(X.head())
# print(y1.head())
# print(y2.head())
# print(y3.tail())
# print(y4.tail())

ax1 = plt.subplot(2,2,1)
plt.hist(y4)
plt.title("packaging material")

ax2 = plt.subplot(2,2,2)
plt.hist(y3)
plt.title("package thickness class-0,1,2")

ax3 = plt.subplot(2,2,3)
plt.hist(y2)
plt.title("extra protection-binary")


ax4 = plt.subplot(2,2,4)
plt.hist(y1)
plt.title("package fragility-0,1,2")

plt.show()