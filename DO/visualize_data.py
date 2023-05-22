import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv("delivery_data.csv")
X = ds.iloc[:, :-2]
y1 = ds.iloc[:, -2]
y2 = ds.iloc[:, -1]

Z = np.arange(1, 1001)

print(X.head())
print(y1.head())
print(y2.head())

# visualize distribution of target variables

print(type(y1))
print(type(y2))
print(type(X))

plt.scatter(Z, y2)
# plt.xlim([1, 1000])
plt.title('predicted delivery days')
plt.ylim([1, 28])
plt.show()

# take a smaller sample instead of plotting the whole 1000
# with that sample plot the prediction on an error bar graph
# prediction of days for delivery is still incomplete and accuracy is low!!!!
