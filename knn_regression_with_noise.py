"""
Project: KNN Regression with Noise
Author: Dogukan Somuncu
Date: 2025
Description:
    A regression project using K-Nearest Neighbors on synthetic data.
    The target values are based on a sine function to simulate non-linear patterns.
    Noise is added to demonstrate how KNN handles imperfect data.
    The script compares two weighting methods: 'uniform' and 'distance'.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
X = np.sort(5 * np.random.rand(40,1), axis = 0) # features
y = np.sin(X).ravel() # target based on a smooth non-linear function (sin used to simulate real-world curve)


#plt.scatter(X,y)

#add noise
y[::5] += 1 * (0.5 - np.random.rand(8))

#plt.scatter(X,y)
T = np.linspace(0, 5, 500)[:, np.newaxis]

# using enumerate to loop with both index and weight
for i, weight in enumerate(["uniform", "distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X,y).predict(T)
    
    plt.subplot(2,1, i+1) # show both plots in the same figure
    plt.scatter(X, y, color = "green", label = "data")
    plt.plot(T, y_pred, color = "blue", label = "prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weight = {}".format(weight))
    
plt.tight_layout()
plt.show()
#weight = "uniform"
#knn.fit(X,y).predict(T)
