from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


housing_data = io.loadmat("data/housing_data.mat")
training_X = housing_data['Xtrain']
training_Y = housing_data['Ytrain']
validation_X = housing_data['Xvalidate']
validation_Y = housing_data['Yvalidate']

class linear_regression:
    def __init__(self, bias=False):
        self.bias = bias
        self.coefficient = None

    def fit(self, X, Y):
        if self.bias:
            X = np.insert(X, X.shape[1], 1, axis=1)
        a = np.dot(X.T, X)
        b = np.dot(X.T, Y)
        a_invese = np.linalg.inv(a)
        self.coefficient = np.dot(a_invese, b)

    def predict(self, X):
        if self.bias:
            X = np.insert(X, X.shape[1], 1, axis=1)
        return np.dot(X, self.coefficient)

    def RSS(self, X, Y):
        if self.bias:
            X = np.insert(X, X.shape[1], 1, axis=1)
        predicted = np.dot(X, self.coefficient)
        square_diff = (validation_Y - predicted) ** 2
        return np.sum(square_diff)



housing_classifier = linear_regression(bias=True)
housing_classifier.fit(training_X, training_Y)

rss = housing_classifier.RSS(validation_X, validation_Y)
print("The RSS of the housing validation set is %f" % rss)

predicted = housing_classifier.predict(validation_X)
min_price = min(predicted)
max_price = max(predicted)
print("The median home value range from %f to %f" % (min_price, max_price))


#### Plotting Coefficients
coefficients = plt.figure(1)
plt.title("Regression coefficients")
indices = range(len(housing_classifier.coefficient)-1)

plt.plot(indices, housing_classifier.coefficient[0:-1], linestyle='-', marker='o')
plt.xlabel('Indices', fontsize=14, color='blue')
plt.ylabel('Coefficient', fontsize=14, color='blue')
plt.grid(True)


#### Plotting residuals
residuals = plt.figure(2)
plt.title("Residuals")

residuals_val = housing_classifier.predict(validation_X) - validation_Y
plt.hist(residuals_val, 75, facecolor='green')

