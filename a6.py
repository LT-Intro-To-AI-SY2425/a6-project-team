import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("stuff.csv")
x = data["observation_date"].values
y = data["GFDEBTN"].values

# Create the model
# model = LinearRegression().fit(x,y)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 

# coef = round(float(model.coef_[0]), 2)
# intercept = round(float(model.intercept_), 2)
# r_squared = model.score(x, y)

# Predict the the blood pressure of someone who is 43 years old.
# Print out the prediction
# prediction = model.predict([[43]])
# x_predict = 43
# print()

# Print out the linear equation and r squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

# Testing the model
xtest = xtest.reshape(-1, 1)
predict = model.predict(xtest)
predict = np.around(predict, 2)

print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index]
    predicted_y = predict[index]
    x_coord = xtest[index]
    print("x value:", float(x_coord[0]), "Predicted y value:", predicted_y, "Actual y value:", actual)

# Plot the data and predictions
plt.figure(figsize=(6, 4))

# Scatter plot for training and testing data
plt.scatter(xtrain, ytrain, c="purple", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing Data")

# Scatter plot for predictions
plt.scatter(xtest, predict, c="red", label="Predictions")

# Label the axes
plt.title("debt over time")
plt.xlabel("Observation Date")
plt.ylabel("Debt")

# Create the model in matplotlib and include the line of best fit
plt.figure(figsize = (6,4))
plt.scatter(x,y, c="purple")
# plt.scatter(43, prediction, c = 'blue')



# plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

# show the plot and legend
plt.legend()
plt.show()
