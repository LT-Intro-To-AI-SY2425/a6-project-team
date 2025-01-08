import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("stuff.csv")
x = data["observation_date"].values
y = data["GFDEBTN"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)

# Create the model
model = LinearRegression().fit(x,y)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 

coef = round(float(model.coef_[0]), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

# # Predict the the blood pressure of someone who is 43 years old.
# # Print out the prediction
# prediction = model.predict([[43]])
# x_predict = 43
# print()

# # Print out the linear equation and r squared value
# print(f"Model's Linear Equation: y = {coef}x + {intercept}")
# print(f"R Squared value: {r_squared}")
# print(f"Prediction when x is {x_predict}: {prediction}")


# # Create the model in matplotlib and include the line of best fit
# plt.figure(figsize = (6,4))
# plt.scatter(x,y, c="purple")
# plt.scatter(43, prediction, c = 'blue')



# plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

# show the plot and legend
plt.legend()
plt.show()