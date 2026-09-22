import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

x = np.array([2, 4, 6, 8, 10, 12, 14, 16]).reshape(-1, 1)
y = np.array([42, 48, 55, 63, 70, 78, 85, 94])


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=10
)


regressor = LinearRegression()
regressor.fit(x_train, y_train)


predictions = regressor.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)


print("===== LINEAR REGRESSION RESULTS =====")
print("Coefficient (Slope):", round(regressor.coef_[0], 4))
print("Constant (Intercept):", round(regressor.intercept_, 4))

print("\n===== MODEL PERFORMANCE =====")
print("Mean Absolute Error :", round(mae, 2))
print("Mean Squared Error  :", round(mse, 2))
print("Root Mean Squared Error :", round(rmse, 2))
print("R-Squared Score :", round(r2, 4))


print("\n===== PREDICTIONS =====")
for actual, predicted in zip(y_test, predictions):
    print("Actual:", actual, " Predicted:", round(predicted, 2))


plt.figure(figsize=(8, 5))

plt.scatter(
    x,
    y,
    color="green",
    marker="o",
    label="Observed Data"
)

plt.plot(
    x,
    regressor.predict(x),
    color="orange",
    linewidth=2,
    label="Best Fit Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.legend()
plt.grid(alpha=0.4)

plt.show()
