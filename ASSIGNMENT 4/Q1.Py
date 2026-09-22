import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

data = fetch_california_housing()


housing_df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)


housing_df["Price"] = data.target


print("===== DATASET PREVIEW =====")
print(housing_df.head())

print("\n===== DATASET SIZE =====")
print("Number of rows:", housing_df.shape[0])
print("Number of columns:", housing_df.shape[1])


features = housing_df[["AveRooms"]]
target = housing_df["Price"]


X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42
)


regression_model = LinearRegression()


regression_model.fit(X_train, y_train)


predicted_values = regression_model.predict(X_test)


mae_value = mean_absolute_error(y_test, predicted_values)
mse_value = mean_squared_error(y_test, predicted_values)
rmse_value = np.sqrt(mse_value)
r2_value = r2_score(y_test, predicted_values)


print("\n===== MODEL PERFORMANCE =====")
print(f"MAE  : {mae_value:.4f}")
print(f"MSE  : {mse_value:.4f}")
print(f"RMSE : {rmse_value:.4f}")
print(f"R²   : {r2_value:.4f}")


slope = regression_model.coef_[0]
intercept = regression_model.intercept_

print("\n===== REGRESSION EQUATION =====")
print(f"Price = {slope:.4f} × AveRooms + {intercept:.4f}")


room_count = float(
    input("\nEnter average number of rooms: ")
)


input_data = pd.DataFrame({
    "AveRooms": [room_count]
})

estimated_price = regression_model.predict(input_data)[0]

print(
    f"Estimated house price: "
    f"${estimated_price * 100000:,.2f}"
)


plt.figure(figsize=(8, 5))

plt.scatter(
    X_test["AveRooms"],
    y_test,
    color="blue",
    alpha=0.5,
    label="Actual Values"
)


plt.plot(
    X_test["AveRooms"],
    predicted_values,
    color="red",
    linewidth=2,
    label="Predicted Regression Line"
)

plt.xlabel("Average Number of Rooms")
plt.ylabel("House Price")
plt.title("California Housing Price Prediction")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

