import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



data = fetch_california_housing()


housing_df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

housing_df["Price"] = data.target


print("===== DATASET PREVIEW =====")
print(housing_df.head())

print("\n===== DATASET SHAPE =====")
print("Rows:", housing_df.shape[0])
print("Columns:", housing_df.shape[1])



inputs = housing_df[["MedInc", "AveRooms"]]
output = housing_df["Price"]



X_train, X_test, y_train, y_test = train_test_split(
    inputs,
    output,
    test_size=0.20,
    random_state=42
)



regressor = LinearRegression()
regressor.fit(X_train, y_train)



predictions = regressor.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, predictions)



print("\n===== MODEL EVALUATION =====")
print(f"Mean Absolute Error  : {mae:.4f}")
print(f"Mean Squared Error   : {mse:.4f}")
print(f"Root Mean Squared Error : {rmse:.4f}")
print(f"R² Score             : {r_squared:.4f}")

print("\n===== MODEL COEFFICIENTS =====")
print(f"MedInc Coefficient   : {regressor.coef_[0]:.4f}")
print(f"AveRooms Coefficient : {regressor.coef_[1]:.4f}")
print(f"Intercept            : {regressor.intercept_:.4f}")



print("\n===== REGRESSION EQUATION =====")

equation = (
    f"Price = {regressor.intercept_:.4f} "
    f"+ ({regressor.coef_[0]:.4f} × MedInc) "
    f"+ ({regressor.coef_[1]:.4f} × AveRooms)"
)

print(equation)


income_value = float(input("\nEnter median income: "))
room_value = float(input("Enter average number of rooms: "))



new_data = pd.DataFrame(
    [[income_value, room_value]],
    columns=["MedInc", "AveRooms"]
)



estimated_price = regressor.predict(new_data)[0]

print(
    f"\nPredicted House Price: "
    f"${estimated_price * 100000:,.2f}"
)



plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions,
    color="green",
    alpha=0.5,
    edgecolors="black"
)


minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

