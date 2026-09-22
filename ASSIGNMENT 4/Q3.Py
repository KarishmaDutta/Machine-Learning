import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score



dataset = fetch_california_housing()


data = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

data["Price"] = dataset.target


features = data[["MedInc", "AveRooms"]]
target = data["Price"]



X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42
)



linear = LinearRegression()
linear.fit(X_train, y_train)

linear_output = linear.predict(X_test)

linear_score = r2_score(
    y_test,
    linear_output
)



transformer = PolynomialFeatures(
    degree=3,
    include_bias=True
)

train_poly = transformer.fit_transform(X_train)
test_poly = transformer.transform(X_test)

polynomial = LinearRegression()
polynomial.fit(train_poly, y_train)

polynomial_output = polynomial.predict(test_poly)

polynomial_score = r2_score(
    y_test,
    polynomial_output
)




print("=" * 50)
print("REGRESSION MODEL COMPARISON")
print("=" * 50)

print(f"Linear Regression R²     : {linear_score:.4f}")
print(f"Polynomial Regression R² : {polynomial_score:.4f}")

print("\n" + "-" * 50)
print("Comparison")
print("-" * 50)

models = {
    "Linear Regression": linear_score,
    "Polynomial Regression": polynomial_score
}

for name, score in models.items():
    print(f"{name:<25} : {score:.4f}")

print("-" * 50)

if polynomial_score > linear_score:
    print("Polynomial model has the larger R² value.")
elif linear_score > polynomial_score:
    print("Linear model has the larger R² value.")
else:
    print("Both models have the same R² value.")


