import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

np.random.seed(42)

rows = 100

df = pd.DataFrame({
    "Age": np.random.randint(22, 60, rows).astype(float),
    "Salary": np.random.normal(65000, 15000, rows).round(-2),
    "Department": np.random.choice(
        ["HR", "Engineering", "Marketing", "Sales"],
        rows
    ),
    "Years_Of_Experience": np.random.randint(1, 35, rows).astype(float)
})


for column in df.columns:
    missing = np.random.rand(rows) < 0.15
    df.loc[missing, column] = np.nan


print("--- Raw Dataset ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isna().sum())


numeric_cols = ["Age", "Salary", "Years_Of_Experience"]
categorical_cols = ["Department"]


numeric_steps = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", MinMaxScaler())
])

categorical_steps = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(
        drop="first",
        sparse_output=False
    ))
])


transformer = ColumnTransformer([
    ("numbers", numeric_steps, numeric_cols),
    ("categories", categorical_steps, categorical_cols)
])


result = transformer.fit_transform(df)


new_category_names = transformer.named_transformers_[
    "categories"
].named_steps["encode"].get_feature_names_out(categorical_cols)


columns = numeric_cols + list(new_category_names)

clean_df = pd.DataFrame(result, columns=columns)


print("\n--- Preprocessed Dataset ---")
print(clean_df.head())



