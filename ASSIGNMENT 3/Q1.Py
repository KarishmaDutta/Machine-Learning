import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


np.random.seed(42)

# Number of records
n = 100


df = pd.DataFrame({
    "Age": np.random.randint(22, 60, n).astype(float),
    "Salary": np.random.normal(65000, 15000, n).round(-2),
    "Department": np.random.choice(
        ["HR", "Engineering", "Marketing", "Sales"], n
    ),
    "Years_Of_Experience": np.random.randint(1, 35, n).astype(float)
})


missing_rate = 0.15

for column in df.columns:
    missing_rows = np.random.random(n) < missing_rate
    df.loc[missing_rows, column] = np.nan


print("--- Raw Dataset ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isna().sum())


numeric_columns = [
    "Age",
    "Salary",
    "Years_Of_Experience"
]

categorical_columns = ["Department"]

num_imputer = SimpleImputer(strategy="median")
df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])


cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_columns] = cat_imputer.fit_transform(
    df[categorical_columns]
)

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


encoder = OneHotEncoder(
    drop="first",
    sparse_output=False
)

encoded = encoder.fit_transform(df[categorical_columns])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_columns),
    index=df.index
)


df = df.drop(columns=categorical_columns)


df_processed = pd.concat([df, encoded_df], axis=1)


print("\n--- Preprocessed Dataset ---")
print(df_processed.head())


