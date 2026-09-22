import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer


data = {
    "Age": [20, 21, None, 23, 24],
    "Income": [25000, 30000, 28000, None, 40000],
    "City": ["Kolkata", "Durgapur", "Kolkata", "Asansol", "Durgapur"],
    "Purchased": [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)


X = df.drop(columns="Purchased")
y = df["Purchased"]

num_cols = ["Age", "Income"]
cat_cols = ["City"]


num_pipeline = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler())
])

cat_pipeline = Pipeline([
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = make_column_transformer(
    (num_pipeline, num_cols),
    (cat_pipeline, cat_cols)
)

X_processed = preprocessor.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42
)


print("--- Original Dataset ---")
print(df)

print("\n--- Processed Feature Matrix Shape ---")
print(X_processed.shape)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


