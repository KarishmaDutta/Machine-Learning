import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine


sns.set_style("whitegrid")


wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)


df["target"] = wine.target
df["class_name"] = [wine.target_names[i] for i in wine.target]



print("=== First Five Records ===")
print(df.iloc[:5])

print("\n=== Dataset Dimensions ===")
rows, columns = df.shape
print("Rows:", rows)
print("Columns:", columns)

print("\n=== Data Types & Non-Null Counts ===")
df.info()

print("\n=== Missing Values ===")
print(df.isna().sum())

print("\n=== Summary Statistics ===")
summary = df.describe().T
print(summary.loc[:, ["mean", "std", "min", "50%", "max"]])

print("\n=== Target Class Distribution ===")
print(df.groupby("class_name").size())


plt.figure(figsize=(12, 8))

numeric_data = df.select_dtypes(include="number")

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidth=0.5
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()



plt.figure(figsize=(6, 4))

sns.barplot(
    x=df["class_name"].value_counts().index,
    y=df["class_name"].value_counts().values,
    hue=df["class_name"].value_counts().index,
    palette="Set2",
    legend=False
)

plt.title("Wine Class Count Distribution")
plt.xlabel("Wine Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(
    data=df,
    x="class_name",
    y="alcohol",
    hue="class_name",
    palette="Set2",
    legend=False,
    ax=ax[0]
)

ax[0].set_title("Alcohol Level by Wine Class")
ax[0].set_xlabel("Wine Class")
ax[0].set_ylabel("Alcohol")

sns.boxplot(
    data=df,
    x="class_name",
    y="flavanoids",
    hue="class_name",
    palette="Set2",
    legend=False,
    ax=ax[1]
)

ax[1].set_title("Flavanoid Content by Wine Class")
ax[1].set_xlabel("Wine Class")
ax[1].set_ylabel("Flavanoids")

plt.tight_layout()
plt.show()
