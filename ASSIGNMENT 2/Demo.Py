import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df = df.assign(target=iris.target)


print("First Five Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


print("\nMissing Values:")
print(df.isna().sum())


print("\nCorrelation Matrix:")
print(df.select_dtypes("number").corr())


sns.set_style("whitegrid")

sns.scatterplot(
    x=df["sepal length (cm)"],
    y=df["petal length (cm)"],
    hue=df["target"],
    palette="viridis"
)

plt.title("Sepal Length vs Petal Length by Class")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

sns.histplot(
    data=df,
    x="sepal length (cm)",
    kde=True,
    color="blue"
)

plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.show()
