import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()


df = pd.DataFrame(wine.data, columns=wine.feature_names)

correlation = df.corr()


plt.figure(figsize=(12, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap - Wine Dataset",
    fontsize=14,
    fontweight="bold",
    pad=15
)

plt.tight_layout()


plt.savefig("wine_correlation_heatmap.png", dpi=300)

plt.close()


highest_value = -1
feature_a = ""
feature_b = ""

columns = correlation.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        value = correlation.iloc[i, j]

        if value > highest_value:
            highest_value = value
            feature_a = columns[i]
            feature_b = columns[j]

print("=" * 50)
print("Correlation Analysis Results")
print("=" * 50)

print(
    f"Strongest Positive Correlation: "
    f"{feature_a} <--> {feature_b}"
)

print(
    f"Pearson Correlation Coefficient (r): "
    f"{highest_value:.4f}"
)

print("=" * 50)
print("Saved heatmap image to: wine_correlation_heatmap.png")

What is different?

Your original code uses:

np.fill_diagonal()


and:

corr_matrix.unstack().sort_values()


The alternative code uses nested for loops to compare every unique pair of features:

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):


This avoids checking the same pair twice. For example, it checks A → B but doesn't separately check B → A, since correlation is symmetric.
