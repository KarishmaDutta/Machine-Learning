import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine


wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)


fig, axes = plt.subplots(4, 4, figsize=(16, 12))


axes = axes.flatten()


for ax, column in zip(axes, df.columns):
    sns.boxplot(
        data=df,
        y=column,
        color="skyblue",
        ax=ax
    )

    ax.set_title(
        column,
        fontsize=10,
        fontweight="bold"
    )
    ax.set_ylabel("")


for ax in axes[len(df.columns):]:
    ax.set_visible(False)

fig.suptitle(
    "Boxplots for Numerical Attributes (Wine Dataset)",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.show()
