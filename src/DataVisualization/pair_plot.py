import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PairPlot:
    """
    Display pairwise relationships between the numeric columns of a dataset.
    """

    def plot(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a Pandas DataFrame.")
        if data.empty:
            raise ValueError("The dataset is empty.")

        hue = "Hogwarts House" if "Hogwarts House" in data.columns else None

        # Select only numeric columns (no infinite values)
        # and drop columns with all NaN values
        features = data.select_dtypes(include="number")
        features = features.replace([float("inf"), float("-inf")], float("nan"))
        features = features.dropna(axis=1, how="all")

        if len(features.columns) < 2:
            raise ValueError(
                "At least two numeric columns are required to create a pair plot."
            )

        plot_data = features.copy()
        if hue:
            plot_data[hue] = data[hue]

        grid = sns.pairplot(
            plot_data,
            vars=features.columns,
            hue=hue,
            diag_kind="hist",
            corner=True,
            plot_kws={"alpha": 0.4, "s": 15},
            diag_kws={"alpha": 0.5}
        )

        # Improve spacing
        grid.fig.subplots_adjust(
            top=0.90,
            bottom=0.12,
            left=0.12,
            right=0.95,
            hspace=0.35,
            wspace=0.35,
        )

        grid.fig.suptitle("Pair Plot - Hogwarts House", y=1.02)
        plt.show()
        return self


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(f"{sys.argv[0]} : need .csv")

    df = pd.read_csv(sys.argv[1], index_col="Index")

    pair_plot = PairPlot()
    pair_plot.plot(df)
