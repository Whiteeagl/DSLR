import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


class Scatter:

    def __init__(self):
        pass

    def plot(self, data, x_col, y_col):
        """
        Display a scatter plot for two features / columns
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a Pandas DataFrame.")

        if data.empty:
            raise ValueError("The dataset is empty.")
        if x_col not in data.columns:
            raise ValueError(f"Column '{x_col}' not found.")
        if y_col not in data.columns:
            raise ValueError(f"Column '{y_col}' not found.")
        if not pd.api.types.is_numeric_dtype(data[x_col]):
            raise TypeError(f"Column '{x_col}' must be numeric.")
        if not pd.api.types.is_numeric_dtype(data[y_col]):
            raise TypeError(f"Column '{y_col}' must be numeric.")

        plot_data = data[[x_col, y_col]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if len(plot_data) < 2:
            raise ValueError(
                "Not enough valid data points to create a scatter plot."
            )

        x = plot_data[x_col]
        y = plot_data[y_col]

        if x.nunique() < 2 or y.nunique() < 2:
            raise ValueError(
                "Both columns must have at least two unique values for a scatter plot."
            )

        correlation = x.corr(y)
        print(f"Correlation between '{x_col}' and '{y_col}': {correlation}")

        # Calculate regression line
        slope, intercept = np.polyfit(x, y, 1)
        regression_line = slope * x + intercept

        # Plot
        plt.figure(figsize=(9, 6))
        plt.scatter(
            x,
            y,
            alpha=0.6,
            label="Observations"
        )

        plt.plot(
            x,
            regression_line,
            color='red',
            label="Regression line",
            linewidth=2
        )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(
            f"{x_col} vs {y_col}\n"
            f"Pearson correlation: r = {correlation:.3f}"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return self

    def find_strongest_correlation(self, data):
        """
        Find the two features with the strongest correlation in the dataset
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a Pandas DataFrame.")

        if data.empty:
            raise ValueError("The dataset is empty.")

        correlation_matrix = data.corr(numeric_only=True, method='pearson')

        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        correlations = upper_triangle.stack().dropna().sort_values(ascending=False)
        if correlations.empty:
            raise ValueError(
                "No valid feature correlation could be calculated."
            )

        most_similar = correlations.abs().idxmax()
        feature_1, feature_2 = most_similar
        correlation_value = correlations.loc[most_similar]

        return feature_1, feature_2, correlation_value

    def plot_correlation_matrix(self, data):
        """
        Display a heatmap of the Pearson correlations matrix for the dataset

        This section isn’t required, but it allows us to see straight away
        which features would be worth keeping and which to ignore
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a Pandas DataFrame.")

        if data.empty:
            raise ValueError("The dataset is empty.")

        correlation_matrix = data.corr(numeric_only=True, method='pearson')

        plt.figure(figsize=(12, 9))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            linecolor='black'
        )

        plt.title("Correlation Matrix Heatmap")
        plt.tight_layout()
        plt.show()

        return self
