import matplotlib.pyplot as plt
import pandas as pd


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

        plt.figure(figsize=(8, 6))
        plt.scatter(
            data[x_col],
            data[y_col],
            alpha=0.6
        )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{x_col} vs {y_col}")

        plt.grid(True)
        plt.show()

        return self
