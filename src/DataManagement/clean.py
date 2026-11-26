from src.DataManagement.data_transformer import DataTransformer
from src.DataManagement.stats import Stats
import pandas as pd


class Clean(DataTransformer):

    def __init__(self):
        self.fill_values = {}
        self.columns = None

    def fit(self, data):
        """
        Learn the replacement values for missing data.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a Pandas DataFrame.")

        if data.empty:
            raise ValueError("The dataset is empty.")

        if data.columns.duplicated().any():
            raise ValueError("Duplicate column names detected.")

        self.columns = data.columns.tolist()
        self.fill_values = {}
        my_stats = Stats(data=data)
        # describe_median = Describe().quartile_50()

        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                value = my_stats.median(column=column)
            else:
                mode = data[column].mode(dropna=True)
                if mode.empty:
                    raise ValueError(
                        f"Column '{column}' contains only missing values."
                    )
                value = mode.iloc[0]

            if pd.isna(value):
                raise ValueError(
                    f"Cannot calculate a replacement for '{column}'."
                )
            self.fill_values[column] = value
        return self

    def transform(self, data):
        """
        Replace missing values using the values learned in fit().
        """

        if self.columns is None:
            raise ValueError("You must call fit() before transform().")
        if data.empty:
            raise TypeError("The dataset is empty.")
        if data.columns.tolist() != self.columns:
            raise ValueError("The columns do not match the fitted dataset.")

        cleaned_data = data.copy()

        for column, value in self.fill_values.items():
            cleaned_data[column] = cleaned_data[column].fillna(value)

        return cleaned_data

    def transform_reverse(self):
        """
        Missing-value imputation is not directly reversible.
        """

        raise NotImplementedError(
            "Clean transformations cannot be reversed "
            "without preserving the original missing-value mask."
        )
