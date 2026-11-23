
import pandas as pd
from math import sqrt
from typing import Optional


class Stats:
    """
    Utilisation :

    mystat = Stats(mydataFrame)

    Statistiques globales :
        _min = mystat.stack().min()
        _std = mystat.stack().std()

    Statistiques par colonne :
        _min = mystat.min(column="Age")
        _std = mystat.std(column="Age")
    """

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        self.data = data
        self.num_data = data.select_dtypes(include="number")

        # Cache des données par colonne
        self._column_cache = {}

        # Cache des données globales
        self.cleaned_num_data = None

        # Préparation des colonnes
        for column in self.num_data.columns:
            self._column_cache[column] = [
                value
                for value in self.num_data[column].tolist()
                if pd.notna(value)
            ]

    # STACK()
    def stack(self):
        """Prépare les données globales une seule fois."""

        if self.cleaned_num_data is None:
            self.cleaned_num_data = sorted([
                value
                for values in self._column_cache.values()
                for value in values
            ])

        return self

    # GET VALUES
    def _get_values(self, column=None):

        if column is not None:
            if column not in self._column_cache:
                raise ValueError(f"Unknown numeric column: {column}")

            values = self._column_cache[column]

        else:
            if self.cleaned_num_data is None:
                raise RuntimeError("Call stack() before global statistics")

            values = self.cleaned_num_data

        if not values:
            raise ValueError("No valid numerical values available")

        return values

    # MIN
    def min(self, column: Optional[str] = None):
        values = self._get_values(column)

        return values[0] if column is None else min(values)

    # MAX
    def max(self, column: Optional[str] = None):
        values = self._get_values(column)

        return values[-1] if column is None else max(values)

    # MEAN
    def mean(self, column: Optional[str] = None):
        values = self._get_values(column)

        return sum(values) / len(values)

    # STD
    def std(self, column: Optional[str] = None):
        values = self._get_values(column)

        mean = sum(values) / len(values)

        variance = sum(
            (x - mean) ** 2 for x in values
        ) / len(values)

        return sqrt(variance)

    # COUNT
    def count(self, column: Optional[str] = None):
        return len(self._get_values(column))

    # QUANTILE
    def quantile(self, q: float, column: Optional[str] = None):
        """Quantile avec interpolation linéaire."""

        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")

        values = self._get_values(column)

        # Les données globales sont déjà triées.
        # Les données d'une colonne doivent être triées.
        if column is not None:
            values = sorted(values)

        if len(values) == 1:
            return values[0]

        pos = (len(values) - 1) * q

        i = int(pos)
        t = pos - i

        if i == len(values) - 1:
            return values[i]

        a = values[i]
        b = values[i + 1]

        return a + (b - a) * t

    # MEDIAN
    def median(self, column: Optional[str] = None):
        return self.quantile(0.5, column)
