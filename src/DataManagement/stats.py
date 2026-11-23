
from pandas import DataFrame, notna
from math import sqrt
from typing import Optional, Union


ColumnSelector = Optional[Union[str, list[str], tuple[str, ...]]]
ALL_COLUMNS = object()

class Stats:
    """
    Utilisation :

    mystat = Stats(mydataFrame)

    Statistiques globales :
        mystat.stack().min()
        mystat.stack().std()

    Une colonne :
        mystat.mean(column="Age")

    Plusieurs colonnes :
        mystat.mean(column=["Age", "Height"])

    Toutes les colonnes :
        mystat.mean(column=ALL_COLUMNS)


    Pour recuperer les noms de colonnes :

        names: list = self.keys()

    Pour recuperer les valeurs par colonnes :

        values: list = self.values()
    """

    def __init__(self, data: DataFrame):

        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        self.data = data
        self.num_data = data.select_dtypes(include="number")
        if self.num_data.empty:
            raise ValueError(f"{self.csv_file} has no numerical data.")

        # Cache des colonnes
        self._column_cache = {}
        self._sorted_cache = {}

        # Cache global
        self.cleaned_num_data = None

        for column in self.num_data.columns:

            values = [
                value
                for value in self.num_data[column].tolist()
                if notna(value)
            ]

            self._column_cache[column] = values

    # STACK
    def stack(self):

        if self.cleaned_num_data is None:

            self.cleaned_num_data = sorted([
                value
                for values in self._column_cache.values()
                for value in values
            ])

        return self

    # GET VALUES
    def _get_values(self, column=None):

        if column is None:

            if self.cleaned_num_data is None:
                raise RuntimeError("Call stack() before global statistics")

            values = self.cleaned_num_data

        else:

            if column not in self._column_cache:
                raise ValueError(f"Unknown numeric column: {column}")

            values = self._column_cache[column]

        if not values:
            raise ValueError("No valid numerical values available")

        return values

    # SORTED VALUES
    def _get_sorted(self, column=None):

        if column is None:
            return self._get_values()

        if column not in self._sorted_cache:
            self._sorted_cache[column] = sorted(
                self._get_values(column)
            )

        return self._sorted_cache[column]

    # COMPUTE
    def _compute(self, function, column=None):

        if column is ALL_COLUMNS:
            return {
                c: function(c)
                for c in self._column_cache
            }

        if isinstance(column, (list, tuple)):
            return {
                c: function(c)
                for c in column
            }

        return function(column)

    # MIN
    def min(self, column: ColumnSelector = None):

        def _min(c):
            values = self._get_sorted(c)
            return values[0]

        return self._compute(_min, column)

    # MAX
    def max(self, column: ColumnSelector = None):

        def _max(c):
            values = self._get_sorted(c)
            return values[-1]

        return self._compute(_max, column)

    # SUM
    def sum(self, column: ColumnSelector = None):

        def _sum(c):
            return sum(self._get_values(c))

        return self._compute(_sum, column)

    # COUNT
    def count(self, column: ColumnSelector = None):

        def _count(c):
            return len(self._get_values(c))

        return self._compute(_count, column)

    # MEAN
    def mean(self, column: ColumnSelector = None):

        def _mean(c):
            values = self._get_values(c)
            return sum(values) / len(values)

        return self._compute(_mean, column)

    # VARIANCE
    def var(self, column: ColumnSelector = None):

        def _var(c):
            values = self._get_values(c)
            mean = sum(values) / len(values)

            return sum(
                (x - mean) ** 2 for x in values
            ) / len(values)

        return self._compute(_var, column)

    # STD
    def std(self, column: ColumnSelector = None):

        def _std(c):
            return sqrt(self.var(c))

        return self._compute(_std, column)

    # RANGE
    def range(self, column: ColumnSelector = None):

        def _range(c):
            values = self._get_sorted(c)
            return values[-1] - values[0]

        return self._compute(_range, column)

    # QUANTILE
    def quantile(self, q: float, column: ColumnSelector = None):
        """Quantile avec interpolation linéaire."""

        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")

        def _quantile(c):

            values = self._get_sorted(c)

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

        return self._compute(_quantile, column)

    # MEDIAN
    def median(self, column: ColumnSelector = None):

        return self.quantile(0.5, column)


    def keys(self):

        return [k for k in self._column_cache.keys()]

    def values(self):

        return [v for v in self._column_cache.values()]

