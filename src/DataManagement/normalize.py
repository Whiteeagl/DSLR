from data_transformer import DataTransformer as DT
from stats import Stats
import numpy as np


class _MinMax(DT):

    def __init__(self):
        self._min = None
        self._max = None
        self._data = None
        self._data_transformed = None

    def fit(self, data: Stats) -> "_MinMax":
        if not isinstance(data, Stats):
            raise TypeError("<class: _MinMax>, <parameter: data> : data must be a Stats instance")

        self._data = np.asarray(data.cleaned_num_data, dtype=np.float64)
        self._min = data.min()
        self._max = data.max()
        self._data_transformed = None

        return self

    def transform(self, data=None) -> np.ndarray:
        arr = self._data if data is None else np.asarray(data, dtype=np.float64)
        self._data_transformed = (arr - self._min) / (self._max - self._min)
        return self._data_transformed

    def transform_reverse(self, data=None) -> np.ndarray:
        if data is None:
            if self._data_transformed is None:
                raise ValueError("<class: _MinMax> transform_reverse requires transform before")
            arr = self._data_transformed
            self._data_transformed = None
        else:
            arr = np.asarray(data, dtype=np.float64)

        return arr * (self._max - self._min) + self._min


class _Standard(DT):

    def __init__(self):
        self._mean = None
        self._std = None
        self._data = None
        self._data_transformed = None

    def fit(self, data: Stats) -> "_Standard":
        if not isinstance(data, Stats):
            raise TypeError("<class: _Standard>, <parameter: data> : data must be a Stats instance")

        self._data = np.asarray(data.cleaned_num_data, dtype=np.float64)
        self._mean = data.mean()
        self._std = data.std()
        self._data_transformed = None

        return self

    def transform(self, data=None) -> np.ndarray:
        arr = self._data if data is None else np.asarray(data, dtype=np.float64)
        self._data_transformed = (arr - self._mean) / self._std
        return self._data_transformed

    def transform_reverse(self, data=None) -> np.ndarray:
        if data is None:
            if self._data_transformed is None:
                raise ValueError("<class: _Standard> transform_reverse requires transform before")
            arr = self._data_transformed
            self._data_transformed = None
        else:
            arr = np.asarray(data, dtype=np.float64)

        return arr * self._std + self._mean


class _Robust(DT):

    def __init__(self):
        self._q25 = None
        self._q50 = None
        self._q75 = None
        self._data = None
        self._data_transformed = None

    def fit(self, data: Stats) -> "_Robust":
        if not isinstance(data, Stats):
            raise TypeError("<class: _Robust>, <parameter: data> : data must be a Stats instance")

        self._data = np.asarray(data.cleaned_num_data, dtype=np.float64)
        self._q25 = data.quantile(q=0.25)
        self._q50 = data.quantile(q=0.50)
        self._q75 = data.quantile(q=0.75)
        self._data_transformed = None

        return self

    def transform(self, data=None) -> np.ndarray:
        arr = self._data if data is None else np.asarray(data, dtype=np.float64)
        self._data_transformed = (arr - self._q50) / (self._q75 - self._q25)
        return self._data_transformed

    def transform_reverse(self, data=None) -> np.ndarray:
        if data is None:
            if self._data_transformed is None:
                raise ValueError("<class: _Robust> transform_reverse requires transform before")
            arr = self._data_transformed
            self._data_transformed = None
        else:
            arr = np.asarray(data, dtype=np.float64)

        return arr * (self._q75 - self._q25) + self._q50


class Normalize(DT):

    ALGORITHM = {
        "min-max": _MinMax,
        "standard": _Standard,
        "robust": _Robust,
    }

    def __init__(self, algo: str = "min-max"):
        if algo not in self.ALGORITHM:
            raise ValueError('<Class: Normalize> : <attribut: "algo"> : key not in dico')

        self.algo = algo
        self.strat: DT = self.ALGORITHM[algo]()
        self.fitted = False

    def fit(self, data: Stats) -> "Normalize":
        self.strat.fit(data=data)
        self.fitted = True
        return self

    def transform(self, data=None) -> np.ndarray:
        if not self.fitted:
            raise ValueError("<Class: Normalize> : transform requires fit(data) before.")
        return self.strat.transform(data=data)

    def transform_reverse(self, data=None) -> np.ndarray:
        if not self.fitted:
            raise ValueError("<Class: Normalize> : transform_reverse requires fit(data) before.")
        return self.strat.transform_reverse(data=data)
