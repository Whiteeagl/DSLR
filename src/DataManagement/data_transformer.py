from abc import ABC, abstractmethod


class DataTransformer(ABC):

    @abstractmethod
    def fit(self, data=None):
        pass

    @abstractmethod
    def transform(self, data=None):
        pass

    @abstractmethod
    def transform_reverse(self, data=None):
        pass
