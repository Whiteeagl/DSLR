from abc import ABC, abstractmethod


class DataTransformer(ABC):

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def transform_reverse(self, data):
        pass
