from src.DataManagement.clean import Clean


class DataPreprocessor:
    def __init__(self):
        self.transformers = [
            Clean()
        ]
        self.isFitted = False

    def fit(self, data):
        """
        Fit every transformer in the pipeline
        """

        current_data = data.copy()
        for transformer in self.transformers:
            transformer.fit(current_data)
            current_data = transformer.transform(current_data)
        self.isFitted = True
        return self

    def transform(self, data):
        """
        Apply every fitted transformation
        """
        if not self.isFitted:
            raise ValueError("You must call fit() before transform().")

        current_data = data.copy()
        for transformer in self.transformers:
            current_data = transformer.transform(current_data)
        return current_data

    def fit_and_transform(self, data):
        """
        Fit and transform directly the dataset
        """

        return self.fit(data).transform(data)

    def transform_reverse(self, data):
        """
        Reverse transformations in reverse order.
        """
        if not self.isFitted:
            raise ValueError("You must call fit() before transform_reverse().")

        current_data = data.copy()
        for transformer in self.transformers:
            current_data = transformer.transform_reverse(current_data)

        return current_data
