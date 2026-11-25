import pandas as pd
from src.DataManagement.data_preprocessor import DataPreprocessor
from src.DataVisualization.scatter import Scatter

# Load dataset
csv_file = "datasets/dataset_train.csv"
data_load = pd.read_csv(csv_file, index_col=0)

print("Dataset information:")
print(data_load.info())

print("\nMissing values before cleaning:")
print(data_load.isnull().sum())

# Fit phase
preprocessor = DataPreprocessor()
preprocessor.fit(data_load)

# Transform phase
train_transform = preprocessor.transform(data_load)

# Details phase
print("Original data:")
print(data_load.head())

print("\nCleaned data:")
print(train_transform.head())

print("\nMissing values remaining:")
print(train_transform.isnull().sum())

assert train_transform.isnull().sum().sum() == 0

# Testing scatter plot
# -> Attendre de normaliser les donnees pour que
# les resultats aient plus de sens !!
scatter = Scatter()

scatter.plot(
    train_transform,
    "Potions",
    "Divination"
)
