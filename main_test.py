import pandas as pd
import numpy as np
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

# ---------------- Scatter plot -------------------- #
# -> Attendre de normaliser les donnees pour que
# les resultats aient plus de sens !!

# Calcul de la correlation la plus forte entre les features
scatter = Scatter()
feature_1, feature_2, correlation = scatter.find_strongest_correlation(train_transform)

print("Most similar features:")
print(f"{feature_1} and {feature_2}")

print("Pearson correlation coefficient:")
print(f"r = {correlation:.3f}")

scatter.plot(
    train_transform,
    feature_1,
    feature_2
)

# Plot the correlation matrix (additional for scatter plot)
scatter.plot_correlation_matrix(train_transform)
