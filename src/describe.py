from dataclasses import dataclass, field
from sys import argv
from pandas import read_csv, errors, DataFrame
from DataManagement.stats import Stats, ALL_COLUMNS


@dataclass(kw_only=True)
class Describe:
    csv_file: str
    df_original: DataFrame = field(init=False, repr=False, compare=False)
    clean_data: dict[str, list[float]] = field(init=False)

    def __post_init__(self):
        """
        Initialise ce qui n'est pas défini dans le constructeur.
        Ici, tous les éléments déclarés avec field(init=False, ...)
        """

        try:
            # On extrait les données du CSV et on écarte la première colonne (index)
            self.df_original = read_csv(self.csv_file, index_col=0)

        except (FileNotFoundError, errors.EmptyDataError) as e:  # en cas d'erreur...
            raise ValueError(f"{self.csv_file} : {e}") from e

        self.clean_data = Stats(self.df_original)


    def to_dataframe(self) -> DataFrame:
        """Crée et renvoie un nouveau DataFrame pandas à partir des métriques calculées"""

        stats = {
            "Count": self.clean_data.count(column=ALL_COLUMNS),
            "Mean": self.clean_data.mean(column=ALL_COLUMNS),
            "Std": self.clean_data.std(column=ALL_COLUMNS),
            "Min": self.clean_data.min(column=ALL_COLUMNS),
            "25%": self.clean_data.quantile(q=0.25, column=ALL_COLUMNS),
            "50%": self.clean_data.quantile(q=0.50, column=ALL_COLUMNS),
            "75%": self.clean_data.quantile(q=0.75, column=ALL_COLUMNS),
            "Max": self.clean_data.max(column=ALL_COLUMNS)
        }

        return DataFrame(stats, index=self.clean_data.keys()).T


if "__main__" == __name__:
    if len(argv) != 2:
        raise ValueError(f"{argv[0]} : need .csv")

    d = Describe(csv_file=argv[1])

    # Afficher les données de notre Describe sur le même format que le natif
    print(d.to_dataframe())

    # Vrai describe de pandas
    print(d.clean_data.num_data.describe())
