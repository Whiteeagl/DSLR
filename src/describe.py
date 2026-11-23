from dataclasses import dataclass, field
from sys import argv
import pandas as pd


@dataclass(kw_only=True)
class Describe:
    csv_file: str
    df_original: pd.DataFrame = field(init=False, repr=False, compare=False)
    df_numerical: pd.DataFrame = field(init=False, repr=False, compare=False)
    clean_data: dict[str, list[float]] = field(init=False)

    def __post_init__(self):
        """
        Initialise ce qui n'est pas défini dans le constructeur.
        Ici, tous les éléments déclarés avec field(init=False, ...)
        """

        try:
            # On extrait les données du CSV et on écarte la première colonne (index)
            self.df_original = pd.read_csv(self.csv_file, index_col=0)

        except (FileNotFoundError, pd.errors.EmptyDataError) as e:  # en cas d'erreur...
            raise ValueError(f"{self.csv_file} : {e}") from e

        # On récupère seulement les colonnes avec des valeurs numériques
        self.df_numerical = self.df_original.select_dtypes(include="number")

        # Si rien n'est trouvé -> erreur
        if self.df_numerical.empty:
            raise ValueError(f"{self.csv_file} has no numerical data.")

        # On définit une liste triée et nettoyée (sans les NaN) de notre dataset numérique
        self.clean_data = {
            c: sorted(vals)
            for c in self.df_numerical.columns
            if (vals := self.df_numerical[c].dropna().tolist())
        }

        if not self.clean_data:
            raise ValueError(f"{self.csv_file} has no usable numerical data.")

    @staticmethod
    def _percentile(elmt: list[float], p: float = 0.5) -> float:
        """
        Calcule les quartiles en fonction de p (le pourcentage de coupe).
        Utilise l'interpolation linéaire : A + (B - A) * t pour une précision optimale.
        """

        pos = p * (len(elmt) - 1)
        i = int(pos)
        t = pos - i

        if i + 1 >= len(elmt):
            return elmt[i]

        a, b = elmt[i], elmt[i + 1]
        return a + (b - a) * t

    def count(self) -> list[int]:
        """Compte combien d'éléments il y a dans chaque colonne"""
        return [len(v) for v in self.clean_data.values()]

    def mean(self) -> list[float]:
        """Calcule la moyenne des données de chaque colonne"""
        return [sum(v) / len(v) for v in self.clean_data.values()]

    def std(self) -> list[float]:
        """Calcule l'écart type, qui mesure la dispersion des données autour de la moyenne."""

        values = self.clean_data.values()
        result = []

        for v in values:
            n = len(v)

            if n < 2:
                result.append(float("nan"))
                continue

            # Calcule la moyenne (idem que self.mean mais sur la colonne v seulement)
            mu = sum(v) / n

            # Calcule la variance
            variance = sum(((_v - mu) ** 2) for _v in v) / (n - 1)

            # Calcule l'écart type (racine carrée de la variance)
            result.append(variance**0.5)

        return result

    def min(self) -> list[float]:
        """Retourne la liste de l'élément le plus petit de chaque colonne"""
        return [v[0] for v in self.clean_data.values()]

    def quartile_25(self) -> list[float]:
        """Retourne le quartile de 25 %"""
        return [Describe._percentile(v, p=0.25) for v in self.clean_data.values()]

    def quartile_50(self) -> list[float]:
        """Retourne le quartile de 50 %"""
        return [Describe._percentile(v, p=0.50) for v in self.clean_data.values()]

    def quartile_75(self) -> list[float]:
        """Retourne le quartile de 75 %"""
        return [Describe._percentile(v, p=0.75) for v in self.clean_data.values()]

    def max(self) -> list[float]:
        """Retourne la liste de l'élément le plus grand de chaque colonne"""
        return [v[-1] for v in self.clean_data.values()]

    def to_dataframe(self) -> pd.DataFrame:
        """Crée et renvoie un nouveau DataFrame pandas à partir des métriques calculées"""

        stats = {
            "Count": self.count(),
            "Mean": self.mean(),
            "Std": self.std(),
            "Min": self.min(),
            "25%": self.quartile_25(),
            "50%": self.quartile_50(),
            "75%": self.quartile_75(),
            "Max": self.max(),
        }

        return pd.DataFrame(stats, index=self.clean_data.keys()).T


if "__main__" == __name__:
    if len(argv) != 2:
        raise ValueError(f"{argv[0]} : need .csv")

    d = Describe(csv_file=argv[1])

    # Afficher les données de notre Describe sur le même format que le natif
    print(d.to_dataframe())

    # Vrai describe de pandas
    print(d.df_numerical.describe())
