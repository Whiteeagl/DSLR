import sys
import pandas as pd
import numpy as np

def get_dico(df: pd.DataFrame) -> dict:

    # 1. Sauvegarde la colonne des maisons avant de filtrer les nombres
    houses = df["Hogwarts House"]
    
    # 2. Ne garde que les colonnes numériques (les 13 cours)
    course_df = df.select_dtypes(include="number")

    # 3. Regroupe par maison, supprime les NaN de chaque colonne et convertit en liste
    grouped_df = course_df.groupby(houses).agg(lambda col: col.dropna().tolist())

    # 4. Transforme le tableau (lignes=maisons, colonnes=cours) en dict {Cours: {Maison: [notes]}}
    return grouped_df

    

from DataManagement.normalize import Normalize
from DataManagement.stats import Stats

import matplotlib.pyplot as plt




def compute_std_from_mean(maison: str = None):

    df = df_group[maison].explode().astype(float).reset_index(name="score")
    

    test = Stats(data=df)
    n = Normalize(algo="standard").fit(test.stack())
    
    my = {
        k: Stats(pd.DataFrame(n.transform(data=df_group[maison][k]))).stack().mean()
        for k in df_group[maison].keys()
    }

    mean_stats = Stats(pd.DataFrame(list(my.values()))).stack()

    return mean_stats.std()




    """ fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 2. Premier graphique à gauche (ax1)
    ax1.hist(mG, bins=20, edgecolor="black")
    ax1.set_title("Avant normalisation")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Nombre d'élèves")

    # 3. Deuxième graphique à droite (ax2)
    ax2.hist(n.transform(data=mG), bins=20, edgecolor="black")
    ax2.set_title("Après normalisation")
    ax2.set_xlabel("Score normalisé")

    # 4. Ajuste automatiquement les marges pour éviter que les textes se chevauchent
    plt.tight_layout()
    plt.show() """

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(f"{argv[0]} : need .csv")


    df = pd.read_csv(sys.argv[1], index_col="Index")
    
    df_group = get_dico(df)
    _all_std = {
        k: compute_std_from_mean(k)
        for k in df_group.keys()
    }

    for k, v in _all_std.items():
        print(k, ": ", v)

    print("\nCours ayant la std la plus faible : ")
    print("\n 🥳​", min(_all_std, key=_all_std.get))



    courses_to_compare = ["Arithmancy", "Care of Magical Creatures"]
    house_colors = {
        "Gryffindor": "red",
        "Hufflepuff": "gold",
        "Ravenclaw": "blue",
        "Slytherin": "green",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, course in zip(axes, courses_to_compare):
        for house, color in house_colors.items():
            # Récupère les notes nettoyées de la maison pour ce cours
            scores = df[df["Hogwarts House"] == house][course].dropna()

            ax.hist(
                scores,
                bins=25,
                alpha=0.5,          # Transparence pour voir les superpositions
                label=house,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_title(course)
        ax.set_xlabel("Score")
        ax.set_ylabel("Nombre d'élèves")
        ax.legend()

    plt.tight_layout()
    plt.show()


