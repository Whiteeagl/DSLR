import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from macro import HOUSE_COLORS
from DataManagement.stats import Stats
from DataManagement.normalize import Normalize


# ==============================================================================
#                              PRÉPARATION DES DONNÉES
# ==============================================================================

def get_dico(df: pd.DataFrame) -> pd.DataFrame:
    """Regroupe les notes numériques par maison en supprimant les valeurs manquantes."""
    # 1. Sauvegarde la colonne des maisons avant de filtrer les nombres
    houses = df["Hogwarts House"]

    # 2. Ne garde que les colonnes numériques (les 13 cours)
    course_df = df.select_dtypes(include="number")

    # 3. Regroupe par maison, supprime les NaN de chaque colonne et convertit en liste
    grouped_df = course_df.groupby(houses).agg(lambda col: col.dropna().tolist())

    return grouped_df


# ==============================================================================
#                              ANALYSE STATISTIQUE
# ==============================================================================

def compute_std_from_mean(course: str = None) -> list[float]:
    """Calcule l'écart-type des moyennes et des dispersions entre les 4 maisons."""
    df_course = df_group[course].explode().astype(float).reset_index(name="score")

    stats_global = Stats(data=df_course)
    normalizer = Normalize(algo="standard").fit(stats_global.stack())

    house_metrics = {
        house: [s.mean(), s.std()]
        for house, scores in df_group[course].items()
        if (s := Stats(pd.DataFrame(normalizer.transform(data=scores))).stack()) is not None
    }

    values = np.array(list(house_metrics.values()))
    means_across_houses = Stats(values[:, 0]).stack()
    stds_across_houses = Stats(values[:, 1]).stack()

    return [means_across_houses.std(), stds_across_houses.std()]


# ==============================================================================
#                            VISUALISATION MATPLOTLIB
# ==============================================================================

def create_histogram(courses: list | str, show_all: bool = False) -> None:
    """Affiche l'histogramme du cours le plus homogène avec un bouton pour révéler les autres."""
    if isinstance(courses, str):
        courses = [courses]

    fig = plt.figure(figsize=(10, 6))
    state = {"expanded": show_all}

    # --- Sous-fonction : tracé d'un cours ---
    def draw_course(ax, course: str, rank: int) -> None:
        for house, color in HOUSE_COLORS.items():
            scores = df[df["Hogwarts House"] == house][course].dropna()
            ax.hist(
                scores,
                bins=25,
                alpha=0.5,
                label=house,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_title(f"#{rank} - {course}", fontsize=10 if state["expanded"] else 13)
        if not state["expanded"]:
            ax.set_xlabel("Score")
            ax.set_ylabel("Nombre d'élèves")
            ax.legend()

    # --- Sous-fonction : rafraîchissement de la grille ---
    def render() -> None:
        for ax in list(fig.axes):
            if ax is not btn_ax:
                fig.delaxes(ax)
        fig.legends.clear()

        if not state["expanded"]:
            ax = fig.add_subplot(1, 1, 1)
            draw_course(ax, courses[0], 1)
            btn.label.set_text(f"Afficher les {len(courses) - 1} autres cours")
            fig.subplots_adjust(top=0.90, bottom=0.16)
        else:
            cols = 4
            rows = math.ceil(len(courses) / cols)
            for idx, course in enumerate(courses):
                ax = fig.add_subplot(rows, cols, idx + 1)
                draw_course(ax, course, idx + 1)
                if idx == 0:
                    handles, labels = ax.get_legend_handles_labels()

            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=4,
                frameon=True,
                fontsize=9,
                bbox_to_anchor=(0.5, 0.99),
            )
            btn.label.set_text("Revenir au #1 uniquement")
            fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.75, wspace=0.30)

        fig.canvas.draw_idle()

    # --- Sous-fonction : événement clic bouton ---
    def toggle(event) -> None:
        state["expanded"] = not state["expanded"]
        render()

    btn_ax = fig.add_axes([0.35, 0.02, 0.3, 0.05])
    btn = Button(btn_ax, "", color="#e0e0e0", hovercolor="#c8e6c9")
    btn.on_clicked(toggle)

    render()
    plt.show()


# ==============================================================================
#                                  POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(f"{sys.argv[0]} : need .csv")

    df = pd.read_csv(sys.argv[1], index_col="Index")
    df_group = get_dico(df)

    _all_std = {k: compute_std_from_mean(k) for k in df_group.keys()}

    print("\nCours ayant une moyenne de notes la plus similaire :")
    print(" 🟧", min(_all_std, key=lambda k: _all_std[k][0]))

    print("\nCours ayant une dispersion entre maisons la plus similaire :")
    print(" 🟪", min(_all_std, key=lambda k: _all_std[k][1]))

    # Trie tous les cours du plus homogène au moins homogène
    ranked = sorted(_all_std, key=lambda k: _all_std[k][0] + _all_std[k][1])
    print(f"\n✅ Cours le plus homogène : {ranked[0]}")

    # Affiche uniquement ranked[0] par défaut, avec le bouton pour révéler les 12 autres
    create_histogram(ranked)
