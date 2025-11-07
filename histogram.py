import matplotlib.pyplot as plt
from utils import (
    read_dataset, compute_house_stats, homogeneity_score, _to_floats,
    HOUSE_COLORS
)


def plot_histograms(df, stats, courses):
    """Display score distributions by house for each Hogwarts course."""
    houses = df['Hogwarts House'].unique()
    n_courses = len(courses)
    n_cols = 3
    rows = (n_courses + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(rows, n_cols, figsize=(14, 3.2 * rows))
    axes = axes.flatten()

    for i, course in enumerate(courses):
        ax = axes[i]

        # Plot one histogram per house
        for house in houses:
            values = _to_floats(df.loc[df['Hogwarts House'] == house, course])
            if not values:
                continue

            ax.hist(values, bins=20, alpha=0.5, color=HOUSE_COLORS.get(house, 'gray'),
                    label=house, edgecolor='black', linewidth=0.5)

            # Plot mean as vertical line
            m = stats[course][house]
            ax.axvline(m, color=HOUSE_COLORS.get(house, 'gray'),
                       linestyle='--', linewidth=1)
            ax.text(m, ax.get_ylim()[1] * 0.85, f"{m:.1f}", rotation=90,
                    color=HOUSE_COLORS.get(house, 'gray'),
                    fontsize=7, va='top')

        ax.set_title(course, fontsize=9, pad=6)
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    fig.subplots_adjust(
        top=0.90,
        bottom=0.07,
        left=0.06,
        right=0.98,
        hspace=0.7,
        wspace=0.35
    )
    fig.suptitle("Hogwarts Courses: Score Distributions by House",
                 fontsize=14, y=0.995)
    plt.show()


def plot_most_homogeneous(df, stats):
    """Highlight the course with the smallest score gap between houses."""
    homogeneity = {c: homogeneity_score(hm) for c, hm in stats.items()}
    best_course = min(homogeneity, key=homogeneity.get)

    print(f"📊 Most homogeneous course: {best_course}")

    houses = df['Hogwarts House'].unique()
    plt.figure(figsize=(7, 5))

    for house in houses:
        values = _to_floats(df.loc[df['Hogwarts House'] == house, best_course])
        if not values:
            continue
        plt.hist(values, bins=20, alpha=0.5, color=HOUSE_COLORS.get(house, 'gray'),
                 label=house, edgecolor='black', linewidth=0.5)
        m = stats[best_course][house]
        plt.axvline(m, color=HOUSE_COLORS.get(house, 'gray'),
                    linestyle='--', linewidth=1)
        plt.text(m, plt.gca().get_ylim()[1] * 0.9, f"{m:.1f}", rotation=90,
                 color=HOUSE_COLORS.get(house, 'gray'), fontsize=8, va='top')

    plt.title(f"Most Homogeneous Course: {best_course}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()


def main():
    """Entry point of the analysis."""
    df, numeric_cols = read_dataset("datasets/dataset_train.csv")

    # Select relevant courses only
    ignore = ['Index', 'Arithmancy']
    courses = [c for c in numeric_cols if c not in ignore]

    stats = compute_house_stats(df, courses)

    # Print summary data to console
    print("\n========== HOUSE MEAN SCORES BY COURSE ==========")
    for course, means in stats.items():
        print(f"\n{course}:")
        for house, m in means.items():
            print(f"  {house:<12}: {m:.2f}")

    print("\n========== HOMOGENEITY (VARIANCE BETWEEN HOUSES) ==========")
    homogeneity = {c: homogeneity_score(hm) for c, hm in stats.items()}
    for course, h in sorted(homogeneity.items(), key=lambda x: x[1]):
        print(f"{course:<30} -> variance = {h:.3f}")

    best_course = min(homogeneity, key=homogeneity.get)
    print(f"\n🏆 Most homogeneous course: {best_course} (variance = {homogeneity[best_course]:.3f})\n")

    # Plot all histograms
    plot_histograms(df, stats, courses)
    plot_most_homogeneous(df, stats)


if __name__ == "__main__":
    main()
