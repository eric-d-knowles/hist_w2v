import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(csv_file):
    """
    Load the evaluation results CSV file.
    """
    try:
        return pd.read_csv(csv_file)
    except Exception as e:
        logging.error(f"Failed to load CSV file {csv_file}: {e}")
        return None


def plot_metrics(
    df,
    metric,
    x_vars,
    plot_type="line",
    output_file=None,
    plot_title=None,
    label_map=None
):
    """
    Grade results and generate a grouped plot or contour/surface plot 
    based on the specified metric and metadata.
    """

    label_map = {
        'vector_size': 'Vector Dimensions',
        'epochs': 'Training Epochs',
        'weight_by': 'Weighting Method',
        'min_count': 'Minimum Token Count',
        'window': 'Context Window',
        'similarity_score': 'Similarity Score',
        'analogy_score': 'Analogy Score'
    }

    # Set seaborn theme and context
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10
        }
    )

    # If no label_map passed, use an empty dict
    if label_map is None:
        label_map = {}

    # Ensure x_vars is a list
    if isinstance(x_vars, str):
        x_vars = [x_vars]

    # Check that required columns are present
    for col in x_vars + [metric]:
        if col not in df.columns:
            logging.error(f"Column {col} not found in the DataFrame.")
            return

    # Group by the specified x_vars and calculate the mean of the metric
    grouped = df.groupby(x_vars)[metric].mean().reset_index()

    # Sort for better visualization
    grouped = grouped.sort_values(by=x_vars)

    # Create a figure (sometimes we'll override it with a new Figure below)
    plt.figure(figsize=(8, 5))

    if plot_type == "line":
        if len(x_vars) == 1:
            # Simple line plot with seaborn
            sns.lineplot(
                data=grouped,
                x=x_vars[0],
                y=metric,
                marker="o",
                color="blue"
            )
            plt.title(
                plot_title
                if plot_title else
                f"{label_map.get(metric, metric)} vs {label_map.get(x_vars[0], x_vars[0])}"
            )
            plt.xlabel(label_map.get(x_vars[0], x_vars[0]))
            plt.ylabel(label_map.get(metric, metric))
            plt.grid(True)

        elif len(x_vars) == 2:
            # Grouped line plot using the second var as a hue
            sns.lineplot(
                data=grouped,
                x=x_vars[0],
                y=metric,
                hue=x_vars[1],
                marker="o",
                palette="tab10"
            )
            plt.title(
                plot_title
                if plot_title else
                f"{label_map.get(metric, metric)} vs {label_map.get(x_vars[0], x_vars[0])} "
                f"grouped by {label_map.get(x_vars[1], x_vars[1])}"
            )
            plt.xlabel(label_map.get(x_vars[0], x_vars[0]))
            plt.ylabel(label_map.get(metric, metric))
            # Adjust the legend title with a mapped label
            plt.legend(title=label_map.get(x_vars[1], x_vars[1]))
            plt.grid(True)

        else:
            logging.error("Line plots do not support more than two x_vars.")
            return

    elif plot_type == "contour":
        if len(x_vars) == 2:
            # 2D contour plot
            x = grouped[x_vars[0]].values
            y = grouped[x_vars[1]].values
            z = grouped[metric].values
            x_unique, y_unique = np.unique(x), np.unique(y)
            X, Y = np.meshgrid(x_unique, y_unique)
            Z = np.zeros_like(X, dtype=float)

            for i in range(len(x)):
                xi = np.where(x_unique == x[i])[0][0]
                yi = np.where(y_unique == y[i])[0][0]
                Z[yi, xi] = z[i]

            cmap = sns.color_palette("viridis", as_cmap=True)
            cont = plt.contourf(X, Y, Z, cmap=cmap)
            cbar = plt.colorbar(cont, label=label_map.get(metric, metric))

            plt.title(
                plot_title if plot_title else f"{label_map.get(metric, metric)} Contour Plot"
            )
            plt.xlabel(label_map.get(x_vars[0], x_vars[0]))
            plt.ylabel(label_map.get(x_vars[1], x_vars[1]))

        elif len(x_vars) == 3:
            # 3D scatter that uses color to represent the metric
            x = grouped[x_vars[0]].values
            y = grouped[x_vars[1]].values
            z = grouped[x_vars[2]].values
            metric_values = grouped[metric].values

            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection='3d')

            sc = ax.scatter(x, y, z, c=metric_values, cmap="viridis")
            cbar = fig.colorbar(sc, ax=ax, label=label_map.get(metric, metric))

            ax.set_xlabel(label_map.get(x_vars[0], x_vars[0]))
            ax.set_ylabel(label_map.get(x_vars[1], x_vars[1]))
            ax.set_zlabel(label_map.get(x_vars[2], x_vars[2]))
            plt.title(
                plot_title if plot_title else f"{label_map.get(metric, metric)} 3D Scatter Plot"
            )
        else:
            logging.error("Contour plots require two or three x_vars.")
            return

    elif plot_type == "surface":
        if len(x_vars) == 2:
            x = grouped[x_vars[0]].values
            y = grouped[x_vars[1]].values
            z = grouped[metric].values

            x_unique, y_unique = np.unique(x), np.unique(y)
            X, Y = np.meshgrid(x_unique, y_unique)
            Z = np.zeros_like(X, dtype=float)

            for i in range(len(x)):
                xi = np.where(x_unique == x[i])[0][0]
                yi = np.where(y_unique == y[i])[0][0]
                Z[yi, xi] = z[i]

            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection='3d')

            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
            cbar = fig.colorbar(surf, ax=ax, label=label_map.get(metric, metric))

            ax.set_xlabel(label_map.get(x_vars[0], x_vars[0]))
            ax.set_ylabel(label_map.get(x_vars[1], x_vars[1]))
            ax.set_zlabel(label_map.get(metric, metric))

            plt.title(
                plot_title if plot_title else "Hyperparameter Plot"
            )
        else:
            logging.error("Surface plots require exactly two x_vars.")
            return

    else:
        logging.error(f"Unsupported plot type: {plot_type}")
        return

    plt.tight_layout()

    # Save or show the plot
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        logging.info(f"Plot saved to {output_file}")
    else:
        plt.show()


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Grade evaluation results and generate plots."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the evaluation results CSV file."
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["similarity_score", "analogy_score"],
        help="Metric to graph."
    )
    parser.add_argument(
        "--x_vars",
        type=str,
        nargs="+",
        required=True,
        help="Var(s) for the x-axis (e.g., 'vector_size', 'weight_by')."
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["line", "contour", "surface"],
        default="line",
        help="Type of plot to generate ('line', 'contour', or 'surface')."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the plot (optional)."
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default=None,
        help="Title for the plot (optional)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()

    # Load DataFrame
    results_df = load_results(args.csv_file)
    if results_df is not None:
        # Pass the global 'label_map' in
        plot_metrics(
            df=results_df,
            metric=args.metric,
            x_vars=args.x_vars,
            plot_type=args.plot_type,
            output_file=args.output_file,
            plot_title=args.plot_title,
            label_map=label_map
        )