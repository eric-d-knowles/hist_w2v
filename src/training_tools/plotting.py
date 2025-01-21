import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting capabilities
import numpy as np


def load_results(csv_file):
    """
    Load the evaluation results CSV file.

    Args:
        csv_file (str): Path to the evaluation results CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        return pd.read_csv(csv_file)
    except Exception as e:
        logging.error(f"Failed to load CSV file {csv_file}: {e}")
        return None


def plot_metrics(
    df, metric, x_vars, plot_type="line", output_file=None, plot_title=None
):
    """
    Grade results and generate a grouped plot or contour plot based on the specified metric and metadata.

    Args:
        df (pd.DataFrame): DataFrame containing evaluation results.
        metric (str): Metric to graph (e.g., "similarity_score" or "analogy_score").
        x_vars (list): Var(s) for the x-axis (e.g., ["vector_size"] or ["vector_size", "weight_by"]).
        plot_type (str): Type of plot ("line" or "contour").
        output_file (str): Path to save the plot (optional).
        plot_title (str): Title for the plot (optional).

    Returns:
        None
    """

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

    plt.figure(figsize=(10, 6))

    if plot_type == "line":
        if len(x_vars) == 1:
            # Simple line plot
            grouped.plot(
                x=x_vars[0],
                y=metric,
                kind="line",
                marker="o",
                figsize=(10, 6),
                title=plot_title if plot_title else f"{metric} vs {x_vars[0]}",
                grid=True,
            )
        elif len(x_vars) == 2:
            # Grouped line plot
            ax = None
            for key, group in grouped.groupby(x_vars[1]):
                group.plot(
                    x=x_vars[0],
                    y=metric,
                    kind="line",
                    marker="o",
                    ax=ax,
                    label=f"{x_vars[1]}: {key}",
                    figsize=(10, 6),
                    grid=True,
                )
            plt.title(
                plot_title
                if plot_title
                else f"{metric} vs {x_vars[0]} grouped by {x_vars[1]}"
            )
            plt.legend(title=x_vars[1])
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

            plt.contourf(X, Y, Z, cmap="viridis")
            plt.colorbar(label=metric)
            plt.title(plot_title if plot_title else f"{metric} Contour Plot")
            plt.xlabel(x_vars[0])
            plt.ylabel(x_vars[1])
        elif len(x_vars) == 3:
            # 3D contour plot
            x = grouped[x_vars[0]].values
            y = grouped[x_vars[1]].values
            z = grouped[x_vars[2]].values
            metric_values = grouped[metric].values
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            sc = ax.scatter(x, y, z, c=metric_values, cmap="viridis")
            fig.colorbar(sc, ax=ax, label=metric)
            ax.set_xlabel(x_vars[0])
            ax.set_ylabel(x_vars[1])
            ax.set_zlabel(x_vars[2])
            plt.title(plot_title if plot_title else f"{metric} 3D Scatter Plot")
        else:
            logging.error("Contour plots require two or three x_vars.")
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

    Returns:
        argparse.Namespace: Parsed arguments.
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
        choices=["line", "contour"],
        default="line",
        help="Type of plot to generate ('line' or 'contour')."
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

    results_df = load_results(args.csv_file)
    if results_df is not None:
        plot_metrics(
            df=results_df,
            metric=args.metric,
            x_vars=args.x_vars,
            plot_type=args.plot_type,
            output_file=args.output_file,
            plot_title=args.plot_title
        )