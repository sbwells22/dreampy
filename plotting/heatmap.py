import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from utils import palettes

def heatmap(
    data,
    significant=None,
    significant_threshold = 0.05,
    figsize=(2, 2),
    title=None,
    xlabel=None,
    ylabel=None,
    save=None,
    cbar_kws=None,
    **kwargs
):
    """
    Create a heatmap with optional significance markers and customizable display settings.

    Parameters:
        data (pd.DataFrame): Matrix of values to display in the heatmap.
        significant (pd.DataFrame, optional): Matrix of p-values (same shape as `data`). 
            Adds asterisks for p < significant_threshold and hatching for NaN.
        significant_threshold (int, 0.05): Value for significance.
        figsize (tuple, optional): Figure size in inches (width, height). Default is (2, 2).
        title (str, optional): Title of the heatmap.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        save (str, optional): File path to save the figure. If None, does not save.
        cbar_kws (dict, optional): Keyword arguments for the colorbar.
        **kwargs: Additional keyword arguments passed to `sns.heatmap`.

    Returns:
        None
    """
    # Default colorbar settings
    default_cbar_kws = {
        "shrink":0.15,
        "ticks":[2, 1, 0, -1, -2]
    }
    cbar_kws = {**default_cbar_kws, **(cbar_kws or {})}

    # Default heatmap kwargs
    default_kwargs = {
        "annot":False,
        "fmt":".2f",
        "cmap":palettes.scmap,
        "square":True,
        "linewidths":0.25,
        "cbar":True,
        "center":0,
        "vmax":2,
        "vmin":-2
    }
    # Update defaults with user-provided kwargs
    kwargs = {**default_kwargs, **(kwargs or {})}

    fig, ax = plt.subplots(figsize=figsize)

    hm = sns.heatmap(
        data,
        ax=ax,
        cbar_kws=cbar_kws,
        **kwargs
    )

    # Customize colorbar appearance
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(width=0.5, length=2, labelsize=5)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)

    # Add significance annotations if provided
    if significant is not None:
        for i in range(significant.shape[0]):
            for j in range(significant.shape[1]):
                p = significant.iloc[i, j]
                if p < significant_threshold:
                    ax.text(j + 0.5, i + 0.7, "*", ha='center', va='center', color='white', size=5)
                elif np.isnan(p):
                    ax.add_patch(
                        mpl.patches.Rectangle(
                            (j, i), 1, 1, fill=True, facecolor='grey',
                            linewidth=0.5
                        )
                    )

    # Axis formatting
    ax.tick_params(top=False, bottom=True, left=True, right=False,
                   labelleft=True, labelbottom=True)
    ax.tick_params(axis='x', labelrotation=90, length=2, width=0.5)
    ax.tick_params(axis='y', labelrotation=0, length=2, width=0.5)

    # Set axis labels
    ax.set_xlabel(xlabel if xlabel else "")
    ax.set_ylabel(ylabel if ylabel else "")

    # Set title
    ax.set_title(title if title else "")

    # Save figure if path provided
    if save is not None:
        plt.savefig(save, bbox_inches='tight')

    plt.show()