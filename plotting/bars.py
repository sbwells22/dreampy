import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import palettes
from utils.constants import MM

def timebar(
    data=None,
    x=None,
    y=None,
    hue=None,
    xlabel=None,
    xticklabels=None, 
    ylabel=None, 
    figsize=None, 
    ax=None,
    ylim = None,
    title=None,
    legend=True,
    **kwargs
):
    """
    Creates a custom boxplot with specified labels, axis limits, and additional parameters.
    
    Parameters:
    - data: DataFrame containing the data to plot.
    - xticklabels: List of strings for the x-axis labels.
    - ylabel: Label for the y-axis.
    - figsize: Size of the figure.
    - vmin: Minimum value for the y-axis (default is -5).
    - vmax: Maximum value for the y-axis (default is 100).
    - title: Title of the plot (default is "CD8s").
    - save: Path to save the figure.
    - **kwargs: Additional keyword arguments to override default boxplot styling.
    """
        
    default_props = dict(
        errwidth=0.5,
        width=0.5,
        saturation=1,
    )
        # Merge default properties with user-specified overrides
    plot_kwargs = {**default_props, **kwargs}
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=data, x=x, y=y, hue=hue,
        ax=ax, legend=legend,
        **plot_kwargs
    )

    ax.set_facecolor([0, 0, 0, 0])
    ax.spines[['left', 'bottom']].set_visible(True)
    ax.spines[['left', 'bottom']].set_color('k')
    ax.spines[['left', 'bottom']].set_linewidth(0.5)
    ax.tick_params(axis='x', labelrotation=90, length=2, width=0.5, pad=0)
    ax.tick_params(axis='y', labelrotation=0, length=2, width=0.5, pad=0.5)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)

    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(labels=xticklabels, fontsize=5)

    if ylim is not None:
        ax.set_ylim(ylim) 
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=5)
    if xlabel is not None:    
        ax.set_xlabel(xlabel, fontsize=5)
    ax.set_title(title, fontsize=5) if title is not None else ""

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        legend = ax.legend(
            handles, labels,
            title="",
            frameon=False,
            handlelength=0.5,
            handleheight=0.5,
            handletextpad=0.3,
            bbox_to_anchor=(-2.2, -0.1),
            loc="upper left",
        )

        for patch in legend.legend_handles:
            patch.set_linewidth(0)

    return ax
    
def numberbar(
    data=None,
    x=None,
    y=None,
    hue=None,
    xlabel=None,
    xticklabels=None, 
    ylabel=None, 
    figsize=None, 
    ax=None,
    ylim = None,
    title=None,
    legend=True,
    **kwargs
):
    """
    Creates a custom boxplot with specified labels, axis limits, and additional parameters.
    
    Parameters:
    - data: DataFrame containing the data to plot.
    - xticklabels: List of strings for the x-axis labels.
    - ylabel: Label for the y-axis.
    - figsize: Size of the figure.
    - vmin: Minimum value for the y-axis (default is -5).
    - vmax: Maximum value for the y-axis (default is 100).
    - title: Title of the plot (default is "CD8s").
    - save: Path to save the figure.
    - **kwargs: Additional keyword arguments to override default boxplot styling.
    """
        
    default_props = dict(
        errwidth=0.5,
        width=0.8,
        saturation=1,
    )
    # Merge default properties with user-specified overrides
    plot_kwargs = {**default_props, **kwargs}
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=data, x=x, y=y, hue=hue,
        ax=ax, legend=legend,
        **plot_kwargs
    )

    ax.set_facecolor([0, 0, 0, 0])
    ax.spines[['bottom']].set_visible(True)
    ax.spines[['bottom']].set_color('k')
    ax.spines[['bottom']].set_linewidth(0.5)
    ax.tick_params(axis='x', labelrotation=90, length=2, width=0.5, pad=0)
    ax.tick_params(axis='y', labelrotation=0, length=2, width=0.5, pad=0.5)
    ax.tick_params(top=False, bottom=True, left=False, right=False, labelleft=False, labelbottom=True)

    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(labels=xticklabels, fontsize=5)

    if ylim is not None:
        ax.set_ylim(ylim) 
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=5)
    if xlabel is not None:    
        ax.set_xlabel(xlabel, fontsize=5)
    ax.set_title(title, fontsize=5) if title is not None else ""

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%d',
            fontsize=4,
            padding=0,
        )

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        legend = ax.legend(
            handles, labels,
            title="",
            frameon=False,
            handlelength=0.5,
            handleheight=0.5,
            handletextpad=0.3,
            bbox_to_anchor=(.85, 1.05),
            loc="upper left",
        )

        for patch in legend.legend_handles:
            patch.set_linewidth(0)

    return ax