import matplotlib.pyplot as plt
from utils.constants import MM

def scatter(data, x, y, s, c, tick_labels=None,
            annotate=None, annotate_pos=None, 
            title=None, xlabel=None, ylabel=None, lim=None, save=None):
    
    spine_settings = {'visible': ['left', 'bottom'], 'color': 'k', 'linewidth': 0.5}
    tick_params = {'axis_params': {'x': {'labelrotation': 0}, 'y': {'labelrotation': 0}},
        'common': {'length': 1.5, 'width': 0.5, 'top': False, 'bottom': True, 
         'left': True, 'right': False}}

    fig, ax = plt.subplots(1,1, figsize=(MM*45, MM*45))
    plt.axline((0,0), slope=1, linewidth=0.25, c='#000000', linestyle="--", alpha=0.5)
    ax.scatter(data=data, x=x, y=y, s=s, c=c, linewidths=0)

    # Configure spines
    for spine in ax.spines:
        if spine in spine_settings['visible']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(spine_settings['color'])
            ax.spines[spine].set_linewidth(spine_settings['linewidth'])
        else:
            ax.spines[spine].set_visible(False)

    # Configure ticks
    for axis in ['x', 'y']:
        if axis in tick_params['axis_params']:
            ax.tick_params(
                axis=axis,
                **tick_params['axis_params'][axis],
                **tick_params['common']
                )
            
    if annotate is not None:
        if annotate_pos is None:
            annotate_pos = (1.05, 0.85)      
        ax.text(annotate_pos[0], annotate_pos[1], annotate, fontsize=5)

    if tick_labels is not None:
        ax.set_xticks(tick_labels)
        ax.set_yticks(tick_labels)

    ax.set_title(title) if title is not None else ""
    ax.set_xlabel(xlabel) if xlabel is not None else ""
    ax.set_ylabel(ylabel) if ylabel is not None else ""

    if lim is not None:
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    ax.set_facecolor([0, 0, 0, 0])
    
    if save is not None:
        plt.savefig(save, dpi=400, bbox_inches="tight")
    plt.show()