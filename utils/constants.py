RANDOM_STATE=11

MM = 1/25.4

GENERAL_RCPARAMS = {
    "pdf.fonttype": 42,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "markers.fillstyle": "full",
    "savefig.bbox": "tight",
    "font.size":5,
    "legend.fontsize":5,
    "legend.title_fontsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    "figure.titlesize": 7,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    'axes.facecolor': 'white',
    'font.sans-serif':['Arial'],
    'font.family':'sans-serif'
    }

ASSAY_FIG_NAMES = {
    "pdc": "pDC",
    "m":"M",
    "b":"B",
    "cM":"cMono",
    "T8":"CD8",
    "T4": "CD4",
    "ncM": "ncMono"   
}

DICTIONARIES = {
    "general_rcParams":GENERAL_RCPARAMS,
    "assay_fig_names":ASSAY_FIG_NAMES
}
