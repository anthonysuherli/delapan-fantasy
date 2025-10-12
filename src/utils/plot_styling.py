import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


MODERN_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#F77F00',
    'danger': '#D62828',
    'purple': '#7209B7',
    'teal': '#06A77D',
    'navy': '#264653',
    'coral': '#F4A261',
    'sky': '#4ECDC4',
    'rose': '#E63946',
}

PALETTE_GRADIENT = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
PALETTE_COOL = ['#06A77D', '#2E86AB', '#7209B7', '#A23B72', '#F18F01']
PALETTE_WARM = ['#F77F00', '#F4A261', '#E76F51', '#D62828', '#A23B72']


def apply_modern_style():
    """
    Apply modern, beautiful styling to matplotlib plots.
    """
    plt.style.use('seaborn-v0_8-darkgrid')

    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.facecolor': 'white',
        'axes.facecolor': '#F8F9FA',
        'axes.edgecolor': '#DEE2E6',
        'axes.linewidth': 1.2,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'axes.labelcolor': '#212529',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.color': '#ADB5BD',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.color': '#495057',
        'ytick.color': '#495057',
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#DEE2E6',
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    })


def create_gradient_cmap(colors=None):
    """
    Create a custom gradient colormap.

    Args:
        colors: List of hex colors. Defaults to PALETTE_GRADIENT.

    Returns:
        Matplotlib colormap
    """
    import matplotlib.colors as mcolors

    if colors is None:
        colors = PALETTE_GRADIENT

    return mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)


def add_value_labels(ax, spacing=5, format_str='{:.1f}'):
    """
    Add value labels on top of bars in a bar chart.

    Args:
        ax: Matplotlib axes object
        spacing: Vertical spacing between bar and label
        format_str: Format string for labels
    """
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        label = format_str.format(y_value)

        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, spacing),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='#212529'
        )


def style_scatter(ax, title, xlabel, ylabel, color=None):
    """
    Apply consistent styling to scatter plots.

    Args:
        ax: Matplotlib axes object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Main color (defaults to primary)
    """
    if color is None:
        color = MODERN_COLORS['primary']

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def style_histogram(ax, title, xlabel, ylabel, color=None, alpha=0.75):
    """
    Apply consistent styling to histograms.

    Args:
        ax: Matplotlib axes object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Main color (defaults to primary)
        alpha: Transparency level
    """
    if color is None:
        color = MODERN_COLORS['primary']

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def style_boxplot(bp, palette=None):
    """
    Apply modern styling to boxplot elements.

    Args:
        bp: Boxplot object returned by plt.boxplot()
        palette: Color palette to use
    """
    if palette is None:
        palette = PALETTE_GRADIENT

    for patch, color in zip(bp['boxes'], palette * (len(bp['boxes']) // len(palette) + 1)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)

    for whisker in bp['whiskers']:
        whisker.set(color='#495057', linewidth=1.5, linestyle='--')

    for cap in bp['caps']:
        cap.set(color='#495057', linewidth=1.5)

    for median in bp['medians']:
        median.set(color='#212529', linewidth=2.5)

    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='#E76F51', markersize=6, alpha=0.6)


def add_correlation_annotation(ax, x, y, loc='upper left'):
    """
    Add correlation coefficient annotation to a scatter plot.

    Args:
        ax: Matplotlib axes object
        x: X data array
        y: Y data array
        loc: Location of text box
    """
    corr = np.corrcoef(x, y)[0, 1]

    textstr = f'r = {corr:.3f}'

    props = dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        edgecolor=MODERN_COLORS['primary'],
        linewidth=1.5,
        alpha=0.9
    )

    ax.text(
        0.05 if 'left' in loc else 0.95,
        0.95 if 'upper' in loc else 0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='top' if 'upper' in loc else 'bottom',
        horizontalalignment='left' if 'left' in loc else 'right',
        bbox=props
    )
