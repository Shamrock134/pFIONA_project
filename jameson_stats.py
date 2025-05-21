import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal
from scipy.stats import wilcoxon
from scipy.stats import linregress
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from itertools import combinations
import statsmodels.formula.api as smf
from IPython.display import display, HTML
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import mannwhitneyu
from scipy.stats import friedmanchisquare
from scipy.stats import stats, kruskal, mannwhitneyu, shapiro, levene, f_oneway


"""
***Table of Contents***
def plot_lo_to_hi
def left_aligned
def wilx_test
def boxplot_single
def boxplot_multiple
def plot_line_graphs
def run_regression
def plot_regression_side_by_side
def kruskal_test
def pairwise_mannwhitney_labeled
def pairwise_mannwhitney
def boxplot_carryover
def plot_calibration_curve
"""


custom_colors = {
    'carry_12': '#1f77b4',            #blue
    'carry_23': '#2ca02c',            # green
    'carry_34': '#ff7f0e',            # orange
    'carry_group_13_24': '#17becf',   # cyan
    'carry_group_13_14': '#bcbd22',   # olive green
    'carry_group_14_24': '#1f77b4',   # blue
    'rsd_13': '#9467bd',              # purple
    'rsd_24': '#8c564b',              # brown-red
    'rsd_14': '#e377c2',              # pink
    'No Treatment': '#2ca02c',       # green
    'Sample': '#ff7f0e',             # orange
    'NaOH': '#bcbd22',               # olive green
    'MilliQ': '#17becf',             # cyan
    'Other': '#7f7f7f',              # gray
    'Ascorbic Acid': '#1f77b4',      # blue
    'SDS MO': '#8c564b',             # brown
    'SDS Carrier': '#9467bd',        # purple
    'SDS MO & Carrier': '#d62728'}   # red}


def plot_lo_to_hi(dataframe, treatment_order=None, figsize=(True)):
    """
    Plots the mean absorbance values for selected treatments using:
    - ABS4 from blank_lo (as ABS4 [0])
    - ABS1 to ABS4 from concentration 45 (as ABS1 [45] ... ABS4 [45])
    Each treatment is plotted as a separate line with legend.

    Parameters:
    - dataframe: pandas DataFrame containing the absorbance data
    - treatments: list of treatment strings to include in the legend
    """
    labels = ['ABS4 [0]', 'ABS1 [45]', 'ABS2 [45]', 'ABS3 [45]', 'ABS4 [45]']

    plt.figure(figsize=figsize)
    if treatment_order is None:
        treatment_order = [
            'No Treatment', 'Sample', 'NaOH', 'MilliQ', 'Other',
            'Ascorbic Acid', 'SDS MO', 'SDS Carrier', 'SDS MO & Carrier']
    for treatment in treatment_order:
        df_treat = dataframe[dataframe['treatment'] == treatment]

        # Mean ABS4 from blank_lo
        abs4_blank_vals = df_treat[df_treat['conc'] == 'blank_lo']['abs4'].dropna()
        mean_abs4_blank = abs4_blank_vals.mean() if not abs4_blank_vals.empty else None

        # Mean ABS1-ABS4 from 45
        df_45 = df_treat[df_treat['conc'] == '45'][['abs1', 'abs2', 'abs3', 'abs4']].dropna()
        mean_abs_45 = df_45.mean() if not df_45.empty else [None] * 4

        # Combine into plot line if valid
        if mean_abs4_blank is not None and all(pd.notnull(mean_abs_45)):
            values = [mean_abs4_blank] + mean_abs_45.tolist()
            color = custom_colors.get(treatment, 'gray')
            plt.plot(labels, values, marker='o', linestyle='-', label=treatment, color=color)

    plt.ylabel('Mean Absorbance', fontsize=14)
    plt.xlabel(r"Concentration of $[\mathrm{SiO}_4^{2-}]$ (µmol/L)", fontsize=14)
    plt.title('Mean Absorbance Transition Across Treatments', fontsize=15, fontweight='bold')
    plt.grid(True)
    plt.legend(title='Treatment', loc='lower right')
    plt.tight_layout()
    plt.show()

def left_aligned(df):
    """
    Returns a left-aligned HTML-styled DataFrame for display in Jupyter notebooks.

    Parameters:
    - df (DataFrame): DataFrame to be displayed.

    Returns:
    - HTML: Left-aligned styled HTML object.
    """
    styles = """
    <style>
        th, td {text-align: left !important; font-family: monospace;}
    </style>
    """
    return HTML(styles + df.to_html(index=False, escape=False))
    

def wilx_test(df, treatment, metric):
    """
    Performs a paired Wilcoxon signed-rank test on absorbance or RSD metrics.

    Parameters:
    - df (DataFrame): Full dataset with absorbance values.
    - treatment (str): The treatment group to analyze.
    - metric (str): One of 'carry_12', 'carry_group_13_24', or 'rsd_group_13_24'.

    Returns:
    - dict: Summary statistics and Wilcoxon test result.
    """
    subset = df[df['treatment'] == treatment].dropna(subset=['abs1', 'abs2', 'abs3', 'abs4']).copy()

    if metric == 'carry_12':
        x1 = subset['abs1']
        x2 = subset['abs2']
        label = r"Carryover % $\Delta$ Abs1 vs Abs2"
        delta = lambda x1, x2: ((x2 - x1) / x1) * 100
    elif metric == 'carry_group_13_24':
        x1 = subset[['abs1', 'abs2', 'abs3']].mean(axis=1)
        x2 = subset[['abs2', 'abs3', 'abs4']].mean(axis=1)
        label = r"Carryover % $\Delta$ Abs1:3 vs Abs2:4"
        delta = lambda x1, x2: ((x2 - x1) / x1) * 100
    elif metric == 'rsd_group_13_24':
        mean_13 = subset[['abs1', 'abs2', 'abs3']].mean(axis=1)
        mean_24 = subset[['abs2', 'abs3', 'abs4']].mean(axis=1)
        rsd_13 = subset[['abs1', 'abs2', 'abs3']].std(axis=1) / mean_13 * 100
        rsd_24 = subset[['abs2', 'abs3', 'abs4']].std(axis=1) / mean_24 * 100
        x1, x2 = rsd_13, rsd_24
        label = r"RSD $\Delta$ Abs1:3 vs Abs2:4"
        delta = lambda x1, x2: x1 - x2
    else:
        raise ValueError("Unsupported metric")

    paired = pd.DataFrame({'x1': x1, 'x2': x2}).dropna()
    stat, p = wilcoxon(paired['x1'], paired['x2'])

    return {
        "Treatment": treatment,
        "Metric": label,
        "Mean x₁": round(paired['x1'].mean(), 4),
        "Mean x₂": round(paired['x2'].mean(), 4),
        "Δ %": round(delta(paired['x1'].mean(), paired['x2'].mean()), 4),
        "W": round(stat, 4),
        "p-value": round(p, 4),
        "n": len(paired)}

def boxplot_single(df, treatment=True, label=True, fig_size=(21, 5)):
    """
    Generates a 3-panel boxplot for a single treatment group showing:
    1. Carryover % between Abs1 and Abs2
    2. Grouped carryover % (Abs1–3 vs Abs2–4)
    3. RSD % for Abs1–3

    Parameters:
    - df (DataFrame): Dataset with absorbance values (abs1–abs4).
    - treatment (str): Treatment group to filter.
    - label (str): Label used for titles.
    - fig_size (tuple): Figure size.

    Returns:
    - None. Displays a matplotlib figure.
    """
    subset = df[df['treatment'] == treatment].copy()
    subset = subset.dropna(subset=['abs1', 'abs2', 'abs3', 'abs4'])

    # Carry_12
    subset['carry_12'] = ((subset['abs2'] - subset['abs1']) / subset['abs1']) * 100

    # Carry groups
    subset['mean_13'] = subset[['abs1', 'abs2', 'abs3']].mean(axis=1)
    subset['mean_24'] = subset[['abs2', 'abs3', 'abs4']].mean(axis=1)
    subset['carry_group_13_24'] = ((subset['mean_24'] - subset['mean_13']) / subset['mean_13']) * 100

    # RSD
    subset['rsd_13'] = subset[['abs1', 'abs2', 'abs3']].std(axis=1) / subset[['abs1', 'abs2', 'abs3']].mean(axis=1) * 100

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size, sharey=True)

    # Left: Carryover % Abs1 → Abs2
    sns.boxplot(y=subset['carry_12'], ax=ax1,
                color=custom_colors['carry_12'], width=0.37125, showfliers=False)
    ax1.set_title(f'Carryover % (Abs1 → Abs2): {label}', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Carryover % / RSD %', fontsize=14)
    ax1.set_xlabel('Absorbance Transition', fontsize=14)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Abs1 → Abs2'])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax1.axhline(3, color='green', linestyle='--', linewidth=1.5)

    # Middle: carry_group24 in cyan
    sns.boxplot(y=subset['carry_group_13_24'], ax=ax2,
                color=custom_colors['carry_group_13_24'], width=0.37125, showfliers=False)  # Cyan
    ax2.set_title(f'Carryover % (Abs1:3 vs Abs2:4): {label}', fontsize=15, fontweight='bold')
    ax2.set_ylabel('')
    ax2.set_xlabel('Absorbance Group', fontsize=14)
    ax2.set_xticks([0])
    ax2.set_xticklabels(['Abs1:3 vs Abs2:4'])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax2.axhline(3, color='green', linestyle='--', linewidth=1.5)

    # Right: RSD13
    sns.boxplot(y=subset['rsd_13'], ax=ax3,
                color=custom_colors['rsd_13'], width=0.37125, showfliers=False)  # Purple
    ax3.set_title(f'RSD (Abs1:3): {label}', fontsize=15, fontweight='bold')
    ax3.set_ylabel('')
    ax3.set_xlabel('Absorbance Group RSD', fontsize=14)
    ax3.set_xticks([0])
    ax3.set_xticklabels(['RSD(Abs1:3)'])
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax3.axhline(3, color='green', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plt.show()


def boxplot_multiple(df, treatment=True, label=True, fig_size=(21, 5)):
    """
    Generates three boxplots for a given treatment group:
    1. Carryover % across adjacent absorbance transitions (Abs1→Abs2, etc.)
    2. Grouped carryover % comparisons (e.g., Abs1–3 vs Abs2–4)
    3. Relative standard deviation (RSD %) for Abs1–3, Abs2–4, and Abs1–4

    Parameters:
    - df (DataFrame): Dataset containing absorbance values.
    - treatment (str): The treatment group to visualize.
    - label (str): Title label for figure headings.
    - fig_size (tuple): Size of the resulting figure.

    Returns:
    - None. Displays a matplotlib plot.
    """
    subset = df[df['treatment'] == treatment].copy()
    subset = subset.dropna(subset=['abs1', 'abs2', 'abs3', 'abs4'])

    # Carryover transitions
    subset['carry_12'] = ((subset['abs2'] - subset['abs1']) / subset['abs1']) * 100
    subset['carry_23'] = ((subset['abs3'] - subset['abs2']) / subset['abs2']) * 100
    subset['carry_34'] = ((subset['abs4'] - subset['abs3']) / subset['abs3']) * 100
    melted_transitions = subset[['carry_12', 'carry_23', 'carry_34']].melt(
        var_name='Transition', value_name='Carryover (%)')

    # Grouped carryover
    subset['mean_13'] = subset[['abs1', 'abs2', 'abs3']].mean(axis=1)
    subset['mean_24'] = subset[['abs2', 'abs3', 'abs4']].mean(axis=1)
    subset['mean_14'] = subset[['abs1', 'abs2', 'abs3', 'abs4']].mean(axis=1)
    subset['carry_group_13_24'] = ((subset['mean_24'] - subset['mean_13']) / subset['mean_13']) * 100
    subset['carry_group_14_24'] = ((subset['mean_24'] - subset['mean_14']) / subset['mean_14']) * 100
    subset['carry_group_13_14'] = ((subset['mean_14'] - subset['mean_13']) / subset['mean_13']) * 100
    melted_groups = subset[['carry_group_13_24', 'carry_group_14_24', 'carry_group_13_14']].melt(
        var_name='Transition', value_name='Carryover (%)')

    # RSD groups
    subset['rsd_13'] = subset[['abs1', 'abs2', 'abs3']].std(axis=1) / subset[['abs1', 'abs2', 'abs3']].mean(axis=1) * 100
    subset['rsd_24'] = subset[['abs2', 'abs3', 'abs4']].std(axis=1) / subset[['abs2', 'abs3', 'abs4']].mean(axis=1) * 100
    subset['rsd_14'] = subset[['abs1', 'abs2', 'abs3', 'abs4']].std(axis=1) / subset[['abs1', 'abs2', 'abs3', 'abs4']].mean(axis=1) * 100
    melted_rsd = subset[['rsd_13', 'rsd_14', 'rsd_24']].melt(var_name='Window', value_name='RSD (%)')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size, sharey=True)

    # Left: Carryover transitions
    sns.boxplot(data=melted_transitions,x='Transition', y='Carryover (%)',
        hue='Transition',palette={
            'carry_12': custom_colors['carry_12'],
            'carry_23': custom_colors['carry_23'],
            'carry_34': custom_colors['carry_34']},
        ax=ax1, showfliers=False)
    
    ax1.set_title(f'Carryover % Across Transitions: {label}', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Absorbance Transition Comparison', fontsize=14)
    ax1.set_ylabel('Carryover % / RSD %', fontsize=14)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['Abs1 → Abs2', 'Abs2 → Abs3', 'Abs3 → Abs4'])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax1.axhline(3, color='green', linestyle='--', linewidth=1.5)

    # Middle: Grouped carryover
    sns.boxplot(data=melted_groups,x='Transition', y='Carryover (%)',
        hue='Transition',palette={
            'carry_group_13_24': custom_colors['carry_group_13_24'],
            'carry_group_14_24': custom_colors['carry_group_14_24'],
            'carry_group_13_14': custom_colors['carry_group_13_14']},
        ax=ax2, showfliers=False)
    
    ax2.set_title(f'Carryover % Absorbance Groups: {label}', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Absorbance Group Comparison', fontsize=14)
    ax2.set_ylabel('')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Abs1:3 vs Abs2:4', 'Abs1:3 vs Abs1:4', 'Abs1:4 vs Abs2:4'])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax2.axhline(3, color='green', linestyle='--', linewidth=1.5)

    # Right: RSD group comparison
    sns.boxplot(data=melted_rsd,x='Window', y='RSD (%)',
        hue='Window',palette={
            'rsd_13': custom_colors['rsd_13'],
            'rsd_24': custom_colors['rsd_24'],
            'rsd_14': custom_colors['rsd_14']},
        ax=ax3, showfliers=False)
    
    ax3.set_title(f'RSD % Absorbance Groups: {label}', fontsize=15, fontweight='bold')
    ax3.set_xlabel('Absorbance Group RSD Comparison', fontsize=14)
    ax3.set_ylabel('Relative Standard Deviation %')
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['RSD(Abs1:3)', 'RSD(Abs1:4)', 'RSD(Abs2:4)'])
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax3.axhline(3, color='green', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plt.show()


def plot_line_graphs(df, treatment_col='treatment', treatment_order=None, ylim=None, fig_size=(21, 6)):
    """
    Creates a three-panel line plot of mean carryover and RSD metrics across treatments:
    1. Absorbance-to-absorbance carryover %
    2. Grouped absorbance comparisons
    3. RSD % for different groupings

    Parameters:
    - df (DataFrame): DataFrame with absorbance data (abs1–abs4).
    - treatment_col (str): Column name identifying treatment groups.
    - treatment_order (list): Order in which treatments appear in the plots.
    - ylim (tuple): Optional y-axis limits.
    - fig_size (tuple): Size of the entire figure.

    Returns:
    - None. Displays a matplotlib plot.
    """
    df = df.copy()

    if treatment_order is None:
        treatment_order = [
            'No Treatment', 'Sample', 'NaOH', 'MilliQ', 'Other',
            'Ascorbic Acid', 'SDS MO', 'SDS Carrier', 'SDS MO & Carrier']
        
    transition_keys = [('abs1', 'abs2'), ('abs2', 'abs3'), ('abs3', 'abs4')]
    transition_labels = ['Abs1 → Abs2', 'Abs2 → Abs3', 'Abs3 → Abs4']
    transition_colors = [custom_colors['carry_12'], custom_colors['carry_23'], custom_colors['carry_34']]

    if treatment_order is None:
        treatment_order = sorted(df[treatment_col].dropna().unique())
    df[treatment_col] = pd.Categorical(df[treatment_col], categories=treatment_order, ordered=True)

    # Calculate metrics
    df['carry_12'] = (df['abs2'] - df['abs1']) / df['abs1'] * 100
    df['carry_23'] = (df['abs3'] - df['abs2']) / df['abs2'] * 100
    df['carry_34'] = (df['abs4'] - df['abs3']) / df['abs3'] * 100

    df['mean_13'] = df[['abs1', 'abs2', 'abs3']].mean(axis=1)
    df['mean_24'] = df[['abs2', 'abs3', 'abs4']].mean(axis=1)
    df['mean_14'] = df[['abs1', 'abs2', 'abs3', 'abs4']].mean(axis=1)
    df['carry_group_13_24'] = ((df['mean_24'] - df['mean_13']) / df['mean_13']) * 100
    df['carry_group_14_24'] = ((df['mean_24'] - df['mean_14']) / df['mean_14']) * 100
    df['carry_group_13_14'] = ((df['mean_14'] - df['mean_13']) / df['mean_13']) * 100

    df['rsd_13'] = df[['abs1', 'abs2', 'abs3']].std(axis=1) / df[['abs1', 'abs2', 'abs3']].mean(axis=1) * 100
    df['rsd_24'] = df[['abs2', 'abs3', 'abs4']].std(axis=1) / df[['abs2', 'abs3', 'abs4']].mean(axis=1) * 100
    df['rsd_14'] = df[['abs1', 'abs2', 'abs3', 'abs4']].std(axis=1) / df[['abs1', 'abs2', 'abs3', 'abs4']].mean(axis=1) * 100

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size)

    # LEFT: carry_12, carry_23, carry_34
    for col, label, color in zip(['carry_12', 'carry_23', 'carry_34'], transition_labels, transition_colors):
        means = df.groupby(treatment_col, observed=True)[col].mean().reindex(treatment_order)
        ax1.plot(means.index, means.values, marker='o', linestyle='-', label=label, color=color)

    ax1.set_title('Carryover % - Transitions per Treatment', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Treatment', fontsize=14)
    ax1.set_ylabel('Mean Carryover (%)', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    if ylim: ax1.set_ylim(ylim)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax1.axhline(3, color='green', linestyle='--', linewidth=1.5)
    ax1.legend()

    # MIDDLE: carry_group24 and carry_group14
    for col, label, color in zip(['carry_group_13_24', 'carry_group_14_24', 'carry_group_13_14'],
                                 ['Abs1:3 vs Abs2:4', 'Abs1:3 vs Abs1:4','Abs1:4 vs Abs2:4'],
                                 [custom_colors['carry_group_13_24'], custom_colors['carry_group_14_24'], custom_colors['carry_group_13_14']]):
        means = df.groupby(treatment_col, observed=True)[col].mean().reindex(treatment_order)
        ax2.plot(means.index, means.values, marker='o', linestyle='-', label=label, color=color)

    ax2.set_title('Carryover % - Absorbance Groups per Treatment', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Treatment', fontsize=14)
    ax2.set_ylabel('')
    ax2.tick_params(axis='x', rotation=45)
    if ylim: ax2.set_ylim(ylim)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax2.axhline(3, color='green', linestyle='--', linewidth=1.5)
    ax2.legend()

    # RIGHT: rsd_13, rsd_24, rsd_14
    for col, label, color in zip(['rsd_13', 'rsd_24', 'rsd_14'],
                                 ['RSD(Abs1:3)', 'RSD(Abs2:4)', 'RSD(Abs1:4)'],
                                 [custom_colors['rsd_13'], custom_colors['rsd_24'], custom_colors['rsd_14']]):
        means = df.groupby(treatment_col, observed=True)[col].mean().reindex(treatment_order)
        ax3.plot(means.index, means.values, marker='o', linestyle='-', label=label, color=color)

    ax3.set_title('RSD % per Treatment', fontsize=15, fontweight='bold')
    ax3.set_xlabel('Treatment', fontsize=14)
    ax3.set_ylabel('')
    ax3.tick_params(axis='x', rotation=45)
    if ylim: ax3.set_ylim(ylim)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax3.axhline(3, color='green', linestyle='--', linewidth=1.5)
    ax3.legend()

    plt.tight_layout()
    plt.show()


def regression_model(df, metric='carry_12', treatment_order=None, show_summary=False):
    """
    Fits an OLS regression model for a specified carryover or RSD metric across treatments.

    Parameters:
    - df (DataFrame): Dataset containing absorbance measurements.
    - metric (str): One of 'carry_12', 'carry_group_13_24', or 'rsd_13'. Determines which metric to compute and model.
    - treatment_order (list): Optional list to specify the order of treatments in the model.
    - show_summary (bool): If True, prints model summary tables.

    Returns:
    - model (RegressionResultsWrapper): Fitted OLS regression model.
    """
    df = df.copy()

    if treatment_order is None:
        treatment_order = [
            'No Treatment', 'Sample', 'NaOH', 'MilliQ', 'Other',
            'Ascorbic Acid', 'SDS MO', 'SDS Carrier', 'SDS MO & Carrier']

    available = df['treatment'].dropna().unique().tolist()
    treatment_order = [t for t in treatment_order if t in available]
    df['treatment'] = df['treatment'].astype('category')
    df['treatment'] = df['treatment'].cat.reorder_categories(treatment_order, ordered=False)

    # Compute selected metric
    if metric == 'carry_12':
        df[metric] = (df['abs2'] - df['abs1']) / df['abs1'] * 100
    elif metric == 'carry_group_13_24':
        df['mean_13'] = df[['abs1', 'abs2', 'abs3']].mean(axis=1)
        df['mean_24'] = df[['abs2', 'abs3', 'abs4']].mean(axis=1)
        df[metric] = ((df['mean_24'] - df['mean_13']) / df['mean_13']) * 100
    elif metric == 'rsd_13':   
        df['rsd_13'] = df[['abs1', 'abs2', 'abs3']].std(axis=1) / df[['abs1', 'abs2', 'abs3']].mean(axis=1) * 100
    else:
        raise ValueError("Metric must be one of: 'carry_12', 'carry_group_13_24', 'rsd_group_13_24'")

    # Fit OLS model
    model = smf.ols(f"{metric} ~ C(treatment)", data=df).fit()

    if show_summary:
        summary = model.summary2()
        overview_df = summary.tables[0]
        coef_df = summary.tables[1]
        diagnostics_df = summary.tables[2]
        display(HTML("<div style='display: flex; gap: 2em;'>"
                     f"<div>{overview_df.to_html()}</div>"
                     f"<div>{coef_df.to_html()}</div>"
                     f"<div>{diagnostics_df.to_html()}</div>"
                     "</div>"))
    return model


def plot_regression_side_by_side(
    model_carry_12, model_carry_group, model_rsd_group,
    fig_size=(21, 6),
    reference_line_12=-3.74,
    reference_line_groups=0.87,
    reference_line_rsd=-1.03,
    title_12="Carryover % Reduction (Abs1 → Abs2)",
    title_groups="Carryover % Reduction (Abs1:3 vs Abs2:4)",
    title_rsd="Relative Standard Deviation (Abs1:3)"):
    """
    Plots regression effect sizes and confidence intervals side-by-side for three metrics.

    Parameters:
    - model_carry_12: OLS model for carryover % between Abs1 and Abs2
    - model_carry_group: OLS model for grouped carryover % (Abs1–3 vs Abs2–4)
    - model_rsd_group: OLS model for RSD % (Abs1–3)
    - fig_size (tuple): Figure size for the plot
    - reference_line_* (float): Optional green dashed line for visual reference
    - title_* (str): Custom titles for each subplot

    Returns:
    - None (displays a matplotlib figure)
    """
    def extract_effects(model, treatments_to_include):
        summary = model.summary2().tables[1].reset_index()
        summary.columns = ['Term', 'Coef', 'StdErr', 't', 'p_value', 'CI_low', 'CI_high']
        effects = summary[summary['Term'].str.startswith('C(treatment)')].copy()
        effects['Treatment'] = effects['Term'].str.replace('C(treatment)[T.', '', regex=False).str.replace(']', '', regex=False)
        effects['Significance'] = effects['p_value'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')
        effects['Summary'] = effects.apply(lambda row: f"{row['Coef']:.2f} {row['Significance']}", axis=1)
        effects = effects.set_index('Treatment').reindex(treatments).dropna().reset_index()
        return effects

    treatments = [
        'SDS MO & Carrier', 'SDS MO', 'SDS Carrier', 'Ascorbic Acid',
        'Other', 'Sample', 'MilliQ', 'NaOH']

    effects_12 = extract_effects(model_carry_12, treatments)
    effects_groups = extract_effects(model_carry_group, treatments)
    effects_rsd = extract_effects(model_rsd_group, treatments)

    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    titles = [title_12, title_groups, title_rsd]
    refs = [reference_line_12, reference_line_groups, reference_line_rsd]

    for ax, effects, title, ref in zip(axes, [effects_12, effects_groups, effects_rsd], titles, refs):
        bar_colors = [custom_colors.get(t, 'gray') for t in effects['Treatment']]
        ax.barh(
            y=effects['Treatment'],
            width=effects['Coef'],
            xerr=[effects['Coef'] - effects['CI_low'], effects['CI_high'] - effects['Coef']],
            color=bar_colors,
            edgecolor='black')
        
        ax.set_yticks(range(len(effects)))
        ax.set_yticklabels(effects['Treatment'])
        label_x = ax.get_xlim()[0] + 0.5
        for i, row in effects.iterrows():
            ax.text(label_x, i + 0.25, row['Summary'], va='center', ha='left', fontsize=10, color='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        if ref is not None:
            ax.axvline(ref, color='green', linestyle='--', linewidth=1.5)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Estimated Change', fontsize=14)
        ax.set_ylabel('Treatment', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def kruskal_test(df, metric='carry_12', treatments=None):
    """
    Performs a Kruskal-Wallis H-test across multiple treatment groups for a specified metric.

    Parameters:
    - df (DataFrame): Dataset containing absorbance data.
    - metric (str): Metric to calculate ('carry_12', 'carry_group_13_24', or 'rsd_group_13_24').
    - treatments (list): List of treatment names to include. If None, includes all standard treatments.

    Returns:
    - dict: H-statistic, p-value, and metric name. Includes note if any group has no data.
    """
    df = df.copy()

    if treatments is None:
        treatments = [
            'No Treatment', 'Sample', 'NaOH', 'MilliQ', 'Other',
            'Ascorbic Acid', 'SDS MO', 'SDS Carrier', 'SDS MO & Carrier'
        ]

    df = df[df['treatment'].isin(treatments)]

    # Compute metric
    if metric == 'carry_12':
        df[metric] = (df['abs2'] - df['abs1']) / df['abs1'] * 100
    elif metric == 'carry_group_13_24':
        df['mean_13'] = df[['abs1', 'abs2', 'abs3']].mean(axis=1)
        df['mean_24'] = df[['abs2', 'abs3', 'abs4']].mean(axis=1)
        df[metric] = ((df['mean_24'] - df['mean_13']) / df['mean_13']) * 100
    elif metric == 'rsd_group_13_24':
        mean_13 = df[['abs1', 'abs2', 'abs3']].mean(axis=1)
        mean_24 = df[['abs2', 'abs3', 'abs4']].mean(axis=1)
        rsd_13 = df[['abs1', 'abs2', 'abs3']].std(axis=1) / mean_13 * 100
        rsd_24 = df[['abs2', 'abs3', 'abs4']].std(axis=1) / mean_24 * 100
        df[metric] = rsd_24 - rsd_13
    else:
        raise ValueError("Metric must be one of: 'carry_12', 'carry_group_13_24', 'rsd_group_13_24'")

    groups = [df[df['treatment'] == t][metric].dropna() for t in treatments]
    if all(len(g) > 0 for g in groups):
        h_stat, p_value = kruskal(*groups)
        return {
            'metric': metric,
            'H-statistic': h_stat,
            'p-value': p_value
        }
    else:
        return {
            'metric': metric,
            'H-statistic': None,
            'p-value': None,
            'note': 'One or more groups had no data.'
        }


def pairwise_mannwhitney_labeled(df, metric='carry_12', treatments=None):
    """
    Performs pairwise Mann-Whitney U tests for a specified metric across all treatments.

    Parameters:
    - df (DataFrame): Dataset containing absorbance data.
    - metric (str): Metric to calculate ('carry_12', 'carry_group_13_24', or 'rsd_group_13_24').
    - treatments (list): List of treatments to compare. Defaults to standard list if None.

    Returns:
    - DataFrame: Contains U statistics and p-values for all pairwise comparisons.
    """
    df = df.copy()

    if treatments is None:
        treatments = [
            'No Treatment', 'Sample', 'NaOH', 'MilliQ', 'Other',
            'Ascorbic Acid', 'SDS MO', 'SDS Carrier', 'SDS MO & Carrier'
        ]

    df = df[df['treatment'].isin(treatments)]

    if metric == 'carry_12':
        df[metric] = (df['abs2'] - df['abs1']) / df['abs1'] * 100
        label = r"Carryover % $\Delta$ Abs1 vs Abs2"
    elif metric == 'carry_group_13_24':
        df['mean_13'] = df[['abs1', 'abs2', 'abs3']].mean(axis=1)
        df['mean_24'] = df[['abs2', 'abs3', 'abs4']].mean(axis=1)
        df[metric] = ((df['mean_24'] - df['mean_13']) / df['mean_13']) * 100
        label = r"Carryover % $\Delta$ Abs1:3 vs Abs2:4"
    elif metric == 'rsd_group_13_24':
        mean_13 = df[['abs1', 'abs2', 'abs3']].mean(axis=1)
        mean_24 = df[['abs2', 'abs3', 'abs4']].mean(axis=1)
        rsd_13 = df[['abs1', 'abs2', 'abs3']].std(axis=1) / mean_13 * 100
        rsd_24 = df[['abs2', 'abs3', 'abs4']].std(axis=1) / mean_24 * 100
        df[metric] = rsd_24 - rsd_13
        label = r"RSD % $\Delta$ Abs1:3 vs Abs2:4"
    else:
        raise ValueError("Metric must be one of: 'carry_12', 'carry_group_13_24', 'rsd_group_13_24'")

    results = []
    for a, b in combinations(treatments, 2):
        group_a = df[df['treatment'] == a][metric].dropna()
        group_b = df[df['treatment'] == b][metric].dropna()
        if len(group_a) > 0 and len(group_b) > 0:
            stat, p = mannwhitneyu(group_a, group_b, alternative='two-sided')
            results.append({
                'Metric': label,
                'Group 1': a,
                'Group 2': b,
                'U statistic': stat,
                'p-value': p
            })
        else:
            results.append({
                'Metric': label,
                'Group 1': a,
                'Group 2': b,
                'U statistic': None,
                'p-value': None,
                'note': 'Missing data in one or both groups'
            })

    return pd.DataFrame(results)

# Run all 3 labeled metrics
def pairwise_mannwhitney(df, treatments=None):
    """
    Runs pairwise Mann-Whitney U tests for three key metrics and combines results into a single DataFrame.

    Parameters:
    - df (DataFrame): Dataset with absorbance data.
    - treatments (list): Optional list of treatments to compare. Uses defaults if None.

    Returns:
    - DataFrame: Combined pairwise test results for all metrics.
    """
    metrics = ['carry_12', 'carry_group_13_24', 'rsd_group_13_24']
    all_results = []

    for metric in metrics:
        result_df = pairwise_mannwhitney_labeled(df, metric=metric, treatments=treatments)
        all_results.append(result_df)

    return pd.concat(all_results, ignore_index=True)


def boxplot_carryover(df, treatments, figsize=(24, 6), show_points=False):
    """
    Generates a 3-panel boxplot for multiple treatments showing:
    1. Carryover % (Abs1 → Abs2)
    2. Grouped carryover % (Abs1–3 vs Abs2–4)
    3. RSD % (Abs1–3)

    Parameters:
    - df (DataFrame): Dataset with absorbance and treatment data.
    - treatments (list): List of treatment names to include.
    - figsize (tuple): Size of the figure.
    - show_points (bool): If True, overlays individual data points.

    Returns:
    - None. Displays a matplotlib figure.
    """
    df = df.copy()

    # Calculate metrics
    df['carry_12'] = (df['abs2'] - df['abs1']) / df['abs1'] * 100
    df['mean_13'] = df[['abs1', 'abs2', 'abs3']].mean(axis=1)
    df['mean_24'] = df[['abs2', 'abs3', 'abs4']].mean(axis=1)
    df['carry_group_13_24'] = ((df['mean_24'] - df['mean_13']) / df['mean_13']) * 100
    df['rsd_13'] = df[['abs1', 'abs2', 'abs3']].std(axis=1) / df['mean_13'] * 100
    df['rsd_24'] = df[['abs2', 'abs3', 'abs4']].std(axis=1) / df['mean_24'] * 100
    df['rsd_group_13_24'] = df['rsd_24'] - df['rsd_13']

    df = df[df['treatment'].isin(treatments)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, sharey=True)
    palette = {t: custom_colors.get(t, 'gray') for t in treatments}

    # Plot 1: carry_12
    sns.boxplot(data=df, x='treatment', y='carry_12', hue='treatment',
                order=treatments, showfliers=False, palette=palette, ax=ax1, legend=False)
    if show_points:
        sns.stripplot(data=df, x='treatment', y='carry_12', order=treatments,
                      color='black', alpha=0.6, jitter=0.2, ax=ax1)
    ax1.set_title("Carryover % (Abs1 → Abs2)", fontsize=15, fontweight='bold')
    ax1.set_xlabel('Treatment', fontsize=14)
    ax1.set_ylabel('Carryover / RSD (%)', fontsize=14)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax1.axhline(3, color='green', linestyle='--', linewidth=1.5)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: carry_group_13_24
    sns.boxplot(data=df, x='treatment', y='carry_group_13_24', hue='treatment',
                order=treatments, showfliers=False, palette=palette, ax=ax2, legend=False)
    if show_points:
        sns.stripplot(data=df, x='treatment', y='carry_group_13_24', order=treatments,
                      color='black', alpha=0.6, jitter=0.2, ax=ax2)
    ax2.set_title("Carryover % (Abs1:3 vs Abs2:4)", fontsize=15, fontweight='bold')
    ax2.set_xlabel('Treatment', fontsize=14)
    ax2.set_ylabel('')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax2.axhline(3, color='green', linestyle='--', linewidth=1.5)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: rsd_133
    sns.boxplot(data=df, x='treatment', y='rsd_13', hue='treatment',
                order=treatments, showfliers=False, palette=palette, ax=ax3, legend=False)
    if show_points:
        sns.stripplot(data=df, x='treatment', y='rsd_13', order=treatments,
                      color='black', alpha=0.6, jitter=0.2, ax=ax3)
    ax3.set_title("RSD % (Abs1:3)", fontsize=15, fontweight='bold')
    ax3.set_xlabel('Treatment', fontsize=14)
    ax3.set_ylabel('')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.axhline(-3, color='green', linestyle='--', linewidth=1.5)
    ax3.axhline(3, color='green', linestyle='--', linewidth=1.5)
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
    
def plot_calibration_curve(data, treatment_filter=None, show_residuals=False):
    treatment_labels = {
        'No Treatment': 'No SDS',
        'SDS MO': 'SDS Molybdate',
        'SDS Carrier': 'SDS Carrier',
        'SDS MO & Carrier': 'SDS Mo & Car'
    }
    """
    Plots calibration curves (absorbance vs concentration) with optional residual plots.

    Parameters:
    - data (DataFrame): Data containing 'conc', 'abs1' to 'abs3', and 'treatment'.
    - treatment_filter (str): Optional single treatment to isolate. If None, plots all.
    - show_residuals (bool): If True, plots residuals in a lower subplot.

    Returns:
    - None. Displays a matplotlib plot.
    """
    # Filter to calibration curve data
    df = data[(data['purpose'] == 'cal_curve') & (data['flag'] == 1)].copy()
    df['mean_abs'] = df[['abs1', 'abs2', 'abs3']].mean(axis=1)
    df['conc'] = pd.to_numeric(df['conc'], errors='coerce')
    df = df.dropna(subset=['conc', 'mean_abs', 'treatment'])

    # Determine which treatments to include
    treatments_to_plot = [treatment_filter] if treatment_filter else list(treatment_labels.keys())

    colors = plt.cm.tab10.colors
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(6, 5))

    for i, treatment in enumerate(treatments_to_plot):
        sub_df = df[df['treatment'] == treatment]
        if sub_df.empty:
            continue

        x = sub_df['conc'].values
        y = sub_df['mean_abs'].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_sorted = np.sort(x)
        y_fit = slope * x_sorted + intercept

        # Residuals
        y_pred = slope * x + intercept
        residuals = y - y_pred

        label = f"{treatment_labels.get(treatment, treatment)} (R² = {r_value**2:.3f})"
        ax1.scatter(x, y, color=colors[i % len(colors)], alpha=0.6)
        ax1.plot(x_sorted, y_fit, color=colors[i % len(colors)], label=label)

        if show_residuals:
            ax2.scatter(x, residuals, color=colors[i % len(colors)], alpha=0.6)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Labels and formatting
    ax1.set_ylabel('Mean Absorbance (abs1–3)', fontsize=16)
    title_suffix = treatment_labels.get(treatment_filter, 'All Treatments') if treatment_filter else 'All Treatments'
    ax1.set_title(f'Calibration Curve - {title_suffix}', fontsize=18, fontweight='bold')
    ax1.legend()
    ax1.grid(True)

    if show_residuals:
        ax2.set_xlabel(r"Concentration of $[\mathrm{SiO}_4^{2-}]$ (µmol/L)", fontsize=16) 
        ax2.set_ylabel('Residuals', fontsize=14)
        ax2.grid(True)
    else:
        ax1.set_xlabel(r"Concentration of $[\mathrm{SiO}_4^{2-}]$ (µmol/L)", fontsize=16)

    plt.tight_layout()
    plt.show()