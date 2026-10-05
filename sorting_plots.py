"""
Gal Vishne 2025-2026
contact me at gal.vishne@gmail.com

Figures of the postprocessing pipeline (also usable on their own, e.g. in a notebook):
    single unit summaries, potential merge summaries, distributions of the unit metrics, classifier probabilities.
The functions only plot: they get the analyzer / unit metrics table (postprocessing_pipeline.get_unit_metrics) / labels they need,
and the file to save to. They don't read anything from the recording folders.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import spikeinterface.widgets as sw

# %% single units

def plot_single_unit(analyzer, unit_id):
    """
    Plot the single unit info for a given unit id.
    these are the included plots:
        sw.plot_unit_locations(sorting_analyzer, unit_ids=[unit_id], figure=fig)
        sw.plot_unit_templates(sorting_analyzer, unit_ids=[unit_id], figure=fig)
        sw.plot_unit_waveforms_density_map(sorting_analyzer, unit_ids=[unit_id], figure=fig, use_max_channel=True)  # I think this can give a hint of whether the unit is including multiple neurons or not
        sw.plot_autocorrelograms(sorting_analyzer, unit_ids = [unit_id], figure=fig)
        sw.plot_amplitudes(sorting_analyzer, unit_ids=[unit_id], figure=fig)
    may also be interesting:  sw.plot_isi_distribution(sorting_analyzer.sorting, unit_ids=[unit_id], figure=fig)
    """
    figsize = (12, 6) # notice this is in inches
    fig = plt.figure(figsize=figsize)
    sw.plot_unit_summary(analyzer, unit_id=unit_id, figure=fig)
    return fig

def export_sorting_single_units(analyzer, file_name, unit_ids=None, overwrite=False):
    """
    Create and save a multi-page PDF with one summary figure per unit.
    :param analyzer: SpikeInterface SortingAnalyzer
    :param file_name: output PDF file path
    :param unit_ids: list of unit IDs to plot (default: None, which means all units)
    :param overwrite: whether to overwrite if file already exists
    """
    if unit_ids is None:
        unit_ids = analyzer.unit_ids
    if os.path.exists(file_name) and not overwrite:
        raise FileExistsError(f"File '{file_name}' already exists. Use overwrite=True to replace it.")

    # create folder named filename
    os.makedirs(file_name.parent / file_name.stem, exist_ok=True)

    # fix bad unit locations (non-finite) -- this causes a bug also in plotting other units so we replace the analyzer with 'analyzer_good'
    ul = analyzer.get_extension("unit_locations").get_data()
    bad = ~np.all(np.isfinite(ul), axis=1)
    if np.any(bad):
        good_ids = analyzer.unit_ids[~bad]
        bad_ids = analyzer.unit_ids[bad]
        # Create an analyzer view containing only good units (keeps all extensions filtered)
        analyzer_use = analyzer.select_units(good_ids, format="memory")
    else:
        bad_ids = []
        analyzer_use = analyzer

    with PdfPages(file_name) as pdf:
        for unit_id in unit_ids:
            if unit_id not in bad_ids:
                fig = plot_single_unit(analyzer_use, unit_id)
            else:
                n_spikes = analyzer.sorting.get_unit_spike_train(unit_id=unit_id).size
                custom_text = f"No unit summary — unit location could not be computed,\nlikely due to too few spikes ({n_spikes})."
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.set_title(f"Unit {unit_id}", fontsize=18, fontweight="bold")
                ax.text(0.5, 0.5, custom_text, ha="center", va="center", fontsize=14, wrap=True)
            pdf.savefig(fig, dpi=25)  # Save the current figure to the PDF, with 25 DPI (used to be 100/300 but it was too large - probably need to redo those too)
            fig.savefig(file_name.parent / file_name.stem / f"unit_{unit_id}.jpeg")             # save the fig also as jpeg in the folder
            plt.close(fig)

# %% potential merges

def plot_potential_merge(analyzer, unit_ids, unit_metrics, sm_fields=None):
    """
    Create an A4 portrait summary figure for the a given suggested merge: (can change the proportions below from A4 to something else)
    - Crosscorrelogram matries (top-left)
    - Unit templates (top-right)
    - Amplitudes (middle)
    - Sorting metrics (bottom) - from unit_metrics (one row per unit, see postprocessing_pipeline.get_unit_metrics)
    """
    sm_df = unit_metrics
    if sm_fields is None:
        sm_fields = ['num_spikes', 'firing_rate', 'presence_ratio', 'snr',
                     'rp_contamination', 'rp_violations', 'amplitude_median',
                     'd_prime', 'nn_hit_rate', 'nn_miss_rate']
        # maybe also add: 'firing_range', 'isolation_distance', 'silhouette'
        sm_fields = [f for f in sm_fields if f in sm_df.columns]
        # add all fields that have "classifier" in them
        sm_fields += [f for f in sm_df.columns if ('classifier' in f.lower()) and (f not in sm_fields)]

    # Generate distinct colors
    palette = sns.color_palette("hls", len(unit_ids))  # or try "tab10", "Set2", etc.
    unit_colors = {uid: palette[i] for i, uid in enumerate(unit_ids)}

    figsize = (8.3, 11.7) # notice this is in inches
    fig = plt.figure(figsize=figsize)

    # proportions of the full height
    bottom, hspace, top = 0.02, 0.06, 0.9
    height = (top - bottom - 2 * hspace) / 3
    # proportions of the width in the top 2 thirds
    left, wspace, right = 0.05, 0.05, 0.95
    width = (right - left - wspace) / 2

    # --- Top third: Crosscorrelograms and Templates ---
    ccg_pos = [left, top-height, width, height]
    ccg_ax = fig.add_axes(ccg_pos)
    ccg_ax.axis("off")  # Hide the axis for the main plot
    #ccg_ax.set_title("Crosscorrelograms")
    ccg_ax.text(0.5, 1.06, "Crosscorrelograms", fontsize=12, va='bottom', ha='center')
    # make crosscorrelogram axes
    def make_ccg_axes(position, n, spacing=0.02):
        """
        Create a grid of axes for crosscorrelograms.
        Parameters:
        - position: [left, bottom, width, height] in figure coords
        - n: number of plots in each dimension
        - spacing: fraction of width/height between plots
        """
        left, bottom, width, height = position
        ax_w = (width - spacing * (n - 1)) / n
        ax_h = (height - spacing * (n - 1)) / n
        if ax_w <= 0 or ax_h <= 0:
            spacing = 0.01
            ax_w = (width - spacing * (n - 1)) / n
            ax_h = (height - spacing * (n - 1)) / n
        # Create grid positions (top to bottom)
        x0s, y0s = np.meshgrid(
            left + np.arange(n) * (ax_w + spacing),
            bottom + np.arange(n)[::-1] * (ax_h + spacing)  # flip y to top-bottom
        )
        # Create a 2D array of Axes
        axes = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                axes[i, j] = fig.add_axes([x0s[i, j], y0s[i, j], ax_w, ax_h])
        return axes
    axes = make_ccg_axes(ccg_pos, n=len(unit_ids))
    # add the ccgs
    sw.plot_crosscorrelograms(analyzer, unit_ids=unit_ids, axes=axes, unit_colors=unit_colors)
    for ax in axes.flatten():
        ax.title.set_fontsize(10)
        ax.tick_params(labelsize=8)

    ax_templates = fig.add_axes([left+width+wspace, top-height, width, height])
    sw.plot_unit_templates(analyzer, unit_ids=unit_ids, same_axis=True, ax=ax_templates, unit_colors=unit_colors)
    ax_templates.set_title("Unit Templates")

    # --- Middle third: Amplitudes ---
    ax_ampl = fig.add_axes([left, bottom+height+hspace, right-left, height])
    sw.plot_amplitudes(analyzer, unit_ids=unit_ids, ax=ax_ampl, unit_colors=unit_colors)
    ax_ampl.set_title("Amplitudes")

    # --- Bottom third: Sorting Metrics ---
    ax_text = fig.add_axes([left, bottom, 1, height])  # we can span the full width here (not crucial)
    ax_text.axis("off")
    text_str = sm_df.loc[unit_ids, sm_fields].to_string(line_width=110)
    ax_text.text(0, 1, text_str, fontsize=8, va='top', family='monospace')
    ax_text.set_title("Sorting Metrics")

    # Title + legend
    fig.suptitle("Summary for units:", fontsize=13)
    leg = fig.legends[0]  # this is from the templates (shows the unit ids and the colors)
    leg.set_bbox_to_anchor((0.5, 0.92))  # slightly below suptitle
    leg.set_loc("lower center")
    leg.set_frame_on(False)
    fig.legends[1].remove()  # this is from the templates + amplitudes, so it's redundant

    return fig

def export_merge_summary_plots(analyzer, merges_list, unit_metrics, file_name, overwrite=False):
    """
    Create and save a multi-page PDF with one summary figure per unit group.

    Parameters:
    - analyzer: SpikeInterface SortingAnalyzer
    - merges_list: list of lists of unit IDs
    - unit_metrics: one row per unit (see postprocessing_pipeline.get_unit_metrics)
    - file_name: output PDF file path
    - overwrite: whether to overwrite if file already exists
    """
    if os.path.exists(file_name) and not overwrite:
        raise FileExistsError(f"File '{file_name}' already exists. Use overwrite=True to replace it.")

    # create folder named filename
    os.makedirs(file_name.parent / file_name.stem, exist_ok=True)

    with PdfPages(file_name) as pdf:
        for merge_id, unit_ids in enumerate(merges_list):
            fig = plot_potential_merge(analyzer, unit_ids, unit_metrics)
            fig.savefig(file_name.parent / file_name.stem / f"merge_{merge_id}.jpeg")             # save the fig also as jpeg in the folder
            pdf.savefig(fig, dpi=150)  # Save with higher DPI for better quality
            plt.close(fig)

# %% unit metrics & classification

def plot_metrics(unit_metrics, metrics=None, file_name=None, overwrite=False, max_categories=10, n_cols=4):
    """
    Plots the distribution of each metric across units: a histogram for numeric metrics (e.g. 'snr', 'firing_rate'),
    and the count of units per value for categorical ones (e.g. KSLabel, the classifier predictions).
    :param unit_metrics: one row per unit (see postprocessing_pipeline.get_unit_metrics)
    :param metrics: list of metrics to plot (default: None - all numeric columns, and categorical ones with at most max_categories values)
    :param file_name: output file path (optional), otherwise returns the figure
    """
    if metrics is None:
        metrics = [m for m in unit_metrics.columns if pd.api.types.is_numeric_dtype(unit_metrics[m])
                   or unit_metrics[m].nunique() <= max_categories]
    metrics = [m for m in metrics if m in unit_metrics.columns and unit_metrics[m].notna().any()]

    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        values = unit_metrics[metric].dropna()
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            sns.histplot(values, bins=30, ax=ax)
        else:  # categorical: number of units per value
            counts = values.astype(str).value_counts()
            ax.bar(counts.index, counts.values)
            ax.tick_params(axis='x', labelrotation=45)
        ax.set_title(metric)
        ax.set_xlabel('')
        ax.set_ylabel('Count')
    for ax in axes.flat[len(metrics):]:  # empty panels
        ax.axis('off')
    fig.tight_layout()

    if file_name:
        if os.path.exists(file_name) and not overwrite:
            raise FileExistsError(f"{file_name} already exists. Use `overwrite=True` to overwrite.")
        fig.savefig(file_name)
        plt.close(fig)
    else:
        return fig

def prob_cutoff_xtab(labels, cutoff=0.6):
    # create a cross-tabulation of the predictions and the probabilities
    xtab = pd.crosstab(labels['prediction'], labels['probability'] > cutoff)
    xtab.columns = ['n_low_prob', 'n_high_prob']
    xtab.loc['total'] = xtab.sum(axis=0)     # add total row
    xtab.loc[:, 'total'] = xtab.sum(axis=1)     # add total column
    xtab['prop_low_prob'] = xtab['n_low_prob'] / xtab['total'] * 100
    xtab['prop_high_prob'] = xtab['n_high_prob'] / xtab['total'] * 100
    return xtab

def plot_classification(labels, cutoff=0.6, file_name=None, overwrite=False):
    """
    Plot the classification results - distribution of probabilities for each given class.
    labels: pd DataFrame with fields ['prediction', 'probability']
    cutoff: float, probability threshold for classification (optional)
    file_name: str, path to save the plot (optional), otherwise returns the FacetGrid object
    """
    class_names = labels['prediction'].unique()
    g = sns.FacetGrid(labels, col='prediction', hue='prediction', sharex=True, sharey=True, col_order=class_names, hue_order=class_names)
    g.map(sns.histplot, 'probability')
    # add to the title the number of units in each class
    for ax, class_name in zip(g.axes.flat, class_names):
        n_units = len(labels[labels['prediction'] == class_name])
        ax.set_title(f"{class_name} ({n_units} units)")
        if cutoff is not None:
            ax.axvline(x=cutoff, color='red', linestyle='--', label='Cutoff')
            # add text mentioning the number of units above the cutoff
            prob_cutoff = prob_cutoff_xtab(labels, cutoff=cutoff)
            txt = 'prob > %0.2g\n#%d, %%%0.1f' % (cutoff, prob_cutoff.loc[class_name, 'n_high_prob'], prob_cutoff.loc[class_name, 'prop_high_prob'])
            ax.text(1, 1, txt, transform=ax.transAxes, fontsize=8, color='red', ha='right', va='top')
    if file_name is not None:
        if os.path.exists(file_name) and not overwrite:
            raise FileExistsError(f"File '{file_name}' already exists. Use overwrite=True to replace it.")
        g.figure.savefig(file_name)
    else:
        return g

# %% all sorting plots of a stage

def export_sorting_plots(analyzer, unit_metrics, figures_folder, overwrite=False):
    """
    Export the sorting plots to a folder: single unit summaries & the distributions of the unit metrics.
    """
    export_sorting_single_units(analyzer, file_name=figures_folder / 'single_units_summary.pdf', overwrite=overwrite)
    plot_metrics(unit_metrics, file_name=figures_folder / 'sorting_metrics.pdf', overwrite=overwrite)
