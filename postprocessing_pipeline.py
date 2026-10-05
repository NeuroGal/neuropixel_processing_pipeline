"""
Gal Vishne 2025-2026
contact me at gal.vishne@gmail.com

Step 2: Kilosort output -> SpikeInterface sorting analyzers, in three stages
(each saved under postprocessing/<stage>_sorting with analyzer, classification, figures, settings subfolders):
    original - the Kilosort4 units, with all analyzer extensions (waveforms, templates, quality & template metrics, ...)
    merged   - units merged automatically (iterative auto-merge, see compute_potential_merges)
    cleaned  - merged units without the ones classified as noise by UnitRefine
Each stage is also classified (UnitRefine noise/neural + SUA/MUA), plotted (sorting_plots.py), and exported to phy.
Run from the command line, e.g.:
    python postprocessing_pipeline.py --monkeys Jones --dates 20260702 --regions Pul LIP
"""
# %% imports

import spikeinterface as si
import spikeinterface.curation as sc
from spikeinterface.exporters import export_to_phy as si_export_to_phy
from pathlib import Path
import numpy as np
import pandas as pd
import os
import networkx as nx
from sklearn.tree import export_text
from dataclasses import dataclass, field
import config
import warnings
from sorting_plots import export_sorting_plots, export_merge_summary_plots, plot_classification
from pipelines_functions import (RecordingID, log_or_print, save_json, delete_outputs, is_nonempty_dir, require_output, require_no_output,
                                 update_info_df, copy_to_server, server_sync_enabled, run_batch_pipeline, main_pipeline_parser)

# %% parameters

@dataclass
class PostprocessingParams:
    """
    All parameters of the postprocessing pipeline, in one place (the reasoning behind each is in the docstrings of the functions using them).
    The defaults are what we use; to change any of them for your setup, set e.g. POSTPROCESSING_PARAMS = dict(noise_cutoff=0.6) in local_config.py.
    """
    # analyzer extensions (see compute_analyzer_extensions); can use analyzer.get_computable_extensions()
    extensions: list = field(default_factory=lambda: [
        'random_spikes', 'correlograms', 'isi_histograms', 'noise_levels',
        'waveforms', 'templates', 'principal_components',
        'amplitude_scalings', 'spike_amplitudes', 'spike_locations',
        'template_metrics', 'template_similarity', 'unit_locations', 'quality_metrics'])
    # this is all for now, the rest use defaults (multi_channel_metrics are needed for classification)
    extension_params: dict = field(default_factory=lambda: dict(template_metrics=dict(include_multi_channel_metrics=True)))
    # auto-merge presets and their step parameters (see compute_potential_merges)
    merge_presets: dict = field(default_factory=lambda: dict(
        similarity_correlograms=dict(min_spikes=0, contamination_thresh=np.inf, refractory_period_ms=1.0, censored_period_ms=0.0, max_distance_um=150,
                                     template_diff_thresh=0.25, corr_diff_thresh=0.25, censor_correlograms_ms=0, sigma_smooth_ms=0.5,
                                     adaptative_window_thresh=0.5, firing_contamination_balance=1.5),
        x_contaminations=dict(min_spikes=0, contamination_thresh=np.inf, refractory_period_ms=1.8, censored_period_ms=0.0, max_distance_um=150,
                              template_diff_thresh=0.22, cc_thresh=0.05, p_value=0.2, firing_contamination_balance=1.5)))
    n_merge_rounds: int = 3            # merges are suggested & applied iteratively (see do_iterative_merges)
    merging_mode: str = 'soft'         # see do_sorting_merges ("hard" was never used)
    sparsity_overlap: float = 0.15     # used to be 0.25 but saw cases where there was a good match and it was even lower (electrode movement)
    # UnitRefine classifiers (see get_repo_id / https://huggingface.co/collections/SpikeInterface/curation-models-66ebf53d7f17088991062c2c;
    # paper: Jain et al. 2025, https://www.biorxiv.org/content/10.1101/2025.03.30.645770v1)
    classifier_repos: dict = field(default_factory=lambda: dict(noise_neural="SpikeInterface/UnitRefine_noise_neural_classifier",
                                                                sua_mua="SpikeInterface/UnitRefine_sua_mua_classifier"))
    noise_cutoff: float = 0.5          # units classified as noise with probability > noise_cutoff are removed in the cleaned stage
    classification_plot_cutoff: float = 0.6   # only for the classification plots

def load_postprocessing_params():
    """PostprocessingParams with the overrides from POSTPROCESSING_PARAMS in the config."""
    return PostprocessingParams(**config.POSTPROCESSING_PARAMS)

# %% create sorting analyzer & compute extensions

def make_sorting_analyzer(recording, sorting, logger=None, data_location='local'):
    """
    Create a sorting analyzer for a given sorting and recording.
    Saved under: session_path/postprocessing/original_sorting/analyzer
    """
    # annotations aren't saved with the analyzer - load_sorting_analyzer adds the recording keys & sorting_type again when loading
    sorting.annotate(sorting_type='original')

    # Create the sorting analyzer
    sorting_analyzer_folder = RecordingID.from_annotations(sorting).folder('original_sorting/analyzer', location=data_location, mkdir=False)
    require_no_output(sorting_analyzer_folder, 'original sorting analyzer')
    sorting_analyzer_folder.parent.mkdir(parents=True, exist_ok=True)
    log_or_print('Creating sorting analyzer. May take some time (sparseness computations).', logger=logger)
    sorting_analyzer = si.create_sorting_analyzer(recording=recording, sorting=sorting, folder=sorting_analyzer_folder, format="binary_folder")  # sparse=True by default
    sorting_analyzer.sorting.annotate(sorting_type='original')  # for this session only (not saved, see above)
    return sorting_analyzer

def load_sorting_analyzer(rec_id, sorting_type='original', logger=None, data_location='local'):
    """
    Load a sorting analyzer.
    """
    analyzer_folder = rec_id.folder(f'{sorting_type}_sorting/analyzer', location=data_location, mkdir=False)
    require_output(analyzer_folder, f'{sorting_type} sorting analyzer', 'run postprocessing_pipeline.py')
    sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_folder)
    log_or_print(f'Loading {sorting_type} sorting analyzer for {rec_id}.', logger=logger)
    sorting_analyzer.sorting.annotate(**rec_id.kwargs, sorting_type=sorting_type)
    return sorting_analyzer

def compute_analyzer_extensions(analyzer, extensions=None, overwrite=False, logger=None, save_settings=True, data_location='local', params=None):
    """
    Compute SpikeInterface analyzer extensions, optionally skipping existing ones and saving settings.

    Parameters:
    - analyzer: SortingAnalyzer object
    - extensions: list of extensions to compute (default: all)
    - overwrite: whether to recompute even if already computed
    - logger: optional logger for info/warning messages
    - save_settings: whether to save the settings to a JSON file

    See for more details on the extensions:
        https://spikeinterface.readthedocs.io/en/stable/modules/postprocessing.html
    some of these seem like repetitions of kilosort, but there it's done with the precomputed templates, so this is more accurate.

    We compute everything!
        For classification we need the following extensions:
            - quality_metrics, relies on:
                - templates > waveforms > random_spikes
                - noise_levels
                - principal_components
                - spike_locations (drift metrics)
                - spike_amplitudes (not strictly mandatory, but highly recommended before some amplitude related metrics) - we do this even though it's a very time consuming step
            - template_metrics (*including multi-channel metrics), relies on:
                - templates > waveforms > random_spikes
        For merges we also need:
            - template_similarity
            - unit_locations
            - correlograms
            (spike_locations & spike_amplitudes are needed in some presets, but not what I'm using currently)
        which leaves:
            - isi_histograms - not that time consuming, and may be useful for plots
            - amplitude_scalings - computed anyway with spike_amplitudes (and may be needed for some metrics / plots, I didn't check in detail)
        These are still computed for plots, but we can consider removing them in the future.

    A lot of these can be configured by multiple parameters. I mostly stayed with the defaults, but it may be worth checking them out (some below, not all).
    In particular:
    - for template metrics see: si.postprocessing.template_metrics.get_default_tm_params()
    - for quality metrics see: si.qualitymetrics.get_default_qm_params()
    - correlograms & isi_histograms use: window_ms = 50, bin_ms = 1 by default.
    - waveforms: ms_before=1, ms_after=2 by default (configurable, e.g. ms_before=2, ms_after=5)
    - PCA: 2 Parameters - n_components & mode: --- this is different than the KS default, but this is actually more accurate I think
        • “by_channel_local” (default): fits one PCA model for each by_channel
        • “by_channel_global”: fits the same PCA model to all channels (also termed temporal PCA)
        • “concatenated”: concatenates all channels and fits a PCA model on the concatenated data
    - might be worth changing some settings to get more info about nn_metrics
        (not crucial, see for more details: https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics/nearest_neighbor.html)
    """

    params = params or load_postprocessing_params()
    all_extensions = params.extensions
    all_extensions_params = params.extension_params

    if (extensions is None) or (extensions == 'all'):
        compute_extensions = all_extensions
    else:
        if isinstance(extensions, str):
            extensions = [extensions]
        if not isinstance(extensions, list):
            raise ValueError("extensions should be a list of strings.")
        compute_extensions = [ext for ext in extensions if ext in all_extensions]

    if not overwrite:
        to_compute = [ext for ext in compute_extensions if not analyzer.has_extension(ext)] # maybe use analyzer.get_loaded_extension_names()
        skipped = [ext for ext in compute_extensions if analyzer.has_extension(ext)]
        if skipped:
            log_or_print(f"Skipping already computed extensions: {skipped}", logger=logger)
        if to_compute:
            log_or_print(f"Computing new extensions: {to_compute}.", logger=logger)
        if skipped and to_compute: # there are some extensions that are already computed and some that are not
            warnings.warn("Note that children of extensions that are computed now will be deleted automatically to avoid inconsistencies.")
        compute_extensions = to_compute
    else:
        log_or_print(f"Computing extensions (overwriting if already computed): {compute_extensions}", logger=logger)

    # do the actual computation
    compute_extensions_params = {ext: params for ext, params in all_extensions_params.items() if ext in compute_extensions}
    # the amplitude group goes through the recording together (one pass instead of three)
    amplitude_group = [ext for ext in ['amplitude_scalings', 'spike_amplitudes', 'spike_locations'] if ext in compute_extensions]
    # On Windows, the parallel workers of this group get their inputs (incl. all spikes) through a pipe, which fails for large sortings
    # (OSError: [Errno 22] / "pickle data was truncated" - seen with 21M spikes). Threads share memory, so nothing goes through a pipe.
    amplitude_job_kwargs = dict(pool_engine='thread') if os.name == 'nt' else {}
    amplitude_group_ran = False
    for ext in compute_extensions:  # doing it one by one for documentation, except amplitude group
        if ext in amplitude_group:
            if not amplitude_group_ran:
                ext_params = {}
                for group_ext in amplitude_group:
                    ext_params = {**ext_params, **compute_extensions_params.get(group_ext, {})}
                log_or_print(f"Computing extensions: {amplitude_group} with params: {ext_params} {amplitude_job_kwargs}", logger=logger)
                analyzer.compute(amplitude_group, **ext_params, **amplitude_job_kwargs)
                log_or_print(f"Done with extensions: {amplitude_group}.", logger=logger)
                amplitude_group_ran = True
        else:
            ext_params = compute_extensions_params.get(ext, {})
            log_or_print(f"Computing extension: {ext} with params: {ext_params}", logger=logger)
            analyzer.compute(ext, **ext_params)
            log_or_print(f"Done with extension: {ext}.", logger=logger)

    settings_dict = dict(extensions=compute_extensions, extension_params=compute_extensions_params)

    if save_settings:
        settings_folder = RecordingID.from_annotations(analyzer.sorting).folder(f"{analyzer.sorting.get_annotation(key='sorting_type')}_sorting/analyzer", location=data_location)
        save_json(settings_dict, settings_folder / 'extension_settings.json', keep_existing=True, logger=logger)
    #return analyzer -- no need, it's all saved

def get_unit_metrics(analyzer, data_location='local', include=('quality', 'template', 'classifier', 'sorting')):
    """
    One table with everything known about each unit (index: unit ids), used for the figures and for the units table of the task data.
    include: which parts to add -
        'quality'    - quality metrics (presence ratio, firing rate, snr, ...)
        'template'   - template metrics (peak to valley, half width, ...)
        'classifier' - UnitRefine labels from the classification folder of this stage: <classifier>_classifier_prediction / _probability
                       (e.g. noise_neural_classifier_prediction). Skipped for analyzers that are not a saved stage (e.g. in the middle of merging).
        'sorting'    - properties of the sorting object (e.g. Kilosort's KSLabel, ContamPct)
    Parts that weren't computed are left out. Units missing from a part (e.g. new merged units in the classifier labels) get NaN.
    """
    unit_ids = analyzer.sorting.get_unit_ids()
    parts = []
    for part, extension in [('quality', 'quality_metrics'), ('template', 'template_metrics')]:
        if part in include and analyzer.has_extension(extension):
            parts.append(analyzer.get_extension(extension).get_data())

    sorting_type = analyzer.sorting.get_annotation(key='sorting_type')
    if 'classifier' in include and sorting_type is not None:
        classifier_folder = RecordingID.from_annotations(analyzer.sorting).folder(f'{sorting_type}_sorting/classification', location=data_location, mkdir=False)
        for f in sorted(classifier_folder.glob('*_classifier_labels.csv')):
            labels = pd.read_csv(f, index_col=0)  # index: unit ids
            prefix = f.stem[:-len('_labels')]     # e.g. noise_neural_classifier
            parts.append(labels.add_prefix(f'{prefix}_'))

    if 'sorting' in include:
        properties = {}
        for key in analyzer.sorting.get_property_keys():
            values = np.asarray(analyzer.sorting.get_property(key))
            if values.ndim == 1:
                properties[key] = values
            else:  # e.g. locations - one column per dimension
                properties.update({f'{key}_{i}': values[:, i] for i in range(values.shape[1])})
        parts.append(pd.DataFrame(properties, index=unit_ids))

    unit_metrics = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=unit_ids)
    unit_metrics = unit_metrics.loc[:, ~unit_metrics.columns.duplicated()]   # e.g. classifier labels also saved as sorting properties
    return unit_metrics.reindex(unit_ids).rename_axis('unit_id')

# %% merges

def compute_potential_merges(analyzer, logger=None, save_settings=False, settings_file=None, params=None):
    """
    Compute potential units to merge - see https://spikeinterface.readthedocs.io/en/stable/modules/curation.html.
    *** Notice this function is under development and the API might change in the future. beware of this if changing versions! ***

    * they recommend using sc.compute_merge_unit_groups but,
     a) it seemed to be ignoring some params (e.g. template_diff_thresh) and
     b) it didn't export outputs when I set extra_outputs=True,
    so I'm using the old sc.get_potential_auto_merge. (note that if the new function is used you simply need to insert step_params=dict() in the function call, not as I do below)

    I am using two function presets:
        - 'similarity_correlograms' - uses the following steps:
            (1) "num_spikes" - removes units with n_spikes < `min_spikes`. The default here is 100, but I change it to 0, effectively removing this step.
                                I found it doesn't matter for the correlograms preset (correlograms for units with very few spikes don't look like anything anyway)
            (2) "remove_contaminated" - removes units with a contamination score > `contamination_thresh`.
                                The default is 0.2, but I set it to np.inf, effectively removing this step as well (which may matter more than the min_spikes, but I found it's fine for now - merges contaminated with other contaminated only).
                                Two more settings which matter here: `refractory_period_ms` (default 1.0) and `censored_period_ms` (default 0.3). These are used to compute the contamination score (what period should be clean, and what period we don't regard respectively)
                                I change the first one only for the second preset, which I will describe below, and the second one I change always to 0.
            (3) "unit_locations" - only suggest pairs that are close enough in space (< `max_distance_um`, default 150, kept as is).
            (4) "template_similarity" - only suggest pairs that are similar enough in template shape (< `template_diff_thresh`, default 0.25, kept as is here, but not in the second preset).
            (5) "correlogram" - if I recall correctly this filters only pairs that are similar enough in the cross-correlogram to individual correlograms (but worth double checking).
                                relevant parameters are `corr_diff_thresh` (default 0.16, I relax this to 0.25 and it still works well)
                                                        `censor_correlograms_ms` (default 0.15, I set to 0) [I assume this means to ignore part of the correlogram, I don't see why]
                                                        `sigma_smooth_ms` (default 0.6, I change to 0.5, doesn't matter much) [smoothing parameter for the correlogram before similarity is computed]
                                                        `adaptative_window_thresh` (default 0.5, don't change) [don't remember what this is]
            (6) "quality_score" - only keep merges where the merged quality score increases relative to the original units (need to dig into how this is computed).
                                  used parameters: `firing_contamination_balance` - how much care about increase in FR relative to increase in contamination (default 1.5) - need to dig into this more
                                                   `refractory_period_ms`, `censored_period_ms` - same as used for remove contaminated, so it matters here too.
        Overall I found this to be pretty good and has barely any false positives (suggests merges that are not good), but it misses a lot, so that's why I added the second preset.

        - 'x_contaminations' - uses the following steps:
            (1) "num_spikes" - as above (changing the default to 0 currently). Here it might be smart to remove units with very few spikes, because they are often not contaminating other units, but then it can bridge two other units with high FR
                               (e.g. unit A has few spikes and is suggested to be merged with B and with C -> contamination when merging B and C). Currently I try to avoid this by other ways, but it's worth remembering.
            (2) "remove_contaminated" - as above (parameters here: `contamination_thresh`=np.inf, `refractory_period_ms`=1.8 (different than above!), `censored_period_ms`=0).
            (3) "unit_locations" - as above (`max_distance_um`=150).
            (4) "template_similarity" - as above (`template_diff_thresh`=0.22 here to be a bit stricter cause this preset is more liberal with merges).
            (5) "cross_contamination" - this is the main step of this preset, and it uses the following parameters: `cc_thresh` (default is 0.1, I change to 0.05), `p_value` (default is 0.2), `refractory_period_ms`, `censored_period_ms`
                 The step checks if merging the two units generates contamination that is significantly higher than `cc_thresh`, and takes only pairs of units where this is not significant, with p_value > `p_value` (would have been more appropriate to use something bayesian, but this is how it's done here...)
                 `refractory_period_ms`, `censored_period_ms` are the same parameters used in "remove_contaminated" and "quality_score" steps.
                 * here I found that using the default refractory period of 1.0 is not good, because it allows a lot more contamination than the correlograms have before merging, so I increased it to 1.8 for this step, and we can increase even a bit more.
            (6) "quality_score" - as above (parameters here: `firing_contamination_balance`=default=1.5, `refractory_period_ms`=1.8, `censored_period_ms`=0).
        This preset is more liberal, and it may still do a few merges that are not ideal with these parameters, but I decided it's worth it with the increase in good merges it brings.

    for more information about other function presets and parameters see:
        https://github.com/SpikeInterface/spikeinterface/blob/12b3470c9c63131eebecd0fe4008c6030e29c6c0/src/spikeinterface/curation/auto_merge.py#L90
    """
    params = params or load_postprocessing_params()
    steps_params_similc = params.merge_presets['similarity_correlograms']
    steps_params_xcont = params.merge_presets['x_contaminations']

    log_or_print("Computing potential merges with 'similarity_correlograms' preset.", logger=logger)
    merges_similc = sc.get_potential_auto_merge(sorting_analyzer=analyzer, resolve_graph=True, preset='similarity_correlograms', **steps_params_similc)
    log_or_print(f"Found {str(len(merges_similc))} potential merges with 'similarity_correlograms'", logger=logger)
    log_or_print("Computing potential merges with 'x_contaminations' preset.", logger=logger)
    merges_xcont = sc.get_potential_auto_merge(sorting_analyzer=analyzer, resolve_graph=True, preset='x_contaminations', **steps_params_xcont)
    # resolve_graph=True ---> instead of just exporting the pairs that were suggested to merge, it will resolve the graph and suggest the groups of units to merge (e.g. if A-B and B-C are suggested, it will suggest A-B-C).
    log_or_print(f"Found {str(len(merges_xcont))} potential merges with 'x_contaminations'", logger=logger)

    # combine suggestions from both presets
    def combine_mgs(mg1, mg2):
        G = nx.Graph()                   # Create a graph
        for group in mg1 + mg2:          # Add edges from both lists
            G.add_edges_from((group[i], group[i + 1]) for i in range(len(group) - 1))
        # Extract connected components as merged groups
        mgs_comb = [sorted(comp) for comp in nx.connected_components(G)]
        mgs_comb.sort(key=lambda x: (x[0], x))
        return mgs_comb

    potential_merges = combine_mgs(merges_similc, merges_xcont)
    log_or_print(f"Total: {str(len(potential_merges))} merges.", logger=logger)

    if save_settings:
        save_json(dict(steps_params_similc=steps_params_similc, steps_params_xcont=steps_params_xcont, merges_similc=merges_similc, merges_xcont=merges_xcont),
                  settings_file, keep_existing=True, logger=logger)
    return potential_merges

def do_sorting_merges(analyzer, merge_groups, logger=None, save_settings=False, settings_file=None, params=None):
    """
    create a new sorting analyzer with units merged by the given groups.
    see https://spikeinterface.readthedocs.io/en/stable/api.html (under merge_units) for more details

    - merging_mode parameter can be set to "soft" (default) or "hard".
        "soft" mode will try to smartly combine the existing extension data (e.g. templates, template similarity, etc.) to estimate the merged units' data, when possible.
            This is the fastest mode, but it can be less accurate.
        "hard" mode will simply merge the spike trains of the units and recompute the extensions on the merged spike train.
            This is more accurate but slower, especially for the extensions that need to traverse the raw data (e.g., spike amplitudes, spike locations, amplitude scalings, etc.).
    ---> I use soft for now, hopefuly it works well enough.

    - sparsity_overlap parameter is used to determine how much overlap is allowed between the units that are merged (relevant only for soft mode, default is 0.75, but I saw totally reasonable cases where it was less, so changing it to 0.25).
    - If censored_period_ms is set, the function will remove spikes that are too close in time after the merge. -- consider using this.
    """
    params = params or load_postprocessing_params()
    merge_settings = dict(merge_unit_groups=merge_groups, merging_mode=params.merging_mode, sparsity_overlap=params.sparsity_overlap)
    log_or_print(f"Merging units in the analyzer, using merge settings: {merge_settings}", logger=logger)
    analyzer_merged = analyzer.merge_units(**merge_settings)
    # default is to save to memory, unless a different format is given (e.g. format='binary_folder', folder=sorting_path / 'interim_sorting_analyzer_output')

    # fix up <NA> to np.nan so the classifier can work with it - maybe only do this after the final step, also, I need to verify it is saved correctly
    for ext in ['quality_metrics', 'template_metrics']:
        analyzer_merged.get_extension(ext).data['metrics'][analyzer_merged.get_extension(ext).data['metrics'].isna()] = np.nan
    if save_settings:
        save_json(merge_settings, settings_file, keep_existing=True, logger=logger)

    return analyzer_merged

def do_iterative_merges(analyzer, logger=None, save_settings=True, data_location='local', params=None):
    """
    * maybe in the future interleve this with removal of clear noise units
    Compute potential merges (and merge them) iteratively params.n_merge_rounds times (default=3).
    Currently using the same parameters for all iterations, so the only reason new merges may be suggested is based on the new merged units.
    """
    rec_id = RecordingID.from_annotations(analyzer.sorting)
    require_no_output(rec_id.folder('merged_sorting', location=data_location, mkdir=False), 'merged sorting')
    merges_settings_folder = rec_id.folder('merged_sorting/settings', location=data_location, mkdir=save_settings)
    merges_figures_folder = rec_id.folder('merged_sorting/figures', location=data_location)
    params = params or load_postprocessing_params()
    n_rep = params.n_merge_rounds
    analyzer_in = analyzer
    for rep in range(n_rep):
        log_or_print(f"Doing merge {rep+1}/{n_rep}.", logger=logger)
        potential_merges = compute_potential_merges(analyzer=analyzer_in, logger=logger, save_settings=save_settings, settings_file=merges_settings_folder / f'potential_merges_info_rep_{rep+1}.json', params=params)
        if len(potential_merges) == 0:
            log_or_print("No potential merges found.", logger=logger)
            break
        else:
            log_or_print("Exporting potential merges summary plots.", logger=logger)
            export_merge_summary_plots(analyzer=analyzer_in, merges_list=potential_merges, unit_metrics=get_unit_metrics(analyzer_in, data_location=data_location),
                                       file_name=merges_figures_folder / f'potential_merges_summary_rep_{rep+1}.pdf')
            analyzer_merged = do_sorting_merges(analyzer=analyzer_in, merge_groups=potential_merges, logger=logger, save_settings=save_settings, settings_file=merges_settings_folder / f'merge_sorting_settings_rep_{rep+1}.json', params=params)
            analyzer_merged.sorting.annotate(**rec_id.kwargs)
            analyzer_in = analyzer_merged
    # save the merged analyzer (it stays in memory otherwise)
    analyzer_in.sorting.annotate(sorting_type='merged')  # for this session only (annotations aren't saved; added again when loading)
    merged_analyzer_folder = rec_id.folder('merged_sorting/analyzer', location=data_location, mkdir=False)
    log_or_print(f"Saving merged analyzer to {merged_analyzer_folder}.", logger=logger)
    analyzer_in.save_as(format='binary_folder', folder=merged_analyzer_folder)
    return analyzer_in

# %% classification

def classify_units(analyzer, repo_id):
    """
    Using a pre-trained classifier to classify the units, based on sorting of multiple neuropixel sessions by the same sorters
        (see here: https://spikeinterface.readthedocs.io/en/latest/how_to/auto_curation_prediction.html)
    * don't forget to cite them! * (Jain et al. 2025, UnitRefine, https://www.biorxiv.org/content/10.1101/2025.03.30.645770v1)

    repo_id: recommended to give the precise repo id of the classifier you want to use, e.g.:
        - SpikeInterface/UnitRefine_noise_neural_classifier
        - AnoushkaJain3/noise_neural_classifier
    but it is also possible to use 'noise_neural' or 'sua_mua' (see below).
    """
    if repo_id in ['noise_neural', 'sua_mua']:
        repo_id = get_repo_id(repo_id)
        warnings.warn(f"Using repo_id: {repo_id} for classification.")
    # warnings.simplefilter("ignore", category=UserWarning) # ignore InconsistentVersionWarning
    labels = sc.auto_label_units(sorting_analyzer=analyzer, repo_id=repo_id, trust_model=True, enforce_metric_params=True)
    return labels

def get_repo_id(c_type, params=None):
    """
    Get the repo id for the classifier based on the type of classifier. Possibe to switch this to one of these repos (as of writing this):
        noise_neural:
        - SpikeInterface/UnitRefine_noise_neural_classifier
        - AnoushkaJain3/noise_neural_classifier
        - AnoushkaJain3/noise_neural_classifier_lightweight
        sua_mua:
        - SpikeInterface/UnitRefine_sua_mua_classifier
        - AnoushkaJain3/sua_mua_classifier
        - AnoushkaJain3/sua_mua_classifier_lightweight
    (see here: https://huggingface.co/collections/SpikeInterface/curation-models-66ebf53d7f17088991062c2c)
    """
    params = params or load_postprocessing_params()
    if c_type not in params.classifier_repos:
        raise ValueError(f"Unknown classifier type: {c_type}.")
    return params.classifier_repos[c_type]

def get_classifier_info(repo_id, print_properties=False, save_properties=False, file_name=None):
    """
    Gets the classifier info from the HuggingFace repo & print / export some information about it.
    """
    # warnings.simplefilter("ignore", category=UserWarning) # ignore InconsistentVersionWarning
    model, model_info = sc.load_model(repo_id=repo_id, trust_model=True)
    model_properties = dict(**model_info, repo_id=repo_id, feature_names_in=model.feature_names_in_, classes=model.classes_, named_steps=model.named_steps)

    # Loop over all trees for all details
    rf = model.named_steps['classifier']
    # rf.estimators_
    rf_details = ''
    for i, tree in enumerate(rf.estimators_):
        rf_details += f"\n--- Tree {i} ---\n"
        tree_rules = export_text(tree, feature_names=model.feature_names_in_)
        rf_details += tree_rules

    # Get the feature importances
    importances = rf.feature_importances_
    feature_importances = pd.DataFrame({'feature': model.feature_names_in_, 'importance': importances})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    if print_properties:
        print(model_properties)
        print(feature_importances)
        print(rf_details)

    if save_properties:
        if file_name is None:
            raise ValueError("file_name is None, please provide a file to save the properties.")
        else: # Ensure file_name is a Path object and strip suffix
            base_path = Path(file_name).with_suffix('')
            def suffixer(s):
                return base_path.with_name(base_path.name + s)
            # named_steps (the sklearn pipeline) is saved as its text description - not loadable as a model (maybe just save the model itself?)
            save_json(model_properties, suffixer('_model_properties.json'))
            feature_importances.to_csv(suffixer('_feature_importances.csv'), index=False)
            with open(suffixer('_trees_details.txt'), 'w') as f:
                f.write(rf_details)

    return model_properties, feature_importances

def do_classification(analyzer, logger=None, save_settings=True, export_plot=True, data_location='local', params=None):
    """
    Classify the units in the sorting using the given classifier types. & save the labels in the sorting object and as csv, alongside the settings & plot.
    """
    rec_id = RecordingID.from_annotations(analyzer.sorting)
    sorting_type = analyzer.sorting.get_annotation(key='sorting_type')
    classification_folder = rec_id.folder(f'{sorting_type}_sorting/classification', location=data_location, mkdir=False)
    require_no_output(classification_folder, f'{sorting_type} sorting classification')
    classification_folder.mkdir(parents=True, exist_ok=True)
    all_labels = {}
    params = params or load_postprocessing_params()
    for c_type in params.classifier_repos:
        repo_id = get_repo_id(c_type, params=params)
        log_or_print(f"Classifying units with {c_type} classifier (repo_id: {repo_id}).", logger=logger)
        labels = classify_units(analyzer, repo_id=repo_id)
        # the labels are automatically saved as properties in the sorting object with "classifier_label", "classifier_probability" keys - adjust these to indicated which classifier
        analyzer.sorting.set_property(key=f'{c_type}_classifier_label', values=labels['prediction'])
        analyzer.sorting.set_property(key=f'{c_type}_classifier_probability', values=labels['probability'])
        analyzer.sorting.delete_property(key='classifier_label')
        analyzer.sorting.delete_property(key='classifier_probability')
        # also save the labels as csv in the classification folder
        labels.to_csv(classification_folder / f'{c_type}_classifier_labels.csv', index=True)
        all_labels[c_type] = labels
        log_or_print(f"Classifier {c_type} labels saved to {classification_folder / f'{c_type}_classifier_labels.csv'}.", logger=logger)
        if export_plot:
            file_name = classification_folder / f'{c_type}_labels_prob.pdf'  # maybe move to jpeg
            plot_classification(labels, cutoff=params.classification_plot_cutoff, file_name=file_name, overwrite=False)
            log_or_print(f"Classifier {c_type} labels plot saved to {file_name}.", logger=logger)
        # also save the classifier info in the folder
        if save_settings:
            settings_base_name = classification_folder / f'{c_type}_classifier_settings'
            get_classifier_info(repo_id=repo_id, print_properties=False, save_properties=True, file_name=settings_base_name)
            log_or_print(f"Classifier {c_type} settings saved to {settings_base_name}_* (several files).", logger=logger)
    return all_labels

# %% cleaning noise units

def do_noise_units_removal(analyzer, logger=None, save_settings=True, data_location='local', params=None):
    """
    Remove noise units from the sorting analyzer based on the given cutoff.
    :param analyzer: SpikeInterface SortingAnalyzer
    params.noise_cutoff: cutoff for noise units (e.g. 0.6) --> anything that is classified as noise with probability > cutoff will be removed. Default is 0.5.
    """
    params = params or load_postprocessing_params()
    noise_cutoff = params.noise_cutoff
    rec_id = RecordingID.from_annotations(analyzer.sorting)
    require_no_output(rec_id.folder('cleaned_sorting', location=data_location, mkdir=False), 'cleaned sorting')

    # core
    classifier_file = rec_id.folder(f"{analyzer.sorting.get_annotation(key='sorting_type')}_sorting/classification", location=data_location) / 'noise_neural_classifier_labels.csv'
    if not classifier_file.exists():
        raise ValueError(f"Classifier data not found in {classifier_file}. Please run classification first.")
    classifier_data = pd.read_csv(classifier_file)
    if len(classifier_data) != len(analyzer.sorting.unit_ids):
        raise ValueError(f"Classifier file {classifier_file} has {len(classifier_data)} units but the analyzer has {len(analyzer.sorting.unit_ids)}. Rerun the classification.")
    noise_neural_labels = classifier_data['prediction'].values   # same order as analyzer.sorting.unit_ids (saved from the same analyzer)
    noise_neural_prob = classifier_data['probability'].values
    # if 'noise_neural_classifier_label' not in analyzer.sorting.get_property_keys():
    #     raise ValueError("Sorting does not include noise classifier labels. Please run classification first.")
    # log_or_print("Loading noise classifier labels from the sorting object.", logger=logger)
    # noise_neural_labels = analyzer.sorting.get_property(key='noise_neural_classifier_label')
    # noise_neural_prob = analyzer.sorting.get_property(key='noise_neural_classifier_probability')
    unit_ids = analyzer.sorting.unit_ids
    remove_ids = [unit_id for unit_id, label, prob in zip(unit_ids, noise_neural_labels, noise_neural_prob) if ((label=='noise') and (prob > noise_cutoff))]
    n_units_remove = len(remove_ids)
    pcnt_units_remove = (n_units_remove / len(unit_ids)) * 100
    log_or_print(f"Removing {n_units_remove} noise units, based on {noise_cutoff} certainty cutoff ({pcnt_units_remove:.2f}% of all units).", logger=logger)
    analyzer_cleaned = analyzer.remove_units(remove_unit_ids=remove_ids)

    # settings
    cleaned_settings_folder = rec_id.folder('cleaned_sorting/settings', mkdir=save_settings, location=data_location)
    if save_settings:
        save_json(dict(noise_cutoff=noise_cutoff, n_units_remove=n_units_remove, pcnt_units_remove=pcnt_units_remove),
                  cleaned_settings_folder / 'cleaning_settings.json', keep_existing=True, logger=logger)

    # save the cleaned analyzer (it stays in memory otherwise) - see https://spikeinterface.readthedocs.io/en/stable/api.html (under remove_units)
    analyzer_cleaned.sorting.annotate(**rec_id.kwargs, sorting_type='cleaned')  # for this session only (annotations aren't saved; added again when loading)
    cleaned_analyzer_folder = rec_id.folder('cleaned_sorting/analyzer', location=data_location, mkdir=False)
    log_or_print(f"Saving noise-free analyzer to {cleaned_analyzer_folder}.", logger=logger)
    analyzer_cleaned.save_as(format='binary_folder', folder=cleaned_analyzer_folder)
    return analyzer_cleaned

# %% pipeline

def create_stage_analyzer(rec_id, sorting_type, logger=None, data_location='local', params=None):
    """
    Create (and save) the analyzer of a stage from the previous one:
        original - from the Kilosort output (or the sorting_folder of the recording table) and the recording it was sorted from
                   (sorting_pipeline.load_sorted_recording: motion corrected, else preprocessed, else raw + preprocessing on the fly)
        merged   - original + iterative auto-merges (do_iterative_merges)
        cleaned  - merged without the noise units (do_noise_units_removal)
    """
    log_or_print(f"Creating the {sorting_type} analyzer.", logger=logger)
    if sorting_type == 'original':
        from sorting_pipeline import load_sorted_recording, load_sorting
        recording = load_sorted_recording(rec_id, logger=logger)
        sorting = load_sorting(rec_id, logger=logger)
        log_or_print("Removing excess spikes from the sorting.", logger=logger)
        sorting = sc.remove_excess_spikes(sorting=sorting, recording=recording)  # removes spikes that are not in the recording range - seems to be a common bug after KS
        sorting.annotate(**rec_id.kwargs)   # apparently these weren't saved in the sorting object
        return make_sorting_analyzer(sorting=sorting, recording=recording, logger=logger, data_location=data_location)
    if sorting_type == 'merged':
        analyzer_original = load_sorting_analyzer(rec_id, sorting_type='original', logger=logger, data_location=data_location)
        return do_iterative_merges(analyzer=analyzer_original, logger=logger, data_location=data_location, params=params)
    if sorting_type == 'cleaned':
        analyzer_merged = load_sorting_analyzer(rec_id, sorting_type='merged', logger=logger, data_location=data_location)
        return do_noise_units_removal(analyzer=analyzer_merged, logger=logger, data_location=data_location, params=params)
    raise ValueError(f"Unknown sorting type: {sorting_type}. Must be 'original', 'merged' or 'cleaned'.")

def postprocess_stage(rec_id, sorting_type, logger=None, data_location='local', params=None):
    """
    Everything for one stage ('original', 'merged' or 'cleaned'); each step is skipped if its output already exists:
        analyzer (loaded, or created from the previous stage) -> extensions (original only) -> classification -> figures -> phy export
    EXPORT_TO_PHY (config): whether to export the sorting to phy format
        this should be pretty fast after extensions are computed, but there is no need to save excessive data at the moment.
        see https://spikeinterface.readthedocs.io/en/latest/modules/exporters.html
    """
    params = params or load_postprocessing_params()
    stage_folder = lambda sub: rec_id.folder(f'{sorting_type}_sorting/{sub}', location=data_location, mkdir=False)

    # analyzer
    if is_nonempty_dir(stage_folder('analyzer')):
        analyzer = load_sorting_analyzer(rec_id, sorting_type=sorting_type, logger=logger, data_location=data_location)
    else:
        analyzer = create_stage_analyzer(rec_id, sorting_type, logger=logger, data_location=data_location, params=params)
    analyzer.sorting.annotate(**rec_id.kwargs)

    # extensions - computed on the original analyzer (the merged / cleaned ones inherit them); only missing ones are computed
    if sorting_type == 'original':
        compute_analyzer_extensions(analyzer, extensions="all", overwrite=False, logger=logger, save_settings=True, data_location=data_location, params=params)

    # classification
    if is_nonempty_dir(stage_folder('classification')):
        log_or_print(f"Classification folder {stage_folder('classification')} already exists. Not reclassifying.", logger=logger)
    else:
        do_classification(analyzer=analyzer, logger=logger, data_location=data_location, params=params)

    # figures
    figures_folder = rec_id.folder(f'{sorting_type}_sorting/figures', location=data_location)
    log_or_print(f"Exporting sorting plots to {figures_folder}.", logger=logger)
    try:
        export_sorting_plots(analyzer, unit_metrics=get_unit_metrics(analyzer, data_location=data_location), figures_folder=figures_folder)
    except FileExistsError:
        log_or_print("At least one of the sorting figures already exists, moving on without overwriting. Delete the folder to recompute all.", logger=logger)

    # phy
    if config.EXPORT_TO_PHY:
        phy_folder = stage_folder('analyzer') / 'phy'
        if is_nonempty_dir(phy_folder):
            log_or_print(f"Phy output folder {phy_folder} already exists. Not re-exporting.", logger=logger)
        else:
            log_or_print(f"Exporting sorting to phy format to {phy_folder}.", logger=logger)
            # copy_binary=False: phy reads the motion-corrected data in preprocessing/ (its path is in params.py) instead of a
            # full copy of it per stage (~260 GB for a 3 h recording). If that folder moves, update dat_path in params.py.
            si_export_to_phy(sorting_analyzer=analyzer, output_folder=phy_folder, copy_binary=False)
    return analyzer

def postprocessing_pipeline(rec_id, logger=None, data_location='local'):
    """
    Run the postprocessing pipeline for a given recording. Used in the pipeline with logging function.
    Stages that already have an analyzer folder are loaded instead of recomputed (see postprocess_stage).
    The pipeline runs on the local folders (data_location='local'; the batch runner doesn't pass anything else). Loading an analyzer
    from the server works (load_sorting_analyzer(..., data_location='server'), as the behavior pipeline does), but creating stages there
    isn't supported - use SERVER_SYNC['postprocessing'] to copy the results to the server.
    """
    root_folder = rec_id.folder('root', location=data_location)
    log_or_print(f"Starting postprocessing pipeline for session {rec_id} ({root_folder}).", logger=logger)
    params = load_postprocessing_params()
    if config.OVERWRITE_PROCESSING:
        delete_outputs(rec_id, ['original_sorting', 'merged_sorting', 'cleaned_sorting'], logger=logger, location=data_location)
    for sorting_type in ['original', 'merged', 'cleaned']:
        log_or_print(f"Generating {sorting_type}_sorting.", logger=logger)
        postprocess_stage(rec_id, sorting_type, logger=logger, data_location=data_location, params=params)

    update_info_df(rec_id, columns=['sorting_analyzed', 'postprocessing_finalized'], logger=logger)
    log_or_print(f"Pipeline for postprocessing completed for session {rec_id} ({root_folder}).", logger=logger)

    # copy to server
    if server_sync_enabled('postprocessing') and data_location=='local':  # note the log was local in any case so if data_location=='server' it's not copied like this
        log_or_print(f"Copying postprocessing for {rec_id} to server.", logger=logger)
        folders_to_copy = ['postprocessing']
        copy_to_server(rec_id, folders_to_copy=folders_to_copy, logger=logger)
        update_info_df(rec_id, columns=['postprocessing_on_server'], logger=logger)
        log_or_print(f"Finished copying postprocessing for {rec_id} to server.", logger=logger)

# %% main

if __name__ == "__main__":
    batch_kwargs = main_pipeline_parser()
    run_batch_pipeline(postprocessing_pipeline, 'postprocessing', batch_kwargs=batch_kwargs)