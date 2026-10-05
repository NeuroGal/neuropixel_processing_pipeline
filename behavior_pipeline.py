"""
Gal Vishne 2025-2026
contact me at gal.vishne@gmail.com

Step 3: task events + sorted units -> trial-aligned data (see trial_data.py for the output files and how to load them).
    1. load the event codes (and eye data, if any) of the second system - Blackrock / Plexon (raw_io.py)
       (the continuous eye data is saved in preprocessing/eye)
    2. map their clock to the neuropixel (probe) clock (see "clock alignment" below)
    3. parse the codes into a trial table with the task's parser (tasks/), and mark the trials to keep;
       cut the eye data (and optionally the LFP) around each trial
    4. for each sorting in BEHAVIOR_SORTING_TYPES - the postprocessing stages (original / merged / cleaned), or 'kilosort' (the Kilosort
       output directly, no postprocessing needed): cut each unit's spikes around each trial, get the unit metrics and mark the units to
       keep (GOOD_UNITS_QUERY in the config). With BEHAVIOR_SORTING_TYPES = [], only the trials, eye data and trial LFP are made.
Run from the command line, e.g.:
    python behavior_pipeline.py --monkeys Jones --dates 20260702
"""
import numpy as np
import pandas as pd
import config
from scipy.interpolate import interp1d
from tasks import get_task
from raw_io import load_events, load_sync_times, load_eye_data, load_nidq_codes, load_nidq_sync_times, load_imec_sync_times, get_imec_stream_id
from trial_data import (get_task_data_files, save_session_data, save_sorting_data, save_trial_eye, save_trial_lfp,
                        save_continuous_eye, load_continuous_eye, get_continuous_eye_file)
from pipelines_functions import (copy_to_server, copy_from_server, delete_outputs,
                                 server_sync_enabled, log_or_print, run_batch_pipeline, main_pipeline_parser, update_info_df)

# %% clock alignment
# Second-system times are mapped to the neuropixel clock with a function built from points recorded on both systems,
# with linear interpolation between them and linear extrapolation before the first / after the last one:
#     'codes'       - task codes recorded on both systems (the codes that occur the same number of times in both)
#     'sync_signal' - shared sync pulses
# This gives NIDQ times. With ALIGN_NIDQ_TO_IMEC (default), they are then mapped to the probe clock (= spike times) using the
# SpikeGLX sync wave recorded in both - like TPrime. The same mapping is used for the events and the eye data.

# %% clock alignment - second system -> NIDQ, method 1: shared codes

def get_code_counts(system1_codes, system2_codes):
    """
    Get the counts of each code in both systems.
    Returns a DataFrame with the counts and the absolute difference between them, sorted by the absolute difference and the average count.
    """
    code_counts = pd.DataFrame({'n_system1': pd.Series(system1_codes).value_counts(), 'n_system2': pd.Series(system2_codes).value_counts()})
    code_counts = code_counts.fillna(0).reset_index(names=['code'])
    code_counts['n_abs_diff'] = np.abs(code_counts['n_system1'] - code_counts['n_system2'])
    code_counts['n_avg'] = (code_counts['n_system1'] + code_counts['n_system2']) / 2
    return code_counts.sort_values(by=['n_abs_diff', 'n_avg'], ascending=[True, False]).reset_index(drop=True)

def get_matching_codes(system1_codes, system2_codes, remove_zero=True, n_codes_to_use=None):
    """
    Get the codes that have the same number of occurrences in both systems.
    Code 0 is left out by default: on the NIDQ digital word, 0 can be the resting value between codes, so its count doesn't track task events.
    We count on the fact that codes are matching iff they are exactly the same code (and not one was missed by neuropixels & another by the other system etc).
    """
    code_counts = get_code_counts(system1_codes, system2_codes)
    matching_codes = code_counts[code_counts['n_abs_diff'] == 0]['code'].values
    if remove_zero:
        matching_codes = matching_codes[matching_codes != 0]
    if len(matching_codes) == 0:
        raise ValueError("No matching codes found between the two systems.")
    return matching_codes[:n_codes_to_use]  # n_codes_to_use=None -> all

def mapping_from_codes(system2_times, system2_codes, neuropixel_times, neuropixel_codes, n_codes_to_use=None):
    """Mapping using the times of the codes that occur the same number of times in both systems as interpolation points."""
    matching_codes = get_matching_codes(system2_codes, neuropixel_codes, n_codes_to_use=n_codes_to_use)
    system2_match_times = system2_times[np.isin(system2_codes, matching_codes)]
    neuropixel_match_times = neuropixel_times[np.isin(neuropixel_codes, matching_codes)]
    return interp1d(system2_match_times, neuropixel_match_times, kind='linear', fill_value='extrapolate')

# %% clock alignment - second system -> NIDQ, method 2: sync signal

def fill_missing_pulses(sync_times, name, logger=None):
    """Insert pulses where the gap between pulses is > 1.5 times the median gap (as many as fit in the gap)."""
    gaps = np.diff(sync_times)
    median_gap = np.median(gaps)
    for gap_i in np.where(gaps > 1.5 * median_gap)[0][::-1]:  # from the end, so earlier indices stay valid
        n_missing = int(round(gaps[gap_i] / median_gap)) - 1
        new_times = sync_times[gap_i] + median_gap * np.arange(1, n_missing + 1)
        sync_times = np.insert(sync_times, gap_i + 1, new_times)
        log_or_print(f"Inserted {n_missing} missing sync time(s) after {sync_times[gap_i]:.3f} s in the {name} sync times.", logger=logger)
    return sync_times

def mapping_from_sync(sync_times_system2, sync_times_neuropixel, logger=None):
    """Mapping using the sync pulse times of both systems as interpolation points."""
    log_or_print(f"Number of sync times in system2: {len(sync_times_system2)}, number of sync times in Neuropixel: {len(sync_times_neuropixel)}", logger=logger)
    if len(sync_times_system2) != len(sync_times_neuropixel):
        log_or_print("Warning: the number of sync times does not match - trying to fill in skipped pulses.", logger=logger)
        sync_times_neuropixel = fill_missing_pulses(sync_times_neuropixel, 'Neuropixel', logger=logger)
        sync_times_system2 = fill_missing_pulses(sync_times_system2, 'system2', logger=logger)
        if len(sync_times_system2) != len(sync_times_neuropixel):
            raise ValueError(f"Sync times still don't match after filling in missing pulses ({len(sync_times_system2)} vs {len(sync_times_neuropixel)}). "
                             "Check whether one system started / stopped recording earlier than the other.")
    return interp1d(sync_times_system2, sync_times_neuropixel, kind='linear', fill_value='extrapolate')

# %% clock alignment - NIDQ -> probe

def nidq_to_imec_mapping(rec_id, raw_path, nidq_sync_channel, imec_sync_bit, logger=None):
    """
    Function mapping NIDQ times to probe times, by matching the sync wave edges recorded in both
    (this is what TPrime does). Assumes the same sync wave goes into the NIDQ sync channel and the probe SY channel.
    Logs the offset between the clocks at the start and end of the recording, so you can see whether it matters for you.
    """
    nidq_edges = load_nidq_sync_times(raw_path, nidq_sync_channel)
    imec_edges = load_imec_sync_times(raw_path, get_imec_stream_id(rec_id, raw_path), imec_sync_bit)
    # pair each NIDQ edge with the nearest probe edge (the streams start nearly together, so offsets are << the sync period)
    period = np.median(np.diff(nidq_edges))
    nearest = np.clip(np.searchsorted(imec_edges, nidq_edges), 1, len(imec_edges) - 1)
    nearest = np.where(np.abs(imec_edges[nearest - 1] - nidq_edges) < np.abs(imec_edges[nearest] - nidq_edges), nearest - 1, nearest)
    matched = np.abs(imec_edges[nearest] - nidq_edges) < period / 4
    if matched.sum() < 2:
        raise ValueError("Could not match the NIDQ and probe sync edges - check NIDQ_SYNC_CHANNEL / IMEC_SYNC_BIT and that both record the same sync wave.")
    nidq_matched, imec_matched = nidq_edges[matched], imec_edges[nearest[matched]]
    offsets_ms = (imec_matched - nidq_matched) * 1000
    # sanity checks that both channels really record the same sync wave: same period, most edges matched, and an offset that
    # changes smoothly (linear clock drift) - otherwise unrelated edges may have been paired
    imec_period = np.median(np.diff(imec_edges))
    residuals_ms = offsets_ms - np.polyval(np.polyfit(nidq_matched, offsets_ms, 1), nidq_matched)
    if abs(imec_period - period) > 0.01 * period or matched.mean() < 0.9 or np.abs(residuals_ms).max() > 1:
        raise ValueError(f"The NIDQ and probe sync edges don't look like the same sync wave (periods {period:.4f} / {imec_period:.4f} s, "
                         f"{matched.mean():.0%} of edges matched, max deviation from linear drift {np.abs(residuals_ms).max():.2f} ms). "
                         "Check NIDQ_SYNC_CHANNEL / IMEC_SYNC_BIT.")
    log_or_print(f"NIDQ->probe clock: matched {matched.sum()}/{len(nidq_edges)} sync edges; probe - NIDQ offset is {offsets_ms[0]:.2f} ms at the start "
                 f"and {offsets_ms[-1]:.2f} ms at the end of the recording.", logger=logger)
    return interp1d(nidq_matched, imec_matched, kind='linear', fill_value='extrapolate')

# %% clock alignment - the mapping used by the pipeline

def get_clock_mapping(rec_id, second_system, alignment_method, system2_times=None, system2_codes=None, data_location='server', logger=None):
    """
    Function mapping second-system times to the neuropixel clock, and the name of that clock ('imec' = probe, or 'nidq').
    alignment_method: 'sync_signal' or 'codes' (needs system2_times & system2_codes) - ALIGNMENT_METHOD in the config.
    If ALIGN_NIDQ_TO_IMEC is set but the NIDQ -> probe mapping fails, this falls back to the NIDQ clock with a warning in the log.
    """
    raw_path = rec_id.folder('raw_data', location=data_location, mkdir=False)
    if alignment_method == 'codes':
        neuropixel_times, neuropixel_codes = load_nidq_codes(raw_path, channel=config.NIDQ_CODES_CHANNEL)
        to_nidq = mapping_from_codes(system2_times, system2_codes, neuropixel_times, neuropixel_codes)
    elif alignment_method == 'sync_signal':
        if config.SYNC_CHANNEL is None:
            raise ValueError("Set SYNC_CHANNEL in the config for sync_signal alignment (or use ALIGNMENT_METHOD = 'codes').")
        sync_times_system2 = load_sync_times(second_system, rec_id, data_location=data_location)
        sync_times_neuropixel = load_nidq_sync_times(raw_path, channel=config.NIDQ_SYNC_CHANNEL)
        to_nidq = mapping_from_sync(sync_times_system2, sync_times_neuropixel, logger=logger)
    else:
        raise ValueError(f"Unknown alignment method: {alignment_method}. Use 'codes' or 'sync_signal'.")
    log_or_print(f"Aligned {second_system} to the NIDQ clock using {alignment_method}.", logger=logger)

    if not config.ALIGN_NIDQ_TO_IMEC:
        return to_nidq, 'nidq'
    try:
        nidq_to_imec = nidq_to_imec_mapping(rec_id, raw_path, config.NIDQ_SYNC_CHANNEL, config.IMEC_SYNC_BIT, logger=logger)
    except Exception as e:
        log_or_print(f"WARNING: NIDQ -> probe clock alignment failed ({e}). Behavior stays in the NIDQ clock "
                     "(set ALIGN_NIDQ_TO_IMEC = False to skip this step).", logger=logger)
        return to_nidq, 'nidq'
    return (lambda times: nidq_to_imec(to_nidq(times))), 'imec'

# %% units

def mark_good_units(unit_metrics, query=None, logger=None):
    """
    Add a keep_unit column: True for units matching the query (GOOD_UNITS_QUERY in the config by default),
    e.g. "presence_ratio > 0.75 and firing_rate / presence_ratio > 3" (present most of the session, and > 3 Hz while present).
    GOOD_UNITS_QUERY = None -> keep all units. Units with missing metrics (nan) are not kept.
    """
    query = query if query is not None else config.GOOD_UNITS_QUERY
    if query is None:
        unit_metrics['keep_unit'] = True
    else:
        unit_metrics['keep_unit'] = unit_metrics.index.isin(unit_metrics.query(query).index)
    log_or_print(f"Marked {unit_metrics['keep_unit'].sum()}/{len(unit_metrics)} units as good (criteria: {query}).", logger=logger)
    return unit_metrics

# %% spikes

def parse_spikes(sorting, trials, t_before, t_after):
    """
    Cut the spikes of each unit around each trial.
    Returns:
        spike_times: np.array (dtype=object) of shape (n_trials, n_units) where each element is an array of spike times in seconds for that unit in that trial,
                     from t_start - t_before to t_end + t_after, relative to t_start (like the event times in the trial table).
        ! Only the first segment of the recording is used (multi-segment recordings are not supported).
    """
    unit_ids = sorting.get_unit_ids()
    fs = sorting.get_sampling_frequency()
    spikes_by_unit = [sorting.get_unit_spike_train(unit_id=unit) / fs for unit in unit_ids]

    # (filled element by element: np.array(list_of_lists, dtype=object) would silently make a 3D array if all arrays happen to have the same length)
    spike_times = np.empty((len(trials), len(unit_ids)), dtype=object)
    for tr_i, (t_start, t_end) in enumerate(trials[['t_start', 't_end']].itertuples(index=False)):
        for u_i, spikes in enumerate(spikes_by_unit):
            spike_times[tr_i, u_i] = spikes[(spikes >= t_start - t_before) & (spikes <= t_end + t_after)] - t_start
    return spike_times

def kilosort_unit_metrics(sorting, duration, presence_bin_s=60.0):
    """
    Unit table of a Kilosort sorting without postprocessing: Kilosort's own unit properties (KSLabel, ContamPct, Amplitude, and the
    group/quality from phy curation if any), plus num_spikes, firing_rate and presence_ratio (the fraction of presence_bin_s bins with
    a spike, as SpikeInterface computes it), so the default GOOD_UNITS_QUERY works.
    """
    fs = sorting.get_sampling_frequency()
    units = pd.DataFrame({key: sorting.get_property(key) for key in sorting.get_property_keys()}, index=pd.Index(sorting.get_unit_ids(), name='unit_id'))
    bins = np.arange(0, duration + presence_bin_s, presence_bin_s)
    trains = [sorting.get_unit_spike_train(unit_id=u) / fs for u in units.index]
    units['num_spikes'] = [len(t) for t in trains]
    units['firing_rate'] = units['num_spikes'] / duration
    units['presence_ratio'] = [np.mean(np.histogram(t, bins)[0] > 0) for t in trains]
    return units

def load_sorting_and_units(rec_id, sorting_type, data_location='local', logger=None):
    """(sorting, unit metrics) of a postprocessing stage ('original' / 'merged' / 'cleaned') or of the Kilosort output ('kilosort')."""
    if sorting_type == 'kilosort':
        import spikeinterface.extractors as se
        from sorting_pipeline import load_sorting
        sorting = load_sorting(rec_id, logger=logger, location=data_location)
        raw_path = rec_id.folder('raw_data', location=data_location, mkdir=False)
        duration = se.read_spikeglx(raw_path, stream_id=get_imec_stream_id(rec_id, raw_path)).get_total_duration()
        return sorting, kilosort_unit_metrics(sorting, duration)
    from postprocessing_pipeline import load_sorting_analyzer, get_unit_metrics
    analyzer = load_sorting_analyzer(rec_id=rec_id, sorting_type=sorting_type, data_location=data_location, logger=logger)
    return analyzer.sorting, get_unit_metrics(analyzer, data_location=data_location)

# %% pipeline

def get_eye(rec_id, second_system, to_neuropixel_clock, logger=None):
    """
    Continuous eye signals in the neuropixel clock (preprocessing/eye): loaded if they exist (unless OVERWRITE_PROCESSING), else
    read from the second system, mapped to the neuropixel clock and saved. None if EYE_DATA is off or no EYE_CHANNELS are set.
    """
    if not config.EYE_DATA or not config.EYE_CHANNELS:
        return None
    data_location = config.BEHAVIOR_DATA_LOCATION
    eye_file = get_continuous_eye_file(rec_id, data_location)
    if eye_file.exists() and not config.OVERWRITE_PROCESSING:
        log_or_print(f"Loading the existing continuous eye data {eye_file} (set OVERWRITE_PROCESSING to recompute).", logger=logger)
        return load_continuous_eye(rec_id, data_location)
    eye = load_eye_data(second_system, rec_id, data_location=data_location, logger=logger)
    eye['times'] = to_neuropixel_clock(eye['times'])
    save_continuous_eye(rec_id, eye, data_location)
    log_or_print(f"Saved the continuous eye data to {eye_file}.", logger=logger)
    return eye

def beh_pipeline(rec_id, logger=None):
    """
    Trial-aligned data of one recording (settings: the behavior pipeline section of the config).
    BEHAVIOR_DATA_LOCATION: 'server' or 'local' - where to load the data from & where to save
        with 'local' and SERVER_SYNC['behavior'], the raw event files are copied from the server first and the outputs copied back after.
    """
    second_system, task_name, alignment_method = config.SECOND_SYSTEM.lower(), config.TASK, config.ALIGNMENT_METHOD
    data_location = config.BEHAVIOR_DATA_LOCATION
    task = get_task(task_name)
    t_before, t_after = config.T_BEFORE_TRIAL, config.T_AFTER_TRIAL
    sync_with_server = server_sync_enabled('behavior') and data_location == 'local'

    if get_task_data_files(rec_id, data_location=data_location)['trials'].exists():
        if not config.OVERWRITE_TASK_DATA:
            log_or_print(f"{rec_id}: task data already exists - skipping (OVERWRITE_TASK_DATA = False).", logger=logger)
            return
        delete_outputs(rec_id, ['task/data'], logger=logger, location=data_location)  # so no files of a previous run are left
    log_or_print(f"{rec_id}: task {task_name}, second system {second_system}, alignment by {alignment_method}.", logger=logger)

    if sync_with_server:
        log_or_print(f"{rec_id}: copying from server to local", logger=logger)
        copy_from_server(rec_id=rec_id, files_to_copy='triggers', logger=logger)
        copy_from_server(rec_id=rec_id, files_to_copy='neuropixel_triggers', logger=logger)

    # 1-2. events & eye data, in the neuropixel clock
    event_times, event_codes = load_events(second_system, rec_id, data_location=data_location)
    log_or_print(f"Loaded {len(event_times)} {second_system} events.", logger=logger)
    to_neuropixel_clock, clock = get_clock_mapping(rec_id, second_system, alignment_method, event_times, event_codes,
                                                   data_location=data_location, logger=logger)
    event_times = to_neuropixel_clock(event_times)
    eye = get_eye(rec_id, second_system, to_neuropixel_clock, logger=logger)

    # 3. trial table (+ eye / LFP per trial)
    trials = task.parse_codes(event_times, event_codes)
    trials['keep_trial'] = task.keep_trial(trials).values
    log_or_print(f"Parsed {len(trials)} trials ({trials['keep_trial'].sum()} to keep).", logger=logger)
    info = dict(**rec_id.kwargs, task=task_name, second_system=second_system, alignment_method=alignment_method,
                clock=clock, t_before=t_before, t_after=t_after, good_units_query=config.GOOD_UNITS_QUERY,
                eye=eye is not None, trial_lfp=config.TRIAL_LFP, trial_lfp_reference=config.TRIAL_LFP_REFERENCE,
                config_file=config.CONFIG_FILE, created=str(pd.Timestamp.now()))
    save_session_data(rec_id, trials=trials, info=info, data_location=data_location)
    if eye is not None:
        save_trial_eye(rec_id, eye, trials, t_before, t_after, data_location=data_location)
        log_or_print("Saved the eye data per trial.", logger=logger)
    if config.TRIAL_LFP:
        save_trial_lfp(rec_id, trials, t_before, t_after, reference=config.TRIAL_LFP_REFERENCE, data_location=data_location, logger=logger)

    # 4. spikes & units per sorting stage
    for sorting_type in config.BEHAVIOR_SORTING_TYPES:
        sorting, unit_metrics = load_sorting_and_units(rec_id, sorting_type, data_location=data_location, logger=logger)
        spikes = parse_spikes(sorting=sorting, trials=trials, t_before=t_before, t_after=t_after)
        units = mark_good_units(unit_metrics, logger=logger).reset_index()  # unit_id as the first column
        save_sorting_data(rec_id, sorting_type=sorting_type, spikes=spikes, units=units, data_location=data_location)
        log_or_print(f"Saved {sorting_type} sorting: {spikes.shape[0]} trials x {spikes.shape[1]} units.", logger=logger)

    update_info_df(rec_id=rec_id, columns=['behavior_parsed'], logger=logger)
    if sync_with_server:
        copy_to_server(rec_id, folders_to_copy=['task/data'], mirror=True, logger=logger)   # task/data is rebuilt on every run
        if eye is not None:
            copy_to_server(rec_id, folders_to_copy=['eye'], copy_log=False, logger=logger)
        log_or_print('Finished copying to server', logger=logger)
        update_info_df(rec_id=rec_id, columns=['behavior_parsed_on_server'], logger=logger)

# %% main

if __name__ == '__main__':
    batch_kwargs = main_pipeline_parser()
    run_batch_pipeline(beh_pipeline, 'behavior', batch_kwargs=batch_kwargs)
