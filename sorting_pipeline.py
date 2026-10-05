"""
Gal Vishne 2025-2026
contact me at gal.vishne@gmail.com

Step 1: raw SpikeGLX data -> preprocessing -> motion correction (MEDiCINe, or Kilosort's own, see SortingParams.motion_correction) -> Kilosort4.

Each step is saved to disk and skipped on later runs if its output exists (see prepro_sorting_pipeline).
With a sorting_folder in the recording table (an existing Kilosort output), the sorting step is skipped and that sorting is used.
Run from the command line, e.g.:
    python sorting_pipeline.py --monkeys Jones --dates 20260702 --regions Pul LIP
"""
# %% imports
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
import config
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.extractors as se
import spikeinterface as si
import medicine
from spikeinterface.core.motion import Motion
from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from raw_io import get_imec_stream_id
from pipelines_functions import (RecordingID,
                                 is_nonempty_dir, save_json, require_output, require_no_output, delete_outputs, log_or_print, update_info_df, copy_to_server, copy_from_server, server_sync_enabled,
                                 run_batch_pipeline, main_pipeline_parser)

# %% parameters

@dataclass
class SortingParams:
    """
    All parameters of the sorting pipeline, in one place. The defaults are what we use;
    to change any of them for your setup, set e.g. SORTING_PARAMS = dict(detect_threshold=5.0) in local_config.py.
    (note: MEDiCINe results are cached in the medicine folder - delete it if you change the peak / MEDiCINe parameters)
    """
    # preprocessing
    freq_min: float = 300
    freq_max: float = 6000
    # motion (drift) correction:
    #   'medicine' - MEDiCINe's estimate, interpolated into a saved motion-corrected recording that Kilosort sorts (Kilosort's own correction off)
    #   'kilosort' - Kilosort4's own drift correction (nblocks below), on the preprocessed recording. The LFP can't reuse it (LFP motion_correction=None)
    #   None       - no motion correction
    motion_correction: str = 'medicine'
    # Kilosort's drift correction blocks, with motion_correction='kilosort': Kilosort recommends 5 for single-shank Neuropixels (nonrigid),
    # 1 is rigid, and 0 for probes with few channels or sparse contacts (https://kilosort.readthedocs.io/en/latest/parameters.html)
    kilosort_nblocks: int = 5
    # peak detection for MEDiCINe (largely agreeing with the defaults: peak_sign='neg', exclude_sweep_ms=0.1, radius_um=50)
    detect_threshold: float = 6.0 # taken from setting used by Gabe in the Jazayeri lab
    peak_sign: str = 'neg'
    exclude_sweep_ms: float = 0.1
    detect_radius_um: float = 50
    # peak localization (the github example had 'monopolar_triangulation' but I tested vs center_of_mass and center_of_mass seemed a tad better)
    # see https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sortingcomponents/peak_localization.py
    localize_method: str = 'center_of_mass'
    localize_radius_um: float = 70.0
    localize_ms_before: float = 0.3
    localize_ms_after: float = 0.6
    # MEDiCINe (recommended in the github page for some troubleshooting cases so I kept using it)
    medicine_amplitude_threshold_quantile: float = 0.25
    # motion interpolation
    interpolation_border_mode: str = 'force_extrapolate'
    # Kilosort4: changes to its defaults (https://kilosort.readthedocs.io/en/latest/parameters.html); chunk_duration comes from --chunk_duration in our config
    # (Kilosort's do_correction / nblocks are set from motion_correction above)
    kilosort_params: dict = field(default_factory=lambda: dict(
        clear_cache=True,
        do_CAR=False,           # we already did this
        highpass_cutoff=10,     # setting to really low since we already do this too
    ))

def load_sorting_params():
    """SortingParams with the overrides from SORTING_PARAMS in the config."""
    params = SortingParams(**config.SORTING_PARAMS)
    if params.motion_correction not in ['medicine', 'kilosort', None]:
        raise ValueError(f"SORTING_PARAMS motion_correction = {params.motion_correction!r} - must be 'medicine', 'kilosort' or None.")
    return params

def sorted_recording_type(params):
    """The saved recording Kilosort sorts: the motion-corrected one with MEDiCINe, otherwise the preprocessed one."""
    return 'motion_corrected' if params.motion_correction == 'medicine' else 'preprocessed'

# %% load data & preprocess

# which step makes each saved recording (for the error messages)
MADE_BY = dict(preprocessed='the preprocessing', motion_corrected='the motion correction')

def load_recording_data(rec_id, type, logger=None):
    """
    Load the raw, preprocessed or motion corrected recording (type: 'raw', 'preprocessed', 'motion_corrected').
    """
    if type not in ['raw', *MADE_BY]:
        raise ValueError(f"Invalid type: {type}. Must be 'raw', 'preprocessed', or 'motion_corrected'.")
    path = rec_id.folder(f'{type}_data', mkdir=False)
    log_or_print(f"Loading {type.replace('_', ' ')} data.", logger=logger)
    if type == 'raw':
        recording = se.read_spikeglx(path, stream_id=get_imec_stream_id(rec_id, path))
    else:
        require_output(path, f"{type.replace('_', ' ')} data", f'run {MADE_BY[type]}')
        recording = si.BaseRecording.load(path)
    recording.annotate(**rec_id.kwargs)
    return recording

def load_sorted_recording(rec_id, logger=None):
    """
    The recording the sorting was made from, for the postprocessing analyzer (waveforms, amplitudes, locations come from it):
    the saved motion-corrected recording if it exists, else the saved preprocessed one, else the raw recording with this pipeline's
    preprocessing applied on the fly (e.g. for a sorting made elsewhere, see sorting_folder in the recording table).
    """
    if is_nonempty_dir(rec_id.folder('motion_corrected_data', mkdir=False)):
        return load_recording_data(rec_id, type='motion_corrected', logger=logger)
    return load_preprocessed_recording(rec_id, logger=logger)

def load_preprocessed_recording(rec_id, logger=None):
    """The saved preprocessed recording, or the raw recording with this pipeline's preprocessing applied on the fly (not saved)."""
    if is_nonempty_dir(rec_id.folder('preprocessed_data', mkdir=False)):
        return load_recording_data(rec_id, type='preprocessed', logger=logger)
    log_or_print("No saved preprocessed recording - using the raw recording with this pipeline's preprocessing (not saved).", logger=logger)
    recording = preprocess(load_recording_data(rec_id, type='raw', logger=logger), load_sorting_params())
    recording.annotate(**rec_id.kwargs, preprocessed=True)
    return recording

def check_step_order(rec, done=(), not_done=()):
    """
    Check which steps a recording went through, using the annotations the steps add ('preprocessed', 'motion_corrected'),
    e.g. check_step_order(rec, done=['preprocessed'], not_done=['motion_corrected']) before motion correction.
    """
    for step in done:
        if not rec.get_annotation(key=step):
            raise ValueError(f"The recording is not {step.replace('_', ' ')} - run {MADE_BY[step]} first.")
    for step in not_done:
        if rec.get_annotation(key=step):
            raise ValueError(f"The recording is already {step.replace('_', ' ')} - load the saved one instead of recomputing.")

def save_step_output(recording, rec_id, step, logger=None, **annotations):
    """Annotate a recording with the step it went through (+ any extra annotations) and save it to <step>_data."""
    recording.annotate(**rec_id.kwargs, **{step: True, f'{step}_date': str(pd.Timestamp.now())}, **annotations)
    log_or_print(f"Saving {step.replace('_', ' ')} data.", logger=logger)
    return recording.save(folder=rec_id.folder(f'{step}_data'), overwrite=True)

def preprocess(rec, params):
    """bandpass filter, phase shift, common reference (lazy: computed when the traces are read)"""
    rec = spre.bandpass_filter(rec, freq_min=params.freq_min, freq_max=params.freq_max)
    rec = spre.phase_shift(rec)
    return spre.common_reference(rec)

def do_prepro(rec, params=None, logger=None):
    """
    The preprocessing step: preprocess + save to preprocessed_data.
    """
    params = params or load_sorting_params()
    check_step_order(rec, not_done=['preprocessed'])
    rec_id = RecordingID.from_annotations(rec)
    require_no_output(rec_id.folder('preprocessed_data', mkdir=False), 'preprocessed data')
    log_or_print("Preprocessing data.", logger=logger)
    recording_preprocessed = preprocess(rec, params)
    return save_step_output(recording_preprocessed, rec_id, 'preprocessed', logger=logger,
                            prepro_settings={'freq_min': params.freq_min, 'freq_max': params.freq_max, 'phase_shift': True, 'common_reference': True})

# %% motion correction with MEDiCINe
# https://github.com/jazlab/medicine/tree/main

def load_or_compute(file, compute, description, logger=None):
    """Load a cached .npy result if it exists, otherwise compute it (compute()) and save it."""
    if file.exists():
        log_or_print(f"Loaded existing {description}.", logger=logger)
        return np.load(file)
    log_or_print(f"Computing {description}.", logger=logger)
    result = compute()
    np.save(file, result)
    log_or_print(f"{description.capitalize()} computed and saved.", logger=logger)
    return result

MEDICINE_FILES = ['motion.npy', 'time_bins.npy', 'depth_bins.npy']

def load_medicine_motion(rec_id, rec):
    """
    The motion estimated by MEDiCINe (saved in the medicine folder) as a SpikeInterface Motion object, in the time of rec (the recording it
    will correct). MEDiCINe's time bins are seconds from the start of the recording (peak sample index / fs), but SpikeInterface gives
    SpikeGLX recordings their start time in the acquisition run (firstSample / fs, here ~1.9 million s), and InterpolateMotionRecording looks
    the motion up at those times - so the bins are shifted by rec's start time. Without this, all times are past the last bin and the
    whole recording is shifted by the last bin's displacement (no drift correction). MEDiCINe's README recipe and SpikeInterface's own
    method='medicine' wrapper have the same problem: https://github.com/SpikeInterface/spikeinterface/issues/4318 (open as of October 2026).
    """
    medicine_folder = rec_id.folder('medicine', mkdir=False)
    if not all((medicine_folder / f).exists() for f in MEDICINE_FILES):
        raise ValueError(f"No MEDiCINe motion in {medicine_folder} - run the motion correction of sorting_pipeline.py first.")
    motion, time_bins, depth_bins = [np.load(medicine_folder / f) for f in MEDICINE_FILES]
    t_start = rec.sample_index_to_time(0)
    return Motion(displacement=motion, temporal_bins_s=time_bins + t_start, spatial_bins_um=depth_bins)

def get_motion_object(rec, params=None, logger=None):
    """
    Core computations for medicine motion correction (called by do_motion_correction)
    Intermediate results (peaks, peak locations, MEDiCINe outputs) are cached in the 'medicine' folder.
    """
    params = params or load_sorting_params()
    rec_id = RecordingID.from_annotations(rec)
    medicine_folder = rec_id.folder('medicine')

    # Detect, extract, and localize peaks
    peaks = load_or_compute(medicine_folder / 'peaks.npy', description='peaks', logger=logger, compute=lambda: detect_peaks(
        rec, method="locally_exclusive", detect_threshold=params.detect_threshold, peak_sign=params.peak_sign,
        exclude_sweep_ms=params.exclude_sweep_ms, radius_um=params.detect_radius_um))
    peak_locations = load_or_compute(medicine_folder / 'peak_locations.npy', description='peak locations', logger=logger, compute=lambda: localize_peaks(
        rec, peaks, method=params.localize_method, radius_um=params.localize_radius_um,
        ms_before=params.localize_ms_before, ms_after=params.localize_ms_after))
    # run medicine unless it already ran
    if not all((medicine_folder / f).exists() for f in MEDICINE_FILES):
        log_or_print("Running MEDiCINe for motion estimation.", logger=logger)
        medicine.run_medicine(peak_amplitudes=peaks['amplitude'], peak_depths=peak_locations['y'],
                              peak_times=peaks['sample_index'] / rec.get_sampling_frequency(),
                              output_dir=medicine_folder, amplitude_threshold_quantile=params.medicine_amplitude_threshold_quantile)
    log_or_print("Loading motion data.", logger=logger)
    return load_medicine_motion(rec_id, rec)

def do_motion_correction(rec, params=None, logger=None):
    params = params or load_sorting_params()
    check_step_order(rec, done=['preprocessed'], not_done=['motion_corrected'])
    rec_id = RecordingID.from_annotations(rec)
    require_no_output(rec_id.folder('motion_corrected_data', mkdir=False), 'motion corrected data')

    # Get motion_object from MEDiCINe
    motion_object = get_motion_object(rec, params=params, logger=logger)

    log_or_print("Interpolating motion to correct recording", logger=logger)
    rec = rec.astype('float32')  # this wasn't in the github instructions, but following Gabe's code I added it
    recording_motion_corrected = InterpolateMotionRecording(rec, motion_object, border_mode=params.interpolation_border_mode)
    recording_motion_corrected = recording_motion_corrected.astype('int16')  # this wasn't in the github instructions, but following Gabe's code I added it
    recording_motion_corrected = save_step_output(recording_motion_corrected, rec_id, 'motion_corrected', logger=logger)
    log_or_print("Motion correction completed.", logger=logger)
    return recording_motion_corrected

# %% sorting

def get_kilosort_params(params=None):
    """
    Kilosort4 defaults + our changes (SortingParams.kilosort_params)
    https://kilosort.readthedocs.io/en/latest/parameters.html
    """
    params = params or load_sorting_params()
    sorter_params = ss.get_default_sorter_params("kilosort4")
    sorter_params.update(params.kilosort_params)
    # Kilosort's own drift correction only with motion_correction='kilosort' (with 'medicine' the recording is already corrected)
    sorter_params['do_correction'] = params.motion_correction == 'kilosort'
    if params.motion_correction == 'kilosort':
        sorter_params['nblocks'] = params.kilosort_nblocks
    sorter_params['chunk_duration'] = si.get_global_job_kwargs()['chunk_duration']  # the default is 1.0, but it crashed in some cases due to RAM issues, so it is set from JOB_KWARGS in the config
    return sorter_params

def run_sorting(rec, params=None, logger=None):
    """
    Run Kilosort4 sorting on the recording.
    Kilosort writes its own log (kilosort4.log) to sorting/sorter_output, and we add sorting_details.json there when done
    (sorting_details.json is also what marks the sorting as complete, see prepro_sorting_pipeline).
    """
    params = params or load_sorting_params()
    check_step_order(rec, done=['preprocessed', 'motion_corrected'] if params.motion_correction == 'medicine' else ['preprocessed'])
    # run_sorter raises an error by itself if the sorting folder already exists
    sorting_folder = RecordingID.from_annotations(rec).folder('sorting', mkdir=False)
    sorter_params = get_kilosort_params(params)
    log_or_print("Running Kilosort4 sorting - this will take a while.", logger=logger)
    sorting = ss.run_sorter(sorter_name="kilosort4", recording=rec, folder=sorting_folder, verbose=True, **sorter_params)
    # add a few details in a json file in the ks folder
    sorting_detail = dict(**RecordingID.from_annotations(rec).kwargs, sorting_date=str(pd.Timestamp.now()), sorter_params=sorter_params,
                          pipeline_params=asdict(params))
    save_json(sorting_detail, sorting_folder / 'sorter_output' / 'sorting_details.json', logger=logger)
    log_or_print("Kilosort4 sorting completed.", logger=logger)
    return sorting

def external_sorting_folder(rec_id):
    """The sorting_folder of the recording table (a Kilosort / phy output made elsewhere), or None."""
    folder = rec_id.info.get('sorting_folder')
    return Path(folder) if isinstance(folder, str) and folder.strip() else None

def load_sorting(rec_id, logger=None, location='local'):
    """
    The Kilosort sorting of a recording: the sorting_folder of the recording table if it's filled (a Kilosort / phy output made
    elsewhere), otherwise this pipeline's Kilosort output (preprocessing/spike_sorting/sorter_output).
    Spike times are samples from the start of the probe's .ap file (as for this pipeline's sorting).
    """
    sorting_folder = external_sorting_folder(rec_id) or rec_id.folder('sorting', location=location, mkdir=False) / "sorter_output"
    require_output(sorting_folder, 'Kilosort output', 'run the sorting (or set sorting_folder in the recording table)')
    sorting = se.KiloSortSortingExtractor(sorting_folder)
    log_or_print(f'Loading the sorting of {rec_id} from {sorting_folder}.', logger=logger)
    return sorting

# %% pipeline

def is_step_done(rec_id, step):
    """
    Check whether a step of the sorting pipeline already finished for this recording.
    """
    step_folder = rec_id.folder(step, mkdir=False)
    if step == 'sorting':
        if step_folder.exists() and not (step_folder / 'sorter_output' / 'sorting_details.json').exists():
            raise ValueError(f"Sorting folder {step_folder} exists but the sorting did not finish (no sorting_details.json). Delete the folder to rerun.")
        return step_folder.exists()
    return is_nonempty_dir(step_folder)

def prepro_sorting_pipeline(rec_id, logger=None):
    """
    Run the entire pipeline for a given recording, skipping steps that were already computed
    (or recomputing everything with OVERWRITE_PROCESSING), + the LFP with EXTRACT_LFP (see lfp.py).
    """
    extract_lfp = config.EXTRACT_LFP
    root_folder = rec_id.folder('root')
    log_or_print(f"Starting preprocessing and sorting pipeline for recording {rec_id} ({root_folder}).", logger=logger)

    if server_sync_enabled('sorting'):
        copy_from_server(rec_id=rec_id, files_to_copy='neuropixel_data', logger=logger)

    params = load_sorting_params()
    if config.OVERWRITE_PROCESSING:
        delete_outputs(rec_id, ['preprocessed_data', 'motion_corrected_data', 'medicine', 'sorting'] + (['lfp'] if extract_lfp else []), logger=logger)

    # check what steps were computed
    steps_to_compute = ['preprocessed_data'] + (['motion_corrected_data'] if params.motion_correction == 'medicine' else []) + ['sorting']
    if external_sorting_folder(rec_id) is not None:
        log_or_print(f"Using the sorting in {external_sorting_folder(rec_id)} (sorting_folder in the recording table) - not sorting.", logger=logger)
        steps_to_compute = [step for step in steps_to_compute if step != 'sorting']
    steps_not_computed = [step for step in steps_to_compute if not is_step_done(rec_id, step)]
    if 'motion_corrected_data' in steps_to_compute and 'motion_corrected_data' not in steps_not_computed:
        # the preprocessed recording is only an input to the motion correction (it may have been deleted, KEEP_PREPROCESSED_DATA)
        steps_not_computed = [step for step in steps_not_computed if step != 'preprocessed_data']

    load_inputs = dict(rec_id=rec_id, logger=logger)
    for step in steps_not_computed:
        if step=='preprocessed_data':
            raw_recording = load_recording_data(**load_inputs, type='raw')
            do_prepro(raw_recording, params=params, logger=logger)
        elif step=='motion_corrected_data':
            preprocessed_recording = load_recording_data(**load_inputs, type='preprocessed')
            do_motion_correction(preprocessed_recording, params=params, logger=logger)
        elif step=='sorting':
            recording_to_sort = load_recording_data(**load_inputs, type=sorted_recording_type(params))
            run_sorting(recording_to_sort, params=params, logger=logger)
    sorting_done = external_sorting_folder(rec_id) is not None or is_step_done(rec_id, 'sorting')
    if (not config.KEEP_PREPROCESSED_DATA and params.motion_correction == 'medicine' and sorting_done
            and is_step_done(rec_id, 'motion_corrected_data')):
        # with MEDiCINe, Kilosort and the postprocessing use the motion-corrected recording (its peaks are cached in the medicine folder)
        delete_outputs(rec_id, ['preprocessed_data'], logger=logger)
    update_info_df(rec_id, columns=['motion_corrected', 'kilosorted'], logger=logger)
    if extract_lfp:
        from lfp import extract_lfp as do_extract_lfp
        do_extract_lfp(rec_id, logger=logger)
        update_info_df(rec_id, columns=['lfp_extracted'], logger=logger)
    log_or_print(f"Pipeline for preprocessing and sorting completed for recording {rec_id} ({root_folder}).", logger=logger)

    if server_sync_enabled('sorting'):
        log_or_print(f"Copying preprocessing and sorting for {rec_id} to server.", logger=logger)
        # 1. Not copying preprocessed data since it's relatively fast to compute
        # 2. Added raw data copying here too, but possible to remove that folder if you want it saved elsewhere
        folders_to_copy = ['raw_data', f'{sorted_recording_type(params)}_data', 'sorting'] + (['medicine'] if params.motion_correction == 'medicine' else []) + (['lfp'] if extract_lfp else [])
        copy_to_server(rec_id=rec_id, folders_to_copy=folders_to_copy, logger=logger)
        update_info_df(rec_id, columns=['raw_on_server', 'motion_corrected_on_server', 'kilosort_on_server'], logger=logger)
        log_or_print(f"Finished copying preprocessing and sorting for {rec_id} to server.", logger=logger)


# %%  main loop

if __name__ == "__main__":
    batch_kwargs = main_pipeline_parser()
    run_batch_pipeline(prepro_sorting_pipeline, 'sorting', batch_kwargs=batch_kwargs)
