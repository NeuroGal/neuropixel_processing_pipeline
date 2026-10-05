# Gal Vishne 2025-2026
# contact me at gal.vishne@gmail.com
"""
UNTESTED: the LFP derived from the AP stream (NP2.0).
The steps follow two references:
    - the IBL LFP preprocessing (ibldsp.voltage.destripe_lfp, https://github.com/int-brain-lab/ibl-neuropixel):
      3rd order butterworth bandpass 0.5-300 Hz -> sample shift correction -> interpolation of bad channels -> (median common reference)
    - the SpikeInterface LFP example (https://spikeinterface.readthedocs.io/en/latest/how_to/drift_with_lfp.html):
      bandpass (order 3, 0.5-250 Hz, long filter margins) -> phase_shift -> resample

LFP extraction: raw probe data -> bandpassed, downsampled LFP saved once per recording (preprocessing/lfp, SpikeInterface binary folder).
Per-trial windows are read from it on demand (TrialData.get_lfp), so nothing is duplicated per trial.
Runs at the end of sorting_pipeline.py (EXTRACT_LFP), or on its own without any sorting:
    python lfp.py --monkeys Jones --dates 20260702

Source:
    - NP1.0 probes record a separate LF stream (*.imecN.lf.bin, 2500 Hz, hardware filtered 0.5-500 Hz) - used if present.
    - NP2.0 probes only have the wideband AP stream (30 kHz) - then the LFP is derived from it (the same filter + resampling).
Steps (parameters in LFPParams below): bandpass freq_range -> phase shift (the ADCs sample channels at different times) -> detect bad channels
       (dead / noisy / outside the brain) and interpolate the dead & noisy ones -> resample to fs -> motion correction -> (keep every channel_step-th channel).
The channel labels are saved as the channel property 'lfp_channel_label' ('good', 'dead', 'noise', 'out').
No common reference is saved (it removes real LFP that is shared across channels, and can't be undone) - TrialData.get_lfp can
subtract the median across the in-brain channels when loading (reference='median', as IBL does).

Motion correction (LFPParams.motion_correction):
    'medicine' (default) - the MEDiCINe motion estimated from the spikes in sorting_pipeline.py, applied to the LFP channels with the same
                           interpolation as the AP data, so LFP and spikes are corrected the same way. If there is no MEDiCINe estimate yet
                           (e.g. LFP only, no sorting), it is estimated here from the AP data with the sorting pipeline's settings, and saved
                           to the medicine folder (the sorting pipeline then reuses it). With Kilosort's own drift correction, use None.
    None                 - no motion correction.

Times: the LF and AP streams of a probe share its clock, so LFP sample k is at k / fs seconds - the same clock as the spike times.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import config
from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording
from raw_io import get_imec_stream_id, find_stream_file
from pipelines_functions import (is_nonempty_dir, require_output, log_or_print, delete_outputs, update_info_df, copy_to_server, copy_from_server,
                                 server_sync_enabled, run_batch_pipeline, main_pipeline_parser)

# %% parameters

@dataclass
class LFPParams:
    """
    All parameters of the LFP extraction. The defaults are what we use; to change any of them, set e.g. LFP_PARAMS = dict(fs=2500) in the config.
    """
    freq_range: tuple = (0.5, 300)          # bandpass (Hz) - IBL's destripe_lfp uses 0.5-300 Hz, 3rd order
    filter_order: int = 3
    fs: float = 1000                        # output sampling rate (Hz)
    channel_step: int = 1                   # keep every n-th channel (2 halves the disk space)
    motion_correction: str = 'medicine'     # 'medicine' (reuse the MEDiCINe motion of the sorting pipeline) or None - see above
    bad_channel_psd_threshold: float = 1.4  # bad (noisy) channel threshold for the LF band - the value IBL uses in destripe_lfp

def load_lfp_params():
    """LFPParams with the overrides from LFP_PARAMS in the config."""
    return LFPParams(**config.LFP_PARAMS)

# %% extraction

def load_raw_lfp_source(rec_id, logger=None):
    """The LF stream if it exists, otherwise the AP stream. Returns (recording, 'lf' / 'ap')."""
    raw_path = rec_id.folder('raw_data', mkdir=False)
    ap_stream = get_imec_stream_id(rec_id, raw_path)
    lf_stream = ap_stream.replace('.ap', '.lf')
    if find_stream_file(raw_path, lf_stream) is not None:
        log_or_print(f"Loading the {lf_stream} stream for the LFP.", logger=logger)
        return se.read_spikeglx(raw_path, stream_id=lf_stream), 'lf'
    log_or_print(f"No {lf_stream} file (NP2.0 probe?) - deriving the LFP from {ap_stream}.", logger=logger)
    return se.read_spikeglx(raw_path, stream_id=ap_stream), 'ap'

def get_lfp_motion(rec, rec_id, motion_correction, logger=None):
    """The motion to correct the LFP with (motion_correction: 'medicine' / None, see the module docstring), or None."""
    if motion_correction is None:
        return None
    if motion_correction == 'medicine':
        from sorting_pipeline import load_medicine_motion, MEDICINE_FILES
        medicine_folder = rec_id.folder('medicine', mkdir=False)
        if not all((medicine_folder / f).exists() for f in MEDICINE_FILES):
            from sorting_pipeline import get_motion_object, load_preprocessed_recording
            log_or_print("No MEDiCINe motion yet - estimating it from the AP data (peaks on the preprocessed data, no sorting).", logger=logger)
            get_motion_object(load_preprocessed_recording(rec_id, logger=logger), logger=logger)
        log_or_print("Correcting the LFP with the spike-based (MEDiCINe) motion.", logger=logger)
        return load_medicine_motion(rec_id, rec)  # the LF and AP streams start within a sample of each other
    raise ValueError(f"Unknown LFP motion_correction: {motion_correction}. Use 'medicine' or None.")

def extract_lfp(rec_id, params=None, logger=None):
    """
    Compute & save the LFP of a recording (skipped if preprocessing/lfp already exists).
    """
    params = params or load_lfp_params()
    lfp_folder = rec_id.folder('lfp', mkdir=False)
    if is_nonempty_dir(lfp_folder):
        log_or_print(f"LFP folder {lfp_folder} already exists - not recomputing.", logger=logger)
        return si.BaseRecording.load(lfp_folder)

    rec, source = load_raw_lfp_source(rec_id, logger=logger)
    freq_min, freq_max = params.freq_range
    # filtering is done in chunks - a 0.5 Hz highpass needs seconds of margin to avoid edge effects (SpikeInterface's example uses 1.5 s)
    rec = spre.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max, filter_order=params.filter_order, margin_ms=5000, dtype='float32')
    rec = spre.phase_shift(rec)

    # bad channels (IBL method, as in the AP preprocessing of SpikeInterface; IBL uses psd_hf_threshold=1.4 for the LF band)
    bad_channel_ids, channel_labels = spre.detect_bad_channels(rec, method='coherence+psd', psd_hf_threshold=params.bad_channel_psd_threshold)
    to_interpolate = rec.channel_ids[np.isin(channel_labels, ['dead', 'noise'])]
    log_or_print(f"LFP channel labels: {dict(zip(*np.unique(channel_labels, return_counts=True)))}; interpolating {len(to_interpolate)} dead/noisy channels.", logger=logger)
    if len(to_interpolate):
        rec = spre.interpolate_bad_channels(rec, bad_channel_ids=to_interpolate)
    rec.set_property('lfp_channel_label', np.asarray(channel_labels).astype(str))

    motion = get_lfp_motion(rec, rec_id, params.motion_correction, logger=logger)
    if rec.get_sampling_frequency() != params.fs:
        rec = spre.resample(rec, resample_rate=params.fs, margin_ms=1000)
    # motion correction before channel subsampling (the interpolation needs the neighboring channels),
    # with the same settings as for the AP data in sorting_pipeline.do_motion_correction
    if motion is not None:
        rec = InterpolateMotionRecording(rec, motion, border_mode='force_extrapolate')
    if params.channel_step > 1:
        rec = rec.select_channels(rec.channel_ids[::params.channel_step])

    settings = dict(source=source, **asdict(params), n_interpolated=len(to_interpolate))
    rec.annotate(**rec_id.kwargs, lfp=True, lfp_settings=settings, lfp_date=str(pd.Timestamp.now()))
    log_or_print(f"Saving LFP for {rec_id} ({settings}) to {lfp_folder}.", logger=logger)
    return rec.save(folder=lfp_folder, overwrite=True)

def load_lfp(rec_id, data_location='local'):
    """The saved LFP recording (lazy - nothing is read until get_traces)."""
    lfp_folder = rec_id.folder('lfp', location=data_location, mkdir=False)
    require_output(lfp_folder, 'LFP', 'run lfp.py (or sorting_pipeline.py with EXTRACT_LFP = True in the config)')
    return si.BaseRecording.load(lfp_folder)

# %% pipeline: LFP only, without sorting

def lfp_pipeline(rec_id, logger=None):
    """
    The LFP of a recording on its own (python lfp.py ...), e.g. when the sorting is done or not needed.
    Skipped if preprocessing/lfp exists, unless OVERWRITE_PROCESSING. With SERVER_SYNC['lfp'], the probe files are copied from the server
    first and the LFP is copied back after.
    """
    if server_sync_enabled('lfp'):
        copy_from_server(rec_id=rec_id, files_to_copy='neuropixel_data', logger=logger)
    if config.OVERWRITE_PROCESSING:
        delete_outputs(rec_id, ['lfp'], logger=logger)
    extract_lfp(rec_id, logger=logger)
    update_info_df(rec_id, columns=['lfp_extracted'], logger=logger)
    if server_sync_enabled('lfp'):
        copy_to_server(rec_id=rec_id, folders_to_copy=['lfp'], logger=logger)

if __name__ == '__main__':
    batch_kwargs = main_pipeline_parser()
    run_batch_pipeline(lfp_pipeline, 'lfp', batch_kwargs=batch_kwargs)
