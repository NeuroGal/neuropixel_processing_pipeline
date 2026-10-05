# Gal Vishne 2025-2026
# contact me at gal.vishne@gmail.com
"""
The output of the behavior pipeline: saving, and loading for analysis.

Continuous data (saved with the preprocessed data, <keys> = e.g. Jones_20260702_Pul):
    preprocessing/eye/<keys>_eye.npz      continuous eye signals in the neuropixel clock: times, signals (samples x channels), channels, fs, units
    preprocessing/lfp/                    continuous LFP (SpikeInterface binary folder, see lfp.py)

Trial-aligned data (in <recording>/task/data):
    <keys>_trials.csv             trial table (one row per trial, incl. keep_trial)
    <keys>_task_data_info.json    settings used (task, alignment, clock, trial window, unit selection, ...)
    <keys>_eye_trials.npy         (if eye data) (n_trials,) object array: samples x (channels..., time) per trial, time relative to t_start
    <keys>_lfp_trials.npy         (if TRIAL_LFP) all trials' LFP windows one after the other (samples x channels, float32, uV) -
    <keys>_lfp_trials_index.npz       too large for an object array in memory; the index has where each trial starts. Use TrialData.get_lfp
                                      (reads one trial at a time from disk, so the file is never kept open).
    <stage>_sorting/<keys>_spikes.npy   (n_trials x n_units) object array: spike times (s) relative to each trial's t_start
    <stage>_sorting/<keys>_units.csv    unit metrics (one row per unit = column of spikes, incl. keep_unit)
All windows run from t_start - t_before to t_end + t_after (see the info file).

Typical use:
    data = load_trial_data(dict(monkey='Jones', date='20260702', region='Pul'), sorting_type='cleaned').good()
    data.spikes[trial_i, unit_j]           # spike times of unit j in trial i
    times, eye = data.get_eye(trial_i)     # eye signals around trial i
    times, lfp = data.get_lfp(trial_i)     # LFP around trial i (if TRIAL_LFP was on)
"""
import json
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from pipelines_functions import RecordingID, log_or_print, save_json

# %% file names

def get_task_data_files(rec_id, sorting_type=None, data_location='local', mkdir=False):
    rec_id = RecordingID.make(rec_id)
    folder = rec_id.folder('task/data', location=data_location, mkdir=mkdir)
    prefix = str(rec_id)
    files = dict(trials=folder / f'{prefix}_trials.csv', info=folder / f'{prefix}_task_data_info.json',
                 eye_trials=folder / f'{prefix}_eye_trials.npy',
                 lfp_trials=folder / f'{prefix}_lfp_trials.npy', lfp_trials_index=folder / f'{prefix}_lfp_trials_index.npz')
    if sorting_type is not None:
        sorting_folder = folder / f'{sorting_type}_sorting'
        if mkdir:
            sorting_folder.mkdir(parents=True, exist_ok=True)
        files.update(spikes=sorting_folder / f'{prefix}_spikes.npy', units=sorting_folder / f'{prefix}_units.csv')
    return files

def get_continuous_eye_file(rec_id, data_location='local', mkdir=False):
    rec_id = RecordingID.make(rec_id)
    return rec_id.folder('eye', location=data_location, mkdir=mkdir) / f'{rec_id}_eye.npz'

# %% continuous eye

def save_continuous_eye(rec_id, eye, data_location='local'):
    np.savez(get_continuous_eye_file(rec_id, data_location, mkdir=True), **eye)

def load_continuous_eye(rec_id, data_location='local'):
    """dict(times (s, neuropixel clock), signals (samples x channels), channels, fs, units)"""
    return load_npz(get_continuous_eye_file(rec_id, data_location))

def load_npz(file):
    """All arrays of an .npz file, with the file closed again (np.load keeps .npz files open otherwise - which blocks overwriting them on Windows)."""
    with np.load(file) as npz:
        return dict(npz)

def read_npy_rows(file, start, stop):
    """Rows start:stop of a 2D .npy file, read directly from disk (the file is opened only for this read)."""
    with open(file, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = (np.lib.format.read_array_header_1_0 if version[0] == 1 else np.lib.format.read_array_header_2_0)(f)
        row_size = int(np.prod(shape[1:]))
        f.seek(f.tell() + start * row_size * dtype.itemsize)
        return np.fromfile(f, dtype=dtype, count=(stop - start) * row_size).reshape((stop - start,) + tuple(shape[1:]))

# %% saving trial-aligned data

def save_session_data(rec_id, trials, info, data_location='local'):
    """Save what is shared by all sorting stages: the trial table and the settings."""
    files = get_task_data_files(rec_id, data_location=data_location, mkdir=True)
    trials.to_csv(files['trials'], index=False)
    save_json(info, files['info'])

def save_sorting_data(rec_id, sorting_type, spikes, units, data_location='local'):
    """Save the spikes & unit table of one sorting stage."""
    files = get_task_data_files(rec_id, sorting_type=sorting_type, data_location=data_location, mkdir=True)
    np.save(files['spikes'], spikes)
    units.to_csv(files['units'], index=False)

def save_trial_eye(rec_id, eye, trials, t_before, t_after, data_location='local'):
    """
    Eye data per trial: (n_trials,) object array, each samples x (eye channels..., time), time relative to that trial's t_start
    (like the spikes and the event times in the trial table).
    """
    eye_trials = np.empty(len(trials), dtype=object)
    for tr_i, (t_start, t_end) in enumerate(trials[['t_start', 't_end']].itertuples(index=False)):
        i0, i1 = np.searchsorted(eye['times'], [t_start - t_before, t_end + t_after])
        eye_trials[tr_i] = np.column_stack([eye['signals'][i0:i1], eye['times'][i0:i1] - t_start]).astype(np.float32)
    np.save(get_task_data_files(rec_id, data_location=data_location, mkdir=True)['eye_trials'], eye_trials)

def read_lfp_traces(lfp, start_frame, end_frame, channel_ids=None):
    """LFP traces in uV (handles the renamed argument in newer SpikeInterface versions)."""
    try:
        return lfp.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=channel_ids, return_scaled=True)
    except TypeError:
        return lfp.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=channel_ids, return_in_uV=True)

def save_trial_lfp(rec_id, trials, t_before, t_after, reference=None, data_location='local', logger=None):
    """
    LFP per trial, from the continuous LFP (preprocessing/lfp). Written trial by trial into a memory-mapped file,
    so the whole thing never has to fit in memory. reference: None or 'median' (subtract the median across the in-brain channels).
    """
    from lfp import load_lfp
    lfp = load_lfp(rec_id, data_location=data_location)
    fs, n_samples = lfp.get_sampling_frequency(), lfp.get_num_samples()
    start_frames = np.clip(np.floor((trials['t_start'].values - t_before) * fs).astype(int), 0, n_samples)
    end_frames = np.clip(np.ceil((trials['t_end'].values + t_after) * fs).astype(int), 0, n_samples)
    offsets = np.concatenate([[0], np.cumsum(end_frames - start_frames)])
    labels = lfp.get_property('lfp_channel_label')
    in_brain = np.ones(len(lfp.channel_ids), dtype=bool) if labels is None else (labels != 'out')

    files = get_task_data_files(rec_id, data_location=data_location, mkdir=True)
    out = np.lib.format.open_memmap(files['lfp_trials'], mode='w+', dtype=np.float32, shape=(int(offsets[-1]), len(lfp.channel_ids)))
    for tr_i, (start, end) in enumerate(zip(start_frames, end_frames)):
        traces = read_lfp_traces(lfp, start, end)
        if reference == 'median':
            traces = traces - np.median(traces[:, in_brain], axis=1, keepdims=True)
        elif reference is not None:
            raise ValueError(f"Unknown reference: {reference}. Use None or 'median'.")
        out[offsets[tr_i]:offsets[tr_i + 1]] = traces
    out.flush()
    del out
    np.savez(files['lfp_trials_index'], offsets=offsets, start_frames=start_frames, fs=fs, channel_ids=np.asarray(lfp.channel_ids).astype(str),
             channel_labels=np.asarray(labels if labels is not None else []).astype(str), reference=str(reference))
    log_or_print(f"Saved the LFP of {len(trials)} trials ({offsets[-1] * len(lfp.channel_ids) * 4 / 1e9:.1f} GB).", logger=logger)

# %% loading

@dataclass
class TrialData:
    rec_id: RecordingID
    sorting_type: str
    trials: pd.DataFrame       # one row per trial (t_start / t_end in the neuropixel clock, other times relative to t_start)
    units: pd.DataFrame        # one row per unit
    spikes: np.ndarray         # (n_trials, n_units) object array of spike times relative to t_start
    info: dict
    eye: np.ndarray = None     # (n_trials,) object array: samples x (eye channels..., time relative to t_start)
    eye_channels: tuple = ()
    lfp_file: str = None       # all trials' LFP windows (read one trial at a time by get_lfp)
    lfp_index: dict = None

    def good(self):
        """Only the trials with keep_trial and the units with keep_unit."""
        keep_trials = self.trials['keep_trial'].values.astype(bool)
        keep_units = self.units['keep_unit'].values.astype(bool)
        return replace(self, trials=self.trials[keep_trials].reset_index(drop=True),
                       units=self.units[keep_units].reset_index(drop=True),
                       spikes=self.spikes[keep_trials][:, keep_units],
                       eye=self.eye[keep_trials] if self.eye is not None else None)

    def get_eye(self, trial_i):
        """Eye signals of a trial: times (s, relative to t_start), signals (samples x channels, see eye_channels)."""
        if self.eye is None:
            raise ValueError("No eye data for this recording.")
        return self.eye[trial_i][:, -1], self.eye[trial_i][:, :-1]

    def get_lfp(self, trial_i):
        """LFP of a trial: times (s, relative to t_start), traces (samples x channels, uV; channels: lfp_index['channel_ids'])."""
        if self.lfp_file is None:
            raise ValueError("No trial LFP for this recording (TRIAL_LFP in the config).")
        trial_num = self.trials.iloc[trial_i]['trial_num']   # row in the full trial table (also after good())
        offsets, start_frames, fs = self.lfp_index['offsets'], self.lfp_index['start_frames'], float(self.lfp_index['fs'])
        traces = read_npy_rows(self.lfp_file, offsets[trial_num], offsets[trial_num + 1])
        times = (start_frames[trial_num] + np.arange(len(traces))) / fs - self.trials.iloc[trial_i]['t_start']
        return times, traces


def load_trial_data(rec_id, sorting_type='cleaned', data_location='local', good_only=False):
    """
    Load the behavior pipeline output of one recording & sorting stage ('original', 'merged', 'cleaned').
    rec_id: a RecordingID or a dict of the recording keys, e.g. dict(monkey='Jones', date='20260702', region='Pul')
    """
    rec_id = RecordingID.make(rec_id)
    files = get_task_data_files(rec_id, sorting_type=sorting_type, data_location=data_location)
    if not files['spikes'].exists():
        raise ValueError(f"{files['spikes']} does not exist. Run behavior_pipeline.py first.")
    with open(files['info']) as f:
        info = json.load(f)
    eye, eye_channels = None, ()
    if files['eye_trials'].exists():
        eye = np.load(files['eye_trials'], allow_pickle=True)
        eye_file = get_continuous_eye_file(rec_id, data_location)
        eye_channels = tuple(load_npz(eye_file)['channels']) if eye_file.exists() else ()
    lfp_file, lfp_index = None, None
    if files['lfp_trials'].exists():
        lfp_file = files['lfp_trials']
        lfp_index = load_npz(files['lfp_trials_index'])
    data = TrialData(rec_id=rec_id, sorting_type=sorting_type, trials=pd.read_csv(files['trials']), units=pd.read_csv(files['units']),
                     spikes=np.load(files['spikes'], allow_pickle=True), info=info, eye=eye, eye_channels=eye_channels,
                     lfp_file=lfp_file, lfp_index=lfp_index)
    return data.good() if good_only else data
