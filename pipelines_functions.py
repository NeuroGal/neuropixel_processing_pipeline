"""
Gal Vishne 2025-2026
contact me at gal.vishne@gmail.com

Shared infrastructure for all pipelines:
- the recording table (INFO_CSV) & folder layout of each recording
- logging (per recording + a summary log per batch run)
- copying to/from the server
- running a pipeline over a batch of recordings from the command line
"""
# %% imports
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import traceback
import shutil
from tqdm import tqdm
import argparse
import json
from dataclasses import dataclass

import config

# %% the recording table (INFO_CSV)

# what the behavior pipeline can take spikes from: the postprocessing stages, or the Kilosort output directly (no postprocessing needed)
BEHAVIOR_SORTING_OPTIONS = ['original', 'merged', 'cleaned', 'kilosort']

def validate_config(pipeline_name):
    """
    Check the settings a pipeline needs before it starts (called by run_batch_pipeline), so a wrong setting fails right away
    instead of in the middle of a run. All problems are reported together.
    """
    problems = []
    # the recording table
    if config.INFO_CSV is None:
        problems.append("INFO_CSV (the path to the recording table) is not set.")
    elif not Path(config.INFO_CSV).exists():
        problems.append(f"INFO_CSV: the recording table {config.INFO_CSV} does not exist.")
    else:
        info_df = pd.read_csv(config.INFO_CSV)
        required = list(config.RECORDING_KEYS) + ['folder_local']
        if config.SERVER_SYNC.get(pipeline_name, False) or get_pipeline_location(pipeline_name) == 'server':
            required.append('folder_server')
        if pipeline_name == 'behavior':
            required.append('trigger_filename')
        missing = [col for col in required if col not in info_df.columns]
        if missing:
            problems.append(f"The recording table {config.INFO_CSV} is missing the columns {missing}.")
        elif info_df[config.RECORDING_KEYS].duplicated().any():
            duplicates = info_df[info_df[config.RECORDING_KEYS].duplicated(keep=False)][config.RECORDING_KEYS]
            problems.append(f"Several rows of the recording table have the same RECORDING_KEYS:\n{duplicates.to_string()}")
    # behavior pipeline settings
    if pipeline_name == 'behavior':
        from raw_io import FILE_SUFFIX
        from tasks import available_tasks
        options = dict(SECOND_SYSTEM=list(FILE_SUFFIX), TASK=available_tasks(), ALIGNMENT_METHOD=['sync_signal', 'codes'], BEHAVIOR_DATA_LOCATION=['local', 'server'])
        for setting, allowed in options.items():
            if getattr(config, setting) not in allowed:
                problems.append(f"{setting} = {getattr(config, setting)!r} - must be one of {allowed}.")
        if config.ALIGNMENT_METHOD == 'sync_signal' and config.SYNC_CHANNEL is None:
            problems.append("SYNC_CHANNEL is not set (needed for ALIGNMENT_METHOD = 'sync_signal').")
        unknown = [s for s in config.BEHAVIOR_SORTING_TYPES if s not in BEHAVIOR_SORTING_OPTIONS]
        if unknown:
            problems.append(f"BEHAVIOR_SORTING_TYPES: unknown {unknown} - use any of {BEHAVIOR_SORTING_OPTIONS} (or [] for no spikes).")
    if problems:
        raise ValueError(f"Config problems ({config.CONFIG_FILE or 'config.py defaults'}):\n- " + "\n- ".join(problems))

def normalize_key_value(x):
    """
    Recording key values are compared as strings. Pandas reads e.g. dates as floats when the column has missing values
    (20260702.0), so integer-valued numbers are converted to '20260702'.
    """
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    return str(x)

def find_info_row(info_df, keys):
    """Index (position) of the row of the recording table matching all the given key values."""
    idx = np.ones(len(info_df), dtype=bool)
    for key, value in keys.items():
        if key not in info_df.columns:
            raise ValueError(f"Key {key} not found in the recording table columns.")
        idx = idx & (info_df[key].apply(normalize_key_value) == normalize_key_value(value)).values
    if sum(idx) == 0:
        raise ValueError(f"No recording found for {keys}. Please check the input parameters.")
    elif sum(idx) > 1:
        raise ValueError(f"Multiple recordings found for {keys}. Please check the input parameters.")
    return np.where(idx)[0][0]

# %% recording identity & folder layout

# folders of each recording, relative to its root folder (folder_local / folder_server in the recording table)
SUBFOLDERS = {
    'root': '.',
    'raw_data': 'raw_data',
    'preprocessing': 'preprocessing',
    'preprocessed_data': 'preprocessing/preprocessed_data',
    'motion_corrected_data': 'preprocessing/motion_corrected_data',
    'medicine': 'preprocessing/medicine',
    'sorting': 'preprocessing/spike_sorting',
    'lfp': 'preprocessing/lfp',
    'eye': 'preprocessing/eye',
    'postprocessing': 'postprocessing',
    **{f'{stage}_sorting{sub}': f'postprocessing/{stage}_sorting{sub}'
       for stage in ['original', 'merged', 'cleaned'] for sub in ['', '/analyzer', '/classification', '/figures', '/settings']},
    'task': 'task',
    'task/data': 'task/data',
    'task/figures': 'task/figures',
}

@dataclass(frozen=True)
class RecordingID:
    """
    Identifies one recording (= one row of the recording table) by the values of its RECORDING_KEYS, e.g. monkey/date/region.
    str(rec_id) gives the name used for logs and files, e.g. 'Jones_20260702_Pul'.
    Create with RecordingID.from_dict(dict(monkey='Jones', date='20260702', region='Pul')) or RecordingID.from_annotations(extractor).
    """
    keys: tuple   # ((key, value), ...) in the order of RECORDING_KEYS, values as strings

    @classmethod
    def from_dict(cls, d):
        missing = [key for key in config.RECORDING_KEYS if key not in d]
        if missing:
            raise ValueError(f"Missing recording keys {missing} (RECORDING_KEYS = {config.RECORDING_KEYS}).")
        return cls(tuple((key, normalize_key_value(d[key])) for key in config.RECORDING_KEYS))

    @classmethod
    def from_annotations(cls, extractor):
        """From the annotations of a SpikeInterface recording / sorting (the pipelines annotate them with the recording keys)."""
        return cls.from_dict({key: extractor.get_annotation(key=key) for key in config.RECORDING_KEYS})

    @classmethod
    def make(cls, x):
        """RecordingID from a RecordingID or a dict."""
        return x if isinstance(x, cls) else cls.from_dict(x)

    @property
    def kwargs(self):
        """The keys as a dict (e.g. to annotate SpikeInterface objects: extractor.annotate(**rec_id.kwargs))."""
        return dict(self.keys)

    def __str__(self):
        return '_'.join(value for _, value in self.keys)

    @property
    def info(self):
        """The row of the recording table for this recording (pd.Series)."""
        info_df = pd.read_csv(config.INFO_CSV)
        return info_df.iloc[find_info_row(info_df, self.kwargs)]

    def folder(self, subfolder='root', location='local', mkdir=True):
        """
        A folder of this recording (see SUBFOLDERS), on the local machine or the server.
        mkdir=True creates it if needed - use mkdir=False when you only want to check if a step was computed.
        The local raw data folder can be anywhere: the raw_data_folder column of the recording table, if filled (default <folder_local>/raw_data).
        """
        info = self.info
        if subfolder == 'raw_data' and location == 'local' and isinstance(info.get('raw_data_folder'), str):
            path = Path(info.raw_data_folder)
        else:
            root = Path(info.folder_local if location == 'local' else info.folder_server)
            path = root / SUBFOLDERS[subfolder] if subfolder != 'root' else root
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)
        return path

def find_file(folder, name):
    """
    A file in a folder or any of its subfolders (e.g. SpikeGLX keeps each probe's files in its own subfolder).
    If it is nowhere, folder / name is returned (so the error later on says which file is missing).
    """
    folder = Path(folder)
    if (folder / name).exists():
        return folder / name
    matches = sorted(folder.rglob(name)) if folder.exists() else []
    if len(matches) > 1:
        raise ValueError(f"Several files named {name} in {folder}: {matches}. Keep one recording per raw data folder.")
    return matches[0] if matches else folder / name

def is_nonempty_dir(path):
    return path.exists() and any(path.iterdir())

def require_output(path, description, how_to_make_it):
    """Raise a clear error if an output a step needs is missing, e.g. require_output(folder, 'preprocessed data', 'run the preprocessing')."""
    if not is_nonempty_dir(path):
        raise ValueError(f"No {description} in {path} - {how_to_make_it} first.")

def require_no_output(path, description):
    """Raise a clear error before overwriting an existing output (the guard of OVERWRITE_PROCESSING = False)."""
    if is_nonempty_dir(path):
        raise ValueError(f"The {description} already exists in {path}. Delete it (or set OVERWRITE_PROCESSING = True) to recompute it.")

def delete_outputs(rec_id, subfolders, logger=None, location='local'):
    """Delete output folders of a recording (used when OVERWRITE_PROCESSING / OVERWRITE_TASK_DATA is set)."""
    for subfolder in subfolders:
        folder = rec_id.folder(subfolder, location=location, mkdir=False)
        if folder.exists():
            log_or_print(f"Overwriting: deleting {folder}.", logger=logger)
            try:
                shutil.rmtree(folder)
            except PermissionError as e:
                raise PermissionError(f"Could not delete {folder} - a file in it is open (e.g. in another program) - "
                                      f"close it and rerun. ({e})") from e

# %% saving json

def save_json(data, file, keep_existing=False, logger=None):
    """
    Save data (dicts / lists / numbers / strings, incl. numpy values) to a json file.
    keep_existing: if the file already exists (e.g. a rerun), save to file_2.json, file_3.json, ... instead of overwriting it.
    Returns the path saved to.
    """
    file = Path(file)
    base, n = file, 1
    while keep_existing and file.exists():
        n += 1
        file = base.with_name(f'{base.stem}_{n}{base.suffix}')
    with open(file, 'w') as f:
        # json can't save numpy arrays / numbers: convert them to lists / python numbers (anything else unknown is saved as text)
        json.dump(data, f, indent=4, default=lambda obj: obj.tolist() if isinstance(obj, (np.ndarray, np.generic)) else str(obj))
    log_or_print(f"Saved {file}.", logger=logger)
    return file

# %% logging

def log_or_print(message, logger=None):
    """
    Log a message to the logger or print it to the console.
    """
    if logger is None:
        print(message)
    else:
        for line in message.splitlines():
            logger.info(line)

def get_pipeline_location(pipeline_name):
    """Where a pipeline reads & writes: the behavior pipeline - BEHAVIOR_DATA_LOCATION; sorting & postprocessing - always local."""
    return config.BEHAVIOR_DATA_LOCATION if pipeline_name == 'behavior' else 'local'

def setup_logging(mode, log_name, rec_id=None, location='local'):
    """
    Setup logging for a specific recording or the summary of all recordings, logging to both a file and the terminal.
    rec_id - RecordingID of the recording; location - which root folder of the recording the log goes to ('local' / 'server')
    """
    if mode == 'recording':
        log_name_full = f"{rec_id}_pipeline_{log_name}"
        log_path = rec_id.folder('root', location=location) / f"{log_name_full}.log"
    elif mode == 'summary': # summary logging
        log_name_full = f"summary_pipeline_{log_name}"
        log_dir = Path(config.SUMMARY_LOG_DIR) if config.SUMMARY_LOG_DIR is not None else Path(__file__).parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{log_name_full}.log'
    else:
        raise ValueError("Invalid mode. Must be 'recording' or 'summary'.")

    log_handler = logging.FileHandler(log_path, mode='a')
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger(log_name_full)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger, log_handler, stream_handler

def pipeline_with_logging(rec_id, pipeline_func, log_name, summary_logger=None):
    """
    Run a processing function for a specific recording, with logging for that recording and a summary logger.
    Errors are logged (with traceback) instead of raised, so a batch continues to the next recording.

    Parameters:
    - rec_id: RecordingID of the recording
    - pipeline_func: The function to execute (should accept rec_id, logger as arguments).
    - log_name: Name for the log file (for both summary and single recordings).
    - summary_logger: Optional logger for summarizing across recordings.
    Returns True if the pipeline completed, False if it raised an error.
    """
    location = get_pipeline_location(log_name)
    recording_logger, log_handler, stream_handler = setup_logging(mode='recording', log_name=log_name, rec_id=rec_id, location=location)
    root_folder = rec_id.folder('root', location=location)
    def write_to_logs(message):
        recording_logger.info(message + " (recording logger)")
        if summary_logger is not None:
            summary_logger.info(message + " (summary logger)")

    success = False
    try:
        write_to_logs(f"Starting pipeline for recording {rec_id} ({root_folder}).")
        pipeline_func(rec_id, logger=recording_logger)
        write_to_logs(f"Pipeline completed for recording {rec_id} ({root_folder}).")
        success = True
    except Exception as e:
        error_message = f"Error in recording {rec_id}: {str(e)}"
        traceback_info = traceback.format_exc()
        write_to_logs(f"{error_message}\nTraceback:\n{traceback_info}")
    finally:
        # Cleanup: Remove handlers to prevent duplication
        recording_logger.removeHandler(log_handler)
        recording_logger.removeHandler(stream_handler)
        log_handler.close()
    return success

# %% updating the info_df

def update_info_df(rec_id, columns, logger=None):
    """
    Mark the given status columns as True for this recording in the recording table (columns are created if missing).
    """
    info_df = pd.read_csv(config.INFO_CSV)
    idx = find_info_row(info_df, rec_id.kwargs)
    for c in columns:
        info_df.loc[idx, c] = True
    info_df.to_csv(config.INFO_CSV, index=False)
    log_or_print(f"Updated info_df for {rec_id}.", logger=logger)

# %% copying to the server

def server_sync_enabled(pipeline_name):
    return config.SERVER_SYNC.get(pipeline_name, False)

def single_file_copy(src_file, dst_file):
    """
    Copy a single file from src_file to dst_file.
    """
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)  # copy2 preserves metadata use shutil.copyfile if there are issues with this
    # Compare file sizes to verify the copy
    if src_file.stat().st_size != dst_file.stat().st_size:
        raise IOError(f"File copy failed: {src_file} -> {dst_file} (size mismatch)")

def same_file(src_file, dst_file):
    """Whether dst_file is already a copy of src_file (same size & modification time - copy2 keeps the time; 2 s tolerance for network drives)."""
    if not dst_file.exists():
        return False
    src, dst = src_file.stat(), dst_file.stat()
    return src.st_size == dst.st_size and abs(src.st_mtime - dst.st_mtime) < 2

def sync_folder(src_folder, dst_folder, mirror=False, logger=None):
    """
    Make dst_folder up to date with src_folder: copy the files that are new or changed (unchanged files are skipped, so reruns are fast).
    mirror: also delete files in dst_folder that are not in src_folder (for outputs that the pipeline rebuilds from scratch).
    """
    src_files = [f for f in src_folder.rglob('*') if f.is_file()]
    to_copy = [f for f in src_files if not same_file(f, dst_folder / f.relative_to(src_folder))]
    for f in tqdm(to_copy, desc=f'Copying {src_folder.name}'):
        single_file_copy(f, dst_folder / f.relative_to(src_folder))
    n_deleted = 0
    if mirror and dst_folder.exists():
        src_relative = {f.relative_to(src_folder) for f in src_files}
        for f in [f for f in dst_folder.rglob('*') if f.is_file() and f.relative_to(dst_folder) not in src_relative]:
            f.unlink()
            n_deleted += 1
    log_or_print(f"{src_folder} -> {dst_folder}: copied {len(to_copy)} new / changed files, {len(src_files) - len(to_copy)} were up to date"
                 + (f", deleted {n_deleted} files no longer in the source." if mirror else "."), logger=logger)

def copy_to_server(rec_id, folders_to_copy, mirror=False, copy_log=True, logger=None):
    """
    Copy pipeline outputs to the server (folder_server in the recording table), updating what is already there:
    new and changed files are copied, unchanged ones skipped. Nothing is deleted locally.
    folders_to_copy: list of subfolder names, as defined in SUBFOLDERS (e.g. 'sorting', 'postprocessing', 'task/data')
    mirror: also delete server files that are not in the local folder (used for task/data, which the behavior pipeline rebuilds
            from scratch - so files of an older run don't stay on the server). Never used for raw data.
    copy_log: also copy this recording's log file.
    """
    for folder in folders_to_copy:
        local_folder = rec_id.folder(folder, location='local', mkdir=False)
        require_output(local_folder, f'{folder} folder', 'run the pipeline')
        sync_folder(local_folder, rec_id.folder(folder, location='server', mkdir=False), mirror=mirror, logger=logger)

    if copy_log:  # the log file of this recording (overwritten on the server)
        if logger is None:
            raise ValueError("Logger must be provided if copy_log is True. Use setup_logging to create a logger.")
        log_src = Path([handler.baseFilename for handler in logger.handlers if isinstance(handler, logging.FileHandler)][0])
        log_dest = rec_id.folder('root', location='server') / log_src.name
        log_or_print(f"Copying log file to server: {log_src} -> {log_dest}", logger=logger)
        single_file_copy(log_src, log_dest)

def copy_from_server(rec_id, files_to_copy='all', skip_if_exists=True, logger=None):
    """
    Copy raw files from the server raw_data folder to the local raw_data folder.
    files_to_copy: 'triggers' (the second system's file(s)), 'neuropixel_triggers' (nidq), 'neuropixel_data' (imec ap), or 'all'
    (file names are taken from the trigger_filename / neuropixel_filename columns of the recording table)
    """
    # server files
    recording_server_folder = rec_id.folder('root', location='server', mkdir=False)
    info = rec_id.info
    server_raw_path = recording_server_folder / 'raw_data'
    def info_value(col):
        value = info.get(col)
        return None if pd.isna(value) else str(value)

    if info_value('trigger_filename'):
        from raw_io import FILE_SUFFIX
        trigger_file = find_file(server_raw_path, Path(info_value('trigger_filename')).with_suffix(FILE_SUFFIX[config.SECOND_SYSTEM]).name)
        triggers = [trigger_file]
        if config.SECOND_SYSTEM == 'blackrock':  # continuous signals (eye, sync inputs) are in the .nsX files next to the .nev
            triggers += sorted(trigger_file.parent.glob(trigger_file.stem + '.ns[0-9]'))
    else:
        triggers = None
    if info_value('neuropixel_filename'):
        neuropixel_filename = info_value('neuropixel_filename')
        nidq_name = neuropixel_filename.split('.')[0]
        neuropixel_triggers = [find_file(server_raw_path, f"{nidq_name}.nidq{ext}") for ext in ['.bin', '.meta']]
        neuropixel_data = [find_file(server_raw_path, Path(neuropixel_filename).with_suffix(ext).name) for ext in ['.bin', '.meta']]
    else:
        neuropixel_triggers = neuropixel_data = None
    file_groups = dict(triggers=triggers, neuropixel_triggers=neuropixel_triggers, neuropixel_data=neuropixel_data)

    if files_to_copy == 'all':
        files_to_copy = [f for group in file_groups.values() if group is not None for f in group]
    elif files_to_copy in file_groups:
        if file_groups[files_to_copy] is None:
            raise ValueError(f"The file name needed for '{files_to_copy}' is not defined in the info_df for recording {rec_id}.")
        files_to_copy = file_groups[files_to_copy]
    else:
        raise ValueError(f"Invalid files_to_copy option: {files_to_copy}. Choose one of {list(file_groups)} or 'all'.")

    # local folder
    session_local_folder = rec_id.folder('raw_data', location='local')

    for server_path in files_to_copy:
        filename = server_path.name
        local_path = find_file(session_local_folder, filename)  # an existing copy may be in a subfolder
        if local_path.exists() and skip_if_exists:
            log_or_print(f"Not copying {filename} to {local_path} because it already exists and skip_if_exists==True.", logger=logger)
        else:
            log_or_print(f"Copying {filename} to {local_path}", logger=logger)
            single_file_copy(server_path, local_path)
            log_or_print(f"Finished copying {filename} to {local_path}", logger=logger)

# %% run the pipeline

def get_filter_columns():
    """Columns of the recording table that recordings can be selected by: the RECORDING_KEYS + FILTER_COLUMNS of the config."""
    return list(dict.fromkeys(list(config.RECORDING_KEYS) + list(config.FILTER_COLUMNS)))

def get_batch_recordings(filters=None, logger=None):
    """
    The recordings to run: the rows of the recording table matching all the filters (no filters - all rows).
    filters: dict of column -> list of allowed values, e.g. dict(monkey=['Jones'], date=['20260702', '20260703'])
    :return: list of RecordingID (used everywhere to load the specific data)
    """
    info_df = pd.read_csv(config.INFO_CSV)
    idx = np.ones(len(info_df), dtype=bool)
    for col, values in (filters or {}).items():
        if col not in info_df.columns:
            raise ValueError(f"Can't select recordings by {col}: the recording table has no {col} column.")
        idx = idx & info_df[col].apply(normalize_key_value).isin([normalize_key_value(v) for v in values]).values
    log_or_print(f"Found {sum(idx)} recordings.", logger=logger)
    batch_recordings = [RecordingID.from_dict(row) for _, row in info_df[idx].iterrows()]
    if len(batch_recordings) == 0:
        raise ValueError("No recordings found for the specified criteria.")
    return batch_recordings

def run_batch_pipeline(pipeline_func, pipeline_name, batch_kwargs=None):
    """
    Run the pipeline_func for a given batch of recordings (loops over them).
    :param pipeline_func: function, the pipeline function to use for each recording (receives rec_id, logger as arguments)
    :pipeline_name: str, name of the pipeline (used for log names) (e.g. 'postprocessing', 'sorting', etc.)
    :param batch_kwargs: which recordings to run - dict of column -> values (see get_batch_recordings / main_pipeline_parser)
    Parallel processing settings come from JOB_KWARGS in the config.
    """
    validate_config(pipeline_name)
    batch_kwargs = batch_kwargs or {}
    summary_logger, log_handler, stream_handler = setup_logging(mode='summary', log_name=pipeline_name)
    log_or_print(f"Starting batch pipeline {pipeline_name} for recordings {batch_kwargs or 'all'} (config file: {config.CONFIG_FILE}).", logger=summary_logger)

    from spikeinterface import set_global_job_kwargs, get_global_job_kwargs
    set_global_job_kwargs(**config.JOB_KWARGS)
    log_or_print(str(get_global_job_kwargs()), logger=summary_logger)

    batch_recordings = get_batch_recordings(batch_kwargs, logger=summary_logger)
    failed = []
    for rec_id in batch_recordings:
        success = pipeline_with_logging(rec_id=rec_id, pipeline_func=pipeline_func, log_name=pipeline_name, summary_logger=summary_logger)
        if not success:
            failed.append(str(rec_id))

    log_or_print(f"Pipeline completed: {len(batch_recordings) - len(failed)}/{len(batch_recordings)} recordings succeeded.", logger=summary_logger)
    if failed:
        log_or_print(f"Failed recordings: {failed}. Check their individual logs for details.", logger=summary_logger)
    summary_logger.removeHandler(log_handler)
    summary_logger.removeHandler(stream_handler)
    log_handler.close()

def main_pipeline_parser():
    """
    Command line of the pipelines: which recordings to run, and (optionally) which config file to use.
    The recordings are selected with --<column>s for each column in get_filter_columns() (RECORDING_KEYS + FILTER_COLUMNS of the config),
    e.g. --monkeys Jones --dates 20260702 20260703. All other settings are in the config (config.py / local_config.py).
    Returns the filters, as a dict of column -> values (see get_batch_recordings).
    """
    # the config file first - the filter arguments depend on its RECORDING_KEYS / FILTER_COLUMNS
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None, help='Config file to use instead of local_config.py (see config.py).')
    config_args, _ = config_parser.parse_known_args()
    if config_args.config is not None:
        config.use_config_file(config_args.config)

    parser = argparse.ArgumentParser(description="Run a pipeline on recordings of the recording table. All settings are in the config file.",
                                     parents=[config_parser])
    for col in get_filter_columns():
        parser.add_argument(f'--{col}s', dest=col, nargs='+', type=str, default=None, help=f'Values of the {col} column to select (default: all).')
    args = parser.parse_args()
    return {col: getattr(args, col) for col in get_filter_columns() if getattr(args, col) is not None}
