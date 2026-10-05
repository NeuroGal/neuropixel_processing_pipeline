# Gal Vishne 2025-2026
# contact me at gal.vishne@gmail.com
"""
Pipeline settings - all settings live here (the command line only selects which recordings to run, and which config file to use).

Don't edit this file for your own setup: copy local_config.example.py to local_config.py and override whatever you need there
(local_config.py is git-ignored, so every user/rig can have its own). To keep several configs (e.g. one per rig / task),
make more files like it and choose one when running: python behavior_pipeline.py --config configs/taylor_rig.py ...
A config file only needs the settings it changes; everything else keeps the defaults below.
"""
import copy
import runpy
from pathlib import Path

# %% general

# Path to the recording table: a csv file with one row per recording (see "Recording table" in the README and recording_info.example.csv),
# e.g. "D:/Code/recording_info.csv".
INFO_CSV = None

# Columns of the recording table that together uniquely identify a recording.
# They are used to select rows, to annotate recordings/sortings, and to name log & output files
# (e.g. ['monkey', 'date', 'region'] -> "Jones_20260702_Pul_...").
RECORDING_KEYS = ['monkey', 'date', 'region']
# Other columns of the recording table to select recordings by on the command line (the RECORDING_KEYS always can):
# each column gets a --<column>s argument, e.g. --monkeys Jones --session_nums 25 26
FILTER_COLUMNS = ['session_num']

# Folder for the summary logs of batch runs (one file per pipeline: summary_pipeline_<name>.log).
# None -> a 'logs' folder inside this code folder.
SUMMARY_LOG_DIR = None

# Whether each pipeline copies data between the local folder and the server folder (folder_local / folder_server in the recording table).
# Leave all False if you work on a single machine. Copying never deletes anything - the local copies stay where they are. When on:
#   sorting:        copies the raw probe data from the server before, and raw_data, motion_corrected_data, medicine, spike_sorting
#                   (and lfp) to the server after (not preprocessed_data - fast to recompute)
#   postprocessing: copies the postprocessing folder to the server after
#   behavior:       copies the second system's file(s) and the NIDQ file from the server before, and task/data (and eye) to the server after
#   lfp:            (lfp.py, LFP only) copies the probe files from the server before, and the lfp folder to the server after
#   (each also copies its log file to the server)
# Files that already exist locally are not copied from the server again, and folders that already exist on the server are not
# copied to it again (so after a rerun, delete the server copy to update it).
SERVER_SYNC = dict(sorting=False, postprocessing=False, behavior=False, lfp=False)

# Parallel processing (SpikeInterface job kwargs, see https://spikeinterface.readthedocs.io/en/stable/modules/core.html#parallel-processing-and-job-kwargs)
#   n_jobs: number of cores, or a fraction of them; chunk_duration: lower it (e.g. '0.5s') if you run out of RAM - it is also passed to Kilosort
#   can also include mp_context (e.g. 'spawn') and total_memory
JOB_KWARGS = dict(n_jobs=0.85, chunk_duration='0.5s')

# %% overwriting existing outputs

# Sorting, postprocessing, LFP and the continuous eye data: by default, steps whose output already exists are skipped (or the run stops
# with a message if an output looks incomplete). True -> the outputs of the pipeline you run are DELETED and recomputed for the selected
# recordings (e.g. sorting_pipeline.py deletes the preprocessed / motion corrected data and the Kilosort output). Use with care.
OVERWRITE_PROCESSING = False
# Trial-aligned data (task/data, written by behavior_pipeline.py): rewritten on every run by default.
# False -> recordings that already have task data are skipped.
OVERWRITE_TASK_DATA = True

# %% sorting pipeline

# Parameter overrides - the defaults (and what each one means) are in SortingParams (sorting_pipeline.py), e.g. dict(detect_threshold=5.0).
# Motion correction: MEDiCINe by default; dict(motion_correction='kilosort') uses Kilosort's own drift correction instead (nblocks=5).
SORTING_PARAMS = {}
# Keep preprocessing/preprocessed_data after the sorting. With MEDiCINe it is only the input to the motion correction, so False deletes it
# once the motion-corrected recording and the sorting exist - saving a copy the size of the raw .ap.bin. (With Kilosort's own drift
# correction or no correction it is what Kilosort and the postprocessing read, so it is always kept.)
KEEP_PREPROCESSED_DATA = True

# LFP (see lfp.py) - computed in sorting_pipeline.py, after the sorting. To compute only the LFP, without sorting: python lfp.py ...
EXTRACT_LFP = False
# Parameter overrides - the defaults (band, sampling rate, motion correction, ...) are in LFPParams (lfp.py), e.g. dict(fs=2500).
# The LFP reuses the MEDiCINe motion (estimated from the AP data if there is none yet); with Kilosort's drift correction, or to skip
# motion correction, set dict(motion_correction=None).
LFP_PARAMS = {}

# %% postprocessing pipeline

# Parameter overrides - the defaults are in PostprocessingParams (postprocessing_pipeline.py), e.g. dict(noise_cutoff=0.6)
POSTPROCESSING_PARAMS = {}
# Export each sorting stage to phy (analyzer/phy) for manual curation. Nothing in the pipeline uses it.
# The export points phy to the motion-corrected data instead of copying it.
EXPORT_TO_PHY = False

# %% behavior pipeline (task events -> trials)

# The system that recorded the task event codes next to the neuropixels: currently implemented for 'plexon' or 'blackrock'.
# To add another system, see "Adapting to your setup" in the README.
SECOND_SYSTEM = 'plexon'
# The task parser: the name of a file in tasks/ ('dots_2d', 'vd_pulse', or your own). To add a task, see "Task parsers" in the README.
TASK = 'dots_2d'
# How the second system's clock is aligned to the neuropixels: 'sync_signal' (shared sync pulses) or 'codes' (task codes recorded on both).
# See "Clock alignment" in the README.
ALIGNMENT_METHOD = 'sync_signal'
# Where the behavior pipeline reads & writes: 'local' or 'server' (the folder_local / folder_server columns of the recording table).
BEHAVIOR_DATA_LOCATION = 'local'
# Which sortings to export trial-aligned spikes for: postprocessing stages ('original', 'merged', 'cleaned') and/or 'kilosort'
# (the Kilosort output directly, without postprocessing: this pipeline's, or the sorting_folder of the recording table).
# [] -> no spikes: only the trials, eye data and trial LFP.
BEHAVIOR_SORTING_TYPES = ['original', 'merged', 'cleaned']

# Spikes (and eye / LFP windows) are taken from t_start - T_BEFORE_TRIAL to t_end + T_AFTER_TRIAL of each trial (seconds).
T_BEFORE_TRIAL = 1.0
T_AFTER_TRIAL = 1.0

# Which units get keep_unit=True in the units table (a pandas query on the unit metrics: quality metrics, template metrics, classifier labels).
# Units are never removed from the saved files - keep_unit only marks them (TrialData.good() selects them).
# Set to None to make this field void.
GOOD_UNITS_QUERY = "presence_ratio > 0.75 and firing_rate / presence_ratio > 3"

# Eye data: saved continuously (preprocessing/eye) and per trial (task/data).
EYE_DATA = True
# Analog channels of the second system with the eye signals: x, y, pupil - verify the channel names on your rig!
#   plexon: e.g. ['AI01', 'AI02', 'AI03'] (Taylor's rig); blackrock: e.g. ['ainp1', 'ainp2', 'ainp3'] (analog inputs in a .nsX file)
EYE_CHANNELS = ['AI01', 'AI02', 'AI03']
# Sampling rate (Hz) the eye signals are saved at: signals recorded faster are downsampled to it (standard anti-aliased resampling,
# scipy.signal.resample_poly); signals recorded at this rate or slower are kept at their original rate.
EYE_SAMPLING_RATE = 1000

# LFP per trial (task/data): needs the LFP from sorting_pipeline.py (EXTRACT_LFP). Large files - one LFP window per trial.
TRIAL_LFP = False
# None, or 'median' to subtract, at each sample, the median across the in-brain channels (as IBL does). Dead / noisy channels were already
# replaced by interpolation from their neighbors when the LFP was extracted, so they are part of the median; channels outside the brain are not.
TRIAL_LFP_REFERENCE = None

# %% second system channels (check these on your rig)

# verify the channel names on your rig!
# Event channel with the task codes - plexon: 'Strobed'; blackrock: 'digital_input_port'
CODES_CHANNEL = 'Strobed'
# Sync pulses shared with the NIDQ (for ALIGNMENT_METHOD = 'sync_signal')
#   plexon: event channel, pulses = code 1, e.g. 'EVT32'; blackrock: analog input channel in a .nsX file, e.g. 'ainp16'
SYNC_CHANNEL = 'EVT32'

# %% SpikeGLX channels (check these on your rig)

# NIDQ channel (index into the saved NIDQ channels) with the sync pulses shared with the second system (analog, thresholded at half its max).
NIDQ_SYNC_CHANNEL = 0
# NIDQ channel with the task codes (a digital word - the last saved channels), for ALIGNMENT_METHOD = 'codes' (see raw_io.load_nidq_codes).
NIDQ_CODES_CHANNEL = -1
# Bit of the imec SY channel that holds the SpikeGLX sync wave (6 for NP1.0 & NP2.0 probes, see the SpikeGLX docs).
IMEC_SYNC_BIT = 6
# Also map the NIDQ clock to the probe (imec) clock (= spike times) using the SpikeGLX sync wave recorded in both, like TPrime.
# See README "Clock alignment". If its checks fail, behavior stays in the NIDQ clock and the log has a WARNING.
ALIGN_NIDQ_TO_IMEC = True

# %% loading a config file (local_config.py by default, or --config on the command line)

_DEFAULTS = {key: copy.deepcopy(value) for key, value in dict(globals()).items() if key.isupper()}
CONFIG_FILE = None

def use_config_file(path):
    """Reset all settings to the defaults above, then apply the settings (UPPERCASE names) defined in a config file."""
    settings = globals()
    settings.update(copy.deepcopy(_DEFAULTS))
    for key, value in runpy.run_path(str(path)).items():
        if key.isupper():
            if key not in _DEFAULTS:
                raise ValueError(f"Unknown setting {key} in {path} (see config.py for the available settings).")
            settings[key] = value
    settings['CONFIG_FILE'] = str(path)

if (Path(__file__).parent / 'local_config.py').exists():
    use_config_file(Path(__file__).parent / 'local_config.py')
