# Neuropixels → trials pipeline

Takes Neuropixels recordings (SpikeGLX) plus the event codes (and eye data) of a task computer recorded on Blackrock or Plexon. It produces sorted, curated units and trial-aligned spikes, eye signals and LFP.

Built on [SpikeInterface](https://spikeinterface.readthedocs.io), [Kilosort4](https://kilosort.readthedocs.io), [MEDiCINe](https://github.com/jazlab/medicine) and [UnitRefine](https://huggingface.co/collections/SpikeInterface/curation-models-66ebf53d7f17088991062c2c).

Gal Vishne, 2025–2026 (gal.vishne@gmail.com)

The pipeline's design and its core procedures are Gal Vishne's own implementation, built on the packages above: the postprocessing stages (auto-merging, classification, cleaning), the clock alignment, the trial parsing and the trial-aligned outputs. For questions, contact Gal.

## Citing

A citation for this pipeline is in progress. If you use it, please contact Gal Vishne (gal.vishne@gmail.com). Please also cite SpikeInterface, Kilosort4, MEDiCINe and UnitRefine (see [References](#references)).

The code is under the MIT license (see `LICENSE`).

## What it does

There are three pipelines, run in this order. Each one works on a batch of recordings (rows of the [recording table](#2-recording-table)), and each step saves its output. You can also start from your own sorting, see [Starting from your own sorting](#starting-from-your-own-sorting).

| Script | Steps | Output (per recording) |
|---|---|---|
| `sorting_pipeline.py` | bandpass 300–6000 Hz, phase shift, common reference → motion correction → Kilosort4 (its CAR turned off). Motion correction is MEDiCINe's estimate interpolated into the recording (default), or Kilosort's own drift correction (`SORTING_PARAMS = dict(motion_correction='kilosort')`). Optional: LFP extraction | `preprocessing/` |
| `postprocessing_pipeline.py` | SpikeInterface analyzer with all extensions → iterative auto-merging → noise-unit removal. Each stage is classified with UnitRefine (noise/neural, SUA/MUA) and plotted, and optionally exported to phy | `postprocessing/{original,merged,cleaned}_sorting/` |
| `behavior_pipeline.py` | load event codes and eye data → map them to the Neuropixels clock → parse trials with a task parser → cut spikes, eye data (and optionally LFP) around each trial → mark good trials and units | `task/data/`, `preprocessing/eye/` |

### Code layout

| File | What it does |
|---|---|
| `config.py` / `local_config.py` | **all settings** (see [Setup](#setup)) |
| `pipelines_functions.py` | the recording table, the folder layout (`RecordingID`), logging, server copying, batch running and the command line |
| `sorting_pipeline.py`, `postprocessing_pipeline.py`, `behavior_pipeline.py` | the three pipelines. Each pipeline's parameters are in one dataclass at the top of its file (`SortingParams`, `PostprocessingParams`), and the clock alignment is in `behavior_pipeline.py` |
| `tasks/` | **task trigger parsers** (event codes → trial table): one file per task (`dots_2d.py`, `vd_pulse.py`), a template for new tasks (`_template.py`), general tools (`trial_codes.py`) and our task computer's conventions (`our_rig_codes.py`) |
| `raw_io.py` | reading raw files: single SpikeGLX channels (sync, codes; little memory), probe selection, and Blackrock / Plexon events, sync pulses and eye signals |
| `sorting_plots.py` | the postprocessing figures (unit summaries, merge candidates, metric distributions, classifier probabilities), also usable on their own |
| `lfp.py` | LFP extraction; also runs on its own (`python lfp.py`) for the LFP only |
| `trial_data.py` | the behavior pipeline's output files, plus `load_trial_data` / `TrialData` for analysis |
| `run_pipelines.ps1` | example batch run on Windows |

## Installation

You need an NVIDIA GPU for Kilosort4: at least 8 GB of GPU memory, 12 GB or more recommended. Everything runs in one Python environment (Python 3.11). SpikeInterface calls Kilosort4 directly, so both must be installed in the same environment.

1. **Install [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/)**, then create the environment from this folder:
   ```
   conda env create -f environment.yml
   conda activate spikesorting
   ```
   This installs everything the pipelines import (see `environment.yml`):

   | Package | Version | Used for |
   |---|---|---|
   | [SpikeInterface](https://spikeinterface.readthedocs.io/en/stable/get_started/installation.html) `[full]` | 0.102.x (the postprocessing uses its auto-merge API) | reading SpikeGLX, preprocessing, running Kilosort, the analyzer, curation, phy export |
   | [Kilosort4](https://github.com/MouseLand/Kilosort) | below 4.1.2 (newer versions break SpikeInterface 0.102's Kilosort wrapper) | spike sorting |
   | [MEDiCINe](https://github.com/jazlab/medicine) (`medicine-neuro`, imported as `medicine`) | | motion estimation |
   | [neo](https://neo.readthedocs.io) | below 0.14.5 (needed by SpikeInterface 0.102 to read SpikeGLX) | reading SpikeGLX, Blackrock and Plexon files |
   | `huggingface_hub`, `skops`, `scikit-learn` | scikit-learn 1.4.2 (the UnitRefine models don't load with newer versions) | downloading and running the UnitRefine classifiers |
   | `networkx` | | combining merge suggestions |
   | `matplotlib`, `seaborn` | | figures |
   | `numpy`, `scipy`, `pandas`, `tqdm` | numpy with OpenBLAS (the MKL build crashed numpy's linear algebra on Windows) | general |

2. **Replace PyTorch with its GPU build.** Step 1 installs the CPU-only PyTorch (as a Kilosort dependency), and Kilosort4 is very slow on it. Installing over it does nothing, so uninstall it first. Pick the CUDA version that matches your GPU and driver from [pytorch.org](https://pytorch.org/get-started/locally/) (see also the [Kilosort installation instructions](https://github.com/MouseLand/Kilosort#installation)). For example, for an RTX 4000 series GPU:
   ```
   pip uninstall -y torch
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
   To check that it worked:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   This should print `True`.

3. **Optional, for looking at sortings:** the Kilosort GUI (`pip install kilosort[gui]`), and [phy](https://github.com/cortex-lab/phy), best in its own environment as its installation page describes.

The UnitRefine classifiers are downloaded from HuggingFace the first time the postprocessing runs, so that run needs internet access.

### Disk space and run time

For one 3.2 h NP1.0 recording (384 channels; raw `.ap.bin` ~270 GB), measured on a 32-core PC with an RTX 4080, with `n_jobs` 0.25–0.5 (other jobs were running):

| Output | Size | Time |
|---|---|---|
| `preprocessing/preprocessed_data` (filtered + CAR) | as large as the raw `.ap.bin` (~250 GB) | ~1.5–2 h |
| `preprocessing/motion_corrected_data` (MEDiCINe) | as large again (~250 GB) | ~1.5 h (peaks + MEDiCINe ~1 h more the first time) |
| `preprocessing/spike_sorting` (Kilosort4) | ~10 GB | ~3 h |
| `preprocessing/lfp` | ~17 GB | ~25 min |
| `postprocessing/` | ~15 GB | ~7–10 h (most of it computing amplitudes and locations of every spike) |
| `task/data` (with the LFP per trial) | ~20 GB (LFP per trial ~18 GB) | ~10 min |

So plan for about twice the raw file's size in free space while sorting. With MEDiCINe, `KEEP_PREPROCESSED_DATA = False` deletes the preprocessed copy once the sorting is done, since nothing reads it afterwards.

## Setup

### 1. Config

All settings are in the config; the command line only selects recordings. Copy `local_config.example.py` to `local_config.py` and set at least `INFO_CSV`. Your config only needs the settings you change: everything else keeps the defaults in `config.py`, where every setting is documented. A misspelled or outdated setting name stops the run with an error rather than being silently ignored. Before each run, the pipeline also checks that the settings it needs are valid and that the recording table has the columns it needs. All problems are listed together, so a config mistake shows up immediately rather than in the middle of a run.

`local_config.py` is git-ignored, so every machine or rig can have its own. To keep several configs (e.g. one per rig or task), make more files like it and choose one with `--config` when running (see [Running](#running)).

| Setting | Meaning |
|---|---|
| `INFO_CSV` | path to the **recording table** (step 2) |
| `RECORDING_KEYS` | columns of the recording table that together identify a recording (default `monkey, date, region`), see [Recording identifiers](#recording-identifiers) |
| `FILTER_COLUMNS` | other columns to select recordings by on the command line (default `session_num`) |
| `SERVER_SYNC` | per pipeline, whether to copy inputs from and outputs to `folder_server`. Copies are updated on reruns: new and changed files are copied, and nothing is deleted locally. Leave everything `False` on a single machine |
| `JOB_KWARGS` | parallel processing: `n_jobs` (number of cores, or a fraction of them) and `chunk_duration` (lower it, e.g. `'0.5s'`, if you run out of RAM; also passed to Kilosort) |
| `OVERWRITE_PROCESSING`, `OVERWRITE_TASK_DATA` | what happens to existing outputs, see [Reruns and overwriting](#reruns-and-overwriting) |
| `KEEP_PREPROCESSED_DATA` | `False` deletes the preprocessed recording after the sorting (with MEDiCINe), see [Disk space and run time](#disk-space-and-run-time) |
| `SORTING_PARAMS`, `POSTPROCESSING_PARAMS`, `LFP_PARAMS` | overrides of the processing parameters, see [Parameters](#parameters) |
| `EXTRACT_LFP` | compute the [LFP](#lfp) in the sorting pipeline (or run `lfp.py` for the LFP only) |
| `EXPORT_TO_PHY` | export each sorting stage to phy for manual curation (default `False`). The export points phy to the motion-corrected data in `preprocessing/` instead of copying it |
| `SECOND_SYSTEM`, `TASK` | the system that recorded the task codes (`blackrock` / `plexon`) and the [task parser](#task-parsers), for the whole recording table. Use separate configs for recordings from different rigs or tasks |
| `ALIGNMENT_METHOD` | `sync_signal` (default) or `codes`, see [Clock alignment](#clock-alignment) |
| `BEHAVIOR_DATA_LOCATION` | `local` (default) or `server`: where the behavior pipeline reads and writes. With `server` it works directly on `folder_server`, including its logs. Sorting and postprocessing always run locally |
| `BEHAVIOR_SORTING_TYPES` | which sortings get trial-aligned spikes: postprocessing stages (`original`, `merged`, `cleaned`) and/or `kilosort` (the Kilosort output directly, see [Starting from your own sorting](#starting-from-your-own-sorting)). `[]` makes only the trials, eye data and trial LFP |
| `T_BEFORE_TRIAL`, `T_AFTER_TRIAL` | data window around each trial (default 1 s each) |
| `GOOD_UNITS_QUERY` | **unit selection**, see [below](#unit-and-trial-selection) |
| `EYE_DATA`, `EYE_CHANNELS`, `EYE_SAMPLING_RATE` | [eye data](#eye-data) |
| `TRIAL_LFP`, `TRIAL_LFP_REFERENCE` | [LFP per trial](#lfp) |
| `CODES_CHANNEL`, `SYNC_CHANNEL`, `NIDQ_SYNC_CHANNEL`, `NIDQ_CODES_CHANNEL`, `IMEC_SYNC_BIT`, `ALIGN_NIDQ_TO_IMEC` | where the codes and sync signals are on your rig (the config lists the Plexon and Blackrock names), see [Clock alignment](#clock-alignment) |

### 2. Recording table

A csv with one row per recording (one probe in one session), at the path given by `INFO_CSV`. See `recording_info.example.csv`.

| Column | Required | Meaning |
|---|---|---|
| `monkey`, `date`, `region` (your `RECORDING_KEYS`) | yes | identify the recording. Their values also name logs and output files, e.g. `Jones_20260702_Pul_...` |
| `folder_local` | yes | root folder of the recording on this machine |
| `folder_server` | with server sync | root folder of the recording on the server |
| `raw_data_folder` | no | where this recording's raw files are, if not in `<folder_local>/raw_data` (e.g. the SpikeGLX run folder, wherever it already is) |
| `neuropixel_filename` | if several probes share a raw data folder | e.g. `run_g0_t0.imec1.ap.bin`: selects the probe, and gives the NIDQ name for server copies |
| `sorting_folder` | no | an existing Kilosort / phy output folder to use instead of sorting with this pipeline, see [Starting from your own sorting](#starting-from-your-own-sorting) |
| `trigger_filename` | for the behavior pipeline | name of the second system's file. The extension is added from `SECOND_SYSTEM`: `.nev` for Blackrock (the `.nsX` files with the same name hold its analog inputs), `.pl2` for Plexon |
| `session_num` (your `FILTER_COLUMNS`) | no | only used to select recordings (`--session_nums`) |
| any other columns | no | ignored by the pipelines - add your own notes freely |

The pipelines also write status columns back to the table, such as `kilosorted`, `postprocessing_finalized` and `behavior_parsed`. Because of this, don't keep it open in Excel while a pipeline runs.

### 3. Raw data

Each recording's raw data folder holds:
- the SpikeGLX files: the probe's `*.imecN.ap.bin/.meta` (and `.lf` if you have it) and `*.nidq.bin/.meta`
- the second system's file(s) (`trigger_filename`)

Where that folder is:
- By default it's `<folder_local>/raw_data/`.
- If the data already sits somewhere else, set `raw_data_folder` in the recording table instead of copying it.

Files are also found in subfolders, so SpikeGLX's own layout works as it is. That layout keeps each probe in its own subfolder (`run_g0/run_g0_imec0/...ap.bin`) with the NIDQ file one level up. Keep one recording (one SpikeGLX run / gate / trigger) per folder.

With `SERVER_SYNC` turned on, missing files are copied from `<folder_server>/raw_data/` (also searched in subfolders).

## Adapting to your setup

The defaults match our rigs: SpikeGLX with Plexon or Blackrock, and the `dots_2d` / `vd_pulse` tasks. Here is what to check or change for yours.

- **Your recordings aren't identified by monkey / date / region** (e.g. subject, session, probe): only the config and the recording table change, see [Recording identifiers](#recording-identifiers).
- **You run a different task:** copy the task template and fill in your codes, see [Adding a task](#adding-a-task).
- **Your signals are recorded on different channels:** the pipeline needs to know which channels carry the task codes, the sync pulses and the eye signals. Set these in your config:

  | Setting | What it is | Ours |
  |---|---|---|
  | `CODES_CHANNEL` | the second system's event channel with the task codes | Plexon `'Strobed'`; Blackrock `'digital_input_port'` |
  | `SYNC_CHANNEL` | the second system's channel with the sync pulses that the NIDQ also records (for `ALIGNMENT_METHOD = 'sync_signal'`) | Plexon `'EVT32'` (event channel, pulses = code 1); Blackrock: an analog input, e.g. `'ainp16'` |
  | `EYE_CHANNELS` | the second system's analog channels with eye x, y, pupil | Plexon `['AI01', 'AI02', 'AI03']` |
  | `NIDQ_SYNC_CHANNEL` | which NIDQ channel has those sync pulses (index in the saved channels) | `0` |
  | `NIDQ_CODES_CHANNEL` | which NIDQ channel has the task codes (only for `ALIGNMENT_METHOD = 'codes'`) | `-1` (the last one) |
  | `IMEC_SYNC_BIT` | the bit of the probe's SY channel with the SpikeGLX sync wave | `6` (the SpikeGLX standard; rarely needs changing) |

  To find the channel names in a Plexon or Blackrock file:
  ```python
  import neo
  reader = neo.rawio.Plexon2RawIO(filename='path/to/file.pl2')   # Blackrock: neo.rawio.BlackrockRawIO(filename='path/to/file.nev')
  reader.parse_header()
  print(reader.header['event_channels']['name'])    # candidates for CODES_CHANNEL (and Plexon's SYNC_CHANNEL)
  print(reader.header['signal_channels']['name'])   # candidates for EYE_CHANNELS (and Blackrock's SYNC_CHANNEL)
  ```
  For the NIDQ, channels are counted in the order SpikeGLX saves them: the analog channels, then the digital word last. The `snsMnMaXaDw` line of the `.nidq.meta` file gives how many there are of each kind.
- **Processing parameters (optional):** the defaults are what we use. You only need to change them to tune something, e.g. spike detection thresholds or how aggressively units are merged. See [Parameters](#parameters).
- **Your task events are recorded by a system other than Blackrock or Plexon:** add your system as described in [Adding a recording system](#adding-a-recording-system).

## Running

```
python sorting_pipeline.py        --monkeys Jones --dates 20260702 --regions Pul LIP
python postprocessing_pipeline.py --monkeys Jones --dates 20260702 --regions Pul LIP
python behavior_pipeline.py       --monkeys Jones --dates 20260702 --regions Pul LIP --config configs/taylor_rig.py
```

| Argument | Meaning |
|---|---|
| `--monkeys`, `--dates`, `--regions`, `--session_nums` | select rows of the recording table (each takes one or more values). There is one `--<column>s` argument for each of your `RECORDING_KEYS` and `FILTER_COLUMNS`, so these names follow your config (`--help` lists them). With no filters, every row is run |
| `--config` | a config file to use instead of `local_config.py` |

Everything else is set in the config. On Windows you can instead edit the settings at the top of `run_pipelines.ps1` and run it.

**Running only part of it:**
- **LFP only, without sorting:** `python lfp.py --monkeys Jones ...` (same arguments). See [LFP](#lfp).
- **Trials and eye data only, without spikes:** `BEHAVIOR_SORTING_TYPES = []` in the config, then run `behavior_pipeline.py`. Add `TRIAL_LFP = True` for the LFP per trial.
- **From a sorting made elsewhere:** see [Starting from your own sorting](#starting-from-your-own-sorting).

**If the computer freezes or crashes during a run** (usually CPU or memory overload), try these:
- Lower `n_jobs` and/or `chunk_duration` in `JOB_KWARGS`.
- Limit the threads each process may start. SpikeInterface runs `n_jobs` processes in parallel, and numpy / scipy / torch inside each can start a thread on every core, which together can be far more threads than the computer has. Before running, set `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` (e.g. to 6, as on our rigs). The lines for this are in `run_pipelines.ps1`; in another terminal use `set OMP_NUM_THREADS=6` (Windows cmd) or `export OMP_NUM_THREADS=6` (Linux/macOS).

**Logs.** Each recording gets a log in its root folder (`<keys>_pipeline_<name>.log`). Each batch also gets a summary log (`summary_pipeline_<name>.log`) that lists the config file used and which recordings failed. An error in one recording is logged, and the batch continues with the next one.

### Check your first recording

Two quick checks before trusting the outputs of a new rig or dataset:
1. **Clock alignment, in the behavior log** (`<keys>_pipeline_behavior.log`):
   - The second system and the NIDQ should have the same number of sync pulses (e.g. `Number of sync times in system2: 11628, number of sync times in Neuropixel: 11628`).
   - The NIDQ → probe line should match all sync edges, with an offset of about a millisecond that changes slowly (e.g. `matched 11628/11628 sync edges; probe - NIDQ offset is 0.66 ms at the start and 0.56 ms at the end`).
   - A `WARNING` there means the behavior stayed in the NIDQ clock.
2. **MEDiCINe's motion estimate**, in `preprocessing/medicine/corrected_motion_raster.png`. The left panel is the uncorrected spike raster (depth over time) and the middle one the estimate.
   - If bands in the left panel drift and the estimate follows them, the correction is right.
   - MEDiCINe can also read a change in which neurons are active as motion. On a session with little real drift, applying such an estimate halved the number of neural units.
   - If the left panel shows no drift, sort with `SORTING_PARAMS = dict(motion_correction=None)` (or `'kilosort'`).

### Reruns and overwriting

**Sorting, postprocessing, LFP and continuous eye data** are protected by default (`OVERWRITE_PROCESSING = False`):
- A step whose output already exists is skipped, or its output is loaded, and the rest continues from there.
- A step that crashed halfway can leave a non-empty folder behind. The pipeline then refuses to continue, with a message telling you which folder to delete. For Kilosort, a sorting folder without `sorter_output/sorting_details.json` counts as unfinished.
- To redo one step, delete its folder.
- To redo everything a pipeline produces for the selected recordings, set `OVERWRITE_PROCESSING = True`. Its outputs are **deleted** and recomputed: e.g. `sorting_pipeline.py` deletes the preprocessed and motion-corrected data, the MEDiCINe cache, the Kilosort output and, with `EXTRACT_LFP`, the LFP.

**Trial-aligned data** (`task/data`) is rewritten on every run by default (`OVERWRITE_TASK_DATA = True`). The folder is cleared first, so no files from earlier runs are left. Set it to `False` to skip recordings that already have task data.

## Output

```
<folder_local>/
├── raw_data/
├── preprocessing/
│   ├── preprocessed_data/        filtered + CAR (SpikeInterface binary folder); input to Kilosort with its own drift correction
│   ├── medicine/                 peaks, peak locations, motion estimate (cached), with MEDiCINe
│   ├── motion_corrected_data/    input to Kilosort, with MEDiCINe
│   ├── spike_sorting/            Kilosort4 output (sorter_output/)
│   ├── lfp/                      continuous LFP, if extracted (SpikeInterface binary folder)
│   └── eye/<keys>_eye.npz        continuous eye signals in the Neuropixels clock, if any
├── postprocessing/
│   └── {original,merged,cleaned}_sorting/
│       ├── analyzer/             SortingAnalyzer (+ analyzer/phy with EXPORT_TO_PHY)
│       ├── classification/       UnitRefine labels (csv) + classifier details
│       ├── figures/              unit summaries, metrics, merge candidates (merged only)
│       └── settings/             merge / cleaning parameters (json)
└── task/data/
    ├── <keys>_trials.csv               trial table, one row per trial (incl. keep_trial)
    ├── <keys>_task_data_info.json      settings used (task, alignment, clock, trial window, unit criteria, config file)
    ├── <keys>_eye_trials.npy           eye data per trial, if any
    ├── <keys>_lfp_trials.npy (+ _index.npz)   LFP per trial, with TRIAL_LFP
    └── {original,merged,cleaned,kilosort}_sorting/   (the BEHAVIOR_SORTING_TYPES)
        ├── <keys>_spikes.npy           (n_trials × n_units) object array of spike-time arrays
        └── <keys>_units.csv            unit metrics, one row per unit (incl. keep_unit)
```

The trial table and the eye and LFP data are shared by the sorting stages. Nothing is saved twice as "all" and "good": `keep_trial` and `keep_unit` mark what to analyze.

### Loading for analysis

```python
from trial_data import load_trial_data

data = load_trial_data(dict(monkey='Jones', date='20260702', region='Pul'), sorting_type='cleaned')
data = data.good()                     # keep_trial trials & keep_unit units only
data.trials, data.units                # DataFrames
data.spikes[trial_i, unit_j]           # spike times (s) relative to that trial's t_start
times, eye = data.get_eye(trial_i)     # eye signals (samples × data.eye_channels) around the trial
times, lfp = data.get_lfp(trial_i)     # LFP (samples × channels, uV) around the trial
```

Timing conventions:
- Each window runs from `t_start − T_BEFORE_TRIAL` to `t_end + T_AFTER_TRIAL`, and all times in it are relative to `t_start`. This applies to spikes, eye and LFP.
- Event times in the trial table (`t_fix_on`, `t_dots_on`, …) are relative to `t_start` too.
- `t_start` and `t_end` themselves are absolute times in the Neuropixels clock.

The raw files load without this code too:
- `eye_trials.npy` and `spikes.npy` are object arrays: use `np.load(path, allow_pickle=True)`. Each trial's eye array is samples × (eye channels…, time).
- `lfp_trials.npy` is a plain float32 array (samples × channels) with all trials one after the other; `lfp_trials_index.npz` has each trial's start offset.

## Starting from your own sorting

What each pipeline reads:

| Pipeline | Reads |
|---|---|
| `sorting_pipeline.py` | the raw SpikeGLX probe files |
| `postprocessing_pipeline.py` | a Kilosort output (this pipeline's, or `sorting_folder`) and the recording it was sorted from: `preprocessing/motion_corrected_data` if it exists, else `preprocessing/preprocessed_data`, else the raw probe data with this pipeline's preprocessing applied on the fly (nothing saved) |
| `behavior_pipeline.py` | the raw data folder (NIDQ files, the probe's `.meta`/`.bin` for its sync channel, the second system's file); the spikes of each sorting in `BEHAVIOR_SORTING_TYPES`; `preprocessing/lfp` with `TRIAL_LFP` |

To use a sorting made elsewhere (Kilosort, or a phy folder after manual curation):
1. **Point to it:** put its folder (the one with `spike_times.npy`, `spike_clusters.npy` and `params.py`) in the `sorting_folder` column of the recording table. Spike times must be in samples from the start of the probe's `.ap.bin` in the raw data folder, which holds when it was sorted directly from that file (or from CatGT output without trimming).
2. **Trial-aligned spikes without postprocessing:** set `BEHAVIOR_SORTING_TYPES = ['kilosort']` and run `behavior_pipeline.py`. The unit table has Kilosort's own unit properties (`KSLabel`, `ContamPct`, `Amplitude`, and `group` from phy curation if any), plus `num_spikes`, `firing_rate` and `presence_ratio`, so the default `GOOD_UNITS_QUERY` works. The output goes to `task/data/kilosort_sorting/`.
3. **With postprocessing** (merging, UnitRefine classification, the full unit metrics): run `postprocessing_pipeline.py`, then the behavior pipeline with the postprocessing stages.
   - The analyzer takes waveforms, amplitudes and locations from the recording, so it should be the signal your sorter saw. If your sorter corrected drift itself, waveforms of drifting units are blurred in an uncorrected recording (spike times aren't affected).
   - Without a saved recording, the preprocessing is recomputed every time the analyzer reads the data, which is slower. To save it once, run `sorting_pipeline.py`: with `sorting_folder` set, it preprocesses (and by default motion-corrects) but doesn't sort.

The same column also works for this pipeline's own sortings if they live somewhere else.

## Postprocessing

For each recording, `postprocessing_pipeline.py` builds three stages, each from the previous one. Each stage is classified with UnitRefine (noise / neural, SUA / MUA), plotted, and optionally exported to phy.

1. **original:** the Kilosort4 units in a SpikeInterface analyzer, with all extensions: waveforms, templates, principal components, amplitudes, locations, quality and template metrics.
2. **merged:** units merged automatically.
   - Potential merges are suggested with two of SpikeInterface's auto-merge presets and combined: `similarity_correlograms` (conservative, very few false merges) and `x_contaminations` (more liberal, with a stricter refractory period).
   - The suggested merges are applied, and the whole procedure is repeated `n_merge_rounds` times (3), since merged units can suggest new merges.
   - A summary figure of every suggested merge is saved in `merged_sorting/figures`, so you can check them.
3. **cleaned:** the merged units, minus those UnitRefine classifies as noise with probability above `noise_cutoff` (0.5).

The parameters are in `PostprocessingParams`, and the reasoning behind each is in the docstrings of `compute_potential_merges`, `do_sorting_merges` and `do_noise_units_removal`.

> The merge and cleaning procedure (the presets, their parameters, the repeated rounds and the cleaning) is Gal Vishne's implementation, not a standard SpikeInterface workflow. For questions, contact Gal (gal.vishne@gmail.com).

## Unit and trial selection

**Units.** `keep_unit` is set by `GOOD_UNITS_QUERY` in the config. This is a [pandas query](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) on the columns of `units.csv`, the table made by `postprocessing_pipeline.get_unit_metrics` (also used for the figures): quality metrics, template metrics, UnitRefine labels such as `noise_neural_classifier_prediction`, and Kilosort's unit properties such as `KSLabel`. The default is:

```python
GOOD_UNITS_QUERY = "presence_ratio > 0.75 and firing_rate / presence_ratio > 3"
```

In words: present for more than 75% of the session, and firing more than 3 Hz while present. Units with missing metrics are not kept.
- To change it, override it in your config. Setting it to `None` keeps every unit.
- The criteria used are recorded in `task_data_info.json`.
- This is in addition to the noise-unit removal in postprocessing: the `cleaned` stage drops units UnitRefine classifies as noise with p > 0.5.

**Trials.** `keep_trial` is set by the task parser's `keep_trial` function. In both current tasks, it keeps trials with a completed response (correct or error).

## Task parsers

Event codes become a trial table through a parser in `tasks/`, chosen with `TASK` in the config.

| Task | For | Notes |
|---|---|---|
| `dots_2d` | the 2D dots task (motion + color) and its 1D motion / 1D color / overlap / memory variants | |
| `vd_pulse` | VGS / MGS / motion discrimination with a motion pulse (VD_pulse_2), before and after injection | `post_inj` is a column, and `task_type` is the same before and after. 0% coherence is stored as ±0.001 so its direction is kept. `choice` is 1 for right and 0 for left (only for completed trials) |

What's in `tasks/`:

| File | What it is |
|---|---|
| `dots_2d.py`, `vd_pulse.py` | our tasks: one file per task |
| `_template.py` | a complete, commented example to copy for a new task |
| `trial_codes.py` | general tools for any task or lab: split the code stream into trials, read event times and values, label categories, list unused codes |
| `our_rig_codes.py` | the conventions of our task computer, read by our task files: trial start/end codes, how values are encoded, display settings and outcome codes. These numbers are specific to our task software |

### Adding a task

A task is a file `tasks/<name>.py` that defines two functions. The file name is the task name: set `TASK = '<name>'` in your config. There's nothing to register, and the startup check lists the available tasks if the name doesn't match.

1. **Copy the template:** `tasks/_template.py` → `tasks/<your_task>.py`. It's a complete, working example with made-up codes; try it with `python -m tasks._template`.
2. **Fill in your codes** at the top of the file, so the parsing reads like a description of the task:
   - the trial start and end codes
   - the event codes whose times you want
   - the outcome codes
   - the value codes (a code followed by its value, usually with an offset and scale)
   - readable labels for categories
3. **Write `parse_trial(tc)`:** one trial's codes → a dict of columns. `tc` gives:
   - `tc.time(code)`: event time relative to the trial start
   - `tc.present(code)`: whether the code is in the trial
   - `tc.value_after(code, offset, scale)`: a value sent as the next code
4. **The two functions the pipeline calls:**
   - `parse_codes(event_times, event_codes)` → the trial table, one row per trial. It must have `t_start` and `t_end`, the absolute trial start/end times; `parse_trials` adds them. Spikes, eye and LFP are cut around these, and all other times are relative to `t_start`.
   - `keep_trial(trials)` → which trials to analyze, e.g. completed trials (saved as `keep_trial`).
5. **Check it on a recording** with `testing_mode=True`, which also lists the codes your parser didn't read (snippet at the top of `_template.py`). Then run `behavior_pipeline.py` on one recording.

If your trials aren't marked by start/end codes (e.g. only a start code, or trials defined by a separate file), build the table any way you like. Only the two functions and the `t_start` / `t_end` columns matter.

If several of your tasks share code conventions (e.g. the same display-settings codes), put them in one shared file that your task files import, as our `our_rig_codes.py` does for `dots_2d` and `vd_pulse`.

## Eye data

With `EYE_DATA = True` (default), eye signals are read from the second system's analog channels listed in `EYE_CHANNELS` (x, y, pupil). Set them for your rig:
- Plexon: e.g. `['AI01', 'AI02', 'AI03']` (the default, as on Taylor's rig: x, y, pupil, at 1 kHz in volts).
- Blackrock: the analog input names in its `.nsX` files, e.g. `['ainp1', 'ainp2', 'ainp3']`.

Processing:
1. The signals are scaled to physical units and downsampled to `EYE_SAMPLING_RATE` (default 1 kHz) with standard anti-aliased resampling ([`scipy.signal.resample_poly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)). Signals recorded at that rate or slower keep their original rate. The resampling runs in chunks with overlapping edges, so long high-rate recordings never have to fit in memory.
2. Their times are mapped to the Neuropixels clock with the same mapping as the task events.
3. They are saved continuously in `preprocessing/eye/<keys>_eye.npz` (`times`, `signals`, `channels`, `fs`, `units`). On later runs this file is loaded instead of recomputed, unless `OVERWRITE_PROCESSING` is set.
4. They are cut around each trial into `task/data/<keys>_eye_trials.npy`, with times relative to `t_start`, like the spikes.

## LFP

Set `EXTRACT_LFP = True` to compute the continuous LFP in `sorting_pipeline.py` (after the sorting), or run `python lfp.py` (same arguments as the other pipelines) to compute only the LFP, without any sorting. Set `TRIAL_LFP = True` to also cut it around each trial in `behavior_pipeline.py`. The LFP derived from the AP stream (NP2.0 probes, no LF stream) is **untested**. The steps follow two published pipelines:
- the IBL LFP preprocessing ([`destripe_lfp`](https://github.com/int-brain-lab/ibl-neuropixel)): 3rd order Butterworth bandpass 0.5–300 Hz → sample-shift correction → bad-channel interpolation → median common reference
- the SpikeInterface LFP example ([LFP processing](https://spikeinterface.readthedocs.io/en/latest/how_to/drift_with_lfp.html)): bandpass (order 3, long filter margins) → phase shift → resample

Continuous LFP (`preprocessing/lfp`; parameters in `LFPParams` in `lfp.py`, overridden with `LFP_PARAMS` in the config):
1. **Source**: the probe's LF stream (NP1.0, 2.5 kHz, hardware-filtered 0.5–500 Hz) if there is one. Otherwise (NP2.0) it derives the LFP from the 30 kHz AP stream with the same filter.
2. Bandpass `freq_range` (0.5–300 Hz, order 3) → phase shift.
3. **Bad channels**: detected with SpikeInterface's IBL method, using IBL's LF threshold (`bad_channel_psd_threshold = 1.4`). Dead and noisy channels are interpolated; channels outside the brain are kept but labeled (channel property `lfp_channel_label`).
4. Resample to `fs` (1 kHz) → motion correction (below) → optional channel subsampling (`channel_step`).
5. Saved once. It is not common-referenced: IBL subtracts the median across channels, which removes shared noise but also real LFP shared across the probe, and can't be undone in the saved file.

**Motion correction** (`motion_correction`). The LFP channels are interpolated to compensate for probe motion, the same way as the AP data:
- `'medicine'` (default): reuses the MEDiCINe motion estimated from the spikes in the sorting pipeline, so LFP and spikes are corrected with the same estimate. If there's no estimate yet (LFP only, no sorting), it is made from the AP data with the sorting pipeline's settings, without saving the preprocessed data. That adds about two passes over the AP file. It's saved in `preprocessing/medicine`, and a later sorting run reuses it.
- `None`: no correction. Use this when the sorting used Kilosort's own drift correction (which has no estimate to reuse), or to keep the LFP-only run fast.

Correction happens before channel subsampling, since the interpolation needs the neighboring channels.

**LFP per trial** (`TRIAL_LFP`, `task/data/<keys>_lfp_trials.npy`):
- Cut from the continuous LFP around each trial. With `TRIAL_LFP_REFERENCE = 'median'`, the median across the in-brain channels is subtracted at each sample, as IBL does. Dead and noisy channels were already replaced by interpolation, so they're part of the median; channels outside the brain aren't.
  - With an external reference, most of the LFP is shared by all channels and looks the same at every depth. Subtract the median (or another reference) in your analysis to see the depth structure. The saved continuous LFP stays unreferenced so that this choice stays open.
- These files are large (one window of all channels per trial), so they're written trial by trial. `TrialData.get_lfp` reads one trial at a time straight from the file, which isn't kept open, so the behavior pipeline can rewrite it while the data is loaded in a notebook.
- The LF and AP streams share the probe clock, so the LFP is in the same clock as the spikes.

## Parameters

Each pipeline's parameters are in one dataclass at the top of its file, with comments on each choice:
- `SortingParams` (`sorting_pipeline.py`): bandpass, motion correction (`'medicine'`, `'kilosort'` with `kilosort_nblocks`, or `None`), peak detection & localization, MEDiCINe, motion interpolation, Kilosort4 settings.
- `PostprocessingParams` (`postprocessing_pipeline.py`): analyzer extensions, the two auto-merge presets, merge rounds and mode, UnitRefine classifiers, noise cutoff.
- `LFPParams` (`lfp.py`): band, filter order, output sampling rate, channel subsampling, motion correction, bad-channel threshold.

To change one for your setup, override it in your config, e.g. `SORTING_PARAMS = dict(detect_threshold=5.0)`, `POSTPROCESSING_PARAMS = dict(noise_cutoff=0.6)` or `LFP_PARAMS = dict(fs=2500)`. The parameters used are saved with the outputs (`sorting_details.json` and the `settings/` folders).

## Clock alignment

There are three clocks: the probe (imec, used for spike times and LFP), the NIDQ card, and the second system (Blackrock / Plexon). The behavior pipeline maps second-system times (task events, eye) to the probe clock in two steps.

### Step 1: second system → NIDQ (`ALIGNMENT_METHOD`)

> Both alignment methods below (and the filling-in of missing sync pulses) are Gal Vishne's implementation. For questions, contact Gal (gal.vishne@gmail.com).

**`sync_signal` (default).** The same sync pulse train is recorded by the NIDQ (`NIDQ_SYNC_CHANNEL`, an analog channel thresholded at half its max) and by the second system (`SYNC_CHANNEL`):
- Plexon: the `EVT32` event channel, pulses = code 1. verify this matches your rig!
- Blackrock: an analog input in its `.nsX` files, e.g. `'ainp16'`. Not set by default: set it for your rig.

The pulses are paired in order and times are linearly interpolated between them. If the two systems have a different number of pulses, missing pulses (gaps longer than 1.5 × the median interval) are filled in. If the counts still differ (e.g. one system started recording earlier), the run stops with an error. Use this method whenever the rig has a shared sync line: it doesn't depend on the task codes.

**`codes`.** Uses the task codes, recorded both by the second system (`CODES_CHANNEL`) and by the NIDQ (`NIDQ_CODES_CHANNEL`). The codes that occur the same number of times in both are taken as matched points, and times are interpolated between them. This needs no extra wiring. The NIDQ codes are read from one digital word (`raw_io.load_nidq_codes`). A new value counts as a code once it's stable for 2 samples, since the word briefly passes through in-between values while its lines switch. On a 3.2 h test session this gave codes and times identical to CatGT's bit-field extraction.

### Step 2: NIDQ → probe (`ALIGN_NIDQ_TO_IMEC`)

**What SpikeGLX does and doesn't do** (from the [SpikeGLX sync docs](https://billkarsh.github.io/SpikeGLX/help/syncEdges/Sync_edges/)):
- SpikeGLX does **not** modify the saved data to align streams. Each stream (each probe, and the NIDQ) is sampled by its own hardware clock.
- To make alignment possible, it records a sync wave (a square wave, typically 1 s period) in every stream: bit 6 of each probe's SY channel, and an NIDQ channel you wire it to.
- It does use the sync edges to make the files of all streams start at a common wall time. So the streams start together, but then drift apart.
- Even with calibrated sample rates, a realistic difference between two stream clocks is ~0.3 Hz. At 30 kHz that is ~10 ppm, or tens of ms per hour.
- The recommended workflow is offline: CatGT extracts the sync-wave edges from each stream, and TPrime maps event times (e.g. NIDQ) to the reference stream (usually imec0). After TPrime, errors are typically ~0.01 ms.

**What this pipeline does.** This step is on by default (`ALIGN_NIDQ_TO_IMEC = True`). It does what TPrime does:
- It pairs the rising edges of the NIDQ sync channel with those of the probe's SY channel, read from the LF file if there is one (small), else the AP file.
- It interpolates between the paired edges.
- It logs the probe − NIDQ offset at the start and end of each recording, so you can see the drift.

This assumes `NIDQ_SYNC_CHANNEL` records the same SpikeGLX sync wave as the probe. If it doesn't, the step checks that the two edge trains have the same period, that most edges pair up, and that the offset changes smoothly. When a check fails, the behavior stays in the NIDQ clock, with a `WARNING` in the log. `task_data_info.json` records which clock was used (`clock`: `imec` or `nidq`).

## Recording identifiers

A recording (one row of the recording table) is identified by the columns in `RECORDING_KEYS`, by default `monkey, date, region`. To use other identifiers, e.g. `subject, session, probe`, **only the config and the table change, not the code**:

1. Add the columns to the recording table.
2. Set them in your config: `RECORDING_KEYS = ['subject', 'session', 'probe']`.

Then everything follows:
- **Selecting recordings:** the command-line filters are generated from the config, `--subjects`, `--sessions`, `--probes` (`--help` lists them).
- **Naming:** log and output files are named from the key values in that order, e.g. `m1_2_A_trials.csv`.
- **In code:** every function identifies a recording by a `RecordingID` (`pipelines_functions.py`) built from these columns. `rec_id.folder('sorting')` gives its folders, `rec_id.info` its row of the table, and `str(rec_id)` its name. In a notebook, `load_trial_data(dict(subject='m1', session=2, probe='A'))`.

Requirements and notes:
- The keys must identify each row uniquely; the startup check reports duplicates.
- **To select recordings by a column that isn't an identifier** (e.g. a task or an experimenter column), add it to `FILTER_COLUMNS`, e.g. `FILTER_COLUMNS = ['session_num', 'task']` gives `--tasks`.
- Changing `RECORDING_KEYS` changes the names of logs and output files. Existing outputs are still in their folders, but files named after the old keys (e.g. the task data) won't be found under the new names. So choose the keys before processing a dataset.

## Adding a recording system

The "second system" is whatever records the task event codes (and eye signals) next to SpikeGLX. Blackrock and Plexon are built in. Everything system-specific is in the "second system" section at the end of `raw_io.py`, so that's the only file to change.

**If [neo](https://neo.readthedocs.io/en/stable/rawiolist.html) can read your system** (it reads most acquisition systems):

1. **Name it and give its file extension:** add it to `FILE_SUFFIX` in `raw_io.py`, e.g. `FILE_SUFFIX = dict(blackrock='.nev', plexon='.pl2', mysystem='.xyz')`. Then set `SECOND_SYSTEM = 'mysystem'` in your config. The startup check accepts every name in `FILE_SUFFIX`, and the recording table's `trigger_filename` column gets this extension.
2. **Open its files:** in `open_reader`, create the neo reader for it, e.g.
   ```python
   if second_system == 'mysystem':
       reader = neo.rawio.MySystemRawIO(filename=filename)
   ```
   The functions that read events, sync pulses and eye signals only use neo's standard reader interface, so they then work for it as they are.
3. **Set its channels** (`CODES_CHANNEL`, `SYNC_CHANNEL`, `EYE_CHANNELS`) in your config. To list the names, use the snippet in [Adapting to your setup](#adapting-to-your-setup) with your system's neo reader.
4. **Sync pulses:** `load_sync_times` reads them in one of two ways:
   - as events with code 1 on an event channel, like Plexon
   - as the rising edges of an analog channel, like Blackrock (the default for any system)

   If your system records them as events, add it to the Plexon branch: `if second_system in ['plexon', 'mysystem']`. If your rig has no shared sync line, use `ALIGNMENT_METHOD = 'codes'` instead.
5. **Check it on one recording** in a notebook:
   ```python
   import config, raw_io
   from pipelines_functions import RecordingID
   from tasks import get_task
   rec_id = RecordingID.from_dict(dict(monkey='Jones', date='20260702', region='Pul'))
   times, codes = raw_io.load_events('mysystem', rec_id, data_location='local')
   print(len(times), times[:10], codes[:10])                              # event times (s) and codes
   print(raw_io.load_sync_times('mysystem', rec_id, data_location='local')[:10])   # sync pulse times (s)
   get_task(config.TASK).parse_codes(times, codes, testing_mode=True)     # trial table + the codes the parser didn't use
   ```
   Then run `behavior_pipeline.py` on that recording and check the log for the alignment and the number of trials.

**If neo can't read it:** write your own loading functions that return the same things, and call them for your system in `load_events`, `load_sync_times` and `load_eye_data`:
- events: `(times, codes)` - numpy arrays of event times (seconds from the start of the recording) and integer codes
- sync pulses: a numpy array of pulse times (seconds, the same clock as the events)
- eye signals: `dict(times, signals, channels, fs, units)` - see `load_eye_signals` for the format

Everything after loading (clock alignment, trial parsing, eye data per trial) is the same for every system.

## Known limitations / TODO

- **Tested end to end on one session:** SpikeGLX NP1.0 with Blackrock, both alignment methods (`sync_signal` and `codes` agree within 1 ms), eye data (also Plexon eye data, read on one session), LFP from the LF stream (also with `lfp.py` alone), and trial-aligned spikes straight from a Kilosort folder (`'kilosort'`). Only run on Windows.
- **Untested:** the LFP from the AP stream (NP2.0 probes), a full sort with Kilosort's own drift correction (`motion_correction='kilosort'`), and a full postprocessing run on a sorting made elsewhere (`sorting_folder`).
- **`postprocessing` uses the legacy auto-merge API** (`sc.get_potential_auto_merge`), which ties the code to SpikeInterface 0.102.x.
- **Sorting and postprocessing always run on the local folders** (`folder_local`); only the behavior pipeline can also run directly on the server folders (`BEHAVIOR_DATA_LOCATION = 'server'`). Use `SERVER_SYNC` to copy the results to the server.
- Only single-segment recordings are supported (a Blackrock or Plexon file paused mid-recording has several segments; only the first is read).

## References

Methods and documentation this pipeline builds on (cited in the code where they are used):
- **SpikeInterface**: [documentation](https://spikeinterface.readthedocs.io), [installation](https://spikeinterface.readthedocs.io/en/stable/get_started/installation.html), [preprocessing](https://spikeinterface.readthedocs.io/en/latest/modules/preprocessing.html), [LFP example](https://spikeinterface.readthedocs.io/en/latest/how_to/drift_with_lfp.html), [motion correction benchmark](https://www.eneuro.org/content/11/2/ENEURO.0229-23.2023) (Garcia et al. 2024)
- **Kilosort4**: [code & installation](https://github.com/MouseLand/Kilosort), [documentation & parameters](https://kilosort.readthedocs.io/en/latest/)
- **MEDiCINe**: [code](https://github.com/jazlab/medicine), [paper](https://www.eneuro.org/content/12/3/ENEURO.0529-24.2025)
- **UnitRefine** classifiers: [paper](https://www.biorxiv.org/content/10.1101/2025.03.30.645770v1) (Jain et al. 2025), [models](https://huggingface.co/collections/SpikeInterface/curation-models-66ebf53d7f17088991062c2c), [how-to](https://spikeinterface.readthedocs.io/en/latest/how_to/auto_curation_prediction.html)
- **SpikeGLX** synchronization: [Sync: aligning with edges](https://billkarsh.github.io/SpikeGLX/help/syncEdges/Sync_edges/), [user manual](https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md)
- **IBL** LFP preprocessing (`destripe_lfp`): [ibl-neuropixel](https://github.com/int-brain-lab/ibl-neuropixel)
- **neo** (Blackrock / Plexon reading): [documentation](https://neo.readthedocs.io), [Blackrock reader](https://github.com/NeuralEnsemble/python-neo/blob/master/neo/rawio/blackrockrawio.py)

**Citing this pipeline:** a citation is in progress. If you use the pipeline, please contact Gal Vishne (gal.vishne@gmail.com), and also cite the packages above.
