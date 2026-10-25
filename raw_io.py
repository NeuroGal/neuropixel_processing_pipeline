# Gal Vishne 2025-2026
# contact me at gal.vishne@gmail.com
"""
Reading raw files: SpikeGLX (single channels: sync pulses, task codes; probe selection) and the "second system" that recorded
the task event codes (and possibly eye data) next to the neuropixels - Blackrock (.nev) or Plexon (.pl2).

Memory: SpikeGLX channels are read in chunks from a memory map, Plexon analog signals (eye) in chunks through neo.
Blackrock / Plexon event files only hold the events (small), so they are read at once.

About clocks: every SpikeGLX stream (each probe, NIDQ) has its own clock, and the second system has its own. Times here are in the
clock of the file they come from (sample / sampling rate, from the start of the file) - behavior_pipeline.py maps them to the probe clock.

To support another second system, add a load_<system>_events function (and optionally sync / eye loaders) and register it in the dicts at the bottom.
"""
import re
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly
import neo
from pathlib import Path
import config
from pipelines_functions import log_or_print, find_file

# %% ================ SpikeGLX ================

# %% files

def read_meta(path):
    """SpikeGLX .meta file -> dict (path can be the .bin or the .meta). Keys starting with ~ keep the ~."""
    meta = {}
    with open(Path(path).with_suffix('.meta')) as f:
        for line in f:
            if '=' in line:
                key, value = line.rstrip('\n').split('=', 1)
                meta[key] = value
    return meta

def find_stream_file(raw_path, stream):
    """
    The .bin file of a stream in the raw data folder or its subfolders (SpikeGLX keeps each probe's files in a subfolder),
    e.g. stream='nidq', 'imec0.ap', 'imec0.lf' (None if it does not exist).
    """
    files = sorted(Path(raw_path).rglob(f'*.{stream}.bin'))
    if len(files) > 1:
        raise ValueError(f"Multiple {stream} files in {raw_path}: {[str(f.relative_to(raw_path)) for f in files]}. "
                         "Keep one recording (one SpikeGLX run / gate / trigger) per raw data folder.")
    return files[0] if files else None

def open_bin(bin_path):
    """Memory map of a SpikeGLX .bin file: (samples x channels int16 array, sampling rate)."""
    meta = read_meta(bin_path)
    n_chans = int(meta['nSavedChans'])
    fs = float(meta['imSampRate'] if meta.get('typeThis') == 'imec' else meta['niSampRate'])
    data = np.memmap(bin_path, dtype=np.int16, mode='r')
    return data.reshape(-1, n_chans), fs

def iter_channel(bin_path, channel, chunk_duration_s=10):
    """
    Yield (first_sample, values) chunks of a single channel. Reads the file chunk by chunk (not memory-mapped), so memory stays at one
    chunk even for long recordings: a memory map of a whole multi-GB file keeps every page it touched in the process's memory.
    """
    meta = read_meta(bin_path)
    n_chans = int(meta['nSavedChans'])
    fs = float(meta['imSampRate'] if meta.get('typeThis') == 'imec' else meta['niSampRate'])
    n_samples = Path(bin_path).stat().st_size // (2 * n_chans)
    channel = channel % n_chans
    chunk = int(chunk_duration_s * fs)
    with open(bin_path, 'rb') as f:
        for start in range(0, n_samples, chunk):
            block = np.fromfile(f, dtype=np.int16, count=min(chunk, n_samples - start) * n_chans).reshape(-1, n_chans)
            yield start, block[:, channel].copy()

def get_imec_stream_id(rec_id, raw_path):
    """
    Which probe (imecN) to load from the raw_data folder.
    Taken from the neuropixel_filename column of the recording table if given (e.g. 'run_g0_t0.imec1.ap.bin' -> 'imec1.ap'),
    otherwise from the files in the folder (which must then contain a single probe).
    """
    neuropixel_filename = rec_id.info.get('neuropixel_filename')
    if isinstance(neuropixel_filename, str) and re.search(r'imec(\d+)', neuropixel_filename):
        imec_id = int(re.search(r'imec(\d+)', neuropixel_filename).group(1))
    else:
        imec_ids = sorted({int(m.group(1)) for f in Path(raw_path).rglob('*') if f.is_file()
                           for m in [re.search(r'imec(\d+)\.(ap|lf)', f.name)] if m})
        if len(imec_ids) == 0:
            raise ValueError(f"No imec files found in raw data folder {raw_path}. Please check the folder and file names.")
        if len(imec_ids) > 1:
            raise ValueError(f"Multiple probes ({['imec%d' % i for i in imec_ids]}) found in {raw_path}. "
                             "Add a neuropixel_filename column to the recording table to specify which one belongs to this recording.")
        imec_id = imec_ids[0]
    return f"imec{imec_id}.ap"

# %% edges / changes in a channel

def rising_edges(bin_path, channel, threshold=None, bit=None):
    """
    Times (s) where a channel crosses threshold upwards (analog), or where a bit of it turns on (digital, bit=...).
    threshold=None -> half of the channel's maximum (computed in a first pass).
    """
    _, fs = open_bin(bin_path)
    if bit is None and threshold is None:
        threshold = 0.5 * max(values.max() for _, values in iter_channel(bin_path, channel))
    edges, previous = [], None
    for start, values in iter_channel(bin_path, channel):
        high = ((values.astype(np.int32) >> bit) & 1).astype(bool) if bit is not None else values >= threshold
        if previous is not None:
            high = np.concatenate(([previous], high))
            edges.append(np.where(~high[:-1] & high[1:])[0] + start)
        else:
            edges.append(np.where(~high[:-1] & high[1:])[0] + 1 + start)
        previous = high[-1]
    return np.concatenate(edges) / fs

def value_changes(bin_path, channel):
    """Times (s) and new values wherever a (digital word) channel changes value."""
    _, fs = open_bin(bin_path)
    indices, values_after, previous = [], [], None
    for start, values in iter_channel(bin_path, channel):
        if previous is not None:
            values_ext = np.concatenate(([previous], values))
            changes = np.where(np.diff(values_ext) != 0)[0]          # index in values_ext of the sample before the change
            indices.append(changes + start)                          # = sample index of the change in the full signal
            values_after.append(values_ext[changes + 1])
        else:
            changes = np.where(np.diff(values) != 0)[0] + 1
            indices.append(changes + start)
            values_after.append(values[changes])
        previous = values[-1]
    return np.concatenate(indices) / fs, np.concatenate(values_after)

# %% NIDQ

def load_nidq_sync_times(raw_path, channel):
    """
    Sync pulse times (s, NIDQ clock) from an analog NIDQ channel (NIDQ_SYNC_CHANNEL in the config), thresholded at half its max.
    """
    nidq_file = find_stream_file(raw_path, 'nidq')
    if nidq_file is None:
        raise ValueError(f"No NIDQ file (*.nidq.bin) in {raw_path}.")
    return rising_edges(nidq_file, channel)

def load_nidq_codes(raw_path, channel, min_stable_samples=2):
    """
    Task codes from a digital word of the NIDQ (NIDQ_CODES_CHANNEL in the config; the digital words are the last saved channels).
    A code is a new value of the word that stays stable for at least min_stable_samples samples: the digital lines don't all switch
    on the same sample, so the word briefly passes through in-between values (1 sample) while a code changes - those are dropped.
    Verified on a 3.2 h session: identical codes and times to CatGT's bit-field extraction (-bf, BFT / BFV files).
    Returns times (s, NIDQ clock), codes.
    """
    nidq_file = find_stream_file(raw_path, 'nidq')
    if nidq_file is None:
        raise ValueError(f"No NIDQ file (*.nidq.bin) in {raw_path}.")
    times, values = value_changes(nidq_file, channel)
    data, fs = open_bin(nidq_file)
    samples = np.round(times * fs).astype(np.int64)
    durations = np.diff(np.append(samples, data.shape[0]))       # how long each value lasted (samples)
    stable = durations >= min_stable_samples
    times, values = times[stable], values[stable].astype(np.int64) & 0xFFFF   # the word as an unsigned 16 bit number
    new = np.concatenate(([True], values[1:] != values[:-1]))      # removing transients can leave the same code twice in a row
    return times[new], values[new]

# %% probe sync wave (for NIDQ -> probe clock alignment)

def load_imec_sync_times(raw_path, imec_stream_id, bit):
    """
    Rising edges of the SpikeGLX sync wave in the SY channel (last saved channel) of a probe (s, probe clock).
    Uses the LF file if there is one (12x smaller than AP, 0.4 ms resolution), otherwise the AP file.
    """
    imec = imec_stream_id.split('.')[0]
    bin_path = find_stream_file(raw_path, f'{imec}.lf') or find_stream_file(raw_path, f'{imec}.ap')
    if bin_path is None:
        raise ValueError(f"No {imec} files in {raw_path}.")
    meta = read_meta(bin_path)
    if not meta.get('snsSaveChanSubset', 'all') == 'all':
        log_or_print(f"Note: not all channels were saved in {bin_path.name} - assuming the last saved channel is still SY.")
    return rising_edges(bin_path, channel=-1, bit=bit)

# %% ================ second system (Blackrock / Plexon) ================
# All times are in seconds from the start of the recording, in the second system's clock.
# Everything (events, sync pulses, eye) goes through the same clock mapping in behavior_pipeline.py.

FILE_SUFFIX = dict(blackrock='.nev', plexon='.pl2')

def get_trigger_file(rec_id, second_system, data_location):
    """
    Path of the second system's file: trigger_filename column of the recording table (extension added per system), in the raw_data
    folder - or in the trigger_folder column's folder, if filled (for second-system files kept apart from the SpikeGLX run folder).
    """
    filename = rec_id.info.get('trigger_filename')
    if not isinstance(filename, str):
        raise ValueError("No trigger_filename in the recording table for this recording.")
    folder = rec_id.info.get('trigger_folder')
    folder = Path(folder) if isinstance(folder, str) and folder.strip() else rec_id.folder('raw_data', location=data_location, mkdir=False)
    return find_file(folder, Path(filename).with_suffix(FILE_SUFFIX[second_system]).name)

def open_reader(second_system, rec_id, data_location='server'):
    """neo reader of the second system's file (header parsed; signals are memory-mapped, not loaded)."""
    filename = get_trigger_file(rec_id, second_system, data_location)
    reader = neo.rawio.BlackrockRawIO(filename=filename) if second_system == 'blackrock' else neo.rawio.Plexon2RawIO(filename=filename)
    reader.parse_header()
    return reader

def load_channel_events(reader, channel):
    """
    Event times & codes from one event channel, e.g. Plexon 'Strobed' (task codes) / 'EVT32' (sync), Blackrock 'digital_input_port' (task codes).
    Times are relative to the start of the recording.
    """
    event_channels = reader.header['event_channels']
    matches = np.where([ch['name'] == channel for ch in event_channels])[0]
    if len(matches) == 0:
        raise ValueError(f"No event channel {channel} - available: {[ch['name'] for ch in event_channels]}.")
    event_times, _, event_codes = reader.get_event_timestamps(event_channel_index=matches[0])
    event_codes = event_codes.astype(int)
    event_times = reader.rescale_event_timestamp(event_times, dtype='float64')
    return event_times - reader.segment_t_start(block_index=0, seg_index=0), event_codes

def find_analog_channels(reader, channel_names):
    """(stream_index, channel indexes within the stream) of analog channels, which must all be in one stream (same sampling rate)."""
    signal_channels = reader.header['signal_channels']
    for stream_index, stream in enumerate(reader.header['signal_streams']):
        stream_channel_names = list(signal_channels[signal_channels['stream_id'] == stream['id']]['name'])
        if all(ch in stream_channel_names for ch in channel_names):
            return stream_index, [stream_channel_names.index(ch) for ch in channel_names]
    raise ValueError(f"Analog channels {channel_names} not found together in one stream - available: {list(signal_channels['name'])}.")

def read_analog(reader, channel_names, i_start, i_stop, scale=True):
    """Samples i_start:i_stop of analog channels (samples x channels). scale: convert to physical units (else raw integers)."""
    stream_index, channel_indexes = find_analog_channels(reader, channel_names)
    raw = reader.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=i_start, i_stop=i_stop,
                                        stream_index=stream_index, channel_indexes=channel_indexes)
    return reader.rescale_signal_raw_to_float(raw, dtype='float32', stream_index=stream_index, channel_indexes=channel_indexes) if scale else raw

def iter_analog_chunks(reader, channel_names, chunk_duration_s=60, scale=True):
    """Yield (first_sample, values (samples x channels)) chunks of analog channels - the whole signal is never loaded at once."""
    info = analog_info(reader, channel_names)
    chunk = int(chunk_duration_s * info['fs'])
    for start in range(0, info['n_samples'], chunk):
        yield start, read_analog(reader, channel_names, start, min(start + chunk, info['n_samples']), scale=scale)

def analog_info(reader, channel_names):
    """Sampling rate, number of samples, start time (relative to the recording start) and units of analog channels."""
    stream_index, channel_indexes = find_analog_channels(reader, channel_names)
    signal_channels = reader.header['signal_channels']
    stream_channels = signal_channels[signal_channels['stream_id'] == reader.header['signal_streams'][stream_index]['id']]
    t_start = reader.get_signal_t_start(block_index=0, seg_index=0, stream_index=stream_index) - reader.segment_t_start(block_index=0, seg_index=0)
    return dict(fs=reader.get_signal_sampling_rate(stream_index=stream_index),
                n_samples=reader.get_signal_size(block_index=0, seg_index=0, stream_index=stream_index),
                t_start=t_start, units=np.array(stream_channels['units'][channel_indexes], dtype=str))

def load_analog_rising_edges(reader, channel_name):
    """Times where an analog channel (e.g. a sync pulse input) crosses half its maximum upwards - read in chunks, two passes."""
    info = analog_info(reader, [channel_name])
    threshold = 0.5 * max(values.max() for _, values in iter_analog_chunks(reader, [channel_name], scale=False))
    edges, previous = [], None
    for start, values in iter_analog_chunks(reader, [channel_name], scale=False):
        high = values[:, 0] >= threshold
        if previous is None:
            edges.append(np.where(~high[:-1] & high[1:])[0] + 1 + start)
        else:
            high = np.concatenate(([previous], high))
            edges.append(np.where(~high[:-1] & high[1:])[0] + start)
        previous = high[-1]
    return info['t_start'] + np.concatenate(edges) / info['fs']

def load_eye_signals(reader, eye_channels, sampling_rate=1000, chunk_duration_s=60, logger=None):
    """
    Eye signals from analog channels, scaled to physical units and downsampled to sampling_rate (Hz) - if they were recorded at
    sampling_rate or less, they are kept at their original rate.
    Downsampling is standard anti-aliased resampling (scipy.signal.resample_poly: low-pass FIR filter + resampling, also for non-integer ratios).
    It is done in chunks, each read with 1 s of extra signal on both sides that is cut off after filtering, so the result is the same as
    resampling the whole signal at once, without loading it all into memory.
    eye_channels: channel names, x, y, pupil (e.g. Plexon ['AI01', 'AI02', 'AI03'], Blackrock ['ainp1', 'ainp2', 'ainp3'] - verify for your rig).
    Returns dict(times (s, second-system clock), signals (samples x channels, float32), channels, fs, units)
    """
    info = analog_info(reader, eye_channels)
    fs, n_samples = info['fs'], info['n_samples']
    if fs <= sampling_rate:
        up = down = 1
    else:
        ratio = Fraction(sampling_rate / fs).limit_denominator(1000)
        up, down = ratio.numerator, ratio.denominator
    fs_out = fs * up / down
    n_samples = n_samples // down * down           # whole number of output samples
    # extra signal read on each side of a chunk: >= 1 s and >= the resampling filter's half-length (10 * max(up, down) samples in
    # resample_poly), rounded to a multiple of down so chunks stay aligned to output samples; none needed without resampling
    margin = 0 if up == down else int(np.ceil(max(fs, 10 * max(up, down)) / down)) * down
    chunk = max(down, int(chunk_duration_s * fs) // down * down)
    signals = np.empty((n_samples * up // down, len(eye_channels)), dtype=np.float32)
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        read_start, read_stop = max(0, start - margin), min(n_samples, stop + margin)
        values = read_analog(reader, eye_channels, read_start, read_stop)
        if up != down:
            values = resample_poly(values, up, down, axis=0, padtype='line')  # 'line': fewer edge artifacts at the start / end of the recording
        first = (start - read_start) * up // down
        signals[start * up // down:stop * up // down] = values[first:first + (stop - start) * up // down]
    times = info['t_start'] + np.arange(len(signals)) / fs_out
    log_or_print(f"Loaded eye signals {eye_channels}: {len(signals)} samples at {fs_out:g} Hz (recorded at {fs:g} Hz).", logger=logger)
    return dict(times=times, signals=signals, channels=np.array(eye_channels), fs=fs_out, units=info['units'])

# %% loading with the config's channels (CODES_CHANNEL, SYNC_CHANNEL, EYE_CHANNELS)
# To add a system: FILE_SUFFIX + open_reader above, and the sync branch below - step by step in the README, "Adding a recording system"

def load_events(second_system, rec_id, data_location='server'):
    """Task event times & codes of the second system."""
    reader = open_reader(second_system, rec_id, data_location)
    return load_channel_events(reader, channel=config.CODES_CHANNEL)

def load_sync_times(second_system, rec_id, data_location='server'):
    """
    Sync pulse times of the second system.
    Plexon: events on the SYNC_CHANNEL event channel (e.g. 'EVT32') with code 1 (assuming code 1 corresponds to sync pulses, adjust if needed).
    Blackrock: rising edges of the SYNC_CHANNEL analog input (e.g. 'ainp16', in one of the .nsX files).
    """
    reader = open_reader(second_system, rec_id, data_location)
    if second_system == 'plexon':
        event_times, event_codes = load_channel_events(reader, channel=config.SYNC_CHANNEL)
        return event_times[event_codes == 1]
    return load_analog_rising_edges(reader, config.SYNC_CHANNEL)

def load_eye_data(second_system, rec_id, data_location='server', logger=None):
    """Eye signals of the second system (EYE_CHANNELS in the config; None if not set)."""
    eye_channels = config.EYE_CHANNELS
    if not eye_channels:
        return None
    reader = open_reader(second_system, rec_id, data_location)
    return load_eye_signals(reader, eye_channels, sampling_rate=config.EYE_SAMPLING_RATE, logger=logger)
