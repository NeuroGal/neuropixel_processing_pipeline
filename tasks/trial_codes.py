"""
# Gal Vishne 2025-2026
# contact me at gal.vishne@gmail.com

General tools for turning a stream of task event codes into a trial table - for any task and any lab (no lab-specific codes here).

The task computer sends a stream of integer codes, each with a time. A task file (see _template.py) usually:
    1. splits the stream into trials, from each trial-start code to the next trial-end code (split_trials / parse_trials)
    2. reads the fields of each trial from its codes, with a TrialCodes object:
        tc.time(code)                         - time of an event code (relative to the trial start; nan if absent)
        tc.present(code)                      - whether a code is in the trial (e.g. correct / error)
        tc.value_after(code, offset, scale)   - a value sent as the next code, decoded as (next_code - offset) / scale
    3. gives categories readable labels (label_column) and checks no codes were missed (report_unused_codes)
"""
import numpy as np
import pandas as pd


class TrialCodes:
    """
    The codes of a single trial, with helpers to read event times and values.
    Times are relative to the first code of the trial (seconds). Every code that is read is added to used_codes
    (so report_unused_codes can list the codes no field used).
    """
    def __init__(self, codes, times, trial_num, used_codes):
        self.codes = codes
        self.times = times - times[0]
        self.trial_num = trial_num
        self.used_codes = used_codes

    def index(self, code, allow_multiple=False):
        """
        Index of the code in the trial (None if absent).
        allow_multiple: take the first one if the code appears more than once (otherwise this raises an error)
        """
        self.used_codes.add(code)
        idx = np.where(self.codes == code)[0]
        if len(idx) == 0:
            return None
        if len(idx) > 1 and not allow_multiple:
            raise ValueError(f"Multiple instances of code {code} found in trial {self.trial_num + 1} (at {idx}, trial codes: {self.codes}).")
        return idx[0]

    def time(self, code, allow_multiple=False):
        """Time of an event code, relative to the trial start (nan if absent)."""
        idx = self.index(code, allow_multiple=allow_multiple)
        return np.nan if idx is None else self.times[idx]

    def value_after(self, code, offset=0, scale=1, allow_multiple=False):
        """Value sent after a code: (next_code - offset) / scale (nan if the code is absent)."""
        idx = self.index(code, allow_multiple=allow_multiple)
        if idx is None:
            return np.nan
        if idx == len(self.codes) - 1:
            raise ValueError(f"Code {code} is the last code in trial {self.trial_num + 1}, but it should have a following code indicating the value.")
        value = self.codes[idx + 1]
        self.used_codes.add(value)
        return (value - offset) / scale

    def present(self, code):
        """Whether a code appears in the trial."""
        self.used_codes.add(code)
        return code in self.codes


def split_trials(event_codes, start_code, end_code):
    """
    Start & end index of each trial: from each start code to the next end code (or the last code, for the last trial).
    """
    start_indices = np.where(event_codes == start_code)[0]
    end_candidates = np.where(event_codes == end_code)[0]
    end_indices = []
    for start in start_indices:
        next_end = end_candidates[end_candidates > start]
        end_indices.append(next_end[0] if len(next_end) else len(event_codes) - 1)
    return list(zip(start_indices, end_indices))


def parse_trials(event_times, event_codes, parse_trial, start_code, end_code):
    """
    Build the trial table: parse_trial(TrialCodes) -> dict of fields for that trial.
    Adds trial_num (first column) and the absolute t_start / t_end (last columns) - the pipeline needs these.
    Returns the table and the set of codes that were used.
    """
    used_codes = {start_code, end_code}
    trials = []
    for trial_num, (start_idx, end_idx) in enumerate(split_trials(event_codes, start_code, end_code)):
        tc = TrialCodes(event_codes[start_idx:end_idx + 1], event_times[start_idx:end_idx + 1], trial_num, used_codes)
        trials.append({'trial_num': trial_num, **parse_trial(tc),
                       't_start': event_times[start_idx], 't_end': event_times[end_idx]})  # absolute times - used to align with the spike data
    return pd.DataFrame(trials), used_codes


def label_column(trials, column, labels):
    """Replace the values of a column by readable labels, e.g. label_column(trials, 'motion_dir', {0: 'right', 180: 'left'})."""
    trials[column] = trials[column].map(labels).astype('category')


def report_unused_codes(event_codes, used_codes, ignore=(), ignore_above=None):
    """
    Print the codes in the recording that no trial field used - to check a parser covers everything.
    ignore: codes that are expected but not part of any trial field; ignore_above: ignore all codes above this number.
    """
    unused_codes = sorted(set(np.unique(event_codes)) - set(used_codes) - set(ignore))
    if ignore_above is not None:
        unused_codes = [code for code in unused_codes if code <= ignore_above]
    if unused_codes:
        counts = sorted(((code, np.sum(event_codes == code)) for code in unused_codes), key=lambda x: x[1], reverse=True)
        print("%d unused event codes detected:" % len(counts))
        for code, count in counts:
            print(f"Code {code} appears {count} times")
    else:
        print("All event codes were accounted for.")
    return unused_codes
