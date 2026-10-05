"""
TEMPLATE FOR A NEW TASK - copy this file to tasks/<your_task>.py, fill it in, and set TASK = '<your_task>' in your config.
(Files starting with _ are not tasks, so this one is never used as is. There is nothing to register - the file name is the task name.)

A task file turns the event codes of one recording into a trial table. The pipeline only needs two functions from it:

    parse_codes(event_times, event_codes) -> pandas DataFrame, one row per trial
        event_times: numpy array of the event times (seconds), already in the neuropixel clock
        event_codes: numpy array of the integer codes, same length
        The table must have the columns
            t_start, t_end - absolute times (s) of the trial start / end: spikes, eye and LFP are cut from t_start - T_BEFORE_TRIAL
                             to t_end + T_AFTER_TRIAL (config), and their times are saved relative to t_start
            trial_num      - the row number (0, 1, 2, ...) - added by parse_trials
        Everything else is up to you: conditions, choices, event times (relative to t_start, like the spikes), ...

    keep_trial(trials) -> boolean pandas Series: the trials to analyze (saved as the keep_trial column; TrialData.good() selects them)

The tools in trial_codes.py do the usual work (see below). If your trials are marked differently (e.g. no end code), you can build the
table any other way - only the two functions and the t_start / t_end columns matter.

To develop and check your parser on a recording, in a notebook:
    import config, raw_io
    from pipelines_functions import RecordingID
    from tasks import get_task
    rec_id = RecordingID.from_dict(dict(monkey='Jones', date='20260702', region='Pul'))   # your RECORDING_KEYS
    times, codes = raw_io.load_events(config.SECOND_SYSTEM, rec_id, data_location='local')
    trials = get_task('<your_task>').parse_codes(times, codes, testing_mode=True)        # + lists the codes your parser didn't read
    trials.head()
(These times are still in the second system's clock - fine for checking the parsing; the pipeline aligns them before parsing.)
"""
import numpy as np
from tasks.trial_codes import parse_trials, label_column, report_unused_codes

# %% 1. the codes of your task - keep all the numbers here, so the parsing below reads like a description of the task
# (example values - replace them with your task computer's codes)

# how trials are marked in the code stream: each trial runs from a start code to the next end code
TRIAL_START_CODE = 1000
TRIAL_END_CODE = 1999

# event codes: we keep their times
FIXATION_ON = 1010
STIMULUS_ON = 1020
RESPONSE = 1030
REWARD = 1040

# outcome codes: we keep whether they happened
CORRECT = 1100
ERROR = 1101

# value codes: the value is sent as the next code, often with an offset (and a scale) so it can't be confused with other codes
CONDITION = 2000                                   # e.g. 2000, then 5003 -> condition 3
COHERENCE = 2001                                   # e.g. 2001, then 5250 -> coherence 0.25
VALUE_OFFSET = 5000
COHERENCE_SCALE = 1000

# codes in the recording that are not part of any trial field (so testing mode doesn't list them)
IGNORED_CODES = []

# readable labels for categorical values
CONDITION_LABELS = {1: 'easy', 2: 'medium', 3: 'hard'}

# %% 2. one trial -> its fields

def parse_trial(tc):
    """
    The fields of one trial (a dict: column name -> value). tc is a TrialCodes object with this trial's codes (see trial_codes.py):
        tc.time(code)                          time of an event code, relative to the trial start (nan if the code is absent)
        tc.present(code)                       whether the code is in this trial
        tc.value_after(code, offset, scale)    the value sent after the code: (next_code - offset) / scale (nan if absent)
    """
    trial = {}
    # trial settings
    trial['condition'] = tc.value_after(CONDITION, offset=VALUE_OFFSET)
    trial['coherence'] = tc.value_after(COHERENCE, offset=VALUE_OFFSET, scale=COHERENCE_SCALE)
    # event times (relative to the trial start)
    trial['t_fixation_on'] = tc.time(FIXATION_ON)
    trial['t_stimulus_on'] = tc.time(STIMULUS_ON)
    trial['t_response'] = tc.time(RESPONSE)
    trial['t_reward'] = tc.time(REWARD)
    # outcome
    trial['correct'] = tc.present(CORRECT)
    trial['error'] = tc.present(ERROR)
    # derived fields
    trial['RT'] = trial['t_response'] - trial['t_stimulus_on']   # nan if either is missing
    return trial

# %% 3. the functions the pipeline calls

def parse_codes(event_times, event_codes, testing_mode=False):
    """
    Event codes -> trial table (one row per trial; trial_num, t_start, t_end are added by parse_trials).
    testing_mode: also print the codes that no field read - to check the parser covers everything.
    """
    trials, used_codes = parse_trials(event_times, event_codes, parse_trial, TRIAL_START_CODE, TRIAL_END_CODE)
    label_column(trials, 'condition', CONDITION_LABELS)
    if testing_mode:
        report_unused_codes(event_codes, used_codes, ignore=IGNORED_CODES)
    return trials

def keep_trial(trials):
    """The trials to analyze - here: trials with a response (correct or error), i.e. not aborted."""
    return trials['correct'] | trials['error']


# %% try it on made-up codes: python -m tasks._template
if __name__ == '__main__':
    codes = np.array([TRIAL_START_CODE, CONDITION, 5003, COHERENCE, 5250, FIXATION_ON, STIMULUS_ON, RESPONSE, CORRECT, REWARD, TRIAL_END_CODE,
                      TRIAL_START_CODE, CONDITION, 5001, COHERENCE, 5500, FIXATION_ON, TRIAL_END_CODE])   # 2nd trial: aborted after fixation
    times = np.arange(len(codes)) * 0.3
    trials = parse_codes(times, codes, testing_mode=True)
    trials['keep_trial'] = keep_trial(trials)
    print(trials.to_string())
