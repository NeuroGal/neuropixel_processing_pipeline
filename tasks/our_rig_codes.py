"""
# Gal Vishne 2025-2026
# contact me at gal.vishne@gmail.com

The code conventions of OUR task computer, shared by our tasks (dots_2d.py, vd_pulse.py):
which codes start / end a trial, how values are encoded, and the codes for the display settings and the trial outcome.

Another lab: these numbers are specific to our task software. Put your own conventions in your task file, or - if several of
your tasks share them - in a file like this one (e.g. tasks/<lab>_rig_codes.py) that your task files import. See _template.py.

Comments marked "CHECK:" are not yet confirmed against the task software (a meaning, unit, offset or scale we inferred from the data).
"""
import numpy as np
from tasks.trial_codes import parse_trials, report_unused_codes

# %% trials and code checking

TRIAL_START_CODE = 1005
TRIAL_END_CODE = 4925

# for testing mode: codes that are expected but not part of any trial field
SYNC_SEQUENCE_CODES = [0, 100, 200, 500, 2000]  # 0-100-200-500-2000-0 sequence used for synching with the start of the neuropixel recordings (we use others)
IGNORED_CODES = [19, 1003, 8000, 8080, 9092, 9093]
IGNORE_CODES_ABOVE = 16000   # codes above 16000 are not task events
# CHECK: the meaning of the ignored codes (they don't affect the trial table, only the list of unread codes in testing mode):
#   19 - comes in every 2D (or dots) trial
#   1003 - possibly the task change
#   8000 - starts a series of codes
#   8080 - possibly the max dots duration in the RT dots task
#   9092, 9093 - almost always right after the trial end (4925), then almost always followed by 8000

def parse_rig_trials(event_times, event_codes, parse_trial, testing_mode=False):
    """
    The trial table of a recording from our rig: trials from TRIAL_START_CODE to TRIAL_END_CODE, each read with parse_trial(TrialCodes).
    testing_mode: also print the codes that no field used (ignoring the ones above, which are expected but not trial fields).
    """
    trials, used_codes = parse_trials(event_times, event_codes, parse_trial, TRIAL_START_CODE, TRIAL_END_CODE)
    if testing_mode:
        report_unused_codes(event_codes, used_codes, ignore=SYNC_SEQUENCE_CODES + IGNORED_CODES, ignore_above=IGNORE_CODES_ABOVE)
    return trials

# %% display settings & outcome - read by our task files
# Positions, sizes and acceptance windows are sent in visual degrees * 10 (divided by 10 here); colors as 0-255
POS_OFFSET, POS_SCALE = 7500, 10
SIZE_OFFSET, SIZE_SCALE = 7000, POS_SCALE
RGB_OFFSET, RGB_SCALE = 7000, 1

def pos_pair(tc, code_x, code_y):
    """Screen position (visual degrees); positions that are effectively off the screen (>= 49) are nan (relevant for the 1D tasks)."""
    x = tc.value_after(code_x, offset=POS_OFFSET, scale=POS_SCALE)
    y = tc.value_after(code_y, offset=POS_OFFSET, scale=POS_SCALE)
    return (x if x < 49 else np.nan), (y if y < 49 else np.nan)

def target_positions(tc, pos_codes):
    """targN_pos_x / targN_pos_y for each (code_x, code_y) pair."""
    trial = {}
    for tidx, (code_x, code_y) in enumerate(pos_codes, start=1):
        trial[f'targ{tidx}_pos_x'], trial[f'targ{tidx}_pos_y'] = pos_pair(tc, code_x, code_y)
    return trial

def display_settings(tc, target_size_codes, target_rgb_codes):
    """
    Settings that should be constant / not important for analyses (saved with extract_important_only=False).
    target_size_codes: one code per target; target_rgb_codes: one (r, g, b) code triplet per target.
    """
    trial = {}
    trial['fix_pos_x'], trial['fix_pos_y'] = pos_pair(tc, 8001, 8002)
    trial['dots_pos_x'], trial['dots_pos_y'] = pos_pair(tc, 8003, 8004)  # location of the dot stimulus

    size = lambda code, **kw: tc.value_after(code, offset=SIZE_OFFSET, scale=SIZE_SCALE, **kw)
    for tidx, code in enumerate(target_size_codes, start=1):
        trial[f'targ{tidx}_size'] = size(code)
    trial['fix_size'] = size(8076)
    trial['dots_diam'] = size(8005)   # aperture diameter for the dots
    trial['dots_size'] = size(8105)   # CHECK: the size of each dot (visual degrees)?
    trial['dots_speed'] = tc.value_after(8006, offset=7000, scale=10)  # CHECK: visual degrees per second (offset & scale inferred)

    # acceptance windows: how close the eye must be for the fixation point / a target to count as acquired
    trial['targ_win_x'], trial['targ_win_y'] = size(8068), size(8069)
    trial['fix_win_x'], trial['fix_win_y'] = size(8066, allow_multiple=True), size(8067, allow_multiple=True)

    rgb = lambda code: tc.value_after(code, offset=RGB_OFFSET, scale=RGB_SCALE)
    for tidx, (r, g, b) in enumerate(target_rgb_codes, start=1):
        trial[f'targ{tidx}_R'], trial[f'targ{tidx}_G'], trial[f'targ{tidx}_B'] = rgb(r), rgb(g), rgb(b)
    trial['fix_R'], trial['fix_G'], trial['fix_B'] = rgb(8133), rgb(8134), rgb(8135)

    # CHECK: milliseconds of valve opening (offset & scale inferred); only sent in some trials. Half reward was never used.
    trial['reward_size'] = tc.value_after(8060, offset=7000)

    # random seeds of the dot generation (could regenerate the dots; they are also saved by the task)
    # CHECK: the offset - the values come out as 7000, 7001, ..., so it is probably 7000 (not subtracted here yet)
    trial['seed_base'] = tc.value_after(8027)
    trial['seed_var'] = tc.value_after(8028)

    trial['monitor_dist'] = tc.value_after(8065, offset=7000)  # CHECK: cm (54 in our data)

    # the task's own outcome code, redundant with the outcome columns (see outcome below):
    #   1 correct, 0 error, -1 no choice, -2 fixation break, -10 fixation never acquired
    #   CHECK: 2 (saccade) and -3 (previous fail) - listed by the task, never seen in our data
    trial['response_code'] = tc.value_after(8091, offset=7500)

    # codes listed by the task software but not seen in our data (not read): targN_lum 8025, 8026, 8059, 8114; dot_dur 8018;
    # seed_range 8014; t_reward_on 1030, t_reward_off 4908; half_reward_size 8150
    return trial

def outcome(tc):
    """Behavioral outcome codes shared by the tasks."""
    trial = {}
    trial['correct'] = tc.present(4905)    # all tasks (not just the dots tasks)
    trial['error'] = tc.present(4906)      # a response was made to the wrong target
    trial['no_choice'] = tc.present(4907)  # no target was chosen (response_code -1)
    # the three are mutually exclusive; trials with none of them are fixation breaks or fixation never acquired (response_code -2 / -10)
    return trial
