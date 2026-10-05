# Gal Vishne 2025-2026
# contact me at gal.vishne@gmail.com
"""
Task parser: 2D dots task (motion + color), with its 1D motion / 1D color / overlap / memory variants (Gal's task).
"""
import numpy as np
from tasks.trial_codes import label_column
from tasks.our_rig_codes import parse_rig_trials, target_positions, display_settings, outcome

TASK_TYPE_LABELS = {10: 'overlap', 11: 'mem', 110: '2D', 113: '1D_motion', 116: '1D_color'}
MOTION_DIR_LABELS = {180: 'left', 0: 'right'}
COLOR_DIR_LABELS = {1: 'blue', 2: 'yellow'}   # maybe change to 'blue (bottom)' and 'yellow (top)' or something like that
TARGET_LABELS = {1: 'bottom-right (R,B)', 2: 'bottom-left (L,B)', 3: 'top-right (R,Y)', 4: 'top-left (L,Y)'}  # maybe switch to say the content (L\R, B\Y) before the location
ACCURACY_DETAILED_LABELS = {3: 'both correct', 4: 'only color correct', 5: 'only motion correct', 6: 'both incorrect'}


def parse_trial(tc, extract_important_only=True):
    trial = {}
    # trial settings
    trial['task_type'] = tc.value_after(8017, offset=7000)
    trial['motion_coh'] = tc.value_after(8011, offset=7000, scale=1000)
    trial['motion_dir'] = tc.value_after(8010, offset=7500)
    trial['color_coh'] = tc.value_after(8101, offset=7000, scale=1000)
    trial['color_dir'] = tc.value_after(8102, offset=7500)

    # fix motion & color for 1D tasks
    if trial['task_type'] == 113:  # 1D motion task
        trial['color_coh'] = np.nan  # the dots are white
        trial['color_dir'] = np.nan
    elif trial['task_type'] == 116:  # 1D color task
        trial['motion_coh'] = np.nan  # technically it is 0 coherence, but this makes more sense
        trial['motion_dir'] = np.nan

    trial['signed_motion_coh'] = trial['motion_coh'] if trial['motion_dir'] == 0 else -trial['motion_coh'] if trial['motion_dir'] == 180 else np.nan
    trial['signed_color_coh'] = trial['color_coh'] if trial['color_dir'] == 1 else -trial['color_coh'] if trial['color_dir'] == 2 else np.nan  # positive - blue=down (that's how they coded it in the past)
    trial['targ_correct_loc'] = tc.value_after(8034, offset=7500)  # this is the target that should be chosen, not the one that was chosen (which is in a different code)

    # event times (relative to trial start)
    trial['t_fix_on'] = tc.time(1010)    # should be present for all tasks
    trial['t_targ_on'] = tc.time(3001)   # relevant for all tasks (may be missing if the fixation was broken) - fyi, 3000 is supposed to be a code for 'targ on' and 3000 for 'targ 1 on', but I see only 3001 so using that
    trial['t_dots_on'] = tc.time(4901)   # only for dot tasks
    trial['t_fix_off'] = tc.time(1025)   # relevant for the memory and overlap tasks (indicates when to initiate saccade)
    trial['t_targ_off'] = tc.time(3005)  # relevant for memory task
    trial['t_dots_off'] = tc.time(1101)  # ~30 ms (21-37 ms) after the saccade is detected
    trial['t_all_off'] = tc.time(4904)   # can be used to fill in some missing times, for tasks when things stay until the end or cases of broken fixations (currently not doing this)

    # behavioral responses
    trial.update(outcome(tc))
    trial['accuracy'] = True if trial['correct'] else False if trial['error'] else np.nan  # only have values for correct\incorrect, and nans for other cases, maybe turn no choice into error too

    # behavioral responses (only relevant for dots tasks)
    trial['targ_chosen_loc'] = tc.value_after(8140, offset=7500)
    trial['chosen_motion'] = 0 if trial['targ_chosen_loc'] in [1, 3] else 180 if trial['targ_chosen_loc'] in [2, 4] else np.nan
    trial['chosen_color'] = 1 if trial['targ_chosen_loc'] in [1, 2] else 2 if trial['targ_chosen_loc'] in [3, 4] else np.nan
    trial['accuracy_detailed'] = tc.value_after(8141, offset=7500)  # 3 both correct, 4 only color correct, 5 only motion correct, 6 both incorrect - will be changed to categorical below.
    # ! ariel had this as 4 different variables (correct_both (3), correct_color (3,4), correct_motion (3,5), error_both (6))
    trial['accuracy_motion'] = True if trial['accuracy_detailed'] in [3, 5] else False if trial['accuracy_detailed'] in [4, 6] else np.nan
    trial['accuracy_color'] = True if trial['accuracy_detailed'] in [3, 4] else False if trial['accuracy_detailed'] in [5, 6] else np.nan
    trial['accuracy_both'] = True if trial['accuracy_detailed'] == 3 else False if trial['accuracy_detailed'] in [4, 5, 6] else np.nan  # correct only if both were correct

    # fix for 1D tasks
    if trial['task_type'] == 113:  # 1D motion task
        trial['chosen_color'] = np.nan
        trial['accuracy_color'] = np.nan
        trial['accuracy_detailed'] = np.nan  # not relevant for the 1D tasks
    elif trial['task_type'] == 116:  # 1D color task
        trial['chosen_motion'] = np.nan
        trial['accuracy_motion'] = np.nan
        trial['accuracy_detailed'] = np.nan  # not relevant for the 1D tasks

    # behavioral times
    sac_time = tc.time(1007)    # defined like this so RT will appear before sacc_detected
    rt_event = trial['t_fix_off'] if (trial['task_type'] in [10, 11]) else trial['t_dots_on']  # which event are we using for RT calculation?
    trial['RT'] = sac_time - rt_event  # should be nan if one of them is missing
    trial['t_fix_acq'] = tc.time(4957)
    trial['t_sacc_detected'] = sac_time
    trial['t_targ_acq'] = tc.time(4919)

    # indicators of a bad trial
    trial['fix_acq'] = tc.present(4957)
    trial['fix_broken'] = tc.present(4955)  # only using as a boolean at this point, didn't feel I need the precise time
    trial['targ_acq'] = tc.present(4919)    # fixation acquired and held without a target acquired = no choice trials (but no choice trials can also have targ_acq)
    trial['bad_trial'] = (not trial['fix_acq'] or trial['fix_broken'] or not trial['targ_acq'])   # maybe define by the negative (i.e. define good trials to keep instead)

    trial.update(target_positions(tc, [(8008, 8009), (8012, 8013), (8057, 8058), (8111, 8112)]))
    if not extract_important_only:
        trial.update(display_settings(tc, target_size_codes=[8078, 8079, 8110, 8113],
                                      target_rgb_codes=[(8121, 8122, 8123), (8124, 8125, 8126), (8127, 8128, 8129), (8130, 8131, 8132)]))
    return trial


def parse_codes(event_times, event_codes, extract_important_only=False, testing_mode=False):
    """
    event codes -> trial table (one row per trial).
    extract_important_only: skip the display settings that should be constant across trials
    testing_mode: also print the codes that no field used (to check the parser covers everything)
    """
    df = parse_rig_trials(event_times, event_codes, lambda tc: parse_trial(tc, extract_important_only and not testing_mode), testing_mode=testing_mode)

    # conversion to categorical variables when needed
    label_column(df, 'task_type', TASK_TYPE_LABELS)
    for column in ['motion_dir', 'chosen_motion']:
        label_column(df, column, MOTION_DIR_LABELS)
    for column in ['color_dir', 'chosen_color']:
        label_column(df, column, COLOR_DIR_LABELS)
    for column in ['targ_correct_loc', 'targ_chosen_loc']:
        label_column(df, column, TARGET_LABELS)
    label_column(df, 'accuracy_detailed', ACCURACY_DETAILED_LABELS)
    return df


def keep_trial(trials):
    """Trials to analyze: a response was completed (i.e. not aborted at some point & not no response)."""
    return ~trials['accuracy'].isna()   # maybe use 'bad_trial' instead?
