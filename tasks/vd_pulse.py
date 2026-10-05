# Gal Vishne 2025-2026
# contact me at gal.vishne@gmail.com
"""
Task parser: visually guided / memory guided saccades (VGS, MGS) and 1D motion discrimination with a motion pulse (VD_pulse_2),
pre and post injection (Taylor's rig).
"""
import numpy as np
from tasks.trial_codes import label_column
from tasks.our_rig_codes import parse_rig_trials, target_positions, display_settings, outcome

TASK_TYPE_LABELS = {2: 'VGS', 3: 'MGS', 20: 'VD_pulse_2'}


def parse_trial(tc, extract_important_only=True):
    trial = {}
    # trial settings
    # post-injection tasks have task_type as 102, 103, and 120 for VGS, MGS, and VD_pulse_2 - subtract the 100 to get the same labels pre and post injection
    task_type = tc.value_after(8017, offset=7000)
    trial['post_inj'] = bool(task_type > 100)
    trial['task_type'] = task_type - 100 if trial['post_inj'] else task_type

    trial['motion_coh'] = tc.value_after(8011, offset=7000, scale=1000)
    trial['motion_coh'] = 0.001 if trial['motion_coh'] == 0 else trial['motion_coh']  # to distinguish left and right 0% coh
    motion_dir = tc.value_after(8010, offset=7500)
    trial['motion_dir'] = motion_dir - 360 if motion_dir > 360 else motion_dir

    # signed motion coherence / direction - positive/1 for rightward, negative/-1 for leftward (nan without dots)
    trial['signed_motion_coh'] = (trial['motion_coh'] if (trial['motion_dir'] <= 90 or trial['motion_dir'] > 270)
                                  else -trial['motion_coh'] if (90 < trial['motion_dir'] <= 270) else np.nan)
    trial['signed_motion_dir'] = np.sign(trial['signed_motion_coh'])  # never 0 since 0% coherence is 0.001

    # motion pulse trial settings
    trial['pulse_size'] = tc.value_after(8055, offset=7500) * trial['signed_motion_dir']  # positive for rightward pulses, negative for leftward pulses, 0 for no pulse trials
    trial['pulse_duration'] = tc.value_after(8056, offset=7000)

    # event times (relative to trial start)
    trial['t_fix_on'] = tc.time(1010)     # should be present for all tasks
    trial['t_targ_on'] = tc.time(3001)    # relevant for all tasks (may be missing if the fixation was broken)
    trial['t_dots_on'] = tc.time(4901)    # only for dot tasks
    trial['t_pulse_on'] = tc.time(1102)   # motion pulse trials
    trial['t_pulse_off'] = tc.time(1103)  # motion pulse trials
    trial['t_fix_off'] = tc.time(1025)    # relevant for the memory and overlap tasks (indicates when to initiate saccade)
    trial['t_targ_off'] = tc.time(3005)   # relevant for memory task
    trial['t_dots_off'] = tc.time(1101)   # ~30 ms (21-37 ms) after the saccade is detected
    trial['t_all_off'] = tc.time(4904)    # can be used to fill in some missing times, for tasks when things stay until the end or cases of broken fixations (currently not doing this)
    trial['t_targ_feedback_on'] = tc.time(3006)  # unused?
    trial['t_reward_on'] = tc.time(3007)  # should be present for all tasks
    trial['dots_duration'] = trial['t_dots_off'] - trial['t_dots_on']  # nan if either is missing

    # behavioral responses
    trial.update(outcome(tc))
    trial['complete'] = True if trial['correct'] else False if trial['error'] else np.nan  # only have values for correct\incorrect, and nans for other cases

    # choice: 1 for choosing the right target, 0 for the left target (nan for trials without a completed choice)
    if not (trial['correct'] or trial['error']) or np.isnan(trial['signed_motion_dir']):
        trial['choice'] = np.nan
    else:
        chose_motion_dir = trial['signed_motion_dir'] if trial['correct'] else -trial['signed_motion_dir']
        trial['choice'] = int(chose_motion_dir > 0)

    # behavioral times
    trial['t_fix_acq'] = tc.time(4957)          # fixation acquired
    trial['t_sacc_detected'] = tc.time(1007)    # saccade detected
    trial['t_sacc_completed'] = tc.time(4919)   # target acquired = saccade completed

    # indicators of a bad trial
    trial['fix_acq'] = tc.present(4957)
    trial['fix_broken'] = tc.present(4955)
    trial['t_fix_broken'] = tc.time(4955) if trial['fix_broken'] else np.nan
    trial['targ_acq'] = tc.present(4919)    # there are cases when fixation was acquired, not broken, and still no target was acquired

    trial.update(target_positions(tc, [(8008, 8009), (8012, 8013)]))
    if not extract_important_only:
        trial.update(display_settings(tc, target_size_codes=[8078, 8079], target_rgb_codes=[(8121, 8122, 8123), (8124, 8125, 8126)]))
    return trial


def parse_codes(event_times, event_codes, extract_important_only=False, testing_mode=False):
    """
    event codes -> trial table (one row per trial).
    extract_important_only: skip the display settings that should be constant across trials
    testing_mode: also print the codes that no field used (to check the parser covers everything)
    """
    df = parse_rig_trials(event_times, event_codes, lambda tc: parse_trial(tc, extract_important_only and not testing_mode), testing_mode=testing_mode)
    label_column(df, 'task_type', TASK_TYPE_LABELS)
    return df


def keep_trial(trials):
    """Trials to analyze: a response was completed (correct or error)."""
    return ~trials['complete'].isna()
