"""
Task parsers: the event codes of a recording -> a trial table (one row per trial). Choose one with TASK in the config.

What's in this folder:
    <task>.py         one file per task - a task is any file here that defines parse_codes and keep_trial (see _template.py).
                      The file name is the task name: TASK = 'dots_2d' uses dots_2d.py. Nothing to register.
                      Ours: dots_2d.py, vd_pulse.py
    _template.py      a complete, commented example to copy for a new task (files starting with _ are not tasks)
    trial_codes.py    general tools for any task / lab: split the code stream into trials, read event times and values,
                      label categories, list unused codes
    our_rig_codes.py  the conventions of OUR task computer (trial start / end codes, display settings, outcome codes), read by our
                      task files. Another lab writes its own (in its task file, or in a shared file like this one)
"""
import importlib
import pkgutil
from pathlib import Path

TASK_FUNCTIONS = ('parse_codes', 'keep_trial')

def is_task(module):
    return all(hasattr(module, f) for f in TASK_FUNCTIONS)

def available_tasks():
    """The task names: files in this folder that define parse_codes and keep_trial (not starting with _)."""
    names = [m.name for m in pkgutil.iter_modules([str(Path(__file__).parent)]) if not m.name.startswith('_')]
    return [name for name in names if is_task(importlib.import_module(f'tasks.{name}'))]

def get_task(name):
    """The module of a task (tasks/<name>.py)."""
    if name not in available_tasks():
        raise ValueError(f"Unknown task: {name}. Available: {available_tasks()}. A task is a file tasks/<name>.py that defines "
                         f"{' and '.join(TASK_FUNCTIONS)} - see tasks/_template.py.")
    return importlib.import_module(f'tasks.{name}')
