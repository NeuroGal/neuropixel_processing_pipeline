# Run the pipelines on a batch of recordings (Windows PowerShell).
# Usage: edit the settings below, then run:  .\run_pipelines.ps1
# (the same commands work from any terminal: python sorting_pipeline.py --monkeys Jones --dates 20260702 ...)
# Settings are in the config (local_config.py, or the file given in $config_file) - see config.py

# ---------------- settings ----------------
$conda_hook = "C:\Users\User\anaconda3\shell\condabin\conda-hook.ps1"   # path to your conda installation
$conda_env  = "taylor_spike_sorting"                                    # conda environment (see environment.yml)
$code_dir   = $PSScriptRoot                                             # folder with the pipeline scripts (this folder)
$config_file = $null                                                    # $null = local_config.py, or e.g. "$code_dir\configs\taylor_rig.py"

# which pipelines to run (in this order)
$run_sorting        = $false
$run_postprocessing = $false
$run_behavior       = $true
# ------------------------------------------

Set-Location $code_dir
& $conda_hook
conda activate $conda_env

# If the computer freezes or crashes during a run (CPU / memory overload): uncomment these two lines.
# SpikeInterface already runs n_jobs processes in parallel (JOB_KWARGS in the config), and numpy / scipy / torch inside each
# of them can start their own threads on every core (through OpenBLAS / OpenMP) - together that can be far more threads than
# the computer has cores. These lines cap the threads each process may start (6-7 worked on our rigs; lower them if it still crashes).
# Lowering n_jobs / chunk_duration in JOB_KWARGS is the other thing to try.
# $env:OPENBLAS_NUM_THREADS = "6"
# $env:OMP_NUM_THREADS = "6"

# turns "name value name value ..." into "--name value --name value ...", skipping $null values; arrays become multiple values
function Format-Args {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        $ArgsList
    )

    $result = @()
    for ($i = 0; $i -lt $ArgsList.Count; $i += 2) {
        $name = "--$($ArgsList[$i])"
        $value = $ArgsList[$i + 1]

        if ($null -eq $value) {
            continue
        }

        if ($value -is [System.Array]) {
            $result += @($name) + ($value | ForEach-Object { "$_" })
        } else {
            $result += @($name, "$value")
        }
    }
    return $result
}

# which recordings to run (rows of the recording table) - leaving all filters out runs everything in the table
# options: one per RECORDING_KEYS / FILTER_COLUMNS column of the config, plural - by default monkeys, dates, regions, session_nums
# (each can be one value or an array, e.g. @("Pul", "LIP"))
# all other settings (task, second system, parallel jobs, LFP, overwriting, ...) are in the config file
$cur_args = Format-Args monkeys "Jones" dates "20260702" regions @("Pul", "LIP") config $config_file

if ($run_sorting)        { python "$code_dir\sorting_pipeline.py" @cur_args }
if ($run_postprocessing) { python "$code_dir\postprocessing_pipeline.py" @cur_args }
if ($run_behavior)       { python "$code_dir\behavior_pipeline.py" @cur_args }


# ---------------- run notes ----------------
# Ali's rig: 18 prepro
# Taylor's rig: 25 postpro

# TODO:
# 1. 0622 (21), 25 Pul postpro with n_jobs = 1
# 2. 0611 (18) prepro
# 0713 (26) prepro
