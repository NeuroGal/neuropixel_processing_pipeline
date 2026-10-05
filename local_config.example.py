# Copy this file to local_config.py and edit it for your machine / rig (or keep several such files and pick one with --config).
# Only the settings you change need to be here - see config.py for all of them and their defaults.

INFO_CSV = "D:/Code/recording_info.csv"

SUMMARY_LOG_DIR = "D:/"

SERVER_SYNC = dict(sorting=False, postprocessing=False, behavior=True)

JOB_KWARGS = dict(n_jobs=0.8, chunk_duration='0.5s')

SECOND_SYSTEM = 'plexon'
TASK = 'vd_pulse'

# channels of the second system - the defaults are for Plexon (Taylor's rig); for Blackrock e.g.:
# CODES_CHANNEL = 'digital_input_port'
# SYNC_CHANNEL = 'ainp16'
# EYE_CHANNELS = ['ainp1', 'ainp2', 'ainp3']
