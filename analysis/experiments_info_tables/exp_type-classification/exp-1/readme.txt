1.  Analysis_interval: This specifies the time interval for the analysis.
    The data for the experiment will be fetched and processed for this period. In this case, it covers the entire month of May 2014.
    "analysis_interval": ["2014-05-01 00:00:00+00:00", "2014-05-31 00:00:00+00:00"]
2.  RADLAB_DATA_DIR: This is the directory path where the radiation dose data files are stored.
"RADLAB_DATA_DIR": "/home/daisysong/2024-HL-Virtual-Dosimeter/data/radlab-private/data_tables/readings_table/per_instrument_padded"

3.  SDOML_DATA_DIR: This is the directory path where the Solar Dynamics Observatory Machine Learning (SDOML) data files are stored.
"SDOML_DATA_DIR": "/home/daisysong/2024-HL-Virtual-Dosimeter/data/solar_data/sdoml_sample"

4. radiation_data: This section contains parameters related to the radiation data:

instrument_ids: A list of instrument IDs whose data will be used in the experiment.
input_window_duration: Duration of the input window in seconds.
target_window_duration: Duration of the target window in seconds.

"radiation_data": {
    "instrument_ids": ["CRaTER-D1D2"],
    "input_window_duration": 43200,
    "target_window_duration": 21600
}

5. solar_data: This section contains parameters related to the solar data:
num_input_sdo_images: Number of input SDO images to be used.

"solar_data": {
    "num_input_sdo_images": 3
}

6. exclusion_window_duration: The duration of the exclusion window in seconds. This parameter helps in defining a window to exclude certain data points from the analysis.
"exclusion_window_duration": 1800

7. exp_save_dir: The directory path where the experiment's metadata and results will be saved.
"exp_save_dir": "/home/daisysong/2024-HL-Virtual-Dosimeter/analysis/experiments_info_tables"

8. use_saved_exp_config: A boolean flag that indicates whether to use a saved experiment configuration if it matches the current one.
"use_saved_exp_config": true



Example Usage in Code
This configuration dictionary is used by the script to control various aspects of the experiment creation and execution process. For example:
exp_config = {
    "analysis_interval": ["2014-05-01 00:00:00+00:00", "2014-05-31 00:00:00+00:00"],
    "RADLAB_DATA_DIR": "/home/daisysong/2024-HL-Virtual-Dosimeter/data/radlab-private/data_tables/readings_table/per_instrument_padded",
    "SDOML_DATA_DIR": "/home/daisysong/2024-HL-Virtual-Dosimeter/data/solar_data/sdoml_sample",
    "radiation_data": {
        "instrument_ids": ["CRaTER-D1D2"],
        "input_window_duration": 43200,
        "target_window_duration": 21600
    },
    "solar_data": {
        "num_input_sdo_images": 3
    },
    "exclusion_window_duration": 1800,
    "exp_save_dir": "/home/daisysong/2024-HL-Virtual-Dosimeter/analysis/experiments_info_tables",
    "use_saved_exp_config": true
}

datapoints_info_table = get_experiment_datapoints_tables(exp_type='classification', exp_config=exp_config)

In this example, the exp_config dictionary is passed to the function get_experiment_datapoints_tables, which uses the specified parameters to fetch data, create targets, and save the experiment's metadata.