def register_experiment(self, exp_type=None):
    analysis_interval = self.config["analysis_interval"]
    start_time, end_time = analysis_interval

    start_time, end_time = utils.str2datetime(start_time), utils.str2datetime(end_time)
    
    print(f"Creating {exp_type} experiment for interval {start_time} to {end_time}")
    
    print(f"Fetching radiation dose timeseries data in the interval from {start_time} to {end_time}")
    rad_dose_timeseries_data = self._load_rad_dose_timeseries_data_in_interval(start_time, end_time)
    print(f"DONE! Loaded radiation dose timeseries data in the interval from {rad_dose_timeseries_data['CRaTER-D1D2']['timestamp_utc'].min()} to {rad_dose_timeseries_data['CRaTER-D1D2']['timestamp_utc'].max()}")

    print(f"Fetching SDOML data filepaths in the interval from {start_time} to {end_time}")
    sdoml_data_filepaths = self._get_sdoml_data_filepaths_in_interval(start_time, end_time)
    if not sdoml_data_filepaths:
        print("No SDOML data files found in the specified interval.")  # Added this print statement

    print(sdoml_data_filepaths)
    if sdoml_data_filepaths:  # Added this condition to check if sdoml_data_filepaths is not empty
        sdoml_data_filepaths_start_time = utils.str2datetime(self._get_datetimestr_from_filepath(sdoml_data_filepaths[0]), formatstr='%Y%m%d_%H%M').astimezone(tz=datetime.timezone.utc)
        sdoml_data_filepaths_end_time = utils.str2datetime(self._get_datetimestr_from_filepath(sdoml_data_filepaths[-1]), formatstr='%Y%m%d_%H%M').astimezone(tz=datetime.timezone.utc)
        print(f"DONE! Found {len(sdoml_data_filepaths)} SDOML data files in the interval from {sdoml_data_filepaths_start_time} to {sdoml_data_filepaths_end_time}")
        print(f"\t First file: {sdoml_data_filepaths[0]}")
        print(f"\t Last file: {sdoml_data_filepaths[-1]}")

    print(f"Creating datapoints info table for the experiment")
    samples_tables = self.make_samples_table(exp_type, sdoml_data_filepaths, rad_dose_timeseries_data)
    print(f"DONE! Created datapoints info table for the experiment.")

    self.experiment_datapoints_tables = samples_tables

