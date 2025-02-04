"""
Data Extraction module for noise analysis.
Provides functionality to extract and process noise measurement data from raw files.
Supports parallel processing for improved performance.
"""

import os, concurrent, time, statistics, logging
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from logging.handlers import QueueHandler
from multiprocessing import freeze_support
from func.ulti import lr, ProcessingConfig, log_queue

# Enable PyDev debugging
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Setup logging
logger = logging.getLogger("Extract Noise")
queue_handler = QueueHandler(log_queue)
logger.setLevel(logging.INFO)
logger.addHandler(queue_handler)

class DataProcessor:
    """
    Processor class for extracting and analyzing noise measurement data.
    Handles parallel processing of multiple devices and data validation.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize data processor with configuration.

        Args:
            config: ProcessingConfig object containing processing parameters
        """
        self.config = config
        self.reset_parameters()
        logger.info("DataProcessor initialized")

    def reset_parameters(self):
        """Reset all processing parameters to initial state."""
        self.wafer_id = None
        self.lot_id = None
        self.die_folders = []
        self.device_list = []
        self.total_dies = None
        self.total_devices = None
        self.results = {}

    def extract_wafer_info(self):
        """
        Extract wafer and lot information from file path.

        Returns:
            tuple: (wafer_id, lot_id)
        """
        try:
            parts = str(self.config.base_path).split('/')
            assumed_wafer_id = parts[-2]
            lot_id = None

            # Check if path follows expected structure
            if parts[-5] == 'W'+assumed_wafer_id and parts[-4] == 'BSIM_W'+ assumed_wafer_id:
                lot_id = parts[-3]
            else:
                assumed_wafer_id = input("Please input wafer id:")
            return (assumed_wafer_id, lot_id)

        except Exception as e:
            logger.error(f"Error extracting wafer info: {e}")
            return ('UNKNOWN', None)

    def scan_structure(self):
        """
        Scan directory structure to identify dies and devices.
        Sets total_dies and total_devices based on found files.
        """
        # Find all die folders
        self.die_folders = [f for f in os.listdir(self.config.base_path)
                          if f.startswith('Die') and
                          os.path.isdir(os.path.join(self.config.base_path, f))]
        self.total_dies = len(self.die_folders)

        # Get device list from first die
        first_die_path = os.path.join(self.config.base_path, self.die_folders[0])
        self.device_list = [f for f in os.listdir(first_die_path)
                        if '_Svg' not in f and
                        os.path.isfile(os.path.join(first_die_path, f))]
        self.total_devices = len(self.device_list)

    def get_data_from_raw(self, file):
        """
        Extract data from raw measurement file.

        Args:
            file: Path to raw data file

        Returns:
            tuple: (header, bias_list, noise_data, num_of_frequency_points)

        Raises:
            FileNotFoundError: If file cannot be read
            ValueError: If data structure is invalid
        """
        try:
            with open(file, "r") as f:
                content = f.readlines()
        except Exception as e:
            raise FileNotFoundError(f"Error reading data file --> {e}")

        try:
            # Parse number of frequency points
            keywords = ['number']
            found_keywords = set()
            for line in content:
                line = line.strip()
                if line.startswith("Noise point Number"):
                    num_of_frequency_points = int(line.split('=', 1)[1].strip())
                    found_keywords.add('number')
                if found_keywords == set(keywords):
                    break
        except Exception as e:
            raise ValueError(f"Error capturing data structure information --> {e}")

        # Extract header and data tables
        header = content[75].strip().split(",")
        header.append("")

        try:
            # Extract bias data
            bias_list = []
            for i in range(1, 21):
                current_line = content[75 + i].strip().split(",")
                if current_line[0][0] == "[":
                    break
                bias_list.append([float(i) for i in current_line])
        except Exception as e:
            raise ValueError(f"Error capturing bias table --> {e}")

        try:
            # Extract noise data
            noise_data = []
            for j in range(num_of_frequency_points):
                current_line = content[76 + i + j].strip().split(",")
                noise_data.append([float(m) for m in current_line])
        except Exception as e:
            raise ValueError(f"Error capturing noise-frequency table --> {e}")

        return header, bias_list, noise_data, num_of_frequency_points

    def transpose_frequency_list(self, data):
        """
        Transpose frequency data for easier processing.

        Args:
            data: Raw frequency data list

        Returns:
            list: Transposed data list
        """
        if not data or not data[0]:
            return []

        num_conditions = len(data[0]) - 1
        result = [[] for _ in range(num_conditions + 1)]

        for row in data:
            result[0].append(row[0])
            for i in range(num_conditions):
                result[i + 1].append(row[i + 1])
        return result

    def check_bias_mismatch(self, var_idx, data):
        """
        Check for bias value mismatches across dies.

        Args:
            var_idx: Index of bias variable to check
            data: List of bias data for all dies

        Raises:
            ValueError: If bias mismatch detected
        """
        first_die_pos_biases = [point[var_idx] for point in data[0]]
        for die_pos_idx, die_pos in enumerate(data[1:], 1):
            current_biases = [point[var_idx] for point in die_pos]
            if not all(abs(b1 - b2) < 1e-10 for b1, b2 in zip(first_die_pos_biases, current_biases)):
                raise ValueError(f"Value mismatch: index {var_idx} in bias list in die position {die_pos_idx}")

    def stack_bias_list(self, data):
        """
        Stack bias data from multiple dies.

        Args:
            data: List of bias data from all dies

        Returns:
            list: Stacked bias data

        Raises:
            ValueError: If data inconsistency detected
        """
        if not data or not data[0]:
            return []

        num_bias_points = len(data[0])

        # Validate bias point count
        if any(len(die_pos) != num_bias_points for die_pos in data):
            raise ValueError("All die positions must have the same number of bias points")

        # Check bias consistency
        self.check_bias_mismatch(0, data)  # Check Vd
        self.check_bias_mismatch(2, data)  # Check Vg

        # Stack data points
        result = []
        for bias_idx in range(num_bias_points):
            bias_point_data = []
            for die_pos in data:
                bias_point_data.append(die_pos[bias_idx])
            result.append(bias_point_data)
        return result

    def transform_position_to_condition(self, position_data):
        """
        Transform position-based data to condition-based format.

        Args:
            position_data: Data organized by position

        Returns:
            list: Data organized by condition
        """
        if not position_data or not position_data[0] or not position_data[0][0]:
            return []

        num_freq_points = len(position_data[0])
        num_conditions = len(position_data[0][0]) - 1

        result = []
        for condition_idx in range(num_conditions):
            condition_data = []
            for freq_idx in range(num_freq_points):
                freq = position_data[0][freq_idx][0]
                pos_data = [pos[freq_idx][condition_idx + 1] for pos in position_data]
                freq_point_data = [freq] + pos_data
                condition_data.append(freq_point_data)
            result.append(condition_data)
        return result

    def insert_separator(self, data, interval, separator=''):
        """
        Insert separators into data list at specified intervals.

        Args:
            data: Input data list
            interval: Interval for separator insertion
            separator: Separator value to insert

        Returns:
            list: Data list with separators
        """
        result = []
        for i in range(len(data)):
            result.append(data[i])
            if (i + 1) % interval == 0 and i != len(data) - 1:
                result.append(separator)
        return result

    def prediction(self, prediction_dict, start_freq, end_freq, interest_freq):
        """
        Perform prediction analysis on noise data.

        Args:
            prediction_dict: Dictionary of prediction data
            start_freq: Starting frequency for prediction
            end_freq: Ending frequency for prediction
            interest_freq: Frequency points of interest

        Returns:
            dict: Prediction results

        Raises:
            ValueError: If prediction parameters are invalid
        """
        # Get frequency range
        frequency_range_list = [data[0] for data in prediction_dict[next(iter(prediction_dict))][1].values.tolist()]
        frequency_range_list_log = np.log10(frequency_range_list)

        # Validate interest frequencies
        if isinstance(interest_freq, float):
            if np.searchsorted(frequency_range_list, interest_freq) == len(frequency_range_list):
                raise ValueError("Frequency to be predicted is outside the data range")
            interest_freq = [interest_freq]
        elif isinstance(interest_freq, list):
            for freq in interest_freq:
                if not isinstance(freq, float):
                    raise ValueError("Element of input list must be a float")
                if np.searchsorted(frequency_range_list, freq) == len(frequency_range_list):
                    raise ValueError("Frequency to be predicted is outside the data range")
        else:
            raise ValueError("Input must be either a float or a list of floats")

        # Get prediction range indices
        try:
            start_index = np.searchsorted(frequency_range_list, start_freq)
            end_index = np.searchsorted(frequency_range_list, end_freq)
        except Exception as e:
            print(f"Cannot calculate prediction frequency range: {str(e)}")

        # Calculate predictions
        prediction_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for sheet_name, (p1, p2) in prediction_dict.items():
            bias_table = p2.values.tolist()
            for FoI in interest_freq:
                for idx in range(1, self.total_dies+1):
                    # Calculate linear regression
                    x_log = frequency_range_list_log[start_index:end_index+1]
                    y_log = np.log10([data[idx] for data in bias_table[start_index:end_index+1]])
                    slope, intercept, _ = lr(x_log, y_log)

                    # Calculate prediction
                    predict_y_result = np.power(10, slope * np.log10(FoI) + intercept)

                    # Store results
                    prediction_data[sheet_name][f'Freq={FoI}']['Raw'][f'Die{idx}'] = bias_table[np.searchsorted(frequency_range_list, FoI)][idx]
                    prediction_data[sheet_name][f'Freq={FoI}']['Predict'][f'Die{idx}'] = predict_y_result
                    prediction_data[sheet_name]['Parameters']['Id (A)'][f'Die{idx}'] = p1.iloc[0,idx]
                    prediction_data[sheet_name]['Parameters']['gm (S)'][f'Die{idx}'] = p1.iloc[1,idx]
                    prediction_data[sheet_name]['Parameters']['Vd (V)'][f'Die{idx}'] = p1.iloc[2,idx]
                    prediction_data[sheet_name]['Parameters']['Vg (V)'][f'Die{idx}'] = p1.iloc[3,idx]

        return prediction_data

    def prediction_export(self, output_file, prediction_data):
        """
        Export prediction results to Excel file.

        Args:
            output_file: Path to output Excel file
            prediction_data: Prediction results to export
        """
        try:
            # Prepare column structure
            sheet_names = []
            freqs = []
            types = []

            for sheet in prediction_data.keys():
                for freq in sorted(prediction_data[sheet].keys()):
                    if 'Freq' in freq:
                        freq_num = freq.split('=')[1]
                        sheet_names.extend([sheet, sheet])
                        freqs.extend([freq_num, freq_num])
                        types.extend(['Raw', 'Prediction'])
                    else:
                        sheet_names.extend([sheet] * 4)
                        freqs.extend(['Parameters'] * 4)
                        types.extend(['Id (A)', 'gm (S)', 'Vd (V)', 'Vg (V)'])

            # Create multi-index columns
            columns = pd.MultiIndex.from_arrays(
                [sheet_names, freqs, types],
                names=['Bias', 'Frequency', 'Type']
            )

            # Create DataFrame
            index = [f'Die{i}' for i in range(1, self.total_dies+1)]
            df = pd.DataFrame(index=index, columns=columns)

            # Fill data
            for sheet in prediction_data.keys():
                for freq in sorted(prediction_data[sheet].keys()):
                    if 'Freq' in freq:
                        freq_num = freq.split('=')[1]
                        for die in prediction_data[sheet][freq]['Raw'].keys():
                            df.loc[die, (sheet, freq_num, 'Raw')] = prediction_data[sheet][freq]['Raw'][die]
                            df.loc[die, (sheet, freq_num, 'Prediction')] = prediction_data[sheet][freq]['Predict'][die]
                            df.loc[die, (sheet, 'Parameters', 'Id (A)')] = prediction_data[sheet]['Parameters']['Id (A)'][die]
                            df.loc[die, (sheet, 'Parameters', 'gm (S)')] = prediction_data[sheet]['Parameters']['gm (S)'][die]
                            df.loc[die, (sheet, 'Parameters', 'Vd (V)')] = prediction_data[sheet]['Parameters']['Vd (V)'][die]
                            df.loc[die, (sheet, 'Parameters', 'Vg (V)')] = prediction_data[sheet]['Parameters']['Vg (V)'][die]

            # Write to Excel with formatting
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Prediction')
                workbook = writer.book
                worksheet = writer.sheets['Prediction']
                header_format = workbook.add_format({
                    'bold': False,
                    'align': 'center',
                    'border': 0
                })

                # Adjust column widths
                for col_num in range(len(df.columns.levels[0]) * len(df.columns.levels[1]) * 2):
                    worksheet.set_column(col_num + 1, col_num + 1, 15)

        except Exception as e:
            logger.error(f"Error writing prediction table: {e}")
            print(f"Error writing prediction table: {str(e)}")

    def process_single_device(self, device_name):
        """
        Process a single device's data.

        Args:
            device_name: Name of device to process

        Returns:
            str: Processing time in seconds
        """
        logger.info(f'Processing: {device_name[:-4]}')
        print(f'Processing: {device_name[:-4]}')
        start_time = time.perf_counter()

        # Initialize data containers
        bias_table = []
        noise_table = []

        # Collect data from all dies
        for die in self.die_folders:
            die_path = os.path.join(self.config.base_path, die)
            dut_path = os.path.join(die_path, device_name)
            if os.path.exists(dut_path):
                bias_table_header, bias_list, noise_list, num_of_frequency_points = self.get_data_from_raw(dut_path)
                bias_table.append(bias_list)
                noise_table.append(noise_list)

        # Process collected data
        stacked_bias_table = self.stack_bias_list(bias_table)
        transformed_noise_table = self.transform_position_to_condition(noise_table)

        # Process valid noise data
        valid_noise_list_ori = {}
        idx = 0
        for p1, p2 in zip(stacked_bias_table, transformed_noise_table):
            idx += 1
            sheet_name = f"Bias{idx}"

            # Skip invalid data
            if all(data == 0 for data in p1[0][1:4]) or all(data == 0 for data in p2[0][1:]):
                continue
            if len(p2[0][1:]) != len(p1) != self.total_dies:
                raise ValueError("Bias table and noise table must have the same number of die positions")

            # Prepare data for export
            die_prefix = [f"Die{j+1}" for j in range(self.total_dies)]
            prefix = ['Id (A)', 'gm (S)', 'Vd (V)', 'Vg (V)']
            selected_columns = zip(*[(row[1], row[5], row[0], row[2]) for row in p1])
            part1 = [
                [header] + list(col)
                for header, col in zip(prefix, selected_columns)
            ] + [[np.nan]]
            part1_header = ["Frequency"] + [f"{col}_Sid" for col in die_prefix]
            df_part1 = pd.DataFrame(part1, columns=part1_header)

            # Calculate statistics for each measurement point
            for row in p2:
                # Calculate Sid statistics
                sid_med = statistics.median(row[1:])
                sid_min = min(row[1:])
                sid_max = max(row[1:])

                # Calculate normalized Id statistics
                id2 = [self.get_normalised(x, y) for x,y in zip(row[1:], part1[0][1:])]
                id2_med = statistics.median(id2)
                id2_min = min(id2)
                id2_max = max(id2)

                # Calculate Gm statistics
                gm2 = [self.get_normalised(x, y) for x,y in zip(row[1:], part1[1][1:])]
                gm2_med = statistics.median(gm2)
                gm2_min = min(gm2)
                gm2_max = max(gm2)

                # Calculate frequency-dependent statistics
                f = [x*row[0] for x in row[1:]]
                f_med = statistics.median(f)
                f_min = min(f)
                f_max = max(f)

                # Combine all data
                row.extend(id2 + gm2 + f)
                row.extend([
                    sid_med, sid_min, sid_max,
                    id2_med, id2_min, id2_max,
                    gm2_med, gm2_min, gm2_max,
                    f_med, f_min, f_max
                ])

            # Create complete headers
            part2_header = (
                part1_header +
                [f"{col}_Sid/id^2" for col in die_prefix] +
                [f"{col}_Svg" for col in die_prefix] +
                [f"{col}_Sid*f" for col in die_prefix] +
                ['Sid_med', 'Sid_min', 'Sid_max',
                 'Sid/id^2_med', 'Sid/id^2_min', 'Sid/id^2_max',
                 'Svg_med', 'Svg_min', 'Svg_max',
                 'Sid*f_med', 'Sid*f_min', 'Sid*f_max']
            )
            df_part2 = pd.DataFrame(p2, columns=part2_header)
            valid_noise_list_ori[sheet_name] = (df_part1, df_part2)

        # Export results if not in prediction-only mode
        if not self.config.prediction_only_flag:
            for sheet_name, (df_part1, df_part2) in valid_noise_list_ori.items():
                # Align columns between parts
                for col in df_part2.columns:
                    if col not in df_part1.columns:
                        df_part1[col] = np.nan
                df_part1 = df_part1[df_part2.columns]

                # Combine and export data
                df_noise_list_ori = pd.concat([df_part1, df_part2], axis=0)
                output_file = os.path.join(
                    self.config.output_path,
                    f"{device_name[:-4]}_W#{self.wafer_id}_{sheet_name}.xlsx"
                )

                # Write to Excel with formatting
                with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                    df_noise_list_ori.to_excel(writer, sheet_name=sheet_name, index=False, header=True)
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]
                    header_format = workbook.add_format({'bold': False, 'border': 0})

                    # Apply formatting
                    for col_num, value in enumerate(df_noise_list_ori.columns):
                        worksheet.write(0, col_num, value, header_format)
                    for col_num in range(df_noise_list_ori.shape[1]):
                        worksheet.set_column(col_num + 1, col_num + 1, 12)

        # Generate and export predictions
        prediction_result = self.prediction(
            valid_noise_list_ori,
            self.config.pred_range_lower,
            self.config.pred_range_upper,
            self.config.interest_freq
        )
        output_file = os.path.join(
            self.config.output_path,
            f"0_Prediction_{device_name[:-4]}_W#{self.wafer_id}.xlsx"
        )
        self.prediction_export(output_file, prediction_result)

        # Return processing time
        total_time = f"{time.perf_counter()-start_time:2f}"
        return total_time

    def process_all_devices(self):
        """
        Process all devices in parallel.
        Manages worker processes and tracks progress.
        """
        print(f"{__name__} is running")

        freeze_support()
        self.scan_structure()
        print(f"Found {self.total_dies} dies and {self.total_devices} devices")

        try:
            # Get wafer information
            if not self.config.debug_flag:
                wafer_info = self.extract_wafer_info()
                self.wafer_id = wafer_info[0]
                self.lot_id = wafer_info[1]
                print(f"Wafer ID: {self.wafer_id}, Lot ID is: {self.lot_id}")
            else:
                self.lot_id = 'DEBUG'
                self.wafer_id = 'DEBUG'

            # Initialize parallel processing
            completed = 0
            num_workers = min(os.cpu_count(), self.total_devices)
            logger.info(f"Submitting worker")

            try:
                # Process devices in parallel
                with ProcessPoolExecutor(max_workers=num_workers) as process_executor:
                    future_list = {
                        process_executor.submit(self.process_single_device, device): device
                        for device in self.device_list
                    }

                    # Track completion and progress
                    for future in concurrent.futures.as_completed(future_list):
                        completed += 1
                        device = future_list[future][:-4]
                        try:
                            result = future.result()
                            print(f"{device} completed in {result} seconds.")
                        except Exception as e:
                            logger.error(f"Error processing {device}: {str(e)}")
                            print(f"Error processing {device}: {str(e)}")

                        # Update progress
                        progress = (completed / self.total_devices) * 100
                        print(f"Progress: {completed}/{self.total_devices} ({progress:.2f}%)")

            except concurrent.futures.process.BrokenProcessPool as e:
                logger.error(f"Process pool broken: {str(e)}")
                print(f"Process pool error: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in process_all_devices: {str(e)}")
            print(f"Error processing devices: {str(e)}")
            raise

        print("All devices processed.")
        self.reset_parameters()

    def get_normalised(self, x, factor):
        """
        Calculate normalized value.

        Args:
            x: Value to normalize
            factor: Normalization factor

        Returns:
            float: Normalized value
        """
        if factor == 0:
            return np.inf
        return x / factor / factor
