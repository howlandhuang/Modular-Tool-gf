"""
Data Extraction module for noise analysis.
Provides functionality to extract and process noise measurement data from raw files.
Supports parallel processing for improved performance.
"""

import os, concurrent, time, statistics, logging, re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from logging.handlers import QueueHandler
from multiprocessing import freeze_support
from func.ulti import lr, ProcessingConfig, log_queue
from PyQt6.QtWidgets import QInputDialog, QWidget

# Enable PyDev debugging
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Initialize module logger - no need to add handler as it inherits from root logger
logger = logging.getLogger(__name__)

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
        logger.info("Initializing DataProcessor")
        self.config = config
        self.reset_parameters()
        logger.debug("DataProcessor initialized successfully")

    def reset_parameters(self):
        """Reset all processing parameters to initial state."""
        logger.debug("Resetting processing parameters")
        self.wafer_id = None
        self.lot_id = None
        self.die_folders = []
        self.device_list = []
        self.total_dies = None
        self.total_devices = None
        self.results = {}
        logger.debug("Processing parameters reset complete")

    def extract_wafer_info_from_path(self):
        """
        Extract wafer and lot information from file path.

        Returns:
            tuple: (wafer_id, lot_id)
        """
        logger.debug("Starting wafer info extraction")
        try:
            parts = str(self.config.base_path).split('/')

            # Check if path follows expected structure
            if parts[-4] == 'BSIM_W'+  parts[-2] or parts[-4] == 'BSIM_'+  parts[-2] :
                wafer_id = parts[-2].replace('w', '').replace('W', '')
                logger.info(f"Found wafer ID from path structure: {wafer_id}")
            else:
                logger.warning("Path structure does not match expected format")
                # Create a temporary QWidget as parent for the dialog
                temp_widget = QWidget()
                while True:  # Loop until valid input is received
                    wafer_id, ok = QInputDialog.getText(
                    temp_widget,
                    'Wafer ID Input',
                    'Path structure does not match expected format.\nPlease input wafer ID:'
                    )
                    if not ok:  # User clicked Cancel or closed the dialog
                        logger.debug("User cancelled wafer id input")
                        wafer_id = 'UNKNOWN'
                        break  # Exit the this loop
                    is_valid, err_msg, wafer_id = self.validator_processor.validate_wafer_id(wafer_id)
                    if is_valid:
                        logger.info(f"User provided wafer ID: {wafer_id}")
                        break
                    else:
                        logger.warning("User inputs an invalid wafer ID")
                temp_widget.deleteLater()  # Clean up the temporary widget


            if match := re.match(r'^\d[a-zA-Z]{3}\d{5}(_RT)?$', parts[-3]):
                lot_id = match.group(0)
                logger.info(f"Found lot ID from path structure: {lot_id}")
            else:
                logger.warning("Path structure does not match expected format")
                # Create a temporary QWidget as parent for the dialog
                temp_widget = QWidget()
                while True:  # Loop until valid input is received
                    lot_id, ok = QInputDialog.getText(
                    temp_widget,
                    'Lot ID Input',
                    'Path structure does not match expected format.\nPlease input lot ID:'
                    )
                    if not ok:  # User clicked Cancel or closed the dialog
                        logger.debug("User cancelled lot id input")
                        lot_id = 'UNKNOWN'
                        break  # Exit the this loop
                    is_valid, err_msg, lot_id = self.validator_processor.validate_lot_id(lot_id)
                    if is_valid:
                        logger.info(f"User provided lot ID: {lot_id}")
                        break
                    else:
                        logger.warning("User inputs an invalid lot ID")
                temp_widget.deleteLater()  # Clean up the temporary widget

            logger.info(f"Extracted wafer info - Wafer ID: {wafer_id}, Lot ID: {lot_id}")
            return (wafer_id, lot_id)

        except Exception as e:
            logger.error(f"Error extracting wafer info: {str(e)}")
            logger.warning("Returning default values due to error")
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
        logger.debug(f"Found {self.total_dies} die folders")

        # Get device list from first die
        first_die_path = os.path.join(self.config.base_path, self.die_folders[0])
        self.device_list = [f for f in os.listdir(first_die_path)
                        if '_Svg' not in f and
                        os.path.isfile(os.path.join(first_die_path, f))]
        self.total_devices = len(self.device_list)
        logger.debug(f"Found {self.total_devices} devices to process")

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
        logger.debug(f"Reading raw data from file: {file}")
        try:
            with open(file, "r") as f:
                content = f.readlines()
        except Exception as e:
            logger.error(f"Error reading data file: {str(e)}")
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
                    logger.debug(f"Found {num_of_frequency_points} frequency points")
                if found_keywords == set(keywords):
                    break
        except Exception as e:
            logger.error(f"Error parsing frequency points: {str(e)}")
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
            logger.debug(f"Extracted {len(bias_list)} bias points")
        except Exception as e:
            logger.error(f"Error extracting bias data: {str(e)}")
            raise ValueError(f"Error capturing bias table --> {e}")

        try:
            # Extract noise data
            noise_data = []
            for j in range(num_of_frequency_points):
                current_line = content[76 + i + j].strip().split(",")
                noise_data.append([float(m) for m in current_line])
            logger.debug(f"Extracted noise data with {len(noise_data)} points")
        except Exception as e:
            logger.error(f"Error extracting noise data: {str(e)}")
            raise ValueError(f"Error capturing noise-frequency table --> {e}")

        logger.info(f"Successfully extracted data from {file}")
        return header, bias_list, noise_data, num_of_frequency_points

    def transpose_frequency_list(self, data):
        """
        Transpose frequency data for easier processing.

        Args:
            data: Raw frequency data list

        Returns:
            list: Transposed data list
        """
        logger.debug("Starting frequency list transposition")
        if not data or not data[0]:
            logger.warning("Empty data provided for transposition")
            return []

        num_conditions = len(data[0]) - 1
        result = [[] for _ in range(num_conditions + 1)]

        for row in data:
            result[0].append(row[0])
            for i in range(num_conditions):
                result[i + 1].append(row[i + 1])

        logger.debug(f"Transposed {len(data)} rows with {num_conditions} conditions")
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
        logger.debug(f"Checking bias mismatch for variable index {var_idx}")
        first_die_pos_biases = [point[var_idx] for point in data[0]]
        for die_pos_idx, die_pos in enumerate(data[1:], 1):
            current_biases = [point[var_idx] for point in die_pos]
            if not all(abs(b1 - b2) < 1e-10 for b1, b2 in zip(first_die_pos_biases, current_biases)):
                logger.error(f"Bias mismatch detected in die position {die_pos_idx}")
                raise ValueError(f"Value mismatch: index {var_idx} in bias list in die position {die_pos_idx}")
        logger.debug("Bias consistency check passed")

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
        logger.debug("Starting bias list stacking")
        if not data or not data[0]:
            logger.warning("Empty data provided for stacking")
            return []

        num_bias_points = len(data[0])
        logger.debug(f"Processing {len(data)} dies with {num_bias_points} bias points each")

        # Validate bias point count
        if any(len(die_pos) != num_bias_points for die_pos in data):
            logger.error("Inconsistent number of bias points across dies")
            raise ValueError("All die positions must have the same number of bias points")

        # Check bias consistency
        logger.debug("Checking bias consistency")
        self.check_bias_mismatch(0, data)  # Check Vd
        self.check_bias_mismatch(2, data)  # Check Vg

        # Stack data points
        result = []
        for bias_idx in range(num_bias_points):
            bias_point_data = []
            for die_pos in data:
                bias_point_data.append(die_pos[bias_idx])
            result.append(bias_point_data)

        logger.debug(f"Successfully stacked {len(result)} bias points")
        return result

    def transform_position_to_condition(self, position_data):
        """
        Transform position-based data to condition-based format.

        Args:
            position_data: Data organized by position

        Returns:
            list: Data organized by condition
        """
        logger.debug("Starting position to condition transformation")
        if not position_data or not position_data[0] or not position_data[0][0]:
            logger.warning("Empty data provided for transformation")
            return []

        num_freq_points = len(position_data[0])
        num_conditions = len(position_data[0][0]) - 1
        logger.debug(f"Processing {num_freq_points} frequency points with {num_conditions} conditions")

        result = []
        for condition_idx in range(num_conditions):
            condition_data = []
            for freq_idx in range(num_freq_points):
                freq = position_data[0][freq_idx][0]
                pos_data = [pos[freq_idx][condition_idx + 1] for pos in position_data]
                freq_point_data = [freq] + pos_data
                condition_data.append(freq_point_data)
            result.append(condition_data)

        logger.debug(f"Successfully transformed data to {len(result)} conditions")
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
        logger.info("Starting prediction analysis")
        logger.debug(f"Prediction range: {start_freq} to {end_freq} Hz")
        logger.debug(f"Interest frequencies: {interest_freq}")

        # Get frequency range
        frequency_range_list = [data[0] for data in prediction_dict[next(iter(prediction_dict))][1].values.tolist()]
        frequency_range_list_log = np.log10(frequency_range_list)
        logger.debug(f"Frequency range established with {len(frequency_range_list)} points")

        # Validate interest frequencies
        try:
            if isinstance(interest_freq, float):
                if np.searchsorted(frequency_range_list, interest_freq) == len(frequency_range_list):
                    logger.error(f"Interest frequency {interest_freq} outside data range")
                    raise ValueError("Frequency to be predicted is outside the data range")
                interest_freq = [interest_freq]
            elif isinstance(interest_freq, list):
                for freq in interest_freq:
                    if not isinstance(freq, float):
                        logger.error(f"Invalid frequency type: {type(freq)}")
                        raise ValueError("Element of input list must be a float")
                    if np.searchsorted(frequency_range_list, freq) == len(frequency_range_list):
                        logger.error(f"Interest frequency {freq} outside data range")
                        raise ValueError("Frequency to be predicted is outside the data range")
            else:
                logger.error(f"Invalid interest_freq type: {type(interest_freq)}")
                raise ValueError("Input must be either a float or a list of floats")
        except Exception as e:
            logger.error(f"Error validating interest frequencies: {str(e)}")
            raise

        # Get prediction range indices
        try:
            start_index = np.searchsorted(frequency_range_list, start_freq)
            end_index = np.searchsorted(frequency_range_list, end_freq)
            logger.debug(f"Prediction range indices: {start_index} to {end_index}")
        except Exception as e:
            logger.error(f"Error calculating prediction range: {str(e)}")
            raise ValueError(f"Cannot calculate prediction frequency range: {str(e)}")

        # Calculate predictions
        prediction_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        try:
            for sheet_name, (p1, p2) in prediction_dict.items():
                logger.debug(f"Processing predictions for sheet: {sheet_name}")
                bias_table = p2.values.tolist()

                for FoI in interest_freq:
                    logger.debug(f"Calculating predictions for frequency {FoI}")
                    for idx in range(1, self.total_dies+1):
                        # Calculate linear regression
                        x_log = frequency_range_list_log[start_index:end_index+1]
                        y_log = np.log10([data[idx] for data in bias_table[start_index:end_index+1]])
                        slope, intercept, r_square = lr(x_log, y_log)
                        logger.debug(f"Linear regression for die {idx}: slope={slope:.4f}, intercept={intercept:.4f}, R²={r_square:.4f}")

                        # Calculate prediction
                        predict_y_result = np.power(10, slope * np.log10(FoI) + intercept)

                        # Store results
                        prediction_data[sheet_name][f'Freq={FoI}']['Raw'][f'Die{idx}'] = bias_table[np.searchsorted(frequency_range_list, FoI)][idx]
                        prediction_data[sheet_name][f'Freq={FoI}']['Predict'][f'Die{idx}'] = predict_y_result
                        prediction_data[sheet_name]['Parameters']['Id (A)'][f'Die{idx}'] = p1.iloc[0,idx]
                        prediction_data[sheet_name]['Parameters']['gm (S)'][f'Die{idx}'] = p1.iloc[1,idx]
                        prediction_data[sheet_name]['Parameters']['Vd (V)'][f'Die{idx}'] = p1.iloc[2,idx]
                        prediction_data[sheet_name]['Parameters']['Vg (V)'][f'Die{idx}'] = p1.iloc[3,idx]
        except Exception as e:
            logger.error(f"Error calculating predictions: {str(e)}")
            raise

        logger.info("Prediction analysis completed successfully")
        return prediction_data

    def prediction_export(self, output_file, prediction_data):
        """
        Export prediction results to Excel file.

        Args:
            output_file: Path to output Excel file
            prediction_data: Prediction results to export
        """
        logger.info(f"Exporting prediction results to {output_file}")
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

            logger.debug("Column structure prepared")

            # Create multi-index columns
            columns = pd.MultiIndex.from_arrays(
                [sheet_names, freqs, types],
                names=['Bias', 'Frequency', 'Type']
            )

            # Create DataFrame
            index = [f'Die{i}' for i in range(1, self.total_dies+1)]
            df = pd.DataFrame(index=index, columns=columns)
            logger.debug(f"Created DataFrame with shape {df.shape}")

            # Fill data
            for sheet in prediction_data.keys():
                logger.debug(f"Filling data for sheet: {sheet}")
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
            logger.debug("Writing to Excel file")
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

            logger.info("Prediction results exported successfully")
        except Exception as e:
            logger.error(f"Error exporting prediction results: {str(e)}")
            raise

    def process_single_device(self, device_name):
        """
        Process a single device's data.

        Args:
            device_name: Name of device to process

        Returns:
            str: Processing time in seconds
        """
        logger.info(f'Processing device: {device_name[:-4]}')
        start_time = time.perf_counter()

        try:
            # Initialize data containers
            bias_table = []
            noise_table = []

            # Collect data from all dies
            logger.debug("Collecting data from all dies")
            for die in self.die_folders:
                die_path = os.path.join(self.config.base_path, die)
                dut_path = os.path.join(die_path, device_name)
                if os.path.exists(dut_path):
                    logger.debug(f"Processing die: {die}")
                    bias_table_header, bias_list, noise_list, num_of_frequency_points = self.get_data_from_raw(dut_path)
                    bias_table.append(bias_list)
                    noise_table.append(noise_list)

            # Process collected data
            logger.debug("Processing collected data")
            stacked_bias_table = self.stack_bias_list(bias_table)
            transformed_noise_table = self.transform_position_to_condition(noise_table)

            # Process valid noise data
            valid_noise_list_ori = {}
            idx = 0
            for p1, p2 in zip(stacked_bias_table, transformed_noise_table):
                idx += 1
                sheet_name = f"Bias{idx}"
                logger.debug(f"Processing {sheet_name}")

                # Skip invalid data
                if all(data == 0 for data in p1[0][1:4]) or all(data == 0 for data in p2[0][1:]):
                    continue
                if len(p2[0][1:]) != len(p1) != self.total_dies:
                    logger.error(f"Data mismatch in {sheet_name}")
                    raise ValueError("Bias table and noise table must have the same number of die positions")

                # Prepare data for export
                logger.debug(f"Preparing data for {sheet_name}")
                die_prefix = [f"Die{j+1}" for j in range(self.total_dies)]
                prefix = ['Id (A)', 'gm (S)', 'Vd (V)', 'Vg (V)']
                selected_columns = zip(*[(row[1], row[5], row[0], row[2]) for row in p1])
                part1 = [
                    [header] + list(col)
                    for header, col in zip(prefix, selected_columns)
                ] + [[np.nan]]
                part1_header = ["Frequency"] + [f"{col}_Sid" for col in die_prefix]
                df_part1 = pd.DataFrame(part1, columns=part1_header)

                # Calculate statistics
                logger.debug("Calculating statistics")
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
                logger.debug("Exporting processed data")
                for sheet_name, (df_part1, df_part2) in valid_noise_list_ori.items():
                    # Align columns between parts
                    missing_cols = [col for col in df_part2.columns if col not in df_part1.columns]
                    missing_data = pd.DataFrame(
                        np.nan,
                        index=df_part1.index,
                        columns=missing_cols
                    )
                    df_part1 = pd.concat([df_part1, missing_data], axis=1)[df_part2.columns]

                    # Combine and export data
                    df_noise_list_ori = pd.concat([df_part1, df_part2], axis=0)
                    output_file = os.path.join(
                        self.config.output_path,
                        f"{device_name[:-4]}_W#{self.wafer_id}_{sheet_name}.xlsx"
                    )
                    logger.info(f"Saving processed data to {output_file}")

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
            logger.debug("Generating predictions")
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
            logger.info(f"Device processing completed in {total_time} seconds")
            return total_time

        except Exception as e:
            logger.error(f"Error processing device {device_name}: {str(e)}")
            raise

    def process_all_devices(self):
        """
        Process all devices in parallel.
        Manages worker processes and tracks progress.
        """
        freeze_support()
        logger.info("Starting parallel processing of all devices")
        try:
            self.scan_structure()
            logger.info(f"Found {self.total_dies} dies and {self.total_devices} devices")

            if not self.config.debug_flag:
                wafer_info = self.extract_wafer_info_from_path()
                self.wafer_id = wafer_info[0]
                self.lot_id = wafer_info[1]
            else:
                self.lot_id = 'DEBUG'
                self.wafer_id = 'DEBUG'
            logger.debug(f"Wafer ID: {self.wafer_id}, Lot ID is: {self.lot_id}")

            # Process devices in parallel
            logger.info("Starting parallel device processing")
            start_time = time.perf_counter()
            execution_times = {}

            with ProcessPoolExecutor() as executor:
                # Submit all tasks and store futures with their device names
                future_to_device = {
                    executor.submit(self.process_single_device, device): device
                    for device in self.device_list
                }

                # Wait for completion and collect results
                for future in concurrent.futures.as_completed(future_to_device):
                    device = future_to_device[future]
                    try:
                        processing_time = float(future.result())  # Get the returned processing time
                        execution_times[device] = processing_time
                        logger.info(f"Device {device[:-4]} completed in {processing_time:.2f} seconds")
                    except Exception as e:
                        logger.error(f"Device {device[:-4]} failed with error: {str(e)}")
                        raise

            # Log execution time statistics
            total_time = time.perf_counter() - start_time
            avg_time = sum(execution_times.values()) / len(execution_times)
            max_time = max(execution_times.values())
            min_time = min(execution_times.values())
            slowest_device = max(execution_times.items(), key=lambda x: x[1])[0]
            fastest_device = min(execution_times.items(), key=lambda x: x[1])[0]

            logger.info("Parallel processing completed")
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            logger.info(f"Average device processing time: {avg_time:.2f} seconds")
            logger.info(f"Fastest device: {fastest_device[:-4]} ({min_time:.2f} seconds)")
            logger.info(f"Slowest device: {slowest_device[:-4]} ({max_time:.2f} seconds)")
            logger.info("All devices processed successfully")
            self.reset_parameters()

        except Exception as e:
            logger.error(f"Error in process_all_devices: {str(e)}")
            raise

    def get_normalised(self, x, factor):
        """
        Calculate normalized value.

        Args:
            x: Value to normalize
            factor: Normalization factor

        Returns:
            float: Normalized value
        """
        try:
            if factor == 0:
                return np.inf
            result = x / factor / factor
            return result
        except Exception as e:
            logger.error(f"Error in normalization calculation: {str(e)}")
            raise
