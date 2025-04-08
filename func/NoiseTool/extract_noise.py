"""
Data Extraction module for noise analysis.
Provides functionality to extract and process noise measurement data from raw files.
Supports parallel processing for improved performance.
"""

import os, concurrent, time, statistics, logging, re
import pandas as pd
import numpy as np
from typing import List
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from multiprocessing import freeze_support
from func.ulti import (
    lr, ProcessingConfig,
    validate_wafer_id, get_user_input, validate_lot_id, validate_width_length,
    parse_device_info, LOT_ID_PATTERN
)


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
            # Find all matches of lot ID pattern in path and take the last one
            # defined in func/ulti.py
            matches = re.finditer(LOT_ID_PATTERN, str(self.config.base_path))
            matches_list = list(matches)
            if matches_list:
                lot_id = matches_list[-1].group(0)  # Take the last match
                logger.info(f"Found lot ID from path structure: {lot_id}")
            else:
                logger.warning("Path structure does not match expected format")

                lot_id = get_user_input(
                    'Lot ID Input',
                    'Please input lot ID:',
                    validate_lot_id
                )
                # if the get_user_input is failed, lot_id falls back to 'UNKNOWN' to make sure the program can run
                if lot_id is None:
                    lot_id = 'UNKNOWN'

        except Exception as e:
            logger.error(f"Error extracting lot id: {str(e)}")
            lot_id = 'UNKNOWN'

        try:
            parts = str(self.config.base_path).split('/')

            # Check if path follows expected structure
            if parts[-4] == 'BSIM_W'+  parts[-2] or parts[-4] == 'BSIM_'+  parts[-2] :
                wafer_id = parts[-2].replace('w', '').replace('W', '')
                logger.info(f"Found wafer ID from path structure: {wafer_id}")
            else:
                logger.warning("Path structure does not match expected format")
                wafer_id= get_user_input(
                    'Wafer ID Input',
                    'Please input wafer ID:',
                    validate_wafer_id
                )
                # if the get_user_input is failed, wafer_id falls back to 'UNKNOWN' to make sure the program can run
                if wafer_id is None:
                    wafer_id = 'UNKNOWN'
                else:
                    # we add 'W' in the final output('W#xxx') so here we only need the number
                    wafer_id = wafer_id.replace('w', '').replace('W', '')
        except Exception as e:
            logger.error(f"Error extracting wafer id: {str(e)}")
            wafer_id = 'UNKNOWN'


        logger.info(f"Extracted wafer info - Wafer ID: {wafer_id}, Lot ID: {lot_id}")
        return (wafer_id, lot_id)

    def analyze_directory_structure(self, base_path: str) -> List[str]:
        """
        Processes the directory structure to identify wafer and die folders.

        Args:
            base_path (str): The base directory path to search for wafers and dies.

        Returns:
            List[str]: A list of paths to wafer directories or the base path if a single wafer is detected.

        Raises:
            FileNotFoundError: If the base path does not exist.
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"The base path '{base_path}' does not exist.")

        # single wafer case
        die_folders = [f for f in os.listdir(base_path)
                    if f.startswith('Die') and
                    os.path.isdir(os.path.join(base_path, f))]
        if die_folders:
            return [base_path]
        # If we found die folders, we're in the single wafer case
        else:
            wafer_candidates = [os.path.join(base_path, f) for f in os.listdir(base_path)
                                if os.path.isdir(os.path.join(base_path, f))]
            return wafer_candidates

    def scan_structure(self, wafer_path: str) -> None:
        """
        Scan directory structure to identify dies and devices.
        Sets total_dies and total_devices based on found files.
        """
        # Find all die folders
        self.die_folders = [f for f in os.listdir(wafer_path)
                          if f.startswith('Die') and
                          os.path.isdir(os.path.join(wafer_path, f))]
        self.total_dies = len(self.die_folders)
        logger.debug(f"Found {self.total_dies} die folders")

        # Get device list from first die
        first_die_path = os.path.join(wafer_path, self.die_folders[0])
        self.device_list = [f for f in os.listdir(first_die_path)
                        if '_Svg' not in f and
                        '.noi' in f and 'temp' not in f and
                        os.path.isfile(os.path.join(first_die_path, f))]
        self.total_devices = len(self.device_list)
        logger.debug(f"Found {self.total_devices} devices to process")

    def update_device_list(self, wafer_path: str):
        """
        Update the device list with device's width and length, also get the wafer_id and lot_id.

        Args:
            wafer_path (str): The path to the wafer directory.

        Returns:
            None

        Raises:
            ValueError: If the width or length is not found in the device name or error occurs during code execution.
        """
        try:
            for idx, device in enumerate(self.device_list):
                wafer_info = parse_device_info(os.path.join(wafer_path, device)) # the joint path here has no actual meaning, just for extract infomation

                # Check if wafer_id and lot_id are already set
                if not self.wafer_id and not self.lot_id:
                    if not wafer_info['lot_id']:
                        # if the lot_id is not found in the path, fallback to manual input
                        logger.warning("Could not find lot ID in the path, fallback to manual input")
                        self.lot_id = get_user_input(
                            'Lot ID Input',
                            'Please input lot ID:',
                            validate_lot_id
                        )
                        # if the get_user_input is failed, lot_id falls back to 'UNKNOWN' to make sure the program can run
                        if self.lot_id is None:
                            self.lot_id = 'UNKNOWN'
                    else:
                        self.lot_id = wafer_info['lot_id']

                    if not wafer_info['wafer_id']:
                        # if the wafer_id is not found in the path, fallback to manual input
                        logger.warning("Could not find wafer ID in the path, fallback to manual input")
                        self.wafer_id = get_user_input(
                            'Wafer ID Input',
                            'Please input wafer ID:',
                            validate_wafer_id
                        )

                        if self.wafer_id is None:
                            self.wafer_id = 'UNKNOWN'
                    else:
                        self.wafer_id = wafer_info['wafer_id']

                    logger.info(f"Wafer ID: {self.wafer_id}, Lot ID is: {self.lot_id}")

                # If auto_size is False, get width and length from user input
                if not self.config.auto_size:
                    width = get_user_input(
                        f'{device} - Width Input',
                        f'{device}\nPlease input device width:',
                        validate_width_length
                    )
                    length = get_user_input(
                        f'{device} - Length Input',
                        f'{device}\nPlease input device length:',
                        validate_width_length
                    )
                    if width is None or length is None:
                        logger.warning(f"Skipping device {device} due to invalid width or length")
                        raise ValueError("Must input valid width and length")
                else:
                    # Extract width and length from device name
                    width = wafer_info['width']
                    length = wafer_info['length']

                width = float(width)
                length = float(length)
                self.device_list[idx] = (device, width, length)

        except Exception as e:
            logger.error(f"Error extracting wafer width and length: {str(e)}")
            raise ValueError("Fail to extract width and length")

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
            # In raw data: [Measured Data] (Vrd, Ird, Vg, Ig, Vs, Is, Vb, Ib, Beta, Gm, Gd, tRd, tRb, fRd, fRb, Amp, Gain, Vn_In, Sampling Noise...)
            keywords = ['number', 'header']
            found_keywords = set()
            for idx, line in enumerate(content):
                line = line.strip()
                if line.startswith("Noise point Number"):
                    num_of_frequency_points = int(line.split('=', 1)[1].strip())
                    found_keywords.add('number')
                    logger.debug(f"Found {num_of_frequency_points} frequency points")
                if line.startswith("[Measured Data]"):
                    header_line_index = idx
                    found_keywords.add('header')

                if found_keywords == set(keywords):
                    break
        except Exception as e:
            logger.error(f"Error parsing frequency points: {str(e)}")
            raise ValueError(f"Error capturing data structure information --> {e}")

        # Extract header and data tables
        header = content[header_line_index].replace('[Measured Data]', '').strip().replace(' ','').strip('()').split(',')
        self.id_idx = header.index('Ird')
        self.gm_idx = header.index('Gm')
        self.vd_idx = header.index('Vrd')
        self.vg_idx = header.index('Vg')
        self.tRd_idx = header.index('tRd')

        try:
            # Extract bias data
            bias_list = []
            for i in range(1, 101): # Why some files have more than 20 bias conditions? :-(
                current_line = content[header_line_index + i].strip().split(",")
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
                current_line = content[header_line_index + 1 + i + j].strip().split(",")
                noise_data.append([float(m) for m in current_line])
            logger.debug(f"Extracted noise data with {len(noise_data)} points")

        except Exception as e:
            logger.error(f"Error extracting noise data: {str(e)}")
            raise ValueError(f"Error capturing noise-frequency table --> {e}")

        logger.info(f"Successfully extracted data from {file}")
        return bias_list, noise_data

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
        # self.check_bias_mismatch(0, data)  # Check Vd
        # self.check_bias_mismatch(2, data)  # Check Vg

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
        num_biases = len(position_data[0][0]) - 1
        logger.debug(f"Processing {num_freq_points} frequency points with {num_biases} conditions")

        result = []
        for bias_idx in range(num_biases):
            condition_data = []
            for freq_idx in range(num_freq_points):
                freq = position_data[0][freq_idx][0]
                pos_data = [pos[freq_idx][bias_idx + 1] for pos in position_data]
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

    def process_single_device(self, device_param):
        """
        Process a single device's data.

        Args:
            device_name: Name of device to process

        Returns:
            str: Processing time in seconds
        """
        device_name, width, length = device_param
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
                    bias_list, noise_list = self.get_data_from_raw(dut_path)
                    bias_table.append(bias_list)
                    noise_table.append(noise_list)
                else:
                    logger.info(f"{die} cannot be found!! making dummy data")
                    bias_list, noise_list = [[None for _ in row] for row in bias_list], [[None for _ in row] for row in noise_list]
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
                prefix = ['Id (A)', 'gm (S)', 'Vd (V)', 'Vg (V)', 'tRd (s)', 'Width (um)', 'Length (um)']
                selected_columns = list(zip(*[(row[self.id_idx], row[self.gm_idx], row[self.vd_idx], row[self.vg_idx], row[self.tRd_idx], width, length) for row in p1]))
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
                    sid = row[1:]

                    # Calculate normalized Id statistics - handle None values
                    id2 = []
                    for x, y in zip(sid, part1[0][1:]):
                        if x is None or y is None:
                            id2.append(None)
                        else:
                            id2.append(self.get_normalised(x, y))

                    # Calculate Gm statistics - handle None values
                    svg = []
                    for x, y in zip(sid, part1[1][1:]):
                        if x is None or y is None:
                            svg.append(None)
                        else:
                            svg.append(self.get_normalised(x, y))

                    # Calculate frequency-dependent statistics - handle None values
                    sid_f = []
                    for x in sid:
                        if x is None or row[0] is None:
                            sid_f.append(None)
                        else:
                            sid_f.append(x * row[0])

                    # Calculate normalized Svg - handle None values
                    svg_norm = []
                    for x in svg:
                        if x is None or width is None or length is None:
                            svg_norm.append(None)
                        else:
                            svg_norm.append(x * width * length)

                    # Combine all data
                    row.extend(id2 + svg + sid_f + svg_norm)

                    # Calculate statistics while filtering out None values
                    valid_sid = [x for x in sid if x is not None]
                    valid_id2 = [x for x in id2 if x is not None]
                    valid_svg = [x for x in svg if x is not None]
                    valid_sid_f = [x for x in sid_f if x is not None]
                    valid_svg_norm = [x for x in svg_norm if x is not None]

                    # Extend row with statistics, using None if no valid values exist
                    row.extend([
                        statistics.median(valid_sid) if valid_sid else None,
                        min(valid_sid) if valid_sid else None,
                        max(valid_sid) if valid_sid else None,
                        statistics.median(valid_id2) if valid_id2 else None,
                        min(valid_id2) if valid_id2 else None,
                        max(valid_id2) if valid_id2 else None,
                        statistics.median(valid_svg) if valid_svg else None,
                        min(valid_svg) if valid_svg else None,
                        max(valid_svg) if valid_svg else None,
                        statistics.median(valid_sid_f) if valid_sid_f else None,
                        min(valid_sid_f) if valid_sid_f else None,
                        max(valid_sid_f) if valid_sid_f else None,
                        statistics.median(valid_svg_norm) if valid_svg_norm else None,
                        min(valid_svg_norm) if valid_svg_norm else None,
                        max(valid_svg_norm) if valid_svg_norm else None
                    ])

                # Create complete headers
                part2_header = (
                    part1_header +
                    [f"{col}_Sid/id^2" for col in die_prefix] +
                    [f"{col}_Svg" for col in die_prefix] +
                    [f"{col}_Sid*f" for col in die_prefix] +
                    [f"{col}_Svg_norm" for col in die_prefix] +
                    ['Sid_med', 'Sid_min', 'Sid_max',
                     'Sid/id^2_med', 'Sid/id^2_min', 'Sid/id^2_max',
                     'Svg_med', 'Svg_min', 'Svg_max',
                     'Sid*f_med', 'Sid*f_min', 'Sid*f_max',
                     'Svg_norm_med', 'Svg_norm_min', 'Svg_norm_max']
                )
                df_part2 = pd.DataFrame(p2, columns=part2_header)
                valid_noise_list_ori[sheet_name] = (df_part1, df_part2)

            # Export results
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

                # Convert all columns to float to avoid pd.cancat featurewarning
                for col in df_part1.columns[1:]:
                    df_part1[col] = pd.to_numeric(df_part1[col], errors='coerce')
                df_part2 = df_part2.apply(pd.to_numeric, errors='coerce')

                # Combine and export data
                df_noise_list_ori = pd.concat([df_part1, df_part2], axis=0)
                output_file = os.path.join(
                    self.config.output_path,
                    f"{device_name[:-4]}_{self.lot_id}_W#{self.wafer_id}_{sheet_name}.xlsx"
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

            # Return processing time
            total_time = f"{time.perf_counter()-start_time:2f}"
            logger.info(f"Device processing completed in {total_time} seconds")
            return total_time

        except Exception as e:
            logger.error(f"Error processing device {device_name}: {str(e)}")
            raise

    def process_devices(self, wafer_path: str):
        """
        Process all devices in parallel.
        Manages worker processes and tracks progress.
        """
        freeze_support()
        logger.info("Starting parallel processing of all devices")
        try:
            self.reset_parameters() # clear any previous infomation
            self.config.base_path = wafer_path
            self.scan_structure(wafer_path) # wafer_path = D:\PythonProject\test_data\multiwafer\7ABC12345_W07
            self.update_device_list(wafer_path)

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
                        logger.debug(f"Device {device[0][:-4]} completed in {processing_time:.2f} seconds")
                    except Exception as e:
                        logger.error(f"Device {device[0][:-4]} failed with error: {str(e)}")
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
            logger.info(f"Fastest device: {fastest_device[0][:-4]} ({min_time:.2f} seconds)")
            logger.info(f"Slowest device: {slowest_device[0][:-4]} ({max_time:.2f} seconds)")
            logger.info("All devices processed successfully\n\n")
            self.reset_parameters()
            return

        except Exception as e:
            logger.error(f"Error in process devices: {str(e)}")
            raise

    def run(self):
        """
        Process all devices across multiple wafers sequentially.
        Enhanced version of process_devices that can handle multiple wafers
        by iterating through each wafer and calling process_devices() for each one.
        """
        freeze_support()
        logger.info("Starting processing of all devices across multiple wafers")

        wafer_paths = self.analyze_directory_structure(self.config.base_path)
        for wafer_path in wafer_paths:
            logger.info(f"Processing devices on wafer: {wafer_path}")
            self.process_devices(wafer_path)

