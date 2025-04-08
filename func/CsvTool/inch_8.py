"""
CSV Tool module for 8-inch wafer data processing.
Provides functionality to extract and process measurement data from raw text files.
Generates CSV outputs with calculated sensitivity values.
"""

import os, re, csv, logging
import numpy as np
import pandas as pd
from collections import defaultdict
from func.ulti import CsvProcessingConfig, lr
from typing import Dict, List, Tuple, Optional, Any

# Initialize module logger
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processor class for extracting and analyzing 8-inch wafer measurement data.
    Handles data parsing, processing, and CSV generation.
    """

    def __init__(self, config: CsvProcessingConfig):
        """
        Initialize CSV data processor with configuration.

        Args:
            config: CsvProcessingConfig object containing processing parameters
        """
        logger.info("Initializing CsvDataProcessor")
        self.config = config
        self.reset_parameters()
        logger.debug("CsvDataProcessor initialized successfully")

    def reset_parameters(self):
        """Reset all processing parameters to initial state."""
        logger.debug("Resetting processing parameters")
        self.metadata = {}
        self.current_wafer_id = None
        self.die_positions = []
        self.unsorted_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))))
        self.sorted_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))))
        self.csv_row_data = []
        logger.debug("Processing parameters reset complete")

    def read_input_file(self, input_file_path: str) -> List[str]:
        """
        Read content from input file.

        Args:
            input_file_path: Specific file path to read

        Returns:
            List[str]: File content as list of lines

        Raises:
            FileNotFoundError: If file cannot be read
        """

        logger.debug(f"Reading input file: {input_file_path}")
        try:
            with open(input_file_path, "r") as f:
                content = f.readlines()
            logger.info(f"Successfully read content from input file: {input_file_path}")
            return content
        except FileNotFoundError:
            logger.error(f"Error: Input file {input_file_path} not found")
            raise FileNotFoundError(f"Error: Input file {input_file_path} not found")
        except Exception as e:
            logger.error(f"Error reading file {input_file_path}: {str(e)}")
            raise

    def parse_measurement_data(self, parameter_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse measurement parameter string to extract relevant information.

        Args:
            parameter_str: String containing measurement parameters

        Returns:
            Dict: Parsed information if match found, None otherwise
        """
        logger.debug(f"Parsing measurement data: {parameter_str}...")
        pattern = re.compile(r"(?P<tk_id>\w+)\_(?P<dut_id>\w+)\_(?P<mfld>\w+[T]).*?~(?P<type>\w+)\_(?P<terminal>\w+)\_(?P<bias>\d+\w+)")
        match = pattern.match(parameter_str)
        if match:
            result = {
                'tk_id': match.group('tk_id'),
                'dut_id': match.group('dut_id'),
                'terminal': match.group('terminal').replace("V",""),
                'bias': match.group('bias'),
                'mfld': float(match.group('mfld').replace('n', '-').replace('p', '.').replace('T', '')),
                'measure_type': 'V' if 'V' in match.group('type') else 'R'
            }
            logger.debug(f"Successfully parsed measurement data: {result}")
            return result
        logger.debug("No match found in parameter string")
        return None

    def convert_bias(self, bias_value: str) -> float:
        """
        Convert bias value to microamperes (uA) or volts.

        Args:
            bias_value: String containing bias value with unit

        Returns:
            float: Converted bias value
        """
        logger.debug(f"Converting bias value: {bias_value}")
        try:
            if "A" in bias_value:
                if "u" in bias_value:
                    return float(bias_value.replace('uA', ''))
                elif "m" in bias_value:
                    return float(bias_value.replace('mA', '')) * 1e3
            elif "V" in bias_value:
                return float(bias_value.replace('V', ''))
            logger.warning(f"Unknown bias unit in {bias_value}, returning 0.0")
            return 0.0
        except Exception as e:
            logger.error(f"Error converting bias value {bias_value}: {str(e)}")
            return 0.0

    def calculate_sensitivity(self, slope_lr: float, bias_value: str, resistance: float) -> Tuple[float, float]:
        """
        Calculate Si and Sv sensitivity values.

        Args:
            slope_lr: Slope from linear regression
            bias_value: Bias value string
            resistance: Resistance value

        Returns:
            tuple: (Si, Sv) sensitivity values
        """
        logger.debug(f"Calculating sensitivity for slope={slope_lr}, bias={bias_value}, resistance={resistance}")
        try:
            bias_numerical = self.convert_bias(bias_value)

            if bias_numerical == 0:
                logger.warning("Zero bias value, cannot calculate sensitivity")
                return 0.0, 0.0

            if "A" in bias_value:
                S_i = abs(slope_lr / bias_numerical) * 1e6                     # V / (A*T)
                S_v = S_i / float(resistance) * 1e3 if resistance != 0 else 0  # mV / (V*T)
            elif "V" in bias_value:
                S_v = abs(slope_lr / bias_numerical) * 1e3                     # mV / (V*T)
                S_i = S_v * float(resistance) / 1e3                            # V / (A*T)
            else:
                logger.warning(f"Unknown bias unit in {bias_value}")
                return 0.0, 0.0

            logger.debug(f"Calculated sensitivities: S_i={S_i:.6f}, S_v={S_v:.6f}")
            return S_i, S_v
        except Exception as e:
            logger.error(f"Error calculating sensitivity: {str(e)}")
            return 0.0, 0.0

    def process_file_content(self, content: List[str]) -> None:
        """
        Process content of input file and extract data.

        Args:
            content: List of lines from input file
        """
        logger.info("Processing file content")
        try:
            for line in content:
                x = re.sub(r"(?<=[\[{(])\s+|\s+(?=[\]})])", '', " ".join(line.split()))

                # Extract metadata
                if match := re.findall(r"^Lot ID:\s+(.*)", x):
                    self.metadata['Lot_ID'] = match[0]
                    logger.debug(f"Found Lot ID: {match[0]}")
                    continue
                if match := re.findall(r"^Product ID:\s+(.*)", x):
                    self.metadata['Product_ID'] = match[0]
                    logger.debug(f"Found Product ID: {match[0]}")
                    continue
                if match := re.findall(r"^Test ID:\s+(.*)", x):
                    self.metadata['Test_ID'] = match[0]
                    logger.debug(f"Found Test ID: {match[0]}")
                    continue
                if match := re.findall(r"^Date:\s+(.*)", x):
                    self.metadata['Time_stamp'] = match[0]
                    logger.debug(f"Found Timestamp: {match[0]}")
                    continue
                if match := re.findall(r"= Wafer ID:\s+(.*) =", x):
                    self.current_wafer_id = match[0]
                    logger.debug(f"Found Wafer ID: {match[0]}")
                    continue

                # Process die positions
                if "Parameter Unit" in x:
                    if die_pos_matches := re.findall(r"\(\d+,\s\d+\)", x):
                        self.die_positions = [pos.replace(' ', '') for pos in die_pos_matches]
                        logger.debug(f"Found {len(self.die_positions)} die positions")
                        continue

                # Process measurement data
                if parsed_data := self.parse_measurement_data(x):
                    data_values = re.findall(r"([+-]\d+\.\d+[Ee][+-]\d+)", x)

                    for j, die_pos in enumerate(self.die_positions):
                        if j < len(data_values):
                            self.unsorted_tree[self.current_wafer_id][die_pos][parsed_data['tk_id']][parsed_data['dut_id']][parsed_data['terminal']][parsed_data['bias']][parsed_data['mfld']][parsed_data['measure_type']] = float(data_values[j])

            logger.info("File content processing completed")
        except Exception as e:
            logger.error(f"Error processing file content: {str(e)}")
            raise

    def sort_data(self) -> None:
        """Sort extracted data by magnetic field."""
        logger.info("Sorting extracted data")
        try:
            for Wafer_id, TK_id_dict in self.unsorted_tree.items():
                for Die_pos, DUT_id_dict in TK_id_dict.items():
                    for TK_id, DUT_site_dict in DUT_id_dict.items():
                        for DUT_id, Terminal_dict in DUT_site_dict.items():
                            for Terminal, Bias_value_dict in Terminal_dict.items():
                                for Bias_value, Mag_field_dict in Bias_value_dict.items():
                                    Sorted_mfld_dict = dict(sorted(Mag_field_dict.items()))
                                    for Mag_field, Measure_dict in Sorted_mfld_dict.items():
                                        for Type, Value in Measure_dict.items():
                                            self.sorted_tree[Wafer_id][Die_pos][TK_id][DUT_id][Terminal][Bias_value][Mag_field][Type] = Value

            logger.info("Data sorting completed")
        except Exception as e:
            logger.error(f"Error sorting data: {str(e)}")
            raise

    def prepare_csv_data(self) -> None:
        """Prepare data for CSV output and calculate sensitivities."""
        logger.info("Preparing CSV data and calculating sensitivities")
        try:
            self.csv_row_data = []
            for Wafer_id, TK_id_dict in self.sorted_tree.items():
                for Die_pos, DUT_id_dict in TK_id_dict.items():
                    x, y = map(int, Die_pos.strip("()").split(","))

                    for TK_id, DUT_site_dict in DUT_id_dict.items():
                        for DUT_id, Terminal_dict in DUT_site_dict.items():
                            for Terminal, Bias_value_dict in Terminal_dict.items():
                                for Bias_value, Mag_field_dict in Bias_value_dict.items():
                                    # Process each magnetic field measurement
                                    for Mag_field in Mag_field_dict.keys():
                                        # Get voltage and resistance values
                                        Voltage = float(self.sorted_tree[Wafer_id][Die_pos][TK_id][DUT_id][Terminal][Bias_value][Mag_field]["V"]) if "ad" not in Terminal else None

                                        if "R" in self.sorted_tree[Wafer_id][Die_pos][TK_id][DUT_id][Terminal][Bias_value][Mag_field]:
                                            Res = float(self.sorted_tree[Wafer_id][Die_pos][TK_id][DUT_id][Terminal][Bias_value][Mag_field]["R"])
                                        else:
                                            try:
                                                Res = float(self.sorted_tree[Wafer_id][Die_pos][TK_id][DUT_id][Terminal[::-1]][Bias_value][Mag_field]["R"])
                                            except KeyError:
                                                logger.warning(f"Resistance data not found for terminal {Terminal} or {Terminal[::-1]}")
                                                Res = 0.0

                                        # Create row data
                                        self.csv_row_data.append([
                                            Wafer_id, x, y, TK_id, DUT_id,
                                            f"{TK_id}_{DUT_id}", Bias_value, Mag_field, Terminal,
                                            Voltage, Res, Res/1000
                                        ])

                                    # Skip sensitivity calculation for 'ad' terminal
                                    if Terminal == 'ad':
                                        logger.debug(f"Skipping sensitivity calculation for 'ad' terminal")
                                        continue

                                    # Calculate sensitivity
                                    magnetic_fields = np.array(self.config.magnetic_fields)

                                    # Make sure we have enough data points
                                    if len(self.csv_row_data) < len(magnetic_fields):
                                        logger.warning(f"Not enough data points for sensitivity calculation: {len(self.csv_row_data)} < {len(magnetic_fields)}")
                                        continue

                                    # Get the last N data points for this terminal/bias combination
                                    last_n_rows = self.csv_row_data[-len(magnetic_fields):]
                                    lr_y = np.array([x[-3] for x in last_n_rows])  # Extract voltage values

                                    slope_lr,_,_ = lr(magnetic_fields, lr_y)
                                    resistance = self.csv_row_data[-len(magnetic_fields)//2][-2]  # Middle data point's resistance

                                    S_i, S_v = self.calculate_sensitivity(slope_lr, Bias_value, resistance)

                                    # Add sensitivity values to all rows for this terminal/bias
                                    for result in self.csv_row_data[-len(magnetic_fields):]:
                                        result.extend([S_i, S_v])

            logger.info(f"CSV data preparation complete: {len(self.csv_row_data)} rows generated")
        except Exception as e:
            logger.error(f"Error preparing CSV data: {str(e)}")
            raise

    def write_csv_output(self, file_name: str) -> None:
        """Write processed data to CSV output file."""
        # Extract the filename without extension and create a proper output filename
        input_basename = os.path.basename(file_name)
        file_name_without_ext = os.path.splitext(input_basename)[0]
        output_filename = f"{file_name_without_ext}_output.csv"
        output_file_path = os.path.join(self.config.output_path, output_filename)

        logger.info(f"Writing data to CSV file: {output_file_path}")
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                fw = csv.writer(f, delimiter=',', lineterminator='\n')

                # Write metadata
                test_info = [
                    ["Lot ID", self.metadata.get('Lot_ID', 'Unknown')],
                    ["Product ID", self.metadata.get('Product_ID', 'Unknown')],
                    ["Test ID", self.metadata.get('Test_ID', 'Unknown')],
                    ["Time", self.metadata.get('Time_stamp', 'Unknown')],
                    []
                ]
                fw.writerows(test_info)

                # Write header and data
                header = [
                    "Wafer_ID", "X", "Y", "TK_ID", "DUT_ID", "TK_DUT",
                    "Bias", "Mfld(T)", "Terminal", "Voltage(V)", "Resistance(Ohm)", "Resistance(kOhm)",
                    "Si(V/A*T)", "Sv(mV/V*T)"
                ]
                fw.writerow(header)
                fw.writerows(self.csv_row_data)

            logger.info("CSV output successfully written")
        except Exception as e:
            logger.error(f"Error writing CSV output: {str(e)}")
            raise

    def process_single_file(self, input_file_path: str) -> None:
        """
        Process data from a single input file and generate output CSV.

        Args:
            input_file_path: Optional path to input file, defaults to config path
            output_file_path: Optional path to output file, defaults to config path
        """

        logger.info(f"Starting processing for file: {self.config.input_path}")

        try:
            # Reset parameters for this file
            self.reset_parameters()
            content = self.read_input_file(input_file_path)
            self.process_file_content(content)
            self.sort_data()
            self.prepare_csv_data()
            self.write_csv_output(input_file_path)

            logger.info(f"Processing of {self.config.input_path} completed successfully")
        except Exception as e:
            logger.error(f"Error during processing of {self.config.input_path}: {str(e)}")
            raise

    def run(self) -> None:
        """
        Run the processing pipeline for all input files.
        Handles both single file and multiple file inputs.
        """
        logger.info("Starting data processing")

        try:
            for file in self.config.input_path:
                logger.info(f"Processing single file: {file}")
                self.process_single_file(file)

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise
