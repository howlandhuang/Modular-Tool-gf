import os
import csv
import re
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

def lr(x: list, y: list) -> Tuple[float, float, float]:
    """
    Calculate linear regression parameters between x and y values
    Args:
        x: x-axis values (magnetic field)
        y: y-axis values (voltage differences)
    Returns:
        tuple: (slope, intercept, R2) of the linear regression
    """
    x = np.array(x)
    y = np.array(y)
    size = len(x)
    x_avg = np.sum(x) / size
    y_avg = np.sum(y) / size
    Sx = np.sum((x - x_avg) ** 2)
    Sxy = np.sum((x - x_avg) * (y - y_avg))
    slope = Sxy / Sx
    intercept = y_avg - slope * x_avg
    y_pred = slope * x + intercept
    SS_res = np.sum((y - y_pred) ** 2)
    SS_tot = np.sum((y - y_avg) ** 2)
    R2 = 1 - (SS_res / SS_tot)
    return slope, intercept, R2

def theil_sen_regression(x: list, y: list, max_subsets: int = 1000) -> Tuple[float, float, float]:
    """
    Theil-Sen robust regression estimator

    This method computes the median of slopes between all pairs of points.
    It's highly robust to outliers (breakdown point of 29.3%) and performs well
    with heteroscedastic data.

    Args:
        x: x-axis values (magnetic field)
        y: y-axis values (voltage differences)
        max_subsets: maximum number of point pairs to sample for large datasets

    Returns:
        tuple: (slope, intercept, R2) of the Theil-Sen regression
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    # For large datasets, use random sampling to limit computation
    if n > 30 and n * (n - 1) // 2 > max_subsets:
        import random
        # Generate random pairs of indices
        indices = []
        while len(indices) < max_subsets:
            i, j = random.sample(range(n), 2)
            if i != j and (i, j) not in indices and (j, i) not in indices:
                indices.append((i, j))
    else:
        # Use all pairs for smaller datasets
        indices = [(i, j) for i in range(n) for j in range(i+1, n)]

    # Calculate slopes for all point pairs
    slopes = []
    for i, j in indices:
        # Avoid division by zero
        if x[i] != x[j]:
            slope = (y[j] - y[i]) / (x[j] - x[i])
            slopes.append(slope)

    if not slopes:
        # If no valid slopes (e.g., all x values are identical), return zero slope
        return 0.0, np.median(y), 0.0

    # Median slope
    slope = np.median(slopes)

    # Calculate intercepts for each point and take the median
    intercepts = y - slope * x
    intercept = np.median(intercepts)

    return slope, intercept

def calculate_sensitivity(slope: float, bias: float, resistance: float, is_voltage_sweep: bool) -> Tuple[float, float]:
    """
    Calculate sensitivity values Si and Sv
    Args:
        slope: Slope from linear regression
        bias: Bias value, in V or uA
        resistance: Resistance value, in Ohm
        is_voltage_sweep: True if voltage sweep, False if current sweep
    Returns:
        tuple: (Si, Sv) sensitivity values
    """
    if is_voltage_sweep:
        S_v = abs(slope / bias) * 1e3                                   # mV / (V*T)
        S_i = S_v * resistance * 1e-3                                   # V / (A*T)
    else:
        S_i = abs(slope / (bias* 1e-6))                                 # V / (A*T)
        S_v = S_i / resistance * 1e3                                    # mV / (V*T)
    return S_i, S_v

def process_file(file_path: str, temperature: str) -> Optional[List[List[Union[str, float]]]]:
    """
    Process a single measurement file and extract relevant data
    Args:
        file_path: Path to the measurement file
        temperature: Temperature value for measurements
    Returns:
        list: Processed measurement data or None if file is invalid
    """
    try:
        with open(file_path, "r") as f:
            content = f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

    try:
        # Get file name for parsing metadata
        file_name = os.path.split(file_path)[1]

        # Parse metadata using regex
        try:
            # Extract coordinates from first line
            if match := re.search(r"^Die:\s*x(?P<x>\d+)\s*y(?P<y>\d+)$", content[0]):
                coor_x = match.group('x')
                coor_y = match.group('y')
                coor = f"x{coor_x.zfill(2)}y{coor_y.zfill(2)}"
            else:
                coor_x = "Unknown"
                coor_y = "Unknown"
                coor = "Unknown"

            # Extract timestamp from second line
            time_stamp = content[1].strip().replace('\t', ' ')

        except ValueError as e:
            print(f"Error parsing metadata in {file_name}: {str(e)}")
            return None

        # Process bias and magnetic field values
        bias_list = []
        mfld_list: Dict[int, int] = {}

        for z in range(5, len(content)):
            line = content[z].strip().split(",")
            if not line or len(line) < 7:  # Basic validation
                continue

            try:
                bias = float(line[5])
                mfld = float(line[6])

                if bias not in bias_list:
                    bias_list.append(bias)
                mfld_list[int(mfld)] = z
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid line in {file_name}: {str(e)}")
                continue

        # Clean and sort bias values
        bias_list = [round(float(x), 3) for x in bias_list]
        if 0 in bias_list:
            bias_list.remove(0)
        if not bias_list:
            print(f"No valid bias values found in {file_name}")
            return None

        is_voltage_sweep = bias_list[0] // 10 == 0

        # Sort and process magnetic field values
        mfld_list = dict(sorted(mfld_list.items()))
        if not mfld_list:
            print(f"No valid magnetic field values found in {file_name}")
            return None

        start_line = next(iter(mfld_list.values()))
        mfld_list_values = [x/1e4 for x in mfld_list.keys()]

        result_list = []

        # Process measurements for each bias and test column
        for i, bias in enumerate(bias_list):
            for test_column in range(3):
                delta_V_list = []

                # Process each magnetic field measurement
                for j, mfld in enumerate(mfld_list_values):
                    try:
                        line_index = start_line + i*3 + j*3*len(bias_list)

                        # Extract voltage measurements
                        current_line_data = content[line_index].split(',')
                        next_line_data = content[line_index + 1].split(',')
                        last_line_data = content[line_index + 2].split(',')

                        V1 = float(current_line_data[test_column + 2])
                        Ref = float(next_line_data[test_column + 2]) + 1e-16
                        V2 = float(last_line_data[test_column + 2])
                        delta_V = round(V1-V2, 6)
                        delta_V_list.append(delta_V)

                        # Calculate resistance
                        Res = bias / Ref if is_voltage_sweep else Ref / bias * 1e6 # bcs when bias current, unit is uA

                        # Store measurement data
                        result_list.append([
                            temperature, coor_x, coor_y, coor, time_stamp, f'Readout_{test_column+1}',
                            bias, mfld, Ref, V1, V2,
                            delta_V, Res, Res/1000
                        ])

                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing measurement at index {line_index} in {file_name}: {str(e)}")
                        continue

                # Calculate linear regression and sensitivities
                if len(delta_V_list) < 7:
                    print(f"Warning: Insufficient data points for linear regression in {file_name}")
                    continue

                lr_x = np.array(mfld_list_values[:len(delta_V_list)])
                lr_y = np.array(delta_V_list)
                slope_lr, intercept, R2 = lr(lr_x, lr_y)
                theil_sen_slope, theil_sen_intercept = theil_sen_regression(lr_x, lr_y)

                # Calculate predicted values and residuals
                pred_y = [slope_lr*i + intercept for i in lr_x]
                theil_sen_pred_y = [theil_sen_slope*i + theil_sen_intercept for i in lr_x]

                # Calculate sensitivities
                resistance = float(result_list[-4][12])
                S_i, S_v = calculate_sensitivity(slope_lr, bias, resistance, is_voltage_sweep)
                theil_sen_S_i, theil_sen_S_v = calculate_sensitivity(theil_sen_slope, bias, resistance, is_voltage_sweep)

                # Update measurement data with calculated values
                last_mfld_group_rows = -1 * len(delta_V_list)
                for idx, result in enumerate(result_list[last_mfld_group_rows:]):
                    result.extend([
                        S_i, S_v, slope_lr, pred_y[idx], R2,
                        theil_sen_S_i, theil_sen_S_v, theil_sen_slope, theil_sen_pred_y[idx]
                    ])

        return result_list

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def process_folder(folder_path: str, temperature: str) -> List[List[Union[str, float]]]:
    """
    Process all CSV files in a single folder
    Args:
        folder_path: Path to folder containing measurement files
        temperature: Temperature value for measurements
    Returns:
        List of processed measurement data
    """
    result = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.csv') and 'output' not in file.lower():
            file_path = os.path.join(folder_path, file)
            if data := process_file(file_path, temperature):
                result.extend(data)
    return result

def main(folder_path: str, output_file_path: str, manual_temperature: Optional[str] = None) -> None:
    """
    Process all measurement files in a folder and generate combined output.
    Handles both single folder and temperature-based folder structures.
    Args:
        folder_path: Path to folder containing measurement files
        output_file_path: Path for output CSV file
        manual_temperature: Default temperature if not found in filename
    """
    try:
        result = []

        # Check if we have temperature-based folders
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        temp_pattern = re.compile(r'^TEL_[nN]?\d+C$')
        temp_folders = [d for d in subdirs if temp_pattern.match(d)]

        if temp_folders:
            # Temperature-based folder structure
            print("Processing temperature-based folder structure...")

            # Process each temperature folder
            for temp_folder in temp_folders:
                # Extract temperature from folder name (e.g. TEL_n20C -> n20C)
                temperature = temp_folder.split('_')[1]
                folder_path_temp = os.path.join(folder_path, temp_folder)
                print(f"Processing temperature folder: {temp_folder}")

                if folder_data := process_folder(folder_path_temp, temperature):  # Only add valid results
                    result.extend(folder_data)

        else:
            # Single folder structure
            print("Processing single folder structure...")
            if manual_temperature is None:
                print("Warning: No temperature specified for single folder structure")
                manual_temperature = 'UNKNOWN'  # Default temperature
            if folder_data := process_folder(folder_path, manual_temperature):  # Only add valid results
                    result.extend(folder_data)


        # Write output file if we have results
        if result:
            with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
                fw = csv.writer(f, delimiter=',', lineterminator='\n')

                header = [
                    "Temperature", "X", "Y", "Site", "Time", "Readout",
                    "Bias(V or uA)", "Mfld(T)", "Vref(V)", "V1(V)", "V2(V)", "delta(V)", "Resistance(Ohm)", "Resistance(kOhm)",
                    "Si(V/A*T)", "Sv(mV/V*T)", "Slope", "Predict voltage (V)", "R square",
                    "Theil-Sen Si(V/A*T)", "Theil-Sen Sv(mV/V*T)", "Theil-Sen Slope", "Theil-Sen Predict voltage (V)"
                ]
                fw.writerow(header)
                fw.writerows(result)

            print(f"Successfully processed {len(result)} measurements!")
            print(f"Output written to: {output_file_path}")
        else:
            print("No valid measurements found to process!")

    except Exception as e:
        print(f"Error in main processing: {str(e)}")
        return

if __name__ == '__main__':
    """
    For extracting TEL results
    Args:
        folder_path: Absolute path of directory containing TEL results
        output_file_path: Output file path in CSV format
        manual_temperature: Temperature value (e.g. 'n20C'/'25C'/'0C')
    """
    # Configuration
    folder_path = r"C:\Users\hhuang10\Documents\Local_Data\MPW1519\W6"
    output_file_path = r"C:\Users\hhuang10\Documents\Local_Data\MPW1519\W6\W6_result_with_robust.csv"

    # Run main processing
    main(folder_path, output_file_path, manual_temperature='125C')