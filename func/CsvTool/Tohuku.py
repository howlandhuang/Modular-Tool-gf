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

def calculate_sensitivity(slope: float, bias: float, resistance: float, is_voltage_sweep: bool) -> Tuple[float, float]:
    """
    Calculate sensitivity values Si and Sv
    Args:
        slope: Slope from linear regression
        bias: Bias value
        resistance: Resistance value
        is_voltage_sweep: True if voltage sweep, False if current sweep
    Returns:
        tuple: (Si, Sv) sensitivity values
    """
    if is_voltage_sweep:
        S_v = abs(slope / bias) * 1e3                                   # mV / (V*T)
        S_i = S_v * resistance / 1e3                                    # V / (A*T)
    else:
        S_i = abs(slope / bias) * 1e6                                   # V / (A*T)
        S_v = S_i / resistance * 1e3                                    # mV / (V*T)
    return S_i, S_v

def robust_lr(x: list, y: list, max_outliers: int = 3, r2_threshold: float = 0.9) -> Tuple[float, float, float]:
    """
    Calculate linear regression with outlier removal
    Args:
        x: x-axis values (magnetic field)
        y: y-axis values (voltage differences)
        max_outliers: maximum number of outliers to remove
        r2_threshold: R² threshold for attempting outlier removal
    Returns:
        tuple: (slope, intercept, R2) of the robust linear regression
    """
    # First try normal linear regression
    slope, intercept, r2 = lr(x, y)

    # If R² is good enough or too many potential outliers, return original results
    if r2 >= r2_threshold or len(x) - max_outliers < 4:
        return slope, intercept, r2

    # Calculate residuals and z-scores
    y_pred = np.array([slope * xi + intercept for xi in x])
    residuals = np.array(y) - y_pred
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))

    # Get indices of inliers (points with lowest z-scores)
    min_points = len(x) - max_outliers
    inlier_indices = np.argsort(z_scores)[:min_points]

    # Perform linear regression on filtered data
    x_filtered = np.array(x)[inlier_indices]
    y_filtered = np.array(y)[inlier_indices]
    return lr(x_filtered, y_filtered)

def process_file(file_path: str, metadata: Dict[str, str], temperature: str) -> Optional[List[List[Union[str, float]]]]:
    """
    Process a single measurement file and extract relevant data
    Args:
        file_path: Path to the measurement file
        temperature: Temperature value for measurements
        metadata: Dictionary containing Lot_id, Wafer_id, TK_id, DUT_id
    Returns:
        list: Processed measurement data or None if file is invalid
    """
    # Magnetic field voltage lookup table
    MFLD_V_table = {
        -0.2736: -2000,
        -0.2394: -1750,
        -0.2052: -1500,
        -0.1710: -1250,
        -0.1368: -1000,
        -0.1026: -750,
        -0.0684: -500,
        -0.0342: -250,
         0.0000: 0,
         0.0342: 250,
         0.0684: 500,
         0.1026: 750,
         0.1368: 1000,
         0.1710: 1250,
         0.2052: 1500,
         0.2394: 1750,
         0.2736: 2000
    }

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

            # Extract timestamp
            time_stamp = content[1].strip().replace('\t', ' ')

        except ValueError as e:
            print(f"Error parsing metadata in {file_name}: {str(e)}")
            return None

        # Process bias and magnetic field values
        bias_list = []
        mfld_list: Dict[int, int] = {}

        for z in range(7, len(content)):
            line = content[z].strip().split(",")
            if not line: # Basic validation
                continue

            try:
                bias = float(line[5])
                mfld = float(line[6])

                if bias not in bias_list:
                    bias_list.append(bias)
                # Record each Mfld value
                voltage = MFLD_V_table.get(round(mfld, 4))
                if voltage is not None:
                    mfld_list[int(voltage)] = z
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

                        # Extract voltage measurements with validation
                        current_line_data = content[line_index].split(',')
                        next_line_data = content[line_index + 1].split(',')
                        last_line_data = content[line_index + 2].split(',')

                        Ref = abs(float(current_line_data[test_column + 2])) + 1e-16
                        V1 = float(next_line_data[test_column + 2])
                        V2 = float(last_line_data[test_column + 2])
                        delta_V = round(V1-V2, 6)
                        delta_V_list.append(delta_V)

                        # Calculate resistance
                        Res = bias / Ref if is_voltage_sweep else Ref / bias * 1e6 # bcs when bias current, unit is uA

                        # Store measurement data
                        result_list.append([
                            metadata['TK_id'], metadata['DUT_id'], f"{metadata['TK_id']}_{metadata['DUT_id']}",
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
                slope_lr, intercept, R2 = robust_lr(lr_x, lr_y)

                # Calculate predicted values and residuals
                pred_y = [slope_lr*i + intercept for i in lr_x]
                last_mfld_group_rows = -1 * len(delta_V_list)
                residual_val = [
                    result_list[last_mfld_group_rows:][j][-3] - pred_y[j]
                    for j in range(len(delta_V_list))
                ]

                # Calculate sensitivities
                resistance = float(result_list[-4][-2])
                S_i, S_v = calculate_sensitivity(slope_lr, bias, resistance, is_voltage_sweep)

                # Update measurement data with calculated values
                for idx, result in enumerate(result_list[last_mfld_group_rows:]):
                    result.extend([
                        S_i, S_v, slope_lr, R2,
                        pred_y[idx], residual_val[idx]
                    ])

        return result_list

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def process_folder(folder_path: str, device_name: str, temperature: str) -> List[List[Union[str, float]]]:
    """
    Process all CSV files in a folder structure
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
            if data := process_file(file_path, device_name, temperature):
                result.extend(data)
    return result

def main(folder_path: str, output_file_path: str, manual_temperature: Optional[str] = None) -> None:
    """
    Process all measurement files in a folder and generate combined output.
    Args:
        folder_path: Path to folder containing measurement files
        output_file_path: Path for output CSV file
        manual_temperature: Temperature value (e.g. 'n20C'/'25C'/'0C')
    """
    try:
        result = []
        folder_path = os.path.normpath(folder_path)
        output_file_path = os.path.normpath(output_file_path)
        print(f"Starting to process folder: {folder_path}")

        if manual_temperature is None:
            print("Warning: No temperature specified for single folder structure")
            manual_temperature = 'UNKNOWN'  # Default temperature
        print(f"Default temperature: {manual_temperature}")

        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        info_pattern = re.compile(r"^(?P<TK_id>^tk\w+)(?P<DUT_id>dut\d+(re)?$)", re.IGNORECASE)

        device_info_dict = {}
        for d in subdirs:
            if match := info_pattern.match(d):
                device_info_dict[d] = {
                    'TK_id': match.group('TK_id'),
                    'DUT_id': match.group('DUT_id')
                }

        if device_info_dict:
            '''
            -tk01dut1
            |----abcd.csv
            |----abcd(2).csv
            -tk02dut3
            |----abcd.csv
            |----abcd(2).csv
            '''
            for device_folder, device_info in device_info_dict.items():
                device_path = os.path.join(folder_path, device_folder)
                print(f"Processing device folder: {device_folder}")

                if os.path.isdir(device_path):
                    folder_data = process_folder(device_path, device_info, manual_temperature)
                    result.extend(folder_data)
        else:
            '''
            -xxxx
            |----abcd.csv
            |----abcd(2).csv
            |----abcd(3).csv
            |----abcd(4).csv
            '''
            device_info = {
                'TK_id': input("Please input TK id"),
                'DUT_id': input("Please input DUT id")
            }
            folder_data = process_folder(folder_path, device_info, manual_temperature)
            result.extend(folder_data)

        # Write results if we have any
        if result:
            with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
                fw = csv.writer(f, delimiter=',', lineterminator='\n')

                header = [
                    "TK id", "DUT id", "TK_DUT",
                    "Temperature", "X", "Y", "Site", "Time", "Readout",
                    "Bias(V or uA)", "Mfld(T)", "Ref(V or A)", "V1(V)", "V2(V)",
                    "delta(V)", "Resistance(Ohm)", "Resistance(kOhm)",
                    "Si(V/A*T)", "Sv(mV/V*T)", "Slope", "R square",
                    "Predict voltage (V)", "Residual voltage (V)"
                ]
                fw.writerow(header)
                fw.writerows(result)
            print(f"Successfully processed {len(result)} measurements")
            print(f"Results written to: {output_file_path}")
        else:
            print("No valid measurements found to process!")

    except Exception as e:
        print(f"Error in main processing: {str(e)}")


if __name__ == '__main__':
    """
    For extracting Tohoku results
    Args:
        folder_path: Absolute path of directory containing measurement files
        output_file_path: Output file path in CSV format
        manual_temperature: Temperature value (e.g. 'n20C'/'25C'/'0C')
    """
    # Configuration
    folder_path = r"C:\Users\hhuang10\Documents\Local_Data\Tohoku\7XYA22665w01"
    output_file_path = r"C:\Users\hhuang10\Documents\Local_Data\Tohoku\7XYA22665w01\Tohoku_demo.csv"

    # Run main processing
    main(folder_path, output_file_path, manual_temperature='25C')
    print("DONE")