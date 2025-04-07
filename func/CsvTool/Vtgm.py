import os
import csv
import re
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np

# Constants
PMOS_MARKER_MULTIPLIER = -70E-09
NMOS_MARKER_MULTIPLIER = 300E-09

def lr(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """
    Calculate linear regression parameters between x and y values
    Args:
        x: x-axis values
        y: y-axis values
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

def find_nearest(input_list: List[float], target: float) -> Tuple[int, int]:
    """
    Find nearest values in sorted list to target value.
    Args:
        input_list: List of values to search
        target: Target value to find nearest neighbors
    Returns:
        Tuple of indices for left and right nearest neighbors
    """
    if not input_list:
        return -1, -1

    ls = sorted(input_list)
    if max(ls) < target or min(ls) > target:
        return -1, -1

    if len(ls) == 1:
        return 0, 0
    if ls[0] == target:
        return 0, 1
    if ls[-1] == target:
        return len(ls)-2, len(ls)-1

    left, right = 0, len(ls)-1
    while left <= right:
        mid = (left + right) // 2
        if ls[mid] == target:
            left = mid - 1
            right = mid + 1
            break
        elif ls[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    res_left = ls[right] if right >= 0 else -1
    res_right = ls[left] if left < len(ls) else -1

    if res_left == -1 or res_right == -1:
        return -1, -1

    if res_left < res_right:
        return (right, left) if input_list[0] == ls[0] else (len(input_list)-1-left, len(input_list)-1-right)
    else:
        return (left, right) if input_list[0] == ls[0] else (len(input_list)-1-right, len(input_list)-1-left)

def fit_linear(id_list: List[float], vg_list: List[float], marker: float) -> float:
    """
    Calculate linear fit and return interpolated value.
    Args:
        id_list: List of current values
        vg_list: List of voltage values
        marker: Target current value
    Returns:
        Interpolated voltage value
    """
    if len(id_list) != len(vg_list):
        raise ValueError("Id Vg data point mismatch!")

    left, right = find_nearest(id_list, marker)
    if left == right == -1:
        return -9

    k = (id_list[right] - id_list[left]) / (vg_list[right] - vg_list[left])
    b = id_list[left] - k * vg_list[left]
    return (marker - b) / k

def fit_vtlgm(gm_list: List[float], vg_list: List[float], id_list: List[float], vd: float) -> Tuple[float, float]:
    """
    Calculate maximum transconductance and threshold voltage.
    Args:
        gm_list: List of transconductance values
        vg_list: List of gate voltage values
        id_list: List of drain current values
        vd: Drain voltage
    Returns:
        Tuple of (max transconductance, threshold voltage)
    """
    if not (len(id_list) == len(vg_list) == len(gm_list)):
        raise ValueError("Id Vg data point mismatch!")

    gm_max = max(gm_list)
    idx_max = gm_list.index(gm_max)
    id_gmmax = id_list[idx_max]
    vth_gmmax = vg_list[idx_max]

    vtlgm = vth_gmmax - id_gmmax / gm_max - vd/2
    return gm_max, vtlgm

def get_gm_list(id_list: List[float], vg_list: List[float]) -> List[float]:
    """
    Calculate differential of current with respect to voltage.
    Args:
        id_list: List of current values
        vg_list: List of voltage values
    Returns:
        List of differential values
    """
    if len(id_list) != len(vg_list):
        raise ValueError("Id Vg data point mismatch!")

    step = abs(vg_list[1] - vg_list[0])
    res = []

    for x in range(len(id_list)):
        if x == 0:
            res.append(abs(id_list[1] - id_list[0]) / step)
        elif x == len(id_list) - 1:
            res.append(abs(id_list[x] - id_list[x-1]) / step)
        else:
            res.append(abs(id_list[x+1] - id_list[x-1]) / (2 * step))

    return res

def extract_device_info(file_name: str,
                     override_params: Optional[Dict[str, Any]] = None, override_mode: bool = False) -> Dict[str, Any]:
    """
    Extract device information from filename with optional parameter override.
    Args:
        file_name: Name of the file to parse
        lot_id: Default Lot ID for the measurement
        wafer_id: Default Wafer ID for the measurement
        temperature: Default Temperature value for the measurement
        override_params: Dictionary of parameters to override the extracted values
        override_mode: If True, use override_params instead of extracting from filename
    Returns:
        Dictionary containing device information

    Cases handled:
    1. No match, no override mode -> return default info
    2. No match, override mode -> return override info
    3. Match found, no override mode -> return match info
    4. Match found, override mode -> return match info updated with override params
    """
    # Define default device info structure
    default_info = {
        'device_name': 'Unknown',
        'width': 1,
        'length': 1,
        'lot_id': 'Unknown',
        'wafer_id': 'Unknown',
        'x': '0',
        'y': '0',
        'site_id': '0',
        'temperature': '25C'
    }

    # Try to extract info from filename
    file_name_pattern = re.compile(r"(?P<Device_name>.*?)_(?P<Width>[wW]\d+)_(?P<Length>L\d+[pP]\d+)\s\[(?P<Lot_id>\d\w{3}\d+)(?P<Wafer_id>W\d+).*?(?P<Pos_x>-?\d+)\s+(?P<Pos_y>-?\d+).*?\((?P<Site>\d+)\)\]")
    match = file_name_pattern.match(file_name)

    if not match:
        # Case 1 & 2: No match found
        if override_mode and override_params:
            # Case 2: No match, override mode
            result = default_info.copy()
            result.update(override_params)
            return result
        else:
            # Case 1: No match, no override
            return default_info
    else:
        # Extract values from filename
        extracted_info = {
            'device_name': match.group('Device_name').replace('_',''),
            'width': float(match.group('Width').replace('W','')),
            'length': float(match.group('Length').replace('L','').replace('P','.').replace('p','.')),
            'lot_id': match.group('Lot_id'),
            'wafer_id': match.group('Wafer_id'),
            'x': match.group('Pos_x'),
            'y': match.group('Pos_y'),
            'site_id': match.group('Site'),
            'temperature': override_params.get('temperature', '25C')
        }

        if override_mode and override_params:
            # Case 4: Match found, override mode
            # Keep matched values but update with override params
            extracted_info.update(override_params)
            return extracted_info
        else:
            # Case 3: Match found, no override
            return extracted_info

def get_columns_for_block(block_type: str) -> List[str]:
    """
    Get the columns of interest for a specific block type.
    Args:
        block_type: Type of the measurement block
    Returns:
        List of column names to track for this block type
    """
    # Define columns of interest for each block type
    if block_type == 'NORMAL_SWEEP':
        return ['Vg', 'Vd', 'Id']  # Basic columns for normal sweep
    elif block_type == 'PD_SWEEP':
        return ['Vg', 'Vd', 'Id', 'Is']  # Example columns for PD sweep
    elif block_type == 'RECORD':
        return ['Time', 'Id', 'Vd']  # Example columns for record type
    else:
        return ['Vg', 'Vd', 'Id']  # Default columns

def process_file(file_path: str, device_info: Dict[str, Any],
                columns_of_interest: Optional[List[str]] = None) -> Optional[Tuple[List[List[Any]], List[Any]]]:
    """
    Process a single data file.
    Args:
        file_path: Path to the file to process
        device_info: Device information dictionary
        columns_of_interest: Optional list of column names to track in data processing
    Returns:
        Tuple of (IV curve data, summary data) or None if processing fails
    """
    try:
        with open(file_path, "r") as f:
            content = f.readlines()

        # Find all blocks in the file
        blocks = []
        current_block = []
        in_block = False

        for line in content:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for block start markers
            if "RecordTime" in line or "Sweep" in line:
                if in_block:
                    blocks.append(current_block)
                current_block = [line]
                in_block = True
                continue

            # Add lines to current block
            if in_block:
                current_block.append(line)

        # Add last block if exists
        if current_block:
            blocks.append(current_block)

        if not blocks:
            print(f"No valid blocks found in {file_path}")
            return None

        # Process each block
        iv_curve_table = []
        summary_list = []

        for block in blocks:
            # Determine block type from first line
            block_type = get_block_type(block[0])

            # Process block based on type
            if block_data := process_block(block, block_type, device_info, columns_of_interest):
                iv_curve_table.extend(block_data[0])
                summary_list.append(block_data[1])

        if not summary_list:
            return None

        summary_list.sort()  # Sort by Vd or other relevant parameter

        # Format final summary based on device type
        base_info = [
            device_info['temperature'], device_info['lot_id'], device_info['wafer_id'],
            device_info['device_name'], device_info['width'], device_info['length'],
            device_info['x'], device_info['y'], device_info['site_id']
        ]

        if 'P' in device_info['device_name']:
            summary = base_info + [summary_list[0][4], summary_list[0][5],
                                 summary_list[-1][4], summary_list[-1][5]]
        else:
            summary = base_info + [summary_list[-1][4], summary_list[-1][5],
                                 summary_list[0][4], summary_list[0][5]]

        return iv_curve_table, summary

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def get_block_type(first_line: str) -> str:
    """
    Determine the type of measurement block based on its first line.
    Args:
        first_line: First line of the block
    Returns:
        String indicating block type
    """
    first_line = first_line.lower()

    if 'sweep' in first_line:
        if 'pd' in first_line:
            return 'PD_SWEEP'
        return 'NORMAL_SWEEP'
    elif 'recordtime' in first_line:
        return 'RECORD'
    else:
        return 'UNKNOWN'

def process_block(block: List[str], block_type: str, device_info: Dict[str, Any],
                 columns_of_interest: Optional[List[str]] = None) -> Optional[Tuple[List[List[Any]], List[Any]]]:
    """
    Process a single measurement block based on its type
    Args:
        block: List of lines in the block
        block_type: Type of the block (determined by get_block_type)
        device_info: Device information dictionary
        columns_of_interest: Optional list of column names to track (overrides default columns)
    Returns:
        Tuple of (measurement data, summary data) or None if processing fails
    """
    try:
        # Find header line and data section
        header_line = None
        data_start = None

        for i, line in enumerate(block):
            if 'Remarks' in line:
                header_line = block[i+1].strip().split(',')
                data_start = i+2
                break

        if not header_line or not data_start:
            return None

        # Get block-specific columns if no override provided
        block_columns = columns_of_interest if columns_of_interest else get_columns_for_block(block_type)

        # Process based on block type
        if block_type == 'NORMAL_SWEEP':
            return process_normal_sweep_block(block[data_start:], header_line, device_info, block_columns)
        elif block_type == 'PD_SWEEP':
            return process_pd_sweep_block(block[data_start:], header_line, device_info, block_columns)
        elif block_type == 'RECORD':
            return process_record_block(block[data_start:], header_line, device_info, block_columns)
        else:
            print(f"Unknown block type: {block_type}")
            return None

    except Exception as e:
        print(f"Error processing block: {str(e)}")
        return None

def process_normal_sweep_block(data_lines: List[str], header: List[str],
                             device_info: Dict[str, Any],
                             columns_of_interest: List[str]) -> Optional[Tuple[List[List[Any]], List[Any]]]:
    """Process normal sweep measurement block"""
    try:
        iv_curve_table = []
        id_list = []
        vg_list = []

        # Get column indices for the specified columns
        col_indices = {col: header.index(col) for col in columns_of_interest if col in header}

        # Extract measurements
        for line in data_lines:
            if not line.strip():
                continue

            values = line.strip().split(',')
            if values[0] == "0":  # Assuming this is still a valid data line marker
                # Extract values based on column indices
                measurement = {col: float(values[idx]) for col, idx in col_indices.items()}

                # Store specific values needed for calculations
                if 'Id' in measurement:
                    id_list.append(measurement['Id'])
                if 'Vg' in measurement:
                    vg_list.append(measurement['Vg'])

                # Store measurement data
                iv_curve_table.append([
                    device_info['temperature'], device_info['lot_id'], device_info['wafer_id'],
                    device_info['device_name'], device_info['width'], device_info['length'],
                    device_info['x'], device_info['y'], device_info['site_id']
                ] + [measurement.get(col, 0.0) for col in columns_of_interest])

        if not id_list or not vg_list:
            return None

        # Calculate parameters if we have necessary data
        if all(col in col_indices for col in ['Vd', 'Id', 'Vg']):
            vd_val = float(values[col_indices['Vd']])
            marker = (PMOS_MARKER_MULTIPLIER if 'P' in device_info['device_name'] else NMOS_MARKER_MULTIPLIER) * device_info['width']/device_info['length']

            # Calculate gm and vtlgm
            gm_list = get_gm_list(id_list, vg_list)
            gmmax, vtlgm = fit_vtlgm(gm_list, vg_list, id_list, vd_val)

            # Calculate other parameters
            vt = fit_linear(id_list, vg_list, marker)
            id_val = id_list[-1]
            idoff = id_list[0]

            # Update measurement data with calculated values
            for idx, row in enumerate(iv_curve_table):
                row.extend([gm_list[idx], vt, id_val, idoff, gmmax, vtlgm])

            return iv_curve_table, [vd_val, vt, id_val, idoff, gmmax, vtlgm]

        return iv_curve_table, []

    except Exception as e:
        print(f"Error processing normal sweep block: {str(e)}")
        return None

def process_pd_sweep_block(data_lines: List[str], header: List[str],
                          device_info: Dict[str, Any],
                          columns_of_interest: List[str]) -> Optional[Tuple[List[List[Any]], List[Any]]]:
    """Process PD sweep measurement block"""
    # TODO: Implement PD sweep processing with block-specific columns
    # This function is left empty for custom implementation
    return None

def process_record_block(data_lines: List[str], header: List[str],
                        device_info: Dict[str, Any],
                        columns_of_interest: List[str]) -> Optional[Tuple[List[List[Any]], List[Any]]]:
    """Process record measurement block"""
    # TODO: Implement record block processing with block-specific columns
    # This function is left empty for custom implementation
    return None

def write_results(output_file_path: str, result: List[List[Any]], summary: List[List[Any]]) -> None:
    """
    Write processing results to output files
    Args:
        output_file_path: Base path for output files (without extension)
        result: List of measurement results
        summary: List of summary data
    """
    try:
        # Write main results
        with open(f"{output_file_path}.csv", 'w', newline='', encoding='utf-8') as f:
            fw = csv.writer(f, delimiter=',', lineterminator='\n')
            header = [
                "Temperature", "Lot_id", "Wafer_id", "Device_Name", "Width", "Length", "X", "Y", "Site",
                'Vg', 'Vs', 'Vd', 'Vb', 'Ig', 'Is', 'Id', 'Ib',
                "Normalized_Id(uA)", 'GM', "Vt", "Ion", "Ioff", "GMMAX", "Vtlgm",
                'Vg_abs', 'Vs_abs', 'Vd_abs', 'Vb_abs', 'Ig_abs', 'Is_abs', 'Id_abs', 'Ib_abs'
            ]
            fw.writerow(header)
            fw.writerows(result)

        # Write summary
        with open(f"{output_file_path}_summary.csv", 'w', newline='', encoding='utf-8') as f:
            fw = csv.writer(f, delimiter=',', lineterminator='\n')
            header = [
                "Temperature", "Lot_id", "Wafer_id", "Device_Name", "Width", "Length", "X", "Y", "Site",
                'GMMAX_halfVdd (A/V)', 'Vtlgm_halfVdd (V)',
                'GMMAX_0.05V (A/V)', 'Vtlgm_0.05V (V)'
            ]
            fw.writerow(header)
            fw.writerows(summary)

    except Exception as e:
        print(f"Error writing results: {str(e)}")

def main(input_folder: str, output_folder: str, output_base_name: str,
         device_info_override: Optional[Dict[str, Any]] = None,
         override_mode: bool = False,
         columns_of_interest: Optional[List[str]] = None) -> None:
    """
    Main processing function for analyzing measurement data files.

    Args:
        input_folder: Path to folder containing input CSV files
        output_folder: Path to folder where output files will be saved
        output_base_name: Base name for output files (without extension)
        device_info_override: Optional dictionary containing device info to override file-extracted info
        override_mode: If True, use override info instead of extracting from filename
        columns_of_interest: Optional list of column names to track in data processing
    """
    try:
        # Validate input/output paths
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder {input_folder} does not exist")

        os.makedirs(output_folder, exist_ok=True)

        # Initialize result containers
        result = []
        summary = []

        # Process all CSV files recursively
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith('.csv') and all(f not in file.lower() for f in ['sub', 'iv', 'output']):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}")

                    # Extract device info from filename or use override
                    device_info = extract_device_info(
                        os.path.basename(file_path),
                        override_params=device_info_override,
                        override_mode=override_mode
                    )

                    # Process file and collect results
                    if ret_val := process_file(file_path, device_info, columns_of_interest):
                        result.extend(ret_val[0])
                        summary.append(ret_val[1])

        # Generate output file paths
        output_main = os.path.join(output_folder, f"{output_base_name}.csv")
        output_summary = os.path.join(output_folder, f"{output_base_name}_summary.csv")

        # Write results if any were found
        if result:
            write_results(output_main, result, summary)
            print(f"Successfully processed {len(result)} measurements")
            print(f"Results written to: {output_main} and {output_summary}")
        else:
            print("No valid measurements found to process!")

    except Exception as e:
        print(f"Error in main processing: {str(e)}")

if __name__ == '__main__':
    """
    Example usage of the script with configuration
    """
    # Example configuration
    CONFIG = {
        'input_folder': r"path/to/input/folder",
        'output_folder': r"path/to/output/folder",
        'output_base_name': "measurement_results",
        'device_info_override': {
            'device_name': 'DGDEPP',
            'width': 10,
            'length': 0.35,
            'lot_id': 'Unknown',
            'wafer_id': 'Unknown',
            'x': '0',
            'y': '0',
            'site_id': '0',
            'temperature': '25C'
        },
        'override_mode': False,  # Set to True to use override params
        'columns_of_interest': ['Vg', 'Vd', 'Id', 'Is']  # Example columns to track
    }

    main(**CONFIG)