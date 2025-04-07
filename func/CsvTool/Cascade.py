import os, csv, re
from typing import List, Optional, Union

def process(file: str, new_column_name: Optional[List[str]], temperature: str) -> Optional[List[List[Union[str, float, None]]]]:
    """
    Process a single cascade measurement file and extract relevant data
    Args:
        file: Path to the cascade measurement file
        new_column_name: List of additional column names to extract
        temperature: Temperature value if not found in filename
    Returns:
        List of measurement data rows or None if file is invalid
    """
    try:
        with open(file, "r") as f:
            content = f.readlines()
    except Exception as e:
        print(f"Error reading file {file}: {str(e)}")
        return None

    # Get file name for parsing metadata
    file_name = os.path.split(file)[1]

    # Parse file metadata
    try:
        # Get Lot ID and Wafer ID
        if match := re.search(r"\[(?P<Lot_id>\w+)(?P<Wafer_id>W\d+)", file_name):
            Lot_id = match.group('Lot_id')
            Wafer_id = match.group('Wafer_id')
        else:
            Lot_id = "Unknown"
            Wafer_id = "Unknown"

        # Get Die Coordinate and Site ID
        if match := re.search(r"_\s+(?P<pos_x>-?\d+)\s+(?P<pos_y>-?\d+)\s+\((?P<site>\d+)\)]", file_name):
            x = match.group('pos_x')
            y = match.group('pos_y')
            site_id = match.group('site')
        else:
            x = "Unknown"
            y = "Unknown"
            site_id = "Unknown"

        TK_id = ''.join(re.findall(r"TK\d+", file_name)) or None
        temp = ''.join(re.findall(r'_([nN]?\d+C)', file_name)) or temperature

    except ValueError as e:
        print(f"Error parsing metadata: {str(e)}")
        return None

    # Locate measurement blocks
    block_starts = []
    for i, line in enumerate(content[3:], start=3):
        if "RecordTime" in line:
            block_starts.append(i-1)

    if not block_starts:
        print(f"No measurement blocks found in file: {file_name}")
        return None

    block_ends = block_starts[1:] + [len(content)-1]  # Last block ends at file end
    csv_contents = []

    # Process each measurement block
    for block_start, block_end in zip(block_starts, block_ends):
        block_data = {}  # Store block data temporarily
        terminal = None
        new_column_indices = []

        # Process block lines
        block_lines = content[block_start:block_end]
        for line in block_lines:
            data = line.strip().split(",")

            # Get DUT ID from measurement name
            if "I/V Sweep" in data:
                block_data['DUT_id'] = (
                    ''.join(re.findall(r"DUT\d+", ''.join(data))) or  # From measurement name
                    ''.join(re.findall(r"DUT\d+", file_name))          # From filename
                )

            # Get timestamp
            elif "RecordTime" in data:
                block_data['timestamp'] = ''.join(data).strip().replace("RecordTime", '')

            # Process header line with column information
            elif 'Ia' in data:
                # Get terminal and force mode information
                for element in data:
                    if 'R' in element:
                        terminal = element.replace("R", "")

                block_data.update({
                    'terminal': terminal,
                    'force_v_flag': 'V' in data[0],
                    'indices': {
                        'v_high': data.index('V' + list(terminal)[0]),
                        'v_offset': data.index('Voffset_' + terminal) if 'Voffset_' + terminal in data else None,
                        'i_high': data.index('I' + list(terminal)[0]),
                        'i_low': data.index('I' + list(terminal)[1]),
                        'i_gnd': data.index('Ig'),
                        'resistance': data.index('R' + terminal)
                    }
                })

                # Get indices for additional columns
                if new_column_name:
                    new_column_indices = [data.index(col) if col in data else None for col in new_column_name]

            # Process measurement data line
            elif terminal:
                try:
                    # Extract measurement values
                    indices = block_data['indices']
                    force_v_flag = block_data['force_v_flag']

                    # Get force values based on mode
                    if force_v_flag:
                        force_v = float(data[indices['v_high']])
                        v_high = None
                        force_i = None
                    else:
                        force_i = float(data[indices['i_high']])
                        v_high = float(data[indices['v_high']])
                        force_v = None

                    # Get other measurements
                    i_high = float(data[indices['i_high']])
                    i_low = float(data[indices['i_low']])
                    i_gnd = float(data[indices['i_gnd']]) if data[indices['i_gnd']] != '' else ''
                    resistance = float(data[indices['resistance']]) if data[indices['resistance']] != "" else 0
                    v_offset = float(data[indices['v_offset']]) if indices['v_offset'] is not None else None

                    # Create measurement row
                    row = [
                        Lot_id, Wafer_id, x, y, site_id,
                        TK_id, block_data['DUT_id'], block_data['timestamp'], temp,
                        force_i, force_v, terminal, resistance, resistance/1000,
                        v_offset, i_high, v_high, i_low, i_gnd,
                        force_i*1e6 if force_i is not None else None,
                        v_offset*1e3 if v_offset is not None else None,
                        i_high*1e6
                    ]

                    # Add additional column values
                    if new_column_name:
                        for idx in new_column_indices:
                            if idx is not None and data[idx] not in [None, ""]:
                                row.append(float(data[idx]))
                            else:
                                row.append(None)

                    csv_contents.append(row)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Error processing measurement line in {file_name}: {str(e)}")
                    continue

    return csv_contents

def process_folder(folder_path: str, temperature: str, new_column_name: Optional[List[str]] = None) -> List[List[Union[str, float, None]]]:
    """
    Process all CSV files in a single folder
    Args:
        folder_path: Path to folder containing measurement files
        temperature: Temperature value for measurements
        new_column_name: List of additional columns to extract
    Returns:
        List of processed measurement data
    """
    result = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.csv') and 'output' not in file.lower():
            file_path = os.path.join(folder_path, file)
            if data := process(file_path, new_column_name, temperature):
                result.extend(data)
    return result

def main(folder_path: str, output_file_path: str, manual_temperature: str, new_column_name: Optional[List[str]] = None) -> None:
    """
    Process cascade measurement files in a folder and generate combined output.
    Handles both single folder and temperature-based folder structures.
    Args:
        folder_path: Path to folder containing measurement files
        output_file_path: Path for output CSV file
        manual_temperature: Default temperature if not found in filename
        new_column_name: List of additional columns to extract
    """

    try:
        result = []

        # Check if we have temperature-based folders
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        temp_pattern = re.compile(r'^[nN]?\d+C$')
        temp_folders = [d for d in subdirs if temp_pattern.match(d)]

        if temp_folders:
            # Temperature-based folder structure
            print("Processing temperature-based folder structure...")

            # Process each temperature folder
            for temp_folder in temp_folders:
                temperature = temp_folder
                folder_path_temp = os.path.join(folder_path, temp_folder)
                print(f"Processing temperature folder: {temp_folder}")

                folder_data = process_folder(folder_path_temp, temperature, new_column_name)
                result.extend(folder_data)

        else:
            # Single folder structure
            print("Processing single folder structure...")
            result.extend(process_folder(folder_path, manual_temperature, new_column_name))

        # Write output file if we have results
        if result:
            with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
                fw = csv.writer(f, delimiter=',', lineterminator='\n')

                # Write header
                header = [
                    "Lot_ID", "Wafer_ID", "X", "Y", "Site_ID", "TK_ID", "DUT_ID", "Time", "Temp",
                    "Force Current(A)", "Force Voltage(V)", "Terminal", "Resistance(Ohm)", "Resistance(kOhm)",
                    "Voffset(V)", "I_high(A)", "V_high(V)", "I_low(A)", "I_gnd(A)",
                    "Force Current(uA)", "Voffset(mV)", "I_high(uA)"
                ]
                if new_column_name:
                    header.extend(new_column_name)

                fw.writerow(header)
                fw.writerows(result)

            print(f"Successfully processed {len(result)} measurements!")
        else:
            print("No valid measurements found to process!")

    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return

if __name__ == '__main__':
    # Configuration
    folder_path = r"C:\Users\hhuang10\Documents\Local_Data\7CXA23211W11"
    output_file_path= r"C:\Users\hhuang10\Documents\Local_Data\7CXA23211W11\single_test_DELETE.csv"
    temperature = '25C'
    main(folder_path, output_file_path, manual_temperature=temperature, new_column_name=["Vg1", "Ig1"])
