import re, csv
import numpy as np
from collections import defaultdict

def lr(new_x, new_y):
    """
    Calculate linear regression slope between x and y values
    Args:
        new_x: x-axis values (magnetic field)
        new_y: y-axis values (resistance/voltage)
    Returns:
        slope: The calculated slope of linear regression
    """
    new_n = len(new_x)
    new_x_avg = np.sum(new_x)/new_n
    new_y_avg = np.sum(new_y)/new_n
    x_star = new_x_avg
    y_star = new_y_avg
    new_Sx = np.sum((new_x-x_star)**2)
    new_Sxy = np.sum((new_x-x_star).reshape(-1) * (new_y-y_star).reshape(-1))
    slope = new_Sxy/new_Sx
    return slope

def parse_measurement_data(parameter_str):
    """
    Parse measurement parameter string to extract relevant information
    Args:
        parameter_str: String containing measurement parameters
    Returns:
        dict: Parsed information if match found, None otherwise
    """
    '''TK02_DUT2_n0p05T~LT~Res_ca_1000uA'''
    pattern = re.compile(r"(?P<tk_id>\w+)\_(?P<dut_id>\w+)\_(?P<mfld>\w+[T]).*?~(?P<type>\w+)\_(?P<terminal>\w+)\_(?P<bias>\d+\w+)")
    match = pattern.match(parameter_str)
    if match:
        return {
            'tk_id': match.group('tk_id'),
            'dut_id': match.group('dut_id'),
            'terminal': match.group('terminal').replace("V",""),
            'bias': match.group('bias'),
            'mfld': float(match.group('mfld').replace('n', '-').replace('p', '.').replace('T', '')),
            'measure_type': 'V' if 'V' in match.group('type') else 'R'
        }
    return None

def convert_bias(bias_value):
    """
    Convert bias value to microamperes (uA)
    Args:
        bias_value: String containing bias value with unit
    Returns:
        float: Bias value in uA
    """
    bias_value = bias_value.replace('p','.')
    if "A" in bias_value:
        if "u" in bias_value:
            return float(bias_value.replace('uA', ''))
        elif "m" in bias_value:
            return float(bias_value.replace('mA', '')) * 1e3
    elif "V" in bias_value:
        return float(bias_value.replace('V', ''))
    return 0.0

def calculate_sensitivity(slope_lr, bias_value, resistance):
    """
    Calculate Si and Sv sensitivity values
    Args:
        slope_lr: Slope from linear regression
        bias_value: Bias value string
        resistance: Resistance value
    Returns:
        tuple: (Si, Sv) sensitivity values
    """
    bias_numerical = convert_bias(bias_value)

    if "A" in bias_value:
        S_i = abs(slope_lr / bias_numerical) * 1e6                     # V / (A*T)
        S_v = S_i / float(resistance) * 1e3                            # mV / (V*T)
    elif "V" in bias_value:
        S_v = abs(slope_lr / bias_numerical) * 1e3                     # mV / (V*T)
        S_i = S_v * float(resistance) / 1e3                            # V / (A*T)

    return S_i, S_v

def main(file_path, output_file_path):
    """
    Main function to process measurement data and generate output CSV
    Args:
        file_path: Input CSV file path
        output_file_path: Output CSV file path
    """
    try:
        # Read input file
        with open(file_path, "r") as f:
            content = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file {file_path} not found.")
        return
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return

    # Extract header information
    try:
        header_line = content[0].strip().split(",")
        data_line = content[1].strip().split(",")

        # Get metadata
        metadata = {
            'Product_ID': data_line[header_line.index("Product")],
            'Lot_ID': data_line[header_line.index("Lot")]
        }

        # Get column indices for required fields
        indices = {
            'wafer': header_line.index("Wafer"),
            'wafer_alias': header_line.index("WaferAlias"),
            'parameter': header_line.index("Parameter"),
            'ucs_x': header_line.index("UCSFlashX"),
            'ucs_y': header_line.index("UCSFlashY"),
            'value': header_line.index("Value"),
            'timestamp': header_line.index("Timestamp_Src")
        }
    except (IndexError, ValueError) as e:
        print(f"Error parsing header: {str(e)}")
        return

    # Initialize data structure with optimized nesting
    Unsorted_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))))

    # Process each line of data
    for line in content[1:]:  # Skip header
        try:
            data = line.strip().split(",")
            parsed_data = parse_measurement_data(data[indices['parameter']])

            if parsed_data:
                # Create wafer identifier
                wafer_id = f"{data[indices['wafer']]}_{data[indices['wafer_alias']].zfill(2)}"
                die_pos = f"({data[indices['ucs_x']]},{data[indices['ucs_y']]})"

                # Store measurement data with timestamp
                Unsorted_tree[wafer_id][die_pos][parsed_data['tk_id']][parsed_data['dut_id']][parsed_data['terminal']][parsed_data['bias']][parsed_data['mfld']].update({
                    parsed_data['measure_type']: float(data[indices['value']]),
                    'Timestamp': data[indices['timestamp']]
                })
        except (IndexError, ValueError) as e:
            print(f"Warning: Skipping malformed line: {str(e)}")
            continue

    # Sort magnetic field values while building the sorted tree
    Sorted_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))))
    for Wafer_id, TK_id_dict in Unsorted_tree.items():
        for Die_pos, DUT_id_dict in TK_id_dict.items():
            for TK_id, DUT_site_dict in DUT_id_dict.items():
                for DUT_id, Terminal_dict in DUT_site_dict.items():
                    for Terminal, Bias_value_dict in Terminal_dict.items():
                        for Bias_value, Mag_field_dict in Bias_value_dict.items():
                            # Sort magnetic field values once
                            sorted_fields = dict(sorted(Mag_field_dict.items()))
                            for Mag_field, Measure_dict in sorted_fields.items():
                                Sorted_tree[Wafer_id][Die_pos][TK_id][DUT_id][Terminal][Bias_value][Mag_field] = Measure_dict

    csv_row_data = []

    for Wafer_id_combine, TK_id_dict in Sorted_tree.items():
        # Split wafer information
        Wafer_name, Wafer_id = Wafer_id_combine[:-3], Wafer_id_combine[-2:]

        for Die_pos, DUT_id_dict in TK_id_dict.items():
            x, y = map(int, Die_pos.strip("()").split(","))

            for TK_id, DUT_site_dict in DUT_id_dict.items():
                for DUT_id, Terminal_dict in DUT_site_dict.items():
                    for Terminal, Bias_value_dict in Terminal_dict.items():
                        for Bias_value, Mag_field_dict in Bias_value_dict.items():
                            # Process each magnetic field measurement
                            for Mag_field in Mag_field_dict.keys():
                                # Get voltage, resistance and timestamp values
                                Voltage = float(Sorted_tree[Wafer_id_combine][Die_pos][TK_id][DUT_id][Terminal][Bias_value][Mag_field]["V"]) if "ad" not in Terminal else None

                                if "R" in Sorted_tree[Wafer_id_combine][Die_pos][TK_id][DUT_id][Terminal][Bias_value][Mag_field]:
                                    Res = float(Sorted_tree[Wafer_id_combine][Die_pos][TK_id][DUT_id][Terminal][Bias_value][Mag_field]["R"])
                                else:
                                    Res = float(Sorted_tree[Wafer_id_combine][Die_pos][TK_id][DUT_id][Terminal[::-1]][Bias_value][Mag_field]["R"]) # start:end:step

                                Time = Sorted_tree[Wafer_id_combine][Die_pos][TK_id][DUT_id][Terminal][Bias_value][Mag_field]["Timestamp"]

                                csv_row_data.append([
                                    Wafer_name, Wafer_id, x, y, Time, TK_id, DUT_id,
                                    f"{TK_id}_{DUT_id}", Bias_value, Mag_field, Terminal,
                                    Voltage, Res, Res/1000
                                ])

                            # Skip sensitivity calculation for 'ad' terminal
                            if Terminal == 'ad':
                                continue

                            magnetic_fields = np.array([-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2])
                            lr_y = np.array([x[-3] for x in csv_row_data[-7:]])
                            slope_lr = lr(magnetic_fields, lr_y)
                            resistance = csv_row_data[-4][-2]
                            S_i, S_v = calculate_sensitivity(slope_lr, Bias_value, resistance)

                            for result in csv_row_data[-7:]:
                                result.extend([S_i, S_v])

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            fw = csv.writer(f, delimiter=',', lineterminator='\n')

            # Write metadata
            test_info = [
                ["Lot ID", metadata['Lot_ID']],
                ["Product ID", metadata['Product_ID']],
                []
            ]
            fw.writerows(test_info)

            header = [
                "Wafer_Name", "Wafer_ID", "X", "Y", "Time", "TK_ID", "DUT_ID", "TK_DUT",
                "Bias", "Mfld(T)", "Terminal", "Voltage(V)", "Resistance(Ohm)", "Resistance(kOhm)",
                "Si(V/A*T)", "Sv(mV/V*T)"
            ]
            fw.writerow(header)
            fw.writerows(csv_row_data)
    except Exception as e:
        print(f"Error writing output file: {str(e)}")
        return

    print("Processing completed successfully!")

# if __name__ == '__main__':
#     # File paths - modify these as needed
#     file_path = r"C:\Users\hhuang10\Downloads\daNav_WET_e13766_24102800552599.csv"
#     output_file_path = r"C:\Users\hhuang10\Downloads\12_NEW.csv"

#     main(file_path, output_file_path)