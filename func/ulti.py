import re, os, logging
import numpy as np
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

def lr(new_x: list, new_y: list):
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    new_x_avg = new_x.mean()
    new_y_avg = new_y.mean()
    new_Sx = np.sum((new_x - new_x_avg) ** 2)
    new_Sxy = np.sum((new_x - new_x_avg) * (new_y - new_y_avg))
    slope = new_Sxy / new_Sx
    intercept = new_y_avg - slope * new_x_avg
    y_pred = slope * new_x + intercept
    SS_res = np.sum((new_y - y_pred) ** 2)
    SS_tot = np.sum((new_y - new_y_avg) ** 2)
    R_square = 1 - (SS_res / SS_tot)
    return slope, intercept, R_square

def split_wafer_file_name(input_string: str):
    pattern = r'(.+)_(W#\w+)_(Bias\d+)'
    match = re.match(pattern, input_string)
    if not match:
        raise ValueError("The input string format does not match the expected pattern.")
    name = match.group(1)
    wafer_id = match.group(2)
    bias = match.group(3)
    return name, wafer_id, bias

def remove_outliers(df, threshold: float, tolerance: float):
        if threshold is None:
            return df
        df_copy = df.copy()

        noise_types = ['Sid', 'Sid/id^2', 'Svg', 'Sid*f']
        noise_columns = [col for col in df_copy.columns if col != "Frequency"]

        for condition in noise_types:
            type_cols = [col for col in noise_columns if col.endswith(f"_{condition}")]
            if not type_cols:
                continue

            median_col = condition + "_med"

            high_mask = df_copy[type_cols].gt(df_copy[median_col].values[:, None] * np.float_power(10, tolerance))
            low_mask = df_copy[type_cols].lt(df_copy[median_col].values[:, None] * np.float_power(10, -tolerance))
            df_copy[type_cols] = df_copy[type_cols].mask(high_mask | low_mask)

            nan_ratios = df_copy[type_cols].isna().mean()
            cols_to_null = nan_ratios[nan_ratios >= threshold].index
            if not cols_to_null.empty:
                df_copy[cols_to_null] = np.nan
                df_copy[median_col] = df_copy[type_cols].median(axis=1)
            cols_to_recover = nan_ratios[(nan_ratios > 0) & (nan_ratios < threshold)].index
            if not cols_to_recover.empty:
                df_copy[cols_to_recover] = df[cols_to_recover]

            # Update min and max columns
            min_col = condition + "_min"
            max_col = condition + "_max"
            df_copy[min_col] = df_copy[type_cols].min(axis=1)
            df_copy[max_col] = df_copy[type_cols].max(axis=1)

        return df_copy

@dataclass
class ProcessingConfig:
    base_path: list
    output_path: str
    basic_info_line_num: int
    pred_range_lower: int
    pred_range_upper: int
    interest_freq: list
    debug_flag: bool
    filter_outliers_flag: bool
    filter_threshold: int
    filter_tolerance: float
    prediction_only_flag: bool

class InputValidator:
    def __init__(self):

        self.invalid_path_chars = r'[<>:"|?*\x00-\x1F]'
        self.single_freq_pattern = r'^\s*(\d+\.?\d*)\s*$'
        self.freq_list_pattern = r'^\s*\d+\.?\d*\s*(?:,\s*\d+\.?\d*\s*)*$'

    def validate_path(self, path_str):
        try:
            if not path_str or not path_str.strip():
                return False, "Path cannot be empty", None

            path_str = path_str.strip()

            if re.search(self.invalid_path_chars, path_str):
                return False, "Path contains invalid characters", None

            reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
                "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3",
                "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}

            if os.name == 'nt':  # Only check on Windows systems
                base_name = os.path.basename(path_str).split('.')[0].upper()
                if base_name in reserved_names:
                    return False, f"The path string contains a reserved name: {base_name}"

            max_length = 260 if os.name == 'nt' else 4096  # NTFS vs most Unix systems
            if len(path_str) > max_length:
                return False, f"The path string exceeds the maximum length of {max_length} characters."

            return True, "", path_str

        except Exception as e:
            return False, f"Invalid path: {str(e)}", None

    def validate_single_number(self, freq_str):
        try:
            if not freq_str or not freq_str.strip():
                return False, "Input cannot be empty", None

            freq_str = freq_str.strip()

            match = re.match(self.single_freq_pattern, freq_str)
            if not match:
                return False, "Invalid format. Please provide a single number", None

            value = float(freq_str)
            if value < 0:
                return False, "Input cannot be negative", None

            return True, "", value

        except ValueError:
            return False, "Invalid number format", None
        except Exception as e:
            return False, f"Invalid input: {str(e)}", None

    def validate_frequency_list(self, freq_str):
        try:
            if not freq_str or not freq_str.strip():
                return False, "Frequency list cannot be empty", None

            freq_str = freq_str.strip()

            # Check for double commas
            if ',,' in freq_str:
                return False, "Double commas are not allowed", None

            # Check if string starts or ends with comma
            if freq_str.startswith(',') or freq_str.endswith(','):
                return False, "Input cannot start or end with a comma", None

            # Match against pattern for strict comma separation
            if not re.match(self.freq_list_pattern, freq_str):
                return False, "Invalid format. Use comma-separated numbers only", None

            # Split and convert to floats
            frequencies = []
            for num in freq_str.split(','):
                try:
                    value = float(num.strip())
                    if value < 0:
                        return False, f"Negative frequency ({value}) is not allowed", None
                    frequencies.append(value)
                except ValueError:
                    return False, f"Invalid number format: {num.strip()}", None

            return True, "", frequencies

        except Exception as e:
            return False, f"Invalid frequency list input: {str(e)}", None

    def validate_range(self, range_str, lower, upper):
            try:
                if not range_str or not range_str.strip():
                    return False, "Input cannot be empty", None

                range_str = range_str.strip()

                # Try to convert the input to a float
                value = float(range_str)
                if value < lower or value > upper:
                    return False, f"Input must be between {lower} and {upper}", None

                return True, "", value

            except ValueError:
                return False, "Invalid number format", None
            except Exception as e:
                return False, f"Invalid input: {str(e)}", None


log_queue = Queue()
def setup_logger(log_file):
    # Root logger setup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler for printing logs to the console
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(module)s - %(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Queue Listener to handle logs from multiprocessing
    listener = QueueListener(log_queue, file_handler, console_handler)
    listener.start()

    # Replace handlers with QueueHandler for all processes
    queue_handler = QueueHandler(log_queue)
    logger.handlers = [queue_handler]

    return listener


