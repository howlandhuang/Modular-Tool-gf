"""
Utility module providing common functionality for data processing.
Includes:
- Linear regression calculation
- File name parsing
- Data filtering
- Configuration management
- Logging setup
"""

import re, os, logging
import numpy as np
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

# Global log queue for multiprocessing
log_queue = Queue()

def setup_logger(log_file):
    """
    Setup logging configuration for multiprocessing.
    File handler will record DEBUG level messages, while console shows only INFO.

    Args:
        log_file: Path to log file

    Returns:
        QueueListener: Configured log listener
    """
    # Root logger setup - set to DEBUG to capture all levels
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Changed to DEBUG to capture all messages

    # Clear any existing handlers
    logger.handlers = []

    # File handler for logging to a file - captures DEBUG and above
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Changed to DEBUG to record all messages

    # Console handler for printing logs to the console - shows only INFO and above
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(module)s - %(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # Kept at INFO for console output

    # Create a filter for the console handler
    class InfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno >= logging.INFO

    console_handler.addFilter(InfoFilter())

    # Queue Listener to handle logs from multiprocessing
    listener = QueueListener(log_queue, file_handler, console_handler, respect_handler_level=True)
    listener.start()

    # Replace handlers with QueueHandler for all processes
    queue_handler = QueueHandler(log_queue)
    logger.handlers = [queue_handler]

    logger.info("Logger initialized successfully")
    return listener

# Initialize module logger
logger = logging.getLogger(__name__)

def lr(new_x: list, new_y: list):
    """
    Calculate linear regression parameters.

    Args:
        new_x: List of x values
        new_y: List of y values

    Returns:
        tuple: (slope, intercept, R_square)
    """
    logger.debug(f"Starting linear regression calculation with {len(new_x)} points")
    try:
        new_x = np.array(new_x)
        new_y = np.array(new_y)

        # Calculate means
        new_x_avg = new_x.mean()
        new_y_avg = new_y.mean()

        # Calculate regression parameters
        new_Sx = np.sum((new_x - new_x_avg) ** 2)
        new_Sxy = np.sum((new_x - new_x_avg) * (new_y - new_y_avg))
        slope = new_Sxy / new_Sx
        intercept = new_y_avg - slope * new_x_avg

        # Calculate R-squared
        y_pred = slope * new_x + intercept
        SS_res = np.sum((new_y - y_pred) ** 2)
        SS_tot = np.sum((new_y - new_y_avg) ** 2)
        R_square = 1 - (SS_res / SS_tot)

        logger.debug(f"Linear regression completed: slope={slope:.4f}, intercept={intercept:.4f}, R²={R_square:.4f}")
        return slope, intercept, R_square
    except Exception as e:
        logger.error(f"Error in linear regression calculation: {str(e)}")
        raise

def split_wafer_file_name(input_string: str):
    """
    Parse wafer file name into components.

    Args:
        input_string: File name to parse

    Returns:
        tuple: (name, wafer_id, bias)

    Raises:
        ValueError: If file name format is invalid
    """
    logger.debug(f"Parsing wafer file name: {input_string}")
    try:
        pattern = r'(.+)_(W#\w+)_(Bias\d+)'
        match = re.match(pattern, input_string)
        if not match:
            logger.error(f"Invalid file name format: {input_string}")
            raise ValueError("The input string format does not match the expected pattern.")
        name = match.group(1)
        wafer_id = match.group(2)
        bias = match.group(3)
        logger.debug(f"Successfully parsed: name={name}, wafer_id={wafer_id}, bias={bias}")
        return name, wafer_id, bias
    except Exception as e:
        logger.error(f"Error parsing wafer file name: {str(e)}")
        raise

def remove_outliers(df, threshold: float, tolerance: float):
    """
    Remove outliers from data based on threshold and tolerance.

    Args:
        df: Input DataFrame
        threshold: Threshold for outlier detection
        tolerance: Tolerance range for values

    Returns:
        DataFrame: Filtered DataFrame with outliers removed
    """
    logger.info(f"Starting outlier removal with threshold={threshold}, tolerance={tolerance}")
    if threshold is None:
        logger.debug("No threshold specified, returning original DataFrame")
        return df

    try:
        df_copy = df.copy()
        total_outliers = 0

        # Process each noise type
        noise_types = ['Sid', 'Sid/id^2', 'Svg', 'Sid*f']
        noise_columns = [col for col in df_copy.columns if col != "Frequency"]

        for condition in noise_types:
            logger.debug(f"Processing noise type: {condition}")

            # Get columns for current noise type
            type_cols = [col for col in noise_columns if col.endswith(f"_{condition}")]
            if not type_cols:
                logger.debug(f"No columns found for noise type: {condition}")
                continue

            median_col = condition + "_med"

            # Identify outliers
            high_mask = df_copy[type_cols].gt(df_copy[median_col].values[:, None] * np.float_power(10, tolerance))
            low_mask = df_copy[type_cols].lt(df_copy[median_col].values[:, None] * np.float_power(10, -tolerance))
            outlier_count = (high_mask | low_mask).sum().sum()
            total_outliers += outlier_count
            logger.debug(f"Found {outlier_count} outliers for {condition}")

            df_copy[type_cols] = df_copy[type_cols].mask(high_mask | low_mask)

            # Handle columns based on outlier ratio
            nan_ratios = df_copy[type_cols].isna().mean()

            # Null columns with too many outliers
            cols_to_null = nan_ratios[nan_ratios >= threshold].index
            if not cols_to_null.empty:
                logger.debug(f"Nulling {len(cols_to_null)} columns with too many outliers")
                df_copy[cols_to_null] = np.nan
                df_copy[median_col] = df_copy[type_cols].median(axis=1)

            # Recover columns with acceptable outlier ratio
            cols_to_recover = nan_ratios[(nan_ratios > 0) & (nan_ratios < threshold)].index
            if not cols_to_recover.empty:
                logger.debug(f"Recovering {len(cols_to_recover)} columns with acceptable outlier ratio")
                df_copy[cols_to_recover] = df[cols_to_recover]

            # Update statistics
            min_col = condition + "_min"
            max_col = condition + "_max"
            df_copy[min_col] = df_copy[type_cols].min(axis=1)
            df_copy[max_col] = df_copy[type_cols].max(axis=1)

        logger.info(f"Outlier removal completed. Total outliers removed: {total_outliers}")
        return df_copy
    except Exception as e:
        logger.error(f"Error in outlier removal: {str(e)}")
        raise

@dataclass
class ProcessingConfig:
    """Configuration class for data processing parameters."""
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

    def __post_init__(self):
        """Log configuration parameters after initialization."""
        logger.info("Processing configuration initialized:")
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Basic info line number: {self.basic_info_line_num}")
        logger.info(f"Prediction range: {self.pred_range_lower} to {self.pred_range_upper}")
        logger.info(f"Interest frequencies: {self.interest_freq}")
        logger.info(f"Debug mode: {self.debug_flag}")
        logger.info(f"Filter outliers: {self.filter_outliers_flag}")
        if self.filter_outliers_flag:
            logger.info(f"Filter threshold: {self.filter_threshold}")
            logger.info(f"Filter tolerance: {self.filter_tolerance}")
        logger.info(f"Prediction only mode: {self.prediction_only_flag}")

class InputValidator:
    """Validator class for user input validation."""

    def __init__(self):
        """Initialize validation patterns."""
        logger.debug("Initializing InputValidator")
        self.invalid_path_chars = r'[<>:"|?*\x00-\x1F]'
        self.single_freq_pattern = r'^\s*(\d+\.?\d*)\s*$'
        self.freq_list_pattern = r'^\s*\d+\.?\d*\s*(?:,\s*\d+\.?\d*\s*)*$'
        self.lot_id_pattern = r'^\d[a-zA-Z]{3}\d{5}(_RT)?$'
        self.wafer_id_pattern = r'^[wW]\d{2}$'

    def validate_path(self, path_str):
        """
        Validate file path string.

        Args:
            path_str: Path string to validate

        Returns:
            tuple: (is_valid, error_message, validated_path)
        """
        logger.debug(f"Validating path: {path_str}")
        try:
            if not path_str or not path_str.strip():
                logger.warning("Empty path provided")
                return False, "Path cannot be empty", None

            path_str = path_str.strip()

            # Check for invalid characters
            if re.search(self.invalid_path_chars, path_str):
                logger.warning(f"Invalid characters found in path: {path_str}")
                return False, "Path contains invalid characters", None

            # Check for reserved names on Windows
            reserved_names = {
                "CON", "PRN", "AUX", "NUL",
                "COM1", "COM2", "COM3", "COM4", "COM5",
                "COM6", "COM7", "COM8", "COM9",
                "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
                "LPT6", "LPT7", "LPT8", "LPT9"
            }

            if os.name == 'nt':  # Only check on Windows systems
                base_name = os.path.basename(path_str).split('.')[0].upper()
                if base_name in reserved_names:
                    logger.warning(f"Reserved name found in path: {base_name}")
                    return False, f"The path string contains a reserved name: {base_name}", None

            # Check path length
            max_length = 260 if os.name == 'nt' else 4096
            if len(path_str) > max_length:
                logger.warning(f"Path exceeds maximum length: {len(path_str)} > {max_length}")
                return False, f"The path string exceeds the maximum length of {max_length} characters.", None

            logger.debug(f"Path validation successful: {path_str}")
            return True, "", path_str

        except Exception as e:
            logger.error(f"Error validating path: {str(e)}")
            return False, f"Invalid path: {str(e)}", None

    def validate_single_number(self, freq_str):
        """
        Validate single number input.

        Args:
            freq_str: Number string to validate

        Returns:
            tuple: (is_valid, error_message, validated_value)
        """
        logger.debug(f"Validating single number: {freq_str}")
        try:
            if not freq_str or not freq_str.strip():
                logger.warning("Empty number string provided")
                return False, "Input cannot be empty", None

            freq_str = freq_str.strip()

            # Check format
            match = re.match(self.single_freq_pattern, freq_str)
            if not match:
                logger.warning(f"Invalid number format: {freq_str}")
                return False, "Invalid format. Please provide a single number", None

            # Convert and validate value
            value = float(freq_str)
            if value < 0:
                logger.warning(f"Negative value provided: {value}")
                return False, "Input cannot be negative", None

            logger.debug(f"Number validation successful: {value}")
            return True, "", value

        except ValueError:
            logger.warning(f"Invalid number format: {freq_str}")
            return False, "Invalid number format", None
        except Exception as e:
            logger.error(f"Error validating number: {str(e)}")
            return False, f"Invalid input: {str(e)}", None

    def validate_frequency_list(self, freq_list):
        """
        Validate frequency list input.

        Args:
            freq_list: Comma-separated frequency list to validate

        Returns:
            tuple: (is_valid, error_message, validated_list)
        """
        logger.debug(f"Validating frequency list: {freq_list}")
        try:
            if not freq_list or not freq_list.strip():
                logger.warning("Empty frequency list provided")
                return False, "Frequency list cannot be empty", None

            freq_list = freq_list.strip()

            # Check for double commas
            if ',,' in freq_list:
                logger.warning("Double commas found in frequency list")
                return False, "Double commas are not allowed", None

            # Check if string starts or ends with comma
            if freq_list.startswith(',') or freq_list.endswith(','):
                logger.warning("Frequency list starts or ends with comma")
                return False, "Input cannot start or end with a comma", None

            # Match against pattern for strict comma separation
            if not re.match(self.freq_list_pattern, freq_list):
                logger.warning(f"Invalid frequency list format: {freq_list}")
                return False, "Invalid format. Use comma-separated numbers only", None

            # Split and convert to floats
            frequencies = []
            for num in freq_list.split(','):
                try:
                    value = float(num.strip())
                    if value < 0:
                        logger.warning(f"Negative frequency found: {value}")
                        return False, f"Negative frequency ({value}) is not allowed", None
                    frequencies.append(value)
                except ValueError:
                    logger.warning(f"Invalid number in frequency list: {num.strip()}")
                    return False, f"Invalid number format: {num.strip()}", None

            logger.debug(f"Frequency list validation successful: {frequencies}")
            return True, "", frequencies

        except Exception as e:
            logger.error(f"Error validating frequency list: {str(e)}")
            return False, f"Invalid frequency list input: {str(e)}", None

    def validate_range(self, range_str, lower, upper):
        """
        Validate numeric range input.

        Args:
            range_str: Range value to validate
            lower: Lower bound of valid range
            upper: Upper bound of valid range

        Returns:
            tuple: (is_valid, error_message, validated_value)
        """
        logger.debug(f"Validating range: {range_str} (bounds: {lower} to {upper})")
        try:
            if not range_str or not range_str.strip():
                logger.warning("Empty range string provided")
                return False, "Input cannot be empty", None

            range_str = range_str.strip()

            # Convert and validate value
            value = float(range_str)
            if value < lower or value > upper:
                logger.warning(f"Value {value} outside valid range [{lower}, {upper}]")
                return False, f"Input must be between {lower} and {upper}", None

            logger.debug(f"Range validation successful: {value}")
            return True, "", value

        except ValueError:
            logger.warning(f"Invalid range format: {range_str}")
            return False, "Invalid number format", None
        except Exception as e:
            logger.error(f"Error validating range: {str(e)}")
            return False, f"Invalid input: {str(e)}", None

    def validate_lot_id(self, lot_id):
        """
        Validate lot id input.

        Args:
            lot_id: Lot id to validate

        Returns:
            tuple: (is_valid, error_message, validated_id)
        """
        logger.debug(f"Validating lot id: {lot_id}")
        try:
            if not lot_id or not lot_id.strip():
                logger.warning("Empty lot id provided")
                return False, "Input cannot be empty", None

            lot_id = lot_id.strip()

            # Check format
            match = re.match(self.lot_id_pattern, lot_id)
            if not match:
                logger.warning(f"Invalid lot id format: {lot_id}")
                return False, "Invalid format. Please provide a correct lot id", None

            logger.debug(f"lot id validation successful: {lot_id}")
            return True, "", lot_id

        except ValueError:
            logger.warning(f"Invalid lot_id format: {lot_id}")
            return False, "Invalid lot id format", None
        except Exception as e:
            logger.error(f"Error validating range: {str(e)}")
            return False, f"Invalid input: {str(e)}", None

    def validate_wafer_id(self, wafer_id):
            """
            Validate wafer id input.

            Args:
                wafer_id: wafer id to validate

            Returns:
                tuple: (is_valid, error_message, validated_id)
            """
            logger.debug(f"Validating wafer id: {wafer_id}")
            try:
                if not wafer_id or not wafer_id.strip():
                    logger.warning("Empty wafer id provided")
                    return False, "Input cannot be empty", None

                wafer_id = wafer_id.strip()

                # Check format
                match = re.match(self.wafer_id_pattern, wafer_id)
                if not match:
                    logger.warning(f"Invalid wafer id format: {wafer_id}")
                    return False, "Invalid format. Please provide a correct wafer id", None

                logger.debug(f"wafer id validation successful: {wafer_id}")
                return True, "", wafer_id

            except ValueError:
                logger.warning(f"Invalid wafer_id format: {wafer_id}")
                return False, "Invalid wafer id format", None
            except Exception as e:
                logger.error(f"Error validating range: {str(e)}")
                return False, f"Invalid input: {str(e)}", None
