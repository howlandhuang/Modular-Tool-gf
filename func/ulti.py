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
import pandas as pd
import numpy as np
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from PyQt6.QtWidgets import QWidget, QInputDialog, QMessageBox
from typing import Any, Callable, Optional, List, Dict, Union
from functools import wraps

# Validation patterns for input validation
INVALID_PATH_CHARS = r'[<>:"|?*\x00-\x1F]'
SINGLE_NUMBER_PATTERN = r'^\s*(\d+(?:\.\d+)?)\s*$'
FREQ_LIST_PATTERN = r'\s*\d+(?:\.\d+)?\s*(?:,\s*\d+(?:\.\d+)?\s*)*'
LOT_ID_PATTERN = r'(\d[a-zA-Z]{3}\d{5}(?:_(?:rt|RT|re|RE))?)'
WAFER_ID_PATTERN = r'[wW]#?(\d+)'
BIAS_ID_PATTERN = r'(Bias\d+)'
DEVICE_WIDTH_LENGTH_PATTERN = r'[wW](\d+(?:\.\d+)?)[lLxX](\d+(?:\.\d+)?)'
RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5",
    "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
    "LPT6", "LPT7", "LPT8", "LPT9"
}

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
        return slope, intercept, R_square
    except Exception as e:
        logger.error(f"Error in linear regression calculation: {str(e)}")
        raise

def parse_device_info(input_string: str) -> Dict[str, str]:
    """
    Parse wafer file name into components.
    This is the primary function for extracting device information from filenames.

    Args:
        input_string: File name to parse
    Returns:
        dict: Dictionary containing parsed components with keys:
            - 'device_name': Device name/identifier
            - 'lot_id': Lot identification number
            - 'wafer_id': Wafer identification number
            - 'bias': Bias information
            - 'width': Device width
            - 'length': Device length

    Raises:
        ValueError: If file name format is invalid
    """
    logger.debug(f"Parsing device info from: {input_string}")
    try:
        # Initialize result dictionary
        result = {
            'device_name': '',
            'lot_id': '',
            'wafer_id': '',
            'bias_id': '',
            'width': '',
            'length': ''
        }
        # Create a copy of input string for manipulation
        remaining_string = os.path.basename(input_string)

        # Search for bias (pattern: Bias1)
        bias_match = re.search(BIAS_ID_PATTERN, remaining_string)
        if bias_match:
            result['bias_id'] = bias_match.group(1)
            remaining_string = remaining_string.replace(bias_match.group(1), '')
            input_string = input_string.replace(bias_match.group(1), '')

        # Search for width/length (pattern: W1.2xL3.4)
        wl_match = re.search(DEVICE_WIDTH_LENGTH_PATTERN, remaining_string)
        if wl_match:
            result['width'] = wl_match.group(1)
            result['length'] = wl_match.group(2)
            # Replace the entire width/length pattern
            remaining_string = remaining_string.replace(wl_match.group(0), '')
            input_string = input_string.replace(wl_match.group(0), '')

        # Search for the last occurrence of lot_id
        lot_matches = list(re.finditer(LOT_ID_PATTERN, input_string))
        if lot_matches:
            # Get the last match
            last_lot_match = lot_matches[-1]
            result['lot_id'] = last_lot_match.group(1)
            remaining_string = remaining_string.replace(last_lot_match.group(1), '')
            input_string = input_string.replace(last_lot_match.group(1), '')

        # Search for wafer_id
        wafer_matches = list(re.finditer(WAFER_ID_PATTERN, input_string))
        if wafer_matches:
            # Get the last match
            last_wafer_match = wafer_matches[-1]
            result['wafer_id'] = last_wafer_match.group(1)
            remaining_string = remaining_string.replace(last_wafer_match.group(0), '')
            input_string = input_string.replace(last_wafer_match.group(0), '')

        # Clean up remaining string and set as device name
        # 1. Remove file extension if present
        remaining_string = re.sub(r'\.\w+$', '', remaining_string)

        # 2. Replace multiple underscores with single underscore
        remaining_string = re.sub(r'_+', '_', remaining_string)

        # 3. Remove trailing/leading underscores and any remaining whitespace
        remaining_string = remaining_string.rstrip('_').lstrip('_').strip()

        result['device_name'] = remaining_string

        logger.debug(f"Successfully parsed: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing device data from {input_string}: {str(e)}")
        raise

def remove_outliers(df, threshold: float, tolerance: float, noise_type: list):
    """
    Remove outliers from data based on threshold and tolerance.

    Args:
        df: Input DataFrame
        threshold: Threshold for outlier detection
        tolerance: Tolerance range for values
        noise_type: List of noise types to process

    Returns:
        tuple: (DataFrame, DataFrame) Filtered DataFrame with outliers removed and removed DataFrame
    """
    logger.info(f"Starting outlier removal with threshold={threshold}, tolerance={tolerance}")
    if (threshold is None) or (tolerance is None):
        logger.debug("No threshold or tolerance specified, returning original DataFrame")
        return df, pd.DataFrame()

    try:
        df_removed = pd.DataFrame()
        df_copy = df.copy()
        total_outliers = 0

        # Process each noise type
        noise_columns = [col for col in df_copy.columns if col != "Frequency"]

        for condition in noise_type:
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
                df_removed = pd.concat([df_removed, df[cols_to_null]], axis=1)
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
        return df_copy, df_removed
    except Exception as e:
        logger.error(f"Error in outlier removal: {str(e)}")
        raise

@dataclass
class CsvProcessingConfig:
    """Configuration class for CSV data processing parameters."""
    input_path: str
    output_path: str
    magnetic_fields: List[float]

@dataclass
class ProcessingConfig:
    """Configuration class for data processing parameters."""
    base_path: Union[List[str], str, None]
    output_path: Optional[str]
    basic_info_line_num: int
    pred_range_lower: Optional[int]
    pred_range_upper: Optional[int]
    interest_freq: Optional[List[float]]
    debug_flag: bool
    filter_outliers_flag: bool
    filter_threshold: float
    filter_tolerance: float
    auto_size: bool

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, value: Any = None, show_warning: bool = True):
        self.message = message
        self.value = value
        self.show_warning = show_warning
        super().__init__(message)

    def handle(self) -> Optional[Any]:
        """
        Handle the validation error by showing warning if needed and returning value if available.

        Returns:
            Optional[Any]: The value if available, None otherwise
        """
        if self.show_warning:
            send_warning(self.message)
        return self.value

def handle_validation(func: Callable) -> Callable:
    """
    Decorator to handle validation errors automatically.
    Shows warnings for ValidationError and returns None for other exceptions.

    Args:
        func: The function to decorate

    Returns:
        Callable: Decorated function that handles validation errors
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Optional[Any]:
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            return e.handle()
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            send_warning(f"An unexpected error occurred: {str(e)}")
            return None
    return wrapper

@handle_validation
def validate_input(value: str, validator: Callable[[str], Any]) -> Any:
    """
    Generic validation function that handles common validation patterns.

    Args:
        value: The input string to validate
        validator: A validation function that returns the validated value or raises ValidationError

    Returns:
        The validated value

    Raises:
        ValidationError: If validation fails
    """
    if not value or not value.strip():
        raise ValidationError("Input cannot be empty")

    value = value.strip()
    try:
        return validator(value)
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Validation error: {str(e)}")

@handle_validation
def validate_filename(value: str) -> str:
    """Validate filename string."""
    if re.search(INVALID_PATH_CHARS, value):
        raise ValidationError("Filename contains invalid characters")

    if os.name == 'nt':
        base_name = os.path.splitext(value)[0].upper()
        if base_name in RESERVED_NAMES:
            raise ValidationError(f"Filename cannot be a reserved name: {base_name}")

    if len(value) > 255:
        raise ValidationError("Filename exceeds maximum length of 255 characters")

    return value

@handle_validation
def validate_single_number(value: str) -> float:
    """Validate single number input."""
    if not re.match(SINGLE_NUMBER_PATTERN, value):
        raise ValidationError("Invalid format. Please provide a single number")

    num = float(value)
    if num < 0:
        raise ValidationError("Input cannot be negative")

    return num

@handle_validation
def validate_frequency_list(value: str) -> list[float]:
    """Validate frequency list input."""
    if re.search(r',{2,}', value):
        raise ValidationError("Multiple consecutive commas are not allowed")

    if value.startswith(',') or value.endswith(','):
        raise ValidationError("Input cannot start or end with a comma")

    frequencies = []
    for num in value.split(','):
        try:
            freq = float(validate_single_number(num.strip()))
            if freq < 0:
                raise ValidationError(f"Negative frequency ({freq}) is not allowed")
            frequencies.append(freq)
        except ValueError:
            raise ValidationError(f"Invalid number format: {num.strip()}")

    return frequencies

@handle_validation
def validate_range(value: str, lower: float, upper: float) -> float:
    """Validate numeric range input."""
    num = float(value)
    if num < lower or num > upper:
        raise ValidationError(f"Input must be between {lower} and {upper}")
    return num

@handle_validation
def validate_lot_id(value: str) -> str:
    """Validate lot id input."""
    if not re.match(LOT_ID_PATTERN, value):
        raise ValidationError("Invalid format. Please provide a correct lot id")
    return value

@handle_validation
def validate_wafer_id(value: str) -> str:
    """Validate wafer id input."""
    if not re.match(WAFER_ID_PATTERN, value):
        raise ValidationError("Invalid format. Please provide a correct wafer id")
    return value

def get_user_input(title: str, prompt: str, validator: Callable[[str], Any]) -> Optional[Any]:
    """
    Get validated user input via dialog.

    Args:
        title: Dialog window title
        prompt: Input prompt message
        validator: Function to validate input

    Returns:
        Optional[Any]: The validated input or None if cancelled/failed
    """
    temp_widget = QWidget()
    try:
        while True:
            value, ok = QInputDialog.getText(temp_widget, title, prompt)
            if not ok:
                logger.debug(f"User cancelled {title} input")
                return None

            try:
                validated_value = validate_input(value, validator)
                logger.info(f"User provided valid {title}: {validated_value}")
                return validated_value

            except ValidationError as e:
                logger.warning(f"Invalid {title} input: {e.message}")
                QMessageBox.warning(temp_widget, f"Invalid {title}", e.message)

            except Exception as e:
                logger.error(f"Unexpected error during {title} validation: {str(e)}")
                return None

    except Exception as e:
        logger.error(f"Error getting user input for {title}: {str(e)}")
        return None
    finally:
        temp_widget.deleteLater()

def send_warning(message: str):
    """
    Send a warning message to the logger and show it to the user.

    Args:
        message: The warning message to display
    """
    logger.warning(message)
    QMessageBox.warning(None, "Warning", message)
