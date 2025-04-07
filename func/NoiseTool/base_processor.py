"""
Base Processor module for noise analysis.
Provides common functionality for data loading and processing used by specialized processors.

This module implements a common base class that encapsulates shared functionality between
different processing classes. This design follows the Don't Repeat Yourself (DRY) principle
and promotes code reuse through inheritance.

Key shared functionality includes:
- File loading and validation
- Data extraction and filtering
- Excel file output formatting
- Error handling patterns
"""

import os
import pandas as pd
import logging
from func.ulti import parse_device_info, ProcessingConfig, remove_outliers

# Initialize module logger
logger = logging.getLogger(__name__)

class BaseProcessor:
    """
    Base processor class with common functionality for noise analysis.
    Implements shared methods for data loading, file handling, and initialization.

    This class is designed to be extended by specialized processor classes such as
    PlotProcessor and StackProcessor. It provides a common interface and implementation
    for operations that would otherwise be duplicated across multiple classes.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize base processor with configuration.

        Args:
            config: ProcessingConfig object containing processing parameters
        """
        self.config = config
        self.dataframes = []

    def load_data(self, for_plot: bool, noise_type: list) -> bool:
        """
        Load data from all input files and prepare for processing.

        Args:
            for_plot: In plot process or not

        Returns:
            bool: True if data loading successful
        """
        logger.info("Loading data from input files")
        # Reset dataframes list to avoid duplication
        self.dataframes = []

        try:
            # Load data from all input files
            for file_path in self.config.base_path:
                logger.debug(f"Loading file: {file_path}")
                # Read Excel file
                df = pd.read_excel(file_path)
                logger.debug(f"Successfully read file with shape: {df.shape}")

                # Skip header rows if in plot
                if for_plot:
                    df = df.iloc[self.config.basic_info_line_num:].reset_index(drop=True)

                # Parse file name for metadata
                result = parse_device_info(os.path.basename(file_path))
                logger.debug(f"Extracted metadata - device: {result['device_name']}, lot: {result['lot_id']}, wafer: {result['wafer_id']}, bias: {result['bias_id']}")

                # Apply outlier filtering if enabled and requested
                if for_plot and self.config.filter_outliers_flag:
                    logger.debug("Applying outlier filtering")
                    df, _ = remove_outliers(df, self.config.filter_threshold, self.config.filter_tolerance, noise_type)

                self.dataframes.append((result['device_name'], result['lot_id'], result['wafer_id'], result['bias_id'], df))
                logger.debug("File processing completed")

            logger.info(f"Successfully loaded {len(self.dataframes)} files")
            return len(self.dataframes) > 0

        except Exception as e:
            logger.error(f"Failed to load data from files: {str(e)}")
            raise

    def save_excel_file(self, df: pd.DataFrame, output_path: str, sheet_name: str = 'Sheet1', use_index: bool = False) -> None:
        """
        Save DataFrame to Excel file with standard formatting.

        Args:
            df: DataFrame to save
            output_path: Path to save Excel file
            sheet_name: Name of the sheet in Excel file
            use_index: Whether to include index in output
        """
        logger.info(f"Saving Excel file to: {output_path}")
        try:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=use_index, header=True)
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]

                # Apply formatting
                logger.debug("Applying Excel formatting")
                header_format = workbook.add_format({'bold': False, 'border': 0})
                for col_num, value in enumerate(df.columns):
                    worksheet.write(0, col_num if not use_index else col_num + 1, value, header_format)

                # Auto-adjust column widths
                # AI generated solution :-)
                for col_num in range(df.shape[1]):
                    width = max(df.iloc[:, col_num].astype(str).map(len).max(), len(str(df.columns[col_num])))
                    width = min(width + 2, 15)  # Add padding but cap width
                    worksheet.set_column(col_num if not use_index else col_num + 1, col_num if not use_index else col_num + 1, width)

            logger.debug(f"File saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save Excel file: {str(e)}")
            raise