"""
Stacked Table module for data organization.
Provides functionality to stack and combine multiple data tables into a unified format.
"""

import os
import numpy as np
import pandas as pd
import logging
from func.ulti import ProcessingConfig, split_wafer_file_name

# Initialize module logger
logger = logging.getLogger(__name__)

class StackProcessor:
    """
    Processor class for stacking and organizing data tables.
    Handles merging of multiple data files while preserving structure and metadata.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize stack processor with configuration.

        Args:
            config: ProcessingConfig object containing processing parameters
        """
        logger.info("Initializing StackProcessor")
        self.config = config
        self.type_of_noise = 4
        logger.debug(f"Stack processor initialized with {self.type_of_noise} noise types")

    def run_stacking(self, save_file):
        """
        Execute stacking process and save results.

        Args:
            save_file: Base name for output file

        Raises:
            ValueError: If initial validation fails
        """
        logger.info(f"Starting stacking process for {save_file}")
        self.dataframes = []
        self.freq = None
        self.die_num = None

        # Initialize and validate data
        if not self.init_process():
            logger.error("Failed to initialize stacking process")
            raise ValueError("Failed to pass initial check")

        self._stacking_files(save_file)
        logger.info("Stacking process completed successfully")

    def init_process(self):
        """
        Initialize data processing.
        Load and validate all input files.

        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing data processing")
        try:
            # Load all input files
            for file_path in self.config.base_path:
                logger.debug(f"Loading file: {file_path}")
                self.get_dataframes(file_path)

            # Validate data consistency
            self.check_column_match()
            logger.info("Data processing initialization successful")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize data processing: {str(e)}")
            raise

    def get_dataframes(self, single_file):
        """
        Load data from a single file.

        Args:
            single_file: Path to input file
        """
        logger.debug(f"Reading file: {single_file}")
        try:
            # Read Excel file
            df = pd.read_excel(single_file)
            logger.debug(f"Successfully read file with shape: {df.shape}")

            # Extract metadata from filename
            device_name, wafer_id, bias_id = split_wafer_file_name(os.path.basename(single_file))
            logger.debug(f"Extracted metadata - device: {device_name}, wafer: {wafer_id}, bias: {bias_id}")

            self.dataframes.append((device_name, wafer_id, bias_id, df))
        except Exception as e:
            logger.error(f"Error reading file {single_file}: {str(e)}")
            raise

    def check_column_match(self):
        """
        Validate column consistency across all dataframes.

        Raises:
            ValueError: If column count mismatch detected
        """
        logger.info("Checking column consistency across dataframes")
        shape = None
        for device_name, wafer_id, bias_id, df in self.dataframes:
            logger.debug(f"Checking columns for {device_name} - {wafer_id} - {bias_id}")
            # Check column count consistency
            if shape is None:
                shape = df.shape[1]
                logger.debug(f"Reference column count set to: {shape}")
            elif df.shape[1] != shape:
                logger.error(f"Column count mismatch: expected {shape}, got {df.shape[1]}")
                raise ValueError("All files must have the same number of columns.")
        logger.info("Column consistency check passed")

    def _stacking_files(self, save_file):
        """
        Stack multiple files into a single combined table.

        Args:
            save_file: Base name for output file
        """
        logger.info(f"Starting file stacking process for {save_file}")
        part1 = None
        part2 = None

        try:
            # Process each input file
            for device_name, wafer_id, bias_id, df in self.dataframes:
                logger.debug(f"Processing file: {device_name} - {wafer_id} - {bias_id}")

                # Add metadata columns
                logger.debug("Adding metadata columns")
                df.insert(0, 'Wafer', [f"{wafer_id}"] * df.shape[0])
                df.insert(0, 'Device', [f"{device_name}"] * df.shape[0])
                columns_to_remove = ['Wafer', 'Device']
                df.loc[self.config.basic_info_line_num-1, columns_to_remove] = np.nan

                # Stack header portion
                if part1 is None:
                    part1 = df.iloc[:self.config.basic_info_line_num]
                    logger.debug("Initialized header portion")
                else:
                    part1 = pd.concat([part1, df.iloc[:self.config.basic_info_line_num]])
                    logger.debug("Added to header portion")

                # Stack data portion with blank row separators
                if part2 is None:
                    part2 = df.iloc[self.config.basic_info_line_num:]
                    blank_row = pd.DataFrame('', columns=df.columns, index=[0])
                    logger.debug("Initialized data portion")
                else:
                    part2 = pd.concat([part2, blank_row, df.iloc[self.config.basic_info_line_num+1:]])
                    logger.debug("Added to data portion with separator")

                # Combine parts
                result = pd.concat([part1, part2])
                logger.debug(f"Combined result shape: {result.shape}")

            # Save to Excel with formatting
            output = os.path.join(self.config.output_path, save_file + '.xlsx')
            logger.info(f"Saving stacked result to: {output}")

            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result.to_excel(writer, sheet_name='Stacked', index=False, header=True)
                workbook = writer.book
                worksheet = writer.sheets['Stacked']

                # Apply formatting
                logger.debug("Applying Excel formatting")
                header_format = workbook.add_format({'bold': False, 'border': 0})
                for col_num, value in enumerate(result.columns):
                    worksheet.write(0, col_num, value, header_format)
                for col_num in range(result.shape[1]):
                    worksheet.set_column(col_num + 1, col_num + 1, 13)

            logger.info("File stacking completed successfully")

        except Exception as e:
            logger.error(f"Error during file stacking: {str(e)}")
            raise

