"""
Stacked Table module for data organization.
Provides functionality to stack and combine multiple data tables into a unified format.
"""

import logging
import os
import re

import numpy as np
import pandas as pd

from func.ulti import ProcessingConfig
from func.NoiseTool.base_processor import BaseProcessor

# Initialize module logger
logger = logging.getLogger(__name__)

class StackProcessor(BaseProcessor):
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
        super().__init__(config)
        logger.debug("Stack processor initialization complete")

    def _stacking_noise_table(self, save_file):
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
            for device_name, lot_id, wafer_id, bias_id, df in self.dataframes:
                logger.debug(f"Processing file: {device_name} - {lot_id} - {wafer_id} - {bias_id}")

                # Add metadata columns
                logger.debug("Adding metadata columns")
                df.insert(0, 'Wafer', [f"{wafer_id}"] * df.shape[0])
                df.insert(0, 'Device', [f"{device_name}"] * df.shape[0])
                df.insert(0, 'Lot', [f"{lot_id}"] * df.shape[0])
                columns_to_remove = ['Wafer', 'Device', 'Lot']
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

            # Save the stacked result using the base class method
            output_path = os.path.join(self.config.output_path, f"{save_file}.xlsx")
            self.save_excel_file(result, output_path, sheet_name='Stacked')
            logger.info("File stacking completed successfully")

        except Exception as e:
            logger.error(f"Error during file stacking: {str(e)}")
            raise

    def run_noise_table_stacking(self, save_file: str) -> None:
        """
        Execute stacking process and save results.

        This method loads all input files, processes them, and generates
        a stacked output file.

        Args:
            save_file: Base name for output file

        Raises:
            ValueError: If data validation fails
            IOError: If file operations fail
        """
        logger.info(f"Starting stacking process for {save_file}")

        # Load data using base class method
        if not self.load_data(for_plot=False, noise_type=[]):
            logger.error("No data loaded for stacking")
            raise ValueError("No data available for stacking")

        # Process and stack the data
        self._stacking_noise_table(save_file)
        logger.info("Stacking process completed successfully")

    def run_prediction_table_stacking(self, save_file: str) -> None:
        """
        Execute prediction table stacking process and save results.

        This method loads all input prediction files, processes them, and generates
        a stacked output file with combined prediction data.

        Args:
            save_file: Base name for output file

        Raises:
            ValueError: If data validation fails
            IOError: If file operations fail
        """
        logger.info(f"Starting prediction table stacking process for {save_file}")
        result_df = pd.DataFrame()

        try:
            # Load and process all input files
            logger.info("Loading prediction input files")
            for file_path in self.config.base_path:
                logger.debug(f"Reading prediction file: {file_path}")
                try:
                    # Read Excel file with multi-level headers
                    df = pd.read_excel(file_path, sheet_name='Prediction', header=[0,1,2], index_col=0)
                    logger.debug(f"Successfully read prediction file with shape: {df.shape}")

                    # Extract metadata from filename
                    if match := re.match(r'0_Prediction_(.+)_(\d[a-zA-Z]{3}\d{5}(?:_[Rr][Tt])?)_W#(\w+)', os.path.basename(file_path)):
                        name = match.group(1)
                        lot_id = match.group(2)
                        wafer_id = match.group(3)
                        logger.debug(f"Extracted metadata - device: {name}, lot: {lot_id}, wafer: {wafer_id}")
                    else:
                        logger.error(f"Invalid file name format: {os.path.basename(file_path)}")
                        raise ValueError("The input string format does not match the expected pattern.")

                    # Process each bias level
                    for bias in df.columns.levels[0]:
                        # Get all columns for this bias
                        params = df[bias]['Parameters']

                        for freq in [i for i in df.columns.levels[1] if i != 'Parameters']:
                            # Combine parameters and frequency data
                            tmp = pd.concat([params[['Vd (V)', 'Vg (V)', 'Id (A)', 'gm (S)', 'tRd (s)', 'Width (um)', 'Length (um)']], df[bias][freq]], axis=1)
                            tmp.insert(0, 'Frequency', [float(freq)] * len(df.index.to_list()))
                            tmp.insert(1, 'Bias', [bias] * len(df.index.to_list()))
                            tmp['Site'] = df.index.to_list()
                            tmp['wafer'] = f'W{wafer_id}'
                            tmp['device'] = name
                            tmp['lot'] = lot_id

                            # Reorder columns for consistency
                            tmp = tmp.set_index('lot')
                            cols = tmp.columns.tolist()
                            cols.remove('Site')
                            cols.remove('wafer')
                            cols.remove('device')
                            cols.insert(0, 'Site')
                            cols.insert(0, 'wafer')
                            cols.insert(0, 'device')
                            tmp = tmp[cols]

                            # Add to result dataframe
                            result_df = pd.concat([result_df, tmp], axis=0)

                except Exception as e:
                    logger.error(f"Error processing prediction file {file_path}: {str(e)}")
                    raise IOError(f"Failed to process prediction file: {file_path}") from e

            # Validate result
            if result_df.empty:
                logger.error("No valid prediction data was processed")
                raise ValueError("No valid prediction data")

            # Save the combined result using the base class method
            excel_output = os.path.join(self.config.output_path, f"{save_file}.xlsx")
            self.save_excel_file(result_df, excel_output, sheet_name='Stacked_Prediction', use_index=True)
            logger.info("Prediction table stacking completed successfully")

        except Exception as e:
            logger.error(f"Prediction stacking process failed: {str(e)}")
            raise
