"""
Stacked Table module for data organization.
Provides functionality to stack and combine multiple data tables into a unified format.
"""

import os, re
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
        # self.type_of_noise = 4
        # logger.debug(f"Stack processor initialized with {self.type_of_noise} noise types")

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
        self.dataframes = []

        try:
            # Load all input files
            logger.info("Loading input files")
            for file_path in self.config.base_path:
                logger.debug(f"Reading file: {file_path}")
                try:
                    # Read Excel file
                    df = pd.read_excel(file_path)
                    logger.debug(f"Successfully read file with shape: {df.shape}")

                    # Extract metadata from filename
                    device_name, wafer_id, bias_id = split_wafer_file_name(os.path.basename(file_path))
                    logger.debug(f"Extracted metadata - device: {device_name}, wafer: {wafer_id}, bias: {bias_id}")

                    self.dataframes.append((device_name, wafer_id, bias_id, df))
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    raise IOError(f"Failed to read input file: {file_path}") from e

            # Validate data consistency
            if not self.dataframes:
                logger.error("No valid data files were loaded")
                raise ValueError("No valid data files")

            # Process and stack the data
            self._stacking_noise_table(save_file)
            logger.info("Stacking process completed successfully")

        except Exception as e:
            logger.error(f"Stacking process failed: {str(e)}")
            raise

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
                    if match := re.search(r"0_Prediction_(.*?)_W#(\d+)", os.path.basename(file_path)):
                        device_name = match.group(1)
                        wafer_num = match.group(2)
                        logger.debug(f"Extracted metadata - device: {device_name}, wafer: {wafer_num}")
                    else:
                        logger.warning(f"Could not extract metadata from filename: {file_path}")
                        continue

                    # Process each bias level
                    for bias in df.columns.levels[0]:
                        # Get all columns for this bias
                        params = df[bias]['Parameters']
                        for freq in [i for i in df.columns.levels[1] if i != 'Parameters']:
                            # Combine parameters and frequency data
                            tmp = pd.concat([params[['Vd (V)', 'Vg (V)', 'Id (A)', 'gm (S)']], df[bias][freq]], axis=1)
                            tmp.insert(0, 'Frequency', [float(freq)] * len(df.index.to_list()))
                            tmp.insert(1, 'Bias', [bias] * len(df.index.to_list()))
                            tmp['Site'] = df.index.to_list()
                            tmp['wafer'] = f'W{wafer_num}'
                            tmp['device'] = device_name

                            # Reorder columns for consistency
                            tmp = tmp.set_index('device')
                            cols = tmp.columns.tolist()
                            cols.remove('Site')
                            cols.remove('wafer')
                            cols.insert(0, 'Site')
                            cols.insert(0, 'wafer')
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

            # # Save the combined result
            # output_path = os.path.join(self.config.output_path, f"{save_file}_prediction.csv")
            # logger.info(f"Saving stacked prediction result to: {output_path}")
            # result_df.to_csv(output_path, index=True)

            # Optionally save as Excel if needed
            excel_output = os.path.join(self.config.output_path, f"{save_file}.xlsx")
            with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, sheet_name='Stacked_Prediction', index=True)
                workbook = writer.book
                worksheet = writer.sheets['Stacked_Prediction']

                # Apply formatting
                header_format = workbook.add_format({'bold': True, 'border': 1})
                for col_num, value in enumerate(result_df.columns):
                    worksheet.write(0, col_num + 1, value, header_format)

            logger.info("Prediction table stacking completed successfully")

        except Exception as e:
            logger.error(f"Prediction stacking process failed: {str(e)}")
            raise
