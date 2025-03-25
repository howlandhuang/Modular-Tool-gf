"""
Noise Plot module for data visualization.
Provides functionality to plot various types of noise measurements and analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from func.ulti import split_wafer_file_name, ProcessingConfig, remove_outliers, check_column_match

# Initialize module logger
logger = logging.getLogger(__name__)

class PlotProcessor:
    """
    Processor class for creating noise analysis plots.
    Handles different types of noise plots including site-specific and statistical plots.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize plot processor with configuration.

        Args:
            config: ProcessingConfig object containing processing parameters
        """
        logger.info("Initializing PlotProcessor")
        self.config = config
        logger.debug("Plot processor initialized successfully")

    def run_plots(self, noise_type_list, fig_type, save_name):
        """
        Create and save noise analysis plots.

        Args:
            noise_type_list: List of noise types to plot ['Sid', 'Sid/id^2', 'Svg', 'Sid*f']
            fig_type: Plot type indicator
                     0: plot by site
                     1: plot median only
                     2: plot min only
                     3: plot max only
            save_name: Base name for saving plot files
        """
        logger.info(f"Starting plot generation for noise types: {noise_type_list}")
        logger.debug(f"Plot type: {fig_type}, Save name: {save_name}")

        self.dataframes = []
        self.freq = None
        self.die_num = None

        # Initialize and validate data
        if not self.init_process(noise_type_list, fig_type):
            logger.error("Failed to initialize plot process")
            raise ValueError("Failed to pass initial check")

        # Create plots for each noise type
        for noise_type in noise_type_list:
            logger.info(f"Creating plot for noise type: {noise_type}")
            self._plot_data(noise_type, fig_type, save_name)

        logger.info("Plot generation completed successfully")

    def init_process(self, noise_type_list, fig_type):
        """
        Initialize data processing for plotting.

        Args:
            noise_type_list: List of noise types to process
            fig_type: Plot type indicator

        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing plot data processing")
        try:
            # Load data from all input files
            for file_path in self.config.base_path:
                logger.debug(f"Loading file: {file_path}")
                self.get_dataframes(file_path)

            # Validate data consistency
            for noise_type in noise_type_list:
                logger.debug(f"Validating data for noise type: {noise_type}")
                self.freq, self.die_num = check_column_match(self.dataframes, noise_type=noise_type, fig_type=fig_type, is_stacking=False)


            logger.info("Plot data processing initialization successful")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize plot data processing: {str(e)}")
            raise

    def get_dataframes(self, single_file):
        """
        Load and preprocess data from a single file.

        Args:
            single_file: Path to input file
        """
        logger.debug(f"Reading file: {single_file}")
        try:
            # Read Excel file
            df = pd.read_excel(single_file)
            logger.debug(f"Successfully read file with shape: {df.shape}")

            # Extract data portion (skip header rows)
            df = df.iloc[self.config.basic_info_line_num:].reset_index(drop=True)
            logger.debug(f"Extracted data portion, new shape: {df.shape}")

            # Parse file name for metadata
            device_name, lot_id, wafer_id, bias_id = split_wafer_file_name(os.path.basename(single_file))
            logger.debug(f"Extracted metadata - device: {device_name}, lot: {lot_id}, wafer: {wafer_id}, bias: {bias_id}")

            # Apply outlier filtering if enabled
            if self.config.filter_outliers_flag:
                logger.debug("Applying outlier filtering")
                df = remove_outliers(df, self.config.filter_threshold, self.config.filter_tolerance)
                logger.debug("Outlier filtering completed")

            self.dataframes.append((device_name, lot_id, wafer_id, bias_id, df))
            logger.debug("File processing completed")
        except Exception as e:
            logger.error(f"Error processing file {single_file}: {str(e)}")
            raise


    def figure_format(self, plt, title):
        """
        Apply standard formatting to plot figure.

        Args:
            plt: matplotlib.pyplot instance
            title: Plot title
        """
        logger.debug(f"Applying formatting to plot: {title}")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.grid(which='both', color='gray', linestyle='-.', linewidth=0.1)
        plt.title(title)
        # Place legend outside the figure at the upper right corner with smaller font
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0,
                  fontsize='small', framealpha=0.9)
        logger.debug("Plot formatting applied")

    def _plot_data(self, noise_type, fig_type, save_name):
        """
        Create and save a single plot.

        Args:
            noise_type: Type of noise to plot
            fig_type: Plot type indicator
            save_name: Base name for saving plot
        """
        logger.info(f"Creating plot for {noise_type} with type {fig_type}")
        try:
            # Create figure with wider width to accommodate legend
            plt.figure(figsize=(14, 8))
            colors = plt.cm.tab10(range(10))
            logger.debug("Figure created")

            # Plot data for each file
            for idx, (device_name, lot_id, wafer_id, bias_id, df) in enumerate(self.dataframes):
                logger.debug(f"Plotting data for {device_name} - {lot_id} - {wafer_id} - {bias_id}")

                # Format label on two lines to save horizontal space
                label_base = f"{device_name}, {lot_id}\n{wafer_id}, {bias_id}"

                if fig_type == 0:
                    # Plot individual die data with reduced opacity
                    logger.debug("Creating site plot with individual dies")
                    for die in range(self.die_num):
                        plt.plot(self.freq, df[f"Die{die+1}_{noise_type}"],
                              color=(*colors[idx][:3], 0.1),
                              label=label_base if die == 0 else "")
                    # Plot median with full opacity
                    plt.plot(self.freq, df[f"{noise_type}_med"],
                        color=colors[idx],
                        label=f"{label_base}, median")
                elif fig_type == 1:
                    # Plot median only
                    logger.debug("Creating median-only plot")
                    plt.plot(self.freq, df[f"{noise_type}_med"],
                        color=colors[idx],
                        label=f"{label_base}, median")
                elif fig_type == 2:
                    # Plot minimum only
                    logger.debug("Creating minimum-only plot")
                    plt.plot(self.freq, df[f"{noise_type}_min"],
                        color=colors[idx],
                        label=f"{label_base}, min")
                elif fig_type == 3:
                    # Plot maximum only
                    logger.debug("Creating maximum-only plot")
                    plt.plot(self.freq, df[f"{noise_type}_max"],
                        color=colors[idx],
                        label=f"{label_base}, max")
                else:
                    logger.error(f"Invalid figure type: {fig_type}")
                    raise ValueError("Invalid fig_type")

            # Format and save plot
            title = f"{noise_type} {'median only' if fig_type else 'by site'}"
            self.figure_format(plt, title)

            # Adjust layout to make room for the legend
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)  # Adjust right margin to make space for legend

            if not self.config.debug_flag:
                output_path = f'{self.config.output_path}/{save_name}_{title.replace("/", "_").replace("*", "x")}.png'
                logger.info(f"Saving plot to: {output_path}")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.debug("Plot saved and closed")
            else:
                logger.debug("Showing plot in debug mode")
                plt.show()

            logger.info("Plot creation completed successfully")
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            raise

    def save_filtered_result(self):
        """Save filtered data results to Excel files."""
        logger.info("Starting filtered result export")
        try:
            for file_path in self.config.base_path:
                logger.debug(f"Processing file: {file_path}")

                # Read original file
                df = pd.read_excel(file_path)
                logger.debug(f"Read file with shape: {df.shape}")

                # Preserve header and apply filtering to data
                header = df.iloc[:self.config.basic_info_line_num]
                data = df.iloc[self.config.basic_info_line_num:]
                logger.debug("Applying outlier filtering")
                data = remove_outliers(data, self.config.filter_threshold, self.config.filter_tolerance)
                modified_df = pd.concat([header, data], ignore_index=True)
                logger.debug(f"Final DataFrame shape: {modified_df.shape}")

                # Generate output filename
                device_info = os.path.basename(file_path)
                device_name, lot_id, wafer_id, bias_id = split_wafer_file_name(device_info)
                output_file = os.path.join(
                    self.config.output_path,
                    f'{os.path.basename(file_path[:-5])}_filtered_threshold{self.config.filter_threshold}_tolerance{self.config.filter_tolerance}.xlsx'
                )
                logger.info(f"Saving filtered result to: {output_file}")

                # Save to Excel with formatting
                with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                    modified_df.to_excel(writer, sheet_name=bias_id, index=False, header=True)
                    workbook = writer.book
                    worksheet = writer.sheets[bias_id]

                    # Apply formatting
                    logger.debug("Applying Excel formatting")
                    header_format = workbook.add_format({'bold': False, 'border': 0})
                    for col_num, value in enumerate(modified_df.columns):
                        worksheet.write(0, col_num, value, header_format)
                    for col_num in range(modified_df.shape[1]):
                        worksheet.set_column(col_num + 1, col_num + 1, 12)

                logger.debug(f"Completed processing file: {file_path}")

            logger.info("Filtered result export completed successfully")
        except Exception as e:
            logger.error(f"Error saving filtered results: {str(e)}")
            raise
