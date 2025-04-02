"""
Noise Plot module for data visualization.
Provides functionality to plot various types of noise measurements and analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from func.ulti import split_wafer_file_name, ProcessingConfig, remove_outliers
from func.NoiseTool.base_processor import BaseProcessor

# Initialize module logger
logger = logging.getLogger(__name__)

class PlotProcessor(BaseProcessor):
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
        super().__init__(config)
        logger.debug("Plot processor initialized successfully")

    def _figure_format(self, plt, title):
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
                freq = df["Frequency"]

                # Get columns for current noise type
                current_columns = len([col for col in df.columns if col.endswith(f"_{noise_type}")])

                # Format label on two lines to save horizontal space
                label_base = f"{device_name}, {lot_id}\n{wafer_id}, {bias_id}"

                if fig_type == 0:
                    # Plot individual die data with reduced opacity
                    logger.debug("Creating site plot with individual dies")
                    for die in range(current_columns):
                        plt.plot(freq, df[f"Die{die+1}_{noise_type}"],
                              color=(*colors[idx][:3], 0.1),
                              label=label_base if die == 0 else "")
                    # Plot median with full opacity
                    plt.plot(freq, df[f"{noise_type}_med"],
                        color=colors[idx],
                        label=f"{label_base}, median")
                elif fig_type == 1:
                    # Plot median only
                    logger.debug("Creating median-only plot")
                    plt.plot(freq, df[f"{noise_type}_med"],
                        color=colors[idx],
                        label=f"{label_base}, median")
                elif fig_type == 2:
                    # Plot minimum only
                    logger.debug("Creating minimum-only plot")
                    plt.plot(freq, df[f"{noise_type}_min"],
                        color=colors[idx],
                        label=f"{label_base}, min")
                elif fig_type == 3:
                    # Plot maximum only
                    logger.debug("Creating maximum-only plot")
                    plt.plot(freq, df[f"{noise_type}_max"],
                        color=colors[idx],
                        label=f"{label_base}, max")
                else:
                    logger.error(f"Invalid figure type: {fig_type}")
                    raise ValueError("Invalid fig_type")

            # Format and save plot
            title = f"{noise_type} {'median only' if fig_type else 'by site'}"
            self._figure_format(plt, title)

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

        # Load and prepare data
        if not self.load_data(for_plot=True):
            logger.error("Failed to load data for plot generation")
            raise ValueError("Failed to load data for plotting")

        # Create plots for each noise type
        for noise_type in noise_type_list:
            logger.info(f"Creating plot for noise type: {noise_type}")
            self._plot_data(noise_type, fig_type, save_name)

        logger.info("Plot generation completed successfully")

    def save_filtered_result(self, noise_type_list):
        """Save filtered data results to Excel files."""
        logger.info("Starting filtered result export")
        try:
            for file_path in self.config.base_path:
                logger.debug(f"Processing file: {file_path}")

                # Read original file (we need both header and data)
                df = pd.read_excel(file_path)
                logger.debug(f"Read file with shape: {df.shape}")

                # Preserve header and apply filtering to data
                header = df.iloc[:self.config.basic_info_line_num]
                data = df.iloc[self.config.basic_info_line_num:]
                logger.debug("Applying outlier filtering")
                data, removed_data = remove_outliers(data, self.config.filter_threshold, self.config.filter_tolerance, noise_type_list)
                modified_df = pd.concat([header, data], ignore_index=True)
                logger.debug(f"Final DataFrame shape: {modified_df.shape}")

                # Generate output filename
                device_info = os.path.basename(file_path)
                result = split_wafer_file_name(device_info)
                output_file = os.path.join(
                    self.config.output_path,
                    f'{os.path.basename(file_path[:-5])}_filtered_threshold{self.config.filter_threshold}_tolerance{self.config.filter_tolerance}.xlsx'
                )
                logger.info(f"Saving filtered result to: {output_file}")

                # Save using the base class method
                with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                    modified_df.to_excel(writer, sheet_name=result['bias_id'], index=False)
                    removed_data.to_excel(writer, sheet_name='filtered', index=False)
                
                logger.debug(f"Completed processing file: {file_path}")

            logger.info("Filtered result export completed successfully")
        except Exception as e:
            logger.error(f"Error saving filtered results: {str(e)}")
            raise
